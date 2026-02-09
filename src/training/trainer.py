"""
VAE Training loop with curriculum batching, early stopping, and LR scheduling.

Handles:
- Forward/backward/optimizer step
- log_sigma_sq clamping after each step
- Curriculum batching transition (synchronous → random)
- Mode F warmup protection for scheduler/early stopping
- Validation ELBO computation (INV-011)
- Monitoring metrics collection

Reference: ISD Section MOD-005 — Sub-task 2.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from src.utils import get_optimal_device, get_dataloader_kwargs, get_amp_config, configure_backend
from src.vae.loss import (
    compute_co_movement_loss,
    compute_loss,
    compute_validation_elbo,
    get_beta_t,
    get_lambda_co,
)
from src.vae.model import VAEModel
from src.training.batching import CurriculumBatchSampler
from src.training.early_stopping import EarlyStopping
from src.training.scheduler import SchedulerWrapper


class VAETrainer:
    """
    VAE training loop with curriculum batching and early stopping.

    Attributes:
        model: VAEModel
        optimizer: Adam optimizer
        loss_mode: str — "P", "F", or "A"
        gamma: float — crisis overweighting factor
        lambda_co_max: float — max co-movement weight
        beta_fixed: float — fixed β for Mode A
        warmup_fraction: float — Mode F warmup fraction
        early_stopping: EarlyStopping
        scheduler: SchedulerWrapper
        device: torch.device
    """

    def __init__(
        self,
        model: VAEModel,
        loss_mode: str = "P",
        gamma: float = 3.0,
        lambda_co_max: float = 0.5,
        beta_fixed: float = 1.0,
        warmup_fraction: float = 0.20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        device: torch.device | None = None,
        log_dir: str | None = None,
        logging_steps: int = 50,
        max_pairs: int = 2048,
        delta_sync: int = 21,
        compile_model: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        """
        :param model (VAEModel): The VAE model
        :param loss_mode (str): Loss mode — "P", "F", or "A"
        :param gamma (float): Crisis overweighting factor
        :param lambda_co_max (float): Maximum co-movement loss weight
        :param beta_fixed (float): Fixed β for Mode A
        :param warmup_fraction (float): Fraction of epochs for Mode F warmup
        :param learning_rate (float): Initial learning rate (η_0)
        :param weight_decay (float): Weight decay for Adam
        :param adam_betas (tuple): Adam betas
        :param adam_eps (float): Adam epsilon
        :param patience (int): Early stopping patience
        :param lr_patience (int): ReduceLROnPlateau patience
        :param lr_factor (float): ReduceLROnPlateau factor
        :param device (torch.device | None): Device for training
        :param log_dir (str | None): TensorBoard log directory (None to disable)
        :param logging_steps (int): Log to TensorBoard every N training steps (0 to disable step logging)
        :param max_pairs (int): Maximum number of pairs for co-movement loss
        :param delta_sync (int): Max date gap for synchronous batching (days)
        :param compile_model (bool): Use torch.compile (requires PyTorch 2.x)
        :param gradient_accumulation_steps (int): Accumulate gradients over N steps
        :param gradient_checkpointing (bool): Recompute activations on backward to save VRAM
        """
        self.model: VAEModel = model
        self.loss_mode = loss_mode
        self.gamma = gamma
        self.lambda_co_max = lambda_co_max
        self.beta_fixed = beta_fixed
        self.warmup_fraction = warmup_fraction
        self.max_pairs = max_pairs
        self.delta_sync = delta_sync
        self._accumulation_steps = max(1, gradient_accumulation_steps)
        self._gradient_checkpointing = gradient_checkpointing
        self.device = device or get_optimal_device()
        self._is_cuda = self.device.type == "cuda"

        # Backend flags: cuDNN benchmark, TF32 (CUDA); no-op on MPS/CPU
        configure_backend(self.device)

        self.model = self.model.to(self.device)

        # torch.compile for op fusion (PyTorch 2.x, CUDA and MPS)
        if compile_model and hasattr(torch, "compile") and self.device.type in ("cuda", "mps"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                logger.info("torch.compile enabled (%s)", self.device.type)
            except Exception as e:
                logger.warning("torch.compile failed, continuing without: %s", e)

        # Optimizer: AdamW (fused on CUDA for 3-5% speedup)
        optim_kwargs: dict[str, object] = {
            "lr": learning_rate,
            "betas": adam_betas,
            "eps": adam_eps,
            "weight_decay": weight_decay,
        }
        if self._is_cuda:
            optim_kwargs["fused"] = True
        try:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), **optim_kwargs,  # type: ignore[arg-type]
            )
        except TypeError:
            # fused=True not supported on this PyTorch version
            optim_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(
                model.parameters(), **optim_kwargs,  # type: ignore[arg-type]
            )

        # Early stopping and scheduler
        self.early_stopping = EarlyStopping(patience=patience)
        self.scheduler = SchedulerWrapper(
            self.optimizer, patience=lr_patience, factor=lr_factor,
        )

        # log_sigma_sq clamping bounds
        self._log_sigma_sq_min = math.log(1e-4)
        self._log_sigma_sq_max = math.log(10.0)

        # AMP (Automatic Mixed Precision) — auto-adapts to device
        amp_cfg = get_amp_config(self.device)
        self.use_amp: bool = bool(amp_cfg["use_amp"])
        self._amp_device_type: str = str(amp_cfg["device_type"])
        self._amp_dtype: torch.dtype = amp_cfg["dtype"]  # type: ignore[assignment]
        if amp_cfg["use_scaler"]:
            self.scaler: torch.amp.GradScaler | None = torch.amp.GradScaler(  # type: ignore[reportPrivateImportUsage]
                device=self._amp_device_type,
            )
        else:
            self.scaler = None

        # TensorBoard logging (optional)
        self._tb_writer: Any = None
        self._logging_steps = logging_steps
        self._global_step = 0
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("tensorboard not installed — TensorBoard logging disabled.")

    def train_epoch(
        self,
        train_loader: DataLoader[Any],
        epoch: int,
        total_epochs: int,
        crisis_fractions: torch.Tensor | None = None,
        progress_bar: tqdm | None = None,
    ) -> dict[str, float]:
        """
        Run one training epoch.

        :param train_loader (DataLoader): Training data loader
        :param epoch (int): Current epoch (0-indexed)
        :param total_epochs (int): Total number of epochs
        :param crisis_fractions (torch.Tensor | None): f_c per window (N,)
        :param progress_bar (tqdm | None): Shared progress bar (updated per step)

        :return metrics (dict): Epoch-level training metrics
        """
        self.model.train()
        _nb = self._is_cuda  # non_blocking transfers (CUDA only)
        accum = self._accumulation_steps

        # Accumulate on-device to avoid per-batch CPU-GPU sync
        epoch_loss = torch.tensor(0.0, device=self.device)
        epoch_recon = torch.tensor(0.0, device=self.device)
        epoch_kl = torch.tensor(0.0, device=self.device)
        epoch_co = torch.tensor(0.0, device=self.device)
        epoch_sigma_sq = torch.tensor(0.0, device=self.device)
        n_batches = 0

        # Zero gradients before first accumulation window
        self.optimizer.zero_grad(set_to_none=True)

        for step_in_epoch, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0].to(self.device, non_blocking=_nb)
                batch_indices = batch_data[1] if len(batch_data) > 1 else None
                raw_ret = batch_data[2].to(self.device, non_blocking=_nb) if len(batch_data) > 2 else None
            else:
                x = batch_data.to(self.device, non_blocking=_nb)
                batch_indices = None
                raw_ret = None

            # Get crisis fractions for this batch (already on device if pre-moved)
            if crisis_fractions is not None and batch_indices is not None:
                cf = crisis_fractions[batch_indices]
                if cf.device != self.device:
                    cf = cf.to(self.device, non_blocking=_nb)
            elif crisis_fractions is not None:
                B = x.shape[0]
                cf = crisis_fractions[:B]
                if cf.device != self.device:
                    cf = cf.to(self.device, non_blocking=_nb)
            else:
                cf = torch.zeros(x.shape[0], device=self.device)

            # Forward pass + loss (under AMP autocast if enabled)
            with torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
                device_type=self._amp_device_type,
                dtype=self._amp_dtype,
                enabled=self.use_amp,
            ):
                if self._gradient_checkpointing:
                    cp_out = checkpoint_utils.checkpoint(
                        self.model, x, use_reentrant=False,
                    )
                    assert cp_out is not None
                    x_hat, mu, log_var = cp_out
                else:
                    x_hat, mu, log_var = self.model(x)

                # Co-movement loss on raw returns (ISD MOD-004: NOT z-scored)
                co_mov_loss: torch.Tensor | None = None
                cur_lambda_co = get_lambda_co(epoch, total_epochs, self.lambda_co_max)
                if raw_ret is not None and cur_lambda_co > 0 and x.shape[0] >= 2:
                    co_mov_loss = compute_co_movement_loss(
                        mu, raw_ret, max_pairs=self.max_pairs,
                    )

                loss, components = compute_loss(
                    x=x,
                    x_hat=x_hat,
                    mu=mu,
                    log_var=log_var,
                    log_sigma_sq=self.model.log_sigma_sq,
                    crisis_fractions=cf,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    mode=self.loss_mode,
                    gamma=self.gamma,
                    lambda_co_max=self.lambda_co_max,
                    beta_fixed=self.beta_fixed,
                    warmup_fraction=self.warmup_fraction,
                    co_movement_loss=co_mov_loss,
                )

            # Backward pass with gradient accumulation
            loss_scaled = loss / accum
            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimizer step every `accum` mini-batches
            if (step_in_epoch + 1) % accum == 0:
                # Gradient clipping to prevent NaN from gradient explosion
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Clamp log_sigma_sq after optimizer step
                with torch.no_grad():
                    self.model.log_sigma_sq.clamp_(
                        self._log_sigma_sq_min, self._log_sigma_sq_max,
                    )

            # Accumulate metrics (tensor += tensor, stays on device — no sync)
            epoch_loss += components["total"]
            epoch_recon += components["recon"]
            epoch_kl += components["kl"]
            epoch_co += components["co_mov"]
            epoch_sigma_sq += components["sigma_sq"]
            n_batches += 1
            self._global_step += 1

            # Per-step TensorBoard logging (sync only when logging)
            if (
                self._tb_writer is not None
                and self._logging_steps > 0
                and self._global_step % self._logging_steps == 0
            ):
                self._tb_writer.add_scalar(
                    "Step/loss", components["total"].item(), self._global_step,
                )
                self._tb_writer.add_scalar(
                    "Step/reconstruction", components["recon"].item(), self._global_step,
                )
                self._tb_writer.add_scalar(
                    "Step/kl_divergence", components["kl"].item(), self._global_step,
                )
                self._tb_writer.add_scalar(
                    "Step/co_movement", components["co_mov"].item(), self._global_step,
                )
                self._tb_writer.add_scalar(
                    "Step/sigma_sq", components["sigma_sq"].item(), self._global_step,
                )

            # Update shared progress bar (throttle postfix to reduce syncs)
            if progress_bar is not None:
                progress_bar.update(1)
                if n_batches % 10 == 1:
                    progress_bar.set_postfix(
                        loss=f"{float(epoch_loss / n_batches):.4f}",
                        epoch=f"{epoch + 1}/{total_epochs}",
                    )

        # Flush leftover accumulated gradients (when n_batches % accum != 0)
        if n_batches > 0 and n_batches % accum != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model.log_sigma_sq.clamp_(
                    self._log_sigma_sq_min, self._log_sigma_sq_max,
                )

        n_batches = max(1, n_batches)

        # Compute AU on the last batch (approximate monitoring)
        au_count = self._compute_batch_au(mu, log_var)

        # Single sync point: convert GPU accumulators to Python floats
        metrics: dict[str, float] = {
            "train_loss": float(epoch_loss / n_batches),
            "train_recon": float(epoch_recon / n_batches),
            "train_kl": float(epoch_kl / n_batches),
            "train_co": float(epoch_co / n_batches),
            "sigma_sq": float(epoch_sigma_sq / n_batches),
            "AU": au_count,
            "lambda_co": get_lambda_co(epoch, total_epochs, self.lambda_co_max),
            "learning_rate": self.scheduler.get_lr(),
        }

        if self.loss_mode == "F":
            metrics["beta_t"] = get_beta_t(epoch, total_epochs, self.warmup_fraction)

        return metrics

    def validate(
        self,
        val_loader: DataLoader[Any],
    ) -> float:
        """
        Compute validation ELBO (INV-011: excludes γ and λ_co, includes σ²).

        :param val_loader (DataLoader): Validation data loader

        :return val_elbo (float): Validation ELBO (lower is better)
        """
        self.model.eval()
        _nb = self._is_cuda
        total_elbo = torch.tensor(0.0, device=self.device)
        n_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device, non_blocking=_nb)
                else:
                    x = batch_data.to(self.device, non_blocking=_nb)

                with torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
                    device_type=self._amp_device_type,
                    dtype=self._amp_dtype,
                    enabled=self.use_amp,
                ):
                    x_hat, mu, log_var = self.model(x)

                    elbo = compute_validation_elbo(
                        x=x,
                        x_hat=x_hat,
                        mu=mu,
                        log_var=log_var,
                        log_sigma_sq=self.model.log_sigma_sq,
                    )

                total_elbo += elbo
                n_batches += 1

        return float(total_elbo / max(1, n_batches))

    def fit(
        self,
        train_windows: torch.Tensor,
        val_windows: torch.Tensor,
        max_epochs: int = 100,
        batch_size: int = 256,
        crisis_fractions: torch.Tensor | None = None,
        raw_returns: torch.Tensor | None = None,
        window_metadata: pd.DataFrame | None = None,
        strata: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Full training loop with early stopping and LR scheduling.

        :param train_windows (torch.Tensor): Training windows (N_train, T, F)
        :param val_windows (torch.Tensor): Validation windows (N_val, T, F)
        :param max_epochs (int): Maximum number of epochs
        :param batch_size (int): Batch size
        :param crisis_fractions (torch.Tensor | None): f_c per training window
        :param raw_returns (torch.Tensor | None): Raw returns per window (N_train, T)
            for co-movement Spearman computation (ISD MOD-004)
        :param window_metadata (pd.DataFrame | None): Window metadata with
            stock_id, start_date, end_date for synchronous batching (INV-010)
        :param strata (np.ndarray | None): Stratum assignment per window (N_train,)
            from k-means clustering on trailing returns

        :return result (dict): Training results with best_epoch, best_val_elbo, history
        """
        # Pre-move crisis_fractions to device (avoids per-batch transfer)
        if crisis_fractions is not None:
            crisis_fractions = crisis_fractions.to(self.device)

        # GPU pre-pin: move all tensors to device before DataLoader creation
        # Eliminates per-batch CPU→GPU transfer (requires num_workers=0)
        if self._is_cuda:
            train_windows = train_windows.to(self.device)
            val_windows = val_windows.to(self.device)
            if raw_returns is not None:
                raw_returns = raw_returns.to(self.device)
            dl_kwargs: dict[str, object] = {"num_workers": 0, "pin_memory": False}
        else:
            dl_kwargs = get_dataloader_kwargs(self.device)

        # Co-movement enabled: CurriculumBatchSampler (INV-010)
        use_co_movement = (
            raw_returns is not None
            and window_metadata is not None
            and self.lambda_co_max > 0
        )

        _sampler: CurriculumBatchSampler | None = None

        if use_co_movement:
            assert raw_returns is not None  # for pyright
            if strata is None:
                strata = np.zeros(len(train_windows), dtype=int)
            _sampler = CurriculumBatchSampler(
                n_windows=len(train_windows),
                batch_size=batch_size,
                window_metadata=window_metadata,
                strata=strata,
                delta_sync=self.delta_sync,
                synchronous=True,
            )
            train_dataset = TensorDataset(
                train_windows,
                torch.arange(len(train_windows)),
                raw_returns,
            )
            # batch_sampler is mutually exclusive with batch_size/shuffle/drop_last
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=_sampler,
                **dl_kwargs,  # type: ignore[arg-type]
            )
        else:
            train_dataset = TensorDataset(
                train_windows,
                torch.arange(len(train_windows)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                **dl_kwargs,  # type: ignore[arg-type]
            )

        val_dataset = TensorDataset(val_windows)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **dl_kwargs,  # type: ignore[arg-type]
        )

        # Mode F warmup handling
        warmup_epochs = int(self.warmup_fraction * max_epochs) if self.loss_mode == "F" else 0
        if self.loss_mode == "F":
            self.scheduler.mode_f_warmup = True

        history: list[dict[str, float]] = []

        # Single progress bar over total steps (like HuggingFace Trainer)
        steps_per_epoch = len(train_loader)
        total_steps = max_epochs * steps_per_epoch
        progress_bar: tqdm | None = (
            tqdm(total=total_steps, desc="    Training", unit="step")
            if total_steps > 1 else None
        )

        for epoch in range(max_epochs):
            # Mode F: disable scheduler/early_stopping during warmup
            if self.loss_mode == "F" and epoch >= warmup_epochs:
                self.scheduler.mode_f_warmup = False

            # INV-010: switch batching mode per curriculum phase
            if _sampler is not None:
                lambda_co = get_lambda_co(epoch, max_epochs, self.lambda_co_max)
                _sampler.set_synchronous(lambda_co > 0)

            # Train one epoch
            train_metrics = self.train_epoch(
                train_loader, epoch, max_epochs, crisis_fractions,
                progress_bar=progress_bar,
            )

            # Validate
            val_elbo = self.validate(val_loader)
            train_metrics["val_elbo"] = val_elbo
            train_metrics["epoch"] = epoch

            history.append(train_metrics)

            # TensorBoard epoch-level logging (uses global_step for consistent x-axis)
            self._log_tensorboard(train_metrics, self._global_step)

            au = int(train_metrics.get("AU", 0))
            lr_val = train_metrics.get("learning_rate", 0)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{train_metrics['train_loss']:.4f}",
                    val=f"{val_elbo:.4f}",
                    AU=au, lr=f"{lr_val:.1e}",
                    epoch=f"{epoch + 1}/{max_epochs}",
                )
            else:
                logger.info(
                    "      Epoch %d/%d — loss=%.3f val=%.3f AU=%d lr=%.1e",
                    epoch + 1, max_epochs,
                    train_metrics["train_loss"], val_elbo, au, lr_val,
                )

            # Scheduler step (protected during Mode F warmup)
            self.scheduler.step(val_elbo)

            # Early stopping (protected during Mode F warmup)
            if self.loss_mode == "F" and epoch < warmup_epochs:
                # During warmup: track progress but never stop.
                # Do NOT call check() — best_loss recorded under β<1 is not
                # comparable to post-warmup ELBO and would cause premature
                # stopping as soon as warmup ends.
                pass
            elif self.loss_mode == "F" and epoch == warmup_epochs:
                # First post-warmup epoch: seed early stopping with current
                # ELBO so all future comparisons use the β=1 regime.
                self.early_stopping.best_loss = val_elbo
                self.early_stopping.best_epoch = epoch
                self.early_stopping.best_state = None  # will be set on next improvement
                self.early_stopping.counter = 0
                self.early_stopping.check(val_elbo, epoch, self.model)
            else:
                should_stop = self.early_stopping.check(val_elbo, epoch, self.model)
                if should_stop:
                    self.early_stopping.restore_best(self.model)
                    break

        if progress_bar is not None:
            progress_bar.close()

        # Flush TensorBoard (caller is responsible for close() after post-training logging)
        if self._tb_writer is not None:
            self._tb_writer.flush()

        # If loop completed without early stopping, restore best
        if not self.early_stopping.stopped:
            self.early_stopping.restore_best(self.model)

        # Overfit diagnostic
        best_val = self.early_stopping.best_loss
        last_train = history[-1]["train_loss"] if history else 1.0
        overfit_ratio = best_val / max(last_train, 1e-8) if last_train > 0 else 0.0
        overfit_flag = overfit_ratio < 0.85 or overfit_ratio > 1.5

        return {
            "best_epoch": self.early_stopping.best_epoch,
            "best_val_elbo": self.early_stopping.best_loss,
            "history": history,
            "overfit_flag": overfit_flag,
            "overfit_ratio": overfit_ratio,
        }

    def _log_tensorboard(self, metrics: dict[str, float], step: int) -> None:
        """
        Log epoch metrics to TensorBoard (no-op if writer is None).

        :param metrics (dict): Epoch metrics from train_epoch + validate
        :param step (int): Epoch number (used as global step)
        """
        if self._tb_writer is None:
            return

        tag_map = {
            "train_loss": "Loss/total",
            "train_recon": "Loss/reconstruction",
            "train_kl": "Loss/kl_divergence",
            "train_co": "Loss/co_movement",
            "val_elbo": "Validation/ELBO",
            "sigma_sq": "Training/sigma_sq",
            "AU": "Training/active_units",
            "learning_rate": "Training/learning_rate",
            "lambda_co": "Training/lambda_co",
            "beta_t": "Training/beta_t",
        }
        for key, tag in tag_map.items():
            if key in metrics:
                self._tb_writer.add_scalar(tag, metrics[key], step)

    def close(self) -> None:
        """
        Flush and close TensorBoard writer (idempotent).
        Call after any post-training logging is done.
        """
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

    def _compute_batch_au(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> int:
        """
        Compute approximate AU on a single batch (monitoring only).

        :param mu (torch.Tensor): Latent mean (B, K)
        :param log_var (torch.Tensor): Latent log-variance (B, K)

        :return au (int): Number of active units (KL_k > 0.01 nats)
        """
        with torch.no_grad():
            # Marginal KL per dimension: (1/B) Σ_i 0.5(μ²_ik + exp(lv_ik) - lv_ik - 1)
            kl_per_dim = 0.5 * torch.mean(
                mu ** 2 + torch.exp(log_var) - log_var - 1.0,
                dim=0,
            )
            au = int((kl_per_dim > 0.01).sum().item())
        return au
