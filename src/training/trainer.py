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

import copy
import logging
import math
import warnings
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
from src.validation import (
    assert_active_units_valid,
    warn_if_au_collapsed,
    warn_if_loss_component_imbalance,
    warn_loss_explosion,
)
from src.vae.loss import (
    compute_co_movement_loss,
    compute_cross_sectional_loss,
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
        es_min_delta: float = 0.0,
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
        sigma_sq_min: float = 1e-4,
        sigma_sq_max: float = 10.0,
        curriculum_phase1_frac: float = 0.30,
        curriculum_phase2_frac: float = 0.30,
        lambda_cs: float = 0.0,
        cs_n_sample_dates: int = 20,
        feature_weights: list[float] | None = None,
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
        :param es_min_delta (float): Minimum ELBO improvement to reset counter
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
        :param sigma_sq_min (float): Lower clamp for observation variance σ²
        :param sigma_sq_max (float): Upper clamp for observation variance σ²
        :param curriculum_phase1_frac (float): Fraction of epochs for full co-movement
        :param curriculum_phase2_frac (float): Fraction of epochs for linear decay
        :param lambda_cs (float): Cross-sectional R² loss weight
        :param cs_n_sample_dates (int): Number of dates to sample for L_cs
        :param feature_weights (list[float] | None): Per-feature recon weights
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
        self._es_min_delta = es_min_delta
        self.curriculum_phase1_frac = curriculum_phase1_frac
        self.curriculum_phase2_frac = curriculum_phase2_frac
        self.lambda_cs = lambda_cs
        self.cs_n_sample_dates = cs_n_sample_dates
        self.feature_weights = feature_weights
        self._feature_weights_tensor: torch.Tensor | None = None  # cached on device
        self.device = device or get_optimal_device()
        self._is_cuda = self.device.type == "cuda"

        # Backend flags: cuDNN benchmark, TF32 (CUDA); no-op on MPS/CPU
        configure_backend(self.device)

        self.model = self.model.to(self.device)

        # Cache normalized feature_weights tensor on device (Step 7: avoid per-batch allocation)
        if self.feature_weights is not None and len(self.feature_weights) > 0:
            fw = torch.tensor(self.feature_weights, device=self.device, dtype=torch.float32)
            self._feature_weights_tensor = fw / fw.sum()

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
        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=self._es_min_delta,
        )
        self.scheduler = SchedulerWrapper(
            self.optimizer, patience=lr_patience, factor=lr_factor,
        )

        # log_sigma_sq clamping bounds (from config)
        self._log_sigma_sq_min = math.log(sigma_sq_min)
        self._log_sigma_sq_max = math.log(sigma_sq_max)
        self._sigma_sq_min = sigma_sq_min
        self._sigma_sq_max = sigma_sq_max
        self._sigma_sq_bounds_hit_streak = 0  # consecutive epochs hitting bounds

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
    ) -> dict[str, Any]:
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
        epoch_cs = torch.tensor(0.0, device=self.device)
        epoch_recon_per_feature: list[float] | None = None  # Per-feature recon loss
        epoch_log_var_stats: dict[str, Any] | None = None  # VAE posterior diagnostics
        epoch_kl_per_dim_accum: torch.Tensor | None = None  # G1: Per-dimension KL trajectory
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

            # Validate batch is non-empty
            assert x.shape[0] > 0, "Empty batch received in train_epoch"

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

                # Capture log_var stats (for VAE posterior diagnostics)
                # GPU accumulation: single .cpu() at epoch end (Step 5)
                with torch.no_grad():
                    log_var_batch = log_var.detach()
                    lower_bound = -6.0
                    upper_bound = 6.0
                    mean_dim = log_var_batch.mean(dim=0)
                    frac_lower = (log_var_batch <= lower_bound + 0.1).float().mean(dim=0)
                    frac_upper = (log_var_batch >= upper_bound - 0.1).float().mean(dim=0)
                    if epoch_log_var_stats is None:
                        epoch_log_var_stats = {
                            "mean_per_dim_accum": mean_dim,
                            "frac_at_lower_accum": frac_lower,
                            "frac_at_upper_accum": frac_upper,
                            "batch_count": 1,
                        }
                    else:
                        epoch_log_var_stats["mean_per_dim_accum"] += mean_dim
                        epoch_log_var_stats["frac_at_lower_accum"] += frac_lower
                        epoch_log_var_stats["frac_at_upper_accum"] += frac_upper
                        epoch_log_var_stats["batch_count"] += 1

                    # G1: Per-dimension KL trajectory tracking
                    kl_per_dim_batch = 0.5 * torch.mean(
                        mu.detach() ** 2 + torch.exp(log_var_batch) - log_var_batch - 1.0,
                        dim=0,
                    )
                    if epoch_kl_per_dim_accum is None:
                        epoch_kl_per_dim_accum = kl_per_dim_batch
                    else:
                        epoch_kl_per_dim_accum += kl_per_dim_batch

                # Co-movement loss on raw returns (ISD MOD-004: NOT z-scored)
                co_mov_loss: torch.Tensor | None = None
                cur_lambda_co = get_lambda_co(
                    epoch, total_epochs, self.lambda_co_max,
                    phase1_frac=self.curriculum_phase1_frac,
                    phase2_frac=self.curriculum_phase2_frac,
                )
                if raw_ret is not None and cur_lambda_co > 0 and x.shape[0] >= 2:
                    # Validate raw_returns shape matches batch (avoid silent correlation errors)
                    assert raw_ret.shape == (x.shape[0], x.shape[1]), (
                        f"raw_ret shape {raw_ret.shape} != expected ({x.shape[0]}, {x.shape[1]})"
                    )
                    co_mov_loss = compute_co_movement_loss(
                        mu, raw_ret, max_pairs=self.max_pairs,
                    )

                # Cross-sectional R² loss: forces encoder to produce factor
                # exposures (mu) that explain cross-sectional return variation
                cs_loss: torch.Tensor | None = None
                if raw_ret is not None and self.lambda_cs > 0 and x.shape[0] >= mu.shape[1] + 1:
                    cs_loss = compute_cross_sectional_loss(
                        mu, raw_ret,
                        n_sample_dates=self.cs_n_sample_dates,
                    )

                # Compute per-feature diagnostics only at logging intervals or last batch
                _do_diag = (
                    step_in_epoch == 0
                    or (self._logging_steps > 0 and self._global_step % self._logging_steps == 0)
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
                    sigma_sq_min=self._sigma_sq_min,
                    sigma_sq_max=self._sigma_sq_max,
                    curriculum_phase1_frac=self.curriculum_phase1_frac,
                    curriculum_phase2_frac=self.curriculum_phase2_frac,
                    cross_sectional_loss=cs_loss,
                    lambda_cs=self.lambda_cs,
                    feature_weights=self.feature_weights,
                    feature_weights_tensor=self._feature_weights_tensor,
                    compute_diagnostics=_do_diag,
                )

            # Loss explosion warning
            if loss.item() > 1e6:
                warnings.warn(
                    f"Loss explosion detected: {loss.item():.2e} at epoch {epoch}, "
                    f"step {step_in_epoch}. Check learning rate or data quality.",
                    stacklevel=2,
                )

            # Backward pass with gradient accumulation
            if torch.isnan(loss) or torch.isinf(loss):
                # Skip backward pass for NaN/Inf loss to prevent corrupting model weights
                logger.warning(
                    "Train step %d: loss is %s (epoch %d). Skipping backward pass. "
                    "Check model outputs for NaN (float16 overflow, torch.compile issue).",
                    step_in_epoch, "NaN" if torch.isnan(loss) else "Inf", epoch,
                )
                n_batches += 1
                self._global_step += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

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
                # Log pre-clip gradient norm for diagnostics
                pre_clip_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self._tb_writer is not None and self._logging_steps > 0 and self._global_step % self._logging_steps == 0:
                    self._tb_writer.add_scalar("Step/grad_norm_pre_clip", float(pre_clip_norm), self._global_step)

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
            epoch_cs += components["cross_sec"]

            # Accumulate per-feature reconstruction loss (CPU list)
            rpf = components.get("recon_per_feature")
            if rpf is not None and len(rpf) > 0:
                if epoch_recon_per_feature is None:
                    epoch_recon_per_feature = [0.0] * len(rpf)
                for i, val in enumerate(rpf):
                    epoch_recon_per_feature[i] += val
            n_batches += 1
            self._global_step += 1

            # Per-step TensorBoard logging (batch .item() calls into single sync)
            if (
                self._tb_writer is not None
                and self._logging_steps > 0
                and self._global_step % self._logging_steps == 0
            ):
                # Stack detached scalars → single .cpu() → single sync point
                tb_keys = ["total", "recon", "kl", "co_mov", "sigma_sq"]
                tb_vals = torch.stack([components[k] for k in tb_keys]).cpu().tolist()
                tb_tags = ["Step/loss", "Step/reconstruction", "Step/kl_divergence",
                           "Step/co_movement", "Step/sigma_sq"]
                for tag, val in zip(tb_tags, tb_vals):
                    self._tb_writer.add_scalar(tag, val, self._global_step)

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

        # VALIDATION: AU must be within valid range [0, K]
        assert_active_units_valid(au_count, self.model.K, f"train_epoch_{epoch}")

        # Single sync point: convert GPU accumulators to Python floats
        metrics: dict[str, Any] = {
            "train_loss": float(epoch_loss / n_batches),
            "train_recon": float(epoch_recon / n_batches),
            "train_kl": float(epoch_kl / n_batches),
            "train_co": float(epoch_co / n_batches),
            "train_cs": float(epoch_cs / n_batches),
            "sigma_sq": float(epoch_sigma_sq / n_batches),
            "AU": au_count,
            "lambda_co": get_lambda_co(
                epoch, total_epochs, self.lambda_co_max,
                phase1_frac=self.curriculum_phase1_frac,
                phase2_frac=self.curriculum_phase2_frac,
            ),
            "learning_rate": self.scheduler.get_lr(),
            "recon_per_feature": [
                v / n_batches for v in epoch_recon_per_feature
            ] if epoch_recon_per_feature is not None else [],
        }

        # Add log_var statistics for VAE posterior diagnostics
        # Single GPU→CPU transfer at epoch end (Step 5)
        if epoch_log_var_stats is not None:
            bc = epoch_log_var_stats["batch_count"]
            metrics["log_var_stats"] = {
                "mean_per_dim": (epoch_log_var_stats["mean_per_dim_accum"] / bc).cpu().tolist(),
                "frac_at_lower": (epoch_log_var_stats["frac_at_lower_accum"] / bc).cpu().tolist(),
                "frac_at_upper": (epoch_log_var_stats["frac_at_upper_accum"] / bc).cpu().tolist(),
            }

        # G1: Per-dimension KL trajectory — averaged across batches (single .cpu() at epoch end)
        if epoch_kl_per_dim_accum is not None:
            kl_per_dim = (epoch_kl_per_dim_accum / n_batches).cpu().tolist()
            n_collapsed = sum(1 for kl in kl_per_dim if kl < 0.01)
            metrics["kl_per_dim"] = kl_per_dim
            metrics["n_collapsed_dims"] = n_collapsed

        # Fix 5: Monitor σ² hitting bounds (DVT monitoring prescription)
        avg_sigma_sq = float(epoch_sigma_sq / n_batches)
        _eps = 1e-6
        hitting_bounds = (
            abs(avg_sigma_sq - self._sigma_sq_min) < _eps * self._sigma_sq_min
            or abs(avg_sigma_sq - self._sigma_sq_max) < _eps * self._sigma_sq_max
        )
        if hitting_bounds:
            self._sigma_sq_bounds_hit_streak += 1
            # Log at milestones only: 5, 10, 20, 50, 100, 200, ...
            if self._sigma_sq_bounds_hit_streak in (5, 10, 20, 50, 100, 200, 500):
                logger.warning(
                    "σ² has been clamped at bounds (%.4g) for %d consecutive epochs. "
                    "Current bounds: [%.4g, %.4g]. Consider adjusting sigma_sq_min/max.",
                    avg_sigma_sq, self._sigma_sq_bounds_hit_streak,
                    self._sigma_sq_min, self._sigma_sq_max,
                )
        else:
            self._sigma_sq_bounds_hit_streak = 0

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
        n_nan_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device, non_blocking=_nb)
                else:
                    x = batch_data.to(self.device, non_blocking=_nb)

                # Model forward pass under AMP autocast (float16 convolutions)
                with torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
                    device_type=self._amp_device_type,
                    dtype=self._amp_dtype,
                    enabled=self.use_amp,
                ):
                    x_hat, mu, log_var = self.model(x)

                # ELBO computed OUTSIDE autocast in float32 for numerical stability
                # (loss functions cast inputs to float32 internally)
                elbo = compute_validation_elbo(
                    x=x,
                    x_hat=x_hat,
                    mu=mu,
                    log_var=log_var,
                    log_sigma_sq=self.model.log_sigma_sq,
                    sigma_sq_min=self._sigma_sq_min,
                    sigma_sq_max=self._sigma_sq_max,
                )

                if torch.isnan(elbo) or torch.isinf(elbo):
                    n_nan_batches += 1
                else:
                    total_elbo += elbo
                n_batches += 1

        if n_nan_batches > 0:
            logger.warning(
                "Validation: %d/%d batches produced NaN/Inf ELBO. "
                "Check AMP settings, sigma_sq bounds, or data quality.",
                n_nan_batches, n_batches,
            )

        if n_nan_batches == n_batches:
            return float("nan")

        valid_batches = max(1, n_batches - n_nan_batches)
        return float(total_elbo / valid_batches)

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
                lambda_co = get_lambda_co(
                    epoch, max_epochs, self.lambda_co_max,
                    phase1_frac=self.curriculum_phase1_frac,
                    phase2_frac=self.curriculum_phase2_frac,
                )
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

            # Warn if AU ratio is outside normal range (collapsed or no regularization)
            K = self.model.K if hasattr(self.model, 'K') else 200
            warn_if_au_collapsed(au, K, min_ratio=0.05, max_ratio=0.80, name="AU")

            # Warn if loss components are severely imbalanced
            kl_loss = float(train_metrics.get("kl_loss", 0))
            recon_loss = float(train_metrics.get("recon_loss", 0))
            warn_if_loss_component_imbalance(kl_loss, recon_loss, max_ratio=100.0, name="loss")

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
                # Capture state dict now — if this epoch is optimal (no future
                # improvement), restore_best() needs a valid checkpoint.
                self.early_stopping.best_loss = val_elbo
                self.early_stopping.best_epoch = epoch
                self.early_stopping.best_state = copy.deepcopy(self.model.state_dict())
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

        # Overfit diagnostic: symmetric ELBO comparison at best epoch (E*)
        # Compute train_elbo using the same formula as val_elbo (excludes γ, λ_co)
        # to ensure the ratio compares like-for-like quantities.
        best_val = self.early_stopping.best_loss
        best_epoch = self.early_stopping.best_epoch

        # Reuse train_dataset (already exists) to avoid ~200MB TensorDataset copy.
        # Create a non-shuffled loader that only extracts windows (index 0).
        train_elbo_loader = DataLoader(
            TensorDataset(train_windows),  # Minimal dataset with windows only
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **dl_kwargs,  # type: ignore[arg-type]
        )
        train_elbo = self.validate(train_elbo_loader)

        overfit_ratio = best_val / max(train_elbo, 1e-8) if train_elbo > 0 else 0.0
        # Thresholds aligned with diagnostics.py health checks (warn=1.3, crit=1.8)
        overfit_flag = overfit_ratio < 0.85 or overfit_ratio > 1.3

        if overfit_flag:
            logger.warning(
                "Overfitting detected: val_elbo/train_elbo ratio=%.2f at best epoch %d "
                "(best_val=%.1f, train_elbo=%.1f). "
                "Consider: increase weight_decay, add dropout, "
                "reduce max_epochs, or try Mode P (learnable sigma_sq).",
                overfit_ratio, best_epoch, best_val, train_elbo,
            )

        return {
            "best_epoch": self.early_stopping.best_epoch,
            "best_val_elbo": self.early_stopping.best_loss,
            "train_elbo": train_elbo,
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
            "train_cs": "Loss/cross_sectional",
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

    def cleanup(self) -> None:
        """
        Release all resources held by trainer after training completes.

        Call this after fit() to free GPU memory before checkpoint saving.
        Releases:
        - TensorBoard writer (~10MB)
        - torch.compile cache (~0.5-2GB)
        - Early stopping best_state (~20MB)
        - Optimizer momentum buffers (~40MB)
        - CUDA/MPS cached allocations

        Total freed: ~0.5-2GB depending on torch.compile usage.
        """
        # Close TensorBoard
        self.close()

        # Clear torch.compile cache (can hold ~0.5-2GB of traced graphs)
        if hasattr(torch, "_dynamo"):
            try:
                torch._dynamo.reset()  # type: ignore[attr-defined]
            except Exception:
                pass  # Ignore errors if dynamo not initialized

        # Clear early stopping state (already None if restore_best was called)
        if hasattr(self, "early_stopping") and self.early_stopping.best_state is not None:
            self.early_stopping.best_state = None

        # Clear optimizer state (momentum buffers)
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad(set_to_none=True)
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    if hasattr(param, "grad") and param.grad is not None:
                        param.grad = None

        # Clear GradScaler state
        if self.scaler is not None:
            # GradScaler doesn't have a close method, but we can clear internal state
            self.scaler = None

        # Clear CUDA/MPS cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        logger.info("Trainer cleanup completed")

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
