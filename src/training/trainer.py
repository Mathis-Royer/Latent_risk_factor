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

import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.vae.loss import (
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
        """
        self.model = model
        self.loss_mode = loss_mode
        self.gamma = gamma
        self.lambda_co_max = lambda_co_max
        self.beta_fixed = beta_fixed
        self.warmup_fraction = warmup_fraction
        self.device = device or torch.device("cpu")

        self.model = self.model.to(self.device)

        # Optimizer: AdamW with weight_decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        # Early stopping and scheduler
        self.early_stopping = EarlyStopping(patience=patience)
        self.scheduler = SchedulerWrapper(
            self.optimizer, patience=lr_patience, factor=lr_factor,
        )

        # log_sigma_sq clamping bounds
        self._log_sigma_sq_min = math.log(1e-4)
        self._log_sigma_sq_max = math.log(10.0)

    def train_epoch(
        self,
        train_loader: DataLoader[Any],
        epoch: int,
        total_epochs: int,
        crisis_fractions: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Run one training epoch.

        :param train_loader (DataLoader): Training data loader
        :param epoch (int): Current epoch (0-indexed)
        :param total_epochs (int): Total number of epochs
        :param crisis_fractions (torch.Tensor | None): f_c per window (N,)

        :return metrics (dict): Epoch-level training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_co = 0.0
        epoch_sigma_sq = 0.0
        n_batches = 0

        for batch_data in train_loader:
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0].to(self.device)
                batch_indices = batch_data[1] if len(batch_data) > 1 else None
            else:
                x = batch_data.to(self.device)
                batch_indices = None

            # Get crisis fractions for this batch
            if crisis_fractions is not None and batch_indices is not None:
                cf = crisis_fractions[batch_indices].to(self.device)
            elif crisis_fractions is not None:
                # Assume sequential access, take first B elements
                B = x.shape[0]
                cf = crisis_fractions[:B].to(self.device)
            else:
                cf = torch.zeros(x.shape[0], device=self.device)

            # Forward pass
            x_hat, mu, log_var = self.model(x)

            # Compute loss
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
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clamp log_sigma_sq after optimizer step
            with torch.no_grad():
                self.model.log_sigma_sq.clamp_(
                    self._log_sigma_sq_min, self._log_sigma_sq_max,
                )

            # Accumulate metrics
            epoch_loss += components["total"]
            epoch_recon += components["recon"]
            epoch_kl += components["kl"]
            epoch_co += components["co_mov"]
            epoch_sigma_sq += components["sigma_sq"]
            n_batches += 1

        n_batches = max(1, n_batches)

        # Compute AU on the last batch (approximate monitoring)
        au_count = self._compute_batch_au(mu, log_var)

        metrics = {
            "train_loss": epoch_loss / n_batches,
            "train_recon": epoch_recon / n_batches,
            "train_kl": epoch_kl / n_batches,
            "train_co": epoch_co / n_batches,
            "sigma_sq": epoch_sigma_sq / n_batches,
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
        total_elbo = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)

                x_hat, mu, log_var = self.model(x)

                elbo = compute_validation_elbo(
                    x=x,
                    x_hat=x_hat,
                    mu=mu,
                    log_var=log_var,
                    log_sigma_sq=self.model.log_sigma_sq,
                )

                total_elbo += elbo.item()
                n_batches += 1

        return total_elbo / max(1, n_batches)

    def fit(
        self,
        train_windows: torch.Tensor,
        val_windows: torch.Tensor,
        max_epochs: int = 100,
        batch_size: int = 256,
        crisis_fractions: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Full training loop with early stopping and LR scheduling.

        :param train_windows (torch.Tensor): Training windows (N_train, T, F)
        :param val_windows (torch.Tensor): Validation windows (N_val, T, F)
        :param max_epochs (int): Maximum number of epochs
        :param batch_size (int): Batch size
        :param crisis_fractions (torch.Tensor | None): f_c per training window

        :return result (dict): Training results with best_epoch, best_val_elbo, history
        """
        # Create data loaders
        train_dataset = TensorDataset(
            train_windows,
            torch.arange(len(train_windows)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dataset = TensorDataset(val_windows)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Mode F warmup handling
        warmup_epochs = int(self.warmup_fraction * max_epochs) if self.loss_mode == "F" else 0
        if self.loss_mode == "F":
            self.scheduler.mode_f_warmup = True

        history: list[dict[str, float]] = []

        for epoch in range(max_epochs):
            # Mode F: disable scheduler/early_stopping during warmup
            if self.loss_mode == "F" and epoch >= warmup_epochs:
                self.scheduler.mode_f_warmup = False

            # Train one epoch
            train_metrics = self.train_epoch(
                train_loader, epoch, max_epochs, crisis_fractions,
            )

            # Validate
            val_elbo = self.validate(val_loader)
            train_metrics["val_elbo"] = val_elbo
            train_metrics["epoch"] = epoch

            history.append(train_metrics)

            # Scheduler step (protected during Mode F warmup)
            self.scheduler.step(val_elbo)

            # Early stopping (protected during Mode F warmup)
            if self.loss_mode == "F" and epoch < warmup_epochs:
                # During warmup, just record best without stopping
                self.early_stopping.check(val_elbo, epoch, self.model)
                self.early_stopping.counter = 0  # Reset counter during warmup
            else:
                should_stop = self.early_stopping.check(val_elbo, epoch, self.model)
                if should_stop:
                    self.early_stopping.restore_best(self.model)
                    break

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
