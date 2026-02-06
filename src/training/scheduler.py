"""
Learning rate scheduler wrapper.

ReduceLROnPlateau monitoring validation ELBO.
Disabled during Mode F warmup (INV-006).

Reference: ISD Section MOD-005 — Sub-task 2.
"""

import torch.optim as optim


class SchedulerWrapper:
    """
    Wrapper around ReduceLROnPlateau with warmup protection for Mode F.

    Attributes:
        scheduler: ReduceLROnPlateau instance
        mode_f_warmup: bool — True during Mode F warmup (scheduler disabled)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-7,
    ) -> None:
        """
        :param optimizer (optim.Optimizer): The optimizer to schedule
        :param patience (int): Epochs to wait before reducing LR
        :param factor (float): Factor to reduce LR by
        :param min_lr (float): Minimum learning rate
        """
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor,
            min_lr=min_lr,
        )
        self.mode_f_warmup = False

    def step(self, val_loss: float) -> None:
        """
        Step the scheduler with the validation loss.

        No-op during Mode F warmup.

        :param val_loss (float): Validation ELBO
        """
        if not self.mode_f_warmup:
            self.scheduler.step(val_loss)

    def get_lr(self) -> float:
        """
        Get the current learning rate.

        :return lr (float): Current learning rate
        """
        return self.scheduler.optimizer.param_groups[0]["lr"]
