"""
Early stopping with best checkpoint restoration.

Monitors validation ELBO (lower is better). Stops after `patience`
consecutive epochs without improvement. Restores best weights.

Reference: ISD Section MOD-005 — Sub-task 3.
"""

import copy

import torch.nn as nn


class EarlyStopping:
    """
    Early stopping with best checkpoint restore.

    Attributes:
        patience: int — epochs without improvement before stopping
        best_loss: float — best validation ELBO seen
        best_epoch: int — epoch of the best checkpoint (E*)
        counter: int — epochs since last improvement
        best_state: dict — best model state_dict (deep copy)
        stopped: bool — whether early stopping was triggered
    """

    def __init__(self, patience: int = 10) -> None:
        """
        :param patience (int): Number of epochs without improvement before stopping
        """
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state: dict | None = None
        self.stopped = False

    def check(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """
        Check if training should stop.

        :param val_loss (float): Current validation ELBO
        :param epoch (int): Current epoch number
        :param model (nn.Module): Model to checkpoint

        :return should_stop (bool): True if patience exhausted
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True

        return False

    def restore_best(self, model: nn.Module) -> None:
        """
        Restore best checkpoint weights to the model.

        :param model (nn.Module): Model to restore
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
