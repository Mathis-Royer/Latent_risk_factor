"""
Early stopping with best checkpoint restoration.

Monitors validation ELBO (lower is better). Stops after `patience`
consecutive epochs without improvement. Restores best weights.

Reference: ISD Section MOD-005 — Sub-task 3.
"""

import copy
import logging
import math

import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping with best checkpoint restore.

    Attributes:
        patience: int — epochs without improvement before stopping
        min_delta: float — minimum ELBO improvement to reset counter
        best_loss: float — best validation ELBO seen
        best_epoch: int — epoch of the best checkpoint (E*)
        counter: int — epochs since last improvement
        best_state: dict — best model state_dict (deep copy)
        stopped: bool — whether early stopping was triggered
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """
        :param patience (int): Number of epochs without improvement before stopping
        :param min_delta (float): Minimum absolute ELBO decrease to count as
            improvement. E.g. min_delta=0.5 means val_loss must drop by at
            least 0.5 below best_loss to reset the counter.
        """
        self.patience = patience
        self.min_delta = max(0.0, min_delta)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state: dict | None = None
        self.stopped = False
        self._nan_streak = 0

    def check(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """
        Check if training should stop.

        NaN val_loss is treated as non-improvement (counter increments).
        Consecutive NaN epochs are tracked and logged.

        :param val_loss (float): Current validation ELBO
        :param epoch (int): Current epoch number
        :param model (nn.Module): Model to checkpoint

        :return should_stop (bool): True if patience exhausted
        """
        if math.isnan(val_loss) or math.isinf(val_loss):
            self._nan_streak += 1
            if self._nan_streak in (1, 5, 10, 20):
                logger.warning(
                    "val_loss is %s at epoch %d (%d consecutive). "
                    "Possible causes: AMP float16 overflow, exploding gradients, "
                    "or degenerate data. Counter: %d/%d.",
                    "NaN" if math.isnan(val_loss) else "Inf",
                    epoch, self._nan_streak, self.counter + 1, self.patience,
                )
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
            return False

        self._nan_streak = 0

        if val_loss < self.best_loss - self.min_delta:
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
            assert set(self.best_state.keys()) == set(model.state_dict().keys()), (
                "Checkpoint keys mismatch: model architecture changed since best checkpoint"
            )
            model.load_state_dict(self.best_state)
