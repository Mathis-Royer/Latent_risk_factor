"""
Phase B: Deployment run.

Re-train encoder on ALL data [start, train_end] for E* epochs
(no validation set, no early stopping). Then run the full downstream
pipeline: AU → B_A → Σ_z → D_ε → portfolio.

Reference: ISD Section MOD-009 — Sub-task 3.
"""

from typing import Any

import numpy as np


def determine_e_star(
    fold_id: int,
    e_star_config: int,
    previous_e_stars: list[int],
    is_holdout: bool = False,
    all_e_stars: list[int] | None = None,
) -> int:
    """
    Determine E* for Phase B deployment.

    - Per-fold: E* = E*_config from Phase A
    - Robust alternative: median of E* across previous folds (expanding)
    - Holdout: median of E* across ALL walk-forward folds

    :param fold_id (int): Current fold ID
    :param e_star_config (int): E* from the selected config in Phase A
    :param previous_e_stars (list[int]): E* from all previous folds
    :param is_holdout (bool): True for holdout fold
    :param all_e_stars (list[int] | None): E* from all walk-forward folds

    :return E_star (int): Number of epochs for deployment training
    """
    assert e_star_config > 0, f"e_star_config must be > 0, got {e_star_config}"

    if is_holdout and all_e_stars:
        return int(np.median(all_e_stars))

    if previous_e_stars and len(previous_e_stars) >= 3:
        # Robust: median of previous folds (CONV-10 point-in-time)
        return int(np.median(previous_e_stars))

    # Default: use the config's E*
    return e_star_config


def check_training_sanity(
    phase_b_loss: float,
    phase_a_loss: float,
    threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Sanity check: if Phase B training loss at E* is > 20% lower
    than Phase A, flag the fold.

    :param phase_b_loss (float): Phase B training loss at E*
    :param phase_a_loss (float): Phase A training loss at best epoch
    :param threshold (float): Relative difference threshold

    :return check (dict): flag and relative difference
    """
    if phase_a_loss == 0:
        return {"flag": False, "relative_diff": 0.0}

    relative_diff = (phase_a_loss - phase_b_loss) / abs(phase_a_loss)

    return {
        "flag": bool(relative_diff > threshold),
        "relative_diff": float(relative_diff),
        "phase_a_loss": phase_a_loss,
        "phase_b_loss": phase_b_loss,
    }
