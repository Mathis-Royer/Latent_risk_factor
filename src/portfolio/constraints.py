"""
Portfolio constraints and penalty functions.

Concentration penalty: P_conc(w) = Σ_i max(0, w_i - w̄)²
Turnover penalty (Almgren-Chriss):
  P_turn = κ₁ · (1/2) Σ|w_i - w_old_i| + κ₂ · Σ max(0, |w_i - w_old_i| - δ̄)²

Hard constraints:
  - w_i ≥ 0 (long-only)
  - Σ w_i = 1 (fully invested)
  - w_i ≤ w_max (5%)
  - (1/2)||w - w_old||_1 ≤ τ_max (30%)

First rebalancing: κ₁ = κ₂ = 0.

Reference: ISD Section MOD-008 — Sub-task 4.
"""

import numpy as np


def concentration_penalty(
    w: np.ndarray,
    w_bar: float = 0.03,
) -> float:
    """
    Concentration penalty: P_conc(w) = Σ_i max(0, w_i - w̄)²

    :param w (np.ndarray): Weights (n,)
    :param w_bar (float): Concentration threshold (3%)

    :return P_conc (float): Penalty value
    """
    excess = np.maximum(0.0, w - w_bar)
    return float(np.sum(excess ** 2))


def concentration_penalty_gradient(
    w: np.ndarray,
    w_bar: float = 0.03,
) -> np.ndarray:
    """
    Gradient of concentration penalty.

    ∂P_conc/∂w_i = 2 · max(0, w_i - w̄)

    :param w (np.ndarray): Weights (n,)
    :param w_bar (float): Concentration threshold

    :return grad (np.ndarray): Gradient (n,)
    """
    return 2.0 * np.maximum(0.0, w - w_bar)


def turnover_penalty(
    w: np.ndarray,
    w_old: np.ndarray,
    kappa_1: float = 0.1,
    kappa_2: float = 7.5,
    delta_bar: float = 0.01,
    is_first: bool = False,
) -> float:
    """
    Almgren-Chriss turnover penalty.

    P_turn = κ₁ · (1/2) Σ|Δw_i| + κ₂ · Σ max(0, |Δw_i| - δ̄)²

    First rebalancing: κ₁ = κ₂ = 0.

    :param w (np.ndarray): Current weights (n,)
    :param w_old (np.ndarray): Previous weights (n,)
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover penalty threshold
    :param is_first (bool): True for first rebalancing (κ₁=κ₂=0)

    :return P_turn (float): Penalty value
    """
    if is_first:
        return 0.0

    delta_w = np.abs(w - w_old)
    linear_part = kappa_1 * 0.5 * np.sum(delta_w)
    excess = np.maximum(0.0, delta_w - delta_bar)
    quadratic_part = kappa_2 * np.sum(excess ** 2)

    return float(linear_part + quadratic_part)


def check_hard_constraints(
    w: np.ndarray,
    w_old: np.ndarray | None,
    w_max: float = 0.05,
    tau_max: float = 0.30,
    tol: float = 1e-6,
) -> dict[str, bool]:
    """
    Check if hard constraints are satisfied.

    :param w (np.ndarray): Weights (n,)
    :param w_old (np.ndarray | None): Previous weights
    :param w_max (float): Maximum weight per stock
    :param tau_max (float): Maximum one-way turnover
    :param tol (float): Numerical tolerance

    :return status (dict): Constraint satisfaction booleans
    """
    status = {
        "long_only": bool(np.all(w >= -tol)),
        "fully_invested": bool(abs(np.sum(w) - 1.0) < tol),
        "weight_cap": bool(np.all(w <= w_max + tol)),
    }

    if w_old is not None:
        one_way_turnover = 0.5 * np.sum(np.abs(w - w_old))
        status["turnover_cap"] = bool(one_way_turnover <= tau_max + tol)
    else:
        status["turnover_cap"] = True

    return status


def project_to_constraints(
    w: np.ndarray,
    w_max: float = 0.05,
    w_min: float = 0.001,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Simple iterative projection onto hard constraints.

    Clip w_max, zero < w_min, renormalize. Iterate 2-3 passes.

    :param w (np.ndarray): Raw weights (n,)
    :param w_max (float): Maximum weight
    :param w_min (float): Minimum active weight
    :param max_iterations (int): Max projection iterations

    :return w_proj (np.ndarray): Projected weights (n,)
    """
    w = w.copy()

    for _ in range(max_iterations):
        # Clip to [0, w_max]
        w = np.clip(w, 0.0, w_max)

        # Zero out weights below w_min
        w[w < w_min] = 0.0

        # Renormalize
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            # Fallback to equal weight
            n = len(w)
            w = np.ones(n) / n

        # Check convergence
        if np.all(w <= w_max + 1e-8) and np.all((w == 0) | (w >= w_min - 1e-8)):
            break

    return w
