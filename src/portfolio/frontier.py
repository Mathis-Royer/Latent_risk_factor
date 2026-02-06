"""
Variance-entropy frontier computation.

For each α in a grid, solve the optimization (μ=0) and record
(variance, entropy, n_active_positions). The elbow of the frontier
determines the operating point.

Reference: ISD Section MOD-008 — Sub-task 5.
"""

import numpy as np
import pandas as pd

from src.portfolio.entropy import compute_entropy_only
from src.portfolio.sca_solver import multi_start_optimize


def compute_variance_entropy_frontier(
    Sigma_assets: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray,
    alpha_grid: list[float] | None = None,
    lambda_risk: float = 1.0,
    w_max: float = 0.05,
    w_min: float = 0.001,
    w_bar: float = 0.03,
    phi: float = 25.0,
    w_old: np.ndarray | None = None,
    kappa_1: float = 0.1,
    kappa_2: float = 7.5,
    delta_bar: float = 0.01,
    tau_max: float = 0.30,
    is_first: bool = True,
    n_starts: int = 5,
    seed: int = 42,
    entropy_eps: float = 1e-30,
) -> pd.DataFrame:
    """
    Compute variance-entropy frontier for a grid of α values.

    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)
    :param alpha_grid (list[float] | None): Grid of α values
    :param lambda_risk (float): Risk aversion
    :param w_max (float): Maximum weight
    :param w_min (float): Minimum active weight
    :param w_bar (float): Concentration threshold
    :param phi (float): Concentration penalty weight
    :param w_old (np.ndarray | None): Previous weights
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover threshold
    :param tau_max (float): Maximum one-way turnover
    :param is_first (bool): First rebalancing flag
    :param n_starts (int): Multi-start count
    :param seed (int): Random seed
    :param entropy_eps (float): Numerical stability

    :return frontier (pd.DataFrame): Columns: alpha, variance, entropy, n_active
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    results: list[dict[str, float]] = []

    for alpha in alpha_grid:
        w_opt, f_opt, H_opt = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=alpha,
            n_starts=n_starts,
            seed=seed,
            lambda_risk=lambda_risk,
            w_max=w_max,
            w_min=w_min,
            w_bar=w_bar,
            phi=phi,
            w_old=w_old,
            kappa_1=kappa_1,
            kappa_2=kappa_2,
            delta_bar=delta_bar,
            tau_max=tau_max,
            is_first=is_first,
            entropy_eps=entropy_eps,
        )

        variance = float(w_opt @ Sigma_assets @ w_opt)
        entropy = compute_entropy_only(w_opt, B_prime, eigenvalues, entropy_eps)
        n_active = int(np.sum(w_opt > w_min))

        results.append({
            "alpha": alpha,
            "variance": variance,
            "entropy": entropy,
            "n_active": float(n_active),
        })

    return pd.DataFrame(results)


def select_operating_alpha(
    frontier: pd.DataFrame,
    delta_H_threshold: float = 0.1,
) -> float:
    """
    Automatic selection of α at the elbow of the variance-entropy frontier.

    Select α where ΔH/ΔVar < threshold (diminishing entropy returns).

    :param frontier (pd.DataFrame): Frontier with columns alpha, variance, entropy
    :param delta_H_threshold (float): Threshold for ΔH/ΔVar

    :return alpha_opt (float): Selected operating α
    """
    if len(frontier) < 2:
        return frontier["alpha"].iloc[0] if len(frontier) > 0 else 0.1

    # Sort by alpha
    df = frontier.sort_values("alpha").reset_index(drop=True)

    for i in range(1, len(df)):
        dH = df["entropy"].iloc[i] - df["entropy"].iloc[i - 1]
        dVar = df["variance"].iloc[i] - df["variance"].iloc[i - 1]

        if dVar == 0:
            continue

        ratio = abs(dH / dVar)
        if ratio < delta_H_threshold:
            return float(df["alpha"].iloc[i - 1])

    # Default: last alpha in grid
    return float(df["alpha"].iloc[-1])
