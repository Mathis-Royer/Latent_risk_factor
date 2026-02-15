"""
Variance-entropy frontier computation.

For each α in a grid, solve the optimization and record
(variance, entropy, n_active_positions). The elbow of the frontier
determines the operating point.

Supports optional expected return signal μ (e.g. cross-sectional momentum).

Reference: ISD Section MOD-008 — Sub-task 5.
"""

import logging

import numpy as np
import pandas as pd

from src.portfolio.entropy import compute_entropy_only
from src.portfolio.sca_solver import multi_start_optimize

logger = logging.getLogger(__name__)


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
    mu: np.ndarray | None = None,
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
    :param mu (np.ndarray | None): Expected return signal (n,)

    :return frontier (pd.DataFrame): Columns: alpha, variance, entropy, n_active
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    results: list[dict[str, float]] = []
    n_alphas = len(alpha_grid)

    for idx, alpha in enumerate(alpha_grid):
        logger.info("    Frontier alpha %d/%d (alpha=%.3f)...", idx + 1, n_alphas, alpha)
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
            mu=mu,
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
) -> float:
    """
    Select α at the elbow of the variance-entropy frontier (Kneedle method).

    Normalizes both axes to [0, 1], then finds the frontier point with maximum
    perpendicular distance from the chord connecting the first (min-var) and
    last (max-entropy) points. This is scale-invariant and requires no
    arbitrary threshold.

    Fallback: if the frontier is degenerate (flat or single-point), returns
    the α that achieves maximum entropy.

    Reference: Satopaa et al. (2011), "Finding a 'Kneedle' in a Haystack".
    DVT §4.7: "The elbow of the variance-entropy frontier is the natural
    operating point."

    :param frontier (pd.DataFrame): Frontier with columns alpha, variance, entropy

    :return alpha_opt (float): Selected operating α
    """
    if len(frontier) < 2:
        return float(frontier["alpha"].iloc[0]) if len(frontier) > 0 else 0.1

    df = frontier.sort_values("alpha").reset_index(drop=True)

    H = np.asarray(df["entropy"], dtype=np.float64)
    V = np.asarray(df["variance"], dtype=np.float64)

    H_range = float(H.max() - H.min())
    V_range = float(V.max() - V.min())

    # Degenerate: flat frontier (all points have same H or same Var)
    if H_range < 1e-15 or V_range < 1e-15:
        best_idx = int(np.argmax(H))
        alpha_sel = float(df["alpha"].iloc[best_idx])
        logger.info(
            "select_operating_alpha: degenerate frontier, selecting α=%.4g "
            "(max entropy).", alpha_sel,
        )
        return alpha_sel

    # Normalize both axes to [0, 1]
    H_n = (H - H.min()) / H_range
    V_n = (V - V.min()) / V_range

    # Chord from first to last point in normalized space
    dx = V_n[-1] - V_n[0]
    dy = H_n[-1] - H_n[0]
    chord_len = float(np.sqrt(dx * dx + dy * dy))

    if chord_len < 1e-15:
        best_idx = int(np.argmax(H))
        return float(df["alpha"].iloc[best_idx])

    # Signed perpendicular distance from each point to the chord.
    # Positive = above the chord (more entropy per unit variance).
    # cross = (P1 - P0) × (P - P0) / |P1 - P0|
    signed_dist = (dx * (H_n - H_n[0]) - dy * (V_n - V_n[0])) / chord_len

    best_idx = int(np.argmax(signed_dist))
    alpha_sel = float(df["alpha"].iloc[best_idx])

    logger.info(
        "select_operating_alpha (Kneedle): α*=%.4g "
        "(dist=%.4f, H=%.4f, Var=%.4e).",
        alpha_sel, float(signed_dist[best_idx]),
        float(H[best_idx]), float(V[best_idx]),
    )

    # Warn if frontier is non-monotonic (SCA found different local optima)
    H_diff = np.diff(H)
    if np.any(H_diff < -1e-6):
        drops = np.where(H_diff < -1e-6)[0]
        for d in drops:
            logger.warning(
                "  Non-monotonic frontier: H dropped from %.4f (α=%.4g) "
                "to %.4f (α=%.4g) — SCA may have found a different local "
                "optimum at higher α.",
                float(H[d]), float(df["alpha"].iloc[d]),
                float(H[d + 1]), float(df["alpha"].iloc[d + 1]),
            )

    return alpha_sel
