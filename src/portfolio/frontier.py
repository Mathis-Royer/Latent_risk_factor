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
    idio_weight: float = 0.2,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
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
    :return weights_by_alpha (dict[float, np.ndarray]): Optimal weights for each α
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    # Sort ascending so warm-start flows from low-alpha to high-alpha
    sorted_grid = sorted(alpha_grid)

    results: list[dict[str, float]] = []
    weights_by_alpha: dict[float, np.ndarray] = {}
    n_alphas = len(sorted_grid)
    prev_w: np.ndarray | None = None

    for idx, alpha in enumerate(sorted_grid):
        logger.info("    Frontier alpha %d/%d (alpha=%.3f)...", idx + 1, n_alphas, alpha)

        # Warm-start: use previous alpha's solution as an additional starting
        # point. Solutions at adjacent alpha values are typically close, so
        # this reduces the number of SCA iterations needed.
        warm_start_w = prev_w

        w_opt, f_opt, H_opt = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=alpha,
            n_starts=n_starts,
            seed=seed + idx,
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
            warm_start_w=warm_start_w,
            idio_weight=idio_weight,
        )

        prev_w = w_opt
        weights_by_alpha[alpha] = w_opt.copy()

        variance = float(w_opt @ Sigma_assets @ w_opt)
        entropy = compute_entropy_only(w_opt, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                                       idio_weight=idio_weight)
        # Factor-only entropy for alpha selection (Meucci 2009):
        # ENB target must apply to H_factor alone, not the combined
        # two-layer entropy.  The idiosyncratic layer (n >> AU terms)
        # trivially inflates ENB, making the target too easy to reach
        # and collapsing the portfolio to near min-variance.
        entropy_factor = compute_entropy_only(
            w_opt, B_prime, eigenvalues, entropy_eps,
            D_eps=None, idio_weight=0.0,
        )
        n_active = int(np.sum(w_opt > w_min))

        results.append({
            "alpha": alpha,
            "variance": variance,
            "entropy": entropy,
            "entropy_factor": entropy_factor,
            "n_active": float(n_active),
        })

    return pd.DataFrame(results), weights_by_alpha


def select_operating_alpha(
    frontier: pd.DataFrame,
    target_enb: float = 0.0,
) -> float:
    """
    Select operating α from the variance-entropy frontier.

    Two modes:
    1. target_enb > 0: Select smallest α where ENB_factor = exp(H_factor)
       >= target_enb.  Uses FACTOR-ONLY entropy (not combined two-layer)
       to prevent the idiosyncratic layer from trivially inflating ENB.
       Recommended: n_signal / 2 (e.g. 2.5 for 5 signal factors).
       Reference: Meucci (2009), "Managing Diversification".
    2. target_enb == 0: Kneedle elbow detection (legacy fallback).
       Reference: Satopaa et al. (2011), "Finding a 'Kneedle' in a Haystack".

    Fallback: if the frontier is degenerate (flat or single-point), returns
    the α that achieves maximum entropy.

    :param frontier (pd.DataFrame): Frontier with columns alpha, variance,
        entropy, and optionally entropy_factor (factor-only entropy).
    :param target_enb (float): Target effective number of bets
        (ENB_factor = exp(H_factor)).  0.0 = use Kneedle (legacy).

    :return alpha_opt (float): Selected operating α
    """
    if len(frontier) < 2:
        return float(frontier["alpha"].iloc[0]) if len(frontier) > 0 else 0.1

    df = frontier.sort_values("alpha").reset_index(drop=True)

    H = np.asarray(df["entropy"], dtype=np.float64)
    V = np.asarray(df["variance"], dtype=np.float64)

    # For ENB target selection, use factor-only entropy (not combined).
    # The idiosyncratic layer (n >> AU terms) trivially inflates ENB,
    # making the target too easy to reach.  With H_combined, the
    # portfolio collapses to near min-variance because even α=0.001
    # satisfies the ENB threshold.  Using H_factor forces the optimizer
    # to actually diversify across latent factors.
    # Reference: Meucci (2009), Roncalli (2013 Ch. 7).
    if "entropy_factor" in df.columns:
        H_factor = np.asarray(df["entropy_factor"], dtype=np.float64)
    else:
        H_factor = H  # backward compatibility

    # Target ENB mode: among all alphas achieving ENB >= target, select
    # the one with MINIMUM VARIANCE.  The old logic picked the smallest
    # qualifying alpha, ignoring that higher-alpha points may achieve
    # lower variance (U-shaped frontier from diversification benefits).
    # Once the ENB target is met, prefer the point with lowest risk.
    # Reference: Meucci (2009), DeMiguel et al. (2009).
    # Apply cumulative max envelope to handle non-monotonic frontiers
    # (SCA may find different local optima at adjacent alpha values).
    if target_enb > 0.0:
        enb = np.exp(H_factor)
        enb_mono = np.maximum.accumulate(enb)
        qualifying_mask = enb_mono >= target_enb
        if qualifying_mask.any():
            qualifying_indices = np.where(qualifying_mask)[0]
            best_idx = qualifying_indices[
                int(np.argmin(V[qualifying_indices]))
            ]
            alpha_sel = float(df["alpha"].iloc[best_idx])
            logger.info(
                "select_operating_alpha (target ENB_factor=%.2f): α*=%.4g "
                "(ENB_factor=%.2f, ENB_mono=%.2f, H_factor=%.4f, "
                "H_combined=%.4f, Var=%.4e, qualifying=%d/%d).",
                target_enb, alpha_sel, float(enb[best_idx]),
                float(enb_mono[best_idx]), float(H_factor[best_idx]),
                float(H[best_idx]), float(V[best_idx]),
                len(qualifying_indices), len(df),
            )
            return alpha_sel

        # No point reaches target: use highest ENB (monotone envelope)
        # with warning.  Using enb_mono instead of raw enb ensures we pick
        # the highest-alpha point even if the frontier is non-monotonic
        # (SCA local optima can produce ENB drops at intermediate alpha).
        best_idx = int(np.argmax(enb_mono))
        alpha_sel = float(df["alpha"].iloc[best_idx])
        logger.warning(
            "select_operating_alpha: no frontier point reaches target "
            "ENB_factor=%.2f (max ENB_factor=%.2f at α=%.4g). "
            "Using max-ENB point.",
            target_enb, float(enb[best_idx]), alpha_sel,
        )
        return alpha_sel

    # Kneedle fallback
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


def compute_adaptive_enb_target(
    eigenvalues: np.ndarray,
    n_signal: int,
) -> float:
    """
    Compute adaptive ENB target from eigenvalue spectrum.

    Uses min(n_signal/2, ENB_spectrum * 0.7) where ENB_spectrum is the
    theoretical maximum ENB if risk contributions were proportional to
    eigenvalues.  With concentrated spectra (lambda_1/lambda_2 >> 1),
    n_signal/2 may be unreachable, so we cap at 70% of ENB_spectrum.

    Floor: max(result, 2.0) ensures at least 2 effective bets.

    Reference: Meucci (2009), "Managing Diversification".
               Roncalli (2013), Ch. 7.

    :param eigenvalues (np.ndarray): Signal eigenvalues (n_signal,)
    :param n_signal (int): Number of signal factors

    :return target_enb (float): Adaptive ENB target (>= 2.0)
    """
    eig_sum = float(eigenvalues.sum())
    if eig_sum > 1e-30:
        eig_norm = eigenvalues / eig_sum
        eig_norm = np.maximum(eig_norm, 1e-30)
        enb_spectrum = float(np.exp(
            -np.sum(eig_norm * np.log(eig_norm))
        ))
    else:
        enb_spectrum = 1.0

    heuristic_enb = max(2.0, n_signal / 2.0)
    target_enb = min(heuristic_enb, enb_spectrum * 0.7)
    target_enb = max(target_enb, 2.0)  # floor

    logger.info(
        "compute_adaptive_enb_target: target=%.2f "
        "(heuristic=%.1f, ENB_spectrum=%.2f, cap=%.2f, n_signal=%d)",
        target_enb, heuristic_enb, enb_spectrum, enb_spectrum * 0.7,
        n_signal,
    )

    return target_enb
