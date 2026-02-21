"""
Conditioning guard for cross-sectional regression.

Uses SVD-based least squares (np.linalg.lstsq) which handles
rank-deficiency and ill-conditioning automatically.

Supports both OLS and WLS (weighted least squares) for
heteroscedasticity correction in factor regression.

Reference: ISD Section MOD-007 — Sub-task 2 (conditioning guard).
Literature: Fama & MacBeth (1973), Barra USE4 (Menchero et al. 2011).
"""

import logging

import numpy as np

from src.validation import assert_finite_2d

logger = logging.getLogger(__name__)


def safe_solve(
    B_t: np.ndarray,
    r_t: np.ndarray,
    conditioning_threshold: float = 1e6,
    ridge_scale: float = 1e-6,
) -> np.ndarray:
    """
    Solve cross-sectional OLS via SVD-based least squares.

    Uses np.linalg.lstsq() which handles rank-deficiency and
    ill-conditioning via SVD truncation — more numerically stable
    than explicit (B^T B)^{-1} B^T r with manual ridge.

    Logs a warning when the effective rank is below the number of
    columns (rank-deficient system), providing traceability for
    poorly-conditioned cross-sections.

    The conditioning_threshold and ridge_scale parameters are retained
    for API compatibility but no longer used internally.

    :param B_t (np.ndarray): Exposures at date t (n_active, AU)
    :param r_t (np.ndarray): Returns at date t (n_active,)
    :param conditioning_threshold (float): Unused (kept for API compat)
    :param ridge_scale (float): Unused (kept for API compat)

    :return z_hat (np.ndarray): Factor returns (AU,)
    """
    z_hat, _, rank, sv = np.linalg.lstsq(B_t, r_t, rcond=None)
    p = B_t.shape[1]
    if rank < p:
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        logger.warning(
            "safe_solve: rank deficient — rank=%d < p=%d, cond=%.1e",
            rank, p, cond,
        )
    assert np.isfinite(z_hat).all(), "safe_solve: z_hat contains NaN/Inf"
    return z_hat


def safe_solve_wls(
    B_t: np.ndarray,
    r_t: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Solve cross-sectional WLS via SVD-based least squares.

    Transforms the problem to OLS: lstsq(W^{1/2} B, W^{1/2} r)
    where W = diag(weights).

    This corrects heteroscedasticity: stocks with higher idiosyncratic
    variance get lower weight, yielding BLUE estimates of factor returns
    (Fama-MacBeth 1973, Barra USE4).

    Logs a warning on rank-deficient systems.

    :param B_t (np.ndarray): Exposures at date t (n_active, AU)
    :param r_t (np.ndarray): Returns at date t (n_active,)
    :param weights (np.ndarray): Regression weights (n_active,), typically
        1/sigma_eps^2 or 1/sigma_eps (inverse idio variance or vol).

    :return z_hat (np.ndarray): Factor returns (AU,)
    """
    sqrt_w = np.sqrt(np.maximum(weights, 1e-20))
    B_w = B_t * sqrt_w[:, np.newaxis]
    r_w = r_t * sqrt_w
    z_hat, _, rank, sv = np.linalg.lstsq(B_w, r_w, rcond=None)
    p = B_w.shape[1]
    if rank < p:
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        logger.warning(
            "safe_solve_wls: rank deficient — rank=%d < p=%d, cond=%.1e",
            rank, p, cond,
        )
    assert np.isfinite(z_hat).all(), "safe_solve_wls: z_hat contains NaN/Inf"
    return z_hat
