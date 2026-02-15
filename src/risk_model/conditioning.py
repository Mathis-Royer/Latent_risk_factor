"""
Conditioning guard for cross-sectional regression.

If κ(B^T B) > threshold, applies minimal ridge regularization:
  λ_ridge = ridge_scale · tr(B^T B) / AU

Reference: ISD Section MOD-007 — Sub-task 2 (conditioning guard).
"""

import numpy as np


def check_conditioning(
    BtB: np.ndarray,
    threshold: float = 1e6,
) -> bool:
    """
    Check if matrix B^T B is ill-conditioned.

    :param BtB (np.ndarray): B^T B matrix (AU, AU)
    :param threshold (float): Condition number threshold

    :return ill_conditioned (bool): True if κ(B^T B) > threshold
    """
    eigenvalues = np.linalg.eigvalsh(BtB)
    if eigenvalues.min() <= 0:
        return True
    kappa = eigenvalues.max() / eigenvalues.min()
    return bool(kappa > threshold)


def apply_ridge(
    BtB: np.ndarray,
    ridge_scale: float = 1e-6,
) -> np.ndarray:
    """
    Apply minimal ridge regularization to B^T B.

    λ_ridge = ridge_scale · tr(B^T B) / AU
    Result = B^T B + λ_ridge · I

    :param BtB (np.ndarray): B^T B matrix (AU, AU)
    :param ridge_scale (float): Scale factor for ridge

    :return BtB_reg (np.ndarray): Regularized matrix (AU, AU)
    """
    AU = BtB.shape[0]
    trace = np.trace(BtB)
    lambda_ridge = ridge_scale * trace / max(1, AU)
    return BtB + lambda_ridge * np.eye(AU)


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

    The conditioning_threshold and ridge_scale parameters are retained
    for API compatibility but no longer used internally.

    :param B_t (np.ndarray): Exposures at date t (n_active, AU)
    :param r_t (np.ndarray): Returns at date t (n_active,)
    :param conditioning_threshold (float): Unused (kept for API compat)
    :param ridge_scale (float): Unused (kept for API compat)

    :return z_hat (np.ndarray): Factor returns (AU,)
    """
    z_hat, _, _, _ = np.linalg.lstsq(B_t, r_t, rcond=None)
    return z_hat
