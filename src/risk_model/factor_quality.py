"""
Factor quality metrics for diagnostic purposes.

These metrics characterize latent factor properties (persistence, breadth, stability)
but are NOT used for portfolio optimization. The optimization remains agnostic to
factor quality — all AU factors receive equal risk contribution.

Reference: Lettau & Pelger (2020), Onatski (2010), Bai & Ng (2002).
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_persistence(factor_returns: np.ndarray) -> np.ndarray:
    """
    Compute half-life of factor return autocorrelation.

    Half-life = ln(2) / ln(|rho_1|), where rho_1 is the lag-1 autocorrelation.
    Higher values indicate more persistent (structural) factors.

    :param factor_returns (np.ndarray): Factor returns (T, k) or (T,) for single factor

    :return half_life (np.ndarray): Half-life in days per factor (k,).
        Returns inf for factors with |rho_1| < 0.01 (noise-like).
    """
    if factor_returns.ndim == 1:
        factor_returns = factor_returns.reshape(-1, 1)

    T, k = factor_returns.shape
    half_lives = np.full(k, np.inf)

    if T < 5:
        return half_lives

    for j in range(k):
        z = factor_returns[:, j]
        z_demean = z - np.mean(z)

        # Lag-1 autocorrelation
        if np.var(z_demean) < 1e-12:
            continue
        rho_1 = np.corrcoef(z_demean[:-1], z_demean[1:])[0, 1]

        if np.isnan(rho_1) or np.abs(rho_1) < 0.01:
            # Noise-like factor: no meaningful persistence
            half_lives[j] = np.inf
        elif np.abs(rho_1) >= 1.0:
            # Perfect autocorrelation: undefined half-life
            half_lives[j] = np.inf
        elif rho_1 < 0:
            # Mean-reverting factor: no positive persistence
            half_lives[j] = np.inf
        else:
            # half_life = -ln(2) / ln(rho_1)
            # For 0 < rho < 1: ln(rho) < 0, so -ln(2)/ln(rho) > 0
            half_lives[j] = -np.log(2) / np.log(rho_1)

    return half_lives


def compute_breadth(
    B_A: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Compute factor breadth: number of stocks with significant loading.

    Breadth = count(|B_ik| > threshold) per factor.
    High breadth indicates a systematic factor; low breadth indicates idiosyncratic.

    :param B_A (np.ndarray): Exposure matrix (n, k)
    :param threshold (float): Absolute loading threshold for significance

    :return breadth (np.ndarray): Number of stocks per factor (k,)
    """
    if B_A.ndim != 2:
        return np.array([])

    # Normalize B_A columns to have unit norm for scale-invariant threshold
    col_norms = np.linalg.norm(B_A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    B_norm = B_A / col_norms

    breadth = np.sum(np.abs(B_norm) > threshold, axis=0)
    return breadth.astype(np.int64)


def compute_eigenvalue_gap(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalue gaps: lambda_k - lambda_{k+1}.

    Large gaps indicate clean separation between signal and noise eigenvalues.
    The largest gap often identifies the boundary between factor and residual.

    :param eigenvalues (np.ndarray): Eigenvalues sorted in descending order (k,)

    :return gaps (np.ndarray): Gaps between consecutive eigenvalues (k-1,)
    """
    if len(eigenvalues) < 2:
        return np.array([])

    gaps = eigenvalues[:-1] - eigenvalues[1:]
    return gaps


def compute_gap_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalue gap ratios: (lambda_k - lambda_{k+1}) / lambda_k.

    Relative measure of spectral separation, more robust to scale.

    :param eigenvalues (np.ndarray): Eigenvalues sorted in descending order (k,)

    :return gap_ratios (np.ndarray): Gap ratios (k-1,)
    """
    if len(eigenvalues) < 2:
        return np.array([])

    # Avoid division by zero
    eig_safe = np.maximum(eigenvalues[:-1], 1e-12)
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    return gaps / eig_safe


def bai_ng_ic2(
    returns_centered: np.ndarray,
    k_max: int = 30,
) -> int:
    """
    Bai & Ng (2002) Information Criterion IC2 for factor number selection.

    IC2(k) = ln(V(k)) + k * ((n+T)/(n*T)) * ln(min(n,T))

    where V(k) = (1/(n*T)) * ||R - F_k Lambda_k^T||^2_F

    :param returns_centered (np.ndarray): Centered returns matrix (T, n)
    :param k_max (int): Maximum number of factors to test

    :return k_star (int): Optimal number of factors minimizing IC2
    """
    T, n = returns_centered.shape
    k_max_eff = min(k_max, min(T, n) - 1)

    if k_max_eff < 1:
        return 1

    # SVD once
    U, S, Vt = np.linalg.svd(returns_centered, full_matrices=False)

    penalty_coeff = ((n + T) / (n * T)) * np.log(min(n, T))

    best_ic = np.inf
    best_k = 1

    for k in range(1, k_max_eff + 1):
        # Reconstruction error using k factors
        R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        V_k = np.sum((returns_centered - R_approx) ** 2) / (n * T)

        ic2 = np.log(max(V_k, 1e-30)) + k * penalty_coeff

        if ic2 < best_ic:
            best_ic = ic2
            best_k = k

    return best_k


def onatski_eigenvalue_ratio(
    eigenvalues: np.ndarray,
    n: int,
    T: int,
    _significance: float = 0.05,
) -> tuple[int, float]:
    """
    Onatski (2010) eigenvalue ratio test for factor number estimation.

    Tests H0: k factors vs H1: k+1 factors using the ratio of consecutive
    eigenvalue differences, compared to Tracy-Widom critical value.

    The test statistic ER(k) = (lambda_k - lambda_{k+1}) / (lambda_{k+1} - lambda_{k+2})
    Under H0 (k factors), ER(k) should be small; large ER suggests k+1.

    :param eigenvalues (np.ndarray): Eigenvalues sorted in descending order
    :param n (int): Number of assets (cross-section dimension)
    :param T (int): Number of time periods
    :param _significance (float): Significance level for test (reserved for future use)

    :return (k_estimated, max_ratio): Estimated number of factors and test statistic
    """
    if len(eigenvalues) < 3:
        return 1, 0.0

    # Compute all ratio statistics
    # ER(k) = (lambda_k - lambda_{k+1}) / (lambda_{k+1} - lambda_{k+2})
    gaps = eigenvalues[:-1] - eigenvalues[1:]

    # Avoid division by zero
    gaps_denom = np.maximum(gaps[1:], 1e-12)
    ratios = gaps[:-1] / gaps_denom

    # Tracy-Widom inspired critical value (simplified approximation)
    # For large n, T: critical value ~ 2 for 5% significance
    gamma = n / T
    critical = 1.0 + np.sqrt(gamma)  # Marchenko-Pastur edge approximation

    # Count factors where ER exceeds critical value
    k_estimated = 1
    for k in range(len(ratios)):
        if ratios[k] > critical:
            k_estimated = k + 1
        else:
            # First ratio below critical ends the spiked region
            break

    max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 0.0

    return k_estimated, max_ratio


def classify_factor(
    half_life: float,
    breadth: int,
    stability: float | None = None,
    n_stocks: int = 100,
) -> str:
    """
    Classify a factor as Structural, Style, or Episodic.

    Categories (Lettau & Pelger 2020 inspired):
    - Structural: high persistence (>100 days), high breadth (>30% of stocks)
    - Style: moderate persistence (25-100 days), any breadth
    - Episodic: low persistence (<25 days) or very low breadth (<10%)

    :param half_life (float): Factor autocorrelation half-life in days
    :param breadth (int): Number of stocks with significant loading
    :param stability (float | None): Optional Spearman rho with previous period
    :param n_stocks (int): Total number of stocks in universe

    :return category (str): "Structural", "Style", or "Episodic"
    """
    breadth_pct = breadth / max(n_stocks, 1)

    # Override with stability if very high
    if stability is not None and stability > 0.95:
        if half_life >= 100 and breadth_pct >= 0.3:
            return "Structural"

    # Primary classification by persistence
    if half_life >= 100 and breadth_pct >= 0.30:
        return "Structural"
    elif half_life >= 25 or (breadth_pct >= 0.15 and np.isfinite(half_life)):
        return "Style"
    else:
        return "Episodic"


def compute_factor_quality_dashboard(
    B_A: np.ndarray,
    eigenvalues: np.ndarray,
    factor_returns: np.ndarray | None = None,
    returns_centered: np.ndarray | None = None,
    stability_rho: float | None = None,
) -> dict[str, Any]:
    """
    Compute complete factor quality dashboard for diagnostics.

    :param B_A (np.ndarray): Exposure matrix (n, AU)
    :param eigenvalues (np.ndarray): Signal eigenvalues (AU,)
    :param factor_returns (np.ndarray | None): Factor returns (T, AU) if available
    :param returns_centered (np.ndarray | None): Centered asset returns for Bai-Ng
    :param stability_rho (float | None): Latent stability correlation from previous fold

    :return dashboard (dict): Factor quality metrics and comparisons
    """
    n_stocks = B_A.shape[0] if B_A.ndim == 2 else 0
    AU = B_A.shape[1] if B_A.ndim == 2 else 0

    # Breadth per factor
    breadth = compute_breadth(B_A, threshold=0.3)

    # Eigenvalue gaps
    gaps = compute_eigenvalue_gap(eigenvalues)
    gap_ratios = compute_gap_ratio(eigenvalues)

    # Persistence (if factor returns available)
    half_lives = np.full(AU, np.inf)
    if factor_returns is not None and factor_returns.shape[0] > 10:
        half_lives = compute_persistence(factor_returns)

    # Factor classification
    categories: list[str] = []
    for k in range(AU):
        hl = half_lives[k] if k < len(half_lives) else np.inf
        br = int(breadth[k]) if k < len(breadth) else 0
        cat = classify_factor(hl, br, stability_rho, n_stocks)
        categories.append(cat)

    # Category summary
    n_structural = sum(1 for c in categories if c == "Structural")
    n_style = sum(1 for c in categories if c == "Style")
    n_episodic = sum(1 for c in categories if c == "Episodic")

    # Bai-Ng IC2 for comparison (if returns provided)
    k_bai_ng: int | None = None
    if returns_centered is not None:
        k_bai_ng = bai_ng_ic2(returns_centered, k_max=min(50, AU + 10))

    # Onatski test (if enough eigenvalues)
    k_onatski: int | None = None
    onatski_ratio: float | None = None
    if len(eigenvalues) >= 5 and n_stocks > 10:
        # Estimate T from context (rough heuristic)
        T_est = max(252, n_stocks)
        k_onatski, onatski_ratio = onatski_eigenvalue_ratio(
            eigenvalues, n_stocks, T_est
        )

    return {
        "AU": AU,
        "n_stocks": n_stocks,
        # Per-factor metrics
        "breadth": breadth.tolist() if isinstance(breadth, np.ndarray) else [],
        "half_lives": half_lives.tolist() if isinstance(half_lives, np.ndarray) else [],
        "categories": categories,
        # Summary
        "n_structural": n_structural,
        "n_style": n_style,
        "n_episodic": n_episodic,
        "pct_structural": n_structural / max(AU, 1),
        "pct_style": n_style / max(AU, 1),
        "pct_episodic": n_episodic / max(AU, 1),
        # Eigenvalue analysis
        "eigenvalue_gaps": gaps.tolist() if isinstance(gaps, np.ndarray) else [],
        "gap_ratios": gap_ratios.tolist() if isinstance(gap_ratios, np.ndarray) else [],
        "max_gap_index": int(np.argmax(gaps)) if len(gaps) > 0 else 0,
        # AU validation
        "k_bai_ng": k_bai_ng,
        "k_onatski": k_onatski,
        "onatski_ratio": onatski_ratio,
        "au_bai_ng_diff": AU - k_bai_ng if k_bai_ng is not None else None,
        "au_onatski_diff": AU - k_onatski if k_onatski is not None else None,
        # Stability
        "stability_rho": stability_rho,
        "stability_ok": stability_rho is None or stability_rho > 0.85,
    }
