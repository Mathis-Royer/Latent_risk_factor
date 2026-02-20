"""
Factor model diagnostics: eigenvalue analysis, shrinkage quality, regression stability.

Implements diagnostics from Ledoit-Wolf (JFEc 2020) and FactSet Risk Model:
- Eigenvalue concentration and gap structure
- Shrinkage distance (Frobenius norm before/after)
- Cross-sectional regression quality per date

Reference: ISD diagnostic gaps B.10-B.18.
"""

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# B.10-B.11: Eigenvalue concentration and gap structure
# ---------------------------------------------------------------------------

def compute_eigenvalue_concentration(
    eigenvalues: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze eigenvalue spectrum for concentration and signal/noise separation.

    Metrics:
    - λ₁/λ_K ratio: Measures dominance of first eigenvalue
    - Cumulative explained variance
    - Gap structure: Identifies signal/noise boundary via gap ratios
    - Effective dimensionality: exp(entropy) of eigenvalue distribution

    :param eigenvalues (np.ndarray): Eigenvalues sorted descending (K,)

    :return analysis (dict): Concentration metrics and gap analysis
    """
    if eigenvalues.size == 0:
        return {"available": False, "reason": "empty eigenvalues"}

    # Ensure sorted descending and positive
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    K = len(eigenvalues)
    total_var = float(np.sum(eigenvalues))

    if total_var <= 0:
        return {"available": False, "reason": "zero total variance"}

    # Concentration ratios
    lambda_1 = float(eigenvalues[0])
    lambda_K = float(eigenvalues[-1]) if eigenvalues[-1] > 0 else 1e-10
    concentration_ratio = lambda_1 / lambda_K

    # Cumulative explained variance
    cumulative = np.cumsum(eigenvalues) / total_var
    var_explained_top1 = float(cumulative[0])
    var_explained_top3 = float(cumulative[min(2, K - 1)])
    var_explained_top10 = float(cumulative[min(9, K - 1)])

    # Number of eigenvalues to explain 90%/95%/99%
    n_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    n_for_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    n_for_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Gap ratios: λ_k / λ_{k+1}
    gap_ratios = []
    for k in range(K - 1):
        denom = max(eigenvalues[k + 1], 1e-15)
        gap_ratios.append(float(eigenvalues[k] / denom))

    # Find max gap (potential signal/noise boundary)
    max_gap_ratio = max(gap_ratios) if gap_ratios else 0.0
    max_gap_index = int(np.argmax(gap_ratios)) if gap_ratios else 0

    # Effective dimensionality (exponential of entropy)
    eig_probs = eigenvalues / total_var
    eig_probs_pos = eig_probs[eig_probs > 1e-20]
    entropy = float(-np.sum(eig_probs_pos * np.log(eig_probs_pos)))
    eff_dim = float(np.exp(entropy))

    # Spectral decay rate (linear regression on log-eigenvalues)
    valid_eigs = eigenvalues[eigenvalues > 1e-15]
    if len(valid_eigs) >= 3:
        log_eigs = np.log(valid_eigs)
        x = np.arange(len(valid_eigs))
        slope, _ = np.polyfit(x, log_eigs, 1)
        decay_rate = float(-slope)  # Positive = decaying
    else:
        decay_rate = 0.0

    return {
        "available": True,
        "K": K,
        "total_variance": total_var,
        # Concentration
        "lambda_1": lambda_1,
        "lambda_K": lambda_K,
        "concentration_ratio": concentration_ratio,
        # Explained variance
        "var_explained_top1": var_explained_top1,
        "var_explained_top3": var_explained_top3,
        "var_explained_top10": var_explained_top10,
        "n_for_90pct": n_for_90,
        "n_for_95pct": n_for_95,
        "n_for_99pct": n_for_99,
        # Gap analysis
        "gap_ratios": gap_ratios[:10],  # First 10 for brevity
        "max_gap_ratio": max_gap_ratio,
        "max_gap_index": max_gap_index,
        "signal_noise_boundary": max_gap_index + 1,
        # Effective dimensionality
        "entropy": entropy,
        "eff_dim": eff_dim,
        "decay_rate": decay_rate,
    }


# ---------------------------------------------------------------------------
# B.12: Shrinkage distance (Frobenius norm)
# ---------------------------------------------------------------------------

def compute_shrinkage_distance(
    Sigma_raw: np.ndarray,
    Sigma_shrunk: np.ndarray,
) -> dict[str, Any]:
    """
    Compute Frobenius distance between raw and shrunk covariance matrices.

    Measures how much the shrinkage procedure modified the sample covariance.
    Large distance indicates high regularization; small distance indicates
    minimal shrinkage (sample covariance was already well-conditioned).

    :param Sigma_raw (np.ndarray): Raw sample covariance (K, K)
    :param Sigma_shrunk (np.ndarray): Shrunk covariance (K, K)

    :return analysis (dict): Distance metrics and eigenvalue changes
    """
    if Sigma_raw.size == 0 or Sigma_shrunk.size == 0:
        return {"available": False, "reason": "empty covariance matrix"}

    if Sigma_raw.shape != Sigma_shrunk.shape:
        return {"available": False, "reason": "shape mismatch"}

    K = Sigma_raw.shape[0]

    # Frobenius norms
    frob_raw = float(np.linalg.norm(Sigma_raw, "fro"))
    frob_shrunk = float(np.linalg.norm(Sigma_shrunk, "fro"))
    frob_diff = float(np.linalg.norm(Sigma_raw - Sigma_shrunk, "fro"))

    # Relative distance
    relative_dist = frob_diff / max(frob_raw, 1e-10)

    # Eigenvalue analysis
    eigs_raw = np.linalg.eigvalsh(Sigma_raw)[::-1]
    eigs_shrunk = np.linalg.eigvalsh(Sigma_shrunk)[::-1]
    eigs_raw = np.maximum(eigs_raw, 0.0)
    eigs_shrunk = np.maximum(eigs_shrunk, 0.0)

    # Eigenvalue reduction ratio per dimension
    eig_reduction = []
    for i in range(K):
        if eigs_raw[i] > 1e-15:
            eig_reduction.append(float(eigs_shrunk[i] / eigs_raw[i]))
        else:
            eig_reduction.append(1.0)

    # Average reduction (< 1 means eigenvalues shrunk on average)
    avg_eig_reduction = float(np.mean(eig_reduction))

    # Condition number change
    cond_raw = float(eigs_raw[0] / max(eigs_raw[-1], 1e-15))
    cond_shrunk = float(eigs_shrunk[0] / max(eigs_shrunk[-1], 1e-15))
    cond_improvement = cond_raw / max(cond_shrunk, 1e-10)

    # Trace preservation check
    trace_raw = float(np.trace(Sigma_raw))
    trace_shrunk = float(np.trace(Sigma_shrunk))
    trace_ratio = trace_shrunk / max(trace_raw, 1e-10)

    return {
        "available": True,
        "K": K,
        # Frobenius metrics
        "frobenius_raw": frob_raw,
        "frobenius_shrunk": frob_shrunk,
        "frobenius_diff": frob_diff,
        "relative_distance": relative_dist,
        # Eigenvalue changes
        "eig_reduction_per_dim": eig_reduction[:10],  # First 10
        "avg_eig_reduction": avg_eig_reduction,
        "max_eig_reduction": float(np.min(eig_reduction)),  # Most reduced
        "min_eig_reduction": float(np.max(eig_reduction)),  # Least reduced
        # Condition number
        "cond_raw": cond_raw,
        "cond_shrunk": cond_shrunk,
        "cond_improvement": cond_improvement,
        # Trace
        "trace_raw": trace_raw,
        "trace_shrunk": trace_shrunk,
        "trace_ratio": trace_ratio,
    }


# ---------------------------------------------------------------------------
# B.12b: DGJ shrinkage validation (Baik-Ben Arous-Péché threshold)
# ---------------------------------------------------------------------------

def validate_dgj_recovery(
    eigs_sample: np.ndarray,
    eigs_shrunk: np.ndarray,
    gamma: float,
) -> dict[str, Any]:
    """
    Validate DGJ (Donoho-Gavish-Johnstone) nonlinear shrinkage quality.

    Uses random matrix theory to check if shrinkage correctly separates
    signal eigenvalues from noise. The Baik-Ben Arous-Péché (BBP) threshold
    marks the phase transition above which eigenvalues carry signal.

    For noise eigenvalues (below BBP), DGJ should shrink them to a flat bulk.
    A low coefficient of variation (CV < 0.1) indicates proper noise flattening.

    :param eigs_sample (np.ndarray): Sample eigenvalues sorted descending
    :param eigs_shrunk (np.ndarray): Shrunk eigenvalues sorted descending
    :param gamma (float): Aspect ratio n_samples / n_features

    :return validation (dict): Signal detection count, noise CV, BBP threshold
    """
    if eigs_sample.size == 0 or eigs_shrunk.size == 0:
        return {
            "available": False,
            "reason": "empty eigenvalues",
            "n_signal": 0,
            "noise_cv": 0.0,
            "bbp_threshold": 0.0,
        }

    if eigs_sample.shape != eigs_shrunk.shape:
        return {
            "available": False,
            "reason": "eigenvalue array shape mismatch",
            "n_signal": 0,
            "noise_cv": 0.0,
            "bbp_threshold": 0.0,
        }

    if gamma <= 0:
        return {
            "available": False,
            "reason": "invalid gamma (must be > 0)",
            "n_signal": 0,
            "noise_cv": 0.0,
            "bbp_threshold": 0.0,
        }

    # Ensure sorted descending and non-negative
    eigs_sample = np.sort(np.maximum(eigs_sample, 0.0))[::-1]
    eigs_shrunk = np.sort(np.maximum(eigs_shrunk, 0.0))[::-1]

    # Noise variance estimate (median of sample eigenvalues)
    sigma_sq = float(np.median(eigs_sample))

    # Baik-Ben Arous-Péché threshold: σ² * (1 + √γ)²
    # Eigenvalues above this carry signal information
    bbp_threshold = sigma_sq * (1.0 + np.sqrt(gamma)) ** 2

    # Count signal eigenvalues (above BBP threshold)
    n_signal_detected = int(np.sum(eigs_sample > bbp_threshold))

    # Extract noise eigenvalues from shrunk spectrum
    noise_mask = eigs_sample <= bbp_threshold
    noise_eigs = eigs_shrunk[noise_mask]

    # Coefficient of variation for noise bulk
    # CV < 0.1 indicates flat noise (DGJ working correctly)
    if len(noise_eigs) > 0 and np.mean(noise_eigs) > 1e-15:
        noise_cv = float(np.std(noise_eigs) / np.mean(noise_eigs))
    else:
        noise_cv = 0.0

    return {
        "available": True,
        "n_signal": n_signal_detected,
        "noise_cv": noise_cv,
        "bbp_threshold": float(bbp_threshold),
        "sigma_sq": sigma_sq,
        "gamma": gamma,
        "n_noise": int(np.sum(noise_mask)),
        "noise_mean": float(np.mean(noise_eigs)) if len(noise_eigs) > 0 else 0.0,
        "noise_std": float(np.std(noise_eigs)) if len(noise_eigs) > 0 else 0.0,
        "dgj_quality": "good" if noise_cv < 0.1 else ("fair" if noise_cv < 0.2 else "poor"),
    }


# ---------------------------------------------------------------------------
# B.13-B.16: Cross-sectional regression quality
# ---------------------------------------------------------------------------

def track_regression_quality(
    B_A_by_date: dict[str, np.ndarray],
    z_hat: np.ndarray,
    returns: np.ndarray,
    valid_dates: list[str],
) -> dict[str, Any]:
    """
    Track cross-sectional regression quality per date.

    Computes per-date:
    - Condition number of B_A^T B_A
    - Effective rank
    - Cross-sectional R²
    - z_hat extreme values

    :param B_A_by_date (dict): date_str -> B_A_t (n_active_t, AU)
    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param returns (np.ndarray): Stock returns (n_dates, n_stocks) or list of arrays
    :param valid_dates (list[str]): Dates corresponding to z_hat rows

    :return analysis (dict): Per-date and aggregate regression quality
    """
    if not B_A_by_date or z_hat.size == 0:
        return {"available": False, "reason": "missing data"}

    n_dates = len(valid_dates)
    AU = z_hat.shape[1]

    condition_numbers: list[float] = []
    effective_ranks: list[float] = []
    cs_r2_values: list[float] = []
    rank_deficient_dates: list[str] = []
    z_hat_extremes: list[float] = []

    for t, date_str in enumerate(valid_dates):
        if date_str not in B_A_by_date:
            continue

        B_t = B_A_by_date[date_str]
        n_stocks_t = B_t.shape[0]

        if n_stocks_t < AU:
            rank_deficient_dates.append(date_str)
            continue

        # Condition number of B^T B
        try:
            BtB = B_t.T @ B_t
            eigs = np.linalg.eigvalsh(BtB)
            eigs = np.maximum(eigs, 0.0)
            if eigs[-1] > 1e-15:
                cond = float(eigs.max() / eigs.min())
            else:
                cond = float("inf")
            condition_numbers.append(cond)

            # Effective rank (normalized nuclear norm / spectral norm)
            eff_rank = float(np.sum(eigs > 1e-10 * eigs.max()))
            effective_ranks.append(eff_rank)
        except np.linalg.LinAlgError:
            condition_numbers.append(float("inf"))
            effective_ranks.append(0.0)

        # Cross-sectional R² (if returns available for this date)
        if t < len(returns):
            r_t = returns[t] if isinstance(returns, list) else returns[t, :]
            if len(r_t) >= n_stocks_t:
                r_t = r_t[:n_stocks_t]
                r_hat = B_t @ z_hat[t]
                ss_res = np.sum((r_t - r_hat) ** 2)
                ss_tot = np.sum((r_t - np.mean(r_t)) ** 2)
                if ss_tot > 1e-15:
                    r2 = 1.0 - ss_res / ss_tot
                    cs_r2_values.append(float(r2))

        # z_hat extremes for this date
        z_t_max = float(np.max(np.abs(z_hat[t])))
        z_hat_extremes.append(z_t_max)

    # Aggregate statistics
    if condition_numbers:
        cond_mean = float(np.mean(condition_numbers))
        cond_std = float(np.std(condition_numbers))
        cond_max = float(np.max(condition_numbers))
        n_ill_cond = sum(1 for c in condition_numbers if c > 1e6)
    else:
        cond_mean = cond_std = cond_max = 0.0
        n_ill_cond = 0

    if cs_r2_values:
        r2_mean = float(np.mean(cs_r2_values))
        r2_std = float(np.std(cs_r2_values))
        r2_min = float(np.min(cs_r2_values))
        r2_max = float(np.max(cs_r2_values))
    else:
        r2_mean = r2_std = r2_min = r2_max = 0.0

    if z_hat_extremes:
        z_extreme_mean = float(np.mean(z_hat_extremes))
        z_extreme_max = float(np.max(z_hat_extremes))
        n_extreme = sum(1 for z in z_hat_extremes if z > 5.0)  # > 5 std
    else:
        z_extreme_mean = z_extreme_max = 0.0
        n_extreme = 0

    return {
        "available": True,
        "n_dates": n_dates,
        "AU": AU,
        # Condition number statistics
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "cond_max": cond_max,
        "n_ill_conditioned": n_ill_cond,
        "condition_numbers": condition_numbers[-50:],  # Last 50
        # Rank analysis
        "effective_ranks": effective_ranks[-50:],
        "eff_rank_mean": float(np.mean(effective_ranks)) if effective_ranks else 0.0,
        "rank_deficient_dates": rank_deficient_dates,
        "n_rank_deficient": len(rank_deficient_dates),
        "rank_deficiency_rate": len(rank_deficient_dates) / max(n_dates, 1),
        # Cross-sectional R²
        "cs_r2_mean": r2_mean,
        "cs_r2_std": r2_std,
        "cs_r2_min": r2_min,
        "cs_r2_max": r2_max,
        "cs_r2_values": cs_r2_values[-50:],  # Last 50
        # z_hat extremes
        "z_hat_extreme_mean": z_extreme_mean,
        "z_hat_extreme_max": z_extreme_max,
        "n_extreme_z_hat": n_extreme,
    }


# ---------------------------------------------------------------------------
# B.17: Column norm distribution of B_A
# ---------------------------------------------------------------------------

def analyze_exposure_norms(
    B_A: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze column (factor) and row (stock) norm distributions of B_A.

    Helps identify:
    - Factors with disproportionately large loadings
    - Stocks with extreme exposure magnitudes
    - Overall scale consistency

    :param B_A (np.ndarray): Exposure matrix (n_stocks, AU)

    :return analysis (dict): Norm statistics and outlier detection
    """
    if B_A.size == 0:
        return {"available": False, "reason": "empty B_A"}

    n_stocks, AU = B_A.shape

    # Column (factor) norms
    col_norms = np.linalg.norm(B_A, axis=0)  # (AU,)
    col_norm_mean = float(np.mean(col_norms))
    col_norm_std = float(np.std(col_norms))
    col_norm_max = float(np.max(col_norms))
    col_norm_min = float(np.min(col_norms))

    # Outlier factors (norm > 3 std from mean)
    threshold = col_norm_mean + 3 * col_norm_std
    outlier_factors = np.where(col_norms > threshold)[0].tolist()

    # Row (stock) norms
    row_norms = np.linalg.norm(B_A, axis=1)  # (n_stocks,)
    row_norm_mean = float(np.mean(row_norms))
    row_norm_std = float(np.std(row_norms))
    row_norm_max = float(np.max(row_norms))

    # Outlier stocks
    row_threshold = row_norm_mean + 3 * row_norm_std
    n_outlier_stocks = int(np.sum(row_norms > row_threshold))

    # Sparsity (fraction of near-zero entries)
    sparsity = float(np.mean(np.abs(B_A) < 1e-6))

    return {
        "available": True,
        "n_stocks": n_stocks,
        "AU": AU,
        # Column (factor) norms
        "col_norm_mean": col_norm_mean,
        "col_norm_std": col_norm_std,
        "col_norm_max": col_norm_max,
        "col_norm_min": col_norm_min,
        "col_norms": col_norms.tolist(),
        "outlier_factors": outlier_factors,
        "n_outlier_factors": len(outlier_factors),
        # Row (stock) norms
        "row_norm_mean": row_norm_mean,
        "row_norm_std": row_norm_std,
        "row_norm_max": row_norm_max,
        "n_outlier_stocks": n_outlier_stocks,
        # Sparsity
        "sparsity": sparsity,
    }


# ---------------------------------------------------------------------------
# B.18: Pre-portfolio factor risk budget
# ---------------------------------------------------------------------------

def compute_pre_portfolio_risk_budget(
    B_A: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute factor risk budget BEFORE portfolio optimization.

    Uses equal weights (1/N) as baseline to see the "natural" risk budget
    implied by the factor model. This helps diagnose if certain factors
    dominate even before optimization.

    :param B_A (np.ndarray): Exposure matrix (n_stocks, AU)
    :param eigenvalues (np.ndarray): Factor eigenvalues (AU,)
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n_stocks,)

    :return analysis (dict): Pre-portfolio risk decomposition
    """
    if B_A.size == 0 or eigenvalues.size == 0:
        return {"available": False, "reason": "missing data"}

    n_stocks, AU = B_A.shape

    # Equal-weight portfolio
    w_eq = np.ones(n_stocks) / n_stocks

    # Portfolio exposure in factor space
    beta = B_A.T @ w_eq  # (AU,)

    # Factor risk contributions: c_k = β_k² * λ_k
    factor_contrib = (beta ** 2) * eigenvalues
    total_factor_risk = float(np.sum(factor_contrib))

    # Idiosyncratic risk (if available)
    if D_eps is not None:
        idio_contrib = (w_eq ** 2) * D_eps
        total_idio_risk = float(np.sum(idio_contrib))
        total_risk = total_factor_risk + total_idio_risk
        factor_pct = total_factor_risk / max(total_risk, 1e-10)
    else:
        total_idio_risk = 0.0
        total_risk = total_factor_risk
        factor_pct = 1.0

    # Factor risk fractions
    if total_factor_risk > 0:
        factor_fractions = factor_contrib / total_factor_risk
    else:
        factor_fractions = np.zeros(AU)

    # Entropy of factor risk budget (higher = more balanced)
    frac_pos = factor_fractions[factor_fractions > 1e-15]
    if len(frac_pos) > 0:
        entropy = float(-np.sum(frac_pos * np.log(frac_pos)))
        max_entropy = np.log(AU)
        enb = float(np.exp(entropy))
        h_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        entropy = 0.0
        enb = 0.0
        h_norm = 0.0

    # Concentration: fraction in top 3 factors
    sorted_frac = np.sort(factor_fractions)[::-1]
    top3_frac = float(np.sum(sorted_frac[:3])) if AU >= 3 else 1.0

    return {
        "available": True,
        "n_stocks": n_stocks,
        "AU": AU,
        # Risk decomposition
        "total_factor_risk": total_factor_risk,
        "total_idio_risk": total_idio_risk,
        "total_risk": total_risk,
        "factor_pct": factor_pct,
        # Factor contributions
        "factor_contrib": factor_contrib.tolist(),
        "factor_fractions": factor_fractions.tolist(),
        # Diversification metrics
        "entropy": entropy,
        "enb": enb,
        "h_norm": h_norm,
        "top3_concentration": top3_frac,
    }
