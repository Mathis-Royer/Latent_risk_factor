"""
Covariance estimation: Sigma_z (Ledoit-Wolf), D_eps (idiosyncratic), Sigma_assets assembly.

- Sigma_z: empirical covariance of z_hat over FULL history + shrinkage
- D_eps: per-stock variance of residuals, floored at 1e-6
- Sigma_assets = B_port Sigma_z (B_port)^T + D_eps
- Eigendecomposition: Sigma_z = V Lambda V^T -> rotated B' = B_port V

Shrinkage methods for Sigma_z eigenvalues:
- "truncation": Legacy percentile-based eigenvalue truncation.
- "spiked": Donoho-Gavish-Johnstone (2018) optimal shrinker for spiked
  covariance models.  Uses the Baik-Ben Arous-Peche transition to separate
  signal from noise eigenvalues, then applies the Frobenius-loss optimal
  correction.  Reference: Ann. Stats. 46(4), 1742-1778.
- "analytical_nonlinear": Ledoit-Wolf (2020) analytical nonlinear shrinkage.
  Reference: Ann. Stats. 48(5), 3043-3065.

Reference: ISD Section MOD-007 -- Sub-tasks 3-4.
"""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Donoho-Gavish-Johnstone spiked covariance shrinker
# ---------------------------------------------------------------------------

def _spiked_shrinkage(
    eigenvalues: np.ndarray,
    n_samples: int,
    p_dims: int,
) -> tuple[np.ndarray, int]:
    """
    Optimal eigenvalue shrinkage for the spiked covariance model.

    Implements the Donoho-Gavish-Johnstone (2018) Frobenius-loss optimal
    shrinker for spiked population eigenvalues.

    The spiked model assumes the population covariance has a few eigenvalues
    (spikes) significantly above 1, and all others equal to sigma^2 (bulk).
    Sample eigenvalues are biased estimators of population eigenvalues:
    large ones are over-estimated, small ones are under-estimated.

    Algorithm:
    1. Estimate bulk variance sigma^2 from the median eigenvalue.
    2. Compute the Baik-Ben Arous-Peche (BBP) transition threshold:
       lambda_+ = sigma^2 * (1 + sqrt(gamma))^2
    3. For each sample eigenvalue above lambda_+:
       a. Invert the biasing function to recover population eigenvalue l.
       b. Compute cosine^2 of the angle between sample/population eigenvectors.
       c. Apply the Frobenius-optimal shrinker: eta = l * c^2 + sigma^2 * s^2.
    4. For eigenvalues below lambda_+: shrink to sigma^2 (bulk).

    :param eigenvalues (np.ndarray): Sample eigenvalues, sorted descending.
    :param n_samples (int): Number of observations (rows of z_hat).
    :param p_dims (int): Number of dimensions (columns of z_hat).

    :return shrunk (np.ndarray): Optimally shrunk eigenvalues, sorted descending.
    :return n_signal (int): Number of eigenvalues above the BBP threshold.
    """
    gamma = p_dims / n_samples
    sqrt_gamma = np.sqrt(gamma)

    # Estimate bulk noise level from the median eigenvalue.
    # In the spiked model, most eigenvalues are near sigma^2;
    # the median is a robust estimator of the bulk level.
    sigma_sq = float(np.median(eigenvalues))
    if sigma_sq <= 0:
        sigma_sq = float(np.mean(eigenvalues[eigenvalues > 0])) if np.any(eigenvalues > 0) else 1.0

    # BBP transition threshold (above this = signal)
    lambda_plus = sigma_sq * (1.0 + sqrt_gamma) ** 2

    shrunk = np.full_like(eigenvalues, sigma_sq)

    n_signal = 0
    for i, lam in enumerate(eigenvalues):
        if lam <= lambda_plus:
            break

        # Invert the biasing function to recover population eigenvalue l:
        #   lambda_sample = l + gamma * sigma^2 * l / (l - sigma^2)
        # Rearranging: l^2 - (lambda - sigma^2*(1+gamma))*l - lambda*sigma^2 = 0
        # (with the constraint l > sigma^2 * (1 + sqrt(gamma)))
        a_coeff = 1.0
        b_coeff = -(lam - sigma_sq * (1.0 + gamma))
        c_coeff = -lam * sigma_sq
        discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
        if discriminant < 0:
            break
        l_hat = (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff)

        # Sanity: population eigenvalue must exceed the critical threshold
        l_critical = sigma_sq * (1.0 + sqrt_gamma)
        if l_hat <= l_critical:
            break

        # Cosine^2 between sample and population eigenvectors
        # c^2(l) = (1 - gamma * sigma^4 / (l - sigma^2)^2) /
        #          (1 + gamma * sigma^2 / (l - sigma^2))
        denom = l_hat - sigma_sq
        if denom <= 1e-15:
            break
        c_sq = (1.0 - gamma * sigma_sq ** 2 / denom ** 2) / \
               (1.0 + gamma * sigma_sq / denom)
        c_sq = max(0.0, min(1.0, c_sq))
        s_sq = 1.0 - c_sq

        # Frobenius-optimal shrinker
        shrunk[i] = l_hat * c_sq + sigma_sq * s_sq
        n_signal += 1

    logger.info(
        "  Spiked shrinkage: gamma=%.3f, sigma_sq=%.4e, lambda_+=%.4e, "
        "n_signal=%d/%d",
        gamma, sigma_sq, lambda_plus, n_signal, p_dims,
    )
    return shrunk, n_signal


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def _apply_ewma_weights(
    z_hat: np.ndarray,
    ewma_half_life: int,
) -> np.ndarray:
    """
    Apply EWMA weights to center z_hat before covariance estimation.

    Returns z_hat_weighted = sqrt(w_t) * (z_hat_t - weighted_mean), so that
    np.cov(z_hat_weighted, rowvar=False, ddof=0) ≈ EWMA covariance.

    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param ewma_half_life (int): Half-life in days for exponential decay

    :return z_weighted (np.ndarray): EWMA-weighted, demeaned z_hat
    """
    n = z_hat.shape[0]
    decay = np.log(2.0) / ewma_half_life
    # weights: most recent = index n-1, oldest = index 0
    raw_w = np.exp(-decay * np.arange(n - 1, -1, -1))
    raw_w /= raw_w.sum()

    # Weighted mean and centering
    w_col = raw_w[:, np.newaxis]  # (n, 1)
    mu = np.sum(w_col * z_hat, axis=0)  # (AU,)
    z_centered = z_hat - mu[np.newaxis, :]

    # sqrt-weight transformation: cov(sqrt(w)*z) = Σ_w z z^T w
    sqrt_w = np.sqrt(raw_w)[:, np.newaxis]
    return z_centered * sqrt_w


def _estimate_n_signal_from_gap(
    Sigma_z: np.ndarray,
    gap_ratio: float = 100.0,
) -> int:
    """
    Estimate n_signal from eigenvalue gap when DGJ is not used.

    Finds the first index where eigenvalue[k]/eigenvalue[k+1] > gap_ratio,
    indicating a transition from signal to noise.  Fallback: all eigenvalues.

    :param Sigma_z (np.ndarray): Covariance matrix (AU, AU)
    :param gap_ratio (float): Minimum ratio between consecutive eigenvalues

    :return n_signal (int): Estimated number of signal eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(Sigma_z)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    p = len(eigenvalues)

    for k in range(p - 1):
        if eigenvalues[k + 1] < 1e-15:
            return k + 1
        if eigenvalues[k] / eigenvalues[k + 1] > gap_ratio:
            return k + 1

    return p


def estimate_sigma_z(
    z_hat: np.ndarray,
    eigenvalue_pct: float = 1.0,
    shrinkage_method: str = "spiked",
    ewma_half_life: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Estimate factor covariance Sigma_z with eigenvalue shrinkage.

    Each shrinkage method is applied as a **standalone** estimator:
    - "truncation": Ledoit-Wolf linear shrinkage + percentile-based
        eigenvalue truncation (eigenvalue_pct).
    - "spiked": Donoho-Gavish-Johnstone (2018) optimal shrinker applied
        to the RAW sample covariance (not LW-shrunk).  DGJ assumes raw
        sample eigenvalues as input — chaining LW then DGJ would inflate
        the bulk noise estimate and misclassify signal as noise.
    - "analytical_nonlinear": LW 2020 (requires covShrinkage package).

    When ewma_half_life > 0, applies EWMA weighting before covariance
    estimation (Barra USE4 methodology: half-life 252 days).  This gives
    more weight to recent observations, better capturing time-varying
    factor covariance structure.

    Also returns n_signal: the number of statistically significant
    eigenvalues (signal vs noise).  For "spiked", this comes from the
    BBP threshold.  For other methods, estimated from eigenvalue gap.

    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param eigenvalue_pct (float): Fraction of Sigma_z variance to retain.
        Only used when shrinkage_method="truncation".
    :param shrinkage_method (str): "truncation", "spiked", or
        "analytical_nonlinear".
    :param ewma_half_life (int): Half-life in days for EWMA weighting.
        0 = equal weights (legacy).  252 = Barra USE4 standard.

    :return Sigma_z (np.ndarray): Shrunk factor covariance (AU, AU)
    :return n_signal (int): Number of signal eigenvalues (above noise floor)
    """
    n_samples, p_dims = z_hat.shape

    # Effective sample size: when EWMA is active, n_eff < n_samples
    # because recent observations carry more weight.
    n_eff = n_samples

    # Apply EWMA weighting if requested
    if ewma_half_life > 0 and n_samples > ewma_half_life:
        z_input = _apply_ewma_weights(z_hat, ewma_half_life)
        # Compute effective sample size: n_eff = 1 / sum(w_i^2)
        # This is the Kish (1965) formula for effective sample size.
        decay = np.log(2.0) / ewma_half_life
        raw_w = np.exp(-decay * np.arange(n_samples - 1, -1, -1))
        raw_w /= raw_w.sum()
        n_eff = max(2, int(1.0 / np.sum(raw_w ** 2)))
        logger.info(
            "  Sigma_z EWMA weighting: half_life=%d, n_eff=%d (raw n=%d)",
            ewma_half_life, n_eff, n_samples,
        )
    else:
        z_input = z_hat

    if shrinkage_method == "truncation":
        # LW linear shrinkage as baseline, then truncate noisy eigenvalues
        lw = LedoitWolf()
        lw.fit(z_input)
        Sigma_z_lw: np.ndarray = lw.covariance_  # type: ignore[assignment]
        Sigma_z_trunc = _truncation_shrinkage(Sigma_z_lw, eigenvalue_pct)
        n_signal = _estimate_n_signal_from_gap(Sigma_z_trunc)
        return Sigma_z_trunc, n_signal

    if shrinkage_method == "analytical_nonlinear":
        # LW 2020 analytical nonlinear (standalone, uses raw z_input)
        # Falls back to spiked if covShrinkage not installed
        Sigma_z_sample = np.cov(z_input, rowvar=False, ddof=1)
        Sigma_z_anl = _analytical_nonlinear_shrinkage(
            z_input, Sigma_z_sample, n_eff, p_dims,
        )
        n_signal = _estimate_n_signal_from_gap(Sigma_z_anl)
        return Sigma_z_anl, n_signal

    # Default: "spiked" (DGJ) — applied to RAW sample covariance
    # DGJ assumes unbiased sample eigenvalues as input; applying LW first
    # would inflate the bulk noise estimate (sigma_sq) and raise the BBP
    # transition threshold, misclassifying signal eigenvalues as noise.
    # Pass n_eff (not n_samples) so DGJ gamma = p/n_eff is correct under EWMA.
    Sigma_z_sample = np.cov(z_input, rowvar=False, ddof=1)
    return _spiked_shrinkage_matrix(Sigma_z_sample, n_eff, p_dims)


def _spiked_shrinkage_matrix(
    Sigma_z: np.ndarray,
    n_samples: int,
    p_dims: int,
) -> tuple[np.ndarray, int]:
    """
    Apply DGJ spiked shrinkage to a covariance matrix.

    :param Sigma_z (np.ndarray): Sample covariance (AU, AU)
    :param n_samples (int): Number of time observations
    :param p_dims (int): Number of factor dimensions

    :return Sigma_z_shrunk (np.ndarray): Optimally shrunk covariance
    :return n_signal (int): Number of eigenvalues above BBP threshold
    """
    eigenvalues, V = np.linalg.eigh(Sigma_z)
    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    V = V[:, sort_idx]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    if eigenvalues.sum() <= 0.0:
        return Sigma_z, 0

    shrunk_eigs, n_signal = _spiked_shrinkage(eigenvalues, n_samples, p_dims)

    # Reconstruct: V diag(shrunk) V^T
    Sigma_shrunk = (V * shrunk_eigs[np.newaxis, :]) @ V.T
    Sigma_shrunk = 0.5 * (Sigma_shrunk + Sigma_shrunk.T)
    return Sigma_shrunk, n_signal


def _truncation_shrinkage(
    Sigma_z: np.ndarray,
    eigenvalue_pct: float,
) -> np.ndarray:
    """
    Legacy eigenvalue truncation: keep top-k explaining eigenvalue_pct.

    :param Sigma_z (np.ndarray): LW-shrunk covariance (AU, AU)
    :param eigenvalue_pct (float): Fraction of variance to retain

    :return Sigma_z_trunc (np.ndarray): Truncated covariance
    """
    if eigenvalue_pct >= 1.0:
        return Sigma_z

    eigenvalues, V = np.linalg.eigh(Sigma_z)
    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    V = V[:, sort_idx]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_var = eigenvalues.sum()

    if total_var <= 0.0:
        return Sigma_z

    cumulative = np.cumsum(eigenvalues) / total_var
    k = int(np.searchsorted(cumulative, eigenvalue_pct)) + 1
    k = min(k, len(eigenvalues))
    eigenvalues[k:] = 0.0

    Sigma_z_trunc = (V * eigenvalues[np.newaxis, :]) @ V.T
    Sigma_z_trunc = 0.5 * (Sigma_z_trunc + Sigma_z_trunc.T)
    return Sigma_z_trunc


def _analytical_nonlinear_shrinkage(
    z_hat: np.ndarray,
    Sigma_z_fallback: np.ndarray,
    n_samples: int,
    p_dims: int,
) -> np.ndarray:
    """
    Ledoit-Wolf (2020) analytical nonlinear shrinkage.

    Requires the covShrinkage package. Falls back to spiked shrinkage
    if the package is not available.

    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param Sigma_z_fallback (np.ndarray): Fallback LW covariance
    :param n_samples (int): Number of observations
    :param p_dims (int): Number of dimensions

    :return Sigma_z (np.ndarray): Nonlinearly shrunk covariance
    """
    try:
        from sklearn.covariance import LedoitWolf as _LW  # noqa: F811
        # Try analytical nonlinear from covShrinkage
        from covShrinkage import AnalyticalNonlinearShrinkage  # type: ignore[import-untyped]
        ans = AnalyticalNonlinearShrinkage()
        ans.fit(z_hat)
        result: np.ndarray = ans.covariance_  # type: ignore[assignment]
        logger.info("  Using Ledoit-Wolf 2020 analytical nonlinear shrinkage")
        return result
    except ImportError:
        logger.warning(
            "  covShrinkage not installed, falling back to spiked shrinkage. "
            "Install with: pip install covShrinkage"
        )
        Sigma_shrunk, _ = _spiked_shrinkage_matrix(Sigma_z_fallback, n_samples, p_dims)
        return Sigma_shrunk


# ---------------------------------------------------------------------------
# Idiosyncratic variance
# ---------------------------------------------------------------------------

def estimate_d_eps(
    residuals_by_stock: dict[int, list[float]],
    stock_ids: list[int],
    d_eps_floor: float = 1e-6,
    shrink_toward_mean: bool = True,
) -> np.ndarray:
    """
    Estimate idiosyncratic variances D_eps from residuals.

    D_{eps,i} = Var(eps_{i,.}) over all dates where stock i was active.
    Floor: D_{eps,i} >= d_eps_floor.

    When shrink_toward_mean=True, applies James-Stein shrinkage towards
    the cross-sectional grand mean of D_eps.  This stabilizes noisy
    per-stock variance estimates, especially for stocks with few valid
    observations.  Reference: Ledoit & Wolf (2004, Section 3.4).

    :param residuals_by_stock (dict): stock_id -> list of residuals
    :param stock_ids (list[int]): Ordered stock IDs (permnos)
    :param d_eps_floor (float): Minimum idiosyncratic variance
    :param shrink_toward_mean (bool): Apply James-Stein shrinkage

    :return D_eps (np.ndarray): Idiosyncratic variances (n,)
    """
    D_eps = np.full(len(stock_ids), d_eps_floor, dtype=np.float64)

    for i, sid in enumerate(stock_ids):
        resids = residuals_by_stock.get(sid, [])
        if len(resids) >= 2:
            var_eps = np.var(resids, ddof=1)
            D_eps[i] = max(var_eps, d_eps_floor)

    # James-Stein shrinkage: D_eps_shrunk = (1 - alpha) * D_eps + alpha * d_bar
    #
    # Correct formula for variance shrinkage (Ledoit-Wolf 2004, §3.4):
    #   alpha = sum_i Var(d_hat_i) / sum_i (d_hat_i - d_bar)^2
    # where Var(d_hat_i) = 2 * d_hat_i^2 / (T_i - 1) for chi-squared
    # distributed sample variances with T_i observations.
    #
    # The old formula (p+2)/(n * sum((d/d_bar - 1)^2)) is Stein's shrinkage
    # for normal means — incorrect for variances which have heavier tails.
    if shrink_toward_mean:
        valid_mask = D_eps > d_eps_floor
        n_valid = int(np.sum(valid_mask))
        if n_valid >= 3:
            d_bar = float(np.mean(D_eps[valid_mask]))
            if d_bar > 1e-15:
                # Compute per-stock T_i from residuals
                d_valid = D_eps[valid_mask]
                valid_sids = [stock_ids[i] for i in range(len(stock_ids)) if valid_mask[i]]
                T_per_stock = np.array([
                    max(len(residuals_by_stock.get(sid, [])), 2)
                    for sid in valid_sids
                ], dtype=np.float64)

                # Var(d_hat_i) = 2 * d_i^2 / (T_i - 1) for chi-squared
                var_d_hat = 2.0 * d_valid ** 2 / np.maximum(T_per_stock - 1.0, 1.0)
                numerator = float(np.sum(var_d_hat))
                denominator = float(np.sum((d_valid - d_bar) ** 2))

                if denominator > 1e-15:
                    alpha_js = min(1.0, numerator / denominator)
                else:
                    alpha_js = 1.0
                D_eps = (1.0 - alpha_js) * D_eps + alpha_js * d_bar
                D_eps = np.maximum(D_eps, d_eps_floor)
                logger.info(
                    "  D_eps James-Stein shrinkage (LW04): alpha=%.4f, "
                    "d_bar=%.4e, n_valid=%d, median_T=%.0f",
                    alpha_js, d_bar, n_valid, float(np.median(T_per_stock)),
                )

    return D_eps


# ---------------------------------------------------------------------------
# Risk model assembly
# ---------------------------------------------------------------------------

def assemble_risk_model(
    B_A_port: np.ndarray,
    Sigma_z: np.ndarray,
    D_eps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Assemble full asset covariance and compute eigendecomposition.

    Sigma_assets = B_port . Sigma_z . (B_port)^T + diag(D_eps)

    Eigendecomposition: Sigma_z = V Lambda V^T
    Rotated exposures: B' = B_port . V

    :param B_A_port (np.ndarray): Portfolio-rescaled exposures (n, AU)
    :param Sigma_z (np.ndarray): Factor covariance (AU, AU)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)

    :return risk_model (dict): Contains Sigma_assets, eigenvalues, V, B_prime_port
    """
    n, AU = B_A_port.shape

    Sigma_assets = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)

    # Eigendecomposition of Sigma_z
    eigenvalues, V = np.linalg.eigh(Sigma_z)

    # Sort descending
    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    V = V[:, sort_idx]

    # INV-007: eigenvalues >= 0 (PSD)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Rotated exposures in principal factor basis
    B_prime_port = B_A_port @ V

    return {
        "Sigma_assets": Sigma_assets,
        "eigenvalues": eigenvalues,
        "V": V,
        "B_prime_port": B_prime_port,
    }
