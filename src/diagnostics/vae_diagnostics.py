"""
VAE-specific diagnostics: posterior quality, KL analysis, sampling significance.

Implements diagnostics from Scale-VAE (LREC 2024) and PCF-VAE (Nature 2025):
- Per-dimension log_var distribution analysis (collapse/explosion detection)
- Per-dimension KL divergence tracking
- Sampling significance ratio (epsilon vs mu magnitude)

Reference: ISD diagnostic gaps A.1-A.9.
"""

from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# A.1-A.2: Log-variance distribution analysis
# ---------------------------------------------------------------------------

def analyze_log_var_distribution(
    log_var: np.ndarray | torch.Tensor,
    lower_bound: float = -6.0,
    upper_bound: float = 6.0,
) -> dict[str, Any]:
    """
    Analyze per-dimension log_var distribution for posterior collapse/explosion.

    Posterior collapse occurs when log_var hits the lower bound (σ → 0),
    meaning the encoder ignores input and outputs a deterministic point.
    Posterior explosion occurs when log_var hits the upper bound (σ → ∞),
    meaning the encoder outputs pure noise.

    :param log_var (np.ndarray | torch.Tensor): Log-variance (B, K) or (K,)
    :param lower_bound (float): Lower clamp bound for log_var
    :param upper_bound (float): Upper clamp bound for log_var

    :return analysis (dict): Per-dimension statistics and health indicators
    """
    if isinstance(log_var, torch.Tensor):
        log_var = log_var.detach().cpu().numpy()

    # Ensure 2D (B, K)
    if log_var.ndim == 1:
        log_var = log_var[np.newaxis, :]

    B, K = log_var.shape
    eps = 0.1  # Tolerance for "hitting" bounds

    # Per-dimension statistics (aggregate over batch)
    mean_per_dim = np.mean(log_var, axis=0)  # (K,)
    std_per_dim = np.std(log_var, axis=0)    # (K,)
    min_per_dim = np.min(log_var, axis=0)    # (K,)
    max_per_dim = np.max(log_var, axis=0)    # (K,)
    median_per_dim = np.median(log_var, axis=0)  # (K,)

    # Fraction of samples hitting bounds per dimension
    frac_at_lower = np.mean(log_var <= (lower_bound + eps), axis=0)  # (K,)
    frac_at_upper = np.mean(log_var >= (upper_bound - eps), axis=0)  # (K,)

    # Global fractions (across all dims and samples)
    global_frac_lower = float(np.mean(log_var <= (lower_bound + eps)))
    global_frac_upper = float(np.mean(log_var >= (upper_bound - eps)))

    # Collapsed dimensions: > 50% of samples hit lower bound
    collapsed_dims = np.where(frac_at_lower > 0.5)[0].tolist()
    n_collapsed = len(collapsed_dims)

    # Exploded dimensions: > 50% of samples hit upper bound
    exploded_dims = np.where(frac_at_upper > 0.5)[0].tolist()
    n_exploded = len(exploded_dims)

    # Health indicators
    collapse_severity = global_frac_lower  # 0-1, higher = worse
    explosion_severity = global_frac_upper  # 0-1, higher = worse

    return {
        "K": K,
        "B": B,
        # Per-dimension statistics
        "mean_per_dim": mean_per_dim.tolist(),
        "std_per_dim": std_per_dim.tolist(),
        "min_per_dim": min_per_dim.tolist(),
        "max_per_dim": max_per_dim.tolist(),
        "median_per_dim": median_per_dim.tolist(),
        # Bounds analysis
        "frac_at_lower_per_dim": frac_at_lower.tolist(),
        "frac_at_upper_per_dim": frac_at_upper.tolist(),
        "global_frac_at_lower": global_frac_lower,
        "global_frac_at_upper": global_frac_upper,
        # Collapsed/exploded dimensions
        "collapsed_dims": collapsed_dims,
        "n_collapsed": n_collapsed,
        "exploded_dims": exploded_dims,
        "n_exploded": n_exploded,
        # Severity scores (0-1)
        "collapse_severity": collapse_severity,
        "explosion_severity": explosion_severity,
        # Bounds used
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


# ---------------------------------------------------------------------------
# A.3: Per-dimension KL divergence
# ---------------------------------------------------------------------------

def compute_kl_per_dimension(
    mu: np.ndarray | torch.Tensor,
    log_var: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """
    Compute KL divergence per latent dimension.

    KL_k = (1/B) * Σ_i 0.5 * (μ²_ik + exp(lv_ik) - lv_ik - 1)

    This identifies which dimensions carry information (high KL) vs which
    are ignored (KL ≈ 0, posterior ≈ prior).

    :param mu (np.ndarray | torch.Tensor): Latent mean (B, K)
    :param log_var (np.ndarray | torch.Tensor): Latent log-variance (B, K)

    :return analysis (dict): Per-dimension KL and derived statistics
    """
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(log_var, torch.Tensor):
        log_var = log_var.detach().cpu().numpy()

    # Ensure 2D
    if mu.ndim == 1:
        mu = mu[np.newaxis, :]
    if log_var.ndim == 1:
        log_var = log_var[np.newaxis, :]

    B, K = mu.shape

    # KL per sample per dimension
    var = np.exp(log_var)
    kl_per_sample_dim = 0.5 * (mu ** 2 + var - log_var - 1.0)  # (B, K)

    # Average over batch → KL per dimension
    kl_per_dim = np.mean(kl_per_sample_dim, axis=0)  # (K,)

    # Total KL (sum over dims, averaged over batch)
    kl_total = float(np.sum(kl_per_dim))

    # Active units (KL > 0.01 nats)
    au_threshold = 0.01
    active_mask = kl_per_dim > au_threshold
    AU = int(np.sum(active_mask))
    active_dims = np.where(active_mask)[0].tolist()

    # KL distribution statistics
    kl_sorted = np.sort(kl_per_dim)[::-1]
    kl_top3_sum = float(np.sum(kl_sorted[:3])) if K >= 3 else float(np.sum(kl_sorted))
    kl_top3_fraction = kl_top3_sum / max(kl_total, 1e-10)

    # Effective number of dimensions (exponential of entropy)
    if kl_total > 0:
        kl_probs = kl_per_dim / kl_total
        kl_probs_pos = kl_probs[kl_probs > 1e-20]
        kl_entropy = float(-np.sum(kl_probs_pos * np.log(kl_probs_pos)))
        eff_dims = float(np.exp(kl_entropy))
    else:
        kl_entropy = 0.0
        eff_dims = 0.0

    # KL balance: entropy normalized by ln(K)
    # 1.0 = perfectly balanced, 0.0 = all in one dimension
    max_entropy = np.log(K) if K > 1 else 1.0
    kl_balance = kl_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "K": K,
        "B": B,
        "kl_per_dim": kl_per_dim.tolist(),
        "kl_sorted": kl_sorted.tolist(),
        "kl_total": kl_total,
        "AU": AU,
        "active_dims": active_dims,
        "au_threshold": au_threshold,
        # Distribution statistics
        "kl_mean": float(np.mean(kl_per_dim)),
        "kl_std": float(np.std(kl_per_dim)),
        "kl_max": float(np.max(kl_per_dim)),
        "kl_min": float(np.min(kl_per_dim)),
        "kl_top3_fraction": kl_top3_fraction,
        # Effective dimensions
        "kl_entropy": kl_entropy,
        "eff_dims": eff_dims,
        "kl_balance": kl_balance,
    }


# ---------------------------------------------------------------------------
# A.4: Sampling significance (epsilon vs mu ratio)
# ---------------------------------------------------------------------------

def compute_sampling_significance(
    mu: np.ndarray | torch.Tensor,
    log_var: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """
    Compute sampling significance: ratio of epsilon magnitude to mu magnitude.

    z = mu + epsilon * sigma, where epsilon ~ N(0, 1)
    If |epsilon * sigma| >> |mu|, sampling dominates (high stochasticity)
    If |epsilon * sigma| << |mu|, mu dominates (near-deterministic)

    Healthy VAE: ratio ≈ 1.0 (balanced contribution)

    :param mu (np.ndarray | torch.Tensor): Latent mean (B, K)
    :param log_var (np.ndarray | torch.Tensor): Latent log-variance (B, K)
    :return analysis (dict): Per-dimension and global significance ratios
    """
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(log_var, torch.Tensor):
        log_var = log_var.detach().cpu().numpy()

    # Ensure 2D
    if mu.ndim == 1:
        mu = mu[np.newaxis, :]
    if log_var.ndim == 1:
        log_var = log_var[np.newaxis, :]

    B, K = mu.shape
    sigma = np.exp(0.5 * log_var)  # (B, K)

    # Average magnitudes over samples
    mu_magnitude = np.abs(mu)  # (B, K)
    mu_avg_magnitude = np.mean(mu_magnitude, axis=0)  # (K,)

    # Expected |epsilon * sigma| = sigma * E[|epsilon|] = sigma * sqrt(2/pi)
    eps_sigma_expected = sigma * np.sqrt(2.0 / np.pi)  # (B, K)
    eps_sigma_avg = np.mean(eps_sigma_expected, axis=0)  # (K,)

    # Ratio per dimension
    ratio_per_dim = eps_sigma_avg / np.maximum(mu_avg_magnitude, 1e-10)  # (K,)

    # Global ratio
    global_mu_mag = float(np.mean(mu_avg_magnitude))
    global_eps_mag = float(np.mean(eps_sigma_avg))
    global_ratio = global_eps_mag / max(global_mu_mag, 1e-10)

    # Classify dimensions
    # High stochasticity: ratio > 2.0 (epsilon dominates)
    # Low stochasticity: ratio < 0.5 (mu dominates)
    # Balanced: 0.5 <= ratio <= 2.0
    high_stoch_dims = np.where(ratio_per_dim > 2.0)[0].tolist()
    low_stoch_dims = np.where(ratio_per_dim < 0.5)[0].tolist()
    balanced_dims = np.where((ratio_per_dim >= 0.5) & (ratio_per_dim <= 2.0))[0].tolist()

    return {
        "K": K,
        "B": B,
        # Per-dimension magnitudes
        "mu_magnitude_per_dim": mu_avg_magnitude.tolist(),
        "eps_sigma_magnitude_per_dim": eps_sigma_avg.tolist(),
        "ratio_per_dim": ratio_per_dim.tolist(),
        # Global statistics
        "global_mu_magnitude": global_mu_mag,
        "global_eps_sigma_magnitude": global_eps_mag,
        "global_ratio": global_ratio,
        # Classification
        "high_stochasticity_dims": high_stoch_dims,
        "low_stochasticity_dims": low_stoch_dims,
        "balanced_dims": balanced_dims,
        "n_high_stoch": len(high_stoch_dims),
        "n_low_stoch": len(low_stoch_dims),
        "n_balanced": len(balanced_dims),
        # Summary statistics
        "ratio_mean": float(np.mean(ratio_per_dim)),
        "ratio_std": float(np.std(ratio_per_dim)),
        "ratio_min": float(np.min(ratio_per_dim)),
        "ratio_max": float(np.max(ratio_per_dim)),
    }


# ---------------------------------------------------------------------------
# Training history analysis
# ---------------------------------------------------------------------------

def analyze_training_log_var_history(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze log_var statistics evolution across training epochs.

    Requires history entries to contain 'log_var_stats' dict with:
    - mean_per_dim: list[float]
    - frac_at_lower: list[float]
    - frac_at_upper: list[float]

    :param history (list[dict]): Training history from VAETrainer.fit()

    :return analysis (dict): Evolution metrics and trend detection
    """
    if not history:
        return {"available": False, "reason": "empty history"}

    # Check if log_var_stats are available
    has_stats = any("log_var_stats" in h for h in history)
    if not has_stats:
        return {"available": False, "reason": "log_var_stats not in history"}

    n_epochs = len(history)

    # Extract per-epoch statistics
    collapse_fracs: list[float] = []
    explosion_fracs: list[float] = []
    mean_log_vars: list[float] = []

    for h in history:
        stats = h.get("log_var_stats", {})
        if not stats:
            continue

        frac_lower = stats.get("frac_at_lower", [])
        frac_upper = stats.get("frac_at_upper", [])
        mean_per_dim = stats.get("mean_per_dim", [])

        if frac_lower:
            collapse_fracs.append(float(np.mean(frac_lower)))
        if frac_upper:
            explosion_fracs.append(float(np.mean(frac_upper)))
        if mean_per_dim:
            mean_log_vars.append(float(np.mean(mean_per_dim)))

    if not collapse_fracs:
        return {"available": False, "reason": "no valid log_var_stats entries"}

    # Trend analysis
    def _compute_trend(values: list[float]) -> str:
        if len(values) < 3:
            return "insufficient_data"
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"

    collapse_trend = _compute_trend(collapse_fracs)
    explosion_trend = _compute_trend(explosion_fracs)
    mean_trend = _compute_trend(mean_log_vars)

    return {
        "available": True,
        "n_epochs": n_epochs,
        "n_entries": len(collapse_fracs),
        # Time series
        "collapse_fracs": collapse_fracs,
        "explosion_fracs": explosion_fracs,
        "mean_log_vars": mean_log_vars,
        # Final values
        "final_collapse_frac": collapse_fracs[-1] if collapse_fracs else 0.0,
        "final_explosion_frac": explosion_fracs[-1] if explosion_fracs else 0.0,
        "final_mean_log_var": mean_log_vars[-1] if mean_log_vars else 0.0,
        # Trends
        "collapse_trend": collapse_trend,
        "explosion_trend": explosion_trend,
        "mean_log_var_trend": mean_trend,
    }


# ---------------------------------------------------------------------------
# G.2: Reconstruction temporal structure analysis
# ---------------------------------------------------------------------------

def analyze_reconstruction_temporal_structure(
    errors: np.ndarray,
) -> dict[str, float]:
    """
    Analyze temporal structure of reconstruction errors.

    Detects two issues:
    1. Temporal autocorrelation: if ACF(1) is high, errors are correlated in time,
       indicating the model fails to capture temporal dynamics.
    2. Boundary effects: if errors are higher at boundaries (start/end of window),
       padding in conv layers may cause artifacts.

    :param errors (np.ndarray): Per-timestamp MSE with shape (B, T)

    :return analysis (dict): Temporal structure metrics
        - acf_1: lag-1 autocorrelation of mean temporal profile
        - boundary_ratio: mean boundary error / mean interior error
    """
    if errors.ndim != 2:
        raise ValueError(f"Expected errors shape (B, T), got {errors.shape}")

    _, T = errors.shape

    # Mean error per timestamp (aggregate over batch)
    mean_per_t = np.mean(errors, axis=0)  # (T,)

    # Lag-1 autocorrelation of temporal profile
    if T > 1:
        acf_1 = float(np.corrcoef(mean_per_t[:-1], mean_per_t[1:])[0, 1])
    else:
        acf_1 = 0.0

    # Handle NaN from constant arrays
    if np.isnan(acf_1):
        acf_1 = 0.0

    # Boundary analysis: compare boundary regions to interior
    boundary = max(T // 10, 1)  # At least 1 timestep

    # Boundary regions: first and last `boundary` timesteps
    boundary_errors = np.concatenate([mean_per_t[:boundary], mean_per_t[-boundary:]])
    interior_errors = mean_per_t[boundary:-boundary] if T > 2 * boundary else mean_per_t

    # Boundary ratio: mean boundary / mean interior
    mean_boundary = float(np.mean(boundary_errors))
    mean_interior = float(np.mean(interior_errors)) if len(interior_errors) > 0 else mean_boundary

    boundary_ratio = mean_boundary / max(mean_interior, 1e-10)

    return {
        "acf_1": acf_1,
        "boundary_ratio": boundary_ratio,
    }


# ---------------------------------------------------------------------------
# G.5: Mutual information estimation (input-latent)
# ---------------------------------------------------------------------------

def estimate_mutual_information(
    x_flat: np.ndarray,
    z: np.ndarray,
    n_neighbors: int = 5,
) -> dict[str, Any]:
    """
    Estimate mutual information between input features and latent dimensions.

    Uses k-NN based MI estimation from sklearn. Higher MI indicates the latent
    dimension captures more information from the input.

    :param x_flat (np.ndarray): Flattened input with shape (B, D) where D = T * F
    :param z (np.ndarray): Latent representations with shape (B, K)
    :param n_neighbors (int): Number of neighbors for MI estimation

    :return analysis (dict): Mutual information metrics
        - total_mi: sum of MI across all latent dimensions
        - mi_per_dim: MI for each latent dimension (list)
    """
    from sklearn.feature_selection import mutual_info_regression

    if x_flat.ndim != 2:
        raise ValueError(f"Expected x_flat shape (B, D), got {x_flat.shape}")
    if z.ndim != 2:
        raise ValueError(f"Expected z shape (B, K), got {z.shape}")
    if x_flat.shape[0] != z.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x_flat has {x_flat.shape[0]}, z has {z.shape[0]}"
        )

    _, K = z.shape

    # Compute MI for each latent dimension
    mi_per_dim: list[float] = []
    for k in range(K):
        # MI between all input features and latent dimension k
        mi_values = mutual_info_regression(
            x_flat, z[:, k], n_neighbors=n_neighbors, random_state=42
        )
        # Sum MI contributions from all input features
        mi_per_dim.append(float(np.sum(mi_values)))

    total_mi = float(np.sum(mi_per_dim))

    return {
        "total_mi": total_mi,
        "mi_per_dim": mi_per_dim,
    }
