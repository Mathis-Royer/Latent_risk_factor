"""
Walk-forward metrics: 3 layers of evaluation.

Layer 1: VAE quality (reconstruction, AU, latent stability)
Layer 2: Risk model quality (realized vs predicted variance, EP)
Layer 3: Portfolio quality (entropy, vol, MDD, returns, Sharpe, ...)

Reference: ISD Section MOD-009 — Sub-task 4.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import subspace_angles

if TYPE_CHECKING:
    from src.walk_forward.oos_rebalancing import OOSRebalancingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: VAE Quality
# ---------------------------------------------------------------------------

def vae_reconstruction_metrics(
    train_mse: float,
    oos_mse: float,
    crisis_mse: float,
    normal_mse: float,
) -> dict[str, float]:
    """
    VAE reconstruction quality metrics.

    :param train_mse (float): Training MSE
    :param oos_mse (float): OOS MSE
    :param crisis_mse (float): MSE on crisis windows
    :param normal_mse (float): MSE on normal windows

    :return metrics (dict): Reconstruction quality metrics
    """
    return {
        "oos_train_mse_ratio": oos_mse / max(train_mse, 1e-10),
        "crisis_normal_mse_ratio": crisis_mse / max(normal_mse, 1e-10),
        "oos_mse": oos_mse,
        "train_mse": train_mse,
    }


def latent_stability(
    B_current: np.ndarray,
    B_previous: np.ndarray,
    ids_current: list[int] | None = None,
    ids_previous: list[int] | None = None,
) -> float:
    """
    Spearman rank correlation of pairwise inter-stock distances
    between retrainings. Target: ρ > 0.85.

    When stock IDs are provided, aligns the matrices by matching stocks
    that exist in both folds before computing distances. This is critical
    because the universe changes between folds (stocks enter/exit).

    :param B_current (np.ndarray): Current exposure matrix (n_curr, AU)
    :param B_previous (np.ndarray): Previous exposure matrix (n_prev, AU)
    :param ids_current (list[int] | None): Stock IDs for B_current rows
    :param ids_previous (list[int] | None): Stock IDs for B_previous rows

    :return rho (float): Spearman correlation of distance matrices
    """
    from scipy.spatial.distance import pdist

    # If stock IDs provided, align matrices by matching stocks
    if ids_current is not None and ids_previous is not None:
        # Find common stocks between folds
        set_current = set(ids_current)
        set_previous = set(ids_previous)
        common_ids = sorted(set_current & set_previous)

        if len(common_ids) < 3:
            logger.debug(
                "latent_stability: only %d common stocks between folds",
                len(common_ids),
            )
            return 0.0

        # Build index maps for alignment
        idx_curr = {sid: i for i, sid in enumerate(ids_current)}
        idx_prev = {sid: i for i, sid in enumerate(ids_previous)}

        # Extract aligned rows (same stock order for both matrices)
        rows_curr = [idx_curr[sid] for sid in common_ids]
        rows_prev = [idx_prev[sid] for sid in common_ids]

        B_curr_aligned = B_current[rows_curr]
        B_prev_aligned = B_previous[rows_prev]

        dist_current = pdist(B_curr_aligned)
        dist_previous = pdist(B_prev_aligned)
    else:
        # Fallback: truncate to min size (legacy behavior, less accurate)
        n = min(B_current.shape[0], B_previous.shape[0])
        if n < 3:
            return 0.0

        dist_current = pdist(B_current[:n])
        dist_previous = pdist(B_previous[:n])

    result = stats.spearmanr(dist_current, dist_previous)
    rho = float(result.statistic)  # type: ignore[union-attr]
    return rho if not np.isnan(rho) else 0.0


# ---------------------------------------------------------------------------
# Layer 2: Risk Model Quality
# ---------------------------------------------------------------------------

def realized_vs_predicted_variance(
    w: np.ndarray,
    Sigma_hat: np.ndarray,
    returns_oos: np.ndarray,
) -> float:
    """
    var(r_p^OOS) / (w^T Σ̂ w). Target: ∈ [0.8, 1.2].

    :param w (np.ndarray): Portfolio weights (n,)
    :param Sigma_hat (np.ndarray): Predicted covariance (n, n)
    :param returns_oos (np.ndarray): OOS returns matrix (T_oos, n)

    :return ratio (float): Realized/predicted variance ratio
    """
    predicted_var = float(w @ Sigma_hat @ w)
    port_returns = returns_oos @ w
    realized_var = float(np.var(port_returns, ddof=1))

    return realized_var / max(predicted_var, 1e-10)


def realized_vs_predicted_correlation(
    Sigma_hat: np.ndarray,
    returns_oos: np.ndarray,
) -> float:
    """
    Spearman rank correlation between predicted and realized per-stock volatilities.

    Target: high positive correlation indicates risk model ranks stocks correctly.

    :param Sigma_hat (np.ndarray): Predicted covariance matrix (n, n)
    :param returns_oos (np.ndarray): OOS returns matrix (T_oos, n)

    :return rho (float): Spearman rank correlation
    """
    n = min(Sigma_hat.shape[0], returns_oos.shape[1])
    predicted_vol = np.sqrt(np.diag(Sigma_hat[:n, :n]))
    realized_vol = np.std(returns_oos[:, :n], axis=0, ddof=1)

    if n < 3 or np.all(predicted_vol == predicted_vol[0]):
        return 0.0

    result = stats.spearmanr(predicted_vol, realized_vol)
    rho = float(result.statistic)  # type: ignore[union-attr]
    return rho if not np.isnan(rho) else 0.0


def factor_explanatory_power(
    returns: np.ndarray,
    B_A: np.ndarray,
    z_hat: np.ndarray,
) -> float:
    """
    Factor explanatory power: 1 - Var(residuals) / Var(returns).

    WARNING: B_A must use the SAME rescaling as the regression that produced
    z_hat. Mixing estimation vs portfolio rescaling gives wrong results.
    Prefer factor_explanatory_power_dynamic() when B_A_by_date is available.

    :param returns (np.ndarray): Returns (T, n)
    :param B_A (np.ndarray): Rescaled exposures at each date (n, AU)
    :param z_hat (np.ndarray): Estimated factor returns (T, AU)

    :return EP (float): Explanatory power ∈ [0, 1]
    """
    predicted = z_hat @ B_A.T  # (T, n)
    residuals = returns - predicted

    total_var = np.var(returns)
    residual_var = np.var(residuals)

    if total_var < 1e-10:
        return 0.0

    return float(1.0 - residual_var / total_var)


def factor_explanatory_power_dynamic(
    B_A_by_date: dict[str, np.ndarray],
    z_hat: np.ndarray,
    returns: "pd.DataFrame",
    universe_snapshots: dict[str, list[int]],
    valid_dates: list[str],
) -> float:
    """
    Factor EP using time-varying estimation rescaling (matches z_hat regression).

    Computes predicted_t = B_A_est(t) @ z_hat_t at each date, using the same
    estimation-rescaled exposures that produced z_hat via OLS. This avoids the
    scale mismatch that occurs when using portfolio-rescaled B_A_port.

    :param B_A_by_date (dict): date_str -> B_A_est(t) (n_active_t, AU)
    :param z_hat (np.ndarray): Estimated factor returns (n_dates, AU)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param universe_snapshots (dict): date_str -> active stock_ids
    :param valid_dates (list[str]): Dates corresponding to z_hat rows

    :return EP (float): Explanatory power in [0, 1] (or negative if model adds noise)
    """
    sid_to_col: dict[int, int] = {
        col: j for j, col in enumerate(returns.columns)
    }
    ret_values = returns.values

    # Build date -> row index mapping
    date_to_row: dict[str, int] = {}
    for i, d in enumerate(returns.index):
        if isinstance(d, pd.Timestamp):
            date_to_row[str(d.date())] = i
        else:
            date_to_row[str(d)] = i

    all_actual: list[float] = []
    all_predicted: list[float] = []

    for t_idx, date_str in enumerate(valid_dates):
        if date_str not in B_A_by_date or date_str not in date_to_row:
            continue

        B_t = B_A_by_date[date_str]
        active_sids = universe_snapshots.get(date_str, [])
        z_t = z_hat[t_idx]
        predicted_t = B_t @ z_t

        row_idx = date_to_row[date_str]
        n_active = min(len(active_sids), B_t.shape[0])

        for i in range(n_active):
            sid = active_sids[i]
            if sid not in sid_to_col:
                continue
            col_idx = sid_to_col[sid]
            r_it = ret_values[row_idx, col_idx]
            if np.isnan(r_it):
                continue
            all_actual.append(float(r_it))
            all_predicted.append(float(predicted_t[i]))

    if len(all_actual) < 10:
        return 0.0

    actual = np.array(all_actual)
    predicted = np.array(all_predicted)
    residuals = actual - predicted

    total_var = float(np.var(actual))
    residual_var = float(np.var(residuals))

    if total_var < 1e-10:
        return 0.0

    return float(1.0 - residual_var / total_var)


def eigenvector_rotation_stability(
    V_curr: np.ndarray,
    V_prev: np.ndarray,
    n_top: int = 10,
) -> dict[str, float]:
    """
    Track rotation of Sigma_z eigenvectors between folds.

    Uses scipy.linalg.subspace_angles to compute principal angles between
    the subspaces spanned by the top n_top eigenvectors of consecutive
    covariance matrices. Large rotations indicate factor instability.

    :param V_curr (np.ndarray): Current eigenvectors (K, K), columns ordered by eigenvalue
    :param V_prev (np.ndarray): Previous eigenvectors (K, K), same ordering
    :param n_top (int): Number of top eigenvectors to compare (default: 10)

    :return result (dict): {
        "mean_alignment": mean of cos(angles), target > 0.8,
        "min_alignment": minimum cos(angle),
        "n_rotated": count of eigenvectors with cos(angle) < 0.5
    }
    """
    # Validate inputs
    if V_curr.ndim != 2 or V_prev.ndim != 2:
        return {"mean_alignment": 0.0, "min_alignment": 0.0, "n_rotated": float(n_top)}

    # Limit n_top to available dimensions
    k_curr = min(V_curr.shape[1], n_top)
    k_prev = min(V_prev.shape[1], n_top)
    k = min(k_curr, k_prev)

    if k == 0:
        return {"mean_alignment": 0.0, "min_alignment": 0.0, "n_rotated": float(n_top)}

    # Extract top eigenvectors (assume columns sorted by descending eigenvalue)
    V1 = V_curr[:, :k]
    V2 = V_prev[:, :k]

    # Handle dimension mismatch: truncate to common row dimension
    min_rows = min(V1.shape[0], V2.shape[0])
    if min_rows < k:
        return {"mean_alignment": 0.0, "min_alignment": 0.0, "n_rotated": float(n_top)}

    V1 = V1[:min_rows, :]
    V2 = V2[:min_rows, :]

    # Compute principal angles between subspaces
    angles = subspace_angles(V1, V2)  # Returns angles in radians

    # Compute alignment as cosine of angles
    alignments = np.cos(angles)

    mean_alignment = float(np.mean(alignments))
    min_alignment = float(np.min(alignments))
    n_rotated = int(np.sum(alignments < 0.5))

    return {
        "mean_alignment": mean_alignment,
        "min_alignment": min_alignment,
        "n_rotated": float(n_rotated),
    }


def au_retention_analysis(
    active_dims_history: list[list[int]],
) -> dict[str, float]:
    """
    Track which latent dimensions remain active across walk-forward folds.

    Identifies "core" dimensions that persist in >80% of folds (indicating
    stable structural factors) vs ephemeral dimensions that come and go
    (potentially overfitting or noise-driven).

    :param active_dims_history (list[list[int]]): Active dimension indices per fold,
        e.g., [[0, 3, 5], [0, 2, 3, 5], [0, 3, 7], ...]

    :return result (dict): {
        "n_core": number of dimensions active in >80% of folds,
        "mean_turnover": average |symmetric_diff| / |union| between consecutive folds
    }
    """
    n_folds = len(active_dims_history)

    if n_folds == 0:
        return {"n_core": 0.0, "mean_turnover": 1.0}

    if n_folds == 1:
        # Single fold: all dimensions are "core", no turnover measurable
        return {"n_core": float(len(active_dims_history[0])), "mean_turnover": 0.0}

    # Count appearances of each dimension
    dim_counts: dict[int, int] = {}
    for dims in active_dims_history:
        for d in dims:
            dim_counts[d] = dim_counts.get(d, 0) + 1

    # Core dimensions appear in >80% of folds
    threshold = 0.80 * n_folds
    n_core = sum(1 for count in dim_counts.values() if count > threshold)

    # Compute turnover between consecutive folds
    turnovers: list[float] = []
    for i in range(1, n_folds):
        set_prev = set(active_dims_history[i - 1])
        set_curr = set(active_dims_history[i])

        symmetric_diff = len(set_prev ^ set_curr)
        union_size = len(set_prev | set_curr)

        if union_size > 0:
            turnovers.append(symmetric_diff / union_size)
        else:
            turnovers.append(0.0)

    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "n_core": float(n_core),
        "mean_turnover": mean_turnover,
    }


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict[str, float]:
    """
    Compute Deflated Sharpe Ratio (DSR) correcting for multiple testing.

    Based on Bailey & Lopez de Prado (2014), accounts for the inflation of
    Sharpe ratios when selecting the best strategy from many trials.
    Non-normality (skewness, excess kurtosis) inflates standard error.

    :param sharpe (float): Observed Sharpe ratio
    :param n_trials (int): Number of strategies/trials tested
    :param T (int): Number of observations (e.g., trading days)
    :param skew (float): Skewness of returns (default: 0.0 = normal)
    :param kurt (float): Kurtosis of returns (default: 3.0 = normal)

    :return result (dict): {
        "dsr_probability": P(true Sharpe > 0 | observed), target > 0.5,
        "expected_max_sharpe": E[max(SR)] under null,
        "sharpe_std_error": standard error of Sharpe estimate
    }
    """
    if T <= 1 or n_trials <= 0:
        return {
            "dsr_probability": 0.0,
            "expected_max_sharpe": 0.0,
            "sharpe_std_error": float("inf"),
        }

    # Euler-Mascheroni constant
    gamma = 0.5772156649

    # Expected maximum Sharpe under null (all true Sharpes = 0)
    # E[max] ≈ (1 - gamma) * Phi^{-1}(1 - 1/n) + gamma * Phi^{-1}(1 - 1/(n*e))
    # Simplified approximation for large n:
    # E[max] ≈ sqrt(2 * ln(n)) - (gamma + ln(2 * sqrt(pi))) / sqrt(2 * ln(n))
    if n_trials == 1:
        e_max = 0.0
    else:
        z_n = stats.norm.ppf(1 - 1 / n_trials)
        z_ne = stats.norm.ppf(1 - 1 / (n_trials * np.e)) if n_trials * np.e > 1 else 0.0
        e_max = (1 - gamma) * z_n + gamma * z_ne

    # Standard error of Sharpe ratio (Lo 2002, adjusted for non-normality)
    # SE(SR) = sqrt((1 + 0.5 * SR^2 - skew * SR + (kurt - 3) / 4 * SR^2) / T)
    # Simplified: SE ≈ sqrt((1 + (kurt - 1) / 4 * SR^2 - skew * SR) / (T - 1))
    excess_kurt = kurt - 3.0
    var_sr = (
        1.0
        + 0.5 * sharpe ** 2
        - skew * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    ) / max(T - 1, 1)
    se = np.sqrt(max(var_sr, 1e-10))

    # DSR probability: P(true SR > 0) = Phi((SR - E_max) / SE)
    dsr_prob = float(stats.norm.cdf((sharpe - e_max) / se))

    return {
        "dsr_probability": dsr_prob,
        "expected_max_sharpe": float(e_max),
        "sharpe_std_error": float(se),
    }


def factor_explanatory_power_oos(
    B_A_port: np.ndarray,
    returns_oos: "pd.DataFrame",
    stock_ids: list[int],
    conditioning_threshold: float = 1e6,
    ridge_scale: float = 1e-6,
) -> dict[str, float]:
    """
    Out-of-sample factor explanatory power using fixed B_A_port.

    For each OOS date t, runs cross-sectional OLS:
        z_hat_t = (B^T B)^{-1} B^T r_t
    then computes R² = 1 - Var(residuals) / Var(returns).

    This is the correct OOS metric: B was estimated on training data,
    and we measure how well its column space captures OOS return cross-sections.

    Also returns in-sample-equivalent metrics for comparison.

    :param B_A_port (np.ndarray): Portfolio-rescaled exposure (n, AU)
    :param returns_oos (pd.DataFrame): OOS log-returns (dates x stocks)
    :param stock_ids (list[int]): Stock IDs matching B_A_port rows
    :param conditioning_threshold (float): κ threshold for ridge fallback
    :param ridge_scale (float): Ridge regularization scale

    :return result (dict): {ep_oos, n_dates, avg_cs_r2, z_hat_oos_std}
    """
    from src.risk_model.conditioning import safe_solve

    n, AU = B_A_port.shape
    sid_to_col: dict[int, int] = {col: j for j, col in enumerate(returns_oos.columns)}
    ret_values = returns_oos.values

    # Map stock_ids to return columns
    col_indices = []
    valid_rows = []
    for i, sid in enumerate(stock_ids[:n]):
        if sid in sid_to_col:
            col_indices.append(sid_to_col[sid])
            valid_rows.append(i)

    if len(valid_rows) < AU:
        return {"ep_oos": 0.0, "n_dates": 0, "avg_cs_r2": 0.0, "z_hat_oos_std": 0.0}

    B_valid = B_A_port[valid_rows]  # (n_valid, AU)
    col_arr = np.array(col_indices)

    all_actual: list[float] = []
    all_predicted: list[float] = []
    cs_r2_list: list[float] = []
    z_hat_oos_list: list[np.ndarray] = []

    for t in range(ret_values.shape[0]):
        r_t = ret_values[t, col_arr].astype(np.float64)
        nan_mask = ~np.isnan(r_t)
        if nan_mask.sum() < AU:
            continue

        r_valid = r_t[nan_mask]
        B_date = B_valid[nan_mask]

        z_hat_t = safe_solve(
            B_date, r_valid,
            conditioning_threshold=conditioning_threshold,
            ridge_scale=ridge_scale,
        )
        z_hat_oos_list.append(z_hat_t)

        predicted_t = B_date @ z_hat_t
        residuals_t = r_valid - predicted_t

        for i in range(len(r_valid)):
            all_actual.append(float(r_valid[i]))
            all_predicted.append(float(predicted_t[i]))

        # Per-date cross-sectional R²
        ss_tot = float(np.sum((r_valid - r_valid.mean()) ** 2))
        ss_res = float(np.sum(residuals_t ** 2))
        if ss_tot > 1e-12:
            cs_r2_list.append(1.0 - ss_res / ss_tot)

    if len(all_actual) < 10:
        return {"ep_oos": 0.0, "n_dates": 0, "avg_cs_r2": 0.0, "z_hat_oos_std": 0.0}

    actual = np.array(all_actual)
    predicted = np.array(all_predicted)
    residuals = actual - predicted

    total_var = float(np.var(actual))
    residual_var = float(np.var(residuals))

    ep = float(1.0 - residual_var / total_var) if total_var > 1e-10 else 0.0

    z_hat_std = 0.0
    if z_hat_oos_list:
        z_oos = np.stack(z_hat_oos_list)
        z_hat_std = float(np.mean(np.std(z_oos, axis=0)))

    return {
        "ep_oos": ep,
        "n_dates": len(cs_r2_list),
        "avg_cs_r2": float(np.mean(cs_r2_list)) if cs_r2_list else 0.0,
        "z_hat_oos_std": z_hat_std,
    }


# ---------------------------------------------------------------------------
# Layer 3: Portfolio Quality
# ---------------------------------------------------------------------------

def portfolio_metrics(
    w: np.ndarray,
    returns_oos: pd.DataFrame,
    universe: list[int],
    H_oos: float = 0.0,
    AU: int = 0,
    K: int = 0,
    Sigma_hat: np.ndarray | None = None,
    n_signal: int = 0,
    H_factor: float = 0.0,
) -> dict[str, float]:
    """
    Complete portfolio metrics (primary + diagnostic).

    :param w (np.ndarray): Portfolio weights (n,)
    :param returns_oos (pd.DataFrame): OOS returns (dates × stocks)
    :param universe (list[int]): Stock identifiers (permnos)
    :param H_oos (float): OOS combined two-layer entropy
    :param AU (int): Active units
    :param K (int): Total latent capacity (for scale-invariant H_norm)
    :param Sigma_hat (np.ndarray | None): Predicted covariance
    :param n_signal (int): Number of signal eigenvalues from DGJ
    :param H_factor (float): Factor-only entropy (no idiosyncratic layer).
        Used for H_norm_signal to ensure it stays in [0, 1].

    :return metrics (dict): Portfolio performance metrics
    """
    available = [s for s in universe if s in returns_oos.columns]
    R_oos = np.nan_to_num(np.asarray(returns_oos[available].values), nan=0.0)
    w_active = w[:len(available)]

    # Guard against NaN weights (solver failure)
    if np.any(np.isnan(w_active)):
        logger.warning("NaN detected in portfolio weights — falling back to equal-weight")
        w_active = np.ones(len(available)) / len(available)

    # Portfolio returns
    port_returns = R_oos @ w_active
    n_days = len(port_returns)

    if n_days == 0:
        return {"error": 1.0}

    # Geometric annualized return — exp(sum(log_r)) for CONV-01 log returns
    cumulative = float(np.exp(np.sum(np.asarray(port_returns))))
    ann_return = cumulative ** (252.0 / n_days) - 1.0 if n_days > 0 else 0.0

    # Annualized volatility
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0

    # Sharpe ratio (geometric return / vol)
    sharpe = ann_return / max(ann_vol, 1e-10)

    # Maximum drawdown (percentage, not log-space)
    cum_returns = np.cumsum(port_returns)
    running_max = np.maximum.accumulate(cum_returns)
    log_drawdowns = cum_returns - running_max
    max_drawdown = float(1.0 - np.exp(np.min(log_drawdowns))) if len(log_drawdowns) > 0 else 0.0

    # Calmar and Sortino
    calmar = ann_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0.0
    downside = port_returns[port_returns < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 1e-10
    sortino = ann_return / max(downside_vol, 1e-10)

    # Normalized entropy:
    # - H_norm_signal uses H_factor / ln(n_signal) — factor-only entropy
    #   ensures the metric stays in [0, 1].  Using H_combined would exceed
    #   1.0 because the idiosyncratic layer inflates H beyond ln(n_signal).
    # - H_norm_au uses H_combined / ln(AU) — secondary diagnostic
    # - H_norm uses H_combined / ln(K) — legacy, for cross-fold comparability
    denom_K = max(np.log(max(K, 1)), 1e-10) if K > 0 else 1e-10
    denom_AU = max(np.log(max(AU, 1)), 1e-10) if AU > 0 else 1e-10
    denom_signal = max(np.log(max(n_signal, 2)), 1e-10) if n_signal > 1 else denom_AU
    H_norm = H_oos / denom_K if K > 0 else H_oos / denom_AU
    H_norm_au = H_oos / denom_AU if AU > 0 else 0.0
    # Use factor-only entropy for H_norm_signal (Meucci 2009)
    H_for_signal = H_factor if H_factor > 0.0 else H_oos
    H_norm_signal = H_for_signal / denom_signal if n_signal > 1 else H_norm_au

    # Effective number of positions
    eff_n = float(1.0 / np.sum(w_active ** 2)) if np.sum(w_active ** 2) > 0 else 0.0

    # Diversification ratio
    dr = 0.0
    if Sigma_hat is not None and ann_vol > 0:
        n_active = len(available)
        sigma_diag = np.sqrt(np.diag(Sigma_hat[:n_active, :n_active]))
        weighted_vol = float(w_active @ sigma_diag)
        port_vol = float(np.sqrt(w_active @ Sigma_hat[:n_active, :n_active] @ w_active))
        dr = weighted_vol / max(port_vol, 1e-10)

    # Turnover (placeholder for single period)
    n_active_positions = int(np.sum(w_active > 0.001))

    return {
        # Primary metrics
        "H_norm_oos": H_norm,
        "ann_vol_oos": ann_vol,
        "max_drawdown_oos": max_drawdown,
        # Diagnostic
        "ann_return": ann_return,
        "sharpe": sharpe,
        "calmar": calmar,
        "sortino": sortino,
        "eff_n_positions": eff_n,
        "diversification_ratio": dr,
        "n_active_positions": float(n_active_positions),
        "n_days_oos": float(n_days),
        "H_norm_au": H_norm_au,
        "H_norm_signal": H_norm_signal,
        "n_signal": float(n_signal),
    }


def crisis_period_return(
    w: np.ndarray,
    returns_oos: pd.DataFrame,
    universe: list[int],
    crisis_mask: np.ndarray,
) -> float:
    """
    Annualized return during crisis periods only (VIX > P80).

    Primary Layer 3 metric: measures diversification failure during stress.

    :param w (np.ndarray): Portfolio weights (n,)
    :param returns_oos (pd.DataFrame): OOS returns (dates × stocks)
    :param universe (list[int]): Stock identifiers (permnos)
    :param crisis_mask (np.ndarray): Boolean mask, True for crisis dates (T_oos,)

    :return ann_return_crisis (float): Annualized return during crisis, or 0.0 if no crisis dates
    """
    available = [s for s in universe if s in returns_oos.columns]
    R_oos = returns_oos[available].values
    w_active = w[:len(available)]

    port_returns = R_oos @ w_active
    mask = crisis_mask[:len(port_returns)]
    crisis_returns = port_returns[mask]

    if len(crisis_returns) == 0:
        return 0.0

    n_crisis = len(crisis_returns)
    cumulative = float(np.exp(np.sum(np.asarray(crisis_returns))))
    return cumulative ** (252.0 / n_crisis) - 1.0


def portfolio_metrics_from_oos_result(
    result: "OOSRebalancingResult",
    AU: int = 0,
    K: int = 0,
    Sigma_hat: np.ndarray | None = None,
    n_signal: int = 0,
) -> dict[str, float]:
    """
    Compute portfolio metrics from OOS rebalancing simulation result.

    Computes standard metrics (annualized return, vol, Sharpe, drawdown) plus
    rebalancing-specific metrics (turnover, transaction cost, entropy trajectory).

    :param result (OOSRebalancingResult): Result from simulate_oos_rebalancing()
    :param AU (int): Active units (for H_norm computation)
    :param K (int): Total latent capacity (for H_norm computation)
    :param Sigma_hat (np.ndarray | None): Final covariance matrix (for DR)
    :param n_signal (int): Number of signal eigenvalues from DGJ

    :return metrics (dict): Portfolio performance metrics
    """
    port_returns = result.daily_returns
    n_days = len(port_returns)

    if n_days == 0:
        return {"error": 1.0}

    # Geometric annualized return
    cumulative = float(np.exp(np.sum(port_returns)))
    ann_return = cumulative ** (252.0 / n_days) - 1.0 if n_days > 0 else 0.0

    # Annualized volatility
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0

    # Sharpe ratio (gross)
    sharpe_gross = ann_return / max(ann_vol, 1e-10)

    # Net return after transaction costs
    ann_tc = result.total_transaction_cost * (252.0 / n_days) if n_days > 0 else 0.0
    ann_return_net = ann_return - ann_tc
    sharpe_net = ann_return_net / max(ann_vol, 1e-10)

    # Maximum drawdown
    cum_returns = np.cumsum(port_returns)
    running_max = np.maximum.accumulate(cum_returns)
    log_drawdowns = cum_returns - running_max
    max_drawdown = float(1.0 - np.exp(np.min(log_drawdowns))) if len(log_drawdowns) > 0 else 0.0

    # Calmar and Sortino
    calmar = ann_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0.0
    downside = port_returns[port_returns < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 1e-10
    sortino = ann_return / max(downside_vol, 1e-10)

    # Entropy trajectory statistics
    H_traj = result.entropy_trajectory
    H_initial = H_traj[0] if H_traj else 0.0
    H_final = H_traj[-1] if H_traj else 0.0
    H_mean = float(np.mean(H_traj)) if H_traj else 0.0
    H_min = float(np.min(H_traj)) if H_traj else 0.0

    # Normalized entropy (same logic as portfolio_metrics)
    denom_K = max(np.log(max(K, 1)), 1e-10) if K > 0 else 1e-10
    denom_AU = max(np.log(max(AU, 1)), 1e-10) if AU > 0 else 1e-10
    denom_signal = max(np.log(max(n_signal, 2)), 1e-10) if n_signal > 1 else denom_AU
    H_norm = H_final / denom_K if K > 0 else H_final / denom_AU
    H_norm_au = H_final / denom_AU if AU > 0 else 0.0
    H_norm_signal = H_final / denom_signal if n_signal > 1 else H_norm_au

    # Effective number of positions (final weights)
    w_final = result.final_weights
    eff_n = float(1.0 / np.sum(w_final ** 2)) if np.sum(w_final ** 2) > 0 else 0.0
    n_active_positions = int(np.sum(w_final > 0.001))

    # Diversification ratio
    dr = 0.0
    if Sigma_hat is not None and ann_vol > 0:
        n_active = len(w_final)
        if Sigma_hat.shape[0] >= n_active:
            sigma_diag = np.sqrt(np.diag(Sigma_hat[:n_active, :n_active]))
            weighted_vol = float(w_final @ sigma_diag)
            port_vol = float(np.sqrt(w_final @ Sigma_hat[:n_active, :n_active] @ w_final))
            dr = weighted_vol / max(port_vol, 1e-10)

    # Rebalancing counts
    n_rebalances = result.n_scheduled_rebalances + result.n_exceptional_rebalances

    return {
        # Primary metrics
        "H_norm_oos": H_norm,
        "ann_vol_oos": ann_vol,
        "max_drawdown_oos": max_drawdown,
        # Return metrics
        "ann_return": ann_return,
        "ann_return_net": ann_return_net,
        "sharpe": sharpe_gross,
        "sharpe_net": sharpe_net,
        "calmar": calmar,
        "sortino": sortino,
        # Position metrics
        "eff_n_positions": eff_n,
        "diversification_ratio": dr,
        "n_active_positions": float(n_active_positions),
        "n_days_oos": float(n_days),
        # Rebalancing metrics
        "cumulative_turnover": result.cumulative_turnover,
        "total_transaction_cost": result.total_transaction_cost,
        "n_rebalances_scheduled": float(result.n_scheduled_rebalances),
        "n_rebalances_exceptional": float(result.n_exceptional_rebalances),
        "n_rebalances_total": float(n_rebalances),
        # Entropy trajectory
        "H_initial": H_initial,
        "H_final": H_final,
        "H_mean": H_mean,
        "H_min": H_min,
        "H_norm_au": H_norm_au,
        "H_norm_signal": H_norm_signal,
        "n_signal": float(n_signal),
    }
