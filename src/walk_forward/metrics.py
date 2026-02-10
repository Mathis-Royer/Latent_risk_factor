"""
Walk-forward metrics: 3 layers of evaluation.

Layer 1: VAE quality (reconstruction, AU, latent stability)
Layer 2: Risk model quality (realized vs predicted variance, EP)
Layer 3: Portfolio quality (entropy, vol, MDD, returns, Sharpe, ...)

Reference: ISD Section MOD-009 — Sub-task 4.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

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
) -> float:
    """
    Spearman rank correlation of pairwise inter-stock distances
    between retrainings. Target: ρ > 0.85.

    :param B_current (np.ndarray): Current exposure matrix (n, AU)
    :param B_previous (np.ndarray): Previous exposure matrix (n, AU)

    :return rho (float): Spearman correlation of distance matrices
    """
    from scipy.spatial.distance import pdist

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


# ---------------------------------------------------------------------------
# Layer 3: Portfolio Quality
# ---------------------------------------------------------------------------

def portfolio_metrics(
    w: np.ndarray,
    returns_oos: pd.DataFrame,
    universe: list[int],
    H_oos: float = 0.0,
    AU: int = 0,
    Sigma_hat: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Complete portfolio metrics (primary + diagnostic).

    :param w (np.ndarray): Portfolio weights (n,)
    :param returns_oos (pd.DataFrame): OOS returns (dates × stocks)
    :param universe (list[int]): Stock identifiers (permnos)
    :param H_oos (float): OOS factor entropy
    :param AU (int): Active units
    :param Sigma_hat (np.ndarray | None): Predicted covariance

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

    # Annualized return
    ann_return = float(np.mean(port_returns) * 252)

    # Annualized volatility
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0

    # Sharpe ratio
    sharpe = ann_return / max(ann_vol, 1e-10)

    # Maximum drawdown
    cum_returns = np.cumsum(port_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns - running_max
    max_drawdown = float(-np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar and Sortino
    calmar = ann_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0.0
    downside = port_returns[port_returns < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 1e-10
    sortino = ann_return / max(downside_vol, 1e-10)

    # Normalized entropy
    H_norm = H_oos / max(np.log(max(AU, 1)), 1e-10) if AU > 0 else 0.0

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

    return float(np.mean(crisis_returns) * 252)
