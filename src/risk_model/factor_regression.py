"""
Cross-sectional factor regression (OLS / iterative WLS).

At each date t: z_hat_t = solve(B_A_t, r_t) via SVD-based lstsq.

When use_wls=True, a two-pass approach corrects for heteroscedasticity:
  Pass 1: OLS -> z_hat_ols, residuals -> per-stock sigma_eps^2
  Pass 2: WLS with weights = 1/sigma_eps^2 (BLUE estimates)

Uses date-specific rescaling (estimation, NOT portfolio).

Reference: ISD Section MOD-007 — Sub-task 2.
Literature: Fama & MacBeth (1973), Barra USE4 (Menchero et al. 2011).
"""

import numpy as np
import pandas as pd

from src.validation import assert_finite_2d
from src.risk_model.conditioning import safe_solve, safe_solve_wls

# Type alias for pre-computed date data
_DateTuple = tuple[str, np.ndarray, np.ndarray, list[int]]


def _prepare_regression_data(
    B_A_by_date: dict[str, np.ndarray],
    returns: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
) -> list[_DateTuple]:
    """
    Pre-compute valid (date, B_t, r_t, stock_ids) tuples for regression.

    Filters out dates with insufficient stocks or NaN returns.
    Shared between OLS and WLS passes to avoid redundant computation.

    :param B_A_by_date (dict): date_str -> B_A_t (n_active_t, AU)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param universe_snapshots (dict): date_str -> list of active stock_ids

    :return date_data (list): Valid (date_str, B_t, r_t, stock_ids) tuples
    """
    sorted_dates = sorted(B_A_by_date.keys())
    ret_matrix = returns.values  # (n_dates, n_stocks)
    ret_dates = returns.index
    ret_col_to_idx = {col: j for j, col in enumerate(returns.columns)}
    ret_date_to_loc: dict[str, int] = {}
    for i, d in enumerate(ret_dates):
        if isinstance(d, str):
            ret_date_to_loc[d] = i
        elif isinstance(d, pd.Timestamp):
            ret_date_to_loc[str(d)] = i
            ret_date_to_loc[str(d.date())] = i
        else:
            ret_date_to_loc[str(d)] = i

    date_data: list[_DateTuple] = []
    for date_str in sorted_dates:
        B_t = B_A_by_date[date_str]  # (n_active, AU)
        if B_t.shape[0] < B_t.shape[1]:
            continue

        active_stocks = universe_snapshots.get(date_str, [])
        date_loc = ret_date_to_loc.get(date_str)
        if date_loc is None:
            continue

        # Filter to stocks with return data
        avail_stocks = [s for s in active_stocks if s in ret_col_to_idx]
        col_indices = [ret_col_to_idx[s] for s in avail_stocks]
        if len(col_indices) < B_t.shape[1]:
            continue

        # BUG FIX: Slice B_t to match avail_stocks (subset of active_stocks)
        # B_t rows correspond to active_stocks, but we only have returns for avail_stocks
        active_to_idx = {sid: i for i, sid in enumerate(active_stocks)}
        avail_rows = [active_to_idx[s] for s in avail_stocks]
        B_t_avail = B_t[avail_rows]  # Now aligned with avail_stocks

        r_t = ret_matrix[date_loc][col_indices].astype(np.float64)

        # Handle NaN in returns
        valid_mask = ~np.isnan(r_t)
        if valid_mask.sum() < B_t.shape[1]:
            continue

        r_t_valid = r_t[valid_mask]
        B_t_valid = B_t_avail[valid_mask]  # Use sliced B_t
        valid_sids = [avail_stocks[j] for j in range(len(avail_stocks))
                      if valid_mask[j]]

        date_data.append((date_str, B_t_valid, r_t_valid, valid_sids))

    return date_data


def estimate_factor_returns(
    B_A_by_date: dict[str, np.ndarray],
    returns: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
    conditioning_threshold: float = 1e6,
    ridge_scale: float = 1e-6,
    use_wls: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Cross-sectional regression at each date t for factor returns z_hat_t.

    When use_wls=False (default): standard OLS via SVD-based lstsq.
    When use_wls=True: iterative two-pass WLS (Fama-MacBeth 1973):
      Pass 1: OLS -> z_hat_ols, eps = r - B z_hat -> per-stock sigma_eps^2
      Pass 2: WLS with weights = 1/sigma_eps^2 (BLUE estimates)

    The conditioning_threshold and ridge_scale parameters are retained
    for API compatibility but no longer used internally.

    :param B_A_by_date (dict): date_str -> B_A_t (n_active_t, AU)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param universe_snapshots (dict): date_str -> list of active stock_ids
    :param conditioning_threshold (float): Unused (kept for API compat)
    :param ridge_scale (float): Unused (kept for API compat)
    :param use_wls (bool): If True, use iterative WLS for BLUE estimates

    :return z_hat (np.ndarray): Factor returns (n_dates, AU)
    :return dates (list[str]): Dates for which z_hat was estimated
    """
    date_data = _prepare_regression_data(
        B_A_by_date, returns, universe_snapshots,
    )

    if not date_data:
        AU = next(iter(B_A_by_date.values())).shape[1] if B_A_by_date else 0
        return np.empty((0, AU), dtype=np.float64), []

    # --- Pass 1: OLS ---
    z_hat_ols: list[np.ndarray] = []
    for _, B_t, r_t, _ in date_data:
        z_t = safe_solve(B_t, r_t)
        assert_finite_2d(z_t.reshape(1, -1), "z_hat_t_ols")
        z_hat_ols.append(z_t)

    if not use_wls:
        return np.stack(z_hat_ols, axis=0), [d[0] for d in date_data]

    # --- WLS: compute per-stock residual variance from OLS ---
    stock_ss: dict[int, float] = {}   # sum of squared residuals
    stock_n: dict[int, int] = {}      # observation count

    for t_idx, (_, B_t, r_t, sids) in enumerate(date_data):
        eps_t = r_t - B_t @ z_hat_ols[t_idx]
        assert np.isfinite(eps_t).all(), f"Residuals contain NaN/Inf at t={t_idx}"
        for i, sid in enumerate(sids):
            stock_ss[sid] = stock_ss.get(sid, 0.0) + float(eps_t[i] ** 2)
            stock_n[sid] = stock_n.get(sid, 0) + 1

    # Per-stock idiosyncratic variance (unbiased estimator)
    stock_var: dict[int, float] = {}
    for sid in stock_ss:
        n = stock_n[sid]
        stock_var[sid] = stock_ss[sid] / max(n - 1, 1)

    # Floor at 1% of median to prevent extreme weights on low-vol stocks
    var_vals = np.array(list(stock_var.values()), dtype=np.float64)
    var_floor = max(float(np.median(var_vals)) * 0.01, 1e-20)

    # --- Pass 2: WLS with weights = 1/sigma_eps^2 ---
    z_hat_wls: list[np.ndarray] = []
    for _, B_t, r_t, sids in date_data:
        w = np.array([
            1.0 / max(stock_var.get(sid, var_floor), var_floor)
            for sid in sids
        ], dtype=np.float64)
        assert np.all(w > 0) and np.isfinite(w).all(), "WLS weights must be positive and finite"
        z_t_wls = safe_solve_wls(B_t, r_t, w)
        assert_finite_2d(z_t_wls.reshape(1, -1), "z_hat_t_wls")
        z_hat_wls.append(z_t_wls)

    return np.stack(z_hat_wls, axis=0), [d[0] for d in date_data]


def compute_residuals(
    B_A_by_date: dict[str, np.ndarray],
    z_hat: np.ndarray,
    returns: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
    dates: list[str],
    stock_ids: list[int],
) -> dict[int, list[float]]:
    """
    Compute idiosyncratic residuals: eps_{i,t} = r_{i,t} - B_A_{i,t} z_hat_t

    Uses date-specific rescaling (estimation), NOT portfolio rescaling.

    :param B_A_by_date (dict): date_str -> B_A_t (n_active_t, AU)
    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param universe_snapshots (dict): date_str -> active stock_ids (permnos)
    :param dates (list[str]): Dates corresponding to z_hat rows
    :param stock_ids (list[int]): All stock IDs (for residual aggregation)

    :return residuals_by_stock (dict): stock_id -> list of residuals
    """
    residuals_by_stock: dict[int, list[float]] = {sid: [] for sid in stock_ids}

    # Pre-extract returns as numpy for fast access
    ret_matrix = returns.values
    ret_dates = returns.index
    ret_col_to_idx = {col: j for j, col in enumerate(returns.columns)}
    ret_date_to_loc: dict[str, int] = {}
    for i, d in enumerate(ret_dates):
        if isinstance(d, str):
            ret_date_to_loc[d] = i
        elif isinstance(d, pd.Timestamp):
            ret_date_to_loc[str(d)] = i
            ret_date_to_loc[str(d.date())] = i
        else:
            ret_date_to_loc[str(d)] = i

    for t_idx, date_str in enumerate(dates):
        if date_str not in B_A_by_date:
            continue

        B_t = B_A_by_date[date_str]
        active_stocks = universe_snapshots.get(date_str, [])
        available_cols = [s for s in active_stocks if s in ret_col_to_idx]

        date_loc = ret_date_to_loc.get(date_str)
        if date_loc is None:
            continue

        # BUG FIX: Slice B_t to match available_cols (subset of active_stocks)
        active_to_idx = {sid: i for i, sid in enumerate(active_stocks)}
        avail_rows = [active_to_idx[s] for s in available_cols]
        B_t_avail = B_t[avail_rows]  # Now aligned with available_cols

        col_indices = [ret_col_to_idx[s] for s in available_cols]
        r_t = ret_matrix[date_loc][col_indices].astype(np.float64)

        # eps_{i,t} = r_{i,t} - B_A_{i,t} z_hat_t
        predicted = B_t_avail @ z_hat[t_idx]
        residuals = r_t - predicted

        for i, sid in enumerate(available_cols):
            if i < len(residuals) and not np.isnan(residuals[i]) and sid in residuals_by_stock:
                residuals_by_stock[sid].append(float(residuals[i]))

    return residuals_by_stock
