"""
Cross-sectional OLS factor regression.

At each date t: ẑ_t = (B̃_{A,t}^T B̃_{A,t})^{-1} B̃_{A,t}^T r_t

Uses date-specific rescaling (estimation, NOT portfolio).
Conditioning guard applied when κ(B^T B) > 10^6.

Reference: ISD Section MOD-007 — Sub-task 2.
"""

import numpy as np
import pandas as pd

from src.risk_model.conditioning import safe_solve


def estimate_factor_returns(
    B_A_by_date: dict[str, np.ndarray],
    returns: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
    conditioning_threshold: float = 1e6,
    ridge_scale: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """
    Cross-sectional OLS at each date t using date-specific rescaled exposures.

    ẑ_t = (B̃_t^T B̃_t)^{-1} B̃_t^T r_t

    :param B_A_by_date (dict): date_str → B̃_{A,t} (n_active_t, AU)
    :param returns (pd.DataFrame): Log-returns (dates × stocks)
    :param universe_snapshots (dict): date_str → list of active stock_ids (permnos)
    :param conditioning_threshold (float): κ threshold for ridge fallback
    :param ridge_scale (float): Ridge scale factor

    :return z_hat (np.ndarray): Factor returns (n_dates, AU)
    :return dates (list[str]): Dates for which z_hat was estimated
    """
    sorted_dates = sorted(B_A_by_date.keys())
    z_hat_list: list[np.ndarray] = []
    valid_dates: list[str] = []

    for date_str in sorted_dates:
        B_t = B_A_by_date[date_str]  # (n_active, AU)
        if B_t.shape[0] < B_t.shape[1]:
            # Underdetermined: skip this date
            continue

        # Get returns for active stocks
        active_stocks = universe_snapshots.get(date_str, [])
        if date_str not in returns.index:
            continue

        # Match stocks: B_t rows correspond to the same stocks as universe_snapshots
        available_cols = [s for s in active_stocks if s in returns.columns]
        if len(available_cols) < B_t.shape[1]:
            continue

        r_t = returns.loc[date_str, available_cols].values.astype(np.float64)

        # Handle NaN in returns: drop stocks with NaN
        valid_mask = ~np.isnan(r_t)
        if valid_mask.sum() < B_t.shape[1]:
            continue

        r_t_valid = r_t[valid_mask]
        B_t_valid = B_t[valid_mask]

        # OLS with conditioning guard
        z_hat_t = safe_solve(
            B_t_valid, r_t_valid,
            conditioning_threshold=conditioning_threshold,
            ridge_scale=ridge_scale,
        )

        z_hat_list.append(z_hat_t)
        valid_dates.append(date_str)

    if not z_hat_list:
        # Return empty arrays with correct AU dimension
        AU = next(iter(B_A_by_date.values())).shape[1] if B_A_by_date else 0
        return np.empty((0, AU), dtype=np.float64), []

    z_hat = np.stack(z_hat_list, axis=0)  # (n_dates, AU)
    return z_hat, valid_dates


def compute_residuals(
    B_A_by_date: dict[str, np.ndarray],
    z_hat: np.ndarray,
    returns: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
    dates: list[str],
    stock_ids: list[int],
) -> dict[int, list[float]]:
    """
    Compute idiosyncratic residuals: ε_{i,t} = r_{i,t} - B̃_{A,i,t} ẑ_t

    Uses date-specific rescaling (estimation), NOT portfolio rescaling.

    :param B_A_by_date (dict): date_str → B̃_{A,t} (n_active_t, AU)
    :param z_hat (np.ndarray): Factor returns (n_dates, AU)
    :param returns (pd.DataFrame): Log-returns (dates × stocks)
    :param universe_snapshots (dict): date_str → active stock_ids (permnos)
    :param dates (list[str]): Dates corresponding to z_hat rows
    :param stock_ids (list[int]): All stock IDs (for residual aggregation)

    :return residuals_by_stock (dict): stock_id → list of residuals
    """
    residuals_by_stock: dict[int, list[float]] = {sid: [] for sid in stock_ids}

    for t_idx, date_str in enumerate(dates):
        if date_str not in B_A_by_date:
            continue

        B_t = B_A_by_date[date_str]
        active_stocks = universe_snapshots.get(date_str, [])
        available_cols = [s for s in active_stocks if s in returns.columns]

        if date_str not in returns.index:
            continue

        r_t = returns.loc[date_str, available_cols].values.astype(np.float64)

        # ε_{i,t} = r_{i,t} - B̃_{A,i,t} ẑ_t
        predicted = B_t[:len(available_cols)] @ z_hat[t_idx]
        residuals = r_t[:len(predicted)] - predicted

        for i, sid in enumerate(available_cols[:len(residuals)]):
            if not np.isnan(residuals[i]) and sid in residuals_by_stock:
                residuals_by_stock[sid].append(float(residuals[i]))

    return residuals_by_stock
