"""
Sliding window creation with per-window z-score normalization.

CONV-02: Z-score per-window, per-feature (mean 0, std 1).
CONV-05: Window shape (batch, T, F) — time is dim 1, features dim 2.

Reference: ISD Section MOD-001 — Sub-task 4.
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view


def _vectorized_ffill_bfill(arr_2d: np.ndarray) -> np.ndarray:
    """
    Per-row forward-fill then backward-fill of NaN values.

    Equivalent to ``pd.Series(row).ffill().bfill()`` applied independently
    to each row, but vectorized across all rows simultaneously.

    :param arr_2d (np.ndarray): Shape (n_windows, T), may contain NaN

    :return filled (np.ndarray): Same shape, NaN replaced where possible
    """
    out = arr_2d.copy()
    _, T = out.shape

    # Forward fill: propagate last valid value rightward
    for t in range(1, T):
        mask = np.isnan(out[:, t])
        out[mask, t] = out[mask, t - 1]

    # Backward fill: propagate first valid value leftward
    for t in range(T - 2, -1, -1):
        mask = np.isnan(out[:, t])
        out[mask, t] = out[mask, t + 1]

    return out


def create_windows(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    stock_ids: list[int],
    T: int = 504,
    stride: int = 1,
    sigma_min: float = 1e-8,
    max_zero_frac: float = 0.20,
) -> tuple[torch.Tensor, pd.DataFrame, torch.Tensor]:
    """
    Create sliding windows from returns and volatility data.

    For each stock i and window ending at date t:
    1. Extract raw returns r_{i, t-T+1:t} and vol v_{i, t-T+1:t}
    2. Exclude window if > max_zero_frac of days have zero return (suspension)
    3. Z-score returns PER WINDOW: r̃ = (r - μ_r) / max(σ_r, σ_min)
    4. Z-score volatility PER WINDOW: ṽ = (v - μ_v) / max(σ_v, σ_min)
    5. Stack → tensor (T, F=2): [r̃, ṽ]

    :param returns_df (pd.DataFrame): Log-returns (n_dates × n_stocks)
    :param vol_df (pd.DataFrame): Rolling 21-day realized vol (n_dates × n_stocks)
    :param stock_ids (list[int]): List of permno IDs to create windows for
    :param T (int): Window length in trading days
    :param stride (int): Step between consecutive windows
    :param sigma_min (float): Minimum std for z-scoring (prevents NaN)
    :param max_zero_frac (float): Maximum fraction of zero-return days

    :return windows (torch.Tensor): Shape (N, T, F), z-scored windows
    :return metadata (pd.DataFrame): Columns: stock_id, start_date, end_date
    :return raw_returns (torch.Tensor): Shape (N, T), NaN-cleaned raw returns
        (before z-scoring) for co-movement loss Spearman computation
    """
    all_windows: list[np.ndarray] = []
    all_raw_returns: list[np.ndarray] = []
    metadata_records: list[dict[str, object]] = []

    dates = returns_df.index

    for permno in stock_ids:
        if permno not in returns_df.columns or permno not in vol_df.columns:
            continue

        ret_series = np.asarray(returns_df[permno].values, dtype=np.float64)
        vol_series = np.asarray(vol_df[permno].values, dtype=np.float64)

        n_dates = len(ret_series)
        if n_dates < T:
            continue

        # Create all windows at once via stride tricks (no-copy view)
        ret_all = sliding_window_view(ret_series, T)[::stride]  # (n_win, T)
        vol_all = sliding_window_view(vol_series, T)[::stride]  # (n_win, T)
        n_windows = ret_all.shape[0]

        if n_windows == 0:
            continue

        # Window start/end indices for metadata
        start_indices = np.arange(0, n_dates - T + 1, stride)[:n_windows]
        end_indices = start_indices + T - 1

        # --- Filter 1: NaN fraction (>5% NaN in ret OR vol → skip) ---
        ret_nan_frac = np.isnan(ret_all).sum(axis=1) / T
        vol_nan_frac = np.isnan(vol_all).sum(axis=1) / T
        nan_ok = (ret_nan_frac <= 0.05) & (vol_nan_frac <= 0.05)

        if not nan_ok.any():
            continue

        ret_wins = ret_all[nan_ok].copy()
        vol_wins = vol_all[nan_ok].copy()
        win_starts = start_indices[nan_ok]
        win_ends = end_indices[nan_ok]

        # --- NaN handling: returns → 0, vol → per-window ffill/bfill then 0 ---
        ret_clean = np.nan_to_num(ret_wins, nan=0.0)
        vol_clean = _vectorized_ffill_bfill(vol_wins)
        vol_clean = np.nan_to_num(vol_clean, nan=0.0)

        # --- Filter 2: zero-return fraction (>max_zero_frac → skip) ---
        zero_frac = (np.abs(ret_clean) < 1e-12).sum(axis=1) / T
        zero_ok = zero_frac <= max_zero_frac

        if not zero_ok.any():
            continue

        ret_clean = ret_clean[zero_ok]
        vol_clean = vol_clean[zero_ok]
        win_starts = win_starts[zero_ok]
        win_ends = win_ends[zero_ok]

        # --- Z-score per feature per window (CONV-02) ---
        ret_mu = ret_clean.mean(axis=1, keepdims=True)
        ret_sigma = ret_clean.std(axis=1, keepdims=True)
        ret_sigma = np.maximum(ret_sigma, sigma_min)
        ret_zscore = (ret_clean - ret_mu) / ret_sigma

        vol_mu = vol_clean.mean(axis=1, keepdims=True)
        vol_sigma = vol_clean.std(axis=1, keepdims=True)
        vol_sigma = np.maximum(vol_sigma, sigma_min)
        vol_zscore = (vol_clean - vol_mu) / vol_sigma

        # Store raw returns before z-scoring (for co-movement Spearman)
        all_raw_returns.append(ret_clean.astype(np.float32))

        # Stack features: (n_valid, T, F=2)
        stacked = np.stack([ret_zscore, vol_zscore], axis=-1).astype(np.float32)
        all_windows.append(stacked)

        # Build metadata for surviving windows
        for i in range(len(win_starts)):
            metadata_records.append({
                "stock_id": permno,
                "start_date": dates[win_starts[i]],
                "end_date": dates[win_ends[i]],
            })

    if not all_windows:
        windows = torch.zeros(0, T, 2, dtype=torch.float32)
        raw_returns = torch.zeros(0, T, dtype=torch.float32)
        metadata = pd.DataFrame(
            columns=["stock_id", "start_date", "end_date"]
        )
        return windows, metadata, raw_returns

    # Concatenate all stocks: (N_total, T, F)
    windows = torch.from_numpy(np.concatenate(all_windows, axis=0))
    raw_returns = torch.from_numpy(np.concatenate(all_raw_returns, axis=0))
    metadata = pd.DataFrame(metadata_records)

    return windows, metadata, raw_returns
