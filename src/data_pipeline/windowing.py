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

from src.validation import (
    assert_finite_2d,
    assert_no_nan_tensor,
    assert_z_score_normalized,
    assert_tensor_bounds,
)


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


def _count_valid_windows_for_stock(
    ret_series: np.ndarray,
    vol_series: np.ndarray,
    T: int,
    stride: int,
    max_zero_frac: float,
) -> int:
    """
    Count how many valid windows a stock produces (pass 1 of two-pass algorithm).

    This duplicates the filtering logic from create_windows but only counts,
    avoiding memory allocation for intermediate arrays.

    :param ret_series (np.ndarray): Returns for one stock (n_dates,)
    :param vol_series (np.ndarray): Volatility for one stock (n_dates,)
    :param T (int): Window length
    :param stride (int): Step between windows
    :param max_zero_frac (float): Maximum fraction of zero-return days

    :return count (int): Number of valid windows
    """
    n_dates = len(ret_series)
    if n_dates < T:
        return 0

    # Create all windows via stride tricks (no-copy view)
    ret_all = sliding_window_view(ret_series, T)[::stride]
    vol_all = sliding_window_view(vol_series, T)[::stride]
    n_windows = ret_all.shape[0]

    if n_windows == 0:
        return 0

    # Filter 1: NaN fraction (>5% NaN in ret OR vol → skip)
    ret_nan_frac = np.isnan(ret_all).sum(axis=1) / T
    vol_nan_frac = np.isnan(vol_all).sum(axis=1) / T
    nan_ok = (ret_nan_frac <= 0.05) & (vol_nan_frac <= 0.05)

    if not nan_ok.any():
        return 0

    # Apply NaN filter to get cleaned returns for zero-frac check
    ret_wins = ret_all[nan_ok]
    ret_clean = np.nan_to_num(ret_wins, nan=0.0)

    # Filter 2: zero-return fraction (>max_zero_frac → skip)
    zero_frac = (np.abs(ret_clean) < 1e-12).sum(axis=1) / T
    zero_ok = zero_frac <= max_zero_frac

    return int(zero_ok.sum())


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

    Memory optimization: Uses two-pass algorithm to pre-allocate output arrays,
    avoiding memory peak from list accumulation + concatenation (~16GB savings).

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
    # CRITICAL: Validate returns and vol DataFrames have same shape (diagnostic fix)
    assert returns_df.shape == vol_df.shape, (
        f"returns_df shape {returns_df.shape} != vol_df shape {vol_df.shape}"
    )

    dates = returns_df.index

    # =========================================================================
    # PASS 1: Count total valid windows per stock (memory-efficient)
    # =========================================================================
    window_counts: dict[int, int] = {}
    total_windows = 0

    for permno in stock_ids:
        if permno not in returns_df.columns or permno not in vol_df.columns:
            continue

        ret_series = np.asarray(returns_df[permno].values, dtype=np.float32)
        vol_series = np.asarray(vol_df[permno].values, dtype=np.float32)

        count = _count_valid_windows_for_stock(
            ret_series, vol_series, T, stride, max_zero_frac,
        )
        if count > 0:
            window_counts[permno] = count
            total_windows += count

    # Early exit if no valid windows
    if total_windows == 0:
        windows = torch.zeros(0, T, 2, dtype=torch.float32)
        raw_returns = torch.zeros(0, T, dtype=torch.float32)
        metadata = pd.DataFrame(columns=["stock_id", "start_date", "end_date"])
        return windows, metadata, raw_returns

    # =========================================================================
    # PASS 2: Pre-allocate and fill arrays (no list accumulation)
    # =========================================================================
    windows_arr = np.empty((total_windows, T, 2), dtype=np.float32)
    raw_returns_arr = np.empty((total_windows, T), dtype=np.float32)

    # Pre-allocate metadata arrays (instead of list of dicts - ~2.1GB savings)
    # Store indices into dates array, then lookup at the end
    meta_stock_ids = np.empty(total_windows, dtype=np.int64)
    meta_start_idx = np.empty(total_windows, dtype=np.int64)
    meta_end_idx = np.empty(total_windows, dtype=np.int64)

    current_idx = 0

    for permno in stock_ids:
        if permno not in window_counts:
            continue

        ret_series = np.asarray(returns_df[permno].values, dtype=np.float32)
        vol_series = np.asarray(vol_df[permno].values, dtype=np.float32)

        n_dates = len(ret_series)
        if n_dates < T:
            continue

        # Create all windows via stride tricks (no-copy view)
        ret_all = sliding_window_view(ret_series, T)[::stride]
        vol_all = sliding_window_view(vol_series, T)[::stride]
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

        # Boolean indexing already returns a copy - no explicit .copy() needed
        ret_wins = ret_all[nan_ok]
        vol_wins = vol_all[nan_ok]
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
        n_valid = len(win_starts)

        # --- Z-score per feature per window (CONV-02) ---
        ret_mu = ret_clean.mean(axis=1, keepdims=True)
        ret_sigma = ret_clean.std(axis=1, keepdims=True)
        ret_sigma = np.maximum(ret_sigma, sigma_min)
        ret_zscore = (ret_clean - ret_mu) / ret_sigma

        vol_mu = vol_clean.mean(axis=1, keepdims=True)
        vol_sigma = vol_clean.std(axis=1, keepdims=True)
        vol_sigma = np.maximum(vol_sigma, sigma_min)
        vol_zscore = (vol_clean - vol_mu) / vol_sigma

        # Validate z-scored returns are finite
        assert_finite_2d(ret_zscore, "z_scored_returns")

        # Validate z-score normalization (mean≈0, std≈1)
        # Only validate if original data had variance (skip constant data)
        if float(ret_sigma.mean()) > sigma_min * 10:
            assert_z_score_normalized(ret_zscore.flatten(), "z_scored_returns")
        if float(vol_sigma.mean()) > sigma_min * 10:
            assert_z_score_normalized(vol_zscore.flatten(), "z_scored_volatility")

        # Fill pre-allocated arrays (no list accumulation)
        # Note: ret_clean/ret_zscore/vol_zscore already float32 from lines 154-155
        end_idx = current_idx + n_valid
        raw_returns_arr[current_idx:end_idx, :] = ret_clean
        windows_arr[current_idx:end_idx, :, 0] = ret_zscore
        windows_arr[current_idx:end_idx, :, 1] = vol_zscore

        # Fill metadata arrays (instead of list append - ~2.1GB savings)
        meta_stock_ids[current_idx:end_idx] = permno
        meta_start_idx[current_idx:end_idx] = win_starts
        meta_end_idx[current_idx:end_idx] = win_ends

        current_idx = end_idx

    # Convert to tensors (no concatenation needed - arrays already contiguous)
    windows = torch.from_numpy(windows_arr[:current_idx])
    raw_returns = torch.from_numpy(raw_returns_arr[:current_idx])

    # Build metadata DataFrame from pre-allocated arrays (not list of dicts)
    # Lookup dates using stored indices
    metadata = pd.DataFrame({
        "stock_id": meta_stock_ids[:current_idx],
        "start_date": dates[meta_start_idx[:current_idx]],
        "end_date": dates[meta_end_idx[:current_idx]],
    })

    # Final tensor validation
    assert_no_nan_tensor(windows, "windows")
    assert metadata.shape[0] == windows.shape[0], (
        f"metadata rows {metadata.shape[0]} != windows batch {windows.shape[0]}"
    )

    return windows, metadata, raw_returns
