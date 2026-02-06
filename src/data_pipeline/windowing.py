"""
Sliding window creation with per-window z-score normalization.

CONV-02: Z-score per-window, per-feature (mean 0, std 1).
CONV-05: Window shape (batch, T, F) — time is dim 1, features dim 2.

Reference: ISD Section MOD-001 — Sub-task 4.
"""

import numpy as np
import pandas as pd
import torch


def create_windows(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    stock_ids: list[int],
    T: int = 504,
    stride: int = 1,
    sigma_min: float = 1e-8,
    max_zero_frac: float = 0.20,
) -> tuple[torch.Tensor, pd.DataFrame]:
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
    """
    all_windows = []
    metadata_records = []

    dates = returns_df.index

    for permno in stock_ids:
        if permno not in returns_df.columns or permno not in vol_df.columns:
            continue

        ret_series = returns_df[permno].values
        vol_series = vol_df[permno].values

        n_dates = len(ret_series)

        # Slide windows with given stride
        for end_idx in range(T - 1, n_dates, stride):
            start_idx = end_idx - T + 1

            ret_window = ret_series[start_idx: end_idx + 1]
            vol_window = vol_series[start_idx: end_idx + 1]

            # Skip if too many NaN values
            ret_nan_frac = np.isnan(ret_window).sum() / T
            vol_nan_frac = np.isnan(vol_window).sum() / T
            if ret_nan_frac > 0.05 or vol_nan_frac > 0.05:
                continue

            # Fill remaining NaN with 0 (for returns) and forward-fill for vol
            ret_arr = np.asarray(ret_window, dtype=np.float64)
            ret_clean = np.nan_to_num(ret_arr, nan=0.0)
            vol_arr = pd.Series(vol_window).ffill().bfill().to_numpy(dtype=np.float64)
            vol_clean = np.nan_to_num(vol_arr, nan=0.0)

            # Exclude windows with too many zero returns (suspension)
            zero_frac = (np.abs(ret_clean) < 1e-12).sum() / T
            if zero_frac > max_zero_frac:
                continue

            # Z-score per feature (CONV-02)
            ret_mu = ret_clean.mean()
            ret_sigma = ret_clean.std()
            ret_sigma = max(ret_sigma, sigma_min)
            ret_zscore = (ret_clean - ret_mu) / ret_sigma

            vol_mu = vol_clean.mean()
            vol_sigma = vol_clean.std()
            vol_sigma = max(vol_sigma, sigma_min)
            vol_zscore = (vol_clean - vol_mu) / vol_sigma

            # Stack: (T, F=2)
            window = np.stack([ret_zscore, vol_zscore], axis=-1).astype(
                np.float32
            )
            all_windows.append(window)

            metadata_records.append({
                "stock_id": permno,
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
            })

    if not all_windows:
        # Return empty tensors with correct shape
        windows = torch.zeros(0, T, 2, dtype=torch.float32)
        metadata = pd.DataFrame(
            columns=["stock_id", "start_date", "end_date"]
        )
        return windows, metadata

    # Stack all windows: (N, T, F)
    windows = torch.from_numpy(np.stack(all_windows, axis=0))
    metadata = pd.DataFrame(metadata_records)

    return windows, metadata
