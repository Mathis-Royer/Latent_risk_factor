"""
Trailing realized volatility computation for the data pipeline.

Two volatility measures:
1. Trailing 252-day annualized volatility (for risk model rescaling)
2. Rolling 21-day realized volatility (second VAE input feature, F=2)

Reference: ISD Section MOD-001 — Sub-task 6.
"""

import numpy as np
import pandas as pd

# Annualization factor for daily returns
TRADING_DAYS_PER_YEAR = 252


def compute_trailing_volatility(
    returns_df: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute trailing annualized volatility for each stock.

    σ_{i,t} = std(r_{i, t-window+1:t}) * sqrt(252)

    Warm-up: the first `window` days of each stock have NaN vol.

    :param returns_df (pd.DataFrame): Log-returns (n_dates × n_stocks)
    :param window (int): Trailing window in trading days

    :return vol_df (pd.DataFrame): Annualized trailing vol, same shape as
        returns_df. First `window` rows per stock are NaN.
    """
    vol_df = returns_df.rolling(window=window, min_periods=window).std()
    vol_df = vol_df * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol_df


def compute_rolling_realized_vol(
    returns_df: pd.DataFrame,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """
    Compute 21-day rolling realized volatility (not annualized).

    v_{i,τ} = std(r_{i, τ-rolling_window+1:τ})

    Used as the second input feature (F=2) for the VAE encoder.

    Warm-up requirement: needs `rolling_window` pre-window returns.
    The first `rolling_window` rows per stock are NaN.

    :param returns_df (pd.DataFrame): Log-returns (n_dates × n_stocks)
    :param rolling_window (int): Rolling window in trading days

    :return vol_df (pd.DataFrame): Rolling realized vol, same shape as
        returns_df. First `rolling_window` rows are NaN.
    """
    result = returns_df.rolling(
        window=rolling_window, min_periods=rolling_window
    ).std()
    return pd.DataFrame(result)
