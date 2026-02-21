"""
Log-return computation with missing value handling and delisting imputation.

CONV-01: Log returns r_t = ln(P^adj_t / P^adj_{t-1}), NEVER arithmetic.

Reference: ISD Section MOD-001 — Sub-task 2.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import mstats

from src.validation import (
    assert_column_exists,
    assert_finite_2d,
    assert_log_input_positive,
    assert_returns_valid,
    warn_if_nan_fraction_exceeds,
    warn_if_price_discontinuity,
)

logger = logging.getLogger(__name__)


# Shumway (1997) delisting return imputation
SHUMWAY_NYSE_AMEX = np.log(1.0 + (-0.30))  # -30% → log return
SHUMWAY_NASDAQ = np.log(1.0 + (-0.55))      # -55% → log return


def compute_log_returns(
    stock_data: pd.DataFrame,
    max_gap_fill: int = 5,
) -> pd.DataFrame:
    """
    Compute log-returns from adjusted prices.

    r_{i,t} = ln(P^adj_{i,t} / P^adj_{i,t-1})

    CONV-01: Log returns, NEVER arithmetic.

    Missing value handling:
    - Isolated gaps ≤ max_gap_fill consecutive days: forward-fill price (= 0 return)
    - Gaps > max_gap_fill consecutive days: NaN (excluded from windowing)

    Delisting returns:
    - If delisting_return is available, use it (converted to log-return)
    - If NaN: impute per Shumway (1997): -30% NYSE/AMEX, -55% NASDAQ

    :param stock_data (pd.DataFrame): Must contain columns: permno, date,
        adj_price, exchange_code, delisting_return. Sorted by (permno, date).
    :param max_gap_fill (int): Maximum consecutive NaN days to forward-fill

    :return returns_df (pd.DataFrame): Log-returns indexed by date,
        columns are permno (int). Shape (n_dates, n_stocks).
    """
    # Validate required columns
    assert_column_exists(stock_data, "adj_price", "stock_data")
    assert_column_exists(stock_data, "permno", "stock_data")
    assert_column_exists(stock_data, "date", "stock_data")
    assert_column_exists(stock_data, "exchange_code", "stock_data")
    assert_column_exists(stock_data, "delisting_return", "stock_data")

    # Pivot to wide format: rows=dates, cols=permnos, values=adj_price
    price_wide = stock_data.pivot_table(
        index="date", columns="permno", values="adj_price", aggfunc="first"
    )
    price_wide = price_wide.sort_index()

    # Build exchange and delisting lookup from stock_data
    stock_info = stock_data.groupby("permno").agg(
        exchange_code=("exchange_code", "first"),
    ).to_dict()["exchange_code"]

    # Get last valid date and delisting return per stock
    delisting_info = _extract_delisting_info(stock_data)

    # Forward-fill small gaps (≤ max_gap_fill consecutive NaNs)
    price_filled = _forward_fill_small_gaps(price_wide, max_gap_fill)

    # Validate prices are positive before log (guards against log(0) or log(negative))
    assert_log_input_positive(price_filled, "prices_before_log")

    # Compute log-returns: r_t = ln(P_t / P_{t-1})
    returns_df = np.log(price_filled / price_filled.shift(1))

    # Apply delisting returns
    returns_df = _apply_delisting_returns(
        returns_df, delisting_info, stock_info
    )

    # Winsorize extreme returns caused by Tiingo API data corruption
    # (inverted split adjustments can create 1000x+ price changes)
    # Clip to |r| <= 2.0 (≈638% move) which is extreme but plausible
    MAX_ABS_RETURN = 2.0
    extreme_mask = np.abs(returns_df.values) > MAX_ABS_RETURN
    n_extreme = int(np.sum(extreme_mask & np.isfinite(returns_df.values)))
    if n_extreme > 0:
        logger.warning(
            "Winsorizing %d extreme returns (|r| > %.1f) caused by data errors",
            n_extreme, MAX_ABS_RETURN,
        )
        returns_df = returns_df.clip(lower=-MAX_ABS_RETURN, upper=MAX_ABS_RETURN)

    # Warn if excessive NaN fraction in returns
    warn_if_nan_fraction_exceeds(returns_df, 0.3, "returns")

    # Validate returns are bounded (max 200% single-day return is suspicious)
    assert_returns_valid(returns_df, "log_returns", max_abs_return=2.0)

    # Warn about potential data errors (price discontinuities)
    warn_if_price_discontinuity(returns_df, 0.5, "log_returns")

    return returns_df


def _forward_fill_small_gaps(
    price_wide: pd.DataFrame,
    max_gap: int,
) -> pd.DataFrame:
    """
    Forward-fill NaN prices for gaps ≤ max_gap consecutive days.
    Gaps > max_gap remain NaN.

    :param price_wide (pd.DataFrame): Wide-format prices (dates × permnos)
    :param max_gap (int): Maximum consecutive NaN days to fill

    :return filled (pd.DataFrame): Prices with small gaps forward-filled
    """
    filled = price_wide.copy()

    for col in filled.columns:
        series: pd.Series = filled[col]  # type: ignore[assignment]
        is_nan: pd.Series = series.isna()  # type: ignore[assignment]

        if not bool(is_nan.any()):
            continue

        # Identify consecutive NaN runs
        groups: pd.Series = is_nan.ne(is_nan.shift()).cumsum()  # type: ignore[assignment]
        nan_groups: pd.Series = groups[is_nan]  # type: ignore[assignment]

        if nan_groups.empty:
            continue

        group_sizes: pd.Series = nan_groups.value_counts()  # type: ignore[assignment]
        small_gap_idx = group_sizes[group_sizes <= max_gap].index  # type: ignore[union-attr]

        # Build a mask of only small gaps to fill
        fill_mask: pd.Series = nan_groups.isin(small_gap_idx)  # type: ignore[assignment]
        fillable_idx = fill_mask[fill_mask].index  # type: ignore[union-attr]

        # Forward-fill only at these positions
        ffilled = series.ffill()
        filled.loc[fillable_idx, col] = ffilled.loc[fillable_idx]

    return filled


def _extract_delisting_info(stock_data: pd.DataFrame) -> dict:
    """
    Extract per-stock delisting information.

    :param stock_data (pd.DataFrame): Raw stock data

    :return delisting_info (dict): permno → {last_date, delisting_return}
    """
    info = {}
    for permno, group in stock_data.groupby("permno"):
        last_row = group.iloc[-1]
        delist_ret = last_row["delisting_return"]

        if pd.notna(delist_ret):
            info[permno] = {
                "last_date": last_row["date"],
                "delisting_return": delist_ret,
            }
        else:
            # Check if there's any non-NaN delisting return in the stock's history
            delist_rows = group[group["delisting_return"].notna()]
            if not delist_rows.empty:
                last_delist = delist_rows.iloc[-1]
                info[permno] = {
                    "last_date": last_delist["date"],
                    "delisting_return": last_delist["delisting_return"],
                }

    return info


def _apply_delisting_returns(
    returns_df: pd.DataFrame,
    delisting_info: dict,
    exchange_codes: dict,
) -> pd.DataFrame:
    """
    Apply delisting returns at the last trading date.

    If delisting_return is provided, convert to log-return and apply.
    If NaN, impute via Shumway (1997): -30% NYSE/AMEX, -55% NASDAQ.

    :param returns_df (pd.DataFrame): Log-returns (dates × permnos)
    :param delisting_info (dict): permno → {last_date, delisting_return}
    :param exchange_codes (dict): permno → exchange_code

    :return returns_df (pd.DataFrame): With delisting returns applied
    """
    returns_df = returns_df.copy()

    for permno, info in delisting_info.items():
        if permno not in returns_df.columns:
            continue

        last_date = info["last_date"]
        delist_ret = info["delisting_return"]

        if last_date not in returns_df.index:
            continue

        if np.isnan(delist_ret):
            # Impute Shumway convention
            exc = exchange_codes.get(permno, 1)
            if exc in (1, 2):  # NYSE, AMEX
                log_ret = SHUMWAY_NYSE_AMEX
            else:  # NASDAQ
                log_ret = SHUMWAY_NASDAQ
        else:
            # Convert arithmetic return to log-return
            log_ret = np.log(1.0 + delist_ret)

        # Validate delisting log-return is within reasonable bounds
        assert -2.0 <= log_ret <= 0.5, (
            f"Delisting log-return {log_ret:.4f} out of bounds [-2.0, 0.5] "
            f"for permno {permno}"
        )
        returns_df.at[last_date, permno] = log_ret

    return returns_df


def winsorize_returns(
    returns_df: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize returns at specified percentiles (cross-sectional per date).

    Optional preprocessing step to reduce impact of extreme outliers.
    Applied per-date to avoid look-ahead bias and maintain cross-sectional
    comparability.

    :param returns_df (pd.DataFrame): Log-returns (dates × stocks)
    :param lower_pct (float): Lower percentile (default: 1%)
    :param upper_pct (float): Upper percentile (default: 99%)

    :return winsorized_df (pd.DataFrame): Winsorized returns
    """
    def winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 10:
            # Too few observations for reliable percentile estimation
            return row
        limits = [lower_pct, 1 - upper_pct]
        winsorized = mstats.winsorize(valid.values, limits=limits)
        result = row.copy()
        result.loc[valid.index] = winsorized
        return result

    result: pd.DataFrame = returns_df.apply(winsorize_row, axis=1)  # type: ignore[assignment]
    return result


def warn_price_discontinuities(
    returns_df: pd.DataFrame,
    threshold: float = 0.15,
) -> int:
    """
    Log warning for suspicious single-day returns exceeding threshold.

    Helps detect data quality issues like stock splits not adjusted,
    corporate actions, or data errors.

    :param returns_df (pd.DataFrame): Log-returns (dates × stocks)
    :param threshold (float): Threshold for extreme returns (default: 15%)

    :return n_extreme (int): Number of extreme return observations
    """
    log_threshold = np.log(1 + threshold)
    extreme_mask = np.abs(returns_df) > log_threshold
    n_extreme = int(extreme_mask.sum().sum())

    if n_extreme > 0:
        # Calculate what percentage of observations are extreme
        total_obs = int(returns_df.notna().sum().sum())
        pct_extreme = 100 * n_extreme / max(total_obs, 1)
        logger.warning(
            "Found %d single-day returns >%.0f%% (%.2f%% of observations). "
            "Check data quality for stock splits, corporate actions, or errors.",
            n_extreme,
            threshold * 100,
            pct_extreme,
        )

    return n_extreme
