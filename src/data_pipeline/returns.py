"""
Log-return computation with missing value handling and delisting imputation.

CONV-01: Log returns r_t = ln(P^adj_t / P^adj_{t-1}), NEVER arithmetic.

Reference: ISD Section MOD-001 — Sub-task 2.
"""

import numpy as np
import pandas as pd


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

    # Compute log-returns: r_t = ln(P_t / P_{t-1})
    returns_df = np.log(price_filled / price_filled.shift(1))

    # Apply delisting returns
    returns_df = _apply_delisting_returns(
        returns_df, delisting_info, stock_info
    )

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

        returns_df.at[last_date, permno] = log_ret

    return returns_df
