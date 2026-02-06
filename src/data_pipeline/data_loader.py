"""
Stock data loading: synthetic CSV generation (Phase A) and file loading.

Phase A: generate_synthetic_csv() produces a CSV with the exact same schema
as production data (EODHD), enabling end-to-end pipeline validation.

Phase B: load_stock_data() loads any CSV/Parquet file with the core schema.

Reference: ISD Section MOD-001 — Sub-task 1.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core schema columns
# ---------------------------------------------------------------------------

CORE_COLUMNS = [
    "permno",
    "date",
    "adj_price",
    "volume",
    "exchange_code",
    "share_code",
    "market_cap",
    "delisting_return",
]

EXTENDED_COLUMNS = ["high", "low", "sector"]

GICS_SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


def generate_synthetic_csv(
    output_path: str,
    n_stocks: int = 200,
    start_date: str = "2000-01-03",
    end_date: str = "2025-12-31",
    n_delistings: int = 20,
    seed: int = 42,
    include_extended: bool = False,
) -> str:
    """
    Generate a synthetic stock data CSV file with realistic properties.

    Price dynamics: geometric Brownian motion per stock.
      P_{t+1} = P_t * exp(μ_i + σ_i * ε_t), ε ~ N(0,1)
      μ_i ~ U(0.0001, 0.0005) (daily drift)
      σ_i ~ U(0.005, 0.03)    (daily vol, annualized ~8%-48%)
      P_0,i ~ U(10, 200)

    :param output_path (str): Path to write the CSV file
    :param n_stocks (int): Number of stocks to generate
    :param start_date (str): Start date (YYYY-MM-DD)
    :param end_date (str): End date (YYYY-MM-DD)
    :param n_delistings (int): Number of stocks that get delisted
    :param seed (int): Random seed for reproducibility
    :param include_extended (bool): Include extended columns (high, low, sector)

    :return output_path (str): Path to the generated CSV file
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    # Per-stock parameters
    mu = rng.uniform(0.0001, 0.0005, size=n_stocks)
    sigma = rng.uniform(0.005, 0.03, size=n_stocks)
    p0 = rng.uniform(10.0, 200.0, size=n_stocks)
    shares_outstanding = rng.uniform(10e6, 500e6, size=n_stocks)

    # Exchange codes: 60% NYSE (1), 10% AMEX (2), 30% NASDAQ (3)
    exchange_codes = rng.choice([1, 2, 3], size=n_stocks, p=[0.60, 0.10, 0.30])
    share_codes = rng.choice([10, 11], size=n_stocks)

    # GBM price simulation
    eps = rng.randn(n_days, n_stocks)
    log_returns = mu[np.newaxis, :] + sigma[np.newaxis, :] * eps
    log_prices = np.log(p0)[np.newaxis, :] + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    # Volume: LogNormal correlated with shares outstanding
    mu_vol = np.log(shares_outstanding * 0.005)
    volumes = np.zeros((n_days, n_stocks), dtype=np.int64)
    for i in range(n_stocks):
        volumes[:, i] = rng.lognormal(mu_vol[i], 1.0, size=n_days).astype(np.int64)

    # Market cap
    market_caps = prices * shares_outstanding[np.newaxis, :]

    # Delisting returns: NaN by default
    delisting_returns = np.full((n_days, n_stocks), np.nan)

    # Delistings in the second half of history
    delisted_after = {}
    if n_delistings > 0 and n_delistings <= n_stocks:
        delist_stocks = rng.choice(n_stocks, size=n_delistings, replace=False)
        half_point = n_days // 2
        for stock_idx in delist_stocks:
            delist_day = rng.randint(half_point, n_days - 1)
            delisted_after[stock_idx] = delist_day

            # ~50% have explicit delisting return, rest NaN (for Shumway imputation)
            if rng.random() < 0.5:
                if exchange_codes[stock_idx] in (1, 2):
                    delisting_returns[delist_day, stock_idx] = -0.30
                else:
                    delisting_returns[delist_day, stock_idx] = -0.55

            # NaN out prices after delisting
            prices[delist_day + 1:, stock_idx] = np.nan
            market_caps[delist_day + 1:, stock_idx] = np.nan
            volumes[delist_day + 1:, stock_idx] = 0

    # ~2% random missing data (NaN adj_price)
    missing_mask = rng.random((n_days, n_stocks)) < 0.02
    missing_mask[0, :] = False  # Keep first day intact
    prices[missing_mask] = np.nan

    # Extended columns
    if include_extended:
        # High/Low: based on price with intraday range
        intraday_range = rng.uniform(0.005, 0.03, size=(n_days, n_stocks))
        highs = prices * (1 + intraday_range / 2)
        lows = prices * (1 - intraday_range / 2)
        sectors = rng.choice(GICS_SECTORS, size=n_stocks)

    # Build long-format DataFrame
    records = []
    for i in range(n_stocks):
        permno = 10001 + i
        for t in range(n_days):
            # Skip post-delisting rows
            if i in delisted_after and t > delisted_after[i]:
                continue

            record = {
                "permno": permno,
                "date": dates[t].strftime("%Y-%m-%d"),
                "adj_price": prices[t, i],
                "volume": volumes[t, i],
                "exchange_code": exchange_codes[i],
                "share_code": share_codes[i],
                "market_cap": market_caps[t, i],
                "delisting_return": delisting_returns[t, i],
            }

            if include_extended:
                record["high"] = highs[t, i]
                record["low"] = lows[t, i]
                record["sector"] = sectors[i]

            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def load_stock_data(
    data_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load daily stock data from CSV or Parquet file.

    Accepts both synthetic data (Phase A) and EODHD data (Phase B) —
    same schema. Must include delisted stocks with full pre-delisting history.

    :param data_path (str): Path to CSV or Parquet file
    :param start_date (str | None): Start date filter (YYYY-MM-DD), inclusive
    :param end_date (str | None): End date filter (YYYY-MM-DD), inclusive

    :return df (pd.DataFrame): DataFrame with core columns sorted by (permno, date).
        date column is pd.Timestamp.
    """
    if data_path.endswith(".parquet"):
        df: pd.DataFrame = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Ensure date is Timestamp
    df["date"] = pd.to_datetime(df["date"])

    # Date filtering
    if start_date is not None:
        df = pd.DataFrame(df[df["date"] >= pd.Timestamp(start_date)])
    if end_date is not None:
        df = pd.DataFrame(df[df["date"] <= pd.Timestamp(end_date)])

    # Validate core columns are present
    missing = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by (permno, date)
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Ensure correct types
    df["permno"] = df["permno"].astype(int)
    df["adj_price"] = df["adj_price"].astype(float)
    vol_numeric = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = pd.Series(vol_numeric).fillna(0).astype(int)
    df["exchange_code"] = df["exchange_code"].astype(int)
    df["share_code"] = df["share_code"].astype(int)
    df["market_cap"] = df["market_cap"].astype(float)
    df["delisting_return"] = df["delisting_return"].astype(float)

    return df
