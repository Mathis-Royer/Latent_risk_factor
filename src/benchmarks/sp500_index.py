"""
S&P 500 index benchmark (price return).

Downloads ^GSPC daily prices from Yahoo Finance via yfinance with
disk persistence. The index is point-in-time by construction: each
daily close reflects the exact composition and cap-weighting on that day.

This benchmark does NOT use the equity universe weights — it evaluates
portfolio metrics directly from the S&P 500 index returns.  Constraints
(w_max, turnover, cardinality) do NOT apply.

Reference: passive market benchmark for context comparison only.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmarks.base import BenchmarkModel

logger = logging.getLogger(__name__)

try:
    import yfinance as yf  # type: ignore[import-untyped]
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False
    logger.info("yfinance not installed — SP500 benchmark unavailable")


# Module-level cache to avoid redundant downloads across folds/runs
_SP500_CACHE: pd.Series | None = None

# Persistence configuration
SP500_DATA_FILENAME = "sp500_index.parquet"
SP500_CACHE_MAX_AGE_DAYS = 1


# ---------------------------------------------------------------------------
# SP500 Data Persistence Functions
# ---------------------------------------------------------------------------

def download_sp500_data(
    data_dir: str = "data/",
    start_date: str = "1995-01-01",
    end_date: str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Download SP500 index data from Yahoo Finance and save to disk.
    Called automatically if cache is missing/stale.

    :param data_dir (str): Output directory
    :param start_date (str): Start date (YYYY-MM-DD)
    :param end_date (str | None): End date, defaults to today
    :param force (bool): Re-download even if cache is fresh

    :return df (pd.DataFrame | None): SP500 data with columns [date, close, log_return]
    """
    if not _HAS_YFINANCE:
        logger.warning("yfinance not installed — cannot download SP500 data")
        return None

    # Ensure output directory exists
    out_path = Path(data_dir) / SP500_DATA_FILENAME
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Check if fresh cache exists
    if not force and out_path.exists():
        mtime = datetime.fromtimestamp(out_path.stat().st_mtime)
        age = datetime.now() - mtime
        if age < timedelta(days=SP500_CACHE_MAX_AGE_DAYS):
            logger.debug("SP500 cache is fresh (age=%s)", age)
            return load_sp500_data(data_dir)

    # Download with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info("Downloading SP500 data (attempt %d/%d)...", attempt + 1, max_retries)
            data: pd.DataFrame | None = yf.download(  # type: ignore[assignment]
                "^GSPC",
                start=start_date,
                end=end_date or "2030-01-01",
                progress=False,
                auto_adjust=True,
            )
            if data is not None and not data.empty:
                break
        except Exception as e:
            logger.warning("SP500 download attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        data = None

    if data is None or data.empty:
        logger.error("SP500 download failed after %d attempts", max_retries)
        return None

    # Process data
    close: pd.Series = data["Close"].squeeze()  # type: ignore[assignment]
    log_return: pd.Series = np.log(close / close.shift(1))

    # Build DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(close.index).tz_localize(None),
        "close": close.values,
        "log_return": log_return.values,
    })
    df = df.dropna(subset=["log_return"])

    # Save to disk
    df.to_parquet(out_path, index=False)
    logger.info("SP500 data saved: %d days to %s", len(df), out_path)

    return df


def load_sp500_data(data_dir: str = "data/") -> pd.DataFrame | None:
    """
    Load cached SP500 data from disk.

    :param data_dir (str): Directory containing sp500_index.parquet

    :return df (pd.DataFrame | None): SP500 data or None if not found
    """
    file_path = Path(data_dir) / SP500_DATA_FILENAME
    if not file_path.exists():
        return None
    try:
        df = pd.read_parquet(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning("Failed to load SP500 cache: %s", e)
        return None


def get_sp500_data(
    data_dir: str = "data/",
    start_date: str = "1995-01-01",
) -> pd.DataFrame | None:
    """
    Load SP500 data from disk, automatically downloading if missing or stale.
    This is the main entry point — no manual download step needed.

    :param data_dir (str): Data directory
    :param start_date (str): Start date for download if needed

    :return df (pd.DataFrame | None): SP500 data with [date, close, log_return]
    """
    file_path = Path(data_dir) / SP500_DATA_FILENAME

    # Check if cache exists and is fresh
    if file_path.exists():
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - mtime
        if age < timedelta(days=SP500_CACHE_MAX_AGE_DAYS):
            df = load_sp500_data(data_dir)
            if df is not None:
                return df

    # Auto-download if missing or stale
    logger.info("SP500 cache missing or stale — downloading...")
    return download_sp500_data(data_dir=data_dir, start_date=start_date)


class SP500TotalReturn(BenchmarkModel):
    """
    S&P 500 price-return benchmark.

    Uses ^GSPC daily prices from Yahoo Finance.  Point-in-time by nature:
    each day's close reflects the actual composition and cap-weighting of
    the index on that date.

    ``evaluate()`` computes OOS metrics from S&P 500 returns directly,
    ignoring the portfolio weights and equity universe passed to it.
    """

    def __init__(self, constraint_params: dict[str, float] | None = None) -> None:
        super().__init__(constraint_params)
        self._log_returns: pd.Series | None = None

    # ------------------------------------------------------------------
    # BenchmarkModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: Any,
    ) -> None:
        """
        Load S&P 500 data from disk cache or auto-download if missing.

        The training returns and universe parameters are ignored — the S&P
        500 index has its own return series.  A module-level cache prevents
        redundant loads across walk-forward folds.

        :param returns (pd.DataFrame): Historical log-returns (used only for date range)
        :param universe (list[str]): Equity universe (unused)
        :param data_dir (str): Data directory for cache (via kwargs)
        """
        global _SP500_CACHE
        self.n = len(universe)

        # Use memory cache if available
        if _SP500_CACHE is not None:
            self._log_returns = _SP500_CACHE
            return

        # Extract parameters
        data_dir = str(kwargs.get("data_dir", "data/"))
        start_date = str(returns.index[0])[:10] if len(returns) > 0 else "1995-01-01"

        # Auto-load from disk or download if missing
        df = get_sp500_data(data_dir=data_dir, start_date=start_date)
        if df is not None and "log_return" in df.columns:
            log_ret: pd.Series = pd.Series(
                df["log_return"].values,
                index=pd.to_datetime(df["date"]),
                name="log_return",
            )
            self._log_returns = log_ret
            _SP500_CACHE = log_ret
            logger.info("SP500 loaded: %d days", len(log_ret))
        else:
            logger.warning("SP500 data unavailable")

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        Return placeholder equal weights (unused in evaluate).

        :param w_old (np.ndarray | None): Previous weights (ignored)
        :param is_first (bool): First rebalancing flag (ignored)

        :return w (np.ndarray): Equal weights (n,)
        """
        if self.n > 0:
            return np.ones(self.n) / self.n
        return np.array([])

    def evaluate(
        self,
        w: np.ndarray,
        returns_oos: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, float]:
        """
        Compute OOS metrics from the S&P 500 index returns.

        Parameters *w* and *universe* are ignored; the method aligns the
        cached ^GSPC daily log-returns with the OOS date range extracted
        from *returns_oos*.

        :param w (np.ndarray): Portfolio weights (ignored)
        :param returns_oos (pd.DataFrame): OOS returns (used for date alignment)
        :param universe (list[str]): Equity universe (ignored)

        :return metrics (dict): OOS portfolio metrics
        """
        nan_result = {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "eff_n_positions": float("nan"),
            "n_days_oos": len(returns_oos),
        }

        if self._log_returns is None:
            return nan_result

        oos_dates = returns_oos.index
        sp500_ret: np.ndarray = np.asarray(
            self._log_returns.reindex(oos_dates).fillna(0.0).values,
            dtype=np.float64,
        )
        n_days = len(sp500_ret)

        if n_days < 2:
            return nan_result

        # Geometric annualized return — exp(sum(log_r)) for log returns
        cumulative = float(np.exp(np.sum(sp500_ret)))
        ann_return = cumulative ** (252.0 / n_days) - 1.0
        ann_vol = float(np.std(sp500_ret, ddof=1) * np.sqrt(252))
        sharpe = ann_return / max(ann_vol, 1e-10)

        # Maximum drawdown (percentage, not log-space)
        cum_returns = np.cumsum(sp500_ret)
        running_max = np.maximum.accumulate(cum_returns)
        log_drawdowns = cum_returns - running_max
        max_drawdown = float(1.0 - np.exp(np.min(log_drawdowns))) if len(log_drawdowns) > 0 else 0.0
        calmar = ann_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0.0

        return {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "eff_n_positions": 500.0,
            "n_days_oos": n_days,
        }

    # ------------------------------------------------------------------
    # Extended interface
    # ------------------------------------------------------------------

    def get_oos_returns(self, returns_oos: pd.DataFrame) -> np.ndarray | None:
        """
        Return daily S&P 500 log-returns aligned to OOS dates.

        :param returns_oos (pd.DataFrame): OOS returns (used for date index)

        :return daily_returns (np.ndarray | None): S&P 500 daily log-returns
        """
        if self._log_returns is None:
            return None
        return np.asarray(
            self._log_returns.reindex(returns_oos.index).fillna(0.0).values,
            dtype=np.float64,
        )

    def rebalance(
        self,
        returns_trailing: pd.DataFrame,
        trailing_vol: pd.DataFrame | None,
        w_old: np.ndarray,
        universe: list[str],
        current_date: str,
    ) -> np.ndarray:
        """
        No-op rebalancing for index benchmark (returns same placeholder weights).

        :return w (np.ndarray): Placeholder equal weights (unused)
        """
        self.n = len(universe)
        return self.optimize()
