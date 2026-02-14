"""
S&P 500 index benchmark (price return).

Downloads ^GSPC daily prices from Yahoo Finance via yfinance.
The index is point-in-time by construction: each daily close reflects
the exact composition and cap-weighting of the index on that trading day.

This benchmark does NOT use the equity universe weights — it evaluates
portfolio metrics directly from the S&P 500 index returns.  Constraints
(w_max, turnover, cardinality) do NOT apply.

Reference: passive market benchmark for context comparison only.
"""

import logging
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
        Download S&P 500 data covering the full date range of *returns*.

        The training returns and universe parameters are ignored — the S&P
        500 index has its own return series.  A module-level cache prevents
        redundant downloads across walk-forward folds.

        :param returns (pd.DataFrame): Historical log-returns (used only for date range)
        :param universe (list[str]): Equity universe (unused)
        """
        global _SP500_CACHE
        self.n = len(universe)

        if _SP500_CACHE is not None:
            self._log_returns = _SP500_CACHE
            return

        if not _HAS_YFINANCE:
            logger.warning(
                "yfinance not installed — SP500 benchmark returns NaN metrics"
            )
            return

        try:
            start = str(returns.index[0])[:10]
            data: pd.DataFrame | None = yf.download(  # type: ignore[assignment]
                "^GSPC",
                start=start,
                end="2030-01-01",
                progress=False,
                auto_adjust=True,
            )
            if data is None or data.empty:
                logger.warning("Empty SP500 data from yfinance")
                return

            close: pd.Series = data["Close"].squeeze()  # type: ignore[assignment]
            log_ret: pd.Series = np.log(close / close.shift(1)).dropna()

            # Normalise to timezone-naive timestamps (match our returns index)
            log_ret.index = pd.to_datetime(log_ret.index).tz_localize(None)

            self._log_returns = log_ret
            _SP500_CACHE = log_ret
            logger.info(
                "SP500 data downloaded: %d days (%s to %s)",
                len(log_ret),
                str(log_ret.index[0])[:10],
                str(log_ret.index[-1])[:10],
            )
        except Exception as e:
            logger.warning("SP500 download failed: %s", e)

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

        ann_return = float(np.mean(sp500_ret) * 252)
        ann_vol = float(np.std(sp500_ret, ddof=1) * np.sqrt(252))
        sharpe = ann_return / max(ann_vol, 1e-10)

        cum_returns = np.cumsum(sp500_ret)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - running_max
        max_drawdown = float(-np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
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
