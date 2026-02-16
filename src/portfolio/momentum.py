"""
Cross-sectional momentum signal (12-1 month).

Computes cumulative log-return over [t-lookback, t-skip] for each stock,
z-scores cross-sectionally, and returns the momentum vector mu.

Reference: Jegadeesh & Titman (1993), "Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency".
"""

import logging
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_momentum_signal(
    returns: pd.DataFrame,
    universe: Sequence[Hashable],
    lookback: int = 252,
    skip: int = 21,
) -> np.ndarray:
    """
    Cross-sectional momentum signal (12-1 month).

    For each stock, compute the cumulative log-return over [t-lookback, t-skip]
    (skipping the most recent month to avoid short-term reversal), then
    z-score cross-sectionally.

    :param returns (pd.DataFrame): Daily log-returns, columns = tickers/permnos
    :param universe (Sequence[Hashable]): Active stock identifiers (defines output order)
    :param lookback (int): Total lookback window in trading days (default 252)
    :param skip (int): Days to skip at the end (default 21, ~1 month)

    :return mu (np.ndarray): Z-scored momentum signal (n,), aligned to universe
    """
    n = len(universe)
    mu = np.zeros(n, dtype=np.float64)

    available = [s for s in universe if s in returns.columns]
    if not available:
        logger.warning("compute_momentum_signal: no stocks found in returns columns.")
        return mu

    T = len(returns)
    if T < lookback:
        logger.warning(
            "compute_momentum_signal: insufficient history (%d < %d). "
            "Returning mu=0.", T, lookback,
        )
        return mu

    # Cumulative log-return over [t-lookback, t-skip]
    # Using the last `lookback` rows, then excluding the last `skip` rows
    ret_window = returns[available].iloc[-(lookback):-(skip)] if skip > 0 else returns[available].iloc[-(lookback):]

    # Cumulative log-return = sum of daily log-returns
    cum_ret = ret_window.sum(axis=0)  # (n_available,)
    cum_ret_arr = np.array(cum_ret, dtype=np.float64, copy=True)

    # Handle NaN: stocks with missing data get 0 momentum
    nan_mask = np.isnan(cum_ret_arr)
    cum_ret_arr[nan_mask] = 0.0

    # Z-score cross-sectionally
    valid_mask = ~nan_mask
    n_valid = int(np.sum(valid_mask))

    if n_valid < 2:
        logger.warning(
            "compute_momentum_signal: fewer than 2 valid stocks. "
            "Returning mu=0.",
        )
        return mu

    mean_val = float(np.mean(cum_ret_arr[valid_mask]))
    std_val = float(np.std(cum_ret_arr[valid_mask], ddof=1))

    if std_val < 1e-12:
        logger.warning(
            "compute_momentum_signal: zero cross-sectional std. "
            "Returning mu=0.",
        )
        return mu

    z_scored = np.where(valid_mask, (cum_ret_arr - mean_val) / std_val, 0.0)

    # Map back to universe order
    avail_to_idx = {s: i for i, s in enumerate(available)}
    for j, ticker in enumerate(universe):
        if ticker in avail_to_idx:
            mu[j] = z_scored[avail_to_idx[ticker]]

    n_nonzero = int(np.sum(np.abs(mu) > 1e-10))
    logger.info(
        "compute_momentum_signal: lookback=%d, skip=%d, "
        "n_valid=%d/%d, z-score range=[%.2f, %.2f]",
        lookback, skip, n_nonzero, n,
        float(np.min(mu)), float(np.max(mu)),
    )

    return mu
