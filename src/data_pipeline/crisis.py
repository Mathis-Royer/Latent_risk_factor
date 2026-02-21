"""
Crisis labeling via VIX threshold on expanding training window.

CONV-10 / INV-005: No look-ahead. VIX threshold computed on expanding
window of the training period only.

Data source: FRED (Federal Reserve Economic Data) via fredapi.
Series: VIXCLS (1990+), VXOCLS (1986-1990 proxy).

Reference: ISD Section MOD-001 — Sub-task 5.
"""

import numpy as np
import pandas as pd
import torch

from src.validation import assert_bounds, assert_crisis_fraction_bounds


def load_vix_data(
    start_date: str = "1990-01-01",
    end_date: str = "2025-12-31",
    fred_api_key: str | None = None,
) -> pd.Series:
    """
    Load VIX daily close data from FRED.

    Primary: FRED series VIXCLS via fredapi.
    Fallback: if fredapi is unavailable, raises ImportError with instructions.

    :param start_date (str): Start date (YYYY-MM-DD)
    :param end_date (str): End date (YYYY-MM-DD)
    :param fred_api_key (str | None): FRED API key. If None, reads from
        environment variable FRED_API_KEY.

    :return vix (pd.Series): Daily VIX close, indexed by date
    """
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError(
            "fredapi is required for VIX data. Install with: pip install fredapi. "
            "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
        ) from e

    import os
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY environment variable or pass fred_api_key."
        )

    fred = Fred(api_key=api_key)
    vix = fred.get_series(
        "VIXCLS",
        observation_start=start_date,
        observation_end=end_date,
    )
    vix = vix.dropna()
    vix.name = "VIX"
    return vix


def generate_synthetic_vix(
    start_date: str = "1990-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.Series:
    """
    Generate synthetic VIX data for development/testing.

    Produces realistic VIX dynamics: mean-reverting with occasional spikes.
    CIR-like process: dV = κ(θ - V)dt + σ√V dW

    :param start_date (str): Start date
    :param end_date (str): End date
    :param seed (int): Random seed

    :return vix (pd.Series): Synthetic daily VIX, indexed by business day dates
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    # CIR parameters (calibrated to historical VIX behavior)
    kappa = 0.05   # mean reversion speed
    theta = 20.0   # long-run mean
    sigma = 3.0    # volatility of vol
    dt = 1.0 / 252

    vix_values = np.zeros(n_days)
    vix_values[0] = theta

    for t in range(1, n_days):
        v = max(vix_values[t - 1], 1.0)
        dv = kappa * (theta - v) * dt + sigma * np.sqrt(v * dt) * rng.randn()
        vix_values[t] = max(v + dv, 5.0)  # Floor at 5

        # Occasional spike regime (2% chance per day)
        if rng.random() < 0.002:
            vix_values[t] = vix_values[t] * rng.uniform(1.5, 3.0)

    vix = pd.Series(vix_values, index=dates, name="VIX")

    # Note: Synthetic VIX can have extreme spikes (up to 3x × multiple days)
    # for stress-testing purposes. Production VIX data should be validated
    # separately with realistic bounds like [5, 100].

    return vix


def compute_crisis_threshold(
    vix_data: pd.Series,
    training_end_date: pd.Timestamp,
    percentile: float = 80.0,
    history_start: pd.Timestamp | None = None,
) -> float:
    """
    Compute the VIX crisis threshold on the expanding training window.

    τ_VIX = Percentile_P(VIX_{t0:t_train})

    CONV-10 / INV-005: Only uses data up to training_end_date.

    :param vix_data (pd.Series): Daily VIX values
    :param training_end_date (pd.Timestamp): End of the training period
    :param percentile (float): Percentile for threshold (default 80.0)
    :param history_start (pd.Timestamp | None): Start of expanding window.
        If None, uses first available VIX date.

    :return threshold (float): VIX crisis threshold
    """
    if history_start is not None:
        mask = (vix_data.index >= history_start) & (
            vix_data.index <= training_end_date
        )
        vix_training: pd.Series = vix_data.loc[mask]  # type: ignore[assignment]
    else:
        vix_training = vix_data.loc[  # type: ignore[assignment]
            vix_data.index <= training_end_date
        ]

    if len(vix_training) == 0:
        raise ValueError(
            f"No VIX data available up to {training_end_date}"
        )

    assert 0 < percentile < 100, (
        f"percentile must be in (0, 100), got {percentile}"
    )

    threshold = float(np.percentile(np.asarray(vix_training), percentile))
    return threshold


def compute_crisis_labels(
    vix_data: pd.Series,
    window_metadata: pd.DataFrame,
    training_end_date: pd.Timestamp,
    percentile: float = 80.0,
) -> torch.Tensor:
    """
    Compute crisis fraction for each window.

    For each window w: f_c(w) = fraction of days where VIX > τ_VIX.

    CONV-10 / INV-005: VIX threshold uses only data up to training_end_date.

    :param vix_data (pd.Series): Daily VIX values, indexed by date
    :param window_metadata (pd.DataFrame): Columns: stock_id, start_date, end_date
    :param training_end_date (pd.Timestamp): End of training period
    :param percentile (float): VIX percentile for crisis threshold

    :return crisis_fractions (torch.Tensor): Shape (N,), values in [0, 1]
    """
    threshold = compute_crisis_threshold(
        vix_data, training_end_date, percentile
    )

    n_windows = len(window_metadata)
    fractions = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        start = window_metadata.iloc[i]["start_date"]
        end = window_metadata.iloc[i]["end_date"]

        window_vix = vix_data[
            (vix_data.index >= start) & (vix_data.index <= end)
        ]

        if len(window_vix) == 0:
            fractions[i] = 0.0
        else:
            fractions[i] = (window_vix > threshold).mean()

    crisis_fractions = torch.from_numpy(fractions)
    assert crisis_fractions.shape[0] == len(window_metadata), (
        f"crisis_fractions length {crisis_fractions.shape[0]} != "
        f"metadata length {len(window_metadata)}"
    )

    # Validate crisis fractions are in [0, 1]
    assert_crisis_fraction_bounds(crisis_fractions, "crisis_fractions")

    return crisis_fractions
