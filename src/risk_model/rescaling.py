"""
Dual rescaling of exposure matrix B_A.

- Estimation rescaling (date-specific): B̃_{A,i,t} = (σ_{i,t} / σ̄_t) · μ̄_{A,i}
- Portfolio rescaling (current-date): B̃^port_{A,i} = (σ_{i,now} / σ̄_now) · μ̄_{A,i}

Ratio σ_{i,t} / σ̄_t is winsorized at [P5, P95] cross-sectionally at each date.

WARNING: Use estimation rescaling for historical factor regression (MOD-007.2),
         and portfolio rescaling for portfolio construction (MOD-008).

Reference: ISD Section MOD-007 — Sub-task 1.
"""

import numpy as np
import pandas as pd


def _compute_winsorized_ratios(
    vol_cross_section: np.ndarray,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> np.ndarray:
    """
    Compute vol ratios relative to cross-sectional median, winsorized.

    R_i = σ_i / median(σ), winsorized at [P_lo, P_hi].

    :param vol_cross_section (np.ndarray): Trailing vols for active stocks (n,)
    :param percentile_lo (float): Lower percentile for winsorization
    :param percentile_hi (float): Upper percentile for winsorization

    :return ratios (np.ndarray): Winsorized vol ratios (n,)
    """
    median_vol = np.median(vol_cross_section)
    if median_vol <= 0:
        return np.ones_like(vol_cross_section)

    ratios = vol_cross_section / median_vol

    # Winsorize cross-sectionally
    lo = np.percentile(ratios, percentile_lo)
    hi = np.percentile(ratios, percentile_hi)
    ratios = np.clip(ratios, lo, hi)

    return ratios


def rescale_estimation(
    B_A: np.ndarray,
    trailing_vol: pd.DataFrame,
    universe_snapshots: dict[str, list[int]],
    stock_ids: list[int],
    percentile_bounds: tuple[float, float] = (5.0, 95.0),
) -> dict[str, np.ndarray]:
    """
    Date-specific rescaling for historical estimation (factor regression).

    B̃_{A,i,t} = R_{i,t} · μ̄_{A,i}   where R = winsorized(σ_{i,t} / σ̄_t)

    :param B_A (np.ndarray): Exposure matrix (n_stocks, AU)
    :param trailing_vol (pd.DataFrame): Trailing 252d annualized vol (dates × stocks)
    :param universe_snapshots (dict): date_str → list of active stock_ids (permnos)
    :param stock_ids (list[int]): Ordered stock IDs matching B_A rows
    :param percentile_bounds (tuple): (lo, hi) percentiles for winsorization

    :return B_A_by_date (dict): date_str → B_A_t (n_active_t, AU)
    """
    # Build stock_id → index mapping
    sid_to_idx = {sid: i for i, sid in enumerate(stock_ids)}

    B_A_by_date: dict[str, np.ndarray] = {}

    for date_str, active_stocks in universe_snapshots.items():
        # Get indices and vols for active stocks
        active_indices = [sid_to_idx[s] for s in active_stocks if s in sid_to_idx]
        if not active_indices:
            continue

        active_sids = [stock_ids[i] for i in active_indices]

        # Get trailing vols at this date
        if date_str not in trailing_vol.index:
            continue

        vols = np.array([
            trailing_vol.loc[date_str, sid]
            if sid in trailing_vol.columns else np.nan
            for sid in active_sids
        ], dtype=np.float64)

        # Impute NaN/zero vols with cross-sectional median (consistent
        # with rescale_portfolio — keeps all stocks, avoids dimension
        # mismatch with downstream factor regression)
        valid_mask = ~np.isnan(vols) & (vols > 0)
        if valid_mask.sum() < 2:
            continue

        median_valid = float(np.median(vols[valid_mask]))
        vols[~valid_mask] = median_valid

        # Winsorized ratios
        ratios = _compute_winsorized_ratios(
            vols, percentile_bounds[0], percentile_bounds[1],
        )

        # Rescale: B̃_{A,i,t} = R_{i,t} · μ̄_{A,i}
        B_A_active = B_A[active_indices]  # (n_active, AU)
        B_A_t = B_A_active * ratios[:, np.newaxis]

        B_A_by_date[date_str] = B_A_t

    return B_A_by_date


def rescale_portfolio(
    B_A: np.ndarray,
    trailing_vol: pd.DataFrame,
    current_date: str,
    universe: list[int],
    stock_ids: list[int],
    percentile_bounds: tuple[float, float] = (5.0, 95.0),
) -> np.ndarray:
    """
    Current-date rescaling for portfolio construction.

    B̃^port_{A,i} = R_{i,now} · μ̄_{A,i}

    :param B_A (np.ndarray): Exposure matrix (n_stocks, AU)
    :param trailing_vol (pd.DataFrame): Trailing vols (dates × stocks)
    :param current_date (str): Rebalancing date
    :param universe (list[int]): Active stocks at rebalancing (permnos)
    :param stock_ids (list[int]): Ordered stock IDs matching B_A rows
    :param percentile_bounds (tuple): (lo, hi) percentiles

    :return B_A_port (np.ndarray): Rescaled exposures (n_active, AU)
    """
    sid_to_idx = {sid: i for i, sid in enumerate(stock_ids)}

    # Get indices for universe stocks
    active_indices = [sid_to_idx[s] for s in universe if s in sid_to_idx]
    active_sids = [stock_ids[i] for i in active_indices]

    # Snap current_date to the most recent valid trading day in the index
    # (fold schedule dates may fall on weekends/holidays)
    lookup_date = trailing_vol.index.asof(pd.Timestamp(current_date))
    if isinstance(lookup_date, float) and np.isnan(lookup_date):
        # No valid date found — return unscaled exposures
        return B_A[[sid_to_idx[s] for s in universe if s in sid_to_idx]]

    # Get current-date trailing vols
    vols = np.array([
        trailing_vol.loc[lookup_date, sid]
        if sid in trailing_vol.columns else np.nan
        for sid in active_sids
    ], dtype=np.float64)

    # Handle missing vols: use cross-sectional median
    valid_mask = ~np.isnan(vols) & (vols > 0)
    if valid_mask.sum() > 0:
        median_valid = np.median(vols[valid_mask])
        vols[~valid_mask] = median_valid
    else:
        vols[:] = 1.0

    # Winsorized ratios
    ratios = _compute_winsorized_ratios(
        vols, percentile_bounds[0], percentile_bounds[1],
    )

    # Rescale
    B_A_active = B_A[active_indices]  # (n_active, AU)
    B_A_port = B_A_active * ratios[:, np.newaxis]

    return B_A_port
