"""
Out-of-sample periodic rebalancing simulation.

Implements scheduled and exceptional rebalancing during OOS evaluation:
- Scheduled rebalancing every N trading days (monthly=21, quarterly=63)
- Exceptional rebalancing when entropy drops below trigger threshold
- Shumway delisting imputation (-30% NYSE/AMEX, -55% NASDAQ)
- Transaction cost and turnover tracking
- Optional risk model refresh at each rebalancing (expanding window)

When refresh_risk_model=True, the risk model (Sigma_z, D_eps, B_port,
eigenvalues) is re-estimated at each rebalancing using an expanding window
[train_start, current_date]. This eliminates the evaluation asymmetry where
benchmarks call fit() at each rebalancing but the VAE would otherwise use a
frozen training-period model during the entire OOS period.

Reference: DVT §4.2 — Rebalancing Protocol, §4.7 — Rescaling Protocol.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.portfolio.entropy import compute_entropy_only
from src.portfolio.sca_solver import multi_start_optimize
from src.risk_model.covariance import assemble_risk_model, estimate_d_eps, estimate_sigma_z
from src.risk_model.factor_regression import compute_residuals, estimate_factor_returns
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.validation import (
    assert_growth_finite,
    assert_weights_sum_to_one,
)

if TYPE_CHECKING:
    from src.benchmarks.base import BenchmarkModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RebalancingEvent:
    """
    Record of a single rebalancing event during OOS simulation.

    :param date (str): Date of rebalancing (YYYY-MM-DD)
    :param trigger (str): "initial", "scheduled", or "exceptional"
    :param turnover (float): One-way turnover (half of |w_new - w_old|)
    :param transaction_cost (float): TC in return units (turnover × tc_bps/10000)
    :param entropy_before (float): H(w_old) before rebalancing
    :param entropy_after (float): H(w_new) after rebalancing
    :param n_delisted (int): Number of stocks that delisted since last rebalance
    :param n_active (int): Number of active positions after rebalancing
    """
    date: str
    trigger: str
    turnover: float
    transaction_cost: float
    entropy_before: float
    entropy_after: float
    n_delisted: int
    n_active: int


@dataclass
class OOSRebalancingResult:
    """
    Complete result of OOS simulation with periodic rebalancing.

    :param daily_returns (np.ndarray): Portfolio daily log-returns (T_oos,)
    :param cumulative_return (float): Total cumulative return (exp(sum(log_r)) - 1)
    :param cumulative_turnover (float): Total two-way turnover accumulated
    :param total_transaction_cost (float): Sum of all TC in return units
    :param n_scheduled_rebalances (int): Count of scheduled rebalances
    :param n_exceptional_rebalances (int): Count of exceptional rebalances
    :param entropy_trajectory (list[float]): H(w) at each rebalancing
    :param rebalancing_events (list[RebalancingEvent]): Detailed event log
    :param final_weights (np.ndarray): Portfolio weights at end of OOS
    :param final_universe (list[int]): Stock IDs at end of OOS
    """
    daily_returns: np.ndarray
    cumulative_return: float
    cumulative_turnover: float
    total_transaction_cost: float
    n_scheduled_rebalances: int
    n_exceptional_rebalances: int
    entropy_trajectory: list[float]
    rebalancing_events: list[RebalancingEvent]
    final_weights: np.ndarray
    final_universe: list[int]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _align_weights_to_new_universe(
    w_old: np.ndarray,
    old_ids: list[int],
    new_ids: list[int],
) -> np.ndarray:
    """
    Re-index portfolio weights from old universe to new universe.

    Stocks present in both get their old weight; new stocks get 0; exited
    stocks are dropped. Result is NOT renormalized (caller handles this).

    :param w_old (np.ndarray): Previous weights (n_old,)
    :param old_ids (list[int]): Stock IDs for w_old
    :param new_ids (list[int]): Stock IDs for new universe

    :return w_aligned (np.ndarray): Aligned weights (n_new,)
    """
    old_map = {sid: i for i, sid in enumerate(old_ids)}
    w_aligned = np.zeros(len(new_ids), dtype=np.float64)
    for j, sid in enumerate(new_ids):
        idx = old_map.get(sid)
        if idx is not None:
            w_aligned[j] = w_old[idx]
    return w_aligned


def _handle_delistings_at_date(
    w: np.ndarray,
    stock_ids: list[int],
    delisted_ids: set[int],
    exchange_codes: dict[int, int],
    delisting_return_nyse_amex: float,
    delisting_return_nasdaq: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Handle delisted stocks: impute Shumway return and zero out weights.

    :param w (np.ndarray): Current weights (n,)
    :param stock_ids (list[int]): Stock IDs matching w
    :param delisted_ids (set[int]): IDs of stocks that delisted on this date
    :param exchange_codes (dict[int, int]): permno -> exchange code (1=NYSE, 2=AMEX, 3=NASDAQ)
    :param delisting_return_nyse_amex (float): Shumway return for NYSE/AMEX delistings
    :param delisting_return_nasdaq (float): Shumway return for NASDAQ delistings

    :return w_new (np.ndarray): Weights with delisted zeroed out
    :return imputed_returns (np.ndarray): Return contribution from delistings (n,)
    :return total_delisting_return (float): Weighted sum of Shumway returns
    """
    w_new = w.copy()
    imputed_returns = np.zeros_like(w)
    total_delisting_return = 0.0

    for i, sid in enumerate(stock_ids):
        if sid in delisted_ids and w[i] > 0:
            exc = exchange_codes.get(sid, 1)
            if exc == 3:  # NASDAQ
                shumway = delisting_return_nasdaq
            else:  # NYSE=1, AMEX=2
                shumway = delisting_return_nyse_amex

            imputed_returns[i] = shumway
            total_delisting_return += w[i] * shumway
            w_new[i] = 0.0

    # Renormalize remaining weights
    total = np.sum(w_new)
    if total > 1e-10:
        w_new = w_new / total

    return w_new, imputed_returns, total_delisting_return


_VT_PRIOR_STRENGTH = 20  # Bayesian shrinkage prior for variance targeting


def _refresh_risk_model(
    B_A_raw: np.ndarray,
    inferred_stock_ids: list[int],
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    current_date: str,
    train_start: str,
    B_A_by_date_cumulative: dict[str, np.ndarray],
    universe_snapshots_cumulative: dict[str, list[int]],
    last_update_date: str,
    risk_model_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Refresh the full risk model at an OOS rebalancing date.

    Uses an expanding window [train_start, current_date] to re-estimate
    Sigma_z, D_eps, and portfolio-rescaled B_port. B_A_raw (VAE exposures)
    are frozen — only the dual rescaling, factor regression, and covariance
    estimation are refreshed with the latest return data.

    Mutates B_A_by_date_cumulative and universe_snapshots_cumulative in-place
    by extending them with new OOS dates (last_update_date, current_date].

    :param B_A_raw (np.ndarray): Raw VAE exposure matrix (n, AU) — frozen
    :param inferred_stock_ids (list[int]): Stock IDs matching B_A_raw rows
    :param returns (pd.DataFrame): Full returns DataFrame (train + OOS)
    :param trailing_vol (pd.DataFrame): Trailing volatilities (full history)
    :param current_date (str): Current rebalancing date (YYYY-MM-DD)
    :param train_start (str): Start of training period (YYYY-MM-DD)
    :param B_A_by_date_cumulative (dict): Mutable dict of date -> B_A_t,
        extended in-place with new OOS dates
    :param universe_snapshots_cumulative (dict): Mutable dict of date -> stock IDs,
        extended in-place with new OOS dates
    :param last_update_date (str): Last date included in B_A_by_date
    :param risk_model_config (dict): Risk model configuration parameters

    :return result (dict): Contains Sigma_assets, eigenvalues_signal,
        B_prime_signal, D_eps, last_update_date
    """
    market_intercept = bool(risk_model_config.get("market_intercept", True))
    winsorize_lo = float(risk_model_config.get("winsorize_lo", 5.0))
    winsorize_hi = float(risk_model_config.get("winsorize_hi", 95.0))
    percentile_bounds = (winsorize_lo, winsorize_hi)

    # 1. Extend universe_snapshots for new OOS dates (last_update_date, current_date]
    oos_returns = returns.loc[last_update_date:current_date]
    oos_date_strs = [str(d.date()) if hasattr(d, "date") else str(d)[:10]
                     for d in oos_returns.index]
    last_update_str = last_update_date[:10]
    for d in oos_date_strs:
        if d <= last_update_str:
            continue
        if d in universe_snapshots_cumulative:
            continue
        row = oos_returns.loc[oos_returns.index[oos_date_strs.index(d)]]
        valid_ids = [
            sid for sid in inferred_stock_ids
            if sid in row.index and pd.notna(row[sid])
        ]
        universe_snapshots_cumulative[d] = valid_ids if valid_ids else inferred_stock_ids

    # 2. Extend B_A_by_date via rescale_estimation() on new dates only
    new_snapshots = {
        d: sids for d, sids in universe_snapshots_cumulative.items()
        if d > last_update_str and d not in B_A_by_date_cumulative
    }
    if new_snapshots:
        new_vol = trailing_vol.loc[last_update_date:current_date]
        new_B_A_by_date = rescale_estimation(
            B_A_raw, new_vol, new_snapshots, inferred_stock_ids,
            percentile_bounds=percentile_bounds,
        )
        # 3. Add market intercept to new entries if configured
        if market_intercept:
            for date_str in new_B_A_by_date:
                n_active = new_B_A_by_date[date_str].shape[0]
                intercept = np.ones((n_active, 1), dtype=np.float64)
                new_B_A_by_date[date_str] = np.hstack(
                    [new_B_A_by_date[date_str], intercept]
                )
        B_A_by_date_cumulative.update(new_B_A_by_date)

    # 4. Factor regression on full expanding window [train_start, current_date]
    full_returns = returns.loc[train_start:current_date]
    z_hat, valid_dates = estimate_factor_returns(
        B_A_by_date_cumulative, full_returns,
        universe_snapshots_cumulative,
        use_wls=bool(risk_model_config.get("use_wls", True)),
    )

    n_factors = z_hat.shape[1] if z_hat.shape[0] > 0 else B_A_raw.shape[1]
    if z_hat.shape[0] < n_factors:
        logger.warning(
            "Risk model refresh: insufficient dates for regression (%d < %d factors)",
            z_hat.shape[0], n_factors,
        )
        return {}

    # Split z_hat: 80% estimation / 20% VT holdout
    n_z = z_hat.shape[0]
    n_vt_holdout = max(20, int(n_z * 0.2))
    n_est = n_z - n_vt_holdout
    z_hat_est = z_hat[:n_est]
    valid_dates_est = valid_dates[:n_est]

    # 5. Residuals from estimation period
    residuals = compute_residuals(
        B_A_by_date_cumulative, z_hat_est,
        full_returns,
        universe_snapshots_cumulative, valid_dates_est,
        inferred_stock_ids,
    )

    # 6. Sigma_z estimation
    Sigma_z, n_signal, _ = estimate_sigma_z(
        z_hat_est,
        eigenvalue_pct=float(risk_model_config.get("eigenvalue_pct", 0.95)),
        shrinkage_method=str(risk_model_config.get("shrinkage_method", "analytical_nonlinear")),
        ewma_half_life=int(risk_model_config.get("ewma_half_life", 0)),
    )

    # 7. D_eps estimation
    D_eps = estimate_d_eps(
        residuals, inferred_stock_ids,
        d_eps_floor=float(risk_model_config.get("d_eps_floor", 1e-6)),
    )

    # 8. Portfolio rescaling at current_date
    B_A_port = rescale_portfolio(
        B_A_raw, trailing_vol, current_date,
        inferred_stock_ids, inferred_stock_ids,
        percentile_bounds=percentile_bounds,
    )
    if market_intercept:
        intercept_port = np.ones((B_A_port.shape[0], 1), dtype=np.float64)
        B_A_port = np.hstack([B_A_port, intercept_port])

    n_port = B_A_port.shape[0]
    D_eps_port = D_eps[:n_port]

    # 9. Assemble risk model
    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
    Sigma_assets = risk_model["Sigma_assets"]
    eigenvalues = risk_model["eigenvalues"]
    B_prime_port = risk_model["B_prime_port"]
    V = risk_model["V"]

    # 10. Signal/noise split (skip PC1 if market_intercept)
    n_signal = max(1, min(n_signal, len(eigenvalues)))
    skip_pc1 = market_intercept
    signal_start = 1 if skip_pc1 else 0
    signal_end = n_signal + (1 if skip_pc1 else 0)
    signal_end = min(signal_end, len(eigenvalues))

    # 11. Per-factor variance targeting on holdout
    z_hat_principal = z_hat @ V
    z_holdout_principal = z_hat_principal[n_est:]
    n_holdout_vt = z_holdout_principal.shape[0]
    vt_min = float(risk_model_config.get("vt_clamp_min", 0.5))
    vt_max = float(risk_model_config.get("vt_clamp_max", 2.0))

    if n_holdout_vt >= 20:
        realized_var = np.var(z_holdout_principal, axis=0, ddof=1)
        pred_var = np.maximum(eigenvalues, 1e-15)
        vt_raw = realized_var / pred_var
        vt_shrunk = (
            n_holdout_vt * vt_raw + _VT_PRIOR_STRENGTH * 1.0
        ) / (n_holdout_vt + _VT_PRIOR_STRENGTH)
        vt_factors = np.clip(vt_shrunk, vt_min, vt_max)
        eigenvalues = eigenvalues * vt_factors

        # Idiosyncratic VT
        residuals_holdout = compute_residuals(
            B_A_by_date_cumulative, z_hat[n_est:],
            full_returns,
            universe_snapshots_cumulative, valid_dates[n_est:],
            inferred_stock_ids,
        )
        idio_ratios: list[float] = []
        for i, sid in enumerate(inferred_stock_ids[:n_port]):
            resids_h = residuals_holdout.get(sid, [])
            if len(resids_h) >= 10 and D_eps_port[i] > 1e-12:
                realized_idio = float(np.var(resids_h, ddof=1))
                idio_ratios.append(realized_idio / float(D_eps_port[i]))
        if len(idio_ratios) >= 10:
            vt_idio_raw = float(np.median(idio_ratios))
            n_idio_obs = len(idio_ratios)
            vt_idio = float(np.clip(
                (n_idio_obs * vt_idio_raw + _VT_PRIOR_STRENGTH * 1.0)
                / (n_idio_obs + _VT_PRIOR_STRENGTH),
                vt_min, vt_max,
            ))
        else:
            vt_idio = 1.0
        D_eps_port = np.maximum(
            D_eps_port * vt_idio,
            float(risk_model_config.get("d_eps_floor", 1e-6)),
        )

    # Rebuild Sigma_assets with VT-calibrated eigenvalues
    eigenvalues_signal = eigenvalues[signal_start:signal_end].copy()
    B_prime_signal = B_prime_port[:, signal_start:signal_end].copy()

    Sigma_sys_vt = B_prime_port @ np.diag(eigenvalues) @ B_prime_port.T
    Sigma_assets = Sigma_sys_vt + np.diag(D_eps_port)
    Sigma_assets = 0.5 * (Sigma_assets + Sigma_assets.T)

    logger.info(
        "  [Risk refresh] %s: n_dates=%d (est=%d, holdout=%d), "
        "n_signal=%d, eigenvalues_signal=[%.2e..%.2e]",
        current_date, n_z, n_est, n_vt_holdout,
        len(eigenvalues_signal),
        float(eigenvalues_signal[-1]) if len(eigenvalues_signal) > 0 else 0.0,
        float(eigenvalues_signal[0]) if len(eigenvalues_signal) > 0 else 0.0,
    )

    return {
        "Sigma_assets": Sigma_assets,
        "eigenvalues_signal": eigenvalues_signal,
        "B_prime_signal": B_prime_signal,
        "D_eps": D_eps_port,
        "last_update_date": current_date,
    }


def _execute_rebalancing(
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    Sigma_assets: np.ndarray,
    D_eps: np.ndarray,
    w_old: np.ndarray,
    n_active: int,
    constraint_params: dict[str, Any],
    is_first: bool,
    seed: int,
    trigger: str = "scheduled",
) -> np.ndarray:
    """
    Execute full portfolio re-optimization.

    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)
    :param w_old (np.ndarray): Previous weights (for turnover)
    :param n_active (int): Number of active stocks
    :param constraint_params (dict): Portfolio constraint parameters
    :param is_first (bool): True for first rebalancing (no turnover constraint)
    :param seed (int): Random seed for multi-start
    :param trigger (str): "scheduled" (routine) or "exceptional" (urgent H degradation)

    :return w_new (np.ndarray): Optimized weights (n,)
    """
    cp = constraint_params

    # Default alpha (can be refined via frontier if needed)
    alpha = float(cp.get("alpha_opt", 0.5))
    idio_weight = float(cp.get("entropy_idio_weight", 0.2))

    # Differentiate solver quality settings based on trigger type:
    # - Scheduled: quick (n_starts=2, max_iter=50) — portfolio already well-positioned
    # - Exceptional: full quality (n_starts=5, max_iter=100) — urgent H degradation
    if trigger == "exceptional":
        n_starts = int(cp.get("oos_n_starts_exceptional", cp.get("n_starts", 5)))
        max_iter = int(cp.get("oos_sca_max_iter_exceptional", 100))
    else:
        n_starts = int(cp.get("oos_n_starts_scheduled", 2))
        max_iter = int(cp.get("oos_sca_max_iter_scheduled", 50))

    w_new: np.ndarray
    try:
        result = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=alpha,
            n_starts=n_starts,
            lambda_risk=float(cp.get("lambda_risk", 252.0)),
            w_max=float(cp.get("w_max", 0.05)),
            w_min=float(cp.get("w_min", 0.001)),
            w_bar=float(cp.get("w_bar", 0.03)),
            phi=float(cp.get("phi", 0.0)),
            w_old=w_old if not is_first else None,
            kappa_1=float(cp.get("kappa_1", 0.1)),
            kappa_2=float(cp.get("kappa_2", 7.5)),
            delta_bar=float(cp.get("delta_bar", 0.01)),
            tau_max=float(cp.get("tau_max", 0.30)),
            is_first=is_first,
            seed=seed,
            D_eps=D_eps,
            idio_weight=idio_weight,
            max_iter=max_iter,  # SCA iteration limit (trigger-dependent)
        )
        # multi_start_optimize returns either ndarray or tuple(ndarray, H, var)
        if isinstance(result, tuple):
            w_new = result[0]
        else:
            w_new = result
    except Exception as e:
        logger.warning("Rebalancing optimization failed: %s — keeping old weights", e)
        w_new = w_old.copy()

    return w_new


# ---------------------------------------------------------------------------
# Main Simulation Function
# ---------------------------------------------------------------------------

def simulate_oos_rebalancing(
    # Frozen risk model
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    B_A_raw: np.ndarray,
    inferred_stock_ids: list[int],
    Sigma_assets_initial: np.ndarray,
    D_eps_initial: np.ndarray,
    # OOS data
    returns_oos: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    exchange_codes: dict[int, int],
    # Initial state
    w_initial: np.ndarray,
    H_initial: float,
    alpha_opt: float,
    # Config
    rebalancing_frequency_days: int = 21,
    entropy_trigger_alpha: float = 0.90,
    tc_bps: float = 10.0,
    delisting_return_nyse_amex: float = -0.30,
    delisting_return_nasdaq: float = -0.55,
    constraint_params: dict[str, Any] | None = None,
    d_eps_lookback: int = 252,
    d_eps_floor: float = 1e-6,
    seed: int = 42,
    idio_weight: float = 0.2,
    # Risk model refresh (DVT §4.7)
    refresh_risk_model: bool = False,
    returns_full: pd.DataFrame | None = None,
    train_start: str | None = None,
    risk_model_config: dict[str, Any] | None = None,
    B_A_by_date_initial: dict[str, np.ndarray] | None = None,
    universe_snapshots_initial: dict[str, list[int]] | None = None,
) -> OOSRebalancingResult:
    """
    Simulate OOS period with periodic rebalancing.

    When refresh_risk_model=False (default), B_prime and eigenvalues are
    FROZEN during OOS — only portfolio weights are re-optimized.

    When refresh_risk_model=True, the risk model (Sigma_z, D_eps, B_port,
    eigenvalues) is re-estimated at each rebalancing using an expanding
    window [train_start, current_date]. B_A_raw (VAE exposures) remain
    frozen — only rescaling, regression, and covariance are refreshed.

    :param B_prime (np.ndarray): Rotated exposures in principal basis (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param B_A_raw (np.ndarray): Raw exposure matrix for factor regression (n, AU)
    :param inferred_stock_ids (list[int]): Stock IDs matching B_prime rows
    :param Sigma_assets_initial (np.ndarray): Initial asset covariance (n, n)
    :param D_eps_initial (np.ndarray): Initial idiosyncratic variances (n,)
    :param returns_oos (pd.DataFrame): OOS daily returns (dates x stocks)
    :param trailing_vol (pd.DataFrame): Trailing volatilities for rescaling
    :param exchange_codes (dict[int, int]): permno -> exchange code for Shumway
    :param w_initial (np.ndarray): Initial portfolio weights from training
    :param H_initial (float): Initial entropy H(w_initial)
    :param alpha_opt (float): Optimal alpha from frontier (for re-optimization)
    :param rebalancing_frequency_days (int): Days between scheduled rebalances (0=buy-hold)
    :param entropy_trigger_alpha (float): Exceptional trigger: H < alpha * H_last
    :param tc_bps (float): Transaction cost in basis points (one-way)
    :param delisting_return_nyse_amex (float): Shumway return for NYSE/AMEX
    :param delisting_return_nasdaq (float): Shumway return for NASDAQ
    :param constraint_params (dict | None): Portfolio constraint parameters
    :param d_eps_lookback (int): Trailing window for D_eps estimation (days)
    :param d_eps_floor (float): Floor for idiosyncratic variance
    :param seed (int): Random seed for optimization
    :param idio_weight (float): Weight for idiosyncratic entropy layer
    :param refresh_risk_model (bool): If True, re-estimate risk model at each
        rebalancing using expanding window. Requires returns_full, train_start,
        risk_model_config, B_A_by_date_initial, universe_snapshots_initial.
    :param returns_full (pd.DataFrame | None): Full returns (train + OOS) for
        risk model refresh. Required when refresh_risk_model=True.
    :param train_start (str | None): Start of training period (YYYY-MM-DD).
        Required when refresh_risk_model=True.
    :param risk_model_config (dict | None): Risk model config parameters.
        Required when refresh_risk_model=True.
    :param B_A_by_date_initial (dict | None): Initial date -> B_A_t dict from
        training. Copied and extended during OOS. Required when refresh=True.
    :param universe_snapshots_initial (dict | None): Initial date -> stock_ids
        dict from training. Copied and extended. Required when refresh=True.

    :return result (OOSRebalancingResult): Complete simulation result
    """
    if constraint_params is None:
        constraint_params = {}

    # Inject alpha_opt for re-optimization
    constraint_params = {**constraint_params, "alpha_opt": alpha_opt}

    n_stocks = len(inferred_stock_ids)
    oos_dates = returns_oos.index.tolist()
    n_days = len(oos_dates)

    # Initialize state
    w_current = w_initial.copy()
    current_universe = inferred_stock_ids.copy()
    H_last_rebalance = H_initial
    days_since_rebalance = 0

    # Use initial risk model (frozen B_prime, eigenvalues, Sigma)
    Sigma_current = Sigma_assets_initial.copy()
    D_eps_current = D_eps_initial.copy()

    # Mutable copies of B_prime and eigenvalues (updated when refresh=True)
    B_prime_current = B_prime.copy()
    eigenvalues_current = eigenvalues.copy()

    # Risk model refresh state
    if refresh_risk_model:
        if returns_full is None or train_start is None or risk_model_config is None:
            raise ValueError(
                "refresh_risk_model=True requires returns_full, train_start, "
                "and risk_model_config"
            )
        if B_A_by_date_initial is None or universe_snapshots_initial is None:
            raise ValueError(
                "refresh_risk_model=True requires B_A_by_date_initial and "
                "universe_snapshots_initial"
            )
        # Deep-copy dicts so we don't mutate the caller's data
        B_A_by_date_cumul = {k: v.copy() for k, v in B_A_by_date_initial.items()}
        universe_snapshots_cumul = {k: list(v) for k, v in universe_snapshots_initial.items()}
        # Last date covered by the initial B_A_by_date
        last_update_date = max(B_A_by_date_cumul.keys()) if B_A_by_date_cumul else train_start
    else:
        B_A_by_date_cumul = {}
        universe_snapshots_cumul = {}
        last_update_date = ""

    # Results accumulators
    daily_returns: list[float] = []
    total_turnover = 0.0
    total_tc = 0.0
    n_scheduled = 0
    n_exceptional = 0
    entropy_trajectory: list[float] = [H_initial]
    rebalancing_events: list[RebalancingEvent] = []

    # Expected number of scheduled rebalances (for logging)
    expected_scheduled = n_days // rebalancing_frequency_days if rebalancing_frequency_days > 0 else 0

    # Record initial event
    rebalancing_events.append(RebalancingEvent(
        date=str(oos_dates[0])[:10] if n_days > 0 else "N/A",
        trigger="initial",
        turnover=0.0,
        transaction_cost=0.0,
        entropy_before=H_initial,
        entropy_after=H_initial,
        n_delisted=0,
        n_active=int(np.sum(w_current > 1e-6)),
    ))

    # Daily simulation
    for t in range(n_days):
        date = oos_dates[t]
        date_str = str(date)[:10]

        # Get available returns for today
        ret_row = returns_oos.iloc[t]
        available_ids = [sid for sid in current_universe if sid in returns_oos.columns]

        # Align weights to available stocks
        w_aligned = _align_weights_to_new_universe(w_current, current_universe, available_ids)

        # Compute daily portfolio return (before any delisting handling)
        port_ret = 0.0
        for i, sid in enumerate(available_ids):
            if w_aligned[i] > 1e-10:
                r = ret_row.get(sid, 0.0)
                if pd.notna(r):
                    port_ret += w_aligned[i] * r

        # Check for delistings (stocks with NaN returns that had weight)
        delisted_today: set[int] = set()
        for i, sid in enumerate(available_ids):
            r = ret_row.get(sid, np.nan)
            if pd.isna(r) and w_aligned[i] > 1e-6:
                delisted_today.add(sid)

        # Handle delistings with Shumway imputation
        if delisted_today:
            w_aligned, imputed_ret, delisting_return = _handle_delistings_at_date(
                w_aligned, available_ids, delisted_today, exchange_codes,
                delisting_return_nyse_amex, delisting_return_nasdaq,
            )
            port_ret += delisting_return

        daily_returns.append(port_ret)
        days_since_rebalance += 1

        # Update weight drift due to returns (mark-to-market)
        if np.sum(w_aligned) > 1e-10:
            # Simple drift: w_i_new ≈ w_i * (1 + r_i) / sum(w_j * (1 + r_j))
            # For log returns: w_i_new ≈ w_i * exp(r_i) / sum(w_j * exp(r_j))
            growth = np.array([
                w_aligned[i] * np.exp(ret_row.get(sid, 0.0) if pd.notna(ret_row.get(sid)) else 0.0)
                for i, sid in enumerate(available_ids)
            ])

            # VALIDATION: Growth factors must be finite and non-negative
            assert_growth_finite(growth, f"oos_growth_{date_str}")

            total_growth = np.sum(growth)
            if total_growth > 1e-10:
                w_aligned = growth / total_growth
            else:
                logger.warning(
                    "Total growth near zero at %s — portfolio may be wiped out",
                    date_str,
                )

        # Validate drifted weights still sum to ~1
        if np.sum(w_aligned) > 1e-10:
            assert_weights_sum_to_one(w_aligned, "w_aligned_after_drift")

        # Update current state
        w_current = w_aligned
        current_universe = available_ids

        # Skip rebalancing logic if buy-and-hold mode
        if rebalancing_frequency_days == 0:
            continue

        # Check for rebalancing triggers
        trigger = None

        # Scheduled rebalancing
        if days_since_rebalance >= rebalancing_frequency_days:
            trigger = "scheduled"
            n_scheduled += 1

        # Exceptional rebalancing: entropy drop
        if trigger is None and len(current_universe) > 0:
            # Need aligned B_prime and D_eps for entropy
            B_aligned = _align_matrix_to_universe(
                B_prime_current, inferred_stock_ids, current_universe,
            )
            D_aligned = _align_vector_to_universe(
                D_eps_current, inferred_stock_ids, current_universe,
            )

            H_current = compute_entropy_only(
                w_current, B_aligned, eigenvalues_current,
                D_eps=D_aligned, idio_weight=idio_weight,
            )

            if H_current < entropy_trigger_alpha * H_last_rebalance:
                trigger = "exceptional"
                n_exceptional += 1
                logger.info(
                    "  [Rebal] Exceptional trigger at %s: H=%.4f < %.2f × %.4f",
                    date_str, H_current, entropy_trigger_alpha, H_last_rebalance,
                )

        # Execute rebalancing
        if trigger is not None:
            # Refresh risk model if enabled (expanding window estimation)
            if refresh_risk_model and returns_full is not None:
                assert train_start is not None
                assert risk_model_config is not None
                refresh_result = _refresh_risk_model(
                    B_A_raw=B_A_raw,
                    inferred_stock_ids=inferred_stock_ids,
                    returns=returns_full,
                    trailing_vol=trailing_vol,
                    current_date=date_str,
                    train_start=train_start,
                    B_A_by_date_cumulative=B_A_by_date_cumul,
                    universe_snapshots_cumulative=universe_snapshots_cumul,
                    last_update_date=last_update_date,
                    risk_model_config=risk_model_config,
                )
                if refresh_result:
                    Sigma_current = refresh_result["Sigma_assets"]
                    D_eps_current = refresh_result["D_eps"]
                    B_prime_current = refresh_result["B_prime_signal"]
                    eigenvalues_current = refresh_result["eigenvalues_signal"]
                    last_update_date = refresh_result["last_update_date"]

            # Align risk model components to current universe
            B_aligned = _align_matrix_to_universe(
                B_prime_current, inferred_stock_ids, current_universe,
            )
            B_A_aligned = _align_matrix_to_universe(
                B_A_raw, inferred_stock_ids, current_universe,
            )
            D_aligned = _align_vector_to_universe(
                D_eps_current, inferred_stock_ids, current_universe,
            )
            Sigma_aligned = _align_covariance_to_universe(
                Sigma_current, inferred_stock_ids, current_universe,
            )

            # Compute entropy before rebalancing
            H_before = compute_entropy_only(
                w_current, B_aligned, eigenvalues_current,
                D_eps=D_aligned, idio_weight=idio_weight,
            )

            # Re-optimize with trigger-dependent quality settings
            w_old_for_tc = w_current.copy()
            w_new = _execute_rebalancing(
                B_prime=B_aligned,
                eigenvalues=eigenvalues_current,
                Sigma_assets=Sigma_aligned,
                D_eps=D_aligned,
                w_old=w_current,
                n_active=len(current_universe),
                constraint_params=constraint_params,
                is_first=False,
                seed=seed + t,
                trigger=trigger,
            )

            # Validate rebalanced weights before turnover computation
            assert_weights_sum_to_one(w_new, "w_new_after_rebalancing")

            # Compute turnover and TC
            turnover = 0.5 * np.sum(np.abs(w_new - w_old_for_tc))
            tc = turnover * (tc_bps / 10000.0)
            total_turnover += 2.0 * turnover  # Two-way
            total_tc += tc

            # Compute entropy after rebalancing
            H_after = compute_entropy_only(
                w_new, B_aligned, eigenvalues_current,
                D_eps=D_aligned, idio_weight=idio_weight,
            )

            # Update state
            w_current = w_new
            H_last_rebalance = H_after
            days_since_rebalance = 0
            entropy_trajectory.append(H_after)

            # Record event
            rebalancing_events.append(RebalancingEvent(
                date=date_str,
                trigger=trigger,
                turnover=turnover,
                transaction_cost=tc,
                entropy_before=H_before,
                entropy_after=H_after,
                n_delisted=len(delisted_today),
                n_active=int(np.sum(w_new > 1e-6)),
            ))

            logger.info(
                "  [Rebal %d/%d] %s at %s: one-way=%.2f%%, two-way=%.2f%%, "
                "cumul=%.1f%%, H=%.4f→%.4f, n_active=%d",
                n_scheduled + n_exceptional,  # Current rebalance number
                expected_scheduled,  # Expected scheduled (excludes exceptional)
                trigger, date_str,
                turnover * 100,  # One-way this rebalance
                turnover * 200,  # Two-way this rebalance
                total_turnover * 100,  # Cumulative two-way
                H_before, H_after,
                int(np.sum(w_new > 1e-6)),
            )

    # Final results
    daily_returns_arr = np.array(daily_returns)
    cumulative = float(np.exp(np.sum(daily_returns_arr))) - 1.0

    return OOSRebalancingResult(
        daily_returns=daily_returns_arr,
        cumulative_return=cumulative,
        cumulative_turnover=total_turnover,
        total_transaction_cost=total_tc,
        n_scheduled_rebalances=n_scheduled,
        n_exceptional_rebalances=n_exceptional,
        entropy_trajectory=entropy_trajectory,
        rebalancing_events=rebalancing_events,
        final_weights=w_current,
        final_universe=current_universe,
    )


# ---------------------------------------------------------------------------
# Alignment Helpers
# ---------------------------------------------------------------------------

def _align_matrix_to_universe(
    M: np.ndarray,
    old_ids: list[int],
    new_ids: list[int],
) -> np.ndarray:
    """
    Align matrix rows from old universe to new universe.

    :param M (np.ndarray): Matrix (n_old, k)
    :param old_ids (list[int]): Stock IDs for M rows
    :param new_ids (list[int]): Target stock IDs

    :return M_aligned (np.ndarray): Aligned matrix (n_new, k)
    """
    old_map = {sid: i for i, sid in enumerate(old_ids)}
    n_new = len(new_ids)
    k = M.shape[1] if M.ndim > 1 else 1

    M_aligned = np.zeros((n_new, k), dtype=np.float64)
    for j, sid in enumerate(new_ids):
        idx = old_map.get(sid)
        if idx is not None:
            M_aligned[j] = M[idx]

    return M_aligned


def _align_vector_to_universe(
    v: np.ndarray,
    old_ids: list[int],
    new_ids: list[int],
    default: float = 1e-6,
) -> np.ndarray:
    """
    Align vector from old universe to new universe.

    :param v (np.ndarray): Vector (n_old,)
    :param old_ids (list[int]): Stock IDs for v elements
    :param new_ids (list[int]): Target stock IDs
    :param default (float): Default value for missing stocks

    :return v_aligned (np.ndarray): Aligned vector (n_new,)
    """
    old_map = {sid: i for i, sid in enumerate(old_ids)}
    n_new = len(new_ids)

    v_aligned = np.full(n_new, default, dtype=np.float64)
    for j, sid in enumerate(new_ids):
        idx = old_map.get(sid)
        if idx is not None:
            v_aligned[j] = v[idx]

    return v_aligned


def _align_covariance_to_universe(
    Sigma: np.ndarray,
    old_ids: list[int],
    new_ids: list[int],
    default_var: float = 1e-4,
) -> np.ndarray:
    """
    Align covariance matrix from old universe to new universe.

    :param Sigma (np.ndarray): Covariance (n_old, n_old)
    :param old_ids (list[int]): Stock IDs for Sigma
    :param new_ids (list[int]): Target stock IDs
    :param default_var (float): Default variance for missing stocks

    :return Sigma_aligned (np.ndarray): Aligned covariance (n_new, n_new)
    """
    old_map = {sid: i for i, sid in enumerate(old_ids)}
    n_new = len(new_ids)

    # Initialize with default diagonal
    Sigma_aligned = np.diag(np.full(n_new, default_var, dtype=np.float64))

    for i, sid_i in enumerate(new_ids):
        idx_i = old_map.get(sid_i)
        if idx_i is None:
            continue
        for j, sid_j in enumerate(new_ids):
            idx_j = old_map.get(sid_j)
            if idx_j is None:
                continue
            Sigma_aligned[i, j] = Sigma[idx_i, idx_j]

    return Sigma_aligned


# ---------------------------------------------------------------------------
# Benchmark Rebalancing Simulation
# ---------------------------------------------------------------------------

def simulate_benchmark_oos_rebalancing(
    benchmark: "BenchmarkModel",  # noqa: F821
    returns_oos: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    w_initial: np.ndarray,
    universe: list[str],
    exchange_codes: dict[int, int],
    rebalancing_frequency_days: int = 21,
    tc_bps: float = 10.0,
    delisting_return_nyse_amex: float = -0.30,
    delisting_return_nasdaq: float = -0.55,
    constraint_params: dict[str, Any] | None = None,
    d_eps_lookback: int = 252,
) -> OOSRebalancingResult:
    """
    Simulate OOS period with periodic rebalancing for a benchmark.

    Unlike VAE rebalancing, benchmarks call their fit() + optimize() methods
    at each rebalancing to re-estimate their risk model.

    :param benchmark: Benchmark model instance (already fit on training data)
    :param returns_oos (pd.DataFrame): OOS daily returns
    :param trailing_vol (pd.DataFrame): Trailing volatilities
    :param w_initial (np.ndarray): Initial weights from training period
    :param universe (list[str]): Stock universe (string IDs)
    :param exchange_codes (dict[int, int]): permno -> exchange code
    :param rebalancing_frequency_days (int): Days between rebalances (0=buy-hold)
    :param tc_bps (float): Transaction cost in basis points
    :param delisting_return_nyse_amex (float): Shumway return for NYSE/AMEX
    :param delisting_return_nasdaq (float): Shumway return for NASDAQ
    :param constraint_params (dict | None): Constraint parameters
    :param d_eps_lookback (int): Lookback for re-estimation

    :return result (OOSRebalancingResult): Simulation result
    """
    if constraint_params is None:
        constraint_params = {}

    oos_dates = returns_oos.index.tolist()
    n_days = len(oos_dates)

    # Convert universe to int for exchange lookup
    universe_int = [int(s) for s in universe]

    # Initialize state
    w_current = w_initial.copy()
    current_universe = list(universe)
    days_since_rebalance = 0

    # Results accumulators
    daily_returns: list[float] = []
    total_turnover = 0.0
    total_tc = 0.0
    n_scheduled = 0
    rebalancing_events: list[RebalancingEvent] = []

    # Record initial event
    rebalancing_events.append(RebalancingEvent(
        date=str(oos_dates[0])[:10] if n_days > 0 else "N/A",
        trigger="initial",
        turnover=0.0,
        transaction_cost=0.0,
        entropy_before=0.0,  # Benchmarks don't track entropy
        entropy_after=0.0,
        n_delisted=0,
        n_active=int(np.sum(w_current > 1e-6)),
    ))

    # Daily simulation
    for t in range(n_days):
        date = oos_dates[t]
        date_str = str(date)[:10]

        # Get available returns for today
        ret_row = returns_oos.iloc[t]
        available_ids = [sid for sid in current_universe if sid in returns_oos.columns]

        # Align weights to available stocks
        w_aligned = _align_weights_to_new_universe(
            w_current,
            [int(s) for s in current_universe],
            [int(s) for s in available_ids],
        )

        # Compute daily portfolio return
        port_ret = 0.0
        for i, sid in enumerate(available_ids):
            if w_aligned[i] > 1e-10:
                r = ret_row.get(sid, 0.0)
                if pd.notna(r):
                    port_ret += w_aligned[i] * r

        # Check for delistings
        delisted_today: set[int] = set()
        for i, sid in enumerate(available_ids):
            r = ret_row.get(sid, np.nan)
            sid_int = int(sid)
            if pd.isna(r) and w_aligned[i] > 1e-6:
                delisted_today.add(sid_int)

        # Handle delistings
        if delisted_today:
            w_aligned, _, delisting_return = _handle_delistings_at_date(
                w_aligned,
                [int(s) for s in available_ids],
                delisted_today,
                exchange_codes,
                delisting_return_nyse_amex,
                delisting_return_nasdaq,
            )
            port_ret += delisting_return

        daily_returns.append(port_ret)
        days_since_rebalance += 1

        # Update weight drift
        if np.sum(w_aligned) > 1e-10:
            growth = np.array([
                w_aligned[i] * np.exp(ret_row.get(sid, 0.0) if pd.notna(ret_row.get(sid)) else 0.0)
                for i, sid in enumerate(available_ids)
            ])

            # VALIDATION: Growth factors must be finite and non-negative
            assert_growth_finite(growth, f"benchmark_growth_{date_str}")

            total_growth = np.sum(growth)
            if total_growth > 1e-10:
                w_aligned = growth / total_growth

        w_current = w_aligned
        current_universe = available_ids

        # Skip if buy-and-hold
        if rebalancing_frequency_days == 0:
            continue

        # Check for scheduled rebalancing
        if days_since_rebalance >= rebalancing_frequency_days:
            n_scheduled += 1

            # Re-fit and re-optimize benchmark
            try:
                # Get trailing returns for re-estimation
                start_idx = max(0, t - d_eps_lookback)
                trailing_returns = returns_oos.iloc[start_idx:t][current_universe]

                # Get trailing vol at rebalancing date
                train_vol = trailing_vol.loc[:date, [int(s) for s in current_universe]].copy()
                train_vol.columns = pd.Index(current_universe)

                benchmark.fit(
                    trailing_returns,
                    current_universe,
                    trailing_vol=train_vol,
                    current_date=date_str,
                )

                w_old_for_tc = w_current.copy()
                w_new = benchmark.optimize(w_old=w_current, is_first=False)

                # Compute turnover and TC
                turnover = 0.5 * np.sum(np.abs(w_new - w_old_for_tc))
                tc = turnover * (tc_bps / 10000.0)
                total_turnover += 2.0 * turnover
                total_tc += tc

                # Validate benchmark rebalanced weights
                assert_weights_sum_to_one(w_new, f"benchmark_w_new_{date_str}")

                w_current = w_new
                days_since_rebalance = 0

                rebalancing_events.append(RebalancingEvent(
                    date=date_str,
                    trigger="scheduled",
                    turnover=turnover,
                    transaction_cost=tc,
                    entropy_before=0.0,
                    entropy_after=0.0,
                    n_delisted=len(delisted_today),
                    n_active=int(np.sum(w_new > 1e-6)),
                ))

            except Exception as e:
                logger.warning("Benchmark rebalancing failed at %s: %s", date_str, e)
                days_since_rebalance = 0  # Reset counter to avoid retry storm

    # Final results
    daily_returns_arr = np.array(daily_returns)
    cumulative = float(np.exp(np.sum(daily_returns_arr))) - 1.0

    return OOSRebalancingResult(
        daily_returns=daily_returns_arr,
        cumulative_return=cumulative,
        cumulative_turnover=total_turnover,
        total_transaction_cost=total_tc,
        n_scheduled_rebalances=n_scheduled,
        n_exceptional_rebalances=0,  # Benchmarks don't have exceptional triggers
        entropy_trajectory=[],  # Benchmarks don't track entropy
        rebalancing_events=rebalancing_events,
        final_weights=w_current,
        final_universe=[int(s) for s in current_universe],
    )
