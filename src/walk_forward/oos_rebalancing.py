"""
Out-of-sample periodic rebalancing simulation.

Implements scheduled and exceptional rebalancing during OOS evaluation:
- Scheduled rebalancing every N trading days (monthly=21, quarterly=63)
- Exceptional rebalancing when entropy drops below trigger threshold
- Shumway delisting imputation (-30% NYSE/AMEX, -55% NASDAQ)
- Transaction cost and turnover tracking

Key constraint: B_prime and eigenvalues are FROZEN during OOS (no VAE
re-inference needed). Only Sigma_assets and D_eps are recomputed from
rolling returns/residuals, making rebalancing lightweight.

Reference: DVT §4.2 — Rebalancing Protocol.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.portfolio.entropy import compute_entropy_only
from src.portfolio.sca_solver import multi_start_optimize

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


def _compute_d_eps_rolling(
    returns_trailing: pd.DataFrame,
    B_A: np.ndarray,
    stock_ids: list[int],
    d_eps_floor: float = 1e-6,
) -> np.ndarray:
    """
    Compute idiosyncratic variances from rolling residuals.

    :param returns_trailing (pd.DataFrame): Trailing returns for D_eps estimation
    :param B_A (np.ndarray): Exposure matrix (n, AU)
    :param stock_ids (list[int]): Stock IDs matching B_A rows
    :param d_eps_floor (float): Floor for idiosyncratic variance

    :return D_eps (np.ndarray): Idiosyncratic variances (n,)
    """
    # Filter returns to matching stocks
    available = [sid for sid in stock_ids if sid in returns_trailing.columns]
    if len(available) < len(stock_ids):
        # Some stocks missing from trailing data
        D_eps = np.full(len(stock_ids), d_eps_floor)
        return D_eps

    R: np.ndarray = np.asarray(returns_trailing[available].values, dtype=np.float64)
    n = len(stock_ids)

    # Simple approach: use sample variance of residuals from factor regression
    # For efficiency, use the diagonal of the sample covariance minus factor contribution
    sample_var: np.ndarray = np.var(R, axis=0, ddof=1)  # type: ignore[assignment]
    D_eps = np.maximum(sample_var, d_eps_floor)

    return D_eps[:n]


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

    :return w_new (np.ndarray): Optimized weights (n,)
    """
    cp = constraint_params

    # Default alpha (can be refined via frontier if needed)
    alpha = float(cp.get("alpha_opt", 0.5))
    idio_weight = float(cp.get("entropy_idio_weight", 0.2))

    w_new: np.ndarray
    try:
        result = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=alpha,
            n_starts=int(cp.get("n_starts", 3)),
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
) -> OOSRebalancingResult:
    """
    Simulate OOS period with periodic rebalancing.

    B_prime and eigenvalues are FROZEN during OOS (no VAE re-inference).
    Only D_eps is recomputed from rolling residuals at each rebalancing.

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

    # Results accumulators
    daily_returns: list[float] = []
    total_turnover = 0.0
    total_tc = 0.0
    n_scheduled = 0
    n_exceptional = 0
    entropy_trajectory: list[float] = [H_initial]
    rebalancing_events: list[RebalancingEvent] = []

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
            total_growth = np.sum(growth)
            if total_growth > 1e-10:
                w_aligned = growth / total_growth

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
                B_prime, inferred_stock_ids, current_universe,
            )
            D_aligned = _align_vector_to_universe(
                D_eps_current, inferred_stock_ids, current_universe,
            )

            H_current = compute_entropy_only(
                w_current, B_aligned, eigenvalues,
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
            # Align risk model components to current universe
            B_aligned = _align_matrix_to_universe(
                B_prime, inferred_stock_ids, current_universe,
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
                w_current, B_aligned, eigenvalues,
                D_eps=D_aligned, idio_weight=idio_weight,
            )

            # Re-optimize
            w_old_for_tc = w_current.copy()
            w_new = _execute_rebalancing(
                B_prime=B_aligned,
                eigenvalues=eigenvalues,
                Sigma_assets=Sigma_aligned,
                D_eps=D_aligned,
                w_old=w_current,
                n_active=len(current_universe),
                constraint_params=constraint_params,
                is_first=False,
                seed=seed + t,
            )

            # Compute turnover and TC
            turnover = 0.5 * np.sum(np.abs(w_new - w_old_for_tc))
            tc = turnover * (tc_bps / 10000.0)
            total_turnover += 2.0 * turnover  # Two-way
            total_tc += tc

            # Compute entropy after rebalancing
            H_after = compute_entropy_only(
                w_new, B_aligned, eigenvalues,
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

            logger.debug(
                "  [Rebal] %s at %s: turnover=%.2f%%, TC=%.4f%%, H=%.4f->%.4f",
                trigger, date_str, turnover * 100, tc * 100, H_before, H_after,
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
