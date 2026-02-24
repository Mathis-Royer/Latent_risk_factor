"""
Diagnostic experiments for systematic component substitution and parameter sweeps.

Implements Tier 0-2 experiments from the V2 post-mortem experimental protocol:
- Tier 0: Component substitution (PCA oracle, diagonal Sigma_z, EW/MinVar baselines)
- Tier 1: Factor quality profiling (CS R², effective rank, random baseline)
- Tier 2: Parameter sensitivity sweeps (sca_tol, phi, shrinkage, VT, signal count)

All experiments reuse existing pipeline components via checkpoint replay,
requiring NO retraining (~30s per experiment instead of ~3h).

Reference: docs/strategic_paths_post_v2.md, Section 5 (Experimental Protocol).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.integration.pipeline_state import load_run_data
from src.risk_model.covariance import assemble_risk_model, estimate_d_eps, estimate_sigma_z
from src.risk_model.factor_regression import compute_residuals, estimate_factor_returns
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VT_PRIOR_STRENGTH = 10.0  # Same as notebook_helpers.py


# ---------------------------------------------------------------------------
# Tier 0: Data loading
# ---------------------------------------------------------------------------


def load_experiment_data(run_dir: str | Path) -> dict[str, Any]:
    """
    Load all data needed for experiments from a completed diagnostic run.

    Wraps pipeline_state.load_run_data() and adds convenience accessors.

    :param run_dir (str | Path): Path to diagnostic run folder

    :return data (dict): Keys: B_A, B_full, stock_ids, weights, diagnostics,
            run_dir, pca_loadings, pca_eigenvalues, literature_comparison, AU
    """
    run_dir_str = str(run_dir)
    data = load_run_data(run_dir_str)

    # Ensure critical keys are present
    missing = []
    for key in ("B_A", "stock_ids", "weights"):
        if key not in data or data[key] is None:
            missing.append(key)
    if missing:
        raise ValueError(
            f"Checkpoint at {run_dir_str} missing critical data: {missing}. "
            f"Ensure the run completed at least through PORTFOLIO_DONE stage."
        )

    return data


# ---------------------------------------------------------------------------
# Tier 0: PCA oracle
# ---------------------------------------------------------------------------


def compute_pca_loadings(
    returns: pd.DataFrame,
    stock_ids: list[int],
    k_max: int = 30,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute PCA loadings for oracle substitution experiment (T0.1).

    Uses SVD on centered returns and Bai-Ng IC2 for factor count selection.

    :param returns (pd.DataFrame): Log-returns DataFrame (dates x stocks)
    :param stock_ids (list[int]): Stock IDs to include
    :param k_max (int): Maximum factors for Bai-Ng IC2

    :return B_PCA (np.ndarray): PCA loadings matrix (n, k_star)
    :return eigenvalues_pca (np.ndarray): PCA eigenvalues (k_star,)
    :return k_star (int): Bai-Ng optimal factor count
    """
    # Filter to available stocks
    available = [s for s in stock_ids if s in returns.columns]
    R = returns[available].dropna(how="all").values.astype(np.float64)
    T, n = R.shape

    # Impute remaining NaN with 0 (standard for factor models)
    R = np.nan_to_num(R, nan=0.0)

    # Center
    R_centered = R - R.mean(axis=0, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
    eigenvalues_all = (S ** 2) / T  # Factor variances

    # Bai-Ng IC2 for factor count selection
    k_star = _bai_ng_ic2(R_centered, eigenvalues_all, T, n, k_max)
    k_star = max(1, k_star)

    # Loadings = first k_star right singular vectors (scaled)
    B_PCA = Vt[:k_star].T  # (n, k_star)
    eigenvalues_pca = eigenvalues_all[:k_star]

    logger.info(
        "PCA: T=%d, n=%d, k_star=%d (Bai-Ng IC2), top eigenvalue=%.4f",
        T, n, k_star, eigenvalues_pca[0],
    )

    return B_PCA, eigenvalues_pca, k_star


def _bai_ng_ic2(
    R_centered: np.ndarray,
    eigenvalues: np.ndarray,
    T: int,
    n: int,
    k_max: int = 30,
) -> int:
    """
    Bai & Ng (2002) Information Criterion IC2 for factor selection.

    IC2(k) = ln(V(k)) + k * ((n+T)/(n*T)) * ln(min(n,T))

    :param R_centered (np.ndarray): Centered returns (T, n)
    :param eigenvalues (np.ndarray): Eigenvalues from SVD (S^2/T)
    :param T (int): Number of time periods
    :param n (int): Number of stocks
    :param k_max (int): Maximum factors to consider

    :return k_star (int): Optimal factor count
    """
    k_max = min(k_max, min(T, n) - 1, len(eigenvalues))
    total_var = float(np.sum(R_centered ** 2)) / (n * T)
    penalty_coef = ((n + T) / (n * T)) * np.log(min(n, T))

    best_ic = np.inf
    best_k = 1

    for k in range(1, k_max + 1):
        # Residual variance
        explained = float(np.sum(eigenvalues[:k]))
        V_k = max(total_var - explained / n, 1e-15)
        ic_k = np.log(V_k) + k * penalty_coef
        if ic_k < best_ic:
            best_ic = ic_k
            best_k = k

    return best_k


# ---------------------------------------------------------------------------
# Tier 0: Risk model with overrides
# ---------------------------------------------------------------------------


def build_risk_model_with_overrides(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    stock_ids: list[int],
    config: PipelineConfig,
    train_start: str,
    train_end: str,
    *,
    B_A_override: np.ndarray | None = None,
    force_diagonal_sigma_z: bool = False,
    sigma_z_shrinkage_override: str | None = None,
    disable_vt: bool = False,
    disable_market_intercept: bool = False,
    eigenvalue_pct_override: float | None = None,
    use_all_factors_for_entropy: bool = False,
) -> dict[str, Any]:
    """
    Build complete risk model with optional component overrides.

    Follows the same pipeline as notebook_helpers.replay_oos_simulation()
    but allows substitution at any step for controlled experiments.

    :param B_A (np.ndarray): VAE exposure matrix (n, AU)
    :param returns (pd.DataFrame): Full returns DataFrame
    :param trailing_vol (pd.DataFrame): Trailing volatilities
    :param stock_ids (list[int]): Ordered stock IDs matching B_A rows
    :param config (PipelineConfig): Pipeline configuration
    :param train_start (str): Training start date (YYYY-MM-DD)
    :param train_end (str): Training end date (YYYY-MM-DD)
    :param B_A_override (np.ndarray | None): Replace VAE B_A with this (e.g. PCA)
    :param force_diagonal_sigma_z (bool): Force Sigma_z to diagonal
    :param sigma_z_shrinkage_override (str | None): Override shrinkage method
    :param disable_vt (bool): Disable variance targeting
    :param disable_market_intercept (bool): Override market intercept setting
    :param eigenvalue_pct_override (float | None): Override eigenvalue truncation
    :param use_all_factors_for_entropy (bool): Use all AU factors (no signal split)

    :return result (dict): Keys: Sigma_assets, eigenvalues_full, eigenvalues_signal,
            B_prime_signal, D_eps_port, n_signal, z_hat, condition_number,
            B_A_used, n_port
    """
    rm = config.risk_model
    B_A_used = B_A_override if B_A_override is not None else B_A
    AU = B_A_used.shape[1]

    t0 = time.monotonic()

    # Build universe snapshots
    train_ret_sub = returns.loc[train_start:train_end]
    train_date_strs = [str(d.date()) if hasattr(d, "date") else str(d)
                       for d in train_ret_sub.index]
    universe_snapshots: dict[str, list[int]] = {}
    for d in train_date_strs:
        row = train_ret_sub.loc[d]
        valid_ids = [
            sid for sid in stock_ids
            if sid in row.index and pd.notna(row[sid])
        ]
        universe_snapshots[d] = valid_ids if valid_ids else stock_ids

    # Dual rescaling
    B_A_by_date = rescale_estimation(
        B_A_used, trailing_vol.loc[train_start:train_end],
        universe_snapshots, stock_ids,
        percentile_bounds=(rm.winsorize_lo, rm.winsorize_hi),
    )
    B_A_port = rescale_portfolio(
        B_A_used, trailing_vol, train_end,
        stock_ids, stock_ids,
        percentile_bounds=(rm.winsorize_lo, rm.winsorize_hi),
    )

    # Market intercept
    use_intercept = rm.market_intercept and not disable_market_intercept
    if use_intercept:
        for date_str in B_A_by_date:
            n_active = B_A_by_date[date_str].shape[0]
            B_A_by_date[date_str] = np.hstack([
                B_A_by_date[date_str],
                np.ones((n_active, 1), dtype=np.float64),
            ])
        B_A_port = np.hstack([
            B_A_port,
            np.ones((B_A_port.shape[0], 1), dtype=np.float64),
        ])

    # Factor regression
    z_hat, valid_dates = estimate_factor_returns(
        B_A_by_date, returns.loc[train_start:train_end],
        universe_snapshots,
        conditioning_threshold=rm.conditioning_threshold,
        ridge_scale=rm.ridge_scale,
        use_wls=rm.use_wls,
    )

    # Estimation/holdout split for VT
    n_z = z_hat.shape[0]
    n_vt_holdout = max(20, int(n_z * 0.2))
    n_est = n_z - n_vt_holdout
    z_hat_est = z_hat[:n_est]
    valid_dates_est = valid_dates[:n_est]

    # Residuals + covariance
    residuals = compute_residuals(
        B_A_by_date, z_hat_est,
        returns.loc[train_start:train_end],
        universe_snapshots, valid_dates_est, stock_ids,
    )

    shrinkage = sigma_z_shrinkage_override or rm.sigma_z_shrinkage
    eig_pct = eigenvalue_pct_override if eigenvalue_pct_override is not None else rm.sigma_z_eigenvalue_pct

    Sigma_z, n_signal, _ = estimate_sigma_z(
        z_hat_est,
        eigenvalue_pct=eig_pct,
        shrinkage_method=shrinkage,
        ewma_half_life=rm.sigma_z_ewma_half_life,
    )

    if force_diagonal_sigma_z:
        Sigma_z = np.diag(np.diag(Sigma_z))

    D_eps = estimate_d_eps(residuals, stock_ids, d_eps_floor=rm.d_eps_floor)
    n_port = B_A_port.shape[0]
    D_eps_port = D_eps[:n_port]

    # Assembly
    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
    Sigma_assets = risk_model["Sigma_assets"]
    eigenvalues = risk_model["eigenvalues"]
    B_prime_port = risk_model["B_prime_port"]
    V = risk_model["V"]

    # Signal/noise split
    if use_all_factors_for_entropy:
        n_signal_eff = len(eigenvalues)
        signal_start = 0
    else:
        n_signal = max(1, min(n_signal, len(eigenvalues)))
        skip_pc1 = use_intercept
        signal_start = 1 if skip_pc1 else 0
        signal_end = min(n_signal + (1 if skip_pc1 else 0), len(eigenvalues))
        n_signal_eff = signal_end - signal_start

    signal_end = signal_start + n_signal_eff
    eigenvalues_signal = eigenvalues[signal_start:signal_end].copy()
    B_prime_signal = B_prime_port[:, signal_start:signal_end].copy()

    # Variance targeting
    if not disable_vt and n_vt_holdout >= 20:
        z_hat_principal = z_hat @ V
        z_holdout_principal = z_hat_principal[n_est:]

        realized_var = np.var(z_holdout_principal, axis=0, ddof=1)
        pred_var = np.maximum(eigenvalues, 1e-15)
        vt_raw = realized_var / pred_var
        n_holdout_vt = z_holdout_principal.shape[0]
        vt_shrunk = (
            n_holdout_vt * vt_raw + _VT_PRIOR_STRENGTH * 1.0
        ) / (n_holdout_vt + _VT_PRIOR_STRENGTH)
        vt_factors = np.clip(vt_shrunk, rm.vt_clamp_min, rm.vt_clamp_max)
        eigenvalues = eigenvalues * vt_factors
        eigenvalues_signal = eigenvalues[signal_start:signal_end].copy()

        # Idiosyncratic VT
        residuals_holdout = compute_residuals(
            B_A_by_date, z_hat[n_est:],
            returns.loc[train_start:train_end],
            universe_snapshots, valid_dates[n_est:], stock_ids,
        )
        idio_ratios: list[float] = []
        for i, sid in enumerate(stock_ids[:n_port]):
            resids_h = residuals_holdout.get(sid, [])
            if len(resids_h) >= 10 and D_eps_port[i] > 1e-12:
                realized_idio = float(np.var(resids_h, ddof=1))
                idio_ratios.append(realized_idio / float(D_eps_port[i]))
        if len(idio_ratios) >= 10:
            vt_idio_raw = float(np.median(idio_ratios))
            n_idio_obs = len(idio_ratios)
            vt_idio = float(np.clip(
                (n_idio_obs * vt_idio_raw + _VT_PRIOR_STRENGTH) /
                (n_idio_obs + _VT_PRIOR_STRENGTH),
                rm.vt_clamp_min, rm.vt_clamp_max,
            ))
        else:
            vt_idio = 1.0
        D_eps_port = np.maximum(D_eps_port * vt_idio, rm.d_eps_floor)

        # Reassemble with VT
        Sigma_sys_vt = B_prime_port @ np.diag(eigenvalues) @ B_prime_port.T
        Sigma_assets = Sigma_sys_vt + np.diag(D_eps_port)
        Sigma_assets = 0.5 * (Sigma_assets + Sigma_assets.T)

    # Condition number
    eigvals_assets = np.linalg.eigvalsh(Sigma_assets)
    eigvals_positive = eigvals_assets[eigvals_assets > 0]
    condition_number = float(eigvals_positive[-1] / eigvals_positive[0]) if len(eigvals_positive) > 0 else np.inf

    elapsed = time.monotonic() - t0
    logger.info(
        "Risk model built in %.1fs: AU=%d, n_signal=%d, cond=%.1f",
        elapsed, AU, n_signal_eff, condition_number,
    )

    return {
        "Sigma_assets": Sigma_assets,
        "eigenvalues_full": eigenvalues,
        "eigenvalues_signal": eigenvalues_signal,
        "B_prime_signal": B_prime_signal,
        "B_prime_port": B_prime_port,
        "D_eps_port": D_eps_port,
        "n_signal": n_signal_eff,
        "z_hat": z_hat,
        "condition_number": condition_number,
        "B_A_used": B_A_used,
        "n_port": n_port,
        "V": V,
    }


# ---------------------------------------------------------------------------
# Tier 0: Portfolio from risk model
# ---------------------------------------------------------------------------


def run_portfolio_from_risk_model(
    risk_model: dict[str, Any],
    config: PipelineConfig,
    *,
    equal_weight_only: bool = False,
    min_variance_only: bool = False,
    phi_override: float | None = None,
    sca_tol_override: float | None = None,
    sca_max_iter_override: int | None = None,
    n_starts_override: int | None = None,
    alpha_grid_override: list[float] | None = None,
) -> dict[str, Any]:
    """
    Run portfolio optimization from a pre-built risk model.

    Supports overrides for controlled parameter experiments.

    :param risk_model (dict): Output from build_risk_model_with_overrides()
    :param config (PipelineConfig): Pipeline configuration
    :param equal_weight_only (bool): Return 1/n weights (skip optimizer)
    :param min_variance_only (bool): Solve min-variance (skip entropy)
    :param phi_override (float | None): Override concentration penalty
    :param sca_tol_override (float | None): Override SCA tolerance
    :param sca_max_iter_override (int | None): Override SCA max iterations
    :param n_starts_override (int | None): Override number of starts
    :param alpha_grid_override (list[float] | None): Override alpha grid

    :return result (dict): Keys: w_opt, alpha_opt, H_opt, variance_opt,
            n_active, solver_stats, frontier
    """
    from src.portfolio.entropy import compute_entropy_only
    from src.portfolio.frontier import (
        compute_adaptive_enb_target,
        compute_variance_entropy_frontier,
        select_operating_alpha,
    )
    from src.portfolio.sca_solver import multi_start_optimize

    n_port = risk_model["n_port"]
    Sigma_assets = risk_model["Sigma_assets"]
    eigenvalues_signal = risk_model["eigenvalues_signal"]
    B_prime_signal = risk_model["B_prime_signal"]
    D_eps_port = risk_model["D_eps_port"]
    pc = config.portfolio

    # Equal-weight baseline
    if equal_weight_only:
        w_ew = np.ones(n_port) / n_port
        H_ew = compute_entropy_only(
            w_ew, B_prime_signal, eigenvalues_signal, pc.entropy_eps,
            D_eps=D_eps_port, idio_weight=pc.entropy_idio_weight,
        )
        var_ew = float(w_ew @ Sigma_assets @ w_ew)
        return {
            "w_opt": w_ew, "alpha_opt": 0.0, "H_opt": H_ew,
            "variance_opt": var_ew, "n_active": n_port,
            "solver_stats": {"method": "equal_weight"}, "frontier": None,
        }

    # Min-variance baseline
    if min_variance_only:
        w_mv = _solve_min_variance(Sigma_assets, pc.w_max)
        H_mv = compute_entropy_only(
            w_mv, B_prime_signal, eigenvalues_signal, pc.entropy_eps,
            D_eps=D_eps_port, idio_weight=pc.entropy_idio_weight,
        )
        var_mv = float(w_mv @ Sigma_assets @ w_mv)
        return {
            "w_opt": w_mv, "alpha_opt": 0.0, "H_opt": H_mv,
            "variance_opt": var_mv, "n_active": int(np.sum(w_mv > 1e-6)),
            "solver_stats": {"method": "min_variance"}, "frontier": None,
        }

    # Full entropy frontier
    phi = phi_override if phi_override is not None else pc.phi
    sca_tol = sca_tol_override if sca_tol_override is not None else pc.sca_tol
    sca_max_iter = sca_max_iter_override if sca_max_iter_override is not None else pc.sca_max_iter
    n_starts = n_starts_override if n_starts_override is not None else pc.n_starts
    alpha_grid = alpha_grid_override if alpha_grid_override is not None else pc.alpha_grid

    # Target ENB
    n_signal = risk_model["n_signal"]
    target_enb = pc.target_enb
    if target_enb == 0.0 and n_signal > 0:
        target_enb = compute_adaptive_enb_target(eigenvalues_signal, n_signal)

    # Inverse-vol warm start
    diag_sigma = np.diag(Sigma_assets)
    inv_vol = 1.0 / np.sqrt(np.maximum(diag_sigma, 1e-15))
    inv_vol_w = inv_vol / inv_vol.sum()
    inv_vol_w = np.clip(inv_vol_w, 0.0, pc.w_max)
    inv_vol_w /= inv_vol_w.sum()

    frontier, weights_by_alpha, stats_by_alpha = compute_variance_entropy_frontier(
        Sigma_assets=Sigma_assets,
        B_prime=B_prime_signal,
        eigenvalues=eigenvalues_signal,
        D_eps=D_eps_port,
        alpha_grid=alpha_grid,
        lambda_risk=pc.lambda_risk,
        w_max=pc.w_max,
        w_min=pc.w_min,
        w_bar=pc.w_bar,
        phi=phi,
        is_first=True,
        n_starts=n_starts,
        max_iter=sca_max_iter,
        seed=config.seed,
        entropy_eps=pc.entropy_eps,
        idio_weight=pc.entropy_idio_weight,
        target_enb=target_enb,
        coarse_grid=pc.frontier_coarse_grid,
        early_stop_patience=pc.frontier_early_stop_patience,
        n_starts_after_target=pc.frontier_n_starts_after_target,
        max_iter_after_target=pc.frontier_max_iter_after_target,
        refine_enabled=pc.frontier_refine_enabled,
        refine_points=pc.frontier_refine_points,
        n_starts_refine=pc.frontier_n_starts_refine,
        max_iter_refine=pc.frontier_max_iter_refine,
        initial_warm_start_w=inv_vol_w,
    )

    alpha_opt = select_operating_alpha(frontier, target_enb=target_enb)
    w_opt_or_none = weights_by_alpha.get(alpha_opt)
    w_opt: np.ndarray = w_opt_or_none if w_opt_or_none is not None else inv_vol_w

    H_opt = compute_entropy_only(
        w_opt, B_prime_signal, eigenvalues_signal, pc.entropy_eps,
        D_eps=D_eps_port, idio_weight=pc.entropy_idio_weight,
    )
    var_opt = float(w_opt @ Sigma_assets @ w_opt)
    solver_stats = stats_by_alpha.get(alpha_opt, {})

    return {
        "w_opt": w_opt, "alpha_opt": alpha_opt, "H_opt": H_opt,
        "variance_opt": var_opt, "n_active": int(np.sum(w_opt > 1e-6)),
        "solver_stats": solver_stats, "frontier": frontier,
    }


def _solve_min_variance(
    Sigma: np.ndarray,
    w_max: float = 0.05,
) -> np.ndarray:
    """
    Solve constrained minimum variance: min w^T Sigma w s.t. sum(w)=1, 0<=w<=w_max.

    :param Sigma (np.ndarray): Covariance matrix (n, n)
    :param w_max (float): Maximum weight per stock

    :return w (np.ndarray): Optimal weights (n,)
    """
    n = Sigma.shape[0]
    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * Sigma @ w

    x0 = np.ones(n) / n
    bounds = [(0.0, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.maximum(result.x, 0.0)
    w /= w.sum()
    return w


# ---------------------------------------------------------------------------
# Tier 0: OOS metrics
# ---------------------------------------------------------------------------


def compute_oos_metrics(
    w: np.ndarray,
    returns_oos: pd.DataFrame,
    stock_ids: list[int],
    transaction_cost_bps: float = 10.0,
) -> dict[str, float]:
    """
    Compute standard OOS performance metrics for a weight vector.

    Simple buy-and-hold (no rebalancing) for fast comparison.

    :param w (np.ndarray): Portfolio weights (n,)
    :param returns_oos (pd.DataFrame): OOS returns (dates x stocks)
    :param stock_ids (list[int]): Stock IDs matching w
    :param transaction_cost_bps (float): One-way transaction cost in bps

    :return metrics (dict): Keys: sharpe, ann_return, ann_vol, max_drawdown,
            calmar, cumulative_return, n_active, herfindahl
    """
    available = [s for s in stock_ids if s in returns_oos.columns]
    n_avail = len(available)
    w_active = w[:n_avail]

    R_oos = np.nan_to_num(np.asarray(returns_oos[available].values, dtype=np.float64), nan=0.0)
    port_returns = R_oos @ w_active
    n_days = len(port_returns)

    if n_days == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "max_drawdown": 1.0, "calmar": 0.0, "cumulative_return": 0.0,
                "n_active": 0, "herfindahl": 1.0}

    # Cumulative
    cum_log = np.cumsum(port_returns)
    cumulative_return = float(np.exp(cum_log[-1]) - 1.0)

    # Annualized
    ann_return = float(np.exp(cum_log[-1]) ** (252.0 / n_days) - 1.0) if n_days > 0 else 0.0
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0
    sharpe = ann_return / max(ann_vol, 1e-10)

    # Max drawdown
    cum_wealth = np.exp(cum_log)
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = (running_max - cum_wealth) / np.maximum(running_max, 1e-15)
    max_drawdown = float(np.max(drawdowns))

    calmar = ann_return / max(max_drawdown, 1e-10)

    # Portfolio concentration
    n_active = int(np.sum(w_active > 1e-6))
    herfindahl = float(np.sum(w_active ** 2))

    return {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "cumulative_return": cumulative_return,
        "n_active": n_active,
        "herfindahl": herfindahl,
    }


# ---------------------------------------------------------------------------
# Tier 1: Factor quality profiling
# ---------------------------------------------------------------------------


def compute_factor_quality_profile(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    stock_ids: list[int],
    n_random_trials: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Tier 1: Comprehensive factor quality profiling.

    Computes multiple metrics that characterize B_A quality independently
    of the risk model and solver.

    :param B_A (np.ndarray): Exposure matrix (n, AU)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param stock_ids (list[int]): Stock IDs matching B_A rows
    :param n_random_trials (int): Number of random baseline trials
    :param seed (int): Random seed for baseline

    :return profile (dict): Keys: cs_r2, cs_r2_std, cs_r2_by_date,
            random_baseline_r2, random_baseline_std,
            effective_rank, condition_number,
            factor_autocorr, factor_autocorr_mean,
            top_1_eigenvalue_pct, top_3_eigenvalue_pct,
            singular_values
    """
    available = [s for s in stock_ids if s in returns.columns]
    sid_to_idx = {sid: i for i, sid in enumerate(stock_ids)}

    # CS R² computation
    cs_r2_list: list[float] = []
    z_hat_list: list[np.ndarray] = []

    R = returns[available].values
    avail_rows = np.array([sid_to_idx[s] for s in available])
    B_sub = B_A[avail_rows]  # (n_avail, AU)

    for t in range(R.shape[0]):
        r_t = R[t]
        valid_mask = np.isfinite(r_t)
        if valid_mask.sum() < B_A.shape[1] + 1:
            continue

        r_valid = r_t[valid_mask]
        B_valid = B_sub[valid_mask]

        # OLS: z_hat = (B^T B)^-1 B^T r
        z_hat_t, residuals_t, _, _ = np.linalg.lstsq(B_valid, r_valid, rcond=None)
        r_pred = B_valid @ z_hat_t
        ss_res = float(np.sum((r_valid - r_pred) ** 2))
        ss_tot = float(np.sum((r_valid - r_valid.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        cs_r2_list.append(r2)
        z_hat_list.append(z_hat_t)

    cs_r2 = float(np.mean(cs_r2_list)) if cs_r2_list else 0.0
    cs_r2_std = float(np.std(cs_r2_list)) if cs_r2_list else 0.0

    # Random baseline R²
    rng = np.random.RandomState(seed)
    random_r2_list: list[float] = []
    for _ in range(n_random_trials):
        B_random = rng.randn(*B_A.shape)
        B_random_sub = B_random[avail_rows]
        # Compute R² for a sample of dates
        sample_indices = rng.choice(R.shape[0], size=min(50, R.shape[0]), replace=False)
        trial_r2s: list[float] = []
        for t_idx in sample_indices:
            r_t = R[t_idx]
            valid_mask = np.isfinite(r_t)
            if valid_mask.sum() < B_A.shape[1] + 1:
                continue
            r_valid = r_t[valid_mask]
            B_valid = B_random_sub[valid_mask]
            z_hat_t, _, _, _ = np.linalg.lstsq(B_valid, r_valid, rcond=None)
            r_pred = B_valid @ z_hat_t
            ss_res = float(np.sum((r_valid - r_pred) ** 2))
            ss_tot = float(np.sum((r_valid - r_valid.mean()) ** 2))
            trial_r2s.append(1.0 - ss_res / max(ss_tot, 1e-15))
        if trial_r2s:
            random_r2_list.append(float(np.mean(trial_r2s)))

    random_baseline_r2 = float(np.mean(random_r2_list)) if random_r2_list else 0.0
    random_baseline_std = float(np.std(random_r2_list)) if random_r2_list else 0.0

    # Singular value analysis of B_A
    _, S, _ = np.linalg.svd(B_A, full_matrices=False)
    S_sq = S ** 2
    S_sq_norm = S_sq / max(S_sq.sum(), 1e-15)

    # Effective rank = exp(entropy of normalized singular values)
    S_sq_safe = S_sq_norm[S_sq_norm > 1e-30]
    effective_rank = float(np.exp(-np.sum(S_sq_safe * np.log(S_sq_safe))))

    # Condition number
    condition_number = float(S[0] / max(S[-1], 1e-15))

    # Eigenvalue concentration
    top_1_pct = float(S_sq_norm[0]) if len(S_sq_norm) > 0 else 1.0
    top_3_pct = float(np.sum(S_sq_norm[:3])) if len(S_sq_norm) >= 3 else 1.0

    # Factor autocorrelation (from z_hat time series)
    factor_autocorr = np.zeros(B_A.shape[1])
    if len(z_hat_list) > 10:
        z_hat_mat = np.array(z_hat_list)  # (n_dates, AU)
        for k in range(min(z_hat_mat.shape[1], B_A.shape[1])):
            z_k = z_hat_mat[:, k]
            if np.std(z_k) > 1e-10 and len(z_k) > 2:
                factor_autocorr[k] = float(np.corrcoef(z_k[:-1], z_k[1:])[0, 1])
    factor_autocorr_mean = float(np.mean(np.abs(factor_autocorr)))

    logger.info(
        "Factor quality: CS R²=%.2f%% (random=%.2f%%), eff_rank=%.1f, "
        "cond=%.1f, top-1 eig=%.1f%%",
        cs_r2 * 100, random_baseline_r2 * 100, effective_rank,
        condition_number, top_1_pct * 100,
    )

    return {
        "cs_r2": cs_r2,
        "cs_r2_std": cs_r2_std,
        "cs_r2_by_date": cs_r2_list,
        "random_baseline_r2": random_baseline_r2,
        "random_baseline_std": random_baseline_std,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "factor_autocorr": factor_autocorr,
        "factor_autocorr_mean": factor_autocorr_mean,
        "top_1_eigenvalue_pct": top_1_pct,
        "top_3_eigenvalue_pct": top_3_pct,
        "singular_values": S,
    }


# ---------------------------------------------------------------------------
# Tier 2: Parameter sweeps
# ---------------------------------------------------------------------------


def run_parameter_sweep(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    stock_ids: list[int],
    config: PipelineConfig,
    train_start: str,
    train_end: str,
    returns_oos: pd.DataFrame,
    param_name: str,
    param_values: list[Any],
) -> pd.DataFrame:
    """
    Tier 2: Sweep one parameter across values, record full metrics.

    :param B_A (np.ndarray): Exposure matrix (n, AU)
    :param returns (pd.DataFrame): Full returns
    :param trailing_vol (pd.DataFrame): Trailing volatilities
    :param stock_ids (list[int]): Stock IDs matching B_A
    :param config (PipelineConfig): Base configuration
    :param train_start (str): Training start date
    :param train_end (str): Training end date
    :param returns_oos (pd.DataFrame): OOS returns
    :param param_name (str): Parameter name (e.g. "phi", "sca_tol", "sigma_z_shrinkage")
    :param param_values (list): Values to sweep

    :return results (pd.DataFrame): Columns: param_value, sharpe, ann_return,
            ann_vol, max_drawdown, n_signal, n_active, condition_number
    """
    # Map param_name to risk model vs portfolio override
    RISK_MODEL_PARAMS = {
        "sigma_z_shrinkage", "force_diagonal_sigma_z",
        "disable_vt", "eigenvalue_pct", "use_all_factors",
        "disable_market_intercept",
    }
    PORTFOLIO_PARAMS = {"phi", "sca_tol", "sca_max_iter", "n_starts"}

    records: list[dict[str, Any]] = []

    for val in param_values:
        t0 = time.monotonic()

        # Build risk model with override
        rm_kwargs: dict[str, Any] = {}
        port_kwargs: dict[str, Any] = {}

        if param_name == "sigma_z_shrinkage":
            rm_kwargs["sigma_z_shrinkage_override"] = val
        elif param_name == "force_diagonal_sigma_z":
            rm_kwargs["force_diagonal_sigma_z"] = val
        elif param_name == "disable_vt":
            rm_kwargs["disable_vt"] = val
        elif param_name == "eigenvalue_pct":
            rm_kwargs["eigenvalue_pct_override"] = val
        elif param_name == "use_all_factors":
            rm_kwargs["use_all_factors_for_entropy"] = val
        elif param_name == "disable_market_intercept":
            rm_kwargs["disable_market_intercept"] = val
        elif param_name == "phi":
            port_kwargs["phi_override"] = val
        elif param_name == "sca_tol":
            port_kwargs["sca_tol_override"] = val
        elif param_name == "sca_max_iter":
            port_kwargs["sca_max_iter_override"] = val
        elif param_name == "n_starts":
            port_kwargs["n_starts_override"] = val
        else:
            logger.warning("Unknown param_name: %s", param_name)
            continue

        try:
            risk_model = build_risk_model_with_overrides(
                B_A, returns, trailing_vol, stock_ids, config,
                train_start, train_end, **rm_kwargs,
            )
            portfolio = run_portfolio_from_risk_model(
                risk_model, config, **port_kwargs,
            )
            oos_metrics = compute_oos_metrics(
                portfolio["w_opt"], returns_oos, stock_ids,
                config.portfolio.transaction_cost_bps,
            )

            elapsed = time.monotonic() - t0
            records.append({
                "param_value": val,
                "sharpe": oos_metrics["sharpe"],
                "ann_return": oos_metrics["ann_return"],
                "ann_vol": oos_metrics["ann_vol"],
                "max_drawdown": oos_metrics["max_drawdown"],
                "n_signal": risk_model["n_signal"],
                "n_active": portfolio["n_active"],
                "condition_number": risk_model["condition_number"],
                "alpha_opt": portfolio["alpha_opt"],
                "H_opt": portfolio["H_opt"],
                "elapsed_s": elapsed,
            })
            logger.info(
                "  %s=%s: Sharpe=%.3f, n_signal=%d, n_active=%d (%.1fs)",
                param_name, val, oos_metrics["sharpe"],
                risk_model["n_signal"], portfolio["n_active"], elapsed,
            )
        except Exception as e:
            logger.error("  %s=%s FAILED: %s", param_name, val, e)
            records.append({
                "param_value": val, "sharpe": np.nan,
                "ann_return": np.nan, "ann_vol": np.nan,
                "max_drawdown": np.nan, "n_signal": 0,
                "n_active": 0, "condition_number": np.nan,
                "alpha_opt": np.nan, "H_opt": np.nan,
                "elapsed_s": time.monotonic() - t0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tier 0: Run all component substitution experiments
# ---------------------------------------------------------------------------


def run_all_tier0(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    stock_ids: list[int],
    config: PipelineConfig,
    train_start: str,
    train_end: str,
    returns_oos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all Tier 0 component substitution experiments.

    Experiments:
      T0.1: Oracle PCA substitution (replace B_A with B_PCA)
      T0.2: Diagonal Sigma_z (remove cross-factor correlations)
      T0.3: Equal-weight baseline (remove optimizer)
      T0.4: Min-variance (replace entropy with min-var)
      T0.5: phi=0 (remove concentration penalty)
      T0.6: All factors for entropy (no signal/noise split)

    :param B_A (np.ndarray): VAE exposure matrix (n, AU)
    :param returns (pd.DataFrame): Full returns
    :param trailing_vol (pd.DataFrame): Trailing volatilities
    :param stock_ids (list[int]): Stock IDs matching B_A
    :param config (PipelineConfig): Pipeline configuration
    :param train_start (str): Training start date
    :param train_end (str): Training end date
    :param returns_oos (pd.DataFrame): OOS returns

    :return comparison (pd.DataFrame): Rows = experiments, columns = metrics
    """
    results: list[dict[str, Any]] = []
    tc_bps = config.portfolio.transaction_cost_bps

    def _add_result(name: str, risk_model: dict, portfolio: dict) -> None:
        oos = compute_oos_metrics(portfolio["w_opt"], returns_oos, stock_ids, tc_bps)
        results.append({
            "experiment": name,
            "sharpe": oos["sharpe"],
            "ann_return": oos["ann_return"],
            "ann_vol": oos["ann_vol"],
            "max_drawdown": oos["max_drawdown"],
            "n_signal": risk_model["n_signal"],
            "n_active": portfolio["n_active"],
            "condition_number": risk_model["condition_number"],
            "alpha_opt": portfolio.get("alpha_opt", 0.0),
            "H_opt": portfolio.get("H_opt", 0.0),
        })

    # Baseline: VAE with current config
    logger.info("T0.0: Baseline (VAE + current config)")
    rm_base = build_risk_model_with_overrides(
        B_A, returns, trailing_vol, stock_ids, config, train_start, train_end,
    )
    port_base = run_portfolio_from_risk_model(rm_base, config)
    _add_result("T0.0_baseline", rm_base, port_base)

    # T0.1: Oracle PCA substitution
    logger.info("T0.1: Oracle PCA substitution")
    try:
        B_PCA, eig_pca, k_star = compute_pca_loadings(
            returns.loc[train_start:train_end], stock_ids,
        )
        rm_pca = build_risk_model_with_overrides(
            B_A, returns, trailing_vol, stock_ids, config, train_start, train_end,
            B_A_override=B_PCA,
        )
        port_pca = run_portfolio_from_risk_model(rm_pca, config)
        _add_result("T0.1_pca_oracle", rm_pca, port_pca)
    except Exception as e:
        logger.error("T0.1 FAILED: %s", e)
        results.append({"experiment": "T0.1_pca_oracle", "sharpe": np.nan})

    # T0.2: Diagonal Sigma_z
    logger.info("T0.2: Diagonal Sigma_z")
    try:
        rm_diag = build_risk_model_with_overrides(
            B_A, returns, trailing_vol, stock_ids, config, train_start, train_end,
            force_diagonal_sigma_z=True,
        )
        port_diag = run_portfolio_from_risk_model(rm_diag, config)
        _add_result("T0.2_diagonal_sigma_z", rm_diag, port_diag)
    except Exception as e:
        logger.error("T0.2 FAILED: %s", e)
        results.append({"experiment": "T0.2_diagonal_sigma_z", "sharpe": np.nan})

    # T0.3: Equal-weight baseline
    logger.info("T0.3: Equal-weight baseline")
    try:
        port_ew = run_portfolio_from_risk_model(rm_base, config, equal_weight_only=True)
        _add_result("T0.3_equal_weight", rm_base, port_ew)
    except Exception as e:
        logger.error("T0.3 FAILED: %s", e)
        results.append({"experiment": "T0.3_equal_weight", "sharpe": np.nan})

    # T0.4: Min-variance
    logger.info("T0.4: Min-variance")
    try:
        port_mv = run_portfolio_from_risk_model(rm_base, config, min_variance_only=True)
        _add_result("T0.4_min_variance", rm_base, port_mv)
    except Exception as e:
        logger.error("T0.4 FAILED: %s", e)
        results.append({"experiment": "T0.4_min_variance", "sharpe": np.nan})

    # T0.5: phi=0 (no concentration penalty)
    logger.info("T0.5: phi=0")
    try:
        port_nophi = run_portfolio_from_risk_model(rm_base, config, phi_override=0.0)
        _add_result("T0.5_phi_zero", rm_base, port_nophi)
    except Exception as e:
        logger.error("T0.5 FAILED: %s", e)
        results.append({"experiment": "T0.5_phi_zero", "sharpe": np.nan})

    # T0.6: All factors for entropy (no signal/noise split)
    logger.info("T0.6: All factors for entropy")
    try:
        rm_allfac = build_risk_model_with_overrides(
            B_A, returns, trailing_vol, stock_ids, config, train_start, train_end,
            use_all_factors_for_entropy=True,
        )
        port_allfac = run_portfolio_from_risk_model(rm_allfac, config)
        _add_result("T0.6_all_factors", rm_allfac, port_allfac)
    except Exception as e:
        logger.error("T0.6 FAILED: %s", e)
        results.append({"experiment": "T0.6_all_factors", "sharpe": np.nan})

    return pd.DataFrame(results)


def run_all_tier1(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    stock_ids: list[int],
    n_random_trials: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run all Tier 1 factor quality profiling.

    :param B_A (np.ndarray): Exposure matrix (n, AU)
    :param returns (pd.DataFrame): Log-returns
    :param stock_ids (list[int]): Stock IDs matching B_A
    :param n_random_trials (int): Number of random baseline trials
    :param seed (int): Random seed

    :return results (dict): Factor quality profile + decision gate
    """
    profile = compute_factor_quality_profile(
        B_A, returns, stock_ids, n_random_trials, seed,
    )

    # Decision gate
    cs_r2 = profile["cs_r2"]
    random_r2 = profile["random_baseline_r2"]

    if random_r2 >= cs_r2:
        decision = "CRITICAL: VAE learned NOTHING useful cross-sectionally"
        decision_code = "no_signal"
    elif cs_r2 < 0.05:
        decision = "Factor model is broken (CS R² < 5%)"
        decision_code = "broken"
    elif cs_r2 < 0.15:
        decision = "Factor model is weak (5% <= CS R² < 15%), test downstream fixes"
        decision_code = "weak"
    else:
        decision = "Factor model is acceptable (CS R² >= 15%)"
        decision_code = "acceptable"

    profile["decision"] = decision
    profile["decision_code"] = decision_code

    logger.info("Tier 1 decision: %s (CS R²=%.2f%%, random=%.2f%%)",
                decision_code, cs_r2 * 100, random_r2 * 100)

    return profile


def run_all_tier2(
    B_A: np.ndarray,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    stock_ids: list[int],
    config: PipelineConfig,
    train_start: str,
    train_end: str,
    returns_oos: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Run all Tier 2 parameter sweeps.

    :param B_A (np.ndarray): Exposure matrix
    :param returns (pd.DataFrame): Full returns
    :param trailing_vol (pd.DataFrame): Trailing volatilities
    :param stock_ids (list[int]): Stock IDs
    :param config (PipelineConfig): Base configuration
    :param train_start (str): Training start date
    :param train_end (str): Training end date
    :param returns_oos (pd.DataFrame): OOS returns

    :return sweeps (dict): Mapping sweep_name -> DataFrame of results
    """
    sweeps: dict[str, pd.DataFrame] = {}

    def _sweep(param_name: str, param_values: list[Any]) -> pd.DataFrame:
        return run_parameter_sweep(
            B_A=B_A, returns=returns, trailing_vol=trailing_vol,
            stock_ids=stock_ids, config=config,
            train_start=train_start, train_end=train_end,
            returns_oos=returns_oos,
            param_name=param_name, param_values=param_values,
        )

    # T2.1: SCA tolerance
    logger.info("T2.1: SCA tolerance sweep")
    sweeps["sca_tol"] = _sweep("sca_tol", [1e-3, 1e-4, 1e-5, 1e-6])

    # T2.2: Phi (concentration penalty)
    logger.info("T2.2: Phi sweep")
    sweeps["phi"] = _sweep("phi", [0.0, 1.0, 5.0, 10.0, 15.0])

    # T2.3: Sigma_z shrinkage method
    logger.info("T2.3: Shrinkage method sweep")
    sweeps["sigma_z_shrinkage"] = _sweep(
        "sigma_z_shrinkage", ["spiked", "analytical_nonlinear", "truncation"],
    )

    # T2.4: VT clamping
    logger.info("T2.4: VT clamping sweep")
    sweeps["disable_vt"] = _sweep("disable_vt", [False, True])

    # T2.5: Signal/noise factor count
    logger.info("T2.5: Signal factor count sweep")
    sweeps["use_all_factors"] = _sweep("use_all_factors", [False, True])

    return sweeps
