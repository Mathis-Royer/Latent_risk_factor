"""
Comprehensive diagnostic collection for the full pipeline.

Gathers diagnostics from every pipeline stage (training, inference, risk model,
portfolio) and runs automated health checks with severity levels.

Used by scripts/run_diagnostic.py and notebooks/dashboard.ipynb.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.walk_forward.metrics import (
    factor_explanatory_power_dynamic,
    factor_explanatory_power_oos,
    realized_vs_predicted_correlation,
    realized_vs_predicted_variance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check thresholds
# ---------------------------------------------------------------------------

from src.config import VAEArchitectureConfig as _VAEArchConfig

_default_vae = _VAEArchConfig()

_THRESHOLDS = {
    "var_ratio_lo": 0.5,
    "var_ratio_hi": 2.0,
    "overfit_ratio_warn": 1.3,
    "overfit_ratio_crit": 1.8,
    "overfit_ratio_underfit": 0.85,
    "au_min_warn": 5,
    "au_min_crit": 2,
    "h_norm_min": 0.3,
    "ep_min_warn": 0.05,
    "ep_min_crit": 0.01,
    "sharpe_min": 0.0,
    "sigma_sq_min_default": _default_vae.sigma_sq_min,
    "sigma_sq_max_default": _default_vae.sigma_sq_max,
    "condition_number_warn": 1e6,
    "condition_number_crit": 1e8,
    "best_epoch_frac_lo": 0.15,
}


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def training_diagnostics(fit_result: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extract training convergence diagnostics from fit_result.

    :param fit_result (dict | None): Return value of VAETrainer.fit()

    :return diag (dict): Training diagnostic metrics
    """
    if fit_result is None:
        return {"available": False, "reason": "pretrained model (no training)"}

    history: list[dict[str, float]] = fit_result.get("history", [])
    n_epochs = len(history)

    if n_epochs == 0:
        return {"available": False, "reason": "empty training history"}

    best_epoch = fit_result.get("best_epoch", n_epochs)
    overfit_flag = fit_result.get("overfit_flag", False)
    overfit_ratio = fit_result.get("overfit_ratio", 1.0)
    best_val_elbo = fit_result.get("best_val_elbo", float("nan"))

    # Extract per-epoch series
    train_loss = [h.get("train_loss", float("nan")) for h in history]
    train_recon = [h.get("train_recon", float("nan")) for h in history]
    train_kl = [h.get("train_kl", float("nan")) for h in history]
    train_co = [h.get("train_co", float("nan")) for h in history]
    val_elbo = [h.get("val_elbo", float("nan")) for h in history]
    sigma_sq = [h.get("sigma_sq", float("nan")) for h in history]
    au_series = [h.get("AU", 0) for h in history]
    lr_series = [h.get("learning_rate", float("nan")) for h in history]

    # Convergence check: is val ELBO still decreasing at end?
    # Use val_elbo (not train_loss) because early stopping acts on val ELBO.
    # A declining train_loss with flat val ELBO signals overfitting, not
    # insufficient training.
    elbo_last_10 = val_elbo[-min(10, n_epochs):]
    still_decreasing = False
    # Filter NaN/Inf before regression (can occur if all batches fail)
    finite_pairs = [(i, v) for i, v in enumerate(elbo_last_10) if np.isfinite(v)]
    if len(finite_pairs) >= 5:
        _x = [p[0] for p in finite_pairs]
        _y = [p[1] for p in finite_pairs]
        regress_result = sp_stats.linregress(_x, _y)
        slope = float(regress_result[0])  # type: ignore[arg-type]
        p_value = float(regress_result[3])  # type: ignore[arg-type]
        still_decreasing = slope < 0 and p_value < 0.05

    # Sigma variance analysis
    sigma_sq_vals = [s for s in sigma_sq if np.isfinite(s)]
    sigma_initial = sigma_sq_vals[0] if sigma_sq_vals else float("nan")
    sigma_final = sigma_sq_vals[-1] if sigma_sq_vals else float("nan")
    sigma_min_hit = sigma_final <= _THRESHOLDS["sigma_sq_min_default"] * 1.01
    sigma_max_hit = sigma_final >= _THRESHOLDS["sigma_sq_max_default"] * 0.99

    # AU evolution
    au_initial = au_series[0] if au_series else 0
    au_final = au_series[-1] if au_series else 0
    au_max_during_training = max(au_series) if au_series else 0

    # Loss decomposition at best epoch
    best_idx = min(best_epoch - 1, n_epochs - 1)
    best_idx = max(0, best_idx)

    return {
        "available": True,
        "n_epochs": n_epochs,
        "best_epoch": best_epoch,
        "best_epoch_fraction": best_epoch / max(n_epochs, 1),
        "best_val_elbo": best_val_elbo,
        "overfit_flag": overfit_flag,
        "overfit_ratio": overfit_ratio,
        "still_decreasing_at_end": still_decreasing,
        # Loss series
        "train_loss": train_loss,
        "train_recon": train_recon,
        "train_kl": train_kl,
        "train_co": train_co,
        "val_elbo": val_elbo,
        # Final-epoch loss decomposition
        "final_loss": train_loss[-1] if train_loss else float("nan"),
        "final_recon": train_recon[-1] if train_recon else float("nan"),
        "final_kl": train_kl[-1] if train_kl else float("nan"),
        "final_co": train_co[-1] if train_co else float("nan"),
        # Best-epoch loss decomposition
        "best_loss": train_loss[best_idx],
        "best_recon": train_recon[best_idx],
        "best_kl": train_kl[best_idx],
        # Sigma evolution
        "sigma_sq_series": sigma_sq,
        "sigma_sq_initial": sigma_initial,
        "sigma_sq_final": sigma_final,
        "sigma_sq_min_hit": sigma_min_hit,
        "sigma_sq_max_hit": sigma_max_hit,
        # AU evolution
        "au_series": au_series,
        "au_initial": au_initial,
        "au_final": au_final,
        "au_max_during_training": au_max_during_training,
        # Learning rate
        "lr_series": lr_series,
        "lr_initial": lr_series[0] if lr_series else float("nan"),
        "lr_final": lr_series[-1] if lr_series else float("nan"),
        "n_lr_reductions": sum(
            1 for i in range(1, len(lr_series))
            if lr_series[i] < lr_series[i - 1]
        ),
    }


# ---------------------------------------------------------------------------
# Latent space diagnostics
# ---------------------------------------------------------------------------

def latent_diagnostics(state_bag: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze latent space properties from inference results.

    :param state_bag (dict): Pipeline state bag with B, B_A, kl_per_dim, etc.

    :return diag (dict): Latent space diagnostic metrics
    """
    kl_per_dim: np.ndarray = state_bag.get("kl_per_dim", np.array([]))
    B: np.ndarray = state_bag.get("B", np.array([]))
    B_A: np.ndarray = state_bag.get("B_A", np.array([]))
    AU: int = state_bag.get("AU", 0)
    active_dims: list[int] = state_bag.get("active_dims", [])
    K = B.shape[1] if B.ndim == 2 else 0

    # KL spectrum analysis
    kl_sorted = np.sort(kl_per_dim)[::-1] if len(kl_per_dim) > 0 else np.array([])
    kl_total = float(np.sum(kl_per_dim)) if len(kl_per_dim) > 0 else 0.0

    # Top-3 KL fraction: how much the 3 strongest dims dominate.
    # Replaces kl_concentration (sum(kl_active)/sum(kl_all)) which was always
    # ~1.0 by construction — active dims capture nearly all KL mass.
    kl_top3_fraction = (
        float(np.sum(kl_sorted[:3]) / max(kl_total, 1e-10))
        if len(kl_sorted) >= 3 else 1.0
    )

    # Effective number of latent dims (exponential entropy of KL distribution)
    if kl_total > 0:
        kl_probs = kl_per_dim / kl_total
        kl_probs = kl_probs[kl_probs > 1e-20]
        kl_entropy = float(-np.sum(kl_probs * np.log(kl_probs)))
        eff_latent_dims = float(np.exp(kl_entropy))
    else:
        kl_entropy = 0.0
        eff_latent_dims = 0.0

    # B matrix statistics
    b_stats: dict[str, Any] = {}
    if B.ndim == 2 and B.size > 0:
        b_norms_per_dim = np.linalg.norm(B, axis=0)
        b_norms_per_stock = np.linalg.norm(B, axis=1)
        b_stats = {
            "sparsity": float(np.mean(np.abs(B) < 1e-6)),
            "mean_dim_norm": float(np.mean(b_norms_per_dim)),
            "std_dim_norm": float(np.std(b_norms_per_dim)),
            "mean_stock_norm": float(np.mean(b_norms_per_stock)),
            "std_stock_norm": float(np.std(b_norms_per_stock)),
            "max_entry": float(np.max(np.abs(B))),
        }

    return {
        "K": K,
        "AU": AU,
        "utilization_ratio": AU / max(K, 1),
        "active_dims": active_dims,
        "kl_per_dim_sorted": kl_sorted.tolist(),
        "kl_total": kl_total,
        "kl_top3_fraction": kl_top3_fraction,
        "kl_entropy": kl_entropy,
        "eff_latent_dims": eff_latent_dims,
        "B_shape": list(B.shape) if B.ndim >= 2 else [],
        "B_A_shape": list(B_A.shape) if B_A.ndim >= 2 else [],
        "B_stats": b_stats,
    }


# ---------------------------------------------------------------------------
# Risk model diagnostics
# ---------------------------------------------------------------------------

def risk_model_diagnostics(
    state_bag: dict[str, Any],
    returns_oos: pd.DataFrame,
    w_vae: np.ndarray,
    inferred_stock_ids: list[int],
) -> dict[str, Any]:
    """
    Analyze risk model quality using realized vs predicted comparisons.

    :param state_bag (dict): Pipeline state bag with risk_model, B_A, z_hat, etc.
    :param returns_oos (pd.DataFrame): Out-of-sample returns
    :param w_vae (np.ndarray): VAE portfolio weights
    :param inferred_stock_ids (list[int]): Stock IDs matching weight vector

    :return diag (dict): Risk model diagnostic metrics
    """
    risk_model: dict[str, Any] = state_bag.get("risk_model", {})
    Sigma_assets: np.ndarray = risk_model.get("Sigma_assets", np.array([]))
    eigenvalues: np.ndarray = risk_model.get("eigenvalues", np.array([]))
    vt_scale_sys: float = state_bag.get("vt_scale_sys", 1.0)
    vt_scale_idio: float = state_bag.get("vt_scale_idio", 1.0)

    diag: dict[str, Any] = {
        "vt_scale_sys": vt_scale_sys,
        "vt_scale_idio": vt_scale_idio,
    }

    # Eigenvalue spectrum
    if eigenvalues.size > 0:
        total_eig = float(np.sum(eigenvalues))
        cumulative = np.cumsum(eigenvalues) / max(total_eig, 1e-10)
        diag["eigenvalues"] = eigenvalues.tolist()
        diag["n_eigenvalues"] = len(eigenvalues)
        diag["top_eigenvalue"] = float(eigenvalues[0])
        diag["top_3_explained"] = float(cumulative[min(2, len(cumulative) - 1)])
        diag["top_10_explained"] = float(cumulative[min(9, len(cumulative) - 1)])
        diag["eigenvalue_ratio_1_2"] = (
            float(eigenvalues[0] / max(eigenvalues[1], 1e-10))
            if len(eigenvalues) > 1 else float("inf")
        )

    # Condition number
    if Sigma_assets.ndim == 2 and Sigma_assets.shape[0] > 0:
        try:
            cond = float(np.linalg.cond(Sigma_assets))
        except np.linalg.LinAlgError:
            cond = float("inf")
        diag["condition_number"] = cond

    # Variance ratio and correlation
    # Use only stocks present in both the risk model and OOS returns, and
    # drop dates where any of these stocks have NaN (avoid fillna(0) bias).
    n_port = Sigma_assets.shape[0] if Sigma_assets.ndim == 2 else 0
    if n_port > 0 and returns_oos.shape[0] > 10:
        all_ids = inferred_stock_ids[:n_port]
        avail_mask = np.array([sid in returns_oos.columns for sid in all_ids])
        avail_idx = np.where(avail_mask)[0]
        if len(avail_idx) > 2:
            avail_ids = [all_ids[i] for i in avail_idx]
            oos_sub = returns_oos[avail_ids].dropna(how="any")
            if len(oos_sub) > 10:
                oos_vals = np.asarray(oos_sub.values)
                w_sub = w_vae[avail_idx]
                # Re-normalize weights to sum to 1 on the available subset
                w_sum = w_sub.sum()
                if w_sum > 1e-10:
                    w_sub = w_sub / w_sum
                Sigma_sub = Sigma_assets[np.ix_(avail_idx, avail_idx)]

                var_ratio = realized_vs_predicted_variance(w_sub, Sigma_sub, oos_vals)
                corr_rank = realized_vs_predicted_correlation(Sigma_sub, oos_vals)
                diag["var_ratio_oos"] = var_ratio
                diag["corr_rank_oos"] = corr_rank

    # In-sample explanatory power — uses estimation-rescaled B_A_by_date and
    # z_hat from training (matches the OLS that produced z_hat). This is
    # the training R² and should always be in [0, 1].
    B_A_by_date: dict[str, np.ndarray] = state_bag.get("B_A_by_date", {})
    z_hat: np.ndarray = state_bag.get("z_hat", np.array([]))
    valid_dates: list[str] = state_bag.get("valid_dates", [])
    universe_snapshots: dict[str, list[int]] = state_bag.get("universe_snapshots", {})
    train_returns: pd.DataFrame | None = state_bag.get("train_returns")
    if (
        B_A_by_date
        and z_hat.ndim == 2
        and z_hat.shape[0] > 0
        and valid_dates
        and train_returns is not None
    ):
        ep_is = factor_explanatory_power_dynamic(
            B_A_by_date, z_hat, train_returns,
            universe_snapshots, valid_dates,
        )
        diag["ep_in_sample"] = ep_is

    # Explanatory power — OOS cross-sectional R² using B_A_port
    # B_A_port is the portfolio-rescaled exposure from training; we run OLS on
    # OOS returns at each date to get proper OOS factor returns and R².
    # Pass returns WITHOUT fillna — factor_explanatory_power_oos handles NaN
    # per-date via nan_mask, which is more correct than replacing with 0.
    B_A_port: np.ndarray = state_bag.get("B_A_port", np.array([]))
    if B_A_port.ndim == 2 and B_A_port.shape[0] > 0:
        ep_result = factor_explanatory_power_oos(
            B_A_port,
            returns_oos.reindex(
                columns=inferred_stock_ids[:B_A_port.shape[0]],
            ),
            inferred_stock_ids[:B_A_port.shape[0]],
        )
        diag["explanatory_power"] = ep_result["ep_oos"]
        diag["ep_oos_n_dates"] = ep_result["n_dates"]
        diag["avg_cs_r2"] = ep_result["avg_cs_r2"]
        diag["z_hat_oos_std"] = ep_result["z_hat_oos_std"]

    # B_A scale diagnostics: detect magnitude issues in raw exposures
    B_A_raw: np.ndarray = state_bag.get("B_A", np.array([]))
    if B_A_raw.ndim == 2:
        diag["B_A_mean_abs"] = float(np.mean(np.abs(B_A_raw)))
        diag["B_A_std"] = float(np.std(B_A_raw))
        diag["B_A_max_abs"] = float(np.max(np.abs(B_A_raw)))
        # Column (factor) norms: if some factors have huge loadings, they
        # dominate the quadratic form in Sigma_assets
        col_norms = np.linalg.norm(B_A_raw, axis=0)
        diag["B_A_col_norm_mean"] = float(np.mean(col_norms))
        diag["B_A_col_norm_max"] = float(np.max(col_norms))

    return diag


# ---------------------------------------------------------------------------
# Portfolio diagnostics
# ---------------------------------------------------------------------------

def portfolio_diagnostics(
    state_bag: dict[str, Any],
    w_vae: np.ndarray,
    vae_metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Analyze portfolio construction quality.

    :param state_bag (dict): Pipeline state bag with alpha_opt, frontier, etc.
    :param w_vae (np.ndarray): VAE portfolio weights
    :param vae_metrics (dict): OOS portfolio metrics

    :return diag (dict): Portfolio diagnostic metrics
    """
    alpha_opt: float = state_bag.get("alpha_opt", 0.0)
    frontier_raw = state_bag.get("frontier", [])
    # frontier may be a pd.DataFrame or list of dicts
    if hasattr(frontier_raw, "to_dict"):
        frontier: list[dict[str, Any]] = frontier_raw.to_dict("records")  # type: ignore[union-attr]
    else:
        frontier = frontier_raw if isinstance(frontier_raw, list) else []
    risk_model: dict[str, Any] = state_bag.get("risk_model", {})
    eigenvalues: np.ndarray = risk_model.get("eigenvalues", np.array([]))
    B_prime_port: np.ndarray = risk_model.get("B_prime_port", np.array([]))

    # Weight statistics
    w_pos = w_vae[w_vae > 1e-8]
    n_active = len(w_pos)
    n_total = len(w_vae)
    eff_n = 1.0 / float(np.sum(w_vae ** 2)) if np.sum(w_vae ** 2) > 0 else 0.0

    # Gini coefficient
    sorted_w = np.sort(w_pos) if len(w_pos) > 0 else np.array([0.0])
    n_w = len(sorted_w)
    gini = 0.0
    if n_w > 1 and np.sum(sorted_w) > 0:
        cumulative = np.cumsum(sorted_w)
        gini = float(
            (n_w + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n_w
        )

    # HHI (Herfindahl-Hirschman Index)
    hhi = float(np.sum(w_vae ** 2))

    # Frontier analysis
    # NOTE: frontier records use keys from compute_variance_entropy_frontier():
    #   "alpha", "variance", "entropy", "n_active"
    frontier_diag: dict[str, Any] = {"available": len(frontier) > 0}
    if frontier:
        alphas = [f.get("alpha", 0) for f in frontier]
        h_values = [f.get("entropy", 0) for f in frontier]
        var_values = [f.get("variance", 0) for f in frontier]
        frontier_diag.update({
            "alpha_grid": alphas,
            "H_values": h_values,
            "variance_values": var_values,
            "H_at_alpha_opt": next(
                (f.get("entropy", 0) for f in frontier if f.get("alpha") == alpha_opt),
                0.0,
            ),
        })

    # Risk decomposition: factor contributions at optimal weights
    risk_decomp: dict[str, Any] = {"available": False}
    if eigenvalues.size > 0 and B_prime_port.size > 0 and len(w_vae) == B_prime_port.shape[0]:
        beta_prime = B_prime_port.T @ w_vae  # (AU,)
        contributions = beta_prime ** 2 * eigenvalues
        total = float(np.sum(contributions))
        if total > 1e-10:
            fractions = contributions / total
            risk_decomp = {
                "available": True,
                "contributions": contributions.tolist(),
                "fractions": fractions.tolist(),
                "top_1_fraction": float(fractions[0]) if len(fractions) > 0 else 0.0,
                "top_3_fraction": float(np.sum(fractions[:3])) if len(fractions) >= 3 else float(np.sum(fractions)),
                "entropy_H": float(
                    -np.sum(fractions[fractions > 1e-15] * np.log(fractions[fractions > 1e-15]))
                ),
                "max_entropy": float(np.log(max(len(fractions), 1))),
            }

    return {
        "alpha_opt": alpha_opt,
        "n_active_positions": n_active,
        "n_total_stocks": n_total,
        "eff_n_positions": eff_n,
        "gini_coefficient": gini,
        "hhi": hhi,
        "w_max": float(np.max(w_vae)) if len(w_vae) > 0 else 0.0,
        "w_min_active": float(np.min(w_pos)) if len(w_pos) > 0 else 0.0,
        "w_mean": float(np.mean(w_pos)) if len(w_pos) > 0 else 0.0,
        "w_std": float(np.std(w_pos)) if len(w_pos) > 0 else 0.0,
        # OOS metrics
        "sharpe": vae_metrics.get("sharpe", 0.0),
        "ann_return": vae_metrics.get("ann_return", 0.0),
        "ann_vol": vae_metrics.get("ann_vol_oos", 0.0),
        "max_drawdown": vae_metrics.get("max_drawdown_oos", 0.0),
        "H_norm_oos": vae_metrics.get("H_norm_oos", 0.0),
        "sortino": vae_metrics.get("sortino", 0.0),
        "calmar": vae_metrics.get("calmar", 0.0),
        # Frontier
        "frontier": frontier_diag,
        # Risk decomposition
        "risk_decomposition": risk_decomp,
    }


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def benchmark_comparison(
    vae_metrics: dict[str, float],
    benchmark_results: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """
    Compare VAE performance against all benchmarks.

    :param vae_metrics (dict): VAE fold metrics
    :param benchmark_results (dict): Benchmark name -> list of fold metrics

    :return comparison (dict): Head-to-head comparison analysis
    """
    compare_metrics = [
        "sharpe", "ann_return", "ann_vol_oos", "max_drawdown_oos",
        "H_norm_oos", "sortino", "calmar", "eff_n_positions",
        "diversification_ratio",
    ]
    # Direction: positive = higher is better, negative = lower is better
    higher_better = {
        "sharpe", "ann_return", "H_norm_oos", "sortino",
        "calmar", "eff_n_positions", "diversification_ratio",
    }

    per_benchmark: dict[str, dict[str, Any]] = {}
    total_wins = 0
    total_losses = 0
    total_ties = 0

    for bench_name, bench_results in benchmark_results.items():
        if not bench_results:
            continue
        # Aggregate across all folds using median (consistent with selection.py)
        bench_m: dict[str, float] = {}
        for m in compare_metrics:
            values = [r.get(m, float("nan")) for r in bench_results]
            valid = [v for v in values if np.isfinite(v)]
            bench_m[m] = float(np.median(valid)) if valid else float("nan")
        wins = 0
        losses = 0
        deltas: dict[str, float] = {}

        ties = 0
        for metric in compare_metrics:
            vae_val = vae_metrics.get(metric, float("nan"))
            bench_val = bench_m.get(metric, float("nan"))
            if np.isnan(vae_val) or np.isnan(bench_val):
                logger.debug(
                    "Skipping %s comparison for %s (vae=%s, bench=%s)",
                    metric, bench_name,
                    "NaN" if np.isnan(vae_val) else f"{vae_val:.4f}",
                    "NaN" if np.isnan(bench_val) else f"{bench_val:.4f}",
                )
                continue
            delta = vae_val - bench_val
            deltas[metric] = delta
            if metric in higher_better:
                if delta > 1e-6:
                    wins += 1
                elif delta < -1e-6:
                    losses += 1
                else:
                    ties += 1
            else:
                if delta < -1e-6:
                    wins += 1
                elif delta > 1e-6:
                    losses += 1
                else:
                    ties += 1

        per_benchmark[bench_name] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "deltas": deltas,
            "bench_metrics": {k: bench_m.get(k, float("nan")) for k in compare_metrics},
        }
        total_wins += wins
        total_losses += losses
        total_ties += ties

    return {
        "per_benchmark": per_benchmark,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_ties": total_ties,
        "vae_metrics": {k: vae_metrics.get(k, float("nan")) for k in compare_metrics},
    }


# ---------------------------------------------------------------------------
# Data quality diagnostics
# ---------------------------------------------------------------------------

def data_quality_diagnostics(
    stock_data: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze input data quality.

    :param stock_data (pd.DataFrame): Raw stock data (long format)
    :param returns (pd.DataFrame): Log-returns (dates x stocks)

    :return diag (dict): Data quality metrics
    """
    n_stocks = returns.shape[1]
    n_dates = returns.shape[0]
    missing_pct = float(returns.isna().mean().mean()) * 100

    date_range_start = str(returns.index[0])[:10] if n_dates > 0 else "N/A"
    date_range_end = str(returns.index[-1])[:10] if n_dates > 0 else "N/A"

    # Per-stock missing data
    missing_per_stock = returns.isna().mean(axis=0)
    stocks_over_20pct_missing = int((missing_per_stock > 0.20).sum())

    # Sector distribution (if available)
    sector_dist: dict[str, int] = {}
    if "sector" in stock_data.columns:
        sector_counts: pd.Series = stock_data.drop_duplicates("permno")["sector"].value_counts()  # type: ignore[assignment]
        sector_dist = sector_counts.to_dict()

    # Stocks per year — mean/min/max daily stock count grouped by year
    stocks_per_date: pd.Series = returns.notna().sum(axis=1)  # type: ignore[assignment]
    year_idx: pd.Series = pd.Series(  # type: ignore[type-arg]
        pd.DatetimeIndex(returns.index).year,  # type: ignore[reportAttributeAccessIssue]
        index=returns.index,
    )
    spy_mean: pd.Series = stocks_per_date.groupby(year_idx).mean()  # type: ignore[assignment]
    spy_min: pd.Series = stocks_per_date.groupby(year_idx).min()  # type: ignore[assignment]
    spy_max: pd.Series = stocks_per_date.groupby(year_idx).max()  # type: ignore[assignment]
    stocks_per_year: dict[str, Any] = {
        "years": [int(y) for y in spy_mean.index.tolist()],
        "mean": [float(v) for v in spy_mean.tolist()],
        "min": [float(v) for v in spy_min.tolist()],
        "max": [float(v) for v in spy_max.tolist()],
    }

    return {
        "n_stocks": n_stocks,
        "n_dates": n_dates,
        "date_range": f"{date_range_start} to {date_range_end}",
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "missing_pct": missing_pct,
        "stocks_over_20pct_missing": stocks_over_20pct_missing,
        "sector_distribution": sector_dist,
        "stocks_per_year": stocks_per_year,
        "years_of_data": n_dates / 252.0,
    }


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def run_health_checks(
    training: dict[str, Any],
    latent: dict[str, Any],
    risk_model: dict[str, Any],
    portfolio: dict[str, Any],
    data_quality: dict[str, Any],
    benchmark_comparison_data: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Run automated health checks across all diagnostic categories.

    :param training (dict): Training diagnostics
    :param latent (dict): Latent space diagnostics
    :param risk_model (dict): Risk model diagnostics
    :param portfolio (dict): Portfolio diagnostics
    :param data_quality (dict): Data quality diagnostics
    :param benchmark_comparison_data (dict | None): Benchmark comparison with per-benchmark metrics

    :return checks (list[dict]): List of {category, check, status, message}
        where status is "OK", "WARNING", or "CRITICAL"
    """
    checks: list[dict[str, str]] = []

    def _add(category: str, check: str, status: str, message: str) -> None:
        checks.append({
            "category": category,
            "check": check,
            "status": status,
            "message": message,
        })

    # --- Training checks ---
    if training.get("available", False):
        # Overfit / underfit
        or_val = training.get("overfit_ratio", 1.0)
        if or_val > _THRESHOLDS["overfit_ratio_crit"]:
            _add("Training", "Overfit ratio", "CRITICAL",
                 f"Overfit ratio = {or_val:.2f} > {_THRESHOLDS['overfit_ratio_crit']}")
        elif or_val > _THRESHOLDS["overfit_ratio_warn"]:
            _add("Training", "Overfit ratio", "WARNING",
                 f"Overfit ratio = {or_val:.2f} > {_THRESHOLDS['overfit_ratio_warn']}")
        elif or_val < _THRESHOLDS["overfit_ratio_underfit"]:
            _add("Training", "Overfit ratio", "WARNING",
                 f"Overfit ratio = {or_val:.2f} < {_THRESHOLDS['overfit_ratio_underfit']}"
                 " — possible underfitting")
        else:
            _add("Training", "Overfit ratio", "OK",
                 f"Overfit ratio = {or_val:.2f}")

        # Best epoch too early (underfitting)
        be_frac = training.get("best_epoch_fraction", 1.0)
        if be_frac < _THRESHOLDS["best_epoch_frac_lo"]:
            _add("Training", "Best epoch timing", "WARNING",
                 f"Best epoch at {be_frac:.0%} of training — possible underfitting")
        else:
            _add("Training", "Best epoch timing", "OK",
                 f"Best epoch at {be_frac:.0%} of training")

        # Still decreasing at end
        if training.get("still_decreasing_at_end", False):
            _add("Training", "Convergence", "WARNING",
                 "Val ELBO still decreasing at end — consider more epochs")
        else:
            _add("Training", "Convergence", "OK", "Training converged")

        # Sigma bounds
        if training.get("sigma_sq_min_hit", False):
            _add("Training", "sigma_sq bounds", "WARNING",
                 f"sigma_sq hit lower bound ({training.get('sigma_sq_final', 0):.2e})")
        elif training.get("sigma_sq_max_hit", False):
            _add("Training", "sigma_sq bounds", "WARNING",
                 f"sigma_sq hit upper bound ({training.get('sigma_sq_final', 0):.2e})")
        else:
            _add("Training", "sigma_sq bounds", "OK",
                 f"sigma_sq = {training.get('sigma_sq_final', 0):.4f}")

    # --- Latent space checks ---
    au = latent.get("AU", 0)
    k = latent.get("K", 0)
    if au < _THRESHOLDS["au_min_crit"]:
        _add("Latent", "Active units", "CRITICAL",
             f"AU = {au} < {_THRESHOLDS['au_min_crit']} — model collapsed")
    elif au < _THRESHOLDS["au_min_warn"]:
        _add("Latent", "Active units", "WARNING",
             f"AU = {au} — very few active dimensions")
    else:
        _add("Latent", "Active units", "OK",
             f"AU = {au} / K = {k} ({latent.get('utilization_ratio', 0):.1%} utilization)")

    # --- Risk model checks ---
    var_ratio = risk_model.get("var_ratio_oos", 1.0)
    if var_ratio < _THRESHOLDS["var_ratio_lo"] or var_ratio > _THRESHOLDS["var_ratio_hi"]:
        _add("Risk Model", "Variance ratio", "CRITICAL",
             f"var_ratio = {var_ratio:.3f} — outside [{_THRESHOLDS['var_ratio_lo']}, {_THRESHOLDS['var_ratio_hi']}]")
    else:
        _add("Risk Model", "Variance ratio", "OK",
             f"var_ratio = {var_ratio:.3f}")

    cond = risk_model.get("condition_number", 1.0)
    if cond > _THRESHOLDS["condition_number_crit"]:
        _add("Risk Model", "Condition number", "CRITICAL",
             f"cond(Sigma) = {cond:.0e} — severely ill-conditioned")
    elif cond > _THRESHOLDS["condition_number_warn"]:
        _add("Risk Model", "Condition number", "WARNING",
             f"cond(Sigma) = {cond:.0e}")
    else:
        _add("Risk Model", "Condition number", "OK",
             f"cond(Sigma) = {cond:.0e}")

    ep = risk_model.get("explanatory_power", 0.0)
    ep_is = risk_model.get("ep_in_sample", None)
    ep_detail = f"EP_oos = {ep:.4f}"
    if ep_is is not None:
        ep_detail += f", EP_is = {ep_is:.4f}"
    if ep < _THRESHOLDS["ep_min_crit"]:
        _add("Risk Model", "Explanatory power", "CRITICAL",
             f"{ep_detail} — factors explain almost nothing OOS")
    elif ep < _THRESHOLDS["ep_min_warn"]:
        _add("Risk Model", "Explanatory power", "WARNING",
             ep_detail)
    else:
        _add("Risk Model", "Explanatory power", "OK",
             ep_detail)

    # --- Portfolio checks ---
    sharpe = portfolio.get("sharpe", 0.0)
    if sharpe < _THRESHOLDS["sharpe_min"]:
        _add("Portfolio", "Sharpe ratio", "WARNING",
             f"Sharpe = {sharpe:.3f} — negative risk-adjusted return")
    else:
        _add("Portfolio", "Sharpe ratio", "OK",
             f"Sharpe = {sharpe:.3f}")

    h_norm = portfolio.get("H_norm_oos", 0.0)
    if h_norm < _THRESHOLDS["h_norm_min"]:
        _add("Portfolio", "Factor entropy", "WARNING",
             f"H_norm = {h_norm:.3f} — poor factor diversification")
    else:
        _add("Portfolio", "Factor entropy", "OK",
             f"H_norm = {h_norm:.3f}")

    mdd = portfolio.get("max_drawdown", 0.0)

    # Extract equal_weight MDD as market proxy for comparison
    ew_mdd: float | None = None
    if benchmark_comparison_data is not None:
        per_bench = benchmark_comparison_data.get("per_benchmark", {})
        ew_data = per_bench.get("equal_weight", {})
        ew_bench_metrics = ew_data.get("bench_metrics", {})
        ew_mdd_val = ew_bench_metrics.get("max_drawdown_oos", float("nan"))
        if np.isfinite(ew_mdd_val):
            ew_mdd = ew_mdd_val

    mdd_context = f"MDD = {mdd:.1%}"
    if ew_mdd is not None:
        mdd_context += f" (EW benchmark: {ew_mdd:.1%})"

    if mdd > 0.3:
        _add("Portfolio", "Max drawdown", "WARNING", mdd_context)
    else:
        _add("Portfolio", "Max drawdown", "OK", mdd_context)

    # --- Data quality checks ---
    miss = data_quality.get("missing_pct", 0.0)
    if miss > 10.0:
        _add("Data", "Missing data", "WARNING",
             f"{miss:.1f}% missing values")
    else:
        _add("Data", "Missing data", "OK",
             f"{miss:.1f}% missing values")

    n_stocks = data_quality.get("n_stocks", 0)
    if n_stocks < 30:
        _add("Data", "Universe size", "WARNING",
             f"Only {n_stocks} stocks — small universe")
    else:
        _add("Data", "Universe size", "OK",
             f"{n_stocks} stocks")

    return checks


# ---------------------------------------------------------------------------
# Master collector
# ---------------------------------------------------------------------------

def collect_diagnostics(
    state_bag: dict[str, Any],
    vae_metrics: dict[str, float],
    benchmark_results: dict[str, list[dict[str, float]]],
    returns_oos: pd.DataFrame,
    stock_data: pd.DataFrame,
    returns: pd.DataFrame,
    w_vae: np.ndarray,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Collect comprehensive diagnostics from all pipeline stages.

    :param state_bag (dict): Pipeline state bag from run_direct()
    :param vae_metrics (dict): VAE OOS metrics
    :param benchmark_results (dict): Benchmark name -> list of fold metrics
    :param returns_oos (pd.DataFrame): Out-of-sample returns
    :param stock_data (pd.DataFrame): Raw stock data
    :param returns (pd.DataFrame): Full log-returns
    :param w_vae (np.ndarray): VAE portfolio weights
    :param config_dict (dict): Pipeline configuration as dict

    :return diagnostics (dict): Complete diagnostic report
    """
    inferred_stock_ids = state_bag.get("inferred_stock_ids", [])

    logger.info("Collecting training diagnostics...")
    training = training_diagnostics(state_bag.get("fit_result"))

    logger.info("Collecting latent space diagnostics...")
    latent = latent_diagnostics(state_bag)

    logger.info("Collecting risk model diagnostics...")
    risk = risk_model_diagnostics(state_bag, returns_oos, w_vae, inferred_stock_ids)

    logger.info("Collecting portfolio diagnostics...")
    portfolio = portfolio_diagnostics(state_bag, w_vae, vae_metrics)

    logger.info("Collecting benchmark comparison...")
    bench_comp = benchmark_comparison(vae_metrics, benchmark_results)

    logger.info("Collecting data quality diagnostics...")
    data_qual = data_quality_diagnostics(stock_data, returns)

    logger.info("Running health checks...")
    checks = run_health_checks(training, latent, risk, portfolio, data_qual, bench_comp)

    n_critical = sum(1 for c in checks if c["status"] == "CRITICAL")
    n_warning = sum(1 for c in checks if c["status"] == "WARNING")
    n_ok = sum(1 for c in checks if c["status"] == "OK")
    logger.info(
        "Health checks complete: %d OK, %d WARNING, %d CRITICAL",
        n_ok, n_warning, n_critical,
    )

    return {
        "training": training,
        "latent": latent,
        "risk_model": risk,
        "portfolio": portfolio,
        "benchmark_comparison": bench_comp,
        "data_quality": data_qual,
        "health_checks": checks,
        "config": config_dict,
        "summary": {
            "n_critical": n_critical,
            "n_warning": n_warning,
            "n_ok": n_ok,
        },
    }
