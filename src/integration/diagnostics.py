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

from src.risk_model.factor_quality import (
    bai_ng_ic2,
    compute_factor_quality_dashboard,
    onatski_eigenvalue_ratio,
)
from src.walk_forward.metrics import (
    factor_explanatory_power_dynamic,
    factor_explanatory_power_oos,
    realized_vs_predicted_correlation,
    realized_vs_predicted_variance,
    eigenvector_rotation_stability,
    au_retention_analysis,
    deflated_sharpe_ratio,
)
from src.integration.composite_scores import compute_all_composite_scores
from src.integration.statistical_tests import compute_pbo

# Import new diagnostic modules
from src.diagnostics.factor_diagnostics import (
    analyze_exposure_norms,
    compute_eigenvalue_concentration,
    track_regression_quality,
    validate_dgj_recovery,
)
from src.diagnostics.vae_diagnostics import (
    analyze_reconstruction_temporal_structure,
    estimate_mutual_information,
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
    # Factor quality thresholds
    "au_bai_ng_diff_warn": 10,  # |AU - k_bai_ng| > 10 triggers warning
    "au_bai_ng_diff_crit": 20,
    "latent_stability_min": 0.85,  # Spearman rho between folds
    # Solver convergence thresholds
    "sca_grad_norm_warn": 1e-3,
    "sca_grad_norm_crit": 1e-2,
    "sca_converged_ratio_warn": 0.5,  # At least 50% of starts should converge
    # Constraint binding thresholds
    "binding_fraction_warn": 0.5,  # More than 50% at w_max is concerning
    # New diagnostic thresholds (G1-G9)
    "kl_collapsed_frac_warn": 0.5,  # G1: >50% dims with KL<0.01 before warmup
    "eigenvector_alignment_warn": 0.70,  # G3: mean alignment < 0.70
    "au_turnover_warn": 0.40,  # G6: mean turnover > 0.40
    "dsr_probability_warn": 0.25,  # G9: DSR < 0.25
    "recon_acf1_warn": 0.3,  # G5: temporal autocorrelation > 0.3
    "recon_boundary_ratio_warn": 1.5,  # G5: boundary ratio > 1.5
    "mutual_info_warn": 0.3,  # G2: total MI < 0.3 nats
    "dgj_noise_cv_warn": 0.2,  # G4: noise CV > 0.2
    "pbo_probability_warn": 0.5,  # G8: PBO > 0.5 indicates overfitting
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

    # Per-feature reconstruction losses (if available)
    recon_per_feature: list[list[float]] = []
    for h in history:
        rpf = h.get("recon_per_feature")
        if isinstance(rpf, list):
            recon_per_feature.append([float(v) for v in rpf])
        else:
            recon_per_feature.append([])
    has_per_feature = any(len(rpf) > 0 for rpf in recon_per_feature)

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
        # Per-feature reconstruction loss (return vs volatility)
        "has_per_feature_recon": has_per_feature,
        "recon_per_feature_series": recon_per_feature if has_per_feature else [],
        "recon_per_feature_final": recon_per_feature[-1] if has_per_feature and recon_per_feature[-1] else [],
        "recon_per_feature_best": recon_per_feature[best_idx] if has_per_feature and best_idx < len(recon_per_feature) and recon_per_feature[best_idx] else [],
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


def factor_quality_diagnostics(
    state_bag: dict[str, Any],
    returns: pd.DataFrame,
    inferred_stock_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Factor quality analysis: persistence, breadth, AU validation.

    Computes diagnostic metrics for factor characterization without
    affecting portfolio optimization (agnostic diversification).

    :param state_bag (dict): Pipeline state bag with B_A, eigenvalues, z_hat
    :param returns (pd.DataFrame): Historical returns for Bai-Ng IC2 comparison
    :param inferred_stock_ids (list[int] | None): Stock IDs matching B_A rows.
        Used to filter returns for consistent Bai-Ng comparison.

    :return diag (dict): Factor quality metrics and AU validation
    """
    B_A: np.ndarray = state_bag.get("B_A", np.array([]))
    risk_model: dict[str, Any] = state_bag.get("risk_model", {})
    eigenvalues: np.ndarray = risk_model.get("eigenvalues", np.array([]))
    z_hat: np.ndarray = state_bag.get("z_hat", np.array([]))
    stability_rho: float | None = state_bag.get("latent_stability_rho")

    if B_A.ndim != 2 or B_A.size == 0:
        return {"available": False, "reason": "B_A not available"}

    n_stocks, AU = B_A.shape

    # Prepare centered returns for Bai-Ng IC2
    # Filter to inferred stocks for consistent comparison with AU
    returns_centered: np.ndarray | None = None
    T_returns: int = 0
    if returns.shape[0] > 50 and returns.shape[1] > 10:
        # Filter columns to inferred stocks if IDs provided
        if inferred_stock_ids:
            avail_cols = [c for c in inferred_stock_ids if c in returns.columns]
            ret_filtered = returns[avail_cols] if avail_cols else returns
        else:
            ret_filtered = returns

        ret_clean = ret_filtered.dropna(axis=0, how="all").dropna(axis=1, how="all")
        T_returns = ret_clean.shape[0]
        if T_returns > 50 and ret_clean.shape[1] >= 10:
            R_mat = ret_clean.values.astype(np.float64)
            returns_centered = R_mat - R_mat.mean(axis=0, keepdims=True)

    # Factor returns for persistence (if available)
    factor_returns: np.ndarray | None = None
    if z_hat.ndim == 2 and z_hat.shape[0] > 10:
        factor_returns = z_hat

    # Compute factor quality dashboard
    dashboard = compute_factor_quality_dashboard(
        B_A=B_A,
        eigenvalues=eigenvalues if eigenvalues.size > 0 else np.ones(AU),
        factor_returns=factor_returns,
        returns_centered=returns_centered,
        stability_rho=stability_rho,
    )

    # Additional AU validation via direct Bai-Ng on returns
    k_bai_ng_direct: int | None = None
    if returns_centered is not None:
        k_bai_ng_direct = bai_ng_ic2(returns_centered, k_max=min(50, AU + 20))

    # Onatski test on eigenvalues
    # Use T_returns (from filtered returns) for consistent dimensions
    k_onatski: int | None = None
    onatski_ratio: float | None = None
    if eigenvalues.size >= 5:
        T_est = max(252, T_returns) if T_returns > 0 else max(252, returns.shape[0])
        k_onatski, onatski_ratio = onatski_eigenvalue_ratio(
            eigenvalues, n_stocks, T_est
        )

    return {
        "available": True,
        "AU": AU,
        "n_stocks": n_stocks,
        # Factor classification summary
        "n_structural": dashboard.get("n_structural", 0),
        "n_style": dashboard.get("n_style", 0),
        "n_episodic": dashboard.get("n_episodic", 0),
        "pct_structural": dashboard.get("pct_structural", 0.0),
        # Per-factor details (first 10 for brevity)
        "breadth_top10": dashboard.get("breadth", [])[:10],
        "half_lives_top10": dashboard.get("half_lives", [])[:10],
        "categories_top10": dashboard.get("categories", [])[:10],
        # AU validation
        "k_bai_ng": k_bai_ng_direct,
        "k_onatski": k_onatski,
        "onatski_ratio": onatski_ratio,
        "au_bai_ng_diff": AU - k_bai_ng_direct if k_bai_ng_direct is not None else None,
        "au_onatski_diff": AU - k_onatski if k_onatski is not None else None,
        # Stability
        "latent_stability_rho": stability_rho,
        "stability_ok": stability_rho is None or stability_rho > 0.85,
        # Eigenvalue gaps
        "max_gap_index": dashboard.get("max_gap_index", 0),
        "eigenvalue_gaps_top5": dashboard.get("eigenvalue_gaps", [])[:5],
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

    n_signal: int = state_bag.get("n_signal", 0)
    shrinkage_intensity: float | None = state_bag.get("shrinkage_intensity")

    diag: dict[str, Any] = {
        "vt_scale_sys": vt_scale_sys,
        "vt_scale_idio": vt_scale_idio,
        "n_signal": n_signal,
        "shrinkage_intensity": shrinkage_intensity,
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
    # Restrict to stocks with non-zero portfolio weight AND present in OOS
    # returns.  Uses fillna(0) to preserve all OOS days instead of dropna
    # which discards rows with any NaN (misleadingly perfect var_ratio on
    # a tiny subset of days).  Missing returns → 0 = flat for that stock.
    n_port = Sigma_assets.shape[0] if Sigma_assets.ndim == 2 else 0
    if n_port > 0 and returns_oos.shape[0] > 10:
        all_ids = inferred_stock_ids[:n_port]
        # Only consider stocks with meaningful weight (held positions)
        active_weight_mask = np.abs(w_vae[:n_port]) > 1e-8
        held_idx = np.where(active_weight_mask)[0]
        # Further filter to stocks present in OOS returns columns
        avail_idx = np.array(
            [i for i in held_idx if all_ids[i] in returns_oos.columns],
            dtype=int,
        )
        if len(avail_idx) > 2:
            avail_ids = [all_ids[i] for i in avail_idx]
            oos_raw = returns_oos[avail_ids]
            # Report coverage: fraction of non-NaN entries
            n_total_cells = oos_raw.shape[0] * oos_raw.shape[1]
            n_valid_cells = int(oos_raw.notna().sum().sum())  # type: ignore[arg-type]
            coverage = n_valid_cells / max(n_total_cells, 1)
            diag["var_ratio_coverage"] = coverage
            # Use fillna(0): missing return = flat for that stock-day
            oos_sub = oos_raw.fillna(0.0)
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
                if coverage < 0.5:
                    diag["var_ratio_oos"] = float("nan")
                    logger.warning(
                        "  var_ratio coverage=%.1f%% < 50%% — unreliable, "
                        "setting to NaN", coverage * 100,
                    )

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
    n_signal: int = state_bag.get("n_signal", 0)

    # Use signal-only eigenvalues/B_prime for entropy computation.
    # The pipeline stores signal-only slices (excluding market PC1 and
    # noise eigenvalues) separately from the full risk_model values.
    # ENB and H_norm_signal must be computed on diversifiable signal
    # factors only (Meucci 2009, "Managing Diversification").
    eigenvalues = state_bag.get("eigenvalues_signal", np.array([]))
    B_prime_port = state_bag.get("B_prime_signal", np.array([]))
    if eigenvalues.size == 0:
        eigenvalues = risk_model.get("eigenvalues", np.array([]))
    if B_prime_port.size == 0:
        B_prime_port = risk_model.get("B_prime_port", np.array([]))

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

    # Effective dimensionality from eigenvalue spectrum.
    # n_eff = exp(H_eigenvalue) where H_eigenvalue = -sum(p_k * ln(p_k))
    # with p_k = lambda_k / sum(lambda).  This gives the number of factors
    # that meaningfully contribute to risk, which is the correct denominator
    # for normalized entropy (not AU, which overcounts inactive dimensions).
    n_eff_eigenvalue: float = 0.0
    h_norm_eff: float = 0.0
    if eigenvalues.size > 0:
        eig_total = float(np.sum(eigenvalues))
        if eig_total > 1e-15:
            eig_probs = eigenvalues / eig_total
            eig_probs_pos = eig_probs[eig_probs > 1e-20]
            h_eig = float(-np.sum(eig_probs_pos * np.log(eig_probs_pos)))
            n_eff_eigenvalue = float(np.exp(h_eig))

    # Risk decomposition: factor contributions at optimal weights
    risk_decomp: dict[str, Any] = {"available": False}
    enb: float = 0.0
    h_norm_signal: float = 0.0
    if eigenvalues.size > 0 and B_prime_port.size > 0 and len(w_vae) == B_prime_port.shape[0]:
        beta_prime = B_prime_port.T @ w_vae  # (n_signal,)
        contributions = beta_prime ** 2 * eigenvalues
        total = float(np.sum(contributions))
        if total > 1e-10:
            fractions = contributions / total
            entropy_H = float(
                -np.sum(fractions[fractions > 1e-15] * np.log(fractions[fractions > 1e-15]))
            )
            enb = float(np.exp(entropy_H))
            max_entropy_au = float(np.log(max(len(fractions), 1)))
            # H_norm_signal: normalized by ln(n_signal) — the number of real
            # factors identified by DGJ spiked shrinkage. This is the correct
            # denominator after eigenvalue truncation (not AU which overcounts
            # noise dimensions, and not n_eff which double-counts spectral decay).
            if n_signal > 1:
                h_norm_signal = entropy_H / float(np.log(n_signal))
            elif n_eff_eigenvalue > 1.0:
                # Fallback when n_signal not available
                h_norm_signal = entropy_H / float(np.log(n_eff_eigenvalue))
            elif max_entropy_au > 0:
                h_norm_signal = entropy_H / max_entropy_au
            # Legacy h_norm_eff for backward compatibility
            if n_eff_eigenvalue > 1.0:
                h_norm_eff = entropy_H / float(np.log(n_eff_eigenvalue))
            elif max_entropy_au > 0:
                h_norm_eff = entropy_H / max_entropy_au
            risk_decomp = {
                "available": True,
                "contributions": contributions.tolist(),
                "fractions": fractions.tolist(),
                "top_1_fraction": float(fractions[0]) if len(fractions) > 0 else 0.0,
                "top_3_fraction": float(np.sum(fractions[:3])) if len(fractions) >= 3 else float(np.sum(fractions)),
                "entropy_H": entropy_H,
                "enb": enb,
                "max_entropy": max_entropy_au,
            }

    # G9: Deflated Sharpe Ratio (if OOS returns available for higher moments)
    dsr_result: dict[str, float] = {}
    oos_returns = state_bag.get("oos_returns_portfolio")  # Daily returns
    sharpe = vae_metrics.get("sharpe", 0.0)
    n_trials = state_bag.get("n_strategies_tested", 7)  # VAE + 6 benchmarks
    if oos_returns is not None and len(oos_returns) > 10:
        try:
            skew = float(sp_stats.skew(oos_returns))
            kurt = float(sp_stats.kurtosis(oos_returns, fisher=True))  # Excess kurtosis
            T = len(oos_returns)
            dsr_result = deflated_sharpe_ratio(sharpe, n_trials, T, skew, kurt)
        except Exception as e:
            logger.debug("Could not compute deflated Sharpe: %s", e)

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
        "sharpe": sharpe,
        "ann_return": vae_metrics.get("ann_return", 0.0),
        "ann_vol": vae_metrics.get("ann_vol_oos", 0.0),
        "max_drawdown": vae_metrics.get("max_drawdown_oos", 0.0),
        "H_norm_oos": vae_metrics.get("H_norm_oos", 0.0),
        "H_norm_eff": h_norm_eff,
        "H_norm_signal": h_norm_signal,
        "enb": enb,
        "n_signal": n_signal,
        "n_eff_eigenvalue": n_eff_eigenvalue,
        "sortino": vae_metrics.get("sortino", 0.0),
        "calmar": vae_metrics.get("calmar", 0.0),
        # G9: Deflated Sharpe Ratio
        "dsr_probability": dsr_result.get("dsr_probability", 0.0),
        "sharpe_deflated": dsr_result.get("sharpe_deflated", 0.0),
        # Frontier
        "frontier": frontier_diag,
        # Risk decomposition
        "risk_decomposition": risk_decomp,
        # Raw weights for health checks (non-trivial selection test)
        "_weights_raw": w_vae,
    }


# ---------------------------------------------------------------------------
# Solver diagnostics
# ---------------------------------------------------------------------------

def solver_diagnostics(state_bag: dict[str, Any]) -> dict[str, Any]:
    """
    Extract SCA solver convergence diagnostics from state_bag.

    :param state_bag (dict): Pipeline state bag with solver_stats

    :return diag (dict): Solver convergence metrics
    """
    solver_stats: dict[str, Any] | None = state_bag.get("solver_stats")

    if solver_stats is None:
        return {"available": False, "reason": "solver_stats not in state_bag"}

    n_starts = solver_stats.get("n_starts", 0)
    converged_count = solver_stats.get("converged_count", 0)
    iterations_list: list[int] = solver_stats.get("iterations", [])
    best_convergence_info: dict[str, Any] = solver_stats.get("best_convergence_info", {})
    best_start_idx = solver_stats.get("best_start_idx", 0)

    # Extract from best start
    best_converged = best_convergence_info.get("converged", False)
    best_grad_norm = best_convergence_info.get("final_grad_norm", float("nan"))
    best_step_sizes: list[float] = best_convergence_info.get("step_sizes", [])
    best_obj_improvements: list[float] = best_convergence_info.get("obj_improvements", [])

    # Compute summary statistics
    converged_ratio = converged_count / max(n_starts, 1)
    avg_iterations = float(np.mean(iterations_list)) if iterations_list else 0.0
    max_iterations = max(iterations_list) if iterations_list else 0

    return {
        "available": True,
        "n_starts": n_starts,
        "converged_count": converged_count,
        "converged_ratio": converged_ratio,
        "best_start_idx": best_start_idx,
        "best_converged": best_converged,
        "best_final_grad_norm": best_grad_norm,
        "best_n_iterations": iterations_list[best_start_idx] if best_start_idx < len(iterations_list) else 0,
        "iterations_list": iterations_list,
        "avg_iterations": avg_iterations,
        "max_iterations": max_iterations,
        # Convergence trajectory (for plotting)
        "best_step_sizes": best_step_sizes[-20:] if best_step_sizes else [],  # Last 20
        "best_obj_improvements": best_obj_improvements[-20:] if best_obj_improvements else [],
    }


# ---------------------------------------------------------------------------
# Constraint binding diagnostics
# ---------------------------------------------------------------------------

def constraint_binding_diagnostics(state_bag: dict[str, Any]) -> dict[str, Any]:
    """
    Extract constraint binding status from state_bag.

    :param state_bag (dict): Pipeline state bag with binding_status

    :return diag (dict): Constraint binding metrics
    """
    binding_status: dict[str, Any] | None = state_bag.get("binding_status")

    if binding_status is None:
        return {"available": False, "reason": "binding_status not in state_bag"}

    return {
        "available": True,
        "n_at_w_max": binding_status.get("n_at_w_max", 0),
        "n_at_w_min": binding_status.get("n_at_w_min", 0),
        "n_above_w_bar": binding_status.get("n_above_w_bar", 0),
        "w_max_binding": binding_status.get("w_max_binding", False),
        "tau_binding": binding_status.get("tau_binding", False),
        "actual_turnover": binding_status.get("actual_turnover", 0.0),
        "concentrated_weight": binding_status.get("concentrated_weight", 0.0),
        "binding_fraction": binding_status.get("binding_fraction", 0.0),
    }


# ---------------------------------------------------------------------------
# VAE Posterior Diagnostics (new diagnostic module integration)
# ---------------------------------------------------------------------------

def vae_posterior_diagnostics(
    fit_result: dict[str, Any] | None,
    state_bag: dict[str, Any],
) -> dict[str, Any]:
    """
    Collect VAE posterior quality diagnostics using new diagnostic modules.

    Integrates analyze_log_var_distribution and compute_kl_per_dimension
    from src.diagnostics.vae_diagnostics.

    :param fit_result (dict | None): Return value of VAETrainer.fit()
    :param state_bag (dict): Pipeline state bag

    :return diag (dict): VAE posterior diagnostics for composite scores
    """
    result: dict[str, Any] = {"available": False}

    # Get log_var stats from training history if available
    if fit_result is not None:
        history: list[dict[str, Any]] = fit_result.get("history", [])
        if history:
            # Get final epoch log_var stats
            last_entry = history[-1]
            log_var_stats = last_entry.get("log_var_stats", {})

            if log_var_stats:
                result["available"] = True
                result["collapse_severity"] = log_var_stats.get(
                    "global_frac_at_lower", 0.0
                )
                result["explosion_severity"] = log_var_stats.get(
                    "global_frac_at_upper", 0.0
                )
                result["n_collapsed"] = log_var_stats.get("n_collapsed", 0)
                result["n_exploded"] = log_var_stats.get("n_exploded", 0)

    # Get KL balance from latent diagnostics
    kl_per_dim = state_bag.get("kl_per_dim", np.array([]))
    if kl_per_dim.size > 0:
        kl_total = float(np.sum(kl_per_dim))
        if kl_total > 0:
            kl_probs = kl_per_dim / kl_total
            kl_probs_pos = kl_probs[kl_probs > 1e-20]
            kl_entropy = float(-np.sum(kl_probs_pos * np.log(kl_probs_pos)))
            K = len(kl_per_dim)
            max_entropy = np.log(K) if K > 1 else 1.0
            kl_balance = kl_entropy / max_entropy if max_entropy > 0 else 0.0
            result["available"] = True
            result["kl_balance"] = kl_balance
            result["kl_entropy"] = kl_entropy
        else:
            result["kl_balance"] = 0.5  # Neutral default
    else:
        result["kl_balance"] = 0.5

    # Get per-feature reconstruction loss from training
    if fit_result is not None:
        history = fit_result.get("history", [])
        if history:
            best_epoch = fit_result.get("best_epoch", len(history))
            best_idx = min(best_epoch - 1, len(history) - 1)
            best_idx = max(0, best_idx)
            rpf = history[best_idx].get("recon_per_feature", [])
            if isinstance(rpf, list) and len(rpf) >= 2:
                result["recon_per_feature"] = [float(v) for v in rpf]

    # G5: Reconstruction temporal structure (if reconstruction errors available)
    recon_errors = state_bag.get("reconstruction_errors")
    if recon_errors is not None and isinstance(recon_errors, np.ndarray) and recon_errors.ndim == 2:
        try:
            temporal_analysis = analyze_reconstruction_temporal_structure(recon_errors)
            result["recon_acf_1"] = temporal_analysis.get("acf_1", 0.0)
            result["recon_boundary_ratio"] = temporal_analysis.get("boundary_ratio", 1.0)
            result["available"] = True
        except Exception as e:
            logger.debug("Could not analyze temporal structure: %s", e)

    # G2: Mutual information (if latent samples and input data available)
    z_samples = state_bag.get("z_samples")  # (B, K) latent representations
    x_flat = state_bag.get("x_flat")  # (B, D) flattened input
    if z_samples is not None and x_flat is not None:
        try:
            mi_analysis = estimate_mutual_information(x_flat, z_samples)
            result["mutual_info_total"] = mi_analysis.get("total_mi", 0.0)
            result["mutual_info_per_dim"] = mi_analysis.get("mi_per_dim", [])
            result["available"] = True
        except Exception as e:
            logger.debug("Could not estimate mutual information: %s", e)

    return result


def factor_model_extended_diagnostics(
    state_bag: dict[str, Any],
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Collect extended factor model diagnostics using new diagnostic modules.

    Integrates compute_eigenvalue_concentration, track_regression_quality,
    and analyze_exposure_norms from src.diagnostics.factor_diagnostics.

    :param state_bag (dict): Pipeline state bag
    :param returns (pd.DataFrame): Historical returns

    :return diag (dict): Extended factor diagnostics for composite scores
    """
    result: dict[str, Any] = {"available": False}

    risk_model: dict[str, Any] = state_bag.get("risk_model", {})
    eigenvalues: np.ndarray = risk_model.get("eigenvalues", np.array([]))
    B_A: np.ndarray = state_bag.get("B_A", np.array([]))
    z_hat: np.ndarray = state_bag.get("z_hat", np.array([]))
    B_A_by_date: dict[str, np.ndarray] = state_bag.get("B_A_by_date", {})
    valid_dates: list[str] = state_bag.get("valid_dates", [])

    # Eigenvalue concentration
    if eigenvalues.size > 0:
        eig_analysis = compute_eigenvalue_concentration(eigenvalues)
        if eig_analysis.get("available", False):
            result["available"] = True
            result["concentration_ratio"] = eig_analysis.get(
                "var_explained_top1", 0.0
            )
            result["eff_dim"] = eig_analysis.get("eff_dim", 0.0)
            result["signal_noise_boundary"] = eig_analysis.get(
                "signal_noise_boundary", 0
            )

    # Regression quality tracking
    if B_A_by_date and z_hat.size > 0 and valid_dates:
        # Prepare returns as array (ensure ndarray type)
        returns_arr: np.ndarray = (
            returns.values if hasattr(returns, "values") else np.asarray(returns)
        )
        reg_analysis = track_regression_quality(
            B_A_by_date, z_hat, returns_arr, valid_dates
        )
        if reg_analysis.get("available", False):
            result["available"] = True
            result["rank_deficiency_rate"] = reg_analysis.get(
                "rank_deficiency_rate", 0.0
            )
            result["cs_r2_mean"] = reg_analysis.get("cs_r2_mean", 0.0)
            result["cs_r2_std"] = reg_analysis.get("cs_r2_std", 0.0)
            result["n_extreme_z_hat"] = reg_analysis.get("n_extreme_z_hat", 0)
            result["n_dates"] = reg_analysis.get("n_dates", 1)
            result["cond_mean"] = reg_analysis.get("cond_mean", 0.0)

    # Exposure norms
    if B_A.size > 0:
        norm_analysis = analyze_exposure_norms(B_A)
        if norm_analysis.get("available", False):
            result["available"] = True
            result["n_outlier_factors"] = norm_analysis.get(
                "n_outlier_factors", 0
            )
            result["sparsity"] = norm_analysis.get("sparsity", 0.0)

    # G4: DGJ shrinkage validation
    eigs_sample = risk_model.get("eigenvalues_sample", np.array([]))
    eigs_shrunk = risk_model.get("eigenvalues_shrunk", eigenvalues)
    gamma = risk_model.get("aspect_ratio", 1.0)
    if eigs_sample.size > 0 and eigs_shrunk.size > 0 and gamma > 0:
        try:
            dgj_analysis = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)
            result["dgj_n_signal"] = dgj_analysis.get("n_signal", 0)
            result["dgj_noise_cv"] = dgj_analysis.get("noise_cv", 0.0)
            result["dgj_bbp_threshold"] = dgj_analysis.get("bbp_threshold", 0.0)
            result["available"] = True
        except Exception as e:
            logger.debug("Could not validate DGJ recovery: %s", e)

    # G3: Eigenvector rotation stability (cross-fold, stored in state_bag)
    eigenvector_rotation = state_bag.get("eigenvector_rotation")
    if eigenvector_rotation is not None:
        result["eigenvector_mean_alignment"] = eigenvector_rotation.get("mean_alignment", 0.0)
        result["eigenvector_min_alignment"] = eigenvector_rotation.get("min_alignment", 0.0)
        result["eigenvector_n_rotated"] = eigenvector_rotation.get("n_rotated", 0)
        result["available"] = True

    return result


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
    factor_quality: dict[str, Any] | None = None,
    solver: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    vae_posterior: dict[str, Any] | None = None,
    factor_extended: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Run automated health checks across all diagnostic categories.

    :param training (dict): Training diagnostics
    :param latent (dict): Latent space diagnostics
    :param risk_model (dict): Risk model diagnostics
    :param portfolio (dict): Portfolio diagnostics
    :param data_quality (dict): Data quality diagnostics
    :param benchmark_comparison_data (dict | None): Benchmark comparison with per-benchmark metrics
    :param factor_quality (dict | None): Factor quality diagnostics
    :param solver (dict | None): Solver convergence diagnostics
    :param constraints (dict | None): Constraint binding diagnostics
    :param vae_posterior (dict | None): VAE posterior diagnostics (G1-G2-G5)
    :param factor_extended (dict | None): Extended factor model diagnostics (G3-G4)

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

    # --- Factor quality checks ---
    if factor_quality is not None and factor_quality.get("available", False):
        # AU validation vs Bai-Ng IC2
        au_bn_diff = factor_quality.get("au_bai_ng_diff")
        k_bn = factor_quality.get("k_bai_ng")
        if au_bn_diff is not None and k_bn is not None:
            if abs(au_bn_diff) > _THRESHOLDS["au_bai_ng_diff_crit"]:
                _add("Factor Quality", "AU vs Bai-Ng IC2", "CRITICAL",
                     f"AU={au}, k_bai_ng={k_bn}, diff={au_bn_diff} — large divergence")
            elif abs(au_bn_diff) > _THRESHOLDS["au_bai_ng_diff_warn"]:
                _add("Factor Quality", "AU vs Bai-Ng IC2", "WARNING",
                     f"AU={au}, k_bai_ng={k_bn}, diff={au_bn_diff}")
            else:
                _add("Factor Quality", "AU vs Bai-Ng IC2", "OK",
                     f"AU={au}, k_bai_ng={k_bn} (consistent)")

        # AU validation vs Onatski
        k_onatski = factor_quality.get("k_onatski")
        au_on_diff = factor_quality.get("au_onatski_diff")
        if k_onatski is not None and au_on_diff is not None:
            if abs(au_on_diff) > _THRESHOLDS["au_bai_ng_diff_crit"]:
                _add("Factor Quality", "AU vs Onatski", "WARNING",
                     f"AU={au}, k_onatski={k_onatski}, diff={au_on_diff}")
            else:
                _add("Factor Quality", "AU vs Onatski", "OK",
                     f"AU={au}, k_onatski={k_onatski}")

        # Latent stability between folds
        stability_rho = factor_quality.get("latent_stability_rho")
        if stability_rho is not None and not np.isnan(stability_rho):
            if stability_rho < _THRESHOLDS["latent_stability_min"]:
                _add("Factor Quality", "Latent stability", "WARNING",
                     f"rho={stability_rho:.3f} < {_THRESHOLDS['latent_stability_min']} — "
                     "factor structure unstable between folds")
            else:
                _add("Factor Quality", "Latent stability", "OK",
                     f"rho={stability_rho:.3f} — factor structure stable")

        # Factor composition summary
        n_struct = factor_quality.get("n_structural", 0)
        n_style = factor_quality.get("n_style", 0)
        n_epis = factor_quality.get("n_episodic", 0)
        if n_struct + n_style + n_epis > 0:
            _add("Factor Quality", "Factor composition", "OK",
                 f"{n_struct} structural, {n_style} style, {n_epis} episodic")

    # --- Risk model checks ---
    var_ratio = risk_model.get("var_ratio_oos", float("nan"))
    if np.isnan(var_ratio):
        _add("Risk Model", "Variance ratio", "WARNING",
             "var_ratio not computed — insufficient OOS data for held positions")
    elif var_ratio < _THRESHOLDS["var_ratio_lo"] or var_ratio > _THRESHOLDS["var_ratio_hi"]:
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
    h_norm_signal = portfolio.get("H_norm_signal", 0.0)
    h_norm_eff = portfolio.get("H_norm_eff", 0.0)
    enb = portfolio.get("enb", 0.0)
    n_signal_port = portfolio.get("n_signal", 0)
    n_eff_eig = portfolio.get("n_eff_eigenvalue", 0.0)
    # Use H_norm_signal (normalized by ln(n_signal)) for the health check.
    # After eigenvalue truncation, n_signal is the number of real factors
    # identified by DGJ — this is the correct denominator.
    h_for_check = h_norm_signal if h_norm_signal > 0 else (h_norm_eff if h_norm_eff > 0 else h_norm)
    h_detail = (
        f"H_norm_signal = {h_norm_signal:.3f} (n_signal={n_signal_port}), "
        f"ENB = {enb:.2f}, H_norm_eff = {h_norm_eff:.3f} (n_eff={n_eff_eig:.1f})"
    )
    if h_for_check < _THRESHOLDS["h_norm_min"]:
        _add("Portfolio", "Factor entropy", "WARNING",
             f"{h_detail} — poor factor diversification")
    else:
        _add("Portfolio", "Factor entropy", "OK", h_detail)

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

    # --- Non-trivial selection check (detect alphabetical/positional bias) ---
    w_raw = portfolio.get("_weights_raw", np.array([]))
    n_total_sel = portfolio.get("n_total_stocks", 0)
    n_active_sel = portfolio.get("n_active_positions", 0)

    if isinstance(w_raw, np.ndarray) and w_raw.size > 0 and n_total_sel >= 10 and n_active_sel >= 3:
        active_indices = np.where(w_raw > 1e-8)[0]
        n_act = len(active_indices)
        if n_act >= 3:
            mean_rank = float(np.mean(active_indices))
            expected_mean = (n_total_sel - 1) / 2.0  # 0-indexed
            rank_ratio = mean_rank / expected_mean if expected_mean > 0 else 1.0

            # Fraction of active stocks in first quintile of sorted list
            quintile_cutoff = max(1, n_total_sel // 5)
            frac_first_q = float(np.sum(active_indices < quintile_cutoff)) / n_act

            detail = (
                f"Mean rank = {mean_rank:.0f}/{n_total_sel} "
                f"(ratio={rank_ratio:.2f}), "
                f"first quintile = {frac_first_q:.0%} of active"
            )
            if rank_ratio < 0.1 or frac_first_q > 0.9:
                _add("Portfolio", "Selection non-trivial", "CRITICAL", detail)
            elif rank_ratio < 0.3 or frac_first_q > 0.7:
                _add("Portfolio", "Selection non-trivial", "WARNING", detail)
            else:
                _add("Portfolio", "Selection non-trivial", "OK", detail)

    # --- Solver convergence checks ---
    if solver is not None and solver.get("available", False):
        best_grad_norm = solver.get("best_final_grad_norm", float("nan"))
        converged_ratio = solver.get("converged_ratio", 0.0)
        best_converged = solver.get("best_converged", False)
        n_starts = solver.get("n_starts", 0)
        converged_count = solver.get("converged_count", 0)

        if not np.isnan(best_grad_norm):
            if best_grad_norm > _THRESHOLDS["sca_grad_norm_crit"]:
                _add("Solver", "Final gradient norm", "CRITICAL",
                     f"grad_norm = {best_grad_norm:.2e} — SCA did not converge")
            elif best_grad_norm > _THRESHOLDS["sca_grad_norm_warn"]:
                _add("Solver", "Final gradient norm", "WARNING",
                     f"grad_norm = {best_grad_norm:.2e}")
            else:
                _add("Solver", "Final gradient norm", "OK",
                     f"grad_norm = {best_grad_norm:.2e}")

        if n_starts > 1:
            if converged_ratio < _THRESHOLDS["sca_converged_ratio_warn"]:
                _add("Solver", "Convergence ratio", "WARNING",
                     f"{converged_count}/{n_starts} starts converged ({converged_ratio:.0%})")
            else:
                _add("Solver", "Convergence ratio", "OK",
                     f"{converged_count}/{n_starts} starts converged ({converged_ratio:.0%})")
        elif not best_converged:
            _add("Solver", "Convergence", "WARNING",
                 "Single-start solver did not converge")
        else:
            _add("Solver", "Convergence", "OK", "Solver converged")

    # --- Constraint binding checks ---
    if constraints is not None and constraints.get("available", False):
        n_at_w_max = constraints.get("n_at_w_max", 0)
        binding_fraction = constraints.get("binding_fraction", 0.0)
        tau_binding = constraints.get("tau_binding", False)
        actual_turnover = constraints.get("actual_turnover", 0.0)

        if binding_fraction > _THRESHOLDS["binding_fraction_warn"]:
            _add("Constraints", "w_max binding", "WARNING",
                 f"{n_at_w_max} positions at w_max ({binding_fraction:.0%} of portfolio)")
        elif n_at_w_max > 0:
            _add("Constraints", "w_max binding", "OK",
                 f"{n_at_w_max} positions at w_max ({binding_fraction:.0%} of portfolio)")
        else:
            _add("Constraints", "w_max binding", "OK", "No positions at w_max")

        if tau_binding:
            _add("Constraints", "Turnover constraint", "WARNING",
                 f"tau_max binding (actual turnover = {actual_turnover:.1%})")
        else:
            _add("Constraints", "Turnover constraint", "OK",
                 f"Turnover = {actual_turnover:.1%}")

    # --- VAE posterior checks (G1, G2, G5) ---
    if vae_posterior is not None and vae_posterior.get("available", False):
        # G5: Reconstruction temporal structure
        recon_acf1 = vae_posterior.get("recon_acf_1")
        recon_boundary = vae_posterior.get("recon_boundary_ratio")
        if recon_acf1 is not None:
            if recon_acf1 > _THRESHOLDS["recon_acf1_warn"]:
                _add("VAE Posterior", "Recon temporal autocorr (G5)", "WARNING",
                     f"ACF(1) = {recon_acf1:.3f} > {_THRESHOLDS['recon_acf1_warn']} — "
                     "systematic temporal patterns")
            else:
                _add("VAE Posterior", "Recon temporal autocorr (G5)", "OK",
                     f"ACF(1) = {recon_acf1:.3f}")

        if recon_boundary is not None:
            if recon_boundary > _THRESHOLDS["recon_boundary_ratio_warn"]:
                _add("VAE Posterior", "Recon boundary ratio (G5)", "WARNING",
                     f"Boundary ratio = {recon_boundary:.2f} > {_THRESHOLDS['recon_boundary_ratio_warn']} — "
                     "boundary effect detected")
            else:
                _add("VAE Posterior", "Recon boundary ratio (G5)", "OK",
                     f"Boundary ratio = {recon_boundary:.2f}")

        # G2: Mutual information
        mi_total = vae_posterior.get("mutual_info_total")
        if mi_total is not None:
            if mi_total < _THRESHOLDS["mutual_info_warn"]:
                _add("VAE Posterior", "Mutual information (G2)", "WARNING",
                     f"MI = {mi_total:.3f} nats < {_THRESHOLDS['mutual_info_warn']} — "
                     "latent may lose input information")
            else:
                _add("VAE Posterior", "Mutual information (G2)", "OK",
                     f"MI = {mi_total:.3f} nats")

        # Collapse/explosion severity (from existing log_var analysis)
        collapse_sev = vae_posterior.get("collapse_severity", 0.0)
        explosion_sev = vae_posterior.get("explosion_severity", 0.0)
        if collapse_sev > 0.3:
            _add("VAE Posterior", "Posterior collapse", "WARNING",
                 f"{collapse_sev:.0%} dims at lower bound — partial collapse")
        elif collapse_sev > 0.1:
            _add("VAE Posterior", "Posterior collapse", "OK",
                 f"{collapse_sev:.0%} dims at lower bound")

        if explosion_sev > 0.3:
            _add("VAE Posterior", "Posterior explosion", "WARNING",
                 f"{explosion_sev:.0%} dims at upper bound — partial explosion")
        elif explosion_sev > 0.1:
            _add("VAE Posterior", "Posterior explosion", "OK",
                 f"{explosion_sev:.0%} dims at upper bound")

    # --- Extended factor model checks (G3, G4) ---
    if factor_extended is not None and factor_extended.get("available", False):
        # G3: Eigenvector rotation stability
        eig_alignment = factor_extended.get("eigenvector_mean_alignment")
        if eig_alignment is not None:
            n_rotated = factor_extended.get("eigenvector_n_rotated", 0)
            if eig_alignment < _THRESHOLDS["eigenvector_alignment_warn"]:
                _add("Factor Model", "Eigenvector stability (G3)", "WARNING",
                     f"Mean alignment = {eig_alignment:.3f} < {_THRESHOLDS['eigenvector_alignment_warn']}, "
                     f"{n_rotated} dims rotated — factor interpretation unstable")
            else:
                _add("Factor Model", "Eigenvector stability (G3)", "OK",
                     f"Mean alignment = {eig_alignment:.3f}, {n_rotated} dims rotated")

        # G4: DGJ shrinkage validation
        dgj_noise_cv = factor_extended.get("dgj_noise_cv")
        dgj_n_signal = factor_extended.get("dgj_n_signal")
        if dgj_noise_cv is not None:
            if dgj_noise_cv > _THRESHOLDS["dgj_noise_cv_warn"]:
                _add("Factor Model", "DGJ shrinkage (G4)", "WARNING",
                     f"Noise CV = {dgj_noise_cv:.3f} > {_THRESHOLDS['dgj_noise_cv_warn']}, "
                     f"n_signal = {dgj_n_signal} — bulk not flat, check shrinkage")
            else:
                _add("Factor Model", "DGJ shrinkage (G4)", "OK",
                     f"Noise CV = {dgj_noise_cv:.3f}, n_signal = {dgj_n_signal}")

        # Eigenvalue concentration (from existing extended diagnostics)
        concentration = factor_extended.get("concentration_ratio")
        if concentration is not None and concentration > 0.5:
            _add("Factor Model", "Eigenvalue concentration", "WARNING",
                 f"Top eigenvalue explains {concentration:.0%} — dominated by single factor")

        # Rank deficiency rate
        rank_def_rate = factor_extended.get("rank_deficiency_rate")
        if rank_def_rate is not None and rank_def_rate > 0.1:
            _add("Factor Model", "Regression rank deficiency", "WARNING",
                 f"{rank_def_rate:.0%} of dates had rank-deficient B_A")

    # --- Portfolio checks (G9: Deflated Sharpe Ratio) ---
    dsr_prob = portfolio.get("dsr_probability")
    if dsr_prob is not None and dsr_prob > 0:
        if dsr_prob < _THRESHOLDS["dsr_probability_warn"]:
            _add("Portfolio", "Deflated Sharpe (G9)", "WARNING",
                 f"DSR probability = {dsr_prob:.3f} < {_THRESHOLDS['dsr_probability_warn']} — "
                 "Sharpe may be overstated due to multiple testing")
        else:
            _add("Portfolio", "Deflated Sharpe (G9)", "OK",
                 f"DSR probability = {dsr_prob:.3f}")

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

    # Inject latent_stability_rho from vae_metrics into state_bag if available
    # (walk-forward computes it outside _run_single_fold and stores in metrics)
    if "latent_stability_rho" not in state_bag and "latent_stability_rho" in vae_metrics:
        state_bag["latent_stability_rho"] = vae_metrics["latent_stability_rho"]

    logger.info("Collecting training diagnostics...")
    training = training_diagnostics(state_bag.get("fit_result"))

    logger.info("Collecting latent space diagnostics...")
    latent = latent_diagnostics(state_bag)

    logger.info("Collecting factor quality diagnostics...")
    factor_qual = factor_quality_diagnostics(state_bag, returns, inferred_stock_ids)

    logger.info("Collecting risk model diagnostics...")
    risk = risk_model_diagnostics(state_bag, returns_oos, w_vae, inferred_stock_ids)

    logger.info("Collecting portfolio diagnostics...")
    portfolio = portfolio_diagnostics(state_bag, w_vae, vae_metrics)

    logger.info("Collecting solver diagnostics...")
    solver = solver_diagnostics(state_bag)

    logger.info("Collecting constraint binding diagnostics...")
    constraints = constraint_binding_diagnostics(state_bag)

    logger.info("Collecting benchmark comparison...")
    bench_comp = benchmark_comparison(vae_metrics, benchmark_results)

    logger.info("Collecting data quality diagnostics...")
    data_qual = data_quality_diagnostics(stock_data, returns)

    # Compute VAE posterior diagnostics (new module integration)
    logger.info("Collecting VAE posterior diagnostics...")
    vae_posterior = vae_posterior_diagnostics(
        state_bag.get("fit_result"), state_bag
    )

    # Compute factor model extended diagnostics (new module integration)
    logger.info("Collecting extended factor model diagnostics...")
    factor_extended = factor_model_extended_diagnostics(state_bag, returns)

    logger.info("Running health checks...")
    checks = run_health_checks(
        training, latent, risk, portfolio, data_qual, bench_comp, factor_qual,
        solver, constraints, vae_posterior, factor_extended,
    )

    n_critical = sum(1 for c in checks if c["status"] == "CRITICAL")
    n_warning = sum(1 for c in checks if c["status"] == "WARNING")
    n_ok = sum(1 for c in checks if c["status"] == "OK")
    logger.info(
        "Health checks complete: %d OK, %d WARNING, %d CRITICAL",
        n_ok, n_warning, n_critical,
    )

    # Compute composite scores
    logger.info("Computing composite scores...")
    n_active = portfolio.get("n_active_positions", 0)
    composite_scores = compute_all_composite_scores(
        solver_stats=solver if solver.get("available", False) else None,
        constraints=constraints if constraints.get("available", False) else None,
        risk_model=risk,
        training=training if training.get("available", False) else None,
        n_active=n_active,
        vae_diagnostics=vae_posterior if vae_posterior.get("available", False) else None,
        factor_diagnostics=factor_extended if factor_extended.get("available", False) else None,
        latent=latent,
        portfolio=portfolio,
        factor_quality=factor_qual if factor_qual.get("available", False) else None,
    )
    overall_score = composite_scores.get("overall", {}).get("score", 0)
    overall_grade = composite_scores.get("overall", {}).get("grade", "F")
    logger.info(
        "Composite scores computed: overall=%.1f (%s)",
        overall_score, overall_grade,
    )

    return {
        "training": training,
        "latent": latent,
        "factor_quality": factor_qual,
        "risk_model": risk,
        "portfolio": portfolio,
        "solver": solver,
        "constraints": constraints,
        "benchmark_comparison": bench_comp,
        "data_quality": data_qual,
        "health_checks": checks,
        "composite_scores": composite_scores,
        "vae_posterior": vae_posterior,
        "factor_extended": factor_extended,
        "config": config_dict,
        "summary": {
            "n_critical": n_critical,
            "n_warning": n_warning,
            "n_ok": n_ok,
            "overall_score": overall_score,
            "overall_grade": overall_grade,
        },
    }
