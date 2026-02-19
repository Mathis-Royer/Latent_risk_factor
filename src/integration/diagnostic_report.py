"""
Diagnostic report generation in Markdown, JSON, and CSV formats.

Produces a comprehensive, human-readable diagnostic report from the
diagnostics collected by src/integration/diagnostics.py.
"""

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.integration.reporting import serialize_for_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _status_icon(status: str) -> str:
    """
    Return a plain-text icon for health check status.

    :param status (str): OK / WARNING / CRITICAL

    :return icon (str): Status marker
    """
    return {"OK": "[OK]", "WARNING": "[WARN]", "CRITICAL": "[CRIT]"}.get(status, "[?]")


def _fmt(val: object, precision: int = 4) -> str:
    """
    Format a value for display.

    :param val (object): Value to format
    :param precision (int): Decimal precision for floats

    :return formatted (str): Formatted string
    """
    if isinstance(val, float):
        if abs(val) < 0.01 and val != 0:
            return f"{val:.2e}"
        return f"{val:.{precision}f}"
    return str(val)


def _extract_ew_mdd(bench: dict[str, Any]) -> float | None:
    """
    Extract equal-weight benchmark MDD as market proxy.

    :param bench (dict): Benchmark comparison diagnostics

    :return ew_mdd (float | None): EW max drawdown, or None if unavailable
    """
    per_bench = bench.get("per_benchmark", {})
    ew_data = per_bench.get("equal_weight", {})
    ew_metrics = ew_data.get("bench_metrics", {})
    val = ew_metrics.get("max_drawdown_oos", None)
    if val is not None and np.isfinite(val):
        return float(val)
    return None


def generate_diagnostic_markdown(
    diagnostics: dict[str, Any],
) -> str:
    """
    Generate a comprehensive diagnostic report in Markdown format.

    :param diagnostics (dict): Full diagnostics from collect_diagnostics()

    :return markdown (str): Complete Markdown report
    """
    lines: list[str] = []

    training = diagnostics.get("training", {})
    latent = diagnostics.get("latent", {})
    risk = diagnostics.get("risk_model", {})
    portfolio = diagnostics.get("portfolio", {})
    bench = diagnostics.get("benchmark_comparison", {})
    data_q = diagnostics.get("data_quality", {})
    checks = diagnostics.get("health_checks", [])
    config = diagnostics.get("config", {})
    summary = diagnostics.get("summary", {})

    # ===== Header =====
    lines.append("# VAE Latent Risk Factor — Diagnostic Report")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ===== Executive Summary =====
    lines.append("## Executive Summary")
    lines.append("")
    n_crit = summary.get("n_critical", 0)
    n_warn = summary.get("n_warning", 0)
    n_ok = summary.get("n_ok", 0)

    if n_crit > 0:
        lines.append(
            f"**Overall Status: ISSUES DETECTED** — "
            f"{n_crit} critical, {n_warn} warnings, {n_ok} OK"
        )
    elif n_warn > 0:
        lines.append(
            f"**Overall Status: REVIEW NEEDED** — "
            f"{n_warn} warnings, {n_ok} OK"
        )
    else:
        lines.append(f"**Overall Status: ALL CLEAR** — {n_ok} checks passed")
    lines.append("")

    # Health check table
    lines.append("### Health Checks")
    lines.append("")
    lines.append("| Status | Category | Check | Details |")
    lines.append("|--------|----------|-------|---------|")
    for c in checks:
        lines.append(
            f"| {_status_icon(c['status'])} | {c['category']} | "
            f"{c['check']} | {c['message']} |"
        )
    lines.append("")

    # ===== 1. Data Quality =====
    lines.append("## 1. Data Quality")
    lines.append("")
    lines.append(f"- **Universe size**: {data_q.get('n_stocks', 'N/A')} stocks")
    lines.append(f"- **Date range**: {data_q.get('date_range', 'N/A')}")
    lines.append(f"- **Trading days**: {data_q.get('n_dates', 'N/A')}")
    lines.append(f"- **Years of data**: {data_q.get('years_of_data', 0):.1f}")
    lines.append(f"- **Missing data**: {data_q.get('missing_pct', 0):.2f}%")
    lines.append(
        f"- **Stocks > 20% missing**: {data_q.get('stocks_over_20pct_missing', 0)}"
    )
    sector_dist = data_q.get("sector_distribution", {})
    if sector_dist:
        lines.append("")
        lines.append("**Sector Distribution:**")
        lines.append("")
        lines.append("| Sector | Count |")
        lines.append("|--------|-------|")
        for sector, count in sorted(sector_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| {sector} | {count} |")
    lines.append("")

    # ===== 2. Training Convergence =====
    lines.append("## 2. Training Convergence")
    lines.append("")
    if not training.get("available", False):
        lines.append(f"*Training diagnostics not available: {training.get('reason', 'unknown')}*")
    else:
        lines.append(f"- **Epochs**: {training.get('n_epochs', 0)}")
        lines.append(
            f"- **Best epoch**: {training.get('best_epoch', 0)} "
            f"({training.get('best_epoch_fraction', 0):.0%} of total)"
        )
        lines.append(f"- **Best val ELBO**: {_fmt(training.get('best_val_elbo', 0))}")
        lines.append(f"- **Overfit flag**: {training.get('overfit_flag', False)}")
        lines.append(f"- **Overfit ratio**: {_fmt(training.get('overfit_ratio', 1.0))}")
        lines.append(
            f"- **Val ELBO still decreasing at end**: {training.get('still_decreasing_at_end', False)}"
        )
        lines.append(f"- **LR reductions**: {training.get('n_lr_reductions', 0)}")
        lines.append("")

        lines.append("### Loss Decomposition (best epoch)")
        lines.append("")
        lines.append("| Component | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Total loss | {_fmt(training.get('best_loss', 0))} |")
        lines.append(f"| Reconstruction | {_fmt(training.get('best_recon', 0))} |")
        lines.append(f"| KL divergence | {_fmt(training.get('best_kl', 0))} |")
        lines.append("")

        lines.append("### Observation Noise (sigma_sq)")
        lines.append("")
        lines.append(f"- **Initial**: {_fmt(training.get('sigma_sq_initial', 0))}")
        lines.append(f"- **Final**: {_fmt(training.get('sigma_sq_final', 0))}")
        lines.append(f"- **Hit lower bound**: {training.get('sigma_sq_min_hit', False)}")
        lines.append(f"- **Hit upper bound**: {training.get('sigma_sq_max_hit', False)}")
        lines.append("")

        lines.append("### Active Units Evolution")
        lines.append("")
        lines.append(f"- **Initial**: {training.get('au_initial', 0)}")
        lines.append(f"- **Final**: {training.get('au_final', 0)}")
        lines.append(f"- **Peak during training**: {training.get('au_max_during_training', 0)}")
    lines.append("")

    # ===== 3. Latent Space =====
    lines.append("## 3. Latent Space Analysis")
    lines.append("")
    lines.append(f"- **K (latent capacity)**: {latent.get('K', 0)}")
    lines.append(f"- **AU (active units)**: {latent.get('AU', 0)}")
    lines.append(
        f"- **Utilization ratio**: {latent.get('utilization_ratio', 0):.1%}"
    )
    lines.append(
        f"- **Effective latent dims**: {_fmt(latent.get('eff_latent_dims', 0), 1)}"
    )
    lines.append(f"- **KL total**: {_fmt(latent.get('kl_total', 0))}")
    lines.append(
        f"- **KL top-3 fraction**: "
        f"{latent.get('kl_top3_fraction', 0):.1%}"
    )

    b_stats = latent.get("B_stats", {})
    if b_stats:
        lines.append("")
        lines.append("### Exposure Matrix B")
        lines.append("")
        lines.append(f"- **Shape**: {latent.get('B_shape', [])}")
        lines.append(f"- **Sparsity**: {b_stats.get('sparsity', 0):.1%}")
        lines.append(f"- **Max absolute entry**: {_fmt(b_stats.get('max_entry', 0))}")
        lines.append(f"- **Mean dim norm**: {_fmt(b_stats.get('mean_dim_norm', 0))}")
        lines.append(f"- **Mean stock norm**: {_fmt(b_stats.get('mean_stock_norm', 0))}")
    lines.append("")

    # ===== 3.5 Factor Quality Dashboard =====
    factor_qual = diagnostics.get("factor_quality", {})
    if factor_qual.get("available", False):
        lines.append("## 3.5 Factor Quality Dashboard")
        lines.append("")
        lines.append("### AU Validation")
        lines.append("")
        au = factor_qual.get("AU", 0)
        k_bn = factor_qual.get("k_bai_ng")
        k_on = factor_qual.get("k_onatski")
        lines.append(f"- **VAE Active Units (AU)**: {au}")
        if k_bn is not None:
            lines.append(f"- **Bai-Ng IC2 optimal k**: {k_bn}")
        if k_on is not None:
            lines.append(f"- **Onatski test k**: {k_on}")

        au_bn_diff = factor_qual.get("au_bai_ng_diff")
        au_on_diff = factor_qual.get("au_onatski_diff")
        if au_bn_diff is not None or au_on_diff is not None:
            lines.append("")
            status = "[OK]" if (
                (au_bn_diff is None or abs(au_bn_diff) <= 10) and
                (au_on_diff is None or abs(au_on_diff) <= 20)
            ) else "[WARN]"
            lines.append(f"**Status**: {status} AU consistent with statistical tests")
        lines.append("")

        # Factor composition
        lines.append("### Factor Composition")
        lines.append("")
        n_struct = factor_qual.get("n_structural", 0)
        n_style = factor_qual.get("n_style", 0)
        n_epis = factor_qual.get("n_episodic", 0)
        lines.append(f"- **Structural factors**: {n_struct} ({factor_qual.get('pct_structural', 0):.0%})")
        lines.append(f"- **Style factors**: {n_style}")
        lines.append(f"- **Episodic factors**: {n_epis}")
        lines.append("")

        # Per-factor table (top 10)
        breadths = factor_qual.get("breadth_top10", [])
        half_lives = factor_qual.get("half_lives_top10", [])
        categories = factor_qual.get("categories_top10", [])
        if breadths or half_lives or categories:
            lines.append("### Top Factors (by KL divergence)")
            lines.append("")
            lines.append("| Factor | Breadth | Persistence (days) | Category |")
            lines.append("|--------|---------|-------------------|----------|")
            for i in range(min(10, len(categories))):
                br = breadths[i] if i < len(breadths) else "?"
                hl_raw = half_lives[i] if i < len(half_lives) else float("inf")
                hl = "inf" if not np.isfinite(hl_raw) else f"{hl_raw:.0f}"
                cat = categories[i] if i < len(categories) else "?"
                lines.append(f"| {i+1} | {br} | {hl} | {cat} |")
            lines.append("")

        # Latent stability
        stability_rho = factor_qual.get("latent_stability_rho")
        lines.append("### Latent Stability")
        lines.append("")
        if stability_rho is not None and not np.isnan(stability_rho):
            status = "[OK]" if stability_rho >= 0.85 else "[WARN]"
            lines.append(
                f"- **Spearman rho between folds**: {stability_rho:.3f} {status}"
            )
        else:
            lines.append(
                "- **Spearman rho between folds**: N/A (single fold or first fold)"
            )
        lines.append("")

    # ===== 4. Risk Model =====
    lines.append("## 4. Risk Model Quality")
    lines.append("")
    lines.append(
        f"- **Variance targeting**: sys={_fmt(risk.get('vt_scale_sys', 1.0))}, "
        f"idio={_fmt(risk.get('vt_scale_idio', 1.0))}"
    )
    var_ratio_val = risk.get("var_ratio_oos", float("nan"))
    if np.isnan(var_ratio_val):
        lines.append("- **Variance ratio (OOS)**: N/A (insufficient data)")
    else:
        lines.append(
            f"- **Variance ratio (OOS)**: {_fmt(var_ratio_val)} "
            f"(target: [0.5, 2.0])"
        )
    lines.append(
        f"- **Rank correlation (OOS)**: {_fmt(risk.get('corr_rank_oos', 0))}"
    )
    lines.append(f"- **Explanatory power (OOS)**: {_fmt(risk.get('explanatory_power', 0))}")
    ep_is = risk.get("ep_in_sample")
    if ep_is is not None:
        lines.append(f"- **Explanatory power (IS)**: {_fmt(ep_is)}")
    if "avg_cs_r2" in risk:
        lines.append(
            f"- **Avg cross-sectional R² (OOS)**: {_fmt(risk.get('avg_cs_r2', 0))}"
        )
    if "B_A_mean_abs" in risk:
        lines.append("")
        lines.append("### Exposure Matrix (B_A) Scale")
        lines.append("")
        lines.append(f"- **Mean |B_A|**: {_fmt(risk.get('B_A_mean_abs', 0))}")
        lines.append(f"- **Std B_A**: {_fmt(risk.get('B_A_std', 0))}")
        lines.append(f"- **Max |B_A|**: {_fmt(risk.get('B_A_max_abs', 0))}")
        lines.append(
            f"- **Column norm (mean)**: {_fmt(risk.get('B_A_col_norm_mean', 0))}"
        )
        lines.append(
            f"- **Column norm (max)**: {_fmt(risk.get('B_A_col_norm_max', 0))}"
        )
    lines.append(
        f"- **Condition number**: {risk.get('condition_number', 0):.2e}"
    )

    if "eigenvalues" in risk:
        lines.append("")
        lines.append("### Eigenvalue Spectrum")
        lines.append("")
        lines.append(
            f"- **Number of eigenvalues**: {risk.get('n_eigenvalues', 0)}"
        )
        lines.append(f"- **Top eigenvalue**: {_fmt(risk.get('top_eigenvalue', 0))}")
        lines.append(
            f"- **Top 3 explained**: {risk.get('top_3_explained', 0):.1%}"
        )
        lines.append(
            f"- **Top 10 explained**: {risk.get('top_10_explained', 0):.1%}"
        )
        lines.append(
            f"- **Ratio #1/#2**: {_fmt(risk.get('eigenvalue_ratio_1_2', 0), 2)}"
        )
    lines.append("")

    # ===== 5. Portfolio Optimization =====
    lines.append("## 5. Portfolio Optimization")
    lines.append("")
    lines.append(f"- **Alpha (risk aversion)**: {_fmt(portfolio.get('alpha_opt', 0))}")
    lines.append(
        f"- **Active positions**: {portfolio.get('n_active_positions', 0)} "
        f"/ {portfolio.get('n_total_stocks', 0)}"
    )
    lines.append(
        f"- **Effective N**: {_fmt(portfolio.get('eff_n_positions', 0), 1)}"
    )
    lines.append(f"- **HHI**: {_fmt(portfolio.get('hhi', 0))}")
    lines.append(f"- **Gini coefficient**: {_fmt(portfolio.get('gini_coefficient', 0))}")
    lines.append(f"- **Max weight**: {_fmt(portfolio.get('w_max', 0))}")
    lines.append(
        f"- **Min active weight**: {_fmt(portfolio.get('w_min_active', 0))}"
    )

    risk_decomp = portfolio.get("risk_decomposition", {})
    if risk_decomp.get("available", False):
        lines.append("")
        lines.append("### Factor Risk Decomposition")
        lines.append("")
        lines.append(
            f"- **Top 1 factor contribution**: "
            f"{risk_decomp.get('top_1_fraction', 0):.1%}"
        )
        lines.append(
            f"- **Top 3 factor contribution**: "
            f"{risk_decomp.get('top_3_fraction', 0):.1%}"
        )
        lines.append(
            f"- **Risk entropy (H)**: {_fmt(risk_decomp.get('entropy_H', 0))}"
        )
        lines.append(
            f"- **Max possible entropy**: {_fmt(risk_decomp.get('max_entropy', 0))}"
        )
    lines.append("")

    # ===== 6. OOS Performance =====
    lines.append("## 6. Out-of-Sample Performance")
    lines.append("")
    lines.append("### VAE Portfolio")
    lines.append("")
    lines.append(f"- **Annualized return**: {portfolio.get('ann_return', 0):.2%}")
    lines.append(f"- **Annualized volatility**: {portfolio.get('ann_vol', 0):.2%}")
    lines.append(f"- **Sharpe ratio**: {_fmt(portfolio.get('sharpe', 0), 3)}")
    lines.append(f"- **Sortino ratio**: {_fmt(portfolio.get('sortino', 0), 3)}")
    lines.append(f"- **Calmar ratio**: {_fmt(portfolio.get('calmar', 0), 3)}")
    # MDD with benchmark context
    mdd_line = f"- **Max drawdown**: {portfolio.get('max_drawdown', 0):.2%}"
    ew_mdd = _extract_ew_mdd(bench)
    if ew_mdd is not None:
        mdd_line += f" (EW benchmark: {ew_mdd:.2%})"
    lines.append(mdd_line)
    lines.append(
        f"- **Normalized entropy (H_norm)**: {_fmt(portfolio.get('H_norm_oos', 0))}"
    )
    h_norm_signal = portfolio.get("H_norm_signal", 0.0)
    enb = portfolio.get("enb", 0.0)
    n_signal_rep = portfolio.get("n_signal", 0)
    h_norm_eff = portfolio.get("H_norm_eff", 0.0)
    n_eff_eig = portfolio.get("n_eff_eigenvalue", 0.0)
    if h_norm_signal > 0:
        lines.append(
            f"- **H_norm_signal (vs n_signal)**: {_fmt(h_norm_signal)} "
            f"(n_signal = {n_signal_rep}, ENB = {enb:.2f})"
        )
    if h_norm_eff > 0:
        lines.append(
            f"- **H_norm_eff (vs effective dims)**: {_fmt(h_norm_eff)} "
            f"(n_eff = {n_eff_eig:.1f})"
        )
    lines.append("")

    # Benchmark comparison table
    per_bench = bench.get("per_benchmark", {})
    if per_bench:
        lines.append("### VAE vs Benchmarks")
        lines.append("")

        # Build comparison table
        metrics_to_show = [
            "sharpe", "ann_return", "ann_vol_oos", "max_drawdown_oos",
            "H_norm_oos", "eff_n_positions",
        ]
        header = "| Metric | VAE |"
        separator = "|--------|-----|"
        for bname in per_bench:
            short = bname.replace("_", " ").title()
            header += f" {short} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        vae_m = bench.get("vae_metrics", {})
        for metric in metrics_to_show:
            row = f"| {metric} | {_fmt(vae_m.get(metric, 0))} |"
            for bname, bdata in per_bench.items():
                bm = bdata.get("bench_metrics", {})
                row += f" {_fmt(bm.get(metric, 0))} |"
            lines.append(row)
        lines.append("")

        # Win/loss summary
        lines.append("### Win/Loss Summary")
        lines.append("")
        lines.append("| Benchmark | VAE Wins | VAE Losses |")
        lines.append("|-----------|----------|------------|")
        for bname, bdata in per_bench.items():
            lines.append(
                f"| {bname} | {bdata.get('wins', 0)} | {bdata.get('losses', 0)} |"
            )
        lines.append("")
        lines.append(
            f"**Total: {bench.get('total_wins', 0)} wins, "
            f"{bench.get('total_losses', 0)} losses**"
        )
    lines.append("")

    # ===== 7. Diagnosis & Recommendations =====
    lines.append("## 7. Diagnosis & Recommendations")
    lines.append("")
    recommendations = _generate_recommendations(diagnostics)
    if recommendations:
        for rec in recommendations:
            lines.append(f"- {rec}")
    else:
        lines.append("No specific recommendations — pipeline appears healthy.")
    lines.append("")

    # ===== Configuration =====
    lines.append("## Appendix: Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(serialize_for_json(config), indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _generate_recommendations(diagnostics: dict[str, Any]) -> list[str]:
    """
    Generate actionable recommendations from diagnostic results.

    :param diagnostics (dict): Full diagnostics

    :return recommendations (list[str]): List of recommendation strings
    """
    recs: list[str] = []
    training = diagnostics.get("training", {})
    latent = diagnostics.get("latent", {})
    risk = diagnostics.get("risk_model", {})
    portfolio = diagnostics.get("portfolio", {})

    # Training recommendations
    if training.get("still_decreasing_at_end", False):
        recs.append(
            "**Increase max_epochs**: val ELBO was still decreasing at the end of training. "
            "The model may benefit from longer training."
        )

    if training.get("overfit_ratio", 1.0) > 1.3:
        recs.append(
            "**Reduce overfitting**: consider increasing dropout, reducing K, "
            "or using more training data. Overfit ratio = "
            f"{training.get('overfit_ratio', 0):.2f}."
        )

    if training.get("sigma_sq_min_hit", False):
        recs.append(
            "**sigma_sq at lower bound**: the model is trying to reduce observation noise "
            "below the allowed minimum. Consider lowering sigma_sq_min or investigating "
            "whether training data variance is too low."
        )

    if training.get("sigma_sq_max_hit", False):
        recs.append(
            "**sigma_sq at upper bound**: the model is trying to increase observation noise "
            "above the allowed maximum. Consider raising sigma_sq_max or investigating "
            "whether reconstruction quality is poor."
        )

    # Latent space recommendations
    au = latent.get("AU", 0)
    k = latent.get("K", 0)
    if au < 5 and k > 20:
        recs.append(
            f"**Low AU ({au}/{k})**: most latent dimensions are unused. "
            "Consider reducing K, increasing training data, or checking "
            "if the KL term is dominating (posterior collapse)."
        )

    if latent.get("utilization_ratio", 0) > 0.8:
        recs.append(
            f"**High utilization ({au}/{k} = {latent.get('utilization_ratio', 0):.0%})**: "
            "consider increasing K to allow the model more capacity."
        )

    # Risk model recommendations
    var_ratio = risk.get("var_ratio_oos", float("nan"))
    if np.isnan(var_ratio):
        recs.append(
            "**Variance ratio not computed**: insufficient OOS data for held positions. "
            "Check that portfolio weights and OOS returns overlap."
        )
    elif var_ratio < 0.5:
        recs.append(
            f"**Covariance overestimation** (var_ratio = {var_ratio:.3f}): "
            "the model predicts much more risk than observed. "
            "Check variance targeting scale and Ledoit-Wolf shrinkage."
        )
    elif var_ratio > 2.0:
        recs.append(
            f"**Covariance underestimation** (var_ratio = {var_ratio:.3f}): "
            "the model predicts less risk than observed. "
            "The risk model may be poorly calibrated."
        )

    ep = risk.get("explanatory_power", 0)
    ep_is = risk.get("ep_in_sample")
    if ep < 0.05:
        ep_msg = (
            f"**Very low OOS explanatory power** (EP_oos = {ep:.4f}"
        )
        if ep_is is not None:
            ep_msg += f", EP_is = {ep_is:.4f}"
            if ep_is > 0.1 and ep < 0.01:
                ep_msg += (
                    "): large IS/OOS gap suggests the factor model overfits "
                    "the training period. Consider reducing AU, using shorter "
                    "aggregation windows, or regularizing the exposures."
                )
            else:
                ep_msg += (
                    "): latent factors explain little return variance even in-sample. "
                    "The VAE-discovered factors may not correspond to true risk drivers."
                )
        else:
            ep_msg += (
                "): latent factors explain almost none of the return variance. "
                "The VAE-discovered factors may not correspond to true risk drivers."
            )
        recs.append(ep_msg)

    # Portfolio recommendations
    sharpe = portfolio.get("sharpe", 0.0)
    bench_comp = diagnostics.get("benchmark_comparison", {})
    total_wins = bench_comp.get("total_wins", 0)
    total_losses = bench_comp.get("total_losses", 0)

    if sharpe < 0 and total_losses > total_wins:
        recs.append(
            "**VAE underperforms benchmarks with negative Sharpe**: "
            "the strategy is not adding value. Consider fundamental "
            "changes to the approach (loss mode, K, training period)."
        )
    elif total_losses > total_wins * 1.5:
        recs.append(
            "**VAE loses to most benchmarks**: the added complexity "
            "of the VAE model does not translate to better portfolios. "
            "Compare against PCA factor RP specifically to isolate "
            "whether the issue is non-linear factors or the pipeline."
        )

    return recs


# ---------------------------------------------------------------------------
# JSON report generation
# ---------------------------------------------------------------------------

def generate_diagnostic_json(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """
    Generate JSON-serializable diagnostic data.

    Excludes ``_raw_*`` keys which hold large numpy arrays / dicts
    intended only for in-memory notebook consumption.

    :param diagnostics (dict): Full diagnostics from collect_diagnostics()

    :return json_data (dict): JSON-safe diagnostic data
    """
    filtered = {k: v for k, v in diagnostics.items() if not k.startswith("_raw")}
    return serialize_for_json(filtered)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Save all diagnostic outputs
# ---------------------------------------------------------------------------

def save_diagnostic_report(
    diagnostics: dict[str, Any],
    output_dir: str = "results/diagnostic",
) -> list[str]:
    """
    Save diagnostic report in all formats (Markdown, JSON, CSV).

    :param diagnostics (dict): Full diagnostics from collect_diagnostics()
    :param output_dir (str): Output directory

    :return files (list[str]): List of written file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    # Markdown report
    md_path = os.path.join(output_dir, "diagnostic_report.md")
    md_content = generate_diagnostic_markdown(diagnostics)
    with open(md_path, "w") as f:
        f.write(md_content)
    written.append(md_path)
    logger.info("Markdown report saved: %s", md_path)

    # JSON data
    json_path = os.path.join(output_dir, "diagnostic_data.json")
    json_data = generate_diagnostic_json(diagnostics)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    written.append(json_path)
    logger.info("JSON data saved: %s", json_path)

    # Health checks CSV
    checks = diagnostics.get("health_checks", [])
    if checks:
        checks_path = os.path.join(output_dir, "health_checks.csv")
        pd.DataFrame(checks).to_csv(checks_path, index=False)
        written.append(checks_path)
        logger.info("Health checks CSV saved: %s", checks_path)

    # Training history CSV
    training = diagnostics.get("training", {})
    if training.get("available", False):
        history_data: dict[str, list[float]] = {}
        for key in ["train_loss", "train_recon", "train_kl", "train_co",
                     "val_elbo", "sigma_sq_series", "au_series", "lr_series"]:
            series = training.get(key, [])
            if series:
                col_name = key.replace("_series", "")
                history_data[col_name] = series

        if history_data:
            max_len = max(len(v) for v in history_data.values())
            for k, v in history_data.items():
                if len(v) < max_len:
                    history_data[k] = v + [float("nan")] * (max_len - len(v))

            history_path = os.path.join(output_dir, "training_history.csv")
            pd.DataFrame(history_data).to_csv(history_path, index_label="epoch")
            written.append(history_path)
            logger.info("Training history CSV saved: %s", history_path)

    # Benchmark comparison CSV
    bench = diagnostics.get("benchmark_comparison", {})
    per_bench = bench.get("per_benchmark", {})
    if per_bench:
        rows = []
        vae_m = bench.get("vae_metrics", {})
        rows.append({"strategy": "VAE", **vae_m})
        for bname, bdata in per_bench.items():
            rows.append({"strategy": bname, **bdata.get("bench_metrics", {})})
        bench_path = os.path.join(output_dir, "strategy_comparison.csv")
        pd.DataFrame(rows).to_csv(bench_path, index=False)
        written.append(bench_path)
        logger.info("Strategy comparison CSV saved: %s", bench_path)

    return written
