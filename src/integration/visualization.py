"""
Visualization helpers for notebook dashboard display.

Provides plotting and formatting functions for walk-forward results.
All functions return matplotlib figures or styled DataFrames for
inline notebook rendering.

Reference: ISD Section MOD-016 — Sub-task 4.
"""

from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fold metrics comparison
# ---------------------------------------------------------------------------


def plot_fold_metrics(
    vae_results: list[dict[str, float]],
    benchmark_results: dict[str, list[dict[str, float]]],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> object:
    """
    Plot per-fold metrics: VAE vs all benchmarks.

    :param vae_results (list[dict]): Per-fold VAE metrics
    :param benchmark_results (dict): Benchmark name -> per-fold metrics
    :param metrics (list[str] | None): Metrics to plot
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    if metrics is None:
        metrics = ["sharpe", "ann_vol_oos", "max_drawdown_oos", "H_norm_oos"]

    vae_df = pd.DataFrame(vae_results)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    fold_ids = vae_df["fold_id"].values if "fold_id" in vae_df.columns else np.arange(len(vae_df))

    for ax, metric in zip(axes, metrics):
        if metric in vae_df.columns:
            ax.plot(fold_ids, vae_df[metric].values, "o-", label="VAE", linewidth=2, markersize=5, color="#2563eb")

        for bench_name, bench_metrics in benchmark_results.items():
            bench_df = pd.DataFrame(bench_metrics)
            if metric in bench_df.columns:
                ax.plot(fold_ids, bench_df[metric].values, "s--", label=bench_name, alpha=0.7, markersize=3)

        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=3)

    axes[-1].set_xlabel("Fold ID", fontsize=11)
    fig.suptitle("Walk-Forward Fold Metrics: VAE vs Benchmarks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E* distribution
# ---------------------------------------------------------------------------


def plot_e_star_distribution(
    e_stars: list[int],
    figsize: tuple[float, float] = (8, 4),
) -> object:
    """
    Plot E* (optimal epochs) distribution across folds.

    :param e_stars (list[int]): E* values per fold
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    e_arr = np.array(e_stars)

    ax.bar(range(len(e_arr)), e_arr, color="#2563eb", alpha=0.8, edgecolor="white")
    ax.axhline(float(np.mean(e_arr)), color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(e_arr):.1f}")

    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("E* (epochs)", fontsize=11)
    ax.set_title("Optimal Training Epochs per Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests heatmap
# ---------------------------------------------------------------------------


def plot_pairwise_heatmap(
    report: dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> object:
    """
    Plot heatmap of pairwise test results (VAE vs benchmarks).

    :param report (dict): Full report from generate_report()
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    tests = report.get("statistical_tests", {})
    pairwise = tests.get("pairwise", {})

    if not pairwise:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No pairwise test results available",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    benchmarks = list(pairwise.keys())
    all_metrics: list[str] = []
    for bench_data in pairwise.values():
        all_metrics.extend(bench_data.keys())
    metrics = sorted(set(all_metrics))

    delta_matrix = np.full((len(benchmarks), len(metrics)), np.nan)
    sig_matrix = np.zeros((len(benchmarks), len(metrics)), dtype=bool)

    for i, bench in enumerate(benchmarks):
        for j, metric in enumerate(metrics):
            result = pairwise.get(bench, {}).get(metric, {})
            if result.get("skipped", False):
                continue
            delta_matrix[i, j] = result.get("median_delta", np.nan)
            sig_matrix[i, j] = result.get("significant_corrected", False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks, fontsize=9)

    for i in range(len(benchmarks)):
        for j in range(len(metrics)):
            val = delta_matrix[i, j]
            if np.isnan(val):
                continue
            marker = "*" if sig_matrix[i, j] else ""
            ax.text(j, i, f"{val:+.3f}{marker}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Median effect (VAE - Benchmark)")
    ax.set_title("Pairwise Tests: VAE vs Benchmarks (* = significant after Holm-Bonferroni)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary styling
# ---------------------------------------------------------------------------


def style_summary_table(
    summary_df: pd.DataFrame,
) -> object:
    """
    Apply conditional styling to a summary statistics DataFrame.

    :param summary_df (pd.DataFrame): From summary_statistics()

    :return styled (Styler): Styled DataFrame for notebook display
    """
    numeric_cols = [c for c in summary_df.columns if c != "metric"]
    styler = summary_df.style.format(
        {c: "{:.4f}" for c in numeric_cols},
    )
    return styler


def style_fold_table(
    fold_df: pd.DataFrame,
    highlight_cols: list[str] | None = None,
) -> object:
    """
    Apply conditional coloring to per-fold metrics DataFrame.

    Green = better, Red = worse (relative to column median).

    :param fold_df (pd.DataFrame): Per-fold metrics
    :param highlight_cols (list[str] | None): Columns to highlight

    :return styled (Styler): Styled DataFrame for notebook display
    """
    if highlight_cols is None:
        highlight_cols = ["sharpe", "H_norm_oos", "ann_vol_oos", "max_drawdown_oos"]

    numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
    styler = fold_df.style.format({c: "{:.4f}" for c in numeric_cols})
    return styler
