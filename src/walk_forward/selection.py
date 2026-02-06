"""
Walk-forward fold scoring and final selection criterion.

Aggregates per-fold metrics, computes summary statistics, and
determines if the VAE strategy should be deployed.

Reference: ISD Section MOD-009 — Sub-task 5.
"""

from typing import Any

import numpy as np
import pandas as pd


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> pd.DataFrame:
    """
    Aggregate per-fold metrics into a DataFrame.

    :param fold_metrics (list[dict]): Metrics from each fold

    :return summary (pd.DataFrame): Fold-level metrics table
    """
    return pd.DataFrame(fold_metrics)


def summary_statistics(
    fold_metrics: pd.DataFrame,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, median, std, min, max) per metric.

    :param fold_metrics (pd.DataFrame): Per-fold metrics
    :param metric_columns (list[str] | None): Columns to summarize

    :return summary (pd.DataFrame): Summary statistics
    """
    if metric_columns is None:
        metric_columns = [c for c in fold_metrics.columns
                          if fold_metrics[c].dtype in (np.float64, np.float32, float)]

    stats: list[dict[str, Any]] = []
    for col in metric_columns:
        if col not in fold_metrics.columns:
            continue
        values_arr = np.asarray(fold_metrics[col].dropna())
        stats.append({
            "metric": col,
            "mean": float(np.mean(values_arr)),
            "median": float(np.median(values_arr)),
            "std": float(np.std(values_arr, ddof=1)) if len(values_arr) > 1 else 0.0,
            "min": float(np.min(values_arr)),
            "max": float(np.max(values_arr)),
            "count": int(len(values_arr)),
        })

    return pd.DataFrame(stats)


def check_deployment_criteria(
    vae_metrics: pd.DataFrame,
    benchmark_metrics: dict[str, pd.DataFrame],
    primary_metrics: list[str] | None = None,
    p_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Check deployment criteria: VAE vs benchmarks.

    Scenario A: VAE outperforms on ≥ 2/4 primary metrics vs all benchmarks
    Scenario C: PCA ≈ VAE (no significant difference)
    Scenario D: 1/N ≥ all

    :param vae_metrics (pd.DataFrame): VAE per-fold metrics
    :param benchmark_metrics (dict): Benchmark name → per-fold metrics
    :param primary_metrics (list[str] | None): Primary metric columns
    :param p_threshold (float): Significance threshold

    :return decision (dict): Deployment recommendation
    """
    if primary_metrics is None:
        primary_metrics = [
            "H_norm_oos", "ann_vol_oos", "max_drawdown_oos", "sharpe",
        ]

    results: dict[str, dict[str, int]] = {}

    for bench_name, bench_df in benchmark_metrics.items():
        wins = 0
        for metric in primary_metrics:
            if metric not in vae_metrics.columns or metric not in bench_df.columns:
                continue

            vae_vals = np.asarray(vae_metrics[metric].dropna())
            bench_vals = np.asarray(bench_df[metric].dropna())

            n = min(len(vae_vals), len(bench_vals))
            if n < 5:
                continue

            # For entropy: higher is better
            # For vol/MDD: lower is better
            if metric in ("H_norm_oos", "sharpe"):
                if float(np.median(vae_vals[:n])) > float(np.median(bench_vals[:n])):
                    wins += 1
            else:
                if float(np.median(vae_vals[:n])) < float(np.median(bench_vals[:n])):
                    wins += 1

        results[bench_name] = {"wins": wins, "total": len(primary_metrics)}

    # Determine scenario
    all_wins = all(v["wins"] >= 2 for v in results.values())
    pca_tie = False
    if "pca_factor_rp" in results:
        pca_tie = results["pca_factor_rp"]["wins"] <= 1

    scenario = "B"  # Default: mixed
    if all_wins:
        scenario = "A"  # VAE outperforms all
    elif pca_tie:
        scenario = "C"  # PCA ≈ VAE

    return {
        "scenario": scenario,
        "per_benchmark": results,
        "recommendation": {
            "A": "Production deployment",
            "B": "Review iterations (DVT Section 6)",
            "C": "Adopt PCA (cost /100)",
            "D": "Use 1/N or relax constraints",
            "E": "Dual-regime system",
        }.get(scenario, "Further analysis needed"),
    }
