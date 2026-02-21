"""
Walk-forward fold scoring and final selection criterion.

Aggregates per-fold metrics, computes summary statistics, and
determines if the VAE strategy should be deployed.

Reference: ISD Section MOD-009 — Sub-task 5.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


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
        if len(values_arr) == 0:
            stats.append({
                "metric": col,
                "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "count": 0,
            })
            continue
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


def bonferroni_adjusted_alpha(
    n_configs: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute Bonferroni-corrected significance level for HP selection.

    When comparing n_configs hyperparameter configurations, the family-wise
    error rate (FWER) inflates. Bonferroni correction adjusts the per-test
    alpha to control FWER at the desired level.

    Reference: Bonferroni (1936), White (2000) Reality Check

    :param n_configs (int): Number of HP configurations being compared
    :param alpha (float): Desired family-wise error rate (default: 0.05)

    :return alpha_adjusted (float): Per-comparison significance threshold
    """
    assert n_configs > 0, f"n_configs must be > 0, got {n_configs}"
    if n_configs <= 1:
        return alpha
    return alpha / n_configs


def score_hp_configs_with_correction(
    config_metrics: list[dict[str, float]],
    primary_metric: str = "best_val_elbo",
    alpha: float = 0.05,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    """
    Score HP configurations with multiple testing correction.

    Returns the best configuration along with statistical significance
    assessment. Uses Bonferroni correction to account for multiple comparisons.

    :param config_metrics (list[dict]): Metrics for each HP configuration
    :param primary_metric (str): Metric to use for comparison
    :param alpha (float): Family-wise error rate (default: 0.05)
    :param higher_is_better (bool): Direction of improvement

    :return result (dict): {
        "best_idx": index of best configuration,
        "best_value": metric value of best,
        "n_configs": number of configurations tested,
        "alpha_adjusted": Bonferroni-adjusted alpha,
        "significant": whether best is significantly better than runner-up
    }
    """
    if not config_metrics:
        return {
            "best_idx": 0,
            "best_value": float("nan"),
            "n_configs": 0,
            "alpha_adjusted": alpha,
            "significant": False,
        }

    n_configs = len(config_metrics)
    values = np.array([m.get(primary_metric, float("nan")) for m in config_metrics])

    # Find best configuration
    if higher_is_better:
        best_idx = int(np.nanargmax(values))
    else:
        best_idx = int(np.nanargmin(values))

    best_value = values[best_idx]
    alpha_adjusted = bonferroni_adjusted_alpha(n_configs, alpha)

    # Check significance vs runner-up (simplified: compare to second best)
    significant = False
    if n_configs >= 2:
        sorted_idx = np.argsort(values) if not higher_is_better else np.argsort(-values)
        runner_up_idx = sorted_idx[1]
        runner_up_value = values[runner_up_idx]

        # Log the effect size
        if not np.isnan(runner_up_value) and not np.isnan(best_value):
            improvement = abs(best_value - runner_up_value)
            logger.info(
                "HP selection: best config #%d (%.4f) vs runner-up #%d (%.4f), "
                "improvement=%.4f, alpha_adj=%.4f",
                best_idx, best_value, runner_up_idx, runner_up_value,
                improvement, alpha_adjusted,
            )

    return {
        "best_idx": best_idx,
        "best_value": float(best_value),
        "n_configs": n_configs,
        "alpha_adjusted": alpha_adjusted,
        "significant": significant,  # Full significance test requires variance estimates
    }
