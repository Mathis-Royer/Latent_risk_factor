"""
Results compilation and report generation.

Aggregates VAE and benchmark results across folds, statistical tests,
and regime decomposition into a structured report.

Reference: ISD Section MOD-016 — Sub-task 1 (step 6).
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.integration.statistical_tests import (
    bootstrap_effect_size,
    holm_bonferroni_correction,
    regime_decomposition,
    wilcoxon_paired_test,
)
from src.walk_forward.selection import aggregate_fold_metrics, check_deployment_criteria


def compile_pairwise_tests(
    vae_fold_metrics: pd.DataFrame,
    benchmark_fold_metrics: dict[str, pd.DataFrame],
    primary_metrics: list[str] | None = None,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run all statistical tests: VAE vs each benchmark on each metric.

    :param vae_fold_metrics (pd.DataFrame): Per-fold VAE metrics
    :param benchmark_fold_metrics (dict): Benchmark name -> per-fold metrics
    :param primary_metrics (list[str] | None): Metrics to test
    :param n_bootstrap (int): Bootstrap iterations
    :param alpha (float): Significance level
    :param seed (int): Random seed

    :return results (dict): Per-benchmark, per-metric test results + corrected p-values
    """
    if primary_metrics is None:
        primary_metrics = [
            "H_norm_oos", "ann_vol_oos", "max_drawdown_oos", "sharpe",
        ]

    all_p_values: list[float] = []
    all_labels: list[str] = []
    raw_results: dict[str, dict[str, dict[str, Any]]] = {}

    for bench_name, bench_df in benchmark_fold_metrics.items():
        raw_results[bench_name] = {}

        for metric in primary_metrics:
            if metric not in vae_fold_metrics.columns or metric not in bench_df.columns:
                continue

            vae_vals = vae_fold_metrics[metric].dropna().values
            bench_vals = bench_df[metric].dropna().values

            n = min(len(vae_vals), len(bench_vals))
            if n < 5:
                raw_results[bench_name][metric] = {
                    "skipped": True,
                    "reason": f"Too few paired observations ({n})",
                }
                continue

            vae_paired = np.asarray(vae_vals[:n], dtype=np.float64)
            bench_paired = np.asarray(bench_vals[:n], dtype=np.float64)

            # Wilcoxon signed-rank
            wilcoxon = wilcoxon_paired_test(vae_paired, bench_paired)

            # Bootstrap effect size
            bootstrap = bootstrap_effect_size(
                vae_paired, bench_paired,
                n_bootstrap=n_bootstrap, seed=seed,
            )

            raw_results[bench_name][metric] = {
                "wilcoxon_stat": wilcoxon["statistic"],
                "p_value": wilcoxon["p_value"],
                "significant": bool(wilcoxon["p_value"] < alpha),
                "median_delta": bootstrap["median_effect"],
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "vae_median": float(np.median(vae_paired)),
                "bench_median": float(np.median(bench_paired)),
            }

            all_p_values.append(wilcoxon["p_value"])
            all_labels.append(f"{bench_name}:{metric}")

    # Holm-Bonferroni correction across all tests
    if all_p_values:
        corrected_list = holm_bonferroni_correction(all_p_values, alpha=alpha)
        for i, label in enumerate(all_labels):
            bench_name, metric = label.split(":", 1)
            if bench_name in raw_results and metric in raw_results[bench_name]:
                result = raw_results[bench_name][metric]
                if not result.get("skipped", False):
                    result["p_corrected"] = corrected_list[i]["p_value"]
                    result["significant_corrected"] = corrected_list[i]["reject"]

    return {
        "pairwise": raw_results,
        "n_tests": len(all_p_values),
        "alpha": alpha,
    }


def compile_regime_analysis(
    vae_fold_metrics: list[dict[str, float]],
    benchmark_fold_metrics: dict[str, list[dict[str, float]]],
    fold_crisis_fractions: list[float] | None = None,
    crisis_threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Decompose results by market regime (crisis vs calm).

    :param vae_fold_metrics (list[dict]): Per-fold VAE metrics
    :param benchmark_fold_metrics (dict): Benchmark per-fold metrics
    :param fold_crisis_fractions (list[float] | None): Crisis fraction per fold
    :param crisis_threshold (float): Fraction of high-VIX days

    :return regime_results (dict): Per-regime metrics for VAE and benchmarks
    """
    if fold_crisis_fractions is None:
        return {"available": False, "reason": "No crisis fractions provided"}

    vae_regime = regime_decomposition(
        vae_fold_metrics, fold_crisis_fractions, crisis_threshold=crisis_threshold,
    )

    bench_regimes: dict[str, dict[str, Any]] = {}
    for bench_name, bench_metrics in benchmark_fold_metrics.items():
        bench_regimes[bench_name] = regime_decomposition(
            bench_metrics, fold_crisis_fractions, crisis_threshold=crisis_threshold,
        )

    return {
        "available": True,
        "vae": vae_regime,
        "benchmarks": bench_regimes,
    }


def generate_report(
    vae_fold_metrics: pd.DataFrame,
    benchmark_fold_metrics: dict[str, pd.DataFrame],
    e_stars: list[int],
    fold_crisis_fractions: list[float] | None = None,
    primary_metrics: list[str] | None = None,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate the complete results report.

    :param vae_fold_metrics (pd.DataFrame): Per-fold VAE metrics
    :param benchmark_fold_metrics (dict): Benchmark per-fold metrics
    :param e_stars (list[int]): E* values per fold
    :param fold_crisis_fractions (list[float] | None): Crisis fraction per fold
    :param primary_metrics (list[str] | None): Metrics for tests
    :param n_bootstrap (int): Bootstrap iterations
    :param alpha (float): Significance level
    :param seed (int): Random seed

    :return report (dict): Complete results report
    """
    # Deployment criteria
    deployment = check_deployment_criteria(
        vae_fold_metrics, benchmark_fold_metrics,
        primary_metrics=primary_metrics,
    )

    # Statistical tests
    statistical_tests = compile_pairwise_tests(
        vae_fold_metrics, benchmark_fold_metrics,
        primary_metrics=primary_metrics,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
    )

    # Regime analysis: convert DataFrames to list[dict] for regime_decomposition
    vae_records: list[dict[str, float]] = vae_fold_metrics.to_dict("records")  # type: ignore[assignment]
    bench_records: dict[str, list[dict[str, float]]] = {
        name: df.to_dict("records")  # type: ignore[misc]
        for name, df in benchmark_fold_metrics.items()
    }
    regime_analysis = compile_regime_analysis(
        vae_records, bench_records,
        fold_crisis_fractions=fold_crisis_fractions,
    )

    # E* summary
    e_star_arr = np.array(e_stars, dtype=np.float64)
    e_star_summary = {
        "values": e_stars,
        "mean": float(np.mean(e_star_arr)) if len(e_stars) > 0 else 0.0,
        "std": float(np.std(e_star_arr)) if len(e_stars) > 0 else 0.0,
        "min": int(np.min(e_star_arr)) if len(e_stars) > 0 else 0,
        "max": int(np.max(e_star_arr)) if len(e_stars) > 0 else 0,
    }

    return {
        "deployment": deployment,
        "statistical_tests": statistical_tests,
        "regime_analysis": regime_analysis,
        "e_star_summary": e_star_summary,
        "n_folds": len(vae_fold_metrics),
        "n_benchmarks": len(benchmark_fold_metrics),
        "benchmark_names": list(benchmark_fold_metrics.keys()),
    }


def serialize_for_json(obj: object) -> object:
    """
    Make an object JSON-serializable (handles numpy, pandas types).

    :param obj (object): Object to serialize

    :return serialized (object): JSON-safe object
    """
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def format_summary_table(report: dict[str, Any]) -> str:
    """
    Format the report as a human-readable text summary.

    :param report (dict): Report from generate_report()

    :return text (str): Formatted summary
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("VAE LATENT RISK FACTOR — RESULTS REPORT")
    lines.append("=" * 70)

    # Deployment decision
    deployment = report["deployment"]
    lines.append(f"\nScenario: {deployment['scenario']}")
    lines.append(f"Recommendation: {deployment['recommendation']}")

    # E* summary
    e_star = report["e_star_summary"]
    lines.append(f"\nE* epochs: mean={e_star['mean']:.1f}, "
                 f"std={e_star['std']:.1f}, "
                 f"range=[{e_star['min']}, {e_star['max']}]")

    # Statistical tests
    tests = report["statistical_tests"]
    lines.append(f"\nStatistical Tests ({tests['n_tests']} comparisons, "
                 f"alpha={tests['alpha']}):")
    lines.append("-" * 50)

    for bench_name, metrics in tests["pairwise"].items():
        lines.append(f"\n  vs {bench_name}:")
        for metric, result in metrics.items():
            if result.get("skipped", False):
                lines.append(f"    {metric}: skipped ({result['reason']})")
                continue

            sig = "*" if result.get("significant_corrected", False) else ""
            lines.append(
                f"    {metric}: Δ={result['median_delta']:+.4f} "
                f"[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}] "
                f"p={result.get('p_corrected', result['p_value']):.4f}{sig}"
            )

    # Regime analysis
    regime = report["regime_analysis"]
    if regime.get("available", False):
        lines.append("\nRegime Decomposition:")
        lines.append("-" * 50)
        vae_regime = regime["vae"]
        lines.append(f"  Crisis folds: {vae_regime.get('n_crisis', 0)}")
        lines.append(f"  Calm folds: {vae_regime.get('n_calm', 0)}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def export_results(
    results: dict[str, Any],
    config_dict: dict[str, Any],
    output_dir: str = "results/",
) -> list[str]:
    """
    Export pipeline results to disk.

    Saves per-fold metric CSVs (VAE + each benchmark), a text report,
    a JSON report, and a config snapshot.

    :param results (dict): Pipeline results with keys vae_results,
        benchmark_results, report, e_stars
    :param config_dict (dict): Config as a dict (from dataclasses.asdict)
    :param output_dir (str): Output directory path

    :return files (list[str]): List of written file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    # VAE fold metrics
    vae_df = aggregate_fold_metrics(results["vae_results"])
    vae_path = os.path.join(output_dir, "vae_fold_metrics.csv")
    vae_df.to_csv(vae_path, index=False)
    written.append(vae_path)

    # Benchmark fold metrics
    for bench_name, bench_metrics in results["benchmark_results"].items():
        bench_df = aggregate_fold_metrics(bench_metrics)
        bench_path = os.path.join(output_dir, f"{bench_name}_fold_metrics.csv")
        bench_df.to_csv(bench_path, index=False)
        written.append(bench_path)

    # Text report
    report_txt_path = os.path.join(output_dir, "report.txt")
    with open(report_txt_path, "w") as f:
        f.write(format_summary_table(results["report"]))
    written.append(report_txt_path)

    # JSON report
    report_json_path = os.path.join(output_dir, "report.json")
    with open(report_json_path, "w") as f:
        json.dump(serialize_for_json(results["report"]), f, indent=2)
    written.append(report_json_path)

    # Config snapshot
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(serialize_for_json(config_dict), f, indent=2)
    written.append(config_path)

    return written
