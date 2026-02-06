"""
Integration tests for MOD-016: E2E pipeline orchestration.

Covers: FullPipeline, statistical tests, Holm-Bonferroni correction,
regime decomposition, report generation.

Reference: ISD Section MOD-016.
"""

from typing import Any

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.integration.pipeline import FullPipeline
from src.integration.statistical_tests import (
    holm_bonferroni_correction,
    regime_decomposition,
    wilcoxon_paired_test,
)
from src.integration.reporting import (
    compile_pairwise_tests,
    format_summary_table,
    generate_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fold_metrics(
    n_folds: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Create minimal per-fold metrics DataFrame for testing."""
    np.random.seed(seed)
    return pd.DataFrame({
        "H_norm_oos": np.random.uniform(0.5, 0.9, n_folds),
        "ann_vol_oos": np.random.uniform(0.10, 0.25, n_folds),
        "max_drawdown_oos": np.random.uniform(0.05, 0.30, n_folds),
        "sharpe": np.random.uniform(0.3, 1.5, n_folds),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Tests for MOD-016 E2E integration."""

    def test_pipeline_e2e_synthetic(self) -> None:
        """FullPipeline can be instantiated and setup() works."""
        from dataclasses import replace
        from src.config import WalkForwardConfig

        config = PipelineConfig()
        wf = replace(
            config.walk_forward,
            total_years=10,
            min_training_years=3,
            holdout_years=1,
        )
        config = replace(config, walk_forward=wf)
        pipeline = FullPipeline(config=config)
        pipeline.setup(start_date="2000-01-03")

        assert len(pipeline.fold_schedule) > 0
        wf_folds = pipeline.get_walk_forward_folds()
        assert len(wf_folds) > 0
        holdout = pipeline.get_holdout_fold()
        assert holdout is not None
        assert holdout["is_holdout"] is True

    def test_statistical_tests_known(self) -> None:
        """Wilcoxon detects significant difference between N(1,1) and N(0,1)."""
        np.random.seed(42)
        n_samples = 30

        vae_scores = np.random.normal(loc=1.0, scale=1.0, size=n_samples)
        bench_scores = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

        result = wilcoxon_paired_test(vae_scores, bench_scores)

        assert result["p_value"] < 0.05, (
            f"Wilcoxon should detect N(1,1) vs N(0,1) difference, "
            f"p={result['p_value']:.4f}"
        )
        assert result["median_diff"] > 0, (
            "Median difference should be positive (VAE > benchmark)"
        )

    def test_holm_bonferroni(self) -> None:
        """Holm-Bonferroni correction produces correct rejection pattern."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.06, 0.10]
        alpha = 0.05

        results = holm_bonferroni_correction(p_values, alpha=alpha)

        assert len(results) == len(p_values)

        # Holm-Bonferroni step-down: sorted p-values are compared to
        # alpha/(m-rank+1) where rank is 1-indexed position after sorting.
        # Sorted: [0.001, 0.01, 0.03, 0.04, 0.06, 0.10], m=6
        #   rank 1: 0.001 <= 0.05/6 = 0.00833 => reject
        #   rank 2: 0.01  <= 0.05/5 = 0.01    => reject
        #   rank 3: 0.03  <= 0.05/4 = 0.0125  => NO => stop rejecting
        # So only p=0.001 and p=0.01 should be rejected
        rejected_p = [r["p_value"] for r in results if r["reject"]]
        not_rejected_p = [r["p_value"] for r in results if not r["reject"]]

        assert 0.001 in rejected_p, "p=0.001 should be rejected"
        assert 0.01 in rejected_p, "p=0.01 should be rejected"
        assert 0.03 in not_rejected_p, "p=0.03 should not be rejected"
        assert 0.06 in not_rejected_p, "p=0.06 should not be rejected"

    def test_regime_decomposition_split(self) -> None:
        """Regime decomposition correctly splits crisis/calm folds."""
        fold_metrics: list[dict[str, float]] = [
            {"sharpe": 0.5},
            {"sharpe": 1.0},
            {"sharpe": 0.3},
            {"sharpe": 1.2},
            {"sharpe": 0.4},
        ]
        fold_crisis_fractions = [0.5, 0.1, 0.3, 0.05, 0.25]
        threshold = 0.20

        result = regime_decomposition(
            fold_metrics, fold_crisis_fractions, crisis_threshold=threshold,
        )

        # Crisis: fractions > 0.20 => indices 0 (0.5), 2 (0.3), 4 (0.25)
        assert result["n_crisis"] == 3, (
            f"Expected 3 crisis folds, got {result['n_crisis']}"
        )
        # Calm: fractions <= 0.20 => indices 1 (0.1), 3 (0.05)
        assert result["n_calm"] == 2, (
            f"Expected 2 calm folds, got {result['n_calm']}"
        )

        # Verify the correct fold metrics are in each group
        crisis_sharpes = sorted([f["sharpe"] for f in result["crisis"]])
        calm_sharpes = sorted([f["sharpe"] for f in result["calm"]])

        assert crisis_sharpes == sorted([0.5, 0.3, 0.4]), (
            f"Crisis sharpes mismatch: {crisis_sharpes}"
        )
        assert calm_sharpes == sorted([1.0, 1.2]), (
            f"Calm sharpes mismatch: {calm_sharpes}"
        )

    def test_report_structure(self) -> None:
        """generate_report returns dict with required keys."""
        n_folds = 10
        vae_metrics = _make_fold_metrics(n_folds, seed=42)

        benchmark_metrics: dict[str, pd.DataFrame] = {
            "equal_weight": _make_fold_metrics(n_folds, seed=43),
            "inverse_vol": _make_fold_metrics(n_folds, seed=44),
            "min_variance": _make_fold_metrics(n_folds, seed=45),
            "erc": _make_fold_metrics(n_folds, seed=46),
            "pca_factor_rp": _make_fold_metrics(n_folds, seed=47),
            "pca_vol": _make_fold_metrics(n_folds, seed=48),
        }

        e_stars = [50 + i for i in range(n_folds)]

        report = generate_report(
            vae_metrics,
            benchmark_metrics,
            e_stars,
            fold_crisis_fractions=None,
        )

        required_keys = [
            "deployment",
            "statistical_tests",
            "regime_analysis",
            "e_star_summary",
            "n_folds",
            "n_benchmarks",
            "benchmark_names",
        ]

        for key in required_keys:
            assert key in report, f"Report missing required key: '{key}'"

        assert report["n_folds"] == n_folds
        assert report["n_benchmarks"] == 6
        assert sorted(report["benchmark_names"]) == sorted(benchmark_metrics.keys())

        # Verify format_summary_table does not crash
        summary_text = format_summary_table(report)
        assert isinstance(summary_text, str)
        assert len(summary_text) > 0

    def test_cli_benchmarks_run(self) -> None:
        """CLI entry point run_benchmarks.py completes on synthetic data."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmarks.py",
                "--synthetic",
                "--n-stocks", "30",
                "--n-years", "5",
                "--seed", "42",
                "--benchmarks", "ew,iv,mv",
                "--output-dir", "/tmp/test_bench_output",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"run_benchmarks.py failed:\nstdout={result.stdout[-500:]}\n"
            f"stderr={result.stderr[-500:]}"
        )
