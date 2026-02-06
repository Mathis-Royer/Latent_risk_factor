"""
Full end-to-end pipeline orchestrator.

Coordinates walk-forward validation:
1. Data pipeline preparation
2. For each fold: Phase A (HP selection) → Phase B (deployment)
3. For each benchmark: fit + optimize
4. Aggregate results + holdout
5. Statistical tests
6. Generate report

Reference: ISD Section MOD-016 — Sub-task 1.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.walk_forward.folds import generate_fold_schedule
from src.walk_forward.metrics import portfolio_metrics
from src.walk_forward.selection import aggregate_fold_metrics, summary_statistics


class FullPipeline:
    """
    End-to-end pipeline orchestrator.

    Attributes:
        config: PipelineConfig — full configuration
        fold_schedule: list[dict] — walk-forward fold specifications
        results: dict — accumulated results per fold
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """
        :param config (PipelineConfig | None): Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.fold_schedule: list[dict[str, object]] = []
        self.vae_results: list[dict[str, float]] = []
        self.benchmark_results: dict[str, list[dict[str, float]]] = {}
        self.e_stars: list[int] = []

    def setup(self, start_date: str) -> None:
        """
        Initialize the pipeline: generate fold schedule.

        :param start_date (str): Data start date
        """
        self.fold_schedule = generate_fold_schedule(
            start_date=start_date,
            total_years=self.config.walk_forward.total_years,
            min_training_years=self.config.walk_forward.min_training_years,
            oos_months=self.config.walk_forward.oos_months,
            embargo_days=self.config.walk_forward.embargo_days,
            holdout_years=self.config.walk_forward.holdout_years,
            val_years=self.config.walk_forward.val_years,
        )

    def get_walk_forward_folds(self) -> list[dict[str, object]]:
        """
        Get non-holdout folds.

        :return folds (list[dict]): Walk-forward folds
        """
        return [f for f in self.fold_schedule if not f["is_holdout"]]

    def get_holdout_fold(self) -> dict[str, object] | None:
        """
        Get the holdout fold.

        :return fold (dict | None): Holdout fold specification
        """
        holdout = [f for f in self.fold_schedule if f["is_holdout"]]
        return holdout[0] if holdout else None

    def record_vae_result(
        self,
        fold_id: int,
        metrics: dict[str, float],
        e_star: int,
    ) -> None:
        """
        Record VAE results for a fold.

        :param fold_id (int): Fold identifier
        :param metrics (dict): Fold metrics
        :param e_star (int): E* used for this fold
        """
        metrics["fold_id"] = float(fold_id)
        self.vae_results.append(metrics)
        self.e_stars.append(e_star)

    def record_benchmark_result(
        self,
        benchmark_name: str,
        fold_id: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Record benchmark results for a fold.

        :param benchmark_name (str): Benchmark identifier
        :param fold_id (int): Fold identifier
        :param metrics (dict): Fold metrics
        """
        if benchmark_name not in self.benchmark_results:
            self.benchmark_results[benchmark_name] = []

        metrics["fold_id"] = float(fold_id)
        self.benchmark_results[benchmark_name].append(metrics)

    def get_summary(self) -> dict[str, Any]:
        """
        Generate summary of all results.

        :return summary (dict): VAE and benchmark summary statistics
        """
        vae_df = aggregate_fold_metrics(self.vae_results)
        vae_summary = summary_statistics(vae_df)

        bench_summaries: dict[str, pd.DataFrame] = {}
        for name, results in self.benchmark_results.items():
            bench_df = aggregate_fold_metrics(results)
            bench_summaries[name] = summary_statistics(bench_df)

        return {
            "vae": {
                "per_fold": vae_df,
                "summary": vae_summary,
            },
            "benchmarks": {
                name: {
                    "per_fold": aggregate_fold_metrics(results),
                    "summary": bench_summaries[name],
                }
                for name, results in self.benchmark_results.items()
            },
            "n_folds": len(self.vae_results),
            "e_stars": self.e_stars,
        }
