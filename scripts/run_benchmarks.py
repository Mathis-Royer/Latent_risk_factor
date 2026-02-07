"""
CLI entry point for benchmarks only (no VAE training).

Usage:
    python scripts/run_benchmarks.py --data-path <path> [options]
    python scripts/run_benchmarks.py --synthetic [options]

Arguments:
    --data-path PATH      Path to stock data CSV (long format)
    --synthetic           Use synthetic data
    --benchmarks LIST     Comma-separated: "ew,iv,mv,erc,pca_rp,pca_vol" (default: all)
    --seed SEED           Global random seed (default: 42)
    --output-dir DIR      Output directory (default: "results/benchmarks/")
    --device DEVICE       Reserved for API consistency (benchmarks are CPU-only)

Output:
    results/benchmarks/
    ├── benchmark_metrics.csv   # Per-fold per-benchmark metrics
    └── benchmark_summary.json  # Summary statistics

Reference: ISD Section MOD-016 — Sub-task 4.
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.equal_weight import EqualWeight
from src.benchmarks.erc import EqualRiskContribution
from src.benchmarks.inverse_vol import InverseVolatility
from src.benchmarks.min_variance import MinimumVariance
from src.benchmarks.pca_factor_rp import PCAFactorRiskParity
from src.benchmarks.pca_vol import PCAVolRiskParity
from src.config import PipelineConfig
from src.data_pipeline.data_loader import generate_synthetic_csv, load_stock_data, load_tiingo_data
from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.features import compute_trailing_volatility
from src.walk_forward.folds import generate_fold_schedule
from src.walk_forward.selection import aggregate_fold_metrics, summary_statistics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BENCHMARK_REGISTRY: dict[str, type] = {
    "ew": EqualWeight,
    "iv": InverseVolatility,
    "mv": MinimumVariance,
    "erc": EqualRiskContribution,
    "pca_rp": PCAFactorRiskParity,
    "pca_vol": PCAVolRiskParity,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return args (Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run benchmark strategies on walk-forward folds (no VAE).",
    )
    parser.add_argument(
        "--data-source", type=str, default=None,
        choices=["synthetic", "tiingo", "csv"],
        help="Data source type (default: auto-detect from --data-path or --synthetic)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to stock data CSV/Parquet (for csv source)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/",
        help="Directory for Tiingo data (default: data/)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data",
    )
    parser.add_argument(
        "--n-stocks", type=int, default=50,
        help="Number of stocks for synthetic data (default: 50)",
    )
    parser.add_argument(
        "--n-years", type=int, default=10,
        help="History length in years for synthetic data (default: 10)",
    )
    parser.add_argument(
        "--benchmarks", type=str, default="ew,iv,mv,erc,pca_rp,pca_vol",
        help="Comma-separated benchmark codes (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/benchmarks/",
        help="Output directory (default: results/benchmarks/)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device (default: auto-detect best available)",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    :return exit_code (int): 0 on success, 1 on failure
    """
    args = parse_args()

    try:
        np.random.seed(args.seed)

        # Parse benchmark selection
        selected = [b.strip() for b in args.benchmarks.split(",")]
        for b in selected:
            if b not in BENCHMARK_REGISTRY:
                logger.error("Unknown benchmark: %s. Available: %s", b, list(BENCHMARK_REGISTRY.keys()))
                return 1

        # Resolve data source
        source = args.data_source
        if source is None:
            if args.synthetic:
                source = "synthetic"
            elif args.data_path:
                source = "csv"
            else:
                logger.error("Must specify --data-source, --data-path, or --synthetic")
                return 1

        # Load data
        if source == "synthetic":
            logger.info("Generating synthetic data: %d stocks, %d years", args.n_stocks, args.n_years)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                csv_path = f.name

            start_year = 2000
            end_year = start_year + args.n_years
            start_date = f"{start_year}-01-03"
            generate_synthetic_csv(
                csv_path,
                n_stocks=args.n_stocks,
                start_date=start_date,
                end_date=f"{end_year}-12-31",
                seed=args.seed,
            )
            stock_data = load_stock_data(csv_path)
            os.unlink(csv_path)
        elif source == "tiingo":
            logger.info("Loading Tiingo data from %s", args.data_dir)
            stock_data = load_tiingo_data(data_dir=args.data_dir)
            start_date = str(stock_data["date"].min().date())
        elif source == "csv":
            if not args.data_path:
                logger.error("--data-path required when --data-source=csv")
                return 1
            logger.info("Loading data from %s", args.data_path)
            stock_data = load_stock_data(args.data_path)
            start_date = str(stock_data["date"].min().date())
        else:
            logger.error("Unknown data source: %s", source)
            return 1

        # Compute returns
        logger.info("Computing log-returns and trailing volatility")
        returns = compute_log_returns(stock_data)
        trailing_vol = compute_trailing_volatility(returns, window=252)

        # Generate fold schedule
        config = PipelineConfig(seed=args.seed)

        if source == "synthetic":
            from dataclasses import replace
            from src.config import WalkForwardConfig
            wf = replace(
                config.walk_forward,
                total_years=args.n_years,
                min_training_years=max(3, args.n_years // 3),
                holdout_years=max(1, args.n_years // 5),
            )
            config = replace(config, walk_forward=wf)

        folds = generate_fold_schedule(
            start_date=start_date,
            total_years=config.walk_forward.total_years,
            min_training_years=config.walk_forward.min_training_years,
            oos_months=config.walk_forward.oos_months,
            embargo_days=config.walk_forward.embargo_days,
            holdout_years=config.walk_forward.holdout_years,
            val_years=config.walk_forward.val_years,
        )

        stock_ids_str = [str(c) for c in returns.columns]

        # Shared constraint parameters (INV-012)
        pc = config.portfolio
        constraint_params: dict[str, float] = {
            "w_max": pc.w_max,
            "w_min": pc.w_min,
            "phi": pc.phi,
            "kappa_1": pc.kappa_1,
            "kappa_2": pc.kappa_2,
            "delta_bar": pc.delta_bar,
            "tau_max": pc.tau_max,
            "lambda_risk": pc.lambda_risk,
        }

        # Run benchmarks on each fold
        all_results: dict[str, list[dict[str, float]]] = {b: [] for b in selected}
        w_olds: dict[str, np.ndarray | None] = {b: None for b in selected}

        for fold_idx, fold in enumerate(folds):
            fold_id = fold["fold_id"]
            train_end = str(fold["train_end"])
            oos_start = str(fold["oos_start"])
            oos_end = str(fold["oos_end"])
            is_first = fold_idx == 0

            train_returns = returns.loc[:train_end]
            returns_oos = returns.loc[oos_start:oos_end]

            logger.info("[Fold %s] Running benchmarks", fold_id)

            for bench_code in selected:
                bench_cls = BENCHMARK_REGISTRY[bench_code]
                benchmark = bench_cls(constraint_params=constraint_params)

                try:
                    benchmark.fit(train_returns, stock_ids_str)
                    w = benchmark.optimize(w_old=w_olds[bench_code], is_first=is_first)
                except Exception as e:
                    logger.warning("Benchmark %s failed at fold %s: %s", bench_code, fold_id, e)
                    w = np.ones(len(stock_ids_str)) / len(stock_ids_str)

                metrics = benchmark.evaluate(w, returns_oos, stock_ids_str)
                metrics["fold_id"] = float(fold_id)  # type: ignore[assignment]
                metrics["benchmark"] = bench_code  # type: ignore[assignment]
                all_results[bench_code].append(metrics)

                w_olds[bench_code] = w

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)

        # benchmark_metrics.csv: all results
        all_rows: list[dict[str, float]] = []
        for bench_code, results_list in all_results.items():
            all_rows.extend(results_list)

        metrics_df = pd.DataFrame(all_rows)
        csv_path = os.path.join(args.output_dir, "benchmark_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        logger.info("Metrics saved to %s", csv_path)

        # benchmark_summary.json
        summaries: dict[str, object] = {}
        for bench_code, results_list in all_results.items():
            bench_df = aggregate_fold_metrics(results_list)
            summary = summary_statistics(bench_df)
            summaries[bench_code] = summary.to_dict("records")

        json_path = os.path.join(args.output_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump(summaries, f, indent=2, default=_json_default)
        logger.info("Summary saved to %s", json_path)

        # Print summary table
        for bench_code, results_list in all_results.items():
            bench_df = aggregate_fold_metrics(results_list)
            summary = summary_statistics(bench_df)
            print(f"\n{'='*50}")
            print(f"Benchmark: {bench_code}")
            print(f"{'='*50}")
            print(summary.to_string(index=False))

        return 0

    except Exception as e:
        logger.exception("Benchmarks failed: %s", e)
        return 1


def _json_default(obj: object) -> object:
    """
    JSON serialization fallback.

    :param obj (object): Object to serialize

    :return serialized (object): JSON-safe value
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main())
