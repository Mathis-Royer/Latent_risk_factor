"""
CLI entry point for the complete walk-forward validation.

Usage:
    python scripts/run_walk_forward.py --data-path <path> [options]
    python scripts/run_walk_forward.py --synthetic [options]

Arguments:
    --data-path PATH      Path to stock data CSV (long format with core columns)
    --synthetic           Use synthetic data (50 stocks, 10 years)
    --n-stocks N          Synthetic: number of stocks (default: 50)
    --n-years N           Synthetic: history length (default: 10)
    --device DEVICE       PyTorch device: "cpu" or "cuda" (default: "cpu")
    --seed SEED           Global random seed (default: 42)
    --output-dir DIR      Output directory for results (default: "results/")
    --config PATH         Optional YAML/JSON config override

Output:
    results/
    ├── report.json       # Full structured report
    ├── report.txt        # Human-readable summary
    ├── fold_metrics.csv  # Per-fold metrics table
    └── statistical_tests.csv  # Pairwise test results

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

from src.config import PipelineConfig
from src.data_pipeline.data_loader import generate_synthetic_csv, load_stock_data
from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.features import compute_trailing_volatility
from src.integration.pipeline import FullPipeline
from src.integration.reporting import format_summary_table, serialize_for_json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return args (Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run full walk-forward validation for VAE latent risk factor pipeline.",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to stock data CSV (long format with core columns)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for development/testing",
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
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device (default: auto-detect best available)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML/JSON config override file",
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

        # Load or generate data
        if args.synthetic:
            logger.info("Generating synthetic data: %d stocks, %d years", args.n_stocks, args.n_years)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                csv_path = f.name

            start_year = 2000
            end_year = start_year + args.n_years
            generate_synthetic_csv(
                csv_path,
                n_stocks=args.n_stocks,
                start_date=f"{start_year}-01-03",
                end_date=f"{end_year}-12-31",
                seed=args.seed,
            )
            stock_data = load_stock_data(csv_path)
            start_date = f"{start_year}-01-03"
            os.unlink(csv_path)
        elif args.data_path:
            logger.info("Loading data from %s", args.data_path)
            stock_data = load_stock_data(args.data_path)
            start_date = str(stock_data["date"].min().date())
        else:
            logger.error("Must specify --data-path or --synthetic")
            return 1

        # Compute returns and trailing vol
        logger.info("Computing log-returns and trailing volatility")
        returns = compute_log_returns(stock_data)
        trailing_vol = compute_trailing_volatility(returns, window=252)

        # Configure pipeline
        config = PipelineConfig(seed=args.seed)

        # Apply config overrides from file
        if args.config:
            config = _apply_config_overrides(config, args.config)

        # For synthetic data, reduce walk-forward params to fit
        if args.synthetic:
            from dataclasses import replace
            from src.config import WalkForwardConfig
            wf = replace(
                config.walk_forward,
                total_years=args.n_years,
                min_training_years=max(3, args.n_years // 3),
                holdout_years=max(1, args.n_years // 5),
            )
            config = replace(config, walk_forward=wf)

        # Run pipeline
        pipeline = FullPipeline(config)
        results = pipeline.run(
            stock_data=stock_data,
            returns=returns,
            trailing_vol=trailing_vol,
            vix_data=None,
            start_date=start_date,
            device=args.device,
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)

        # report.json
        report_path = os.path.join(args.output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(serialize_for_json(results["report"]), f, indent=2)
        logger.info("Report saved to %s", report_path)

        # report.txt
        txt_path = os.path.join(args.output_dir, "report.txt")
        with open(txt_path, "w") as f:
            f.write(format_summary_table(results["report"]))
        logger.info("Summary saved to %s", txt_path)

        # fold_metrics.csv
        from src.walk_forward.selection import aggregate_fold_metrics
        vae_df = aggregate_fold_metrics(results["vae_results"])
        vae_df.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False)

        # Print summary
        print(format_summary_table(results["report"]))

        return 0

    except Exception as e:
        logger.exception("Walk-forward failed: %s", e)
        return 1


def _apply_config_overrides(config: PipelineConfig, config_path: str) -> PipelineConfig:
    """
    Apply overrides from a YAML or JSON config file.

    The file should contain a flat or nested dict of parameter names
    matching the PipelineConfig dataclass fields.

    :param config (PipelineConfig): Base config
    :param config_path (str): Path to YAML/JSON override file

    :return config (PipelineConfig): Updated config
    """
    from dataclasses import fields, replace

    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml") from e
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
    else:
        with open(config_path) as f:
            overrides = json.load(f)

    # Apply nested overrides: e.g. {"training": {"max_epochs": 50}}
    sub_configs: dict[str, object] = {}
    for field_info in fields(config):
        name = field_info.name
        if name in overrides and isinstance(overrides[name], dict):
            sub_obj = getattr(config, name)
            sub_configs[name] = replace(sub_obj, **overrides[name])

    # Apply top-level overrides (e.g. seed)
    top_overrides = {k: v for k, v in overrides.items() if not isinstance(v, dict)}

    return replace(config, **top_overrides, **sub_configs)


if __name__ == "__main__":
    sys.exit(main())
