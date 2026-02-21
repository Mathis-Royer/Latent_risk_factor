"""
CLI entry point for comprehensive pipeline diagnostic.

Runs the full pipeline (data -> VAE training -> inference -> risk model ->
portfolio optimization -> benchmarks) with instrumented diagnostics at
every stage, and produces a detailed diagnostic report in Markdown, JSON,
and PNG formats.

Usage:
    # Quick sanity check (< 10 min)
    python scripts/run_diagnostic.py --profile quick --data-dir data/

    # Full diagnostic (1-3 hours depending on hardware)
    python scripts/run_diagnostic.py --profile full --data-dir data/

    # Synthetic data (no Tiingo required)
    python scripts/run_diagnostic.py --profile quick --synthetic

Output:
    results/diagnostic/
    ├── diagnostic_report.md    # Human-readable Markdown report
    ├── diagnostic_data.json    # Machine-readable JSON
    ├── health_checks.csv       # Health check results
    ├── training_history.csv    # Per-epoch training metrics
    ├── strategy_comparison.csv # VAE vs benchmarks
    └── plots/                  # PNG diagnostic plots
        ├── training_convergence.png
        ├── sigma_and_au.png
        ├── kl_spectrum.png
        ├── eigenvalue_spectrum.png
        ├── weight_distribution.png
        ├── frontier.png
        ├── risk_decomposition.png
        ├── benchmark_comparison.png
        └── health_summary.png
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DataPipelineConfig,
    LossConfig,
    PipelineConfig,
    PortfolioConfig,
    TrainingConfig,
    VAEArchitectureConfig,
)
from src.data_pipeline.data_loader import generate_synthetic_csv, load_stock_data, load_tiingo_data, _filter_universe
from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.features import compute_trailing_volatility
from src.integration.diagnostics import collect_diagnostics
from src.integration.diagnostic_report import save_diagnostic_report
from src.integration.diagnostic_plots import save_all_plots
from src.integration.pipeline import FullPipeline
from src.integration.pipeline_state import DiagnosticRunManager
from src.integration.reporting import export_results


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile definitions (public — importable from notebook)
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "description": "Quick sanity check with reduced parameters (< 10 min)",
        "data": {
            "n_stocks": 50,
            "training_stride": 63,  # max allowed, minimal windows
        },
        "vae": {
            "K": 30,
        },
        "training": {
            "max_epochs": 15,
            "batch_size": 256,
            "compile_model": False,
            "gradient_checkpointing": False,
        },
        "portfolio": {
            "n_starts": 2,
        },
        "holdout_fraction": 0.3,
    },
    "full": {
        "description": "Full diagnostic with production-level parameters (1-3 hours)",
        "data": {
            "n_stocks": 1000,  # 0 = no cap, use all available
            "training_stride": 21,
        },
        "vae": {
            "K": 75,
        },
        "training": {
            "max_epochs": 500,
            "batch_size": 512,
            "compile_model": False,  # Disabled due to MPS stride mismatch bug
            "gradient_checkpointing": False,
        },
        "portfolio": {
            "n_starts": 5,
        },
        "holdout_fraction": 0.2,
    },
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return args (Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run comprehensive pipeline diagnostic with detailed report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", type=str, default="quick",
        choices=list(PROFILES.keys()),
        help="Parameter profile: 'quick' for sanity check, 'full' for real diagnostic"
             " (default: quick)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/",
        help="Directory containing Tiingo parquet data (default: data/)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of Tiingo (no external data needed)",
    )
    parser.add_argument(
        "--n-stocks", type=int, default=None,
        help="Override number of stocks (default: from profile)",
    )
    parser.add_argument(
        "--n-years", type=int, default=0,
        help="Limit to last N years of data (0 = all available, default: 0)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/diagnostic",
        help="Output directory (default: results/diagnostic)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--no-benchmarks", action="store_true",
        help="Skip benchmark comparison (faster)",
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default="runs/diagnostic",
        help="TensorBoard log directory (default: runs/diagnostic)",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--loss-mode", type=str, default="P",
        choices=["P", "F", "A"],
        help="VAE loss mode (default: P)",
    )
    parser.add_argument(
        "--holdout-start", type=str, default=None,
        help="Explicit train/test split date (YYYY-MM-DD). Overrides holdout_fraction.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint if available",
    )
    parser.add_argument(
        "--force-stage", type=str, default=None,
        help="Force restart from a specific pipeline stage "
             "(e.g., INFERENCE_DONE, COVARIANCE_DONE, PORTFOLIO_DONE)",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run folder to resume from (e.g., results/diagnostic_runs/2026-02-21_143052). "
             "If not specified, creates a new timestamped folder.",
    )
    parser.add_argument(
        "--vae-checkpoint", type=str, default=None,
        help="Override VAE checkpoint path (takes priority over run folder checkpoint)",
    )
    parser.add_argument(
        "--base-dir", type=str, default="results/diagnostic_runs",
        help="Base directory for new diagnostic runs (default: results/diagnostic_runs)",
    )
    return parser.parse_args()


def build_config_from_profile(
    profile_name: str,
    *,
    n_stocks: int | None = None,
    loss_mode: str = "P",
    seed: int = 42,
) -> PipelineConfig:
    """
    Build PipelineConfig from a named profile.

    Callable from both CLI and notebook without argparse dependency.

    :param profile_name (str): Profile key in PROFILES ("quick" or "full")
    :param n_stocks (int | None): Override n_stocks (None = use profile default)
    :param loss_mode (str): VAE loss mode ("P", "F", or "A")
    :param seed (int): Random seed

    :return config (PipelineConfig): Pipeline configuration
    """
    if profile_name not in PROFILES:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Available: {list(PROFILES.keys())}"
        )

    profile = PROFILES[profile_name]
    p_data = profile.get("data", {})
    p_vae = profile.get("vae", {})
    p_training = profile.get("training", {})
    p_portfolio = profile.get("portfolio", {})

    effective_n_stocks = n_stocks if n_stocks is not None else p_data.get("n_stocks", 1000)

    data_cfg = DataPipelineConfig(
        n_stocks=effective_n_stocks,
        training_stride=p_data.get("training_stride", 21),
    )
    vae_cfg = VAEArchitectureConfig(
        K=p_vae.get("K", 200),
    )
    loss_cfg = LossConfig(
        mode=loss_mode,
    )
    training_cfg = TrainingConfig(
        max_epochs=p_training.get("max_epochs", 100),
        batch_size=p_training.get("batch_size", 512),
        compile_model=p_training.get("compile_model", True),
        gradient_checkpointing=p_training.get("gradient_checkpointing", False),
    )
    portfolio_cfg = PortfolioConfig(
        n_starts=p_portfolio.get("n_starts", 5),
    )

    return PipelineConfig(
        data=data_cfg,
        vae=vae_cfg,
        loss=loss_cfg,
        training=training_cfg,
        portfolio=portfolio_cfg,
        seed=seed,
    )


def _build_config(args: argparse.Namespace, profile: dict[str, Any]) -> PipelineConfig:
    """
    Build PipelineConfig from profile and CLI overrides (CLI wrapper).

    Delegates to build_config_from_profile().

    :param args (Namespace): CLI arguments
    :param profile (dict): Profile parameter dict

    :return config (PipelineConfig): Pipeline configuration
    """
    return build_config_from_profile(
        args.profile,
        n_stocks=args.n_stocks,
        loss_mode=args.loss_mode,
        seed=args.seed,
    )


def run_diagnostic(
    stock_data: pd.DataFrame,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    config: PipelineConfig,
    *,
    output_dir: str = "results/diagnostic",
    device: str = "auto",
    holdout_fraction: float = 0.2,
    holdout_start: str | None = None,
    loss_mode: str = "P",
    run_benchmarks: bool = True,
    generate_plots: bool = True,
    tensorboard_dir: str | None = "runs/diagnostic",
    profile_name: str = "custom",
    data_source_label: str = "unknown",
    seed: int = 42,
    pretrained_model: str | None = None,
    resume: bool = False,
    force_stage: str | None = None,
    run_dir: str | None = None,
    vae_checkpoint_override: str | None = None,
    base_dir: str = "results/diagnostic_runs",
) -> dict[str, Any]:
    """
    Run comprehensive pipeline diagnostic.

    Callable from both CLI (main()) and notebook (in-process).
    Runs pipeline, collects diagnostics, generates reports.

    :param stock_data (pd.DataFrame): Raw stock price data
    :param returns (pd.DataFrame): Log-returns (dates x stocks)
    :param trailing_vol (pd.DataFrame): Trailing volatility
    :param config (PipelineConfig): Pipeline configuration
    :param output_dir (str): Output directory for reports
    :param device (str): PyTorch device
    :param holdout_fraction (float): Fraction held out for OOS
    :param holdout_start (str | None): Explicit split date (overrides fraction)
    :param loss_mode (str): VAE loss mode (P/F/A)
    :param run_benchmarks (bool): Whether to run benchmark comparison
    :param generate_plots (bool): Whether to generate PNG plots
    :param tensorboard_dir (str | None): TensorBoard log dir (None to disable)
    :param profile_name (str): Profile label for report metadata
    :param data_source_label (str): Data source label for report metadata
    :param seed (int): Random seed
    :param pretrained_model (str | None): Path to checkpoint file.
        When provided, skips VAE training and loads the encoder from disk.
    :param resume (bool): If True, resume from last checkpoint if available.
    :param force_stage (str | None): Force restart from a specific pipeline stage.
    :param run_dir (str | None): Existing run folder for resume/load.
        If None, creates new timestamped folder under base_dir.
    :param vae_checkpoint_override (str | None): Override VAE checkpoint path.
    :param base_dir (str): Base directory for new runs.

    :return results (dict): Full diagnostic results including run_dir path
    """
    t_start = time.monotonic()
    np.random.seed(seed)

    n_stocks_actual = returns.shape[1]
    n_dates_actual = returns.shape[0]
    date_start = str(returns.index[0])[:10]
    date_end = str(returns.index[-1])[:10]

    logger.info(
        "Data: %d stocks, %d dates (%s to %s)",
        n_stocks_actual, n_dates_actual, date_start, date_end,
    )

    # ---- Step 0: Initialize run manager ----
    # Use timestamped folder structure for extended checkpointing.
    # If base_dir is default but output_dir is custom, use output_dir as base.
    effective_base = base_dir
    if base_dir == "results/diagnostic_runs" and output_dir != "results/diagnostic":
        # User specified custom output_dir but not base_dir — use output_dir
        effective_base = output_dir

    run_manager = DiagnosticRunManager(
        base_dir=effective_base,
        run_dir=run_dir,
    )
    active_output_dir = run_manager.get_output_dir()
    active_checkpoint_dir = run_manager.get_checkpoint_dir()

    # Save run configuration
    cli_args = {
        "device": device,
        "holdout_fraction": holdout_fraction,
        "holdout_start": holdout_start,
        "loss_mode": loss_mode,
        "run_benchmarks": run_benchmarks,
        "generate_plots": generate_plots,
        "profile_name": profile_name,
        "data_source_label": data_source_label,
        "seed": seed,
        "resume": resume,
        "force_stage": force_stage,
    }
    run_manager.save_run_config(asdict(config), cli_args)

    # Determine VAE checkpoint to use
    effective_pretrained = pretrained_model
    if vae_checkpoint_override is not None:
        effective_pretrained = vae_checkpoint_override
        logger.info("Using VAE checkpoint override: %s", effective_pretrained)
    elif effective_pretrained is None and resume:
        # Try to find checkpoint in run folder
        found_ckpt = run_manager.get_vae_checkpoint_path()
        if found_ckpt:
            effective_pretrained = found_ckpt
            logger.info("Found VAE checkpoint in run folder: %s", effective_pretrained)

    # ---- Step 1: Configure pipeline ----
    logger.info("Step 1/4: Configuring pipeline...")

    pipeline = FullPipeline(
        config, tensorboard_dir=tensorboard_dir,
        checkpoint_dir=active_checkpoint_dir,
    )

    hp_config = [{"mode": loss_mode, "learning_rate": config.training.learning_rate, "alpha": 1.0}]

    logger.info(
        "Config: K=%d, max_epochs=%d, batch=%d, n_starts=%d, "
        "stride=%d, holdout=%.0f%%, mode=%s",
        config.vae.K, config.training.max_epochs,
        config.training.batch_size, config.portfolio.n_starts,
        config.data.training_stride, holdout_fraction * 100,
        loss_mode,
    )

    # ---- Step 2: Run pipeline ----
    logger.info("Step 2/4: Running pipeline (direct training mode)...")
    t_run = time.monotonic()

    results = pipeline.run_direct(
        stock_data=stock_data,
        returns=returns,
        trailing_vol=trailing_vol,
        vix_data=None,
        start_date=date_start,
        hp_grid=hp_config,
        device=device,
        holdout_start=holdout_start,
        holdout_fraction=holdout_fraction,
        run_benchmarks=run_benchmarks,
        pretrained_model=effective_pretrained,
        resume=resume,
        force_stage=force_stage,
    )

    t_pipeline = time.monotonic() - t_run
    logger.info("Pipeline completed in %.1f seconds", t_pipeline)

    # Extract results
    state_bag: dict[str, Any] = results.get("state", {})
    vae_metrics: dict[str, float] = (
        results["vae_results"][0] if results["vae_results"] else {}
    )
    w_vae = results.get("weights", np.array([]))
    oos_start = results.get("oos_start", "")
    oos_end = results.get("oos_end", "")
    returns_oos = returns.loc[oos_start:oos_end] if oos_start else returns.iloc[-50:]

    # ---- Step 3: Collect diagnostics ----
    logger.info("Step 3/4: Collecting diagnostics...")

    config_dict = asdict(config)
    config_dict["_diagnostic"] = {
        "profile": profile_name,
        "data_source": data_source_label,
        "n_stocks_actual": n_stocks_actual,
        "n_dates_actual": n_dates_actual,
        "date_range": f"{date_start} to {date_end}",
        "pipeline_time_seconds": t_pipeline,
        "holdout_fraction": holdout_fraction,
        "loss_mode": loss_mode,
    }

    diagnostics = collect_diagnostics(
        state_bag=state_bag,
        vae_metrics=vae_metrics,
        benchmark_results=results.get("benchmark_results", {}),
        returns_oos=returns_oos,
        stock_data=stock_data,
        returns=returns,
        w_vae=w_vae,
        config_dict=config_dict,
    )

    # Expose lightweight data for notebook portfolio holdings cell
    diagnostics["_raw_weights"] = w_vae
    diagnostics["_raw_stock_ids"] = state_bag.get("inferred_stock_ids", [])

    # ---- Step 4: Generate reports ----
    logger.info("Step 4/4: Generating reports...")

    # Use active_output_dir (from run manager) for all outputs
    os.makedirs(active_output_dir, exist_ok=True)

    # Save standard pipeline results too
    export_results(results, config_dict, output_dir=active_output_dir)

    # Save diagnostic report (MD + JSON + CSVs) with weights and stock_ids
    inferred_stock_ids = state_bag.get("inferred_stock_ids", [])
    report_files = save_diagnostic_report(
        diagnostics,
        output_dir=active_output_dir,
        weights=w_vae,
        stock_ids=inferred_stock_ids,
    )

    # Save plots
    if generate_plots:
        plots_dir = run_manager.get_plots_dir()
        plot_files = save_all_plots(
            diagnostics, w_vae, output_dir=plots_dir,
            returns_oos=returns_oos,
            benchmark_weights=results.get("benchmark_weights", {}),
            benchmark_results=results.get("benchmark_results", {}),
            inferred_stock_ids=inferred_stock_ids,
            train_end=results.get("train_end", ""),
            oos_start=oos_start,
        )
        report_files.extend(plot_files)
    else:
        logger.info("Plot generation skipped")

    # ---- Summary ----
    t_total = time.monotonic() - t_start

    logger.info("=" * 70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time: %.1f seconds (%.1f min)", t_total, t_total / 60)
    logger.info("Files generated: %d", len(report_files))
    logger.info("Run directory: %s", run_manager.run_dir_str)
    logger.info("")

    # Print health check summary
    checks = diagnostics.get("health_checks", [])
    summary = diagnostics.get("summary", {})
    logger.info(
        "Health: %d OK, %d WARNING, %d CRITICAL",
        summary.get("n_ok", 0),
        summary.get("n_warning", 0),
        summary.get("n_critical", 0),
    )
    for c in checks:
        if c["status"] != "OK":
            logger.info(
                "  [%s] %s / %s: %s",
                c["status"], c["category"], c["check"], c["message"],
            )

    # Print key metrics
    logger.info("")
    logger.info("Key metrics:")
    logger.info("  Sharpe = %.3f", vae_metrics.get("sharpe", 0.0))
    logger.info("  Ann. Return = %.2f%%", vae_metrics.get("ann_return", 0.0) * 100)
    logger.info("  Ann. Vol = %.2f%%", vae_metrics.get("ann_vol_oos", 0.0) * 100)

    # Max DD with EW benchmark context
    vae_mdd = vae_metrics.get("max_drawdown_oos", 0.0)
    bench_res = results.get("benchmark_results", {})
    ew_folds = bench_res.get("equal_weight", [])
    ew_mdd_val = ew_folds[0].get("max_drawdown_oos", None) if ew_folds else None
    if ew_mdd_val is not None:
        logger.info("  Max DD = %.2f%% (EW benchmark: %.2f%%)", vae_mdd * 100, ew_mdd_val * 100)
    else:
        logger.info("  Max DD = %.2f%%", vae_mdd * 100)

    logger.info("  H_norm = %.4f", vae_metrics.get("H_norm_oos", 0.0))
    logger.info("  AU = %s", vae_metrics.get("AU", "?"))
    logger.info("  E* = %s", vae_metrics.get("e_star", "?"))

    logger.info("")
    logger.info("Report: %s", os.path.join(active_output_dir, "diagnostic_report.md"))

    # Return dict with diagnostics and run folder info
    return {
        "diagnostics": diagnostics,
        "run_dir": run_manager.run_dir_str,
        "weights": w_vae,
        "stock_ids": inferred_stock_ids,
        "pipeline_results": results,
    }


def main() -> int:
    """
    Main CLI entry point. Parses args, loads data, delegates to run_diagnostic().

    :return exit_code (int): 0 on success, 1 on failure
    """
    args = parse_args()
    profile = PROFILES[args.profile]

    logger.info("=" * 70)
    logger.info("VAE LATENT RISK FACTOR — DIAGNOSTIC RUN")
    logger.info("Profile: %s — %s", args.profile, profile["description"])
    logger.info("=" * 70)

    try:
        # ---- Load data ----
        logger.info("Loading data...")

        if args.synthetic:
            n_stocks_syn = args.n_stocks or profile.get("data", {}).get("n_stocks", 50)
            n_years_syn = args.n_years or 10
            logger.info(
                "Generating synthetic data: %d stocks, %d years",
                n_stocks_syn, n_years_syn,
            )
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                csv_path = f.name

            start_year = 2000
            generate_synthetic_csv(
                csv_path,
                n_stocks=n_stocks_syn,
                start_date=f"{start_year}-01-03",
                end_date=f"{start_year + n_years_syn}-12-31",
                seed=args.seed,
            )
            stock_data = load_stock_data(csv_path)
            os.unlink(csv_path)
            data_source_label = "synthetic"
        else:
            logger.info("Loading Tiingo data from %s", args.data_dir)
            stock_data = load_tiingo_data(data_dir=args.data_dir)
            n_stocks_cap = args.n_stocks or profile.get("data", {}).get("n_stocks", 0)
            stock_data = _filter_universe(stock_data, n_stocks_cap, args.n_years)
            data_source_label = "tiingo"

        # Compute returns and trailing vol
        logger.info("Computing log-returns and trailing volatility...")
        returns = compute_log_returns(stock_data)
        trailing_vol = compute_trailing_volatility(returns, window=252)

        # ---- Build config ----
        config = _build_config(args, profile)
        holdout_fraction = profile.get("holdout_fraction", 0.2)
        tb_dir = None if args.no_tensorboard else args.tensorboard_dir

        # ---- Run diagnostic ----
        run_diagnostic(
            stock_data=stock_data,
            returns=returns,
            trailing_vol=trailing_vol,
            config=config,
            output_dir=args.output_dir,
            device=args.device,
            holdout_fraction=holdout_fraction,
            holdout_start=args.holdout_start,
            loss_mode=args.loss_mode,
            run_benchmarks=not args.no_benchmarks,
            generate_plots=not args.no_plots,
            tensorboard_dir=tb_dir,
            profile_name=args.profile,
            data_source_label=data_source_label,
            seed=args.seed,
            resume=args.resume,
            force_stage=args.force_stage,
            run_dir=args.run_dir,
            vae_checkpoint_override=args.vae_checkpoint,
            base_dir=args.base_dir,
        )

        return 0

    except Exception as e:
        logger.exception("Diagnostic failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
