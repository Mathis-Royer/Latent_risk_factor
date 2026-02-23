"""
Notebook helper functions for config assembly, JSON export, and diagnostic execution.

These functions simplify notebook cells by extracting complex logic that would
otherwise be inline, making notebook cells cleaner (1-5 lines each).
"""

import json
import logging
import shutil
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DataPipelineConfig,
    InferenceConfig,
    LossConfig,
    PipelineConfig,
    PortfolioConfig,
    RiskModelConfig,
    TrainingConfig,
    VAEArchitectureConfig,
    WalkForwardConfig,
)
from src.utils import get_optimal_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration assembly
# ---------------------------------------------------------------------------


def assemble_quick_config(
    seed: int = 42,
    n_stocks: int = 50,
    K: int = 30,
    max_epochs: int = 15,
    batch_size: int = 256,
    **overrides: Any,
) -> PipelineConfig:
    """
    Assemble quick mode configuration for fast end-to-end testing.

    :param seed (int): Random seed
    :param n_stocks (int): Number of stocks to use
    :param K (int): Latent capacity
    :param max_epochs (int): Maximum training epochs
    :param batch_size (int): Training batch size
    :param overrides: Additional config overrides (window_length, n_features, mode)

    :return config (PipelineConfig): Complete configuration
    """
    data_cfg = DataPipelineConfig(
        n_stocks=n_stocks,
        window_length=overrides.get("window_length", 504),
        n_features=overrides.get("n_features", 2),
    )

    vae_cfg = VAEArchitectureConfig(
        K=K,
        window_length=data_cfg.window_length,
        n_features=data_cfg.n_features,
    )

    loss_cfg = LossConfig(
        mode=overrides.get("mode", "P"),
    )

    training_cfg = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    inference_cfg = InferenceConfig()
    risk_cfg = RiskModelConfig()
    portfolio_cfg = PortfolioConfig()
    walk_forward_cfg = WalkForwardConfig()

    return PipelineConfig(
        data=data_cfg,
        vae=vae_cfg,
        loss=loss_cfg,
        training=training_cfg,
        inference=inference_cfg,
        risk_model=risk_cfg,
        portfolio=portfolio_cfg,
        walk_forward=walk_forward_cfg,
        seed=seed,
    )


def assemble_full_config(
    seed: int = 42,
    n_stocks: int = 0,
    K: int = 75,
    max_epochs: int = 800,
    batch_size: int = 512,
    **overrides: Any,
) -> PipelineConfig:
    """
    Assemble full production configuration.

    :param seed (int): Random seed
    :param n_stocks (int): Number of stocks (0 = all)
    :param K (int): Latent capacity
    :param max_epochs (int): Maximum training epochs
    :param batch_size (int): Training batch size
    :param overrides: Additional config overrides

    :return config (PipelineConfig): Complete configuration
    """
    data_cfg = DataPipelineConfig(
        n_stocks=n_stocks,
        window_length=overrides.get("window_length", 504),
        n_features=overrides.get("n_features", 2),
        training_stride=overrides.get("training_stride", 21),
    )

    vae_cfg = VAEArchitectureConfig(
        K=K,
        window_length=data_cfg.window_length,
        n_features=data_cfg.n_features,
        sigma_sq_init=overrides.get("sigma_sq_init", 1.0),
        sigma_sq_min=overrides.get("sigma_sq_min", 0.1),
        sigma_sq_max=overrides.get("sigma_sq_max", 2.0),
        dropout=overrides.get("dropout", 0.3),
    )

    loss_cfg = LossConfig(
        mode=overrides.get("mode", "P"),
        gamma=overrides.get("gamma", 3.0),
        lambda_co_max=overrides.get("lambda_co_max", 0.5),
    )

    training_cfg = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=overrides.get("learning_rate", 1e-3),
        patience=overrides.get("patience", 50),
        es_min_delta=overrides.get("es_min_delta", 0.5),
        lr_patience=overrides.get("lr_patience", 30),
        lr_factor=overrides.get("lr_factor", 0.75),
        compile_model=overrides.get("compile_model", True),
    )

    inference_cfg = InferenceConfig(
        au_threshold=overrides.get("au_threshold", 0.01),
        aggregation_method=overrides.get("aggregation_method", "mean"),
        aggregation_half_life=overrides.get("aggregation_half_life", 60),
    )

    risk_cfg = RiskModelConfig(
        sigma_z_eigenvalue_pct=overrides.get("sigma_z_eigenvalue_pct", 0.95),
        b_a_shrinkage_alpha=overrides.get("b_a_shrinkage_alpha", 0.0),
    )

    portfolio_cfg = PortfolioConfig(
        lambda_risk=overrides.get("lambda_risk", 252.0),
        w_max=overrides.get("w_max", 0.03),
        w_min=overrides.get("w_min", 0.001),
        w_bar=overrides.get("w_bar", 0.015),
        phi=overrides.get("phi", 15.0),
        kappa_1=overrides.get("kappa_1", 0.1),
        kappa_2=overrides.get("kappa_2", 7.5),
        delta_bar=overrides.get("delta_bar", 0.01),
        tau_max=overrides.get("tau_max", 0.30),
        alpha_grid=overrides.get("alpha_grid", [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]),
        n_starts=overrides.get("sca_n_starts", 3),
        normalize_entropy_gradient=overrides.get("normalize_entropy_gradient", True),
        entropy_budget_mode=overrides.get("entropy_budget_mode", "proportional"),
        rebalancing_frequency_days=overrides.get("rebalancing_frequency", 63),
        entropy_trigger_alpha=overrides.get("entropy_trigger_alpha", 0.90),
    )

    walk_forward_cfg = WalkForwardConfig()

    return PipelineConfig(
        data=data_cfg,
        vae=vae_cfg,
        loss=loss_cfg,
        training=training_cfg,
        inference=inference_cfg,
        risk_model=risk_cfg,
        portfolio=portfolio_cfg,
        walk_forward=walk_forward_cfg,
        seed=seed,
    )


def apply_profile_overrides(
    config: PipelineConfig,
    profile: str,
    speed_overrides: dict[str, dict[str, Any]] | None = None,
) -> PipelineConfig:
    """
    Apply profile-specific speed overrides to a config.

    :param config (PipelineConfig): Base configuration
    :param profile (str): Profile name ("quick" or "full")
    :param speed_overrides (dict | None): Profile-specific overrides
        Default: {"quick": {"n_stocks": 50, "K": 30, "max_epochs": 15, "batch_size": 256}, "full": {}}

    :return config (PipelineConfig): Modified configuration
    """
    if speed_overrides is None:
        speed_overrides = {
            "quick": {"n_stocks": 50, "K": 30, "max_epochs": 15, "batch_size": 256},
            "full": {},
        }

    overrides = speed_overrides.get(profile, {})

    if not overrides:
        return config

    new_data = dc_replace(
        config.data,
        n_stocks=overrides.get("n_stocks", config.data.n_stocks),
    )
    new_vae = dc_replace(
        config.vae,
        K=overrides.get("K", config.vae.K),
    )
    new_training = dc_replace(
        config.training,
        max_epochs=overrides.get("max_epochs", config.training.max_epochs),
        batch_size=overrides.get("batch_size", config.training.batch_size),
    )

    return dc_replace(
        config,
        data=new_data,
        vae=new_vae,
        training=new_training,
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_diagnostic_json(
    diagnostics: dict[str, Any],
    output_path: str | Path,
    include_arrays: bool = False,
) -> str:
    """
    Export validated diagnostic JSON.

    :param diagnostics (dict): Full diagnostics dict
    :param output_path (str | Path): Output file path
    :param include_arrays (bool): Whether to include large arrays (default: False)

    :return path (str): Path to written file
    """
    from src.integration.reporting import serialize_for_json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out raw arrays if not included
    if not include_arrays:
        filtered = {k: v for k, v in diagnostics.items() if not k.startswith("_raw")}
    else:
        filtered = diagnostics

    json_data = serialize_for_json(filtered)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info("Diagnostic JSON exported: %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Ticker/metadata loading helpers
# ---------------------------------------------------------------------------


def load_ticker_metadata(
    data_dir: str | Path,
) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    """
    Load ticker metadata from Tiingo files.

    :param data_dir (str | Path): Data directory containing tiingo_meta/

    :return permno_to_ticker (dict): Mapping permno -> ticker
    :return ticker_meta (dict): Ticker metadata {ticker: {"exchange": ..., "is_sp500": ...}}
    """
    meta_dir = Path(data_dir) / "tiingo_meta"
    permno_to_ticker: dict[int, str] = {}
    ticker_meta: dict[str, dict[str, Any]] = {}

    json_path = meta_dir / "ticker_to_permno.json"
    if json_path.exists():
        with open(json_path) as f:
            t2p = json.load(f)
        permno_to_ticker = {int(v): k for k, v in t2p.items()}

    csv_path = meta_dir / "supported_tickers.csv"
    if csv_path.exists():
        meta_df = pd.read_csv(csv_path)
        for _, row in meta_df.iterrows():
            ticker_meta[str(row["ticker"])] = {
                "exchange": str(row.get("exchange", "")),
                "is_sp500": bool(row.get("is_sp500", False)),
            }

    return permno_to_ticker, ticker_meta


def get_latest_market_caps(
    stock_data: pd.DataFrame | None,
) -> dict[int, float]:
    """
    Extract latest market cap per permno from stock data.

    :param stock_data (pd.DataFrame | None): Stock data with date, permno, market_cap columns

    :return latest_mcap (dict): Mapping permno -> latest market cap
    """
    latest_mcap: dict[int, float] = {}

    if stock_data is None:
        return latest_mcap

    if "date" not in stock_data.columns:
        return latest_mcap

    last_date = stock_data["date"].max()
    latest = stock_data[stock_data["date"] == last_date]

    for _, row in latest.iterrows():
        mcap = row.get("market_cap")
        if mcap is not None and pd.notna(mcap):
            permno_val = row["permno"]
            mcap_val = mcap
            latest_mcap[int(permno_val)] = float(mcap_val)  # type: ignore[arg-type]

    return latest_mcap


# ---------------------------------------------------------------------------
# Diagnostic execution wrappers
# ---------------------------------------------------------------------------


def run_diagnostic_from_notebook(
    stock_data: pd.DataFrame,
    returns: pd.DataFrame,
    trailing_vol: pd.DataFrame,
    config: PipelineConfig,
    output_dir: str = "results/diagnostic",
    device: str | None = None,
    holdout_fraction: float = 0.2,
    holdout_start: str | None = None,
    run_benchmarks: bool = True,
    generate_plots: bool = True,
    tensorboard_dir: str = "runs/diagnostic",
    profile_name: str = "full",
    data_source_label: str = "unknown",
    seed: int = 42,
    pretrained_model: str | None = None,
    run_dir: str | None = None,
    vae_checkpoint_override: str | None = None,
) -> dict[str, Any]:
    """
    Run full diagnostic pipeline from notebook context.

    Wrapper around scripts.run_diagnostic.run_diagnostic() with sensible defaults.

    :param stock_data (pd.DataFrame): Stock data
    :param returns (pd.DataFrame): Log returns
    :param trailing_vol (pd.DataFrame): Trailing volatility
    :param config (PipelineConfig): Pipeline configuration
    :param output_dir (str): Output directory
    :param device (str | None): Device (auto-detect if None)
    :param holdout_fraction (float): Fraction of data for holdout
    :param holdout_start (str | None): Override holdout start date
    :param run_benchmarks (bool): Whether to run benchmarks
    :param generate_plots (bool): Whether to generate plots
    :param tensorboard_dir (str): TensorBoard log directory
    :param profile_name (str): Profile name for logging
    :param data_source_label (str): Data source label
    :param seed (int): Random seed
    :param pretrained_model (str | None): Path to pretrained VAE
    :param run_dir (str | None): Resume from existing run directory
    :param vae_checkpoint_override (str | None): Override VAE checkpoint

    :return result (dict): Diagnostic result with keys: diagnostics, run_dir, weights, stock_ids
    """
    from scripts.run_diagnostic import run_diagnostic

    if device is None:
        device = str(get_optimal_device())

    result = run_diagnostic(
        stock_data=stock_data,
        returns=returns,
        trailing_vol=trailing_vol,
        config=config,
        output_dir=output_dir,
        device=device,
        holdout_fraction=holdout_fraction,
        holdout_start=holdout_start,
        loss_mode=config.loss.mode,
        run_benchmarks=run_benchmarks,
        generate_plots=generate_plots,
        tensorboard_dir=tensorboard_dir,
        profile_name=profile_name,
        data_source_label=data_source_label,
        seed=seed,
        pretrained_model=pretrained_model,
        run_dir=run_dir,
        vae_checkpoint_override=vae_checkpoint_override,
    )

    return result  # type: ignore[return-value]


def load_existing_run(
    run_dir: str | Path,
) -> dict[str, Any]:
    """
    Load diagnostics from existing run folder.

    :param run_dir (str | Path): Path to run directory

    :return data (dict): Loaded run data with diagnostics, weights, stock_ids
    """
    from src.integration.pipeline_state import load_run_data

    run_data = load_run_data(str(run_dir))

    # Also expose weights and stock_ids in diagnostics for compatibility
    diagnostics = run_data.get("diagnostics", {})
    if "weights" in run_data:
        diagnostics["_raw_weights"] = run_data["weights"]
    if "stock_ids" in run_data:
        diagnostics["_raw_stock_ids"] = run_data["stock_ids"]

    # Reconstruct state_bag from loaded arrays/scalars if not already present
    # This enables Section 9c-ter and Section 10 to work after reload
    if "state_bag" not in diagnostics:
        diagnostics["state_bag"] = {}
    state_bag = diagnostics["state_bag"]

    # Populate from run_data if not already in state_bag
    if "B_A" in run_data and state_bag.get("B_A") is None:
        state_bag["B_A"] = run_data["B_A"]
    if "AU" in run_data and state_bag.get("AU") is None:
        state_bag["AU"] = run_data["AU"]
    if "active_dims" in run_data and state_bag.get("active_dims") is None:
        state_bag["active_dims"] = run_data["active_dims"]
    if "stock_ids" in run_data and state_bag.get("inferred_stock_ids") is None:
        state_bag["inferred_stock_ids"] = run_data["stock_ids"]

    # Populate from other diagnostic sub-dicts as fallback
    factor_qual = diagnostics.get("factor_quality", {})
    risk_model_diag = diagnostics.get("risk_model", {})
    latent_diag = diagnostics.get("latent", {})

    if state_bag.get("latent_stability_rho") is None:
        state_bag["latent_stability_rho"] = factor_qual.get("latent_stability_rho")
    if state_bag.get("shrinkage_intensity") is None:
        state_bag["shrinkage_intensity"] = risk_model_diag.get("shrinkage_intensity")
    if state_bag.get("k_bai_ng") is None:
        state_bag["k_bai_ng"] = factor_qual.get("k_bai_ng")
    if state_bag.get("AU") is None:
        state_bag["AU"] = latent_diag.get("AU") or factor_qual.get("AU")

    return {
        "diagnostics": diagnostics,
        "run_dir": str(run_dir),
        "weights": run_data.get("weights"),
        "stock_ids": run_data.get("stock_ids", []),
        "run_data": run_data,
    }


# ---------------------------------------------------------------------------
# Consolidated display functions
# ---------------------------------------------------------------------------


def display_diagnostic_results(
    diagnostics: dict[str, Any],
    run_data: dict[str, Any],
    output_dir: str | Path,
    data_dir: str | Path | None = None,
    stock_data: pd.DataFrame | None = None,
    show_plots: bool = True,
    show_report: bool = True,
    show_holdings: bool = True,
    show_exposures: bool = True,
    show_ml_diagnostics: bool = True,
    export_zip: bool = True,
) -> dict[str, Any]:
    """
    Display all diagnostic results from a completed pipeline run.

    Consolidates notebook cells 9b through 9e-4:
    - 9b: Display PNG plots
    - 9c: Display markdown report
    - 9c-bis: Portfolio holdings table
    - 9c-ter: Factor exposure heatmap
    - 9d: Export ZIP
    - 9e-1: KL per-dim heatmap
    - 9e-2: PCA eigenvalue spectrum
    - 9e-3: Literature comparison table
    - 9e-4: VAE/PCA correlation

    :param diagnostics (dict): Full diagnostics dict
    :param run_data (dict): Loaded run data from load_run_data()
    :param output_dir (str | Path): Diagnostic output directory
    :param data_dir (str | Path | None): Data directory for ticker metadata
    :param stock_data (pd.DataFrame | None): Stock data for market cap lookup
    :param show_plots (bool): Display PNG plots
    :param show_report (bool): Display markdown report
    :param show_holdings (bool): Display portfolio holdings table
    :param show_exposures (bool): Display factor exposure heatmap
    :param show_ml_diagnostics (bool): Display ML diagnostics (KL, PCA, etc.)
    :param export_zip (bool): Export diagnostic folder as ZIP

    :return summary (dict): Display status for each component
    """
    # Defer IPython imports for non-notebook environments
    from IPython.display import Image as IPImage
    from IPython.display import Markdown, display

    import matplotlib.pyplot as plt

    from src.integration.diagnostic_report import (
        build_literature_comparison_table,
        build_portfolio_holdings_table,
    )
    from src.integration.visualization import (
        plot_factor_exposure_comparison,
        plot_kl_per_dim_heatmap,
        plot_pca_eigenvalue_spectrum,
        plot_vae_pca_correlation,
    )

    output_dir = Path(output_dir)
    summary: dict[str, Any] = {
        "plots_displayed": 0,
        "report_displayed": False,
        "holdings_displayed": False,
        "exposures_displayed": False,
        "ml_diagnostics_displayed": False,
        "zip_created": False,
    }

    # --- 9b: Display PNG plots ---
    if show_plots:
        plots_dir = output_dir / "plots"
        if plots_dir.exists():
            pngs = sorted(plots_dir.glob("*.png"))
            if pngs:
                print(f"Displaying {len(pngs)} diagnostic plots from {plots_dir}/\n")
                for p in pngs:
                    display(IPImage(filename=str(p), width=900))
                summary["plots_displayed"] = len(pngs)
            else:
                print(f"No PNG files found in {plots_dir}/")
        else:
            print(f"Plots directory not found: {plots_dir}/")

    # --- 9c: Display markdown report ---
    if show_report:
        report_path = output_dir / "diagnostic_report.md"
        if report_path.exists():
            md_text = report_path.read_text(encoding="utf-8")
            print(f"Report loaded from {report_path} ({len(md_text)} chars)\n")
            display(Markdown(md_text))
            summary["report_displayed"] = True
        else:
            print(f"Report not found: {report_path}")

    # --- 9c-bis: Portfolio holdings table ---
    if show_holdings:
        stock_ids = diagnostics.get("_raw_stock_ids", [])
        w = diagnostics.get("_raw_weights")

        if w is not None and len(stock_ids) > 0:
            w = np.asarray(w)

            # Load ticker metadata
            permno_to_ticker: dict[int, str] = {}
            ticker_meta: dict[str, dict[str, Any]] = {}
            if data_dir is not None:
                permno_to_ticker, ticker_meta = load_ticker_metadata(data_dir)

            latest_mcap = get_latest_market_caps(stock_data) if stock_data is not None else {}

            holdings = build_portfolio_holdings_table(
                weights=w,
                stock_ids=stock_ids,
                permno_to_ticker=permno_to_ticker,
                ticker_meta=ticker_meta,
                latest_mcap=latest_mcap,
            )

            print(f"Portfolio Holdings: {len(holdings)} active positions")
            print(f"  Total weight: {holdings['Weight (%)'].sum():.1f}%")
            print(f"  Weight range: {holdings['Weight (%)'].min():.2f}% – {holdings['Weight (%)'].max():.2f}%")
            n_sp500 = sum(1 for _, r in holdings.iterrows() if r["S&P 500"] == "Yes")
            if n_sp500 > 0:
                print(f"  S&P 500 members: {n_sp500}/{len(holdings)}")
            print()
            display(holdings)
            summary["holdings_displayed"] = True
        else:
            print("No portfolio weights available")

    # --- 9c-ter: Factor exposure heatmap ---
    if show_exposures:
        state = diagnostics.get("state_bag", {})
        # B_A: prefer run_data (NPY file) over state_bag (may not serialize to JSON)
        B_A = run_data.get("B_A") if run_data.get("B_A") is not None else state.get("B_A")
        w = diagnostics.get("_raw_weights")
        # AU: from state_bag or infer from B_A shape
        AU = state.get("AU", 0)
        if AU == 0 and B_A is not None:
            AU = B_A.shape[1] if len(B_A.shape) > 1 else 0

        if B_A is not None and w is not None:
            w = np.asarray(w)

            _ = plot_factor_exposure_comparison(
                B_A=B_A,
                weights=w,
                AU=AU,
                n_factors_show=min(30, AU) if AU > 0 else 30,
            )
            plt.show()

            n_selected = int(np.sum(w > 1e-8))
            print(f"\nExposure Summary:")
            print(f"  Before: {B_A.shape[0]} stocks x {AU} factors (AU)")
            print(f"  After:  {n_selected} stocks selected (w > 0)")
            print(f"  Cardinality reduction: {B_A.shape[0]} -> {n_selected} ({100 * n_selected / B_A.shape[0]:.1f}%)")
            summary["exposures_displayed"] = True
        else:
            print("Exposure data not available")

    # --- 9d: Export ZIP ---
    if export_zip:
        if output_dir.exists() and any(output_dir.iterdir()):
            zip_name = "diagnostic_export"
            archive = shutil.make_archive(str(zip_name), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
            size_mb = Path(archive).stat().st_size / (1024 * 1024)
            print(f"Archive created: {archive}  ({size_mb:.2f} MB)")
            n_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
            print(f"Contains {n_files} files from {output_dir}/")
            summary["zip_created"] = True
            summary["zip_path"] = archive
        else:
            print(f"Directory {output_dir}/ is empty or missing")

    # --- 9e: ML Diagnostics ---
    if show_ml_diagnostics:
        ml_displayed = 0

        # 9e-1: KL per-dim heatmap
        if "kl_per_dim_history" in run_data:
            kl_hist = run_data["kl_per_dim_history"]
            E, K = kl_hist.shape

            _ = plot_kl_per_dim_heatmap(kl_hist, au_threshold=0.01)
            plt.show()

            final_kl = kl_hist[-1, :]
            active_mask = final_kl > 0.01
            n_active = active_mask.sum()
            print(f"Training epochs: {E}")
            print(f"Latent capacity: K={K}")
            print(f"Active units (KL>0.01): AU={n_active} ({100 * n_active / K:.1f}%)")
            print(f"Max KL at final epoch: {final_kl.max():.3f} nats")
            if n_active > 0:
                print(f"Mean KL (active only): {final_kl[active_mask].mean():.3f} nats")
            ml_displayed += 1

        # 9e-2: PCA eigenvalue spectrum
        if "pca_eigenvalues" in run_data and "literature_comparison" in run_data:
            eigenvalues = run_data["pca_eigenvalues"]
            lit_comp = run_data["literature_comparison"]

            _ = plot_pca_eigenvalue_spectrum(eigenvalues, lit_comp)
            plt.show()

            au = lit_comp.get("vae_au", 0)
            mp_signal_count = lit_comp.get("eigenvalues_above_mp", 0)
            total_var = eigenvalues.sum()
            var_explained_au = eigenvalues[:au].sum() / total_var * 100 if au > 0 else 0
            var_explained_mp = eigenvalues[:mp_signal_count].sum() / total_var * 100 if mp_signal_count > 0 else 0
            print(f"Total variance: {total_var:.4f}")
            print(f"Variance explained by AU={au} factors: {var_explained_au:.1f}%")
            print(f"Variance explained by MP={mp_signal_count} factors: {var_explained_mp:.1f}%")
            ml_displayed += 1

        # 9e-3: Literature comparison table
        if "literature_comparison" in run_data:
            lit = run_data["literature_comparison"]

            display(Markdown("### Factor Count Comparison: VAE vs Literature Methods"))
            display(Markdown(build_literature_comparison_table(lit)))

            au = lit.get("vae_au", 0)
            mp = lit.get("eigenvalues_above_mp", 0)

            if au > 0 and mp > 0:
                ratio = au / mp
                if 0.5 <= ratio <= 2.0:
                    verdict = "VAE AU is consistent with MP signal count (within 2x)"
                    status = "[OK]"
                elif ratio > 2.0:
                    verdict = f"VAE may be over-extracting factors (AU/MP = {ratio:.1f}x)"
                    status = "[WARN]"
                else:
                    verdict = f"VAE may be under-extracting factors (AU/MP = {ratio:.1f}x)"
                    status = "[WARN]"
                print(f"\n{status} {verdict}")
            ml_displayed += 1

        # 9e-4: VAE vs PCA correlation
        if "B_A" in run_data and "pca_loadings" in run_data:
            B_vae = run_data["B_A"]
            B_pca = run_data["pca_loadings"]

            au = B_vae.shape[1]
            k_compare = min(20, au, B_pca.shape[1])

            _ = plot_vae_pca_correlation(B_vae, B_pca, k_compare=k_compare)
            plt.show()

            # Compute best match for summary
            from scipy import stats

            corr_matrix = np.zeros((k_compare, k_compare))
            for i in range(k_compare):
                for j in range(k_compare):
                    ranks_vae = stats.rankdata(B_vae[:, i])
                    ranks_pca = stats.rankdata(B_pca[:, j])
                    rho = np.corrcoef(ranks_vae, ranks_pca)[0, 1]
                    corr_matrix[i, j] = abs(rho) if np.isfinite(rho) else 0.0

            best_match = corr_matrix.max(axis=1)
            n_novel = (best_match < 0.3).sum()
            n_similar = (best_match >= 0.5).sum()
            print(f"VAE factors: {au}, PCA factors compared: {k_compare}")
            print(f"VAE factors similar to PCA (|rho|>=0.5): {n_similar} ({100 * n_similar / k_compare:.0f}%)")
            print(f"VAE factors novel (|rho|<0.3): {n_novel} ({100 * n_novel / k_compare:.0f}%)")
            print(f"Mean best-match correlation: {best_match.mean():.3f}")
            ml_displayed += 1

        summary["ml_diagnostics_displayed"] = ml_displayed > 0
        summary["ml_components_shown"] = ml_displayed

    return summary


def run_decision_synthesis(
    diagnostics: dict[str, Any],
    output_dir: str | Path,
    show_analysis: bool = True,
    show_rules_table: bool = True,
    show_causal_diagram: bool = True,
    show_recommendations: bool = True,
    export_json: bool = True,
) -> dict[str, Any]:
    """
    Run decision synthesis: root cause analysis, recommendations, JSON export.

    Consolidates notebook cells 10a through 10e:
    - 10a: Root cause analysis
    - 10b: Matched decision rules table
    - 10c: Causal chain visualization
    - 10d: Configuration recommendations
    - 10e: Validated JSON export

    :param diagnostics (dict): Full diagnostics dict with composite_scores
    :param output_dir (str | Path): Output directory for JSON export
    :param show_analysis (bool): Display root cause analysis summary
    :param show_rules_table (bool): Display matched decision rules table
    :param show_causal_diagram (bool): Display causal chain visualization
    :param show_recommendations (bool): Display configuration recommendations
    :param export_json (bool): Export validated decision_synthesis.json

    :return synthesis (dict): Analysis results and export status
    """
    # Defer IPython imports for non-notebook environments
    from IPython.display import Markdown, display

    import matplotlib.pyplot as plt

    from src.integration.action_specs import get_executable_actions
    from src.integration.decision_rules import (
        format_diagnosis_summary,
        get_root_cause_analysis,
    )
    from src.integration.diagnostic_report import (
        build_decision_rules_table,
        build_recommendations_table,
    )
    from src.integration.diagnostic_schema import (
        create_minimal_output,
        severity_from_score,
        validate_diagnostic_output,
    )
    from src.integration.visualization import plot_causal_chain_diagram

    output_dir = Path(output_dir)
    synthesis: dict[str, Any] = {
        "analysis": None,
        "matched_rules": [],
        "detected_patterns": [],
        "exec_actions": [],
        "json_valid": False,
        "json_path": None,
    }

    # --- Extract composite scores ---
    comp_scores = diagnostics.get("composite_scores", {})

    scores = {
        "solver_score": comp_scores.get("solver", {}).get("score", 100),
        "constraint_score": comp_scores.get("constraint", {}).get("score", 100),
        "covariance_score": comp_scores.get("covariance", {}).get("score", 100),
        "reconstruction_score": comp_scores.get("reconstruction", {}).get("score", 100),
        "vae_health_score": comp_scores.get("vae_health", {}).get("score", 100),
        "factor_model_score": comp_scores.get("factor_model", {}).get("score", 100),
    }

    # Extract raw metrics for pattern detection
    # Sources: state_bag (if available), factor_quality, risk_model, latent, training
    state_bag = diagnostics.get("state_bag", {})
    factor_qual = diagnostics.get("factor_quality", {})
    risk_model_diag = diagnostics.get("risk_model", {})
    latent_diag = diagnostics.get("latent", {})
    training_diag = diagnostics.get("training", {})

    raw_metrics = {
        # Prefer state_bag, fallback to diagnostic sub-dicts
        "latent_stability_rho": (
            state_bag.get("latent_stability_rho")
            or factor_qual.get("latent_stability_rho")
        ),
        "shrinkage_intensity": (
            state_bag.get("shrinkage_intensity")
            or risk_model_diag.get("shrinkage_intensity")
        ),
        "AU": (
            state_bag.get("AU")
            or latent_diag.get("AU")
            or factor_qual.get("AU")
        ),
        "k_bai_ng": (
            state_bag.get("k_bai_ng")
            or factor_qual.get("k_bai_ng")
        ),
        "condition_number": comp_scores.get("covariance", {}).get("details", {}).get("condition_number"),
        "overfit_ratio": (
            diagnostics.get("training_summary", {}).get("overfit_ratio")
            or training_diag.get("overfit_ratio")
        ),
        "explanatory_power": comp_scores.get("covariance", {}).get("details", {}).get("explanatory_power"),
    }

    # --- 10a: Root cause analysis ---
    analysis = get_root_cause_analysis(scores, raw_metrics)
    synthesis["analysis"] = analysis
    synthesis["matched_rules"] = analysis.get("matching_rules", [])
    synthesis["detected_patterns"] = analysis.get("detected_patterns", [])

    if show_analysis:
        print(format_diagnosis_summary(analysis))

    # --- 10b: Matched decision rules table ---
    if show_rules_table:
        matched_rules = analysis.get("matching_rules", [])
        if matched_rules:
            display(Markdown(f"### Matched Decision Rules ({len(matched_rules)})"))
            display(Markdown(build_decision_rules_table(matched_rules)))
        else:
            print("No issues detected - all systems nominal")

        patterns = analysis.get("detected_patterns", [])
        if patterns:
            print(f"\nDetected Patterns ({len(patterns)}):")
            for p in patterns:
                print(f"  - {p['name']}: {p['interpretation']}")
                print(f"    Recommendation: {p['recommendation']}")

    # --- 10c: Causal chain visualization ---
    if show_causal_diagram:
        _ = plot_causal_chain_diagram(analysis)
        plt.show()

    # --- 10d: Configuration recommendations ---
    overall_result = comp_scores.get("overall", {})
    priority_actions_raw = overall_result.get("priority_actions", [])

    # Fallback: use actions from matched rules
    if not priority_actions_raw:
        priority_actions_raw = []
        for rule in analysis.get("matching_rules", [])[:3]:
            for action in rule.get("actions", [])[:2]:
                priority_actions_raw.append({
                    "component": analysis.get("weakest_component", "unknown"),
                    "action": action,
                })

    exec_actions = get_executable_actions(priority_actions_raw)
    synthesis["exec_actions"] = exec_actions

    if show_recommendations:
        if exec_actions:
            display(Markdown("### Configuration Recommendations"))
            display(Markdown(build_recommendations_table(exec_actions)))
        else:
            print("No configuration changes recommended")

    # --- 10e: Validated JSON export ---
    if export_json:
        overall_score = comp_scores.get("overall", {}).get("score", 50.0)
        severity = severity_from_score(overall_score)
        verdict = (
            analysis["matching_rules"][0]["diagnosis"]
            if analysis.get("matching_rules")
            else "All systems nominal"
        )

        output = create_minimal_output(overall_score, severity, verdict)

        # Add component scores
        output["component_scores"] = {}
        for comp in ["solver", "constraint", "covariance", "reconstruction", "vae_health", "factor_model"]:
            if comp in comp_scores:
                output["component_scores"][comp] = {
                    "score": comp_scores[comp].get("score", 0),
                    "grade": comp_scores[comp].get("grade", "F"),
                    "available": True,
                }

        # Add priority actions
        output["priority_actions"] = [
            {"component": a.get("component"), "action": a.get("original_action", a.get("action", ""))}
            for a in exec_actions[:5]
        ]

        # Add key findings
        output["key_findings"] = [
            {
                "metric": "weakest_component",
                "value": analysis.get("weakest_score"),
                "interpretation": f"{analysis.get('weakest_component')} has lowest score",
            }
        ]
        for p in analysis.get("detected_patterns", [])[:3]:
            output["key_findings"].append({
                "metric": p["pattern_id"],
                "interpretation": p["interpretation"],
            })

        # Add metadata
        output["metadata"] = {
            "schema_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }

        # Validate
        is_valid, errors = validate_diagnostic_output(output)
        synthesis["json_valid"] = is_valid

        print(f"Schema Validation: {'VALID' if is_valid else 'INVALID'}")
        if errors:
            for e in errors:
                print(f"  - {e}")

        # Display JSON
        print("\n--- decision_synthesis.json ---")
        print(json.dumps(output, indent=2))

        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "decision_synthesis.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {json_path}")
        synthesis["json_path"] = str(json_path)

    return synthesis
