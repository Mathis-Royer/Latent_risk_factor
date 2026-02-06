"""
Full end-to-end pipeline orchestrator.

Coordinates walk-forward validation:
1. Data pipeline preparation
2. For each fold: Phase A (HP selection) → Phase B (deployment)
3. For each benchmark: fit + optimize
4. Aggregate results + holdout
5. Statistical tests
6. Generate report

Reference: ISD Section MOD-016 — Sub-tasks 1, 3.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.benchmarks.equal_weight import EqualWeight
from src.benchmarks.erc import EqualRiskContribution
from src.benchmarks.inverse_vol import InverseVolatility
from src.benchmarks.min_variance import MinimumVariance
from src.benchmarks.pca_factor_rp import PCAFactorRiskParity
from src.benchmarks.pca_vol import PCAVolRiskParity
from src.config import PipelineConfig
from src.data_pipeline.windowing import create_windows
from src.data_pipeline.features import compute_rolling_realized_vol
from src.inference.active_units import (
    compute_au_max_stat,
    filter_exposure_matrix,
    measure_active_units,
    truncate_active_dims,
)
from src.inference.composite import aggregate_profiles, infer_latent_trajectories
from src.integration.reporting import generate_report, format_summary_table
from src.portfolio.cardinality import enforce_cardinality
from src.portfolio.frontier import (
    compute_variance_entropy_frontier,
    select_operating_alpha,
)
from src.portfolio.sca_solver import multi_start_optimize, sca_optimize
from src.risk_model.covariance import assemble_risk_model, estimate_d_eps, estimate_sigma_z
from src.risk_model.factor_regression import compute_residuals, estimate_factor_returns
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.training.trainer import VAETrainer
from src.vae.build_vae import build_vae
from src.walk_forward.folds import generate_fold_schedule
from src.walk_forward.metrics import (
    factor_explanatory_power,
    portfolio_metrics,
    realized_vs_predicted_variance,
)
from src.walk_forward.phase_a import select_best_config
from src.walk_forward.phase_b import check_training_sanity, determine_e_star
from src.walk_forward.selection import aggregate_fold_metrics, summary_statistics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARK_CLASSES: dict[str, type] = {
    "equal_weight": EqualWeight,
    "inverse_vol": InverseVolatility,
    "min_variance": MinimumVariance,
    "erc": EqualRiskContribution,
    "pca_factor_rp": PCAFactorRiskParity,
    "pca_vol": PCAVolRiskParity,
}


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

    # -------------------------------------------------------------------
    # Main orchestration
    # -------------------------------------------------------------------

    def run(
        self,
        stock_data: pd.DataFrame,
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        vix_data: pd.Series | None = None,
        start_date: str = "2000-01-03",
        hp_grid: list[dict[str, Any]] | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Full walk-forward validation + holdout.

        :param stock_data (pd.DataFrame): Raw stock data (long format)
        :param returns (pd.DataFrame): Log-returns (dates × stocks)
        :param trailing_vol (pd.DataFrame): 252-day trailing vol (dates × stocks)
        :param vix_data (pd.Series | None): VIX daily close (for crisis labels)
        :param start_date (str): Data start date
        :param hp_grid (list[dict] | None): HP grid for Phase A
        :param device (str): PyTorch device

        :return results (dict): Complete walk-forward results + report
        """
        torch_device = torch.device(device)
        if hp_grid is None:
            hp_grid = self._default_hp_grid()

        # Step 1: Setup fold schedule
        self.setup(start_date)
        wf_folds = self.get_walk_forward_folds()
        holdout_fold = self.get_holdout_fold()
        n_folds = len(wf_folds)

        # Rolling realized vol for windowing (21-day)
        rolling_vol = compute_rolling_realized_vol(returns, rolling_window=21)

        # Benchmark w_old trackers (per benchmark)
        bench_w_old: dict[str, np.ndarray | None] = {
            name: None for name in BENCHMARK_CLASSES
        }

        logger.info("Starting walk-forward: %d folds + holdout", n_folds)

        # Step 2: Walk-forward folds
        for fold_idx, fold in enumerate(wf_folds):
            fold_id = int(fold["fold_id"])  # type: ignore[arg-type]
            logger.info("[Fold %d/%d] Starting", fold_id, n_folds)

            # Phase A: HP selection
            best_config = self._run_phase_a(
                fold, returns, trailing_vol, rolling_vol,
                stock_data, vix_data, hp_grid, torch_device,
            )

            if best_config is None:
                logger.warning("[Fold %d] Phase A: all configs eliminated, using defaults", fold_id)
                best_config = {"e_star": self.config.training.max_epochs, "mode": "P",
                               "learning_rate": self.config.training.learning_rate,
                               "alpha": 1.0}

            # Phase B: Deployment
            e_star_config = int(best_config.get("e_star", self.config.training.max_epochs))
            e_star = determine_e_star(
                fold_id, e_star_config, self.e_stars,
                is_holdout=False,
            )

            vae_metrics, w_vae = self._run_single_fold(
                fold, returns, trailing_vol, rolling_vol, stock_data,
                vix_data, best_config, e_star, torch_device,
            )
            self.record_vae_result(fold_id, vae_metrics, e_star)

            n_surviving = best_config.get("n_surviving", "?")
            logger.info(
                "[Fold %d/%d] Phase A: %d configs → %s surviving",
                fold_id, n_folds, len(hp_grid), n_surviving,
            )
            logger.info(
                "[Fold %d/%d] Phase B: E*=%d, AU=%s, H_norm=%.3f",
                fold_id, n_folds, e_star,
                vae_metrics.get("AU", "?"),
                vae_metrics.get("H_norm_oos", 0.0),
            )

            # Benchmarks
            bench_sharpes: list[str] = []
            for bench_name, bench_cls in BENCHMARK_CLASSES.items():
                bench_metrics = self._run_benchmark_fold(
                    bench_name, bench_cls, fold, returns, trailing_vol,
                    bench_w_old[bench_name], fold_idx == 0,
                )
                self.record_benchmark_result(bench_name, fold_id, bench_metrics)

                # Track w_old for turnover constraints
                if "w" in bench_metrics:
                    bench_w_old[bench_name] = np.array(bench_metrics.pop("w"))

                bench_sharpes.append(f"{bench_name[:2].upper()}={bench_metrics.get('sharpe', 0.0):.3f}")

            logger.info("[Fold %d/%d] Benchmarks: %s", fold_id, n_folds, ", ".join(bench_sharpes))

        # Step 3: Holdout fold
        holdout_result: dict[str, Any] | None = None
        if holdout_fold is not None:
            holdout_id = int(holdout_fold["fold_id"])  # type: ignore[arg-type]
            logger.info("[Holdout] Starting")

            e_star_holdout = determine_e_star(
                holdout_id, self.config.training.max_epochs, self.e_stars,
                is_holdout=True, all_e_stars=self.e_stars,
            )

            # Use the best config from the last WF fold
            if best_config is not None:
                holdout_config = best_config
            else:
                holdout_config = {"mode": "P", "learning_rate": self.config.training.learning_rate, "alpha": 1.0}

            holdout_metrics, _ = self._run_single_fold(
                holdout_fold, returns, trailing_vol, rolling_vol, stock_data,
                vix_data, holdout_config, e_star_holdout, torch_device,
            )
            self.record_vae_result(holdout_id, holdout_metrics, e_star_holdout)

            # Holdout benchmarks
            for bench_name, bench_cls in BENCHMARK_CLASSES.items():
                bench_metrics = self._run_benchmark_fold(
                    bench_name, bench_cls, holdout_fold, returns, trailing_vol,
                    bench_w_old[bench_name], False,
                )
                self.record_benchmark_result(bench_name, holdout_id, bench_metrics)

            holdout_result = {
                "fold_id": holdout_id,
                "e_star": e_star_holdout,
                "metrics": holdout_metrics,
            }

        # Steps 4-6: Statistical tests + Report
        vae_df = aggregate_fold_metrics(self.vae_results)
        bench_dfs: dict[str, pd.DataFrame] = {
            name: aggregate_fold_metrics(results)
            for name, results in self.benchmark_results.items()
        }

        report = generate_report(
            vae_df, bench_dfs, self.e_stars,
            fold_crisis_fractions=None,  # Crisis fractions per fold not tracked yet
        )

        logger.info("Walk-forward complete. Scenario: %s", report["deployment"]["scenario"])
        logger.info("\n%s", format_summary_table(report))

        return {
            "fold_schedule": self.fold_schedule,
            "vae_results": self.vae_results,
            "benchmark_results": self.benchmark_results,
            "e_stars": self.e_stars,
            "report": report,
            "holdout": holdout_result,
        }

    # -------------------------------------------------------------------
    # Phase A: HP selection
    # -------------------------------------------------------------------

    def _run_phase_a(
        self,
        fold: dict[str, object],
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        rolling_vol: pd.DataFrame,
        stock_data: pd.DataFrame,
        vix_data: pd.Series | None,
        hp_grid: list[dict[str, Any]],
        device: torch.device,
    ) -> dict[str, Any] | None:
        """
        Phase A: evaluate HP grid on nested validation, return best config.

        :param fold (dict): Fold specification
        :param returns (pd.DataFrame): Log-returns
        :param trailing_vol (pd.DataFrame): Trailing vol
        :param rolling_vol (pd.DataFrame): 21-day rolling vol
        :param stock_data (pd.DataFrame): Raw stock data
        :param vix_data (pd.Series | None): VIX data
        :param hp_grid (list[dict]): HP configurations to evaluate
        :param device (torch.device): Compute device

        :return best_config (dict | None): Best HP config or None
        """
        train_end = str(fold["train_end"])
        val_start = str(fold["val_start"])

        # Get universe at train_end for window creation
        stock_ids = [int(c) for c in returns.columns if str(c).isdigit()]

        # Create windows from training period
        train_returns = returns.loc[:train_end]
        train_rolling_vol = rolling_vol.loc[:train_end]

        if train_returns.empty:
            return None

        windows, metadata = create_windows(
            train_returns, train_rolling_vol, stock_ids,
            T=self.config.data.window_length,
            stride=1,
        )

        if windows.shape[0] == 0:
            return None

        # Split into train/val by date
        val_mask = metadata["end_date"] >= pd.Timestamp(val_start)
        train_mask = ~val_mask

        train_windows = windows[train_mask.values]
        val_windows = windows[val_mask.values]

        if train_windows.shape[0] == 0 or val_windows.shape[0] == 0:
            return None

        # Evaluate each HP config
        config_results: list[dict[str, Any]] = []

        for hp in hp_grid:
            try:
                result = self._evaluate_hp_config(
                    hp, train_windows, val_windows, metadata,
                    returns, trailing_vol, stock_data, fold, device,
                )
                config_results.append(result)
            except Exception as e:
                logger.warning("HP config failed: %s — %s", hp, e)

        if not config_results:
            return None

        # Select best config
        best = select_best_config(
            config_results,
            K=self.config.vae.K,
            mdd_threshold=self.config.walk_forward.score_mdd_threshold,
            lambda_pen=self.config.walk_forward.score_lambda_pen,
            lambda_est=self.config.walk_forward.score_lambda_est,
        )

        if best is not None:
            best["n_surviving"] = len([r for r in config_results
                                       if r.get("AU", 0) >= int(0.15 * self.config.vae.K)])

        return best

    def _evaluate_hp_config(
        self,
        hp: dict[str, Any],
        train_windows: torch.Tensor,
        val_windows: torch.Tensor,
        metadata: pd.DataFrame,
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        stock_data: pd.DataFrame,
        fold: dict[str, object],
        device: torch.device,
    ) -> dict[str, Any]:
        """
        Evaluate a single HP config on nested validation.

        :param hp (dict): HP config with keys: mode, learning_rate, alpha
        :param train_windows (torch.Tensor): Training windows
        :param val_windows (torch.Tensor): Validation windows
        :param metadata (pd.DataFrame): Window metadata
        :param returns (pd.DataFrame): Log-returns
        :param trailing_vol (pd.DataFrame): Trailing vol
        :param stock_data (pd.DataFrame): Raw stock data
        :param fold (dict): Fold specification
        :param device (torch.device): Compute device

        :return result (dict): Evaluation results
        """
        mode = hp.get("mode", "P")
        lr = hp.get("learning_rate", self.config.training.learning_rate)
        alpha = hp.get("alpha", 1.0)

        # Build VAE
        n_stocks = len([c for c in returns.columns if str(c).isdigit()])
        T = self.config.data.window_length
        F = self.config.data.n_features
        K = self.config.vae.K

        model, info = build_vae(
            n=n_stocks, T=T, T_annee=10, F=F, K=K,
            learn_obs_var=(mode != "F"),
        )

        # Train
        trainer = VAETrainer(
            model, loss_mode=mode, learning_rate=lr,
            gamma=self.config.loss.gamma,
            lambda_co_max=self.config.loss.lambda_co_max,
            beta_fixed=self.config.loss.beta_fixed,
            warmup_fraction=self.config.loss.warmup_fraction,
            patience=self.config.training.patience,
            lr_patience=self.config.training.lr_patience,
            lr_factor=self.config.training.lr_factor,
            device=device,
        )

        fit_result = trainer.fit(
            train_windows, val_windows,
            max_epochs=self.config.training.max_epochs,
            batch_size=self.config.training.batch_size,
        )

        # Measure AU
        AU, kl_per_dim, active_dims = measure_active_units(
            model, val_windows, device=device,
        )

        return {
            **hp,
            "AU": AU,
            "e_star": fit_result["best_epoch"],
            "best_val_elbo": fit_result["best_val_elbo"],
            "oos_train_mse_ratio": fit_result.get("overfit_ratio", 1.0),
            "n_obs": train_windows.shape[0],
            "H_oos": 0.0,  # Simplified: full entropy requires downstream pipeline
            "mdd_oos": 0.0,
            "explanatory_power": 0.5,  # Placeholder for Phase A
        }

    # -------------------------------------------------------------------
    # Phase B: Single fold deployment
    # -------------------------------------------------------------------

    def _run_single_fold(
        self,
        fold: dict[str, object],
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        rolling_vol: pd.DataFrame,
        stock_data: pd.DataFrame,
        vix_data: pd.Series | None,
        config: dict[str, Any],
        e_star: int,
        device: torch.device,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Run Phase B for a single fold: train → infer → risk model → portfolio → metrics.

        :param fold (dict): Fold specification
        :param returns (pd.DataFrame): Log-returns
        :param trailing_vol (pd.DataFrame): Trailing vol
        :param rolling_vol (pd.DataFrame): 21-day rolling vol
        :param stock_data (pd.DataFrame): Raw stock data
        :param vix_data (pd.Series | None): VIX data
        :param config (dict): HP config from Phase A
        :param e_star (int): Number of training epochs
        :param device (torch.device): Compute device

        :return metrics (dict): Fold metrics
        :return w (np.ndarray): Portfolio weights
        """
        train_start = str(fold["train_start"])
        train_end = str(fold["train_end"])
        oos_start = str(fold["oos_start"])
        oos_end = str(fold["oos_end"])
        fold_id = int(fold["fold_id"])  # type: ignore[arg-type]

        mode = config.get("mode", "P")
        lr = config.get("learning_rate", self.config.training.learning_rate)
        alpha_val = config.get("alpha", 1.0)

        # Get stock IDs from returns columns
        stock_ids = [int(c) for c in returns.columns if str(c).isdigit()]
        stock_ids_str = [str(c) for c in stock_ids]

        # 1. Create windows from FULL training period
        train_returns = returns.loc[train_start:train_end]
        train_rolling_vol = rolling_vol.loc[train_start:train_end]

        windows, metadata = create_windows(
            train_returns, train_rolling_vol, stock_ids,
            T=self.config.data.window_length,
            stride=1,
        )

        if windows.shape[0] < 10:
            return self._empty_metrics(fold_id), np.ones(len(stock_ids)) / len(stock_ids)

        # 2. Build and train VAE for E* epochs (no early stopping for Phase B)
        n_stocks = len(stock_ids)
        T = self.config.data.window_length
        F = self.config.data.n_features
        K = self.config.vae.K

        # Compute T_annee from training period
        train_days = len(train_returns)
        T_annee = max(1, train_days // 252)

        model, info = build_vae(
            n=n_stocks, T=T, T_annee=T_annee, F=F, K=K,
            learn_obs_var=(mode != "F"),
        )

        # Phase B: train on full data for E* epochs
        # Use all windows for training (no validation split)
        trainer = VAETrainer(
            model, loss_mode=mode, learning_rate=lr,
            gamma=self.config.loss.gamma,
            lambda_co_max=self.config.loss.lambda_co_max,
            beta_fixed=self.config.loss.beta_fixed,
            warmup_fraction=self.config.loss.warmup_fraction,
            patience=e_star + 1,  # Disable early stopping
            lr_patience=self.config.training.lr_patience,
            lr_factor=self.config.training.lr_factor,
            device=device,
        )

        # Use 90% for train, 10% for validation monitor (but don't stop early)
        n_train = int(0.9 * windows.shape[0])
        train_w = windows[:n_train]
        val_w = windows[n_train:]
        if val_w.shape[0] == 0:
            val_w = train_w[-10:]

        fit_result = trainer.fit(
            train_w, val_w,
            max_epochs=e_star,
            batch_size=self.config.training.batch_size,
        )

        # 3. Infer latent trajectories → B
        trajectories = infer_latent_trajectories(
            model, windows, metadata,
            batch_size=self.config.inference.batch_size,
            device=device,
        )
        B, inferred_stock_ids = aggregate_profiles(trajectories, method="mean")

        # 4. Measure AU and truncate
        AU, kl_per_dim, active_dims = measure_active_units(
            model, windows, device=device,
            au_threshold=self.config.inference.au_threshold,
        )
        au_max = compute_au_max_stat(
            n_obs=train_days,
            r_min=self.config.inference.r_min,
        )
        AU, active_dims = truncate_active_dims(AU, kl_per_dim, active_dims, au_max)

        if AU == 0:
            return self._empty_metrics(fold_id), np.ones(n_stocks) / n_stocks

        B_A = filter_exposure_matrix(B, active_dims)

        # 5. Dual rescaling
        # Build universe snapshots for estimation rescaling
        train_dates = [str(d.date()) for d in train_returns.index]
        universe_snapshots = {d: inferred_stock_ids for d in train_dates}

        B_A_by_date = rescale_estimation(
            B_A, trailing_vol.loc[train_start:train_end],
            universe_snapshots, inferred_stock_ids,
            percentile_bounds=(
                self.config.risk_model.winsorize_lo,
                self.config.risk_model.winsorize_hi,
            ),
        )

        B_A_port = rescale_portfolio(
            B_A, trailing_vol, train_end,
            inferred_stock_ids, inferred_stock_ids,
            percentile_bounds=(
                self.config.risk_model.winsorize_lo,
                self.config.risk_model.winsorize_hi,
            ),
        )

        # 6. Factor regression → z_hat
        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns.loc[train_start:train_end],
            universe_snapshots,
            conditioning_threshold=self.config.risk_model.conditioning_threshold,
            ridge_scale=self.config.risk_model.ridge_scale,
        )

        if z_hat.shape[0] < AU:
            return self._empty_metrics(fold_id), np.ones(n_stocks) / n_stocks

        # Residuals for D_eps
        residuals = compute_residuals(
            B_A_by_date, z_hat,
            returns.loc[train_start:train_end],
            universe_snapshots, valid_dates,
            inferred_stock_ids,
        )

        # 7. Covariance: Σ_z, D_ε, Σ_assets
        Sigma_z = estimate_sigma_z(z_hat)
        D_eps = estimate_d_eps(
            residuals, inferred_stock_ids,
            d_eps_floor=self.config.risk_model.d_eps_floor,
        )

        n_port = B_A_port.shape[0]
        D_eps_port = D_eps[:n_port]

        risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
        Sigma_assets = risk_model["Sigma_assets"]
        eigenvalues = risk_model["eigenvalues"]
        B_prime_port = risk_model["B_prime_port"]

        # 8. Portfolio optimization: frontier → α*, SCA → w*
        pc = self.config.portfolio
        frontier = compute_variance_entropy_frontier(
            Sigma_assets, B_prime_port, eigenvalues, D_eps_port,
            alpha_grid=pc.alpha_grid,
            lambda_risk=pc.lambda_risk,
            w_max=pc.w_max, w_min=pc.w_min,
            w_bar=pc.w_bar, phi=pc.phi,
            w_old=None, is_first=True,
            n_starts=pc.n_starts, seed=self.config.seed,
            entropy_eps=pc.entropy_eps,
        )
        alpha_opt = select_operating_alpha(frontier)

        w_opt, f_opt, H_opt = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime_port,
            eigenvalues=eigenvalues,
            D_eps=D_eps_port,
            alpha=alpha_opt,
            n_starts=pc.n_starts,
            seed=self.config.seed,
            lambda_risk=pc.lambda_risk,
            w_max=pc.w_max, w_min=pc.w_min,
            w_bar=pc.w_bar, phi=pc.phi,
            w_old=None, is_first=True,
            entropy_eps=pc.entropy_eps,
        )

        # Cardinality enforcement
        sca_kwargs = {
            "Sigma_assets": Sigma_assets,
            "B_prime": B_prime_port,
            "eigenvalues": eigenvalues,
            "alpha": alpha_opt,
            "lambda_risk": pc.lambda_risk,
            "phi": pc.phi, "w_bar": pc.w_bar,
            "w_max": pc.w_max,
            "w_old": None, "is_first": True,
            "entropy_eps": pc.entropy_eps,
        }
        w_final = enforce_cardinality(
            w_opt, B_prime_port, eigenvalues,
            w_min=pc.w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            max_eliminations=pc.max_cardinality_elim,
            entropy_eps=pc.entropy_eps,
        )

        # 9. OOS metrics
        returns_oos = returns.loc[oos_start:oos_end]

        metrics = portfolio_metrics(
            w_final, returns_oos,
            universe=inferred_stock_ids,
            H_oos=H_opt, AU=AU,
            Sigma_hat=Sigma_assets,
        )
        metrics["AU"] = float(AU)
        metrics["e_star"] = float(e_star)
        metrics["alpha_opt"] = alpha_opt

        # Layer 2: realized vs predicted variance
        if returns_oos.shape[0] > 0:
            oos_values = returns_oos.reindex(columns=inferred_stock_ids[:n_port]).fillna(0.0).values
            var_ratio = realized_vs_predicted_variance(w_final, Sigma_assets, oos_values)
            metrics["var_ratio_oos"] = var_ratio

        # Factor explanatory power
        ep = factor_explanatory_power(
            returns.loc[train_start:train_end].reindex(
                columns=inferred_stock_ids[:n_port]
            ).fillna(0.0).values,
            B_A_port, z_hat[:min(z_hat.shape[0], train_days)],
        )
        metrics["explanatory_power"] = ep

        return metrics, w_final

    # -------------------------------------------------------------------
    # Benchmarks
    # -------------------------------------------------------------------

    def _run_benchmark_fold(
        self,
        bench_name: str,
        bench_cls: type,
        fold: dict[str, object],
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        w_old: np.ndarray | None,
        is_first: bool,
    ) -> dict[str, float]:
        """
        Run a single benchmark on a single fold.

        :param bench_name (str): Benchmark name
        :param bench_cls (type): Benchmark class
        :param fold (dict): Fold specification
        :param returns (pd.DataFrame): Log-returns
        :param trailing_vol (pd.DataFrame): Trailing vol
        :param w_old (np.ndarray | None): Previous weights
        :param is_first (bool): First rebalancing flag

        :return metrics (dict): Benchmark OOS metrics
        """
        train_end = str(fold["train_end"])
        oos_start = str(fold["oos_start"])
        oos_end = str(fold["oos_end"])

        stock_ids_str = [str(c) for c in returns.columns]
        train_returns = returns.loc[:train_end]

        # Shared constraint parameters (INV-012)
        pc = self.config.portfolio
        constraint_params = {
            "w_max": pc.w_max,
            "w_min": pc.w_min,
            "phi": pc.phi,
            "kappa_1": pc.kappa_1,
            "kappa_2": pc.kappa_2,
            "delta_bar": pc.delta_bar,
            "tau_max": pc.tau_max,
            "lambda_risk": pc.lambda_risk,
        }

        benchmark = bench_cls(constraint_params=constraint_params)

        try:
            benchmark.fit(train_returns, stock_ids_str)
            w = benchmark.optimize(w_old=w_old, is_first=is_first)
        except Exception as e:
            logger.warning("Benchmark %s failed: %s", bench_name, e)
            n = len(stock_ids_str)
            w = np.ones(n) / n

        # OOS metrics
        returns_oos = returns.loc[oos_start:oos_end]
        metrics = benchmark.evaluate(w, returns_oos, stock_ids_str)

        # Rename to distinguish from VAE metrics
        renamed: dict[str, float] = {}
        for k, v in metrics.items():
            if k == "ann_vol":
                renamed["ann_vol_oos"] = v
            elif k == "max_drawdown":
                renamed["max_drawdown_oos"] = v
            else:
                renamed[k] = v

        return renamed

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _default_hp_grid(self) -> list[dict[str, Any]]:
        """
        Default HP grid: 3 modes × 2 LRs × 3 alphas = 18 configs.

        :return grid (list[dict]): HP configurations
        """
        grid: list[dict[str, Any]] = []
        for mode in ["P", "F", "A"]:
            for lr in [5e-4, 1e-3]:
                for alpha in [0.5, 1.0, 2.0]:
                    grid.append({
                        "mode": mode,
                        "learning_rate": lr,
                        "alpha": alpha,
                    })
        return grid

    def _empty_metrics(self, fold_id: int) -> dict[str, float]:
        """
        Return empty/default metrics for a failed fold.

        :param fold_id (int): Fold identifier

        :return metrics (dict): Default metrics with NaN values
        """
        return {
            "fold_id": float(fold_id),
            "H_norm_oos": 0.0,
            "ann_vol_oos": 0.0,
            "max_drawdown_oos": 0.0,
            "sharpe": 0.0,
            "AU": 0.0,
            "e_star": 0.0,
            "error": 1.0,
        }
