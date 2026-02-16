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
import os
import time
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
from src.benchmarks.sp500_index import SP500TotalReturn
from src.config import PipelineConfig
from src.data_pipeline.windowing import create_windows
from src.data_pipeline.features import compute_rolling_realized_vol
from src.inference.active_units import (
    compute_au_max_stat,
    filter_exposure_matrix,
    truncate_active_dims,
)
from src.inference.composite import aggregate_profiles, infer_latent_trajectories
from src.integration.reporting import generate_report, format_summary_table
from src.portfolio.cardinality import enforce_cardinality
from src.portfolio.entropy import compute_entropy_only
from src.portfolio.frontier import (
    compute_variance_entropy_frontier,
    select_operating_alpha,
)
from src.portfolio.sca_solver import multi_start_optimize, sca_optimize, _safe_cholesky
from src.risk_model.covariance import assemble_risk_model, estimate_d_eps, estimate_sigma_z
from src.risk_model.factor_regression import compute_residuals, estimate_factor_returns
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.training.batching import compute_strata
from src.training.trainer import VAETrainer
from src.vae.build_vae import (
    C_MIN_DEFAULT,
    C_MIN_SMALL,
    build_vae,
    compute_channel_progression,
    compute_depth,
    compute_final_width,
    compute_temporal_sizes,
    count_decoder_params,
    count_encoder_params,
)
from src.walk_forward.folds import generate_fold_schedule
from src.walk_forward.metrics import (
    crisis_period_return,
    factor_explanatory_power_dynamic,
    portfolio_metrics,
    realized_vs_predicted_correlation,
)
from src.utils import get_optimal_device, configure_backend, clear_device_cache
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
    "sp500_index": SP500TotalReturn,
}

# Variance-targeting: Bayesian shrinkage toward 1.0 (Barra USE4 §4).
# nu_prior acts as the number of "virtual observations" where VT = 1.0.
# With nu_prior=60, a holdout of 400 days barely shrinks (400/(400+60)=0.87),
# while a short holdout of 20 days shrinks aggressively (20/(20+60)=0.25).
_VT_PRIOR_STRENGTH = 60
# Safety clamps after shrinkage (very wide — the shrinkage does the work)
_VT_SCALE_MIN = 0.1
_VT_SCALE_MAX = 10.0


def _align_weights_by_stock_id(
    w_old: np.ndarray,
    old_ids: list[int],
    new_ids: list[int],
) -> np.ndarray:
    """
    Re-index portfolio weights from a previous fold's universe to the current
    fold's universe.  Stocks present in both get their old weight; new stocks
    get weight 0; exited stocks are dropped.

    :param w_old (np.ndarray): Previous weights (n_old,)
    :param old_ids (list[int]): Stock IDs for w_old
    :param new_ids (list[int]): Stock IDs for the current fold

    :return w_aligned (np.ndarray): Aligned weights (n_new,)
    """
    old_map = {sid: i for i, sid in enumerate(old_ids)}
    w_aligned = np.zeros(len(new_ids), dtype=np.float64)
    for j, sid in enumerate(new_ids):
        idx = old_map.get(sid)
        if idx is not None:
            w_aligned[j] = w_old[idx]
    return w_aligned


class FullPipeline:
    """
    End-to-end pipeline orchestrator.

    Attributes:
        config: PipelineConfig — full configuration
        fold_schedule: list[dict] — walk-forward fold specifications
        results: dict — accumulated results per fold
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        tensorboard_dir: str | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        """
        :param config (PipelineConfig | None): Pipeline configuration
        :param tensorboard_dir (str | None): TensorBoard log directory. If set,
            training metrics are logged to TensorBoard and the dashboard is
            auto-launched in a background process.
        :param checkpoint_dir (str | None): Directory for model checkpoints.
            Defaults to 'checkpoints/' if None.
        """
        self.config = config or PipelineConfig()
        self.fold_schedule: list[dict[str, object]] = []
        self.vae_results: list[dict[str, float]] = []
        self.benchmark_results: dict[str, list[dict[str, float]]] = {}
        self.e_stars: list[int] = []
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir

    # -------------------------------------------------------------------
    # Checkpoint save / load
    # -------------------------------------------------------------------

    def save_checkpoint(
        self,
        state_bag: dict[str, Any],
        fold_info: dict[str, object],
        metrics: dict[str, float],
        weights: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        tag: str = "latest",
    ) -> str:
        """
        Save model state_dict + metadata to disk after training.

        :param state_bag (dict): Internal state from _run_single_fold
        :param fold_info (dict): Fold specification (dates, fold_id)
        :param metrics (dict): Fold metrics
        :param weights (np.ndarray): Portfolio weights
        :param checkpoint_dir (str): Output directory
        :param tag (str): Checkpoint name tag (e.g. 'latest', 'fold_03')

        :return path (str): Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        bp = state_bag.get("build_params", {})
        specs = (
            f"N{bp.get('n', '?')}_Y{bp.get('T_annee', '?')}"
            f"_T{bp.get('T', '?')}_K{bp.get('K', '?')}_F{bp.get('F', '?')}"
        )
        path = os.path.join(checkpoint_dir, f"checkpoint_{tag}_{specs}.pt")

        checkpoint: dict[str, Any] = {
            # Model reconstruction
            "model_state_dict": state_bag.get("model_state_dict", {}),
            "build_params": state_bag.get("build_params", {}),
            "vae_info": state_bag.get("vae_info", {}),
            # Training
            "fit_result": (
                {
                    k: v for k, v in fit_res.items()
                    if k != "model"  # Exclude model reference
                }
                if (fit_res := state_bag.get("fit_result")) is not None
                else {}
            ),
            # Inference
            "AU": state_bag.get("AU", 0),
            "active_dims": state_bag.get("active_dims"),
            # Fold & metrics
            "fold_info": fold_info,
            "metrics": metrics,
            "weights": weights,
            # Config (serializable fields only)
            "config": {
                "data": {
                    "window_length": self.config.data.window_length,
                    "n_features": self.config.data.n_features,
                },
                "vae": {"K": self.config.vae.K},
                "seed": self.config.seed,
            },
        }

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
        return path

    @staticmethod
    def load_checkpoint(
        path: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Load a saved checkpoint and reconstruct the VAE model.

        :param path (str): Path to checkpoint file
        :param device (str): Device for model loading

        :return checkpoint (dict): Checkpoint with reconstructed 'model' key
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        build_params = checkpoint.get("build_params", {})

        if build_params:
            model, info = build_vae(**build_params)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            checkpoint["model"] = model
            checkpoint["vae_info"] = info
            logger.info(
                "Model restored from %s (AU=%s, device=%s)",
                path, checkpoint.get("AU", "?"), device,
            )

        return checkpoint

    def _adapt_vae_params(
        self,
        n_stocks: int,
        T_annee: int,
    ) -> dict[str, Any]:
        """
        Auto-adapt K, c_min, r_max, and dropout for the actual universe size.

        Three levers applied in order:
        1. Cap K at min(K_config, max(2 * AU_max_stat, 10))
        2. If r > r_max with C_MIN_DEFAULT, try C_MIN_SMALL (= C_HEAD = 144)
        3. If r still > r_max, relax r_max to r * 1.1 + set dropout=0.2

        :param n_stocks (int): Number of stocks in the universe
        :param T_annee (int): Training history length in years

        :return adapted (dict): Keys: K, r_max, c_min, dropout
        """
        T = self.config.data.window_length
        F = self.config.data.n_features
        K_config = self.config.vae.K
        r_max_config = self.config.vae.r_max

        # Step 1: Cap K based on AU_max_stat (binding downstream constraint)
        n_obs = T_annee * 252
        au_max = compute_au_max_stat(
            n_obs=n_obs,
            r_min=self.config.inference.r_min,
            ewma_half_life=self.config.risk_model.sigma_z_ewma_half_life,
        )
        K_adapted = min(K_config, max(2 * au_max, 10))

        # Step 2: Compute capacity ratio
        L = compute_depth(T)
        T_hist = T_annee * 252
        N_capacity = max(n_stocks * (T_hist - T + 1), 1)

        def _compute_r(K: int, c_min: int) -> float:
            C_L = compute_final_width(K, c_min=c_min)
            channels = compute_channel_progression(L, C_L)
            temporal_sizes = compute_temporal_sizes(T, L)
            T_comp = temporal_sizes[-1]
            P_enc = count_encoder_params(F, K, channels)
            P_dec = count_decoder_params(F, K, channels, T_comp)
            return (P_enc + P_dec) / N_capacity

        c_min = C_MIN_DEFAULT
        r = _compute_r(K_adapted, c_min)

        # Step 3: If r > r_max, try reducing c_min
        if r > r_max_config:
            c_min = C_MIN_SMALL
            r = _compute_r(K_adapted, c_min)

        # Step 4: If still > r_max, relax r_max with 10% headroom
        r_max_adapted = r_max_config
        dropout = self.config.vae.dropout

        if r > r_max_config:
            r_max_adapted = r * 1.1
            dropout = max(dropout, 0.2)  # reinforce regularization for small universes
            logger.warning(
                "Small universe adaptation: n=%d, T_annee=%d, K=%d->%d, "
                "c_min=%d->%d, r=%.2f > r_max=%.1f. "
                "Relaxing r_max to %.2f with reinforced regularization "
                "(dropout=%.1f).",
                n_stocks, T_annee, K_config, K_adapted,
                C_MIN_DEFAULT, c_min, r, r_max_config,
                r_max_adapted, dropout,
            )
        elif c_min < C_MIN_DEFAULT:
            if r > 5.0:
                dropout = max(dropout, 0.2)  # reinforce regularization for small universes
            logger.info(
                "Small universe adaptation: n=%d, K=%d->%d, c_min=%d->%d, "
                "r=%.2f (within r_max=%.1f)%s.",
                n_stocks, K_config, K_adapted, C_MIN_DEFAULT, c_min, r,
                r_max_config,
                " [reinforced reg]" if dropout > 0.1 else "",
            )
        elif K_adapted < K_config:
            logger.info(
                "K adapted: %d -> %d (AU_max_stat=%d), r=%.2f.",
                K_config, K_adapted, au_max, r,
            )

        return {
            "K": K_adapted,
            "r_max": r_max_adapted,
            "c_min": c_min,
            "dropout": dropout,
        }

    def _tb_log_dir(self, fold_id: int, phase: str, hp_idx: int = 0,
                     hp: dict[str, Any] | None = None) -> str | None:
        """
        Build TensorBoard log directory path for a training run.

        :param fold_id (int): Fold identifier
        :param phase (str): "phase_a" or "phase_b"
        :param hp_idx (int): HP config index (Phase A only)
        :param hp (dict | None): HP config dict for naming (Phase A only)

        :return log_dir (str | None): Log directory path, or None if TB disabled
        """
        if self.tensorboard_dir is None:
            return None
        if phase == "phase_a" and hp is not None:
            mode = hp.get("mode", "X")
            lr = hp.get("learning_rate", 0)
            run_name = f"config_{hp_idx:02d}_mode_{mode}_lr_{lr}"
            return os.path.join(
                self.tensorboard_dir, f"fold_{fold_id:02d}", "phase_a", run_name,
            )
        return os.path.join(
            self.tensorboard_dir, f"fold_{fold_id:02d}", phase,
        )


    def _compute_window_strata(
        self,
        returns: pd.DataFrame,
        stock_ids: list[int],
        metadata: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute per-window stratum assignments via k-means on trailing returns.

        :param returns (pd.DataFrame): Log-returns (dates x stocks, int columns)
        :param stock_ids (list[int]): Stock permno IDs
        :param metadata (pd.DataFrame): Window metadata with stock_id column

        :return window_strata (np.ndarray): Stratum per window (N,)
        """
        # compute_strata expects string columns and string stock_ids
        ret_str = returns.copy()
        ret_str.columns = pd.Index([str(c) for c in returns.columns])
        ids_str = [str(s) for s in stock_ids]
        stock_strata = compute_strata(
            ret_str, ids_str, seed=self.config.seed,
        )
        # Map per-stock strata → per-window strata via metadata["stock_id"]
        strata_map = {sid: int(stock_strata[i]) for i, sid in enumerate(stock_ids)}
        mapped = metadata["stock_id"].map(strata_map).fillna(0).astype(int)  # type: ignore[arg-type]
        window_strata: np.ndarray = np.asarray(mapped.values, dtype=int)
        return window_strata

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
        device: str = "auto",
        skip_phase_a: bool = False,
        pretrained_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Full walk-forward validation + holdout.

        When pretrained_model is a checkpoint path (str), the saved encoder is
        loaded once and reused for every fold, skipping VAE training entirely.

        :param stock_data (pd.DataFrame): Raw stock data (long format)
        :param returns (pd.DataFrame): Log-returns (dates × stocks)
        :param trailing_vol (pd.DataFrame): 252-day trailing vol (dates × stocks)
        :param vix_data (pd.Series | None): VIX daily close (for crisis labels)
        :param start_date (str): Data start date
        :param hp_grid (list[dict] | None): HP grid for Phase A
        :param device (str): PyTorch device
        :param skip_phase_a (bool): Skip Phase A HP selection, use default config
        :param pretrained_model (str | None): Path to checkpoint file.
            If None, trains from scratch each fold. If str, loads encoder once
            and reuses it for all folds (skips training).

        :return results (dict): Complete walk-forward results + report
        """
        torch_device = get_optimal_device() if device == "auto" else torch.device(device)
        configure_backend(torch_device)
        if hp_grid is None and not skip_phase_a:
            hp_grid = self._default_hp_grid()

        # Load pretrained model if checkpoint path provided
        loaded_model = None
        if pretrained_model is not None:
            logger.info("Loading pretrained encoder from: %s", pretrained_model)
            ckpt = self.load_checkpoint(pretrained_model, device=str(torch_device))
            loaded_model = ckpt.get("model")
            if loaded_model is None:
                raise ValueError(
                    f"Checkpoint has no model (missing build_params?): {pretrained_model}"
                )
            bp = ckpt.get("build_params", {})
            if bp.get("T") and bp["T"] != self.config.data.window_length:
                raise ValueError(
                    f"Checkpoint T={bp['T']} != config T={self.config.data.window_length}"
                )
            if bp.get("F") and bp["F"] != self.config.data.n_features:
                raise ValueError(
                    f"Checkpoint F={bp['F']} != config F={self.config.data.n_features}"
                )

        # Ensure TensorBoard log directory exists
        if self.tensorboard_dir is not None:
            os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Step 1: Setup fold schedule
        self.setup(start_date)
        wf_folds = self.get_walk_forward_folds()
        holdout_fold = self.get_holdout_fold()
        n_folds = len(wf_folds)

        # Rolling realized vol for windowing (21-day)
        rolling_vol = compute_rolling_realized_vol(returns, rolling_window=21)

        # Weight trackers for turnover measurement (inter-fold)
        w_old_vae: np.ndarray | None = None
        w_old_vae_ids: list[int] | None = None
        bench_w_old: dict[str, np.ndarray | None] = {
            name: None for name in BENCHMARK_CLASSES
        }

        logger.info("Starting walk-forward: %d folds + holdout", n_folds)

        # Default config (used when Phase A is skipped or fails)
        default_config: dict[str, Any] = {
            "e_star": self.config.training.max_epochs, "mode": "P",
            "learning_rate": self.config.training.learning_rate,
            "alpha": 1.0,
        }

        # Step 2: Walk-forward folds
        for fold_idx, fold in enumerate(wf_folds):
            fold_id = int(fold["fold_id"])  # type: ignore[arg-type]

            # Phase A: HP selection (or skip)
            if skip_phase_a:
                logger.info("[Fold %d/%d] Phase A skipped, using default config", fold_id, n_folds)
                best_config = default_config
            else:
                logger.info("[Fold %d/%d] Phase A: HP selection", fold_id, n_folds)
                best_config = self._run_phase_a(
                    fold, returns, trailing_vol, rolling_vol,
                    stock_data, vix_data, hp_grid or [], torch_device,
                )

                if best_config is None:
                    logger.warning("[Fold %d] Phase A: all configs eliminated, using defaults", fold_id)
                    best_config = default_config

            # Phase B: Deployment
            logger.info("[Fold %d/%d] Phase B: deployment training", fold_id, n_folds)
            e_star_config = int(best_config.get("e_star", self.config.training.max_epochs))
            e_star = determine_e_star(
                fold_id, e_star_config, self.e_stars,
                is_holdout=False,
            )

            vae_metrics, w_vae, vae_ids = self._run_single_fold(
                fold, returns, trailing_vol, rolling_vol, stock_data,
                vix_data, best_config, e_star, torch_device,
                use_early_stopping=skip_phase_a,
                pretrained_model=loaded_model,
                w_old=w_old_vae,
                is_first=(fold_idx == 0),
                w_old_ids=w_old_vae_ids,
            )

            # Turnover measurement and net-of-cost Sharpe
            # (DeMiguel et al. 2009, Novy-Marx & Velikov 2016)
            # Align by stock ID to handle universe changes between folds.
            tc_bps = self.config.portfolio.transaction_cost_bps
            if w_old_vae is not None and w_old_vae_ids is not None and w_vae is not None:
                w_old_aligned = _align_weights_by_stock_id(
                    w_old_vae, w_old_vae_ids, vae_ids,
                )
                turnover = 0.5 * float(np.sum(np.abs(w_vae - w_old_aligned)))
                vae_metrics["turnover"] = turnover
                tc_cost = (tc_bps / 1e4) * turnover
                vae_metrics["transaction_cost"] = tc_cost
                n_oos_days = vae_metrics.get("n_days_oos", 126.0)
                ann_tc = tc_cost * (252.0 / max(n_oos_days, 1.0))
                ann_return_net = vae_metrics.get("ann_return", 0.0) - ann_tc
                ann_vol = max(vae_metrics.get("ann_vol_oos", 1e-10), 1e-10)
                vae_metrics["sharpe_net"] = ann_return_net / ann_vol
            else:
                vae_metrics["turnover"] = 0.0
                vae_metrics["transaction_cost"] = 0.0
                vae_metrics["sharpe_net"] = vae_metrics.get("sharpe", 0.0)
            if w_vae is not None:
                w_old_vae = w_vae.copy()
                w_old_vae_ids = vae_ids

            self.record_vae_result(fold_id, vae_metrics, e_star)

            # Benchmarks
            for bench_name, bench_cls in BENCHMARK_CLASSES.items():
                bench_metrics = self._run_benchmark_fold(
                    bench_name, bench_cls, fold, returns, trailing_vol,
                    bench_w_old[bench_name], fold_idx == 0,
                )
                self.record_benchmark_result(bench_name, fold_id, bench_metrics)

                # Track w_old for turnover constraints
                if "w" in bench_metrics:
                    bench_w_old[bench_name] = np.array(bench_metrics.pop("w"))

            logger.info(
                "[Fold %d/%d] E*=%d, AU=%s, H=%.3f, Sharpe=%.3f",
                fold_id, n_folds, e_star,
                vae_metrics.get("AU", "?"),
                vae_metrics.get("H_norm_oos", 0.0),
                vae_metrics.get("sharpe", 0.0),
            )

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

            holdout_metrics, _, _ = self._run_single_fold(
                holdout_fold, returns, trailing_vol, rolling_vol, stock_data,
                vix_data, holdout_config, e_star_holdout, torch_device,
                use_early_stopping=skip_phase_a,
                pretrained_model=loaded_model,
                w_old=w_old_vae,
                is_first=(w_old_vae is None),
                w_old_ids=w_old_vae_ids,
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
        logger.info("Computing statistical tests and generating report")
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
    # Direct training (no walk-forward)
    # -------------------------------------------------------------------

    def run_direct(
        self,
        stock_data: pd.DataFrame,
        returns: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        vix_data: pd.Series | None = None,
        start_date: str = "2000-01-03",
        hp_grid: list[dict[str, Any]] | None = None,
        device: str = "auto",
        skip_phase_a: bool = True,
        holdout_start: str | None = None,
        holdout_fraction: float = 0.1,
        run_benchmarks: bool = True,
        pretrained_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Direct training mode: single train/holdout split, no walk-forward.

        Same signature as run() plus holdout_start/holdout_fraction. Trains
        the VAE on [data_start, holdout_start) with early stopping, evaluates
        on [holdout_start, data_end], optionally runs benchmarks on the same split.

        When pretrained_model is a checkpoint path (str), the saved encoder is
        loaded and training is skipped entirely. Only the portfolio optimization
        head (inference → risk model → portfolio) is re-run.

        :param stock_data (pd.DataFrame): Raw stock data (long format)
        :param returns (pd.DataFrame): Log-returns (dates x stocks)
        :param trailing_vol (pd.DataFrame): 252-day trailing vol (dates x stocks)
        :param vix_data (pd.Series | None): VIX daily close for crisis metrics
        :param start_date (str): Data start date (unused, accepted for API compat)
        :param hp_grid (list[dict] | None): HP configs. First entry used (or defaults)
        :param device (str): PyTorch device ("auto", "cpu", "mps", "cuda")
        :param skip_phase_a (bool): Accepted for API compat (always True)
        :param holdout_start (str | None): Explicit train/test split date.
            If None, computed as (1 - holdout_fraction) of the date range.
        :param holdout_fraction (float): Fraction of dates for holdout (default 0.1)
        :param run_benchmarks (bool): Run 6 benchmarks on the same split
        :param pretrained_model (str | None): Path to checkpoint file.
            If None, trains from scratch. If str, loads encoder and skips training.

        :return results (dict): Results dict compatible with run() output
        """
        torch_device = (
            get_optimal_device() if device == "auto" else torch.device(device)
        )
        configure_backend(torch_device)

        # Ensure TensorBoard log directory exists
        if self.tensorboard_dir is not None:
            os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Reset accumulated state from any previous run
        self.vae_results = []
        self.benchmark_results = {}
        self.e_stars = []
        self.fold_schedule = []

        # --- Step 1: Compute train/test split ---
        all_dates = returns.index
        if holdout_start is None:
            split_idx = int(len(all_dates) * (1.0 - holdout_fraction))
            holdout_start_ts: pd.Timestamp = pd.Timestamp(all_dates[split_idx])  # type: ignore[assignment]
        else:
            holdout_start_ts = pd.Timestamp(holdout_start)  # type: ignore[assignment]

        train_dates = all_dates[all_dates < holdout_start_ts]
        test_dates = all_dates[all_dates >= holdout_start_ts]

        if len(train_dates) < self.config.data.window_length:
            raise ValueError(
                f"Training period too short: {len(train_dates)} days "
                f"< window_length {self.config.data.window_length}"
            )
        if len(test_dates) == 0:
            raise ValueError(
                f"No test data after holdout_start={holdout_start_ts}"
            )

        _first: pd.Timestamp = pd.Timestamp(str(all_dates[0]))  # type: ignore[assignment]
        _train_last: pd.Timestamp = pd.Timestamp(str(train_dates[-1]))  # type: ignore[assignment]
        _test_first: pd.Timestamp = pd.Timestamp(str(test_dates[0]))  # type: ignore[assignment]
        _test_last: pd.Timestamp = pd.Timestamp(str(test_dates[-1]))  # type: ignore[assignment]

        data_start = str(_first.date())
        train_end = str(_train_last.date())
        oos_start = str(_test_first.date())
        oos_end = str(_test_last.date())

        logger.info(
            "Direct training mode: train [%s, %s] (%d days), "
            "test [%s, %s] (%d days)",
            data_start, train_end, len(train_dates),
            oos_start, oos_end, len(test_dates),
        )

        # --- Step 2: Build synthetic fold ---
        fold: dict[str, object] = {
            "fold_id": 0,
            "train_start": data_start,
            "train_end": train_end,
            "val_start": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "is_holdout": False,
        }
        self.fold_schedule = [fold]

        # --- Step 3: HP config ---
        if hp_grid and len(hp_grid) > 0:
            hp_config = hp_grid[0]
            if len(hp_grid) > 1:
                logger.info(
                    "Direct mode: using first HP config from grid (%d configs provided)",
                    len(hp_grid),
                )
        else:
            hp_config = {
                "mode": self.config.loss.mode,
                "learning_rate": self.config.training.learning_rate,
                "alpha": 1.0,
            }

        # --- Step 4: Compute rolling vol ---
        rolling_vol = compute_rolling_realized_vol(returns, rolling_window=21)

        # --- Step 5: Train (or load pretrained encoder) ---
        e_star = self.config.training.max_epochs
        loaded_model = None

        if pretrained_model is not None:
            logger.info("Loading pretrained encoder from: %s", pretrained_model)
            ckpt = self.load_checkpoint(pretrained_model, device=str(torch_device))
            loaded_model = ckpt.get("model")
            if loaded_model is None:
                raise ValueError(
                    f"Checkpoint has no model (missing build_params?): {pretrained_model}"
                )
            # Validate T/F compatibility
            bp = ckpt.get("build_params", {})
            if bp.get("T") and bp["T"] != self.config.data.window_length:
                raise ValueError(
                    f"Checkpoint T={bp['T']} != config T={self.config.data.window_length}"
                )
            if bp.get("F") and bp["F"] != self.config.data.n_features:
                raise ValueError(
                    f"Checkpoint F={bp['F']} != config F={self.config.data.n_features}"
                )
            # Preserve original best_epoch for metrics
            ckpt_fit = ckpt.get("fit_result")
            if ckpt_fit and isinstance(ckpt_fit, dict):
                e_star = int(ckpt_fit.get("best_epoch", 0))

        state_bag: dict[str, Any] = {}
        vae_metrics, w_vae, _ = self._run_single_fold(
            fold, returns, trailing_vol, rolling_vol, stock_data,
            vix_data, hp_config, e_star, torch_device,
            use_early_stopping=True,
            _state_bag=state_bag,
            pretrained_model=loaded_model,
        )

        # When loading from checkpoint, carry over build_params and vae_info
        # (needed for descriptive checkpoint filenames and metadata)
        if loaded_model is not None:
            state_bag.setdefault("build_params", ckpt.get("build_params", {}))
            state_bag.setdefault("vae_info", ckpt.get("vae_info", {}))
            state_bag.setdefault("fit_result", ckpt.get("fit_result"))

        # Fix E* bug: recover true best_epoch from fit_result instead of using max_epochs
        if state_bag.get("fit_result") is not None:
            e_star = state_bag["fit_result"]["best_epoch"]

        self.record_vae_result(0, vae_metrics, e_star)

        # --- Step 6: Benchmarks (optional) ---
        benchmark_weights: dict[str, dict[str, Any]] = {}
        if run_benchmarks:
            for bench_name, bench_cls in BENCHMARK_CLASSES.items():
                bench_metrics = self._run_benchmark_fold(
                    bench_name, bench_cls, fold, returns, trailing_vol,
                    None, True,
                    store_weights=True,
                )
                # Extract weight/return data before recording metrics
                bw_entry: dict[str, Any] = {}
                if "_weights" in bench_metrics:
                    bw_entry["weights"] = bench_metrics.pop("_weights")
                if "_universe" in bench_metrics:
                    bw_entry["universe"] = bench_metrics.pop("_universe")
                if "_daily_returns" in bench_metrics:
                    bw_entry["daily_returns"] = bench_metrics.pop("_daily_returns")
                benchmark_weights[bench_name] = bw_entry
                self.record_benchmark_result(bench_name, 0, bench_metrics)

        # --- Step 7: Generate report ---
        vae_df = aggregate_fold_metrics(self.vae_results)
        bench_dfs: dict[str, pd.DataFrame] = {
            name: aggregate_fold_metrics(results_list)
            for name, results_list in self.benchmark_results.items()
        }
        report = generate_report(
            vae_df, bench_dfs, self.e_stars,
            fold_crisis_fractions=None,
        )

        logger.info(
            "Direct training complete. Sharpe=%.3f, AU=%s",
            vae_metrics.get("sharpe", 0.0),
            vae_metrics.get("AU", "?"),
        )

        # --- Step 7b: Return results (same keys as run() + extras) ---
        return {
            "fold_schedule": self.fold_schedule,
            "vae_results": self.vae_results,
            "benchmark_results": self.benchmark_results,
            "e_stars": self.e_stars,
            "report": report,
            "holdout": None,
            # Direct-mode extras
            "direct_mode": True,
            "state": state_bag,
            "weights": w_vae,
            "benchmark_weights": benchmark_weights,
            "holdout_start": str(holdout_start_ts.date()),
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
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
        train_start = str(fold["train_start"])
        val_start = str(fold["val_start"])

        # Get universe at train_end for window creation
        stock_ids = [int(c) for c in returns.columns if str(c).isdigit()]

        # Create windows from training period
        train_returns = returns.loc[:train_end]
        train_rolling_vol = rolling_vol.loc[:train_end]

        if train_returns.empty:
            return None

        windows, metadata, raw_ret = create_windows(
            train_returns, train_rolling_vol, stock_ids,
            T=self.config.data.window_length,
            stride=self.config.data.training_stride,
        )

        if windows.shape[0] == 0:
            return None

        # Split into train/val by date
        val_mask = metadata["end_date"] >= pd.Timestamp(val_start)
        train_mask = ~val_mask

        train_windows = windows[train_mask.values.copy()].clone()
        val_windows = windows[val_mask.values.copy()].clone()

        if train_windows.shape[0] == 0 or val_windows.shape[0] == 0:
            return None

        # Co-movement data: raw returns + strata (split to training portion)
        train_raw = raw_ret[train_mask.values.copy()].clone()
        window_strata = self._compute_window_strata(
            train_returns, stock_ids, metadata,
        )
        train_strata = window_strata[train_mask.values.copy()]
        train_metadata: pd.DataFrame = metadata[train_mask.values.copy()].reset_index(drop=True)  # type: ignore[assignment]

        # Adapt VAE params for this fold's universe size
        n_stocks = len(stock_ids)
        train_days = len(train_returns)
        T_annee = max(1, train_days // 252)
        adapted = self._adapt_vae_params(n_stocks, T_annee)

        # Evaluate each HP config
        config_results: list[dict[str, Any]] = []

        for hp_idx, hp in enumerate(hp_grid):
            logger.info(
                "  [Phase A] Config %d/%d: mode=%s lr=%s",
                hp_idx + 1, len(hp_grid),
                hp.get("mode", "?"), hp.get("learning_rate", "?"),
            )
            try:
                fold_id = int(fold.get("fold_id", 0))  # type: ignore[arg-type]
                tb_dir = self._tb_log_dir(fold_id, "phase_a", hp_idx=hp_idx, hp=hp)
                result = self._evaluate_hp_config(
                    hp, train_windows, val_windows, metadata,
                    returns, trailing_vol, stock_data, fold, device,
                    adapted_params=adapted, T_annee=T_annee,
                    log_dir=tb_dir,
                    raw_returns_train=train_raw,
                    window_metadata_train=train_metadata,
                    strata_train=train_strata,
                )
                config_results.append(result)
            except Exception as e:
                logger.warning("HP config failed: %s — %s", hp, e)

        if not config_results:
            return None

        # Select best config (use adapted K)
        K_adapted = adapted["K"]
        best = select_best_config(
            config_results,
            K=K_adapted,
            mdd_threshold=self.config.walk_forward.score_mdd_threshold,
            lambda_pen=self.config.walk_forward.score_lambda_pen,
            lambda_est=self.config.walk_forward.score_lambda_est,
            n_stocks=n_stocks,
        )

        if best is not None:
            best["n_surviving"] = len([r for r in config_results
                                       if r.get("AU", 0) >= int(0.15 * K_adapted)])

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
        adapted_params: dict[str, Any] | None = None,
        T_annee: int = 10,
        log_dir: str | None = None,
        raw_returns_train: torch.Tensor | None = None,
        window_metadata_train: pd.DataFrame | None = None,
        strata_train: np.ndarray | None = None,
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
        :param adapted_params (dict | None): Pre-computed adapted K/r_max/c_min/dropout
        :param T_annee (int): Training history in years (computed by caller)
        :param log_dir (str | None): TensorBoard log directory for this run
        :param raw_returns_train (torch.Tensor | None): Raw returns for training windows
        :param window_metadata_train (pd.DataFrame | None): Metadata for training windows
        :param strata_train (np.ndarray | None): Strata for training windows

        :return result (dict): Evaluation results
        """
        mode = hp.get("mode", "P")
        lr = hp.get("learning_rate", self.config.training.learning_rate)
        alpha = hp.get("alpha", 1.0)

        # Build VAE with adapted params
        n_stocks = len([c for c in returns.columns if str(c).isdigit()])
        T = self.config.data.window_length
        F = self.config.data.n_features

        if adapted_params is None:
            adapted_params = self._adapt_vae_params(n_stocks, T_annee)

        K = adapted_params["K"]

        model, info = build_vae(
            n=n_stocks, T=T, T_annee=T_annee, F=F, K=K,
            r_max=adapted_params["r_max"],
            c_min=adapted_params["c_min"],
            dropout=adapted_params["dropout"],
            learn_obs_var=(mode != "F"),
            sigma_sq_min=self.config.vae.sigma_sq_min,
            sigma_sq_max=self.config.vae.sigma_sq_max,
        )

        # Train
        trainer = VAETrainer(
            model, loss_mode=mode, learning_rate=lr,
            gamma=self.config.loss.gamma,
            lambda_co_max=self.config.loss.lambda_co_max,
            beta_fixed=self.config.loss.beta_fixed,
            warmup_fraction=self.config.loss.warmup_fraction,
            patience=self.config.training.patience,
            es_min_delta=self.config.training.es_min_delta,
            lr_patience=self.config.training.lr_patience,
            lr_factor=self.config.training.lr_factor,
            device=device,
            log_dir=log_dir,
            max_pairs=self.config.loss.max_pairs,
            delta_sync=self.config.loss.delta_sync,
            compile_model=self.config.training.compile_model,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            sigma_sq_min=self.config.vae.sigma_sq_min,
            sigma_sq_max=self.config.vae.sigma_sq_max,
        )

        fit_result = trainer.fit(
            train_windows, val_windows,
            max_epochs=self.config.training.max_epochs,
            batch_size=self.config.training.batch_size,
            raw_returns=raw_returns_train,
            window_metadata=window_metadata_train,
            strata=strata_train,
        )
        trainer.close()

        # Measure AU (fused into a single encode pass on val_windows)
        _, kl_per_dim = infer_latent_trajectories(
            model, val_windows,
            window_metadata=metadata.iloc[:len(val_windows)],
            batch_size=self.config.inference.batch_size,
            device=device,
            compute_kl=True,
        )
        assert kl_per_dim is not None
        active_mask = kl_per_dim > self.config.inference.au_threshold
        AU = int(active_mask.sum())

        # Release GPU memory after Phase A eval
        clear_device_cache(device)

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
    # Block-level variance targeting
    # NOTE: Currently calibrates on in-sample training data (last holdout_fraction
    # of the training period).  Ideally, VT should use a separate held-out period
    # not used for factor regression or covariance estimation (Barra USE4 approach).
    # However, this requires re-running factor regression on a subset, which is
    # architecturally complex.  The current approach provides a reasonable
    # approximation because (1) EWMA half-life=252 gives less weight to recent
    # observations in Sigma_z, partially decorrelating prediction from the
    # holdout period, and (2) VT scales are clamped to [0.01, 100].
    # -------------------------------------------------------------------

    @staticmethod
    def _block_variance_targeting(
        B_prime_port: np.ndarray,
        eigenvalues: np.ndarray,
        D_eps_port: np.ndarray,
        train_returns: "pd.DataFrame",
        stock_ids: list[int],
        n_port: int,
        z_hat: np.ndarray | None = None,
        residuals_by_stock: dict[int, list[float]] | None = None,
        B_A_by_date: dict[str, np.ndarray] | None = None,
        valid_dates: list[str] | None = None,
        holdout_start: int | None = None,
        holdout_fraction: float = 0.2,
    ) -> tuple[float, float]:
        """
        Block-level variance targeting via z_hat/residual decomposition.

        Uses the available z_hat (factor returns) and stock-level residuals
        to independently estimate realized systematic and idiosyncratic
        variances of the EW portfolio, then scales each block to match.

        When holdout_start is provided, the holdout is disjoint from the
        estimation period used for Sigma_z/D_eps.  This avoids look-ahead
        bias in the VT calibration (Barra USE4 Section 4).

        For each holdout day t:
          r_sys_t = w_eq^T @ B_A_t @ z_hat_t  (systematic EW return)
          r_idio_t = r_ew_t - r_sys_t          (idiosyncratic EW return)

        Then: vt_sys = Var(r_sys) / pred_sys_var
              vt_idio = Var(r_idio) / pred_idio_var

        Falls back to scalar VT when z_hat decomposition is not available.

        :param B_prime_port (np.ndarray): Rotated exposures (n, n_signal)
        :param eigenvalues (np.ndarray): Principal eigenvalues (n_signal,)
        :param D_eps_port (np.ndarray): Idiosyncratic variances (n,)
        :param train_returns (pd.DataFrame): Training daily log-returns
        :param stock_ids (list[int]): Stock IDs matching Sigma_assets rows
        :param n_port (int): Number of portfolio stocks
        :param z_hat (np.ndarray | None): Factor returns (n_dates, AU)
        :param residuals_by_stock (dict | None): stock_id -> list of residuals
        :param B_A_by_date (dict | None): date_str -> B_A matrix
        :param valid_dates (list[str] | None): Dates with valid factor returns
        :param holdout_start (int | None): Index into valid_dates where VT
            holdout begins.  Dates [holdout_start:] are used for VT.
            When None, falls back to fraction-based holdout.
        :param holdout_fraction (float): Fraction of training for holdout
            (only used when holdout_start is None)

        :return vt_sys (float): Systematic block scale, clamped to [0.01, 100]
        :return vt_idio (float): Idiosyncratic block scale, clamped to [0.01, 100]
        """
        n = B_prime_port.shape[0]
        w_eq = np.ones(n) / n

        # Predicted variance decomposition for EW portfolio
        beta_eq = B_prime_port.T @ w_eq  # (n_signal,)
        pred_sys_var = float(np.sum(beta_eq ** 2 * eigenvalues))
        pred_idio_var = float(np.sum(w_eq ** 2 * D_eps_port))

        if pred_sys_var <= 1e-12 and pred_idio_var <= 1e-12:
            return 1.0, 1.0

        # Realized EW portfolio returns on holdout.
        # Filter days by coverage (>= 90%) instead of fillna(0) which biases
        # variance downward for delisted/suspended stocks.
        available = [s for s in stock_ids[:n_port] if s in train_returns.columns]
        if len(available) < 10:
            return 1.0, 1.0
        R_df = train_returns[available]
        coverage = R_df.notna().mean(axis=1)
        R_df = R_df.loc[coverage >= 0.90]  # type: ignore[arg-type]
        R_arr = R_df.to_numpy()
        # Fill residual NaN with cross-sectional mean (market-neutral)
        row_means = np.nanmean(R_arr, axis=1, keepdims=True)
        nan_mask = np.isnan(R_arr)
        if np.any(nan_mask):
            R_arr = np.where(nan_mask, np.broadcast_to(row_means, R_arr.shape), R_arr)
        R_full = R_arr
        n_total = R_full.shape[0]
        if n_total < 40:
            return 1.0, 1.0

        # Determine holdout dates: use explicit holdout_start if provided,
        # otherwise fallback to fraction-based split
        if holdout_start is not None and valid_dates is not None:
            n_holdout = len(valid_dates) - holdout_start
        else:
            n_holdout = max(20, int(n_total * holdout_fraction))
        R = R_full[-n_holdout:]
        n_avail = len(available)
        w_sub = np.ones(n_avail) / n_avail
        r_ew = R @ w_sub

        # Try z_hat decomposition for independent block VT
        can_decompose = (
            z_hat is not None
            and B_A_by_date is not None
            and valid_dates is not None
            and n_holdout >= 20
        )

        if can_decompose:
            assert z_hat is not None and B_A_by_date is not None and valid_dates is not None
            # Use holdout dates (disjoint from estimation when holdout_start is set)
            if holdout_start is not None:
                holdout_dates = valid_dates[holdout_start:]
            else:
                holdout_dates = valid_dates[-n_holdout:]
            r_sys_list: list[float] = []
            r_idio_list: list[float] = []

            # Pre-build date → z_hat index lookup (O(1) per date instead of O(n))
            date_to_zidx = {d: i for i, d in enumerate(valid_dates)}

            # Date-keyed EW returns for robust alignment
            r_ew_by_date: dict[str, float] = {}
            for i_d, d_val in enumerate(train_returns.index):
                if isinstance(d_val, pd.Timestamp):
                    r_ew_by_date[str(d_val.date())] = float(R_full[i_d] @ w_sub)
                else:
                    r_ew_by_date[str(d_val)] = float(R_full[i_d] @ w_sub)

            for date_str in holdout_dates:
                if date_str not in B_A_by_date:
                    continue
                z_idx = date_to_zidx.get(date_str)
                if z_idx is None or z_idx >= z_hat.shape[0]:
                    continue

                B_A_t = B_A_by_date[date_str]
                z_t = z_hat[z_idx]  # (AU,)

                # Systematic return: use properly normalized EW weights
                n_b = B_A_t.shape[0]
                if n_b >= n_avail:
                    r_sys_t = float(w_sub @ (B_A_t[:n_avail] @ z_t))
                else:
                    w_eq_t = np.ones(n_b) / n_b  # normalized for this date's universe
                    r_sys_t = float(w_eq_t @ (B_A_t @ z_t))
                r_total_t = r_ew_by_date.get(date_str, 0.0)
                r_idio_t = r_total_t - r_sys_t

                r_sys_list.append(r_sys_t)
                r_idio_list.append(r_idio_t)

            if len(r_sys_list) >= 20:
                realized_sys_var = float(np.var(r_sys_list, ddof=1))
                realized_idio_var = float(np.var(r_idio_list, ddof=1))

                # Independent block scales with Bayesian shrinkage toward 1.0
                vt_sys_raw = realized_sys_var / max(pred_sys_var, 1e-15)
                vt_idio_raw = realized_idio_var / max(pred_idio_var, 1e-15)

                n_obs = len(r_sys_list)
                vt_sys = float(np.clip(
                    (n_obs * vt_sys_raw + _VT_PRIOR_STRENGTH) / (n_obs + _VT_PRIOR_STRENGTH),
                    _VT_SCALE_MIN, _VT_SCALE_MAX,
                ))
                vt_idio = float(np.clip(
                    (n_obs * vt_idio_raw + _VT_PRIOR_STRENGTH) / (n_obs + _VT_PRIOR_STRENGTH),
                    _VT_SCALE_MIN, _VT_SCALE_MAX,
                ))

                if not np.isfinite(vt_sys):
                    vt_sys = 1.0
                if not np.isfinite(vt_idio):
                    vt_idio = 1.0

                logger.info(
                    "  Variance targeting (z_hat decomposition, %d holdout days): "
                    "realized_sys_vol=%.4f, realized_idio_vol=%.4f, "
                    "pred_sys_vol=%.4f, pred_idio_vol=%.4f, "
                    "s_sys=%.4f, s_idio=%.4f",
                    len(r_sys_list),
                    np.sqrt(realized_sys_var * 252),
                    np.sqrt(realized_idio_var * 252),
                    np.sqrt(pred_sys_var * 252),
                    np.sqrt(pred_idio_var * 252),
                    vt_sys, vt_idio,
                )
                return vt_sys, vt_idio

        # Fallback: scalar VT (single ratio for both blocks)
        realized_var = float(np.var(r_ew, ddof=1))
        pred_total = pred_sys_var + pred_idio_var
        if pred_total <= 1e-12:
            return 1.0, 1.0

        overall_ratio = realized_var / pred_total
        vt_scalar = float(np.clip(
            (n_holdout * overall_ratio + _VT_PRIOR_STRENGTH) / (n_holdout + _VT_PRIOR_STRENGTH),
            _VT_SCALE_MIN, _VT_SCALE_MAX,
        ))

        logger.info(
            "  Variance targeting (scalar fallback, %d holdout days): "
            "realized_vol=%.4f, pred_vol=%.4f, s=%.4f",
            n_holdout,
            np.sqrt(realized_var * 252),
            np.sqrt(pred_total * 252),
            vt_scalar,
        )
        return vt_scalar, vt_scalar

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
        use_early_stopping: bool = False,
        _state_bag: dict[str, Any] | None = None,
        pretrained_model: Any = None,
        w_old: np.ndarray | None = None,
        is_first: bool = True,
        w_old_ids: list[int] | None = None,
    ) -> tuple[dict[str, float], np.ndarray, list[int]]:
        """
        Run Phase B for a single fold: train → infer → risk model → portfolio → metrics.

        When pretrained_model is provided, training is skipped and the given model
        is used directly for inference onward.

        :param fold (dict): Fold specification
        :param returns (pd.DataFrame): Log-returns
        :param trailing_vol (pd.DataFrame): Trailing vol
        :param rolling_vol (pd.DataFrame): 21-day rolling vol
        :param stock_data (pd.DataFrame): Raw stock data
        :param vix_data (pd.Series | None): VIX data
        :param config (dict): HP config from Phase A
        :param e_star (int): Number of training epochs
        :param device (torch.device): Compute device
        :param use_early_stopping (bool): Enable early stopping (when Phase A is skipped)
        :param pretrained_model (Any): Pre-loaded VAEModel to skip training. None = train.
        :param w_old (np.ndarray | None): Previous fold weights for turnover
            constraints (INV-012 symmetry with benchmarks).
        :param is_first (bool): First rebalancing flag (no turnover penalty).
        :param w_old_ids (list[int] | None): Stock IDs for w_old.  Used to
            align w_old to the current fold's universe when stocks enter/exit.

        :return metrics (dict): Fold metrics
        :return w (np.ndarray): Portfolio weights (n_port,)
        :return inferred_stock_ids (list[int]): Stock IDs corresponding to w
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
        t0 = time.monotonic()
        train_returns = returns.loc[train_start:train_end]
        train_rolling_vol = rolling_vol.loc[train_start:train_end]

        windows, metadata, raw_ret = create_windows(
            train_returns, train_rolling_vol, stock_ids,
            T=self.config.data.window_length,
            stride=self.config.data.training_stride,
        )

        if windows.shape[0] < 10:
            return self._empty_metrics(fold_id), np.ones(len(stock_ids)) / len(stock_ids), stock_ids

        logger.info(
            "  [Fold %d] Windowing: %d train windows (stride=%d) from %d stocks, %d days",
            fold_id, windows.shape[0], self.config.data.training_stride,
            len(stock_ids), len(train_returns),
        )

        # 2. Build and train VAE (or use pretrained model)
        n_stocks = len(stock_ids)
        T = self.config.data.window_length
        F = self.config.data.n_features
        train_days = len(train_returns)
        T_annee = max(1, train_days // 252)

        trainer: VAETrainer | None = None

        if pretrained_model is not None:
            # --- Skip training: use pre-loaded model ---
            model = pretrained_model
            model.to(device)
            model.eval()
            logger.info(
                "  [Fold %d] Using pretrained encoder, skipping training.",
                fold_id,
            )
            if _state_bag is not None:
                _state_bag["fit_result"] = None
                _state_bag["model_state_dict"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            # --- Train from scratch ---
            adapted = self._adapt_vae_params(n_stocks, T_annee)
            K = adapted["K"]

            model, info = build_vae(
                n=n_stocks, T=T, T_annee=T_annee, F=F, K=K,
                r_max=adapted["r_max"],
                c_min=adapted["c_min"],
                dropout=adapted["dropout"],
                learn_obs_var=(mode != "F"),
                sigma_sq_min=self.config.vae.sigma_sq_min,
                sigma_sq_max=self.config.vae.sigma_sq_max,
            )

            # Phase B: train on full data for E* epochs
            patience = (
                self.config.training.patience if use_early_stopping
                else e_star + 1  # Disable early stopping when E* comes from Phase A
            )
            tb_dir = self._tb_log_dir(fold_id, "phase_b")
            trainer = VAETrainer(
                model, loss_mode=mode, learning_rate=lr,
                gamma=self.config.loss.gamma,
                lambda_co_max=self.config.loss.lambda_co_max,
                beta_fixed=self.config.loss.beta_fixed,
                warmup_fraction=self.config.loss.warmup_fraction,
                patience=patience,
                es_min_delta=self.config.training.es_min_delta,
                lr_patience=self.config.training.lr_patience,
                lr_factor=self.config.training.lr_factor,
                device=device,
                log_dir=tb_dir,
                max_pairs=self.config.loss.max_pairs,
                delta_sync=self.config.loss.delta_sync,
                compile_model=self.config.training.compile_model,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                gradient_checkpointing=self.config.training.gradient_checkpointing,
                sigma_sq_min=self.config.vae.sigma_sq_min,
                sigma_sq_max=self.config.vae.sigma_sq_max,
            )

            # Co-movement data: strata from k-means on trailing returns
            window_strata = self._compute_window_strata(
                train_returns, stock_ids, metadata,
            )

            # Use 90% for train, 10% for validation monitor (but don't stop early)
            n_train = int(0.9 * windows.shape[0])
            train_w = windows[:n_train]
            val_w = windows[n_train:]
            if val_w.shape[0] == 0:
                val_w = train_w[-10:]

            # Split co-movement data to training portion
            train_raw = raw_ret[:n_train]
            train_metadata = metadata.iloc[:n_train].reset_index(drop=True)
            train_strata = window_strata[:n_train]

            fit_result = trainer.fit(
                train_w, val_w,
                max_epochs=e_star,
                batch_size=self.config.training.batch_size,
                raw_returns=train_raw,
                window_metadata=train_metadata,
                strata=train_strata,
            )
            t_train = time.monotonic() - t0
            actual_epochs = len(fit_result.get("history", []))
            # Update e_star to actual best epoch (fixes E*=max_epochs bug
            # when called from run_direct with e_star=max_epochs)
            if use_early_stopping:
                e_star = fit_result["best_epoch"]

            logger.info(
                "  [Fold %d] Training: %d/%d epochs in %.1fs (best_epoch=%d%s)",
                fold_id, actual_epochs, e_star, t_train,
                fit_result["best_epoch"],
                ", early stopped" if fit_result.get("best_epoch", 0) < actual_epochs - 1 else "",
            )

            if _state_bag is not None:
                _state_bag["fit_result"] = fit_result
                _state_bag["model_state_dict"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                _state_bag["build_params"] = {
                    "n": n_stocks, "T": T, "T_annee": T_annee, "F": F, "K": K,
                    "r_max": adapted["r_max"], "c_min": adapted["c_min"],
                    "dropout": adapted["dropout"],
                    "learn_obs_var": (mode != "F"),
                }
                _state_bag["vae_info"] = info

            # Save training checkpoint immediately (before portfolio optimization)
            if _state_bag is not None:
                ckpt_dir = self.checkpoint_dir or "checkpoints"
                tag = f"fold{fold_id:02d}_train"
                empty_w = np.ones(n_stocks) / n_stocks
                self.save_checkpoint(
                    _state_bag, fold, {}, empty_w,
                    checkpoint_dir=ckpt_dir, tag=tag,
                )

        # 3. Infer latent trajectories → B (+ KL per dim in same pass)
        #    Inference uses stride=1 for full-resolution exposure matrix (DVT Section 5)
        infer_windows, infer_metadata, _ = create_windows(
            train_returns, train_rolling_vol, stock_ids,
            T=self.config.data.window_length,
            stride=1,
        )
        logger.info(
            "  [Fold %d] Inference: %d windows (stride=1). AU measurement...",
            fold_id, infer_windows.shape[0],
        )
        trajectories, kl_per_dim = infer_latent_trajectories(
            model, infer_windows, infer_metadata,
            batch_size=self.config.inference.batch_size,
            device=device,
            compute_kl=True,
        )
        B, inferred_stock_ids = aggregate_profiles(
            trajectories,
            method="mean",
            half_life=self.config.inference.aggregation_half_life,
        )

        # 4. Derive AU from pre-computed KL (no extra forward pass)
        assert kl_per_dim is not None
        active_mask = kl_per_dim > self.config.inference.au_threshold
        active_dims_unsorted = np.where(active_mask)[0]
        sorted_order = np.argsort(-kl_per_dim[active_dims_unsorted])
        active_dims = active_dims_unsorted[sorted_order].tolist()
        AU = len(active_dims)
        au_max = compute_au_max_stat(
            n_obs=train_days,
            r_min=self.config.inference.r_min,
            ewma_half_life=self.config.risk_model.sigma_z_ewma_half_life,
        )
        # Cross-sectional identification: AU must be < n_stocks for OLS regression
        if au_max >= n_stocks:
            logger.info(
                "  [Fold %d] AU cap reduced: %d (statistical) -> %d (n_stocks-1)",
                fold_id, au_max, n_stocks - 1,
            )
            au_max = n_stocks - 1
        AU, active_dims = truncate_active_dims(AU, kl_per_dim, active_dims, au_max)

        if AU == 0:
            return self._empty_metrics(fold_id), np.ones(n_stocks) / n_stocks, stock_ids

        B_A = filter_exposure_matrix(B, active_dims)
        logger.info("  [Fold %d] AU=%d active units (max=%d)", fold_id, AU, au_max)

        # B_A per-factor z-score normalization (Barra USE4 standard):
        # each column of B_A (factor) is standardized to mean=0, std=1
        # cross-sectionally.  This replaces the ad-hoc global scaling
        # (mean(|B_A|) = 1/sqrt(AU)) with the industry-standard approach.
        if self.config.risk_model.b_a_normalize and AU > 0:
            col_means = np.mean(B_A, axis=0)  # (AU,)
            col_stds = np.std(B_A, axis=0, ddof=1)  # (AU,)
            col_stds = np.maximum(col_stds, 1e-12)
            B_A = (B_A - col_means[np.newaxis, :]) / col_stds[np.newaxis, :]
            logger.info(
                "  [Fold %d] B_A z-score normalized (per-factor): "
                "mean_abs=%.4f, col_std_range=[%.4f, %.4f]",
                fold_id, float(np.mean(np.abs(B_A))),
                float(np.min(col_stds)), float(np.max(col_stds)),
            )

        # B_A shrinkage: reduce spurious cross-stock correlations
        shrinkage_alpha = self.config.risk_model.b_a_shrinkage_alpha
        if shrinkage_alpha > 0.0:
            B_A = B_A * (1.0 - shrinkage_alpha)
            logger.info(
                "  [Fold %d] B_A shrinkage applied: alpha=%.3f, "
                "mean_abs=%.4f",
                fold_id, shrinkage_alpha, float(np.mean(np.abs(B_A))),
            )

        if _state_bag is not None:
            _state_bag["B"] = B
            _state_bag["B_A"] = B_A
            _state_bag["inferred_stock_ids"] = inferred_stock_ids
            _state_bag["AU"] = AU
            _state_bag["active_dims"] = active_dims
            _state_bag["kl_per_dim"] = kl_per_dim

            # Reconstruction MSE for diagnostics (train + OOS)
            _train_recon = float("nan")
            _fit_res = _state_bag.get("fit_result")
            if _fit_res is not None and _fit_res.get("history"):
                _train_recon = float(
                    _fit_res["history"][-1].get("train_recon", float("nan"))
                )
            _oos_mse = float("nan")
            try:
                _oos_ret = returns.loc[oos_start:oos_end]
                _oos_rvol = rolling_vol.loc[oos_start:oos_end]
                if len(_oos_ret) >= T:
                    _oos_w, _, _ = create_windows(
                        _oos_ret, _oos_rvol, stock_ids,
                        T=self.config.data.window_length, stride=T,
                    )
                    if _oos_w.shape[0] > 0:
                        model.eval()
                        with torch.no_grad():
                            _oos_t = _oos_w.to(
                                dtype=torch.float32, device=device,
                            )
                            _x_hat, _, _ = model(_oos_t)
                            _oos_mse = float(
                                torch.nn.functional.mse_loss(_x_hat, _oos_t).item()
                            )
            except Exception:
                pass
            _state_bag["reconstruction"] = {
                "train_mse": _train_recon,
                "oos_mse": _oos_mse,
            }

        # 5. Dual rescaling
        logger.info("  [Fold %d] Dual rescaling (%d stocks, %d dates)...",
                     fold_id, n_stocks, len(train_returns.index))
        # Build universe snapshots for estimation rescaling (point-in-time)
        train_dates = [str(d.date()) for d in train_returns.index]
        train_ret_sub = returns.loc[train_start:train_end]
        universe_snapshots: dict[str, list[int]] = {}
        for d in train_dates:
            if d in train_ret_sub.index.strftime("%Y-%m-%d").values:
                row = train_ret_sub.loc[d]
                valid_ids = [
                    sid for sid in inferred_stock_ids
                    if sid in row.index and pd.notna(row[sid])
                ]
                universe_snapshots[d] = valid_ids if valid_ids else inferred_stock_ids
            else:
                universe_snapshots[d] = inferred_stock_ids

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

        # Market intercept (Barra USE4): append a column of ones to B_A
        # AFTER dual rescaling so the intercept is not affected by vol ratios.
        # This adds an explicit market factor with unit exposure for all
        # stocks, fixing the variance targeting pathology where z-scored
        # factors give beta_eq ~ 0 for the EW portfolio.
        if self.config.risk_model.market_intercept:
            for date_str in B_A_by_date:
                n_active = B_A_by_date[date_str].shape[0]
                intercept = np.ones((n_active, 1), dtype=np.float64)
                B_A_by_date[date_str] = np.hstack(
                    [B_A_by_date[date_str], intercept]
                )
            intercept_port = np.ones(
                (B_A_port.shape[0], 1), dtype=np.float64
            )
            B_A_port = np.hstack([B_A_port, intercept_port])
            logger.info(
                "  [Fold %d] Market intercept added: n_factors=%d (AU=%d + 1)",
                fold_id, B_A_port.shape[1], AU,
            )

        logger.info("  [Fold %d] Dual rescaling done", fold_id)

        # 6. Factor regression → z_hat
        logger.info("  [Fold %d] Factor regression: B_A(%d dates)...", fold_id, len(train_dates))
        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns.loc[train_start:train_end],
            universe_snapshots,
            conditioning_threshold=self.config.risk_model.conditioning_threshold,
            ridge_scale=self.config.risk_model.ridge_scale,
            use_wls=self.config.risk_model.use_wls,
        )

        n_factors = z_hat.shape[1] if z_hat.shape[0] > 0 else AU
        if z_hat.shape[0] < n_factors:
            return self._empty_metrics(fold_id), np.ones(n_stocks) / n_stocks, stock_ids

        # Split z_hat into estimation (80%) / holdout (20%) for unbiased VT.
        # Sigma_z and D_eps are estimated from estimation period only, so the
        # VT holdout compares realized vs predicted without look-ahead bias.
        # (Barra USE4 Section 4, Shephard & Sheppard 2010)
        n_z = z_hat.shape[0]
        n_vt_holdout = max(20, int(n_z * 0.2))
        n_est = n_z - n_vt_holdout  # estimation period end index
        z_hat_est = z_hat[:n_est]
        valid_dates_est = valid_dates[:n_est]

        # Residuals from estimation period only (for D_eps)
        residuals = compute_residuals(
            B_A_by_date, z_hat_est,
            returns.loc[train_start:train_end],
            universe_snapshots, valid_dates_est,
            inferred_stock_ids,
        )

        # 7. Covariance: Σ_z, D_ε, Σ_assets (from estimation period)
        logger.info(
            "  [Fold %d] Factor regression done (%d valid dates, %d estimation / %d VT holdout). "
            "Covariance estimation...",
            fold_id, n_z, n_est, n_vt_holdout,
        )
        Sigma_z, n_signal = estimate_sigma_z(
            z_hat_est,
            eigenvalue_pct=self.config.risk_model.sigma_z_eigenvalue_pct,
            shrinkage_method=self.config.risk_model.sigma_z_shrinkage,
            ewma_half_life=self.config.risk_model.sigma_z_ewma_half_life,
        )
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

        # Split eigenvalues: full set for Sigma_assets (risk), signal-only
        # for entropy.  DGJ (2018) Theorem 2.2: noise eigenvalues should be
        # sigma_sq (not 0) for optimal Frobenius-loss estimation.  Keeping
        # them in Sigma_assets prevents systematic risk underestimation.
        # Entropy only diversifies across n_signal real factors.
        #
        # When market_intercept=True, PC1 captures the market factor (all
        # stocks have unit exposure → lambda_1/lambda_2 >> 1).  The market
        # is not a diversifiable bet (Meucci 2009), so we exclude it from
        # the entropy objective.  Sigma_assets keeps all eigenvalues.
        n_signal = max(1, min(n_signal, len(eigenvalues)))
        skip_pc1 = self.config.risk_model.market_intercept
        signal_start = 1 if skip_pc1 else 0
        signal_end = n_signal + (1 if skip_pc1 else 0)
        signal_end = min(signal_end, len(eigenvalues))
        eigenvalues_signal = eigenvalues[signal_start:signal_end].copy()
        B_prime_signal = B_prime_port[:, signal_start:signal_end].copy()
        n_signal_eff = len(eigenvalues_signal)  # actual count for ENB target
        if n_signal < len(eigenvalues) or skip_pc1:
            logger.info(
                "  [Fold %d] Eigenvalue split: n_signal=%d/%d "
                "(signal for entropy, full for risk%s)",
                fold_id, n_signal_eff, len(eigenvalues),
                ", PC1 market excluded" if skip_pc1 else "",
            )

        # Per-factor variance targeting (Barra USE4 methodology):
        # Calibrate each eigenvalue independently using holdout z_hat
        # projected into the principal factor basis.  This eliminates
        # the dependency on the EW portfolio that caused instability
        # when B_A is z-scored (beta_eq ≈ 0 for latent factors).
        # Reference: Menchero, Orr & Wang (2011), §4.
        V = risk_model["V"]
        z_hat_principal = z_hat @ V  # (n_dates, AU) in principal basis
        z_holdout_principal = z_hat_principal[n_est:]
        n_holdout_vt = z_holdout_principal.shape[0]

        if n_holdout_vt >= 20:
            # Per-factor VT with Bayesian shrinkage toward 1.0
            # (Barra USE4, Menchero et al. 2011 §4).
            # vt_shrunk = (n·vt_raw + ν·1.0) / (n + ν) pulls extreme
            # ratios toward unity, with strength inversely proportional
            # to holdout sample size.
            vt_factors = np.ones(len(eigenvalues), dtype=np.float64)
            for k in range(len(eigenvalues)):
                realized_k = float(np.var(z_holdout_principal[:, k], ddof=1))
                pred_k = max(float(eigenvalues[k]), 1e-15)
                vt_raw_k = realized_k / pred_k
                vt_shrunk_k = (
                    n_holdout_vt * vt_raw_k + _VT_PRIOR_STRENGTH * 1.0
                ) / (n_holdout_vt + _VT_PRIOR_STRENGTH)
                vt_factors[k] = float(np.clip(
                    vt_shrunk_k, _VT_SCALE_MIN, _VT_SCALE_MAX,
                ))
            eigenvalues = eigenvalues * vt_factors
            eigenvalues_signal = eigenvalues[signal_start:signal_end].copy()

            # Idiosyncratic VT: compute holdout residuals and compare
            # per-stock realized idio variance with D_eps.
            residuals_holdout = compute_residuals(
                B_A_by_date, z_hat[n_est:],
                returns.loc[train_start:train_end],
                universe_snapshots, valid_dates[n_est:],
                inferred_stock_ids,
            )
            idio_ratios: list[float] = []
            for i, sid in enumerate(inferred_stock_ids[:n_port]):
                resids_h = residuals_holdout.get(sid, [])
                if len(resids_h) >= 10 and D_eps_port[i] > 1e-12:
                    realized_idio = float(np.var(resids_h, ddof=1))
                    idio_ratios.append(realized_idio / float(D_eps_port[i]))
            if len(idio_ratios) >= 10:
                vt_idio_raw = float(np.median(idio_ratios))
                n_idio_obs = len(idio_ratios)
                vt_idio_shrunk = (
                    n_idio_obs * vt_idio_raw + _VT_PRIOR_STRENGTH * 1.0
                ) / (n_idio_obs + _VT_PRIOR_STRENGTH)
                vt_idio = float(np.clip(
                    vt_idio_shrunk, _VT_SCALE_MIN, _VT_SCALE_MAX,
                ))
            else:
                vt_idio = 1.0
            D_eps_port = D_eps_port * vt_idio

            logger.info(
                "  Variance targeting (per-factor, %d holdout days): "
                "vt_factors=[%.3f..%.3f], median=%.3f, vt_idio=%.3f",
                n_holdout_vt,
                float(np.min(vt_factors)), float(np.max(vt_factors)),
                float(np.median(vt_factors)), vt_idio,
            )
            vt_sys = float(np.median(vt_factors))  # summary for state_bag
        else:
            # Fallback: scalar VT when holdout is too short
            vt_sys, vt_idio = self._block_variance_targeting(
                B_prime_port, eigenvalues, D_eps_port,
                returns.loc[train_start:train_end],
                inferred_stock_ids, n_port,
                z_hat=z_hat,
                residuals_by_stock=residuals,
                B_A_by_date=B_A_by_date,
                valid_dates=valid_dates,
                holdout_start=n_est,
            )
            eigenvalues = eigenvalues * vt_sys
            eigenvalues_signal = eigenvalues_signal * vt_sys
            D_eps_port = D_eps_port * vt_idio

        Sigma_sys_vt = B_prime_port @ np.diag(eigenvalues) @ B_prime_port.T
        Sigma_assets = Sigma_sys_vt + np.diag(D_eps_port)
        Sigma_assets = 0.5 * (Sigma_assets + Sigma_assets.T)  # ensure symmetry

        # Update risk_model dict with variance-targeted values so that
        # downstream consumers (diagnostics, state_bag) see the scaled
        # covariance matrix, not the raw assembly output.
        risk_model["Sigma_assets"] = Sigma_assets
        risk_model["eigenvalues"] = eigenvalues
        risk_model["D_eps_port"] = D_eps_port

        if _state_bag is not None:
            _state_bag["risk_model"] = risk_model
            _state_bag["Sigma_z"] = Sigma_z
            _state_bag["z_hat"] = z_hat
            _state_bag["B_A_port"] = B_A_port
            _state_bag["vt_scale_sys"] = vt_sys
            _state_bag["vt_scale_idio"] = vt_idio
            _state_bag["n_signal"] = n_signal_eff
            _state_bag["B_A_by_date"] = B_A_by_date
            _state_bag["valid_dates"] = valid_dates
            _state_bag["universe_snapshots"] = universe_snapshots
            _state_bag["train_returns"] = returns.loc[train_start:train_end]

        logger.info(
            "  [Fold %d] Risk model: AU=%d, n_signal=%d, B_A(%d×%d), Sigma(%d×%d), "
            "vt_sys=%.4f, vt_idio=%.4f, shrinkage=%s",
            fold_id, AU, n_signal, B_A_port.shape[0], B_A_port.shape[1],
            Sigma_assets.shape[0], Sigma_assets.shape[1],
            vt_sys, vt_idio, self.config.risk_model.sigma_z_shrinkage,
        )

        # 8. Portfolio optimization: frontier → α*, SCA → w*
        pc = self.config.portfolio

        # Cross-sectional momentum signal (optional)
        mu: np.ndarray | None = None
        if pc.momentum_enabled:
            from src.portfolio.momentum import compute_momentum_signal
            mu_raw = compute_momentum_signal(
                returns.loc[:train_end],
                inferred_stock_ids,
                lookback=pc.momentum_lookback,
                skip=pc.momentum_skip,
            )
            mu = mu_raw * pc.momentum_weight
            logger.info(
                "  [Fold %d] Momentum signal: weight=%.2f, "
                "range=[%.2f, %.2f], n_nonzero=%d",
                fold_id, pc.momentum_weight,
                float(np.min(mu)), float(np.max(mu)),
                int(np.sum(np.abs(mu) > 1e-10)),
            )

        idio_weight = pc.entropy_idio_weight

        # Align w_old to the current fold's universe when stock IDs differ.
        # This prevents misaligned turnover penalties in the SCA optimizer
        # when stocks enter or exit the inferred universe between folds.
        if w_old is not None and w_old_ids is not None:
            w_old = _align_weights_by_stock_id(
                w_old, w_old_ids, inferred_stock_ids[:n_port],
            )
            if w_old.shape[0] != n_port:
                logger.warning(
                    "  [Fold %d] w_old alignment size mismatch: %d vs n_port=%d",
                    fold_id, w_old.shape[0], n_port,
                )
                w_old = None

        logger.info(
            "  [Fold %d] Frontier: %d alphas × %d starts...",
            fold_id, len(pc.alpha_grid), pc.n_starts,
        )
        t_port = time.monotonic()
        frontier, frontier_weights = compute_variance_entropy_frontier(
            Sigma_assets, B_prime_signal, eigenvalues_signal, D_eps_port,
            alpha_grid=pc.alpha_grid,
            lambda_risk=pc.lambda_risk,
            w_max=pc.w_max, w_min=pc.w_min,
            w_bar=pc.w_bar, phi=pc.phi,
            w_old=w_old, is_first=is_first,
            n_starts=pc.n_starts, seed=self.config.seed,
            entropy_eps=pc.entropy_eps,
            mu=mu,
            idio_weight=idio_weight,
        )
        # Dynamic target_enb: when config is 0.0 (default), use
        # n_signal_eff / 2 for a robust diversification target (Meucci 2009).
        # n_signal_eff excludes PC1 (market) when market_intercept=True.
        # Set config to -1.0 to force Kneedle legacy mode.
        effective_target_enb = pc.target_enb
        if pc.target_enb == 0.0:
            effective_target_enb = max(2.0, n_signal_eff / 2.0)
            logger.info(
                "  [Fold %d] Dynamic target_enb=%.1f (n_signal_eff=%d / 2)",
                fold_id, effective_target_enb, n_signal_eff,
            )
        elif pc.target_enb < 0.0:
            # Legacy Kneedle mode: pass 0.0 to select_operating_alpha
            effective_target_enb = 0.0
        alpha_opt = select_operating_alpha(frontier, target_enb=effective_target_enb)

        # Guard: alpha=0 means entropy had no effect (min-var under
        # concentration constraints). Force minimum alpha to ensure the
        # entropy term contributes to diversification.
        if alpha_opt <= 0.0:
            alpha_opt = min(a for a in frontier["alpha"] if a > 0.0)
            logger.warning(
                "  [Fold %d] alpha_opt was 0 — forced to %.4g to ensure "
                "entropy contributes to diversification.",
                fold_id, alpha_opt,
            )
            # Re-fetch weights for the new alpha
            if alpha_opt in frontier_weights:
                pass  # will be used below
            else:
                # Nearest alpha in frontier
                frontier_alphas = sorted(frontier_weights.keys())
                alpha_opt = min(
                    (a for a in frontier_alphas if a > 0.0),
                    default=frontier_alphas[-1],
                )

        if _state_bag is not None:
            _state_bag["alpha_opt"] = alpha_opt
            _state_bag["frontier"] = frontier
            if mu is not None:
                _state_bag["mu"] = mu

        # Reuse weights from frontier computation (avoids redundant SCA solve)
        w_opt = frontier_weights[alpha_opt]
        H_opt = compute_entropy_only(
            w_opt, B_prime_signal, eigenvalues_signal, pc.entropy_eps,
            D_eps=D_eps_port, idio_weight=idio_weight,
        )
        # Factor-only entropy for H_norm_signal (stays in [0, 1])
        H_factor = compute_entropy_only(
            w_opt, B_prime_signal, eigenvalues_signal, pc.entropy_eps,
            D_eps=None, idio_weight=0.0,
        )

        logger.info(
            "  [Fold %d] Frontier done (%.1fs), alpha*=%.3f, H=%.4f.",
            fold_id, time.monotonic() - t_port, alpha_opt, H_opt,
        )

        logger.info("  [Fold %d] Cardinality enforcement...", fold_id)

        # Cardinality enforcement
        logger.info("  [Fold %d] Cardinality enforcement (max %d eliminations)...",
                     fold_id, pc.max_cardinality_elim)
        L_sigma = _safe_cholesky(Sigma_assets)
        sca_kwargs = {
            "Sigma_assets": Sigma_assets,
            "B_prime": B_prime_signal,
            "eigenvalues": eigenvalues_signal,
            "alpha": alpha_opt,
            "lambda_risk": pc.lambda_risk,
            "phi": pc.phi, "w_bar": pc.w_bar,
            "w_max": pc.w_max,
            "w_old": w_old, "is_first": is_first,
            "kappa_1": pc.kappa_1, "kappa_2": pc.kappa_2,
            "delta_bar": pc.delta_bar, "tau_max": pc.tau_max,
            "entropy_eps": pc.entropy_eps,
            "_L_sigma": L_sigma,
            "mu": mu,
            "D_eps": D_eps_port,
            "idio_weight": idio_weight,
        }
        w_final = enforce_cardinality(
            w_opt, B_prime_signal, eigenvalues_signal,
            w_min=pc.w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            max_eliminations=pc.max_cardinality_elim,
            entropy_eps=pc.entropy_eps,
            method=pc.cardinality_method,
            D_eps=D_eps_port,
            idio_weight=idio_weight,
        )

        t_fold = time.monotonic() - t0
        logger.info(
            "  [Fold %d] Portfolio: alpha=%.2f, n_active=%d | total %.1fs",
            fold_id, alpha_opt, int(np.sum(w_final > 1e-6)), t_fold,
        )

        # 9. OOS metrics
        returns_oos = returns.loc[oos_start:oos_end]

        metrics = portfolio_metrics(
            w_final, returns_oos,
            universe=inferred_stock_ids,
            H_oos=H_opt, AU=AU, K=self.config.vae.K,
            Sigma_hat=Sigma_assets,
            n_signal=n_signal_eff,
            H_factor=H_factor,
        )
        metrics["AU"] = float(AU)
        metrics["e_star"] = float(e_star)
        metrics["alpha_opt"] = alpha_opt

        # Log post-training metrics to TensorBoard
        if trainer is not None:
            if trainer._tb_writer is not None:
                trainer._tb_writer.add_scalar(
                    "Training/sharpe", metrics.get("sharpe", 0.0), e_star,
                )
            trainer.close()

        # Layer 2: realized vs predicted variance + correlation
        # Compute portfolio return directly via R @ w with fillna(0)
        # instead of dropna("any") which eliminates most OOS rows when
        # any single stock has a missing value on a given day.
        if returns_oos.shape[0] > 0:
            active_mask = w_final > 1e-8
            active_ids = [inferred_stock_ids[i] for i in range(n_port) if active_mask[i]]
            if len(active_ids) > 0:
                w_active = w_final[active_mask]
                w_active = w_active / w_active.sum()
                active_indices = np.where(active_mask)[0]
                Sigma_active = Sigma_assets[np.ix_(active_indices, active_indices)]

                # Portfolio return: fillna(0) means a missing stock
                # contributes 0 return (as if not traded that day).
                oos_active = returns_oos.reindex(columns=active_ids).fillna(0.0)
                port_ret = oos_active.values @ w_active
                if len(port_ret) > 10:
                    realized_var = float(np.var(port_ret, ddof=1))
                    predicted_var = float(w_active @ Sigma_active @ w_active)
                    if predicted_var > 1e-15:
                        var_ratio = realized_var / predicted_var
                        metrics["var_ratio_oos"] = var_ratio

                # Rank correlation still uses dropna for cross-sectional comparison
                oos_df = returns_oos.reindex(columns=active_ids).dropna(how="any")
                if len(oos_df) > 10:
                    corr_rank = realized_vs_predicted_correlation(Sigma_active, oos_df.values)
                    metrics["corr_rank_oos"] = corr_rank

        # Layer 3: crisis-period return
        if vix_data is not None and returns_oos.shape[0] > 0:
            from src.data_pipeline.crisis import compute_crisis_threshold
            train_end_ts: pd.Timestamp = pd.Timestamp(train_end)  # type: ignore[assignment]
            threshold = compute_crisis_threshold(
                vix_data, train_end_ts,
                percentile=self.config.data.vix_lookback_percentile,
            )
            oos_dates = returns_oos.index
            oos_vix = vix_data.reindex(oos_dates)
            crisis_mask = np.asarray(oos_vix > threshold, dtype=bool)
            crisis_mask = np.where(np.isnan(oos_vix), False, crisis_mask)
            crisis_ret = crisis_period_return(
                w_final, returns_oos, inferred_stock_ids, crisis_mask,
            )
            metrics["crisis_return_oos"] = crisis_ret

        # Factor explanatory power — use estimation-rescaled B_A_by_date
        # (matches the rescaling used to estimate z_hat via OLS)
        ep = factor_explanatory_power_dynamic(
            B_A_by_date, z_hat,
            returns.loc[train_start:train_end],
            universe_snapshots, valid_dates,
        )
        metrics["explanatory_power"] = ep

        # GPU memory logging + cleanup after fold
        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
            logger.info("  GPU peak memory this fold: %.0f MB", peak_mb)
            torch.cuda.reset_peak_memory_stats(device)
        clear_device_cache(device)

        return metrics, w_final, inferred_stock_ids

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
        *,
        store_weights: bool = False,
    ) -> dict[str, Any]:
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

        # Ensure returns have string columns for consistent benchmark indexing
        train_returns = returns.loc[:train_end].copy()
        train_returns.columns = pd.Index(stock_ids_str)

        # Clean NaN: drop stocks with >50% NaN, then drop remaining NaN rows
        # DVT 4.2: gaps >5 days remain NaN and must be excluded before estimation
        nan_frac = train_returns.isna().mean(axis=0)
        valid_stocks = list(pd.Series(nan_frac[nan_frac < 0.5]).index)
        if len(valid_stocks) < len(stock_ids_str):
            n_dropped = len(stock_ids_str) - len(valid_stocks)
            logger.debug("Benchmark %s: dropped %d stocks with >50%% NaN", bench_name, n_dropped)
        train_returns = train_returns[valid_stocks].dropna()
        stock_ids_str = valid_stocks

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

        # Prepare trailing vol for benchmarks (INV-012: same inputs as VAE)
        # Filter to same valid stocks as train_returns (after NaN cleanup)
        valid_int_ids = [int(s) for s in stock_ids_str]
        train_vol = trailing_vol.loc[:train_end, valid_int_ids].copy()
        train_vol.columns = pd.Index(stock_ids_str)

        try:
            benchmark.fit(
                train_returns, stock_ids_str,
                trailing_vol=train_vol,
                current_date=train_end,
            )
            w = benchmark.optimize(w_old=w_old, is_first=is_first)
        except Exception as e:
            logger.warning("Benchmark %s failed: %s", bench_name, e)
            n = len(stock_ids_str)
            w = np.ones(n) / n

        # OOS metrics — restrict to same valid stocks, fillna(0) per DVT convention
        returns_oos = returns.loc[oos_start:oos_end, [int(s) for s in stock_ids_str]].copy()
        returns_oos.columns = pd.Index(stock_ids_str)
        metrics = benchmark.evaluate(w, returns_oos, stock_ids_str)

        # Rename to distinguish from VAE metrics
        renamed: dict[str, Any] = {}
        for k, v in metrics.items():
            if k == "ann_vol":
                renamed["ann_vol_oos"] = v
            elif k == "max_drawdown":
                renamed["max_drawdown_oos"] = v
            else:
                renamed[k] = v

        # Optionally store weights / daily returns for cumulative-return plots
        if store_weights:
            renamed["_weights"] = w
            renamed["_universe"] = stock_ids_str
            custom_oos = benchmark.get_oos_returns(returns_oos)
            if custom_oos is not None:
                renamed["_daily_returns"] = custom_oos

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
