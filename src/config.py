"""
Centralized configuration for the VAE Latent Risk Factor pipeline.

All hyperparameters, constants, and default values are defined here
as frozen dataclasses. Modules import from this file — no hardcoding.

Reference: ISD Section 00 — Symbol Glossary + Critical Conventions.
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data Pipeline (MOD-001)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataPipelineConfig:
    """
    Configuration for the data pipeline module.

    :param n_stocks (int): Number of stocks in the universe
    :param window_length (int): Window length T in trading days
    :param n_features (int): Number of features per timestep (return + realized vol)
    :param vol_window (int): Trailing window for annualized volatility (days)
    :param vix_lookback_percentile (float): VIX percentile for crisis threshold
    :param min_valid_fraction (float): Minimum fraction of valid data for a stock
    """

    n_stocks: int = 1000
    window_length: int = 504
    n_features: int = 2
    vol_window: int = 252
    vix_lookback_percentile: float = 80.0
    min_valid_fraction: float = 0.80

    @property
    def D(self) -> int:
        """Number of elements per window: T x F."""
        return self.window_length * self.n_features


# ---------------------------------------------------------------------------
# VAE Architecture (MOD-002)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAEArchitectureConfig:
    """
    Configuration for the VAE architecture.

    :param K (int): Latent capacity ceiling
    :param sigma_sq_init (float): Initial value for observation noise
    :param sigma_sq_min (float): Lower clamp for sigma squared
    :param sigma_sq_max (float): Upper clamp for sigma squared
    :param window_length (int): Window length T (must match DataPipelineConfig)
    :param n_features (int): Number of features F (must match DataPipelineConfig)
    """

    K: int = 200
    sigma_sq_init: float = 1.0
    sigma_sq_min: float = 1e-4
    sigma_sq_max: float = 10.0
    window_length: int = 504
    n_features: int = 2

    @property
    def D(self) -> int:
        """Number of elements per window: T x F."""
        return self.window_length * self.n_features

    @property
    def encoder_depth(self) -> int:
        """L = max(3, ceil(log2(T/63)) + 2)."""
        return max(3, math.ceil(math.log2(self.window_length / 63)) + 2)

    @property
    def final_layer_width(self) -> int:
        """C_L = max(384, ceil(1.3 * 2K))."""
        return max(384, math.ceil(1.3 * 2 * self.K))


# ---------------------------------------------------------------------------
# Loss Function (MOD-004)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossConfig:
    """
    Configuration for loss computation.

    :param mode (str): Loss mode — 'P' (primary), 'F' (fallback), 'A' (advanced)
    :param gamma (float): Crisis overweighting factor
    :param lambda_co_max (float): Maximum co-movement loss weight
    :param beta_fixed (float): Fixed beta for Mode A (ignored in P/F)
    :param warmup_fraction (float): Fraction of epochs for Mode F warmup
    :param max_pairs (int): Maximum number of pairs for co-movement loss
    :param delta_sync (int): Max date gap for synchronization (days)
    """

    mode: str = "P"
    gamma: float = 3.0
    lambda_co_max: float = 0.5
    beta_fixed: float = 1.0
    warmup_fraction: float = 0.20
    max_pairs: int = 2048
    delta_sync: int = 21


# ---------------------------------------------------------------------------
# Training (MOD-005)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the training loop.

    :param max_epochs (int): Maximum number of training epochs
    :param batch_size (int): Batch size
    :param learning_rate (float): Initial learning rate (eta_0)
    :param weight_decay (float): Weight decay for Adam
    :param adam_betas (tuple): Adam betas
    :param adam_eps (float): Adam epsilon
    :param patience (int): Early stopping patience (epochs)
    :param lr_patience (int): ReduceLROnPlateau patience
    :param lr_factor (float): ReduceLROnPlateau factor
    :param n_strata (int): Number of strata for synchronous batching
    :param curriculum_phase1_frac (float): Fraction of epochs for Phase 1
    :param curriculum_phase2_frac (float): Fraction of epochs for Phase 2
    """

    max_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    n_strata: int = 15
    curriculum_phase1_frac: float = 0.30
    curriculum_phase2_frac: float = 0.30


# ---------------------------------------------------------------------------
# Inference (MOD-006)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuration for inference and AU measurement.

    :param batch_size (int): Batch size for inference
    :param au_threshold (float): KL threshold for active unit detection (nats)
    :param r_min (int): Minimum observations-per-parameter ratio for AU_max_stat
    :param aggregation_method (str): Method for aggregating profiles ('mean')
    """

    batch_size: int = 512
    au_threshold: float = 0.01
    r_min: int = 2
    aggregation_method: str = "mean"


# ---------------------------------------------------------------------------
# Risk Model (MOD-007)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskModelConfig:
    """
    Configuration for the risk model.

    :param winsorize_lo (float): Lower percentile for ratio winsorization
    :param winsorize_hi (float): Upper percentile for ratio winsorization
    :param d_eps_floor (float): Floor for idiosyncratic variance
    :param conditioning_threshold (float): Condition number threshold for ridge
    :param ridge_scale (float): Scale factor for minimal ridge regularization
    """

    winsorize_lo: float = 5.0
    winsorize_hi: float = 95.0
    d_eps_floor: float = 1e-6
    conditioning_threshold: float = 1e6
    ridge_scale: float = 1e-6


# ---------------------------------------------------------------------------
# Portfolio Optimization (MOD-008)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioConfig:
    """
    Configuration for portfolio optimization.

    Constraints are IDENTICAL for VAE and all benchmarks (INV-012).

    :param lambda_risk (float): Risk aversion
    :param w_max (float): Maximum weight per stock (hard cap)
    :param w_min (float): Minimum active weight (semi-continuous)
    :param w_bar (float): Concentration penalty threshold
    :param phi (float): Concentration penalty weight
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover penalty threshold
    :param tau_max (float): Maximum one-way turnover (hard cap)
    :param n_starts (int): Number of multi-start initializations
    :param sca_max_iter (int): Maximum SCA iterations
    :param sca_tol (float): SCA convergence tolerance
    :param armijo_c (float): Armijo sufficient decrease constant
    :param armijo_rho (float): Armijo backtracking factor
    :param armijo_max_iter (int): Maximum Armijo backtracking steps
    :param max_cardinality_elim (int): Max cardinality elimination rounds
    :param entropy_eps (float): Numerical stability for log in entropy
    :param alpha_grid (list): Grid of alpha values for frontier
    """

    lambda_risk: float = 1.0
    w_max: float = 0.05
    w_min: float = 0.001
    w_bar: float = 0.03
    phi: float = 25.0
    kappa_1: float = 0.1
    kappa_2: float = 7.5
    delta_bar: float = 0.01
    tau_max: float = 0.30
    n_starts: int = 5
    sca_max_iter: int = 100
    sca_tol: float = 1e-8
    armijo_c: float = 1e-4
    armijo_rho: float = 0.5
    armijo_max_iter: int = 20
    max_cardinality_elim: int = 100
    entropy_eps: float = 1e-30
    alpha_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )


# ---------------------------------------------------------------------------
# Walk-Forward (MOD-009)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    :param total_years (int): Total history length
    :param min_training_years (int): Minimum training window
    :param oos_months (int): Out-of-sample period length
    :param embargo_days (int): Embargo between training and OOS (trading days)
    :param holdout_years (int): Final holdout period
    :param val_years (int): Nested validation window for Phase A
    :param score_lambda_pen (float): MDD penalty weight in composite score
    :param score_lambda_est (float): Estimation quality penalty weight
    :param score_mdd_threshold (float): MDD threshold in composite score
    """

    total_years: int = 30
    min_training_years: int = 10
    oos_months: int = 6
    embargo_days: int = 21
    holdout_years: int = 3
    val_years: int = 2
    score_lambda_pen: float = 5.0
    score_lambda_est: float = 2.0
    score_mdd_threshold: float = 0.20


# ---------------------------------------------------------------------------
# Full Pipeline Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Top-level configuration aggregating all module configs.

    :param data (DataPipelineConfig): Data pipeline configuration
    :param vae (VAEArchitectureConfig): VAE architecture configuration
    :param loss (LossConfig): Loss function configuration
    :param training (TrainingConfig): Training loop configuration
    :param inference (InferenceConfig): Inference configuration
    :param risk_model (RiskModelConfig): Risk model configuration
    :param portfolio (PortfolioConfig): Portfolio optimization configuration
    :param walk_forward (WalkForwardConfig): Walk-forward configuration
    :param seed (int): Global random seed for reproducibility
    """

    data: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    vae: VAEArchitectureConfig = field(default_factory=VAEArchitectureConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    risk_model: RiskModelConfig = field(default_factory=RiskModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    seed: int = 42
