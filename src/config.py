"""
Centralized configuration for the VAE Latent Risk Factor pipeline.

All hyperparameters, constants, and default values are defined here
as frozen dataclasses. Modules import from this file — no hardcoding.

Reference: ISD Section 00 — Symbol Glossary + Critical Conventions.
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


def _validate_range(
    name: str,
    value: float | int,
    default: float | int,
    lo: float | int | None = None,
    hi: float | int | None = None,
    lo_exclusive: bool = False,
    hi_exclusive: bool = False,
) -> None:
    """
    Validate that value is within bounds. Raises ValueError with remediation.

    :param name (str): Parameter name for error message
    :param value (float | int): Current value to check
    :param default (float | int): Suggested default value
    :param lo (float | int | None): Lower bound (None = no lower bound)
    :param hi (float | int | None): Upper bound (None = no upper bound)
    :param lo_exclusive (bool): If True, use strict > for lower bound
    :param hi_exclusive (bool): If True, use strict < for upper bound
    """
    violated = False
    if lo is not None:
        if lo_exclusive and value <= lo:
            violated = True
        elif not lo_exclusive and value < lo:
            violated = True
    if hi is not None:
        if hi_exclusive and value >= hi:
            violated = True
        elif not hi_exclusive and value > hi:
            violated = True

    if violated:
        lo_bracket = "(" if lo_exclusive else "["
        hi_bracket = ")" if hi_exclusive else "]"
        lo_str = str(lo) if lo is not None else "-inf"
        hi_str = str(hi) if hi is not None else "inf"
        raise ValueError(
            f"Invalid parameter '{name}':\n"
            f"  Current value : {value}\n"
            f"  Valid range    : {lo_bracket}{lo_str}, {hi_str}{hi_bracket}\n"
            f"  Suggested      : {default}"
        )


def _validate_in(
    name: str,
    value: str,
    allowed: set[str],
    default: str,
) -> None:
    """
    Validate that value is in the allowed set.

    :param name (str): Parameter name
    :param value (str): Current value
    :param allowed (set[str]): Set of valid values
    :param default (str): Suggested default
    """
    if value not in allowed:
        raise ValueError(
            f"Invalid parameter '{name}':\n"
            f"  Current value : {value!r}\n"
            f"  Allowed        : {sorted(allowed)}\n"
            f"  Suggested      : {default!r}"
        )


def _validate_pair(
    name_lo: str,
    value_lo: float,
    name_hi: str,
    value_hi: float,
    strict: bool = True,
) -> None:
    """
    Validate that value_lo < value_hi (or <=).

    :param name_lo (str): Name of the lower-bound parameter
    :param value_lo (float): Value that should be smaller
    :param name_hi (str): Name of the upper-bound parameter
    :param value_hi (float): Value that should be larger
    :param strict (bool): If True, require strict inequality
    """
    op = "<" if strict else "<="
    violated = (value_lo >= value_hi) if strict else (value_lo > value_hi)
    if violated:
        raise ValueError(
            f"Invalid parameter pair:\n"
            f"  {name_lo} = {value_lo} must be {op} {name_hi} = {value_hi}\n"
            f"  Adjust either {name_lo} (lower) or {name_hi} (higher)."
        )


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
    :param training_stride (int): Stride for training windows (DVT Table 8.1: 1-21).
        Inference always uses stride=1. Default 21 (monthly) reduces redundancy ~20x.
    """

    n_stocks: int = 1000
    window_length: int = 504
    n_features: int = 2
    vol_window: int = 252
    vix_lookback_percentile: float = 80.0
    min_valid_fraction: float = 0.80
    data_source: str = "synthetic"
    data_dir: str = "data/"
    training_stride: int = 21

    def __post_init__(self) -> None:
        _validate_range("n_stocks", self.n_stocks, default=1000, lo=1)
        _validate_range("window_length", self.window_length, default=504, lo=1)
        _validate_range("n_features", self.n_features, default=2, lo=1)
        _validate_range("vol_window", self.vol_window, default=252, lo=2)
        _validate_range("vix_lookback_percentile", self.vix_lookback_percentile,
                        default=80.0, lo=0, hi=100, lo_exclusive=True)
        _validate_range("min_valid_fraction", self.min_valid_fraction,
                        default=0.80, lo=0, hi=1, lo_exclusive=True)
        _validate_in("data_source", self.data_source,
                     {"synthetic", "tiingo", "csv"}, default="synthetic")
        _validate_range("training_stride", self.training_stride, default=21, lo=1, hi=63)

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
    :param r_max (float): Maximum parameter/data ratio for capacity guard
    """

    K: int = 50
    sigma_sq_init: float = 1.0
    sigma_sq_min: float = 1e-4
    sigma_sq_max: float = 10.0
    window_length: int = 504
    n_features: int = 2
    r_max: float = 5.0
    dropout: float = 0.2

    def __post_init__(self) -> None:
        _validate_range("K", self.K, default=50, lo=1)
        _validate_range("window_length", self.window_length, default=504, lo=63)
        _validate_range("n_features", self.n_features, default=2, lo=1)
        _validate_range("sigma_sq_min", self.sigma_sq_min, default=1e-4,
                        lo=0, lo_exclusive=True)
        _validate_range("sigma_sq_max", self.sigma_sq_max, default=10.0,
                        lo=0, lo_exclusive=True)
        _validate_range("r_max", self.r_max, default=5.0, lo=0, lo_exclusive=True)
        _validate_range("dropout", self.dropout, default=0.2, lo=0.0, hi=0.5)
        _validate_pair("sigma_sq_min", self.sigma_sq_min,
                       "sigma_sq_max", self.sigma_sq_max, strict=True)

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

    def __post_init__(self) -> None:
        _validate_in("mode", self.mode, {"P", "F", "A"}, default="P")
        _validate_range("gamma", self.gamma, default=3.0, lo=1.0)
        _validate_range("lambda_co_max", self.lambda_co_max, default=0.5,
                        lo=0, hi=1.0)
        _validate_range("beta_fixed", self.beta_fixed, default=1.0,
                        lo=0, lo_exclusive=True)
        _validate_range("warmup_fraction", self.warmup_fraction, default=0.20,
                        lo=0, hi=1.0, hi_exclusive=True)
        _validate_range("max_pairs", self.max_pairs, default=2048, lo=1)
        _validate_range("delta_sync", self.delta_sync, default=21, lo=1)


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
    :param es_min_delta (float): Minimum ELBO improvement to count as progress.
        val_loss must drop by at least this amount below best to reset the
        patience counter. 0.0 = any improvement counts (default).
    :param lr_patience (int): ReduceLROnPlateau patience
    :param lr_factor (float): ReduceLROnPlateau factor
    :param n_strata (int): Number of strata for synchronous batching
    :param curriculum_phase1_frac (float): Fraction of epochs for Phase 1
    :param curriculum_phase2_frac (float): Fraction of epochs for Phase 2
    :param gradient_accumulation_steps (int): Accumulate gradients over N steps
        before optimizer update. Simulates batch_size * N effective batch on
        memory-constrained GPUs. Default 1 (no accumulation).
    :param gradient_checkpointing (bool): Trade compute for VRAM by recomputing
        activations during backward pass. Saves ~20% VRAM at ~5% speed cost.
    :param compile_model (bool): Use torch.compile for operator fusion on
        CUDA/MPS (15-30% speedup). Initial compilation takes ~30-60s.
    """

    max_epochs: int = 250
    batch_size: int = 512
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    patience: int = 20
    es_min_delta: float = 0.0
    lr_patience: int = 10
    lr_factor: float = 0.75
    n_strata: int = 15
    curriculum_phase1_frac: float = 0.30
    curriculum_phase2_frac: float = 0.30
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    compile_model: bool = True

    def __post_init__(self) -> None:
        _validate_range("max_epochs", self.max_epochs, default=250, lo=1)
        _validate_range("batch_size", self.batch_size, default=512, lo=1)
        _validate_range("learning_rate", self.learning_rate, default=5e-3,
                        lo=0, lo_exclusive=True)
        _validate_range("patience", self.patience, default=20, lo=1)
        _validate_range("es_min_delta", self.es_min_delta, default=0.0, lo=0)
        _validate_range("lr_patience", self.lr_patience, default=10, lo=1)
        _validate_range("lr_factor", self.lr_factor, default=0.75,
                        lo=0, hi=1, lo_exclusive=True, hi_exclusive=True)
        _validate_range("n_strata", self.n_strata, default=15, lo=1)
        _validate_range("curriculum_phase1_frac", self.curriculum_phase1_frac,
                        default=0.30, lo=0)
        _validate_range("curriculum_phase2_frac", self.curriculum_phase2_frac,
                        default=0.30, lo=0)
        _validate_range("gradient_accumulation_steps",
                        self.gradient_accumulation_steps, default=1, lo=1)
        total_curriculum = self.curriculum_phase1_frac + self.curriculum_phase2_frac
        if total_curriculum > 1.0:
            raise ValueError(
                "Invalid parameter combination:\n"
                f"  curriculum_phase1_frac ({self.curriculum_phase1_frac}) + "
                f"curriculum_phase2_frac ({self.curriculum_phase2_frac}) = "
                f"{total_curriculum:.4f} > 1.0\n"
                f"  Their sum must be <= 1.0. "
                f"Suggested: phase1=0.30, phase2=0.30 (sum=0.60)."
            )


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
    :param aggregation_half_life (int): Exponential decay half-life in
        windows for profile aggregation.  0 = uniform mean (original
        behavior).  Typical: 60 (≈ 1260 trading days at stride=21,
        i.e. ~5 years).  Larger values give more weight to recent
        windows, making B_A reflect the current factor structure.
    """

    batch_size: int = 512
    au_threshold: float = 0.01
    r_min: int = 2
    aggregation_method: str = "mean"
    aggregation_half_life: int = 0

    def __post_init__(self) -> None:
        _validate_range("batch_size", self.batch_size, default=512, lo=1)
        _validate_range("au_threshold", self.au_threshold, default=0.01,
                        lo=0, lo_exclusive=True)
        _validate_range("r_min", self.r_min, default=2, lo=1)
        _validate_in("aggregation_method", self.aggregation_method,
                     {"mean"}, default="mean")
        _validate_range("aggregation_half_life", self.aggregation_half_life,
                        default=0, lo=0)


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
    :param sigma_z_shrinkage (str): Eigenvalue shrinkage method for Σ_z.
        "truncation" — legacy: keep top eigenvalues explaining eigenvalue_pct
            of variance, zero out the rest. Simple but arbitrary threshold.
        "spiked" — Donoho-Gavish-Johnstone (2018) optimal shrinker for
            spiked covariance models.  Analytically corrects finite-sample
            eigenvalue bias using Baik-Ben Arous-Péché transition.
            Best for factor models with a few dominant eigenvalues.
        "analytical_nonlinear" — Ledoit-Wolf (2020) analytical nonlinear
            shrinkage.  General-purpose, no structural assumption.
            Requires the covShrinkage package (fallback: "spiked").
        Default: "spiked".
    :param sigma_z_ewma_half_life (int): Half-life in days for EWMA weighting
        of z_hat before Σ_z estimation.  0 = equal weights (legacy).
        252 = Barra USE4 standard.  Gives more weight to recent observations,
        better capturing time-varying factor covariance.  Range: [0, ∞).
    :param sigma_z_eigenvalue_pct (float): Fraction of total Σ_z variance
        retained after eigenvalue truncation.  Only used when
        sigma_z_shrinkage="truncation".  1.0 = keep all eigenvalues.
        0.95 = discard eigenvalues contributing < 5% of variance.
        Range: (0, 1].
    :param b_a_shrinkage_alpha (float): Shrinkage intensity for the
        exposure matrix B_A towards zero.  After filtering active units,
        B_A is multiplied by (1 - alpha).  0.0 = no shrinkage (default).
        NOTE: This is pure scaling, not true column-wise shrinkage.
        With variance targeting, it changes the sys/idio ratio without
        benefit.  Set to 0.0 unless implementing proper James-Stein
        column-wise shrinkage.  Range: [0, 1].
    :param b_a_normalize (bool): If True, apply per-factor cross-sectional
        z-score normalization to B_A after AU filtering (Barra USE4 standard).
        Each column of B_A is standardized to mean=0, std=1.  This corrects
        the unconstrained VAE encoder output scale and ensures factors are
        comparable.  Default: True.
    """

    winsorize_lo: float = 5.0
    winsorize_hi: float = 95.0
    d_eps_floor: float = 1e-6
    conditioning_threshold: float = 1e6
    ridge_scale: float = 1e-6
    sigma_z_shrinkage: str = "spiked"
    sigma_z_eigenvalue_pct: float = 0.95
    sigma_z_ewma_half_life: int = 252
    b_a_shrinkage_alpha: float = 0.0
    b_a_normalize: bool = True

    def __post_init__(self) -> None:
        _validate_range("winsorize_lo", self.winsorize_lo, default=5.0,
                        lo=0, hi=100)
        _validate_range("winsorize_hi", self.winsorize_hi, default=95.0,
                        lo=0, hi=100)
        _validate_range("d_eps_floor", self.d_eps_floor, default=1e-6,
                        lo=0, lo_exclusive=True)
        _validate_range("conditioning_threshold", self.conditioning_threshold,
                        default=1e6, lo=0, lo_exclusive=True)
        _validate_range("ridge_scale", self.ridge_scale, default=1e-6,
                        lo=0, lo_exclusive=True)
        _validate_pair("winsorize_lo", self.winsorize_lo,
                       "winsorize_hi", self.winsorize_hi, strict=True)
        _validate_in("sigma_z_shrinkage", self.sigma_z_shrinkage,
                     {"truncation", "spiked", "analytical_nonlinear"},
                     default="spiked")
        _validate_range("sigma_z_eigenvalue_pct",
                        self.sigma_z_eigenvalue_pct,
                        default=1.0, lo=0, hi=1,
                        lo_exclusive=True)
        _validate_range("b_a_shrinkage_alpha",
                        self.b_a_shrinkage_alpha,
                        default=0.0, lo=0, hi=1)
        _validate_range("sigma_z_ewma_half_life",
                        self.sigma_z_ewma_half_life,
                        default=0, lo=0)


# ---------------------------------------------------------------------------
# Portfolio Optimization (MOD-008)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioConfig:
    """
    Configuration for portfolio optimization.

    Constraints are IDENTICAL for VAE and all benchmarks (INV-012).

    :param lambda_risk (float): Risk aversion coefficient applied to daily
        variance.  Since w^T Sigma w is in daily units (~1e-4), this must
        be large enough to make the risk term commensurate with entropy
        (O(0.01-0.1)).  252 = annualization (risk aversion γ=1 on annual
        variance); use 252-2520 for γ in [1, 10].  Literature: Garlappi
        et al. (2007), DeMiguel et al. (2009).
    :param w_max (float): Maximum weight per stock (hard cap)
    :param w_min (float): Minimum active weight (semi-continuous)
    :param w_bar (float): Concentration penalty threshold.  Positions with
        w_i > w_bar are penalized quadratically.  Must be > 1/target_N to
        avoid penalizing ALL active positions (self-defeating).  With 50-80
        target positions and w_max=0.05, w_bar=0.03 is appropriate.
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
    :param cardinality_method (str): Strategy for semi-continuous enforcement.
        "auto" (best available), "sequential" (original), "gradient" (Taylor approx),
        "miqp" (single MOSEK MIQP), "two_stage" (DVT §4.7 decomposition)
    :param alpha_grid (list): Grid of alpha values for variance-entropy
        frontier.  Needs >= 10 points with log-uniform spacing for
        reliable Kneedle elbow detection (Satopaa et al. 2011).
    :param momentum_enabled (bool): Enable cross-sectional momentum signal
    :param momentum_lookback (int): Lookback window in trading days (12 months)
    :param momentum_skip (int): Skip period in trading days (1 month reversal)
    :param momentum_weight (float): Scaling factor for momentum signal (γ_mom).
        Calibrated so E[|w^T μ|] ≈ lambda_risk × E[w^T Σ w].  With λ_risk=252
        and daily variance ~5e-4, momentum_weight=0.02 gives a return term
        ~0.02, comparable to the risk term ~0.04.
    :param entropy_idio_weight (float): Weight for idiosyncratic entropy layer
        in two-layer entropy formulation.  0.0 = factor-only entropy
        (recommended when eigenvalues are truncated to signal dimensions).
        1.0 = idiosyncratic-only.  Default 0.0 = factor-only entropy.
        The parameter remains available for experimentation.
    :param target_enb (float): Target effective number of bets (ENB = exp(H)).
        When > 0, select the smallest α on the frontier where ENB >= target_enb.
        When 0.0, use Kneedle elbow detection (legacy).
        Recommended: n_signal / 2 (e.g. 2.5 for 5 signal factors).
    """

    lambda_risk: float = 252.0
    w_max: float = 0.05
    w_min: float = 0.001
    w_bar: float = 0.03
    phi: float = 5.0
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
    cardinality_method: str = "auto"
    alpha_grid: list[float] = field(
        default_factory=lambda: [
            0, 0.001, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
        ]
    )
    momentum_enabled: bool = False
    momentum_lookback: int = 252
    momentum_skip: int = 21
    momentum_weight: float = 0.02
    entropy_idio_weight: float = 0.0
    target_enb: float = 0.0

    def __post_init__(self) -> None:
        _validate_range("w_min", self.w_min, default=0.001,
                        lo=0, lo_exclusive=True)
        _validate_range("w_max", self.w_max, default=0.05,
                        lo=0, hi=1, lo_exclusive=True)
        _validate_range("tau_max", self.tau_max, default=0.30,
                        lo=0, hi=1, lo_exclusive=True)
        _validate_range("lambda_risk", self.lambda_risk, default=252.0,
                        lo=0, lo_exclusive=True)
        _validate_range("phi", self.phi, default=5.0, lo=0)
        _validate_range("kappa_1", self.kappa_1, default=0.1, lo=0)
        _validate_range("kappa_2", self.kappa_2, default=7.5, lo=0)
        _validate_range("n_starts", self.n_starts, default=5, lo=1)
        _validate_range("sca_max_iter", self.sca_max_iter, default=100, lo=1)
        _validate_range("sca_tol", self.sca_tol, default=1e-8,
                        lo=0, lo_exclusive=True)
        if not self.alpha_grid:
            raise ValueError(
                "Invalid parameter 'alpha_grid':\n"
                "  Current value : [] (empty)\n"
                "  Requirement   : must contain at least 1 element\n"
                "  Suggested     : [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]"
            )
        _validate_pair("w_min", self.w_min, "w_max", self.w_max, strict=True)
        _validate_pair("w_min", self.w_min, "w_bar", self.w_bar, strict=True)
        if self.w_bar < 2.0 * self.w_min:
            import warnings
            warnings.warn(
                f"w_bar={self.w_bar} is close to w_min={self.w_min} — "
                f"concentration penalty may penalize ALL active positions "
                f"(self-defeating).  Recommended: w_bar >= 1/target_N "
                f"(e.g. 0.03 for ~33 positions).",
                UserWarning,
                stacklevel=2,
            )
        _validate_in("cardinality_method", self.cardinality_method,
                     {"auto", "sequential", "gradient", "miqp", "two_stage"},
                     default="auto")
        _validate_range("momentum_lookback", self.momentum_lookback,
                        default=252, lo=1)
        _validate_range("momentum_skip", self.momentum_skip,
                        default=21, lo=0)
        _validate_range("momentum_weight", self.momentum_weight,
                        default=0.02, lo=0)
        _validate_range("target_enb", self.target_enb,
                        default=0.0, lo=0)
        _validate_range("entropy_idio_weight", self.entropy_idio_weight,
                        default=0.2, lo=0.0, hi=1.0)
        if self.momentum_enabled and self.momentum_lookback <= self.momentum_skip:
            raise ValueError(
                f"Invalid parameter pair:\n"
                f"  momentum_lookback ({self.momentum_lookback}) must be > "
                f"momentum_skip ({self.momentum_skip})"
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

    def __post_init__(self) -> None:
        _validate_range("total_years", self.total_years, default=30, lo=1)
        _validate_range("min_training_years", self.min_training_years,
                        default=10, lo=1)
        _validate_range("oos_months", self.oos_months, default=6, lo=1)
        _validate_range("embargo_days", self.embargo_days, default=21, lo=0)
        _validate_range("holdout_years", self.holdout_years, default=3, lo=0)
        _validate_range("val_years", self.val_years, default=2, lo=1)
        required_min = self.min_training_years + self.holdout_years
        if self.total_years <= required_min:
            raise ValueError(
                "Invalid parameter combination:\n"
                f"  total_years ({self.total_years}) must be > "
                f"min_training_years ({self.min_training_years}) + "
                f"holdout_years ({self.holdout_years}) = {required_min}\n"
                f"  Current total_years leaves no room for OOS folds.\n"
                f"  Suggested: total_years={required_min + 10} "
                f"(default=30)."
            )


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

    def __post_init__(self) -> None:
        if self.data.window_length != self.vae.window_length:
            raise ValueError(
                "Cross-config mismatch:\n"
                f"  data.window_length = {self.data.window_length}\n"
                f"  vae.window_length  = {self.vae.window_length}\n"
                "  These must match. Update one to equal the other."
            )
        if self.data.n_features != self.vae.n_features:
            raise ValueError(
                "Cross-config mismatch:\n"
                f"  data.n_features = {self.data.n_features}\n"
                f"  vae.n_features  = {self.vae.n_features}\n"
                "  These must match. Update one to equal the other."
            )
