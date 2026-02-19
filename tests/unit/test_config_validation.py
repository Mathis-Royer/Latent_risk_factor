"""
Tests for config dataclass __post_init__ validation.

Each dataclass is tested for:
  - Default construction succeeds
  - Representative invalid values raise ValueError with parameter name
  - Cross-field constraints raise ValueError
"""

import pytest

from src.config import (
    DataPipelineConfig,
    VAEArchitectureConfig,
    LossConfig,
    TrainingConfig,
    InferenceConfig,
    RiskModelConfig,
    PortfolioConfig,
    WalkForwardConfig,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# DataPipelineConfig
# ---------------------------------------------------------------------------


class TestDataPipelineConfigValidation:
    """Validation tests for DataPipelineConfig."""

    def test_default_valid(self) -> None:
        DataPipelineConfig()

    def test_n_stocks_zero_is_valid(self) -> None:
        """n_stocks=0 is valid and means 'no cap, use all available stocks'."""
        config = DataPipelineConfig(n_stocks=0)
        assert config.n_stocks == 0

    def test_n_stocks_negative(self) -> None:
        """Negative n_stocks should raise ValueError."""
        with pytest.raises(ValueError, match="n_stocks"):
            DataPipelineConfig(n_stocks=-1)

    def test_window_length_zero(self) -> None:
        with pytest.raises(ValueError, match="window_length"):
            DataPipelineConfig(window_length=0)

    def test_vol_window_one(self) -> None:
        with pytest.raises(ValueError, match="vol_window"):
            DataPipelineConfig(vol_window=1)

    def test_min_valid_fraction_zero(self) -> None:
        with pytest.raises(ValueError, match="min_valid_fraction"):
            DataPipelineConfig(min_valid_fraction=0.0)

    def test_vix_percentile_zero(self) -> None:
        with pytest.raises(ValueError, match="vix_lookback_percentile"):
            DataPipelineConfig(vix_lookback_percentile=0.0)

    def test_vix_percentile_over_100(self) -> None:
        with pytest.raises(ValueError, match="vix_lookback_percentile"):
            DataPipelineConfig(vix_lookback_percentile=101.0)


# ---------------------------------------------------------------------------
# VAEArchitectureConfig
# ---------------------------------------------------------------------------


class TestVAEArchitectureConfigValidation:
    """Validation tests for VAEArchitectureConfig."""

    def test_default_valid(self) -> None:
        VAEArchitectureConfig()

    def test_K_zero(self) -> None:
        with pytest.raises(ValueError, match="K"):
            VAEArchitectureConfig(K=0)

    def test_window_too_short(self) -> None:
        with pytest.raises(ValueError, match="window_length"):
            VAEArchitectureConfig(window_length=10)

    def test_window_63_valid(self) -> None:
        VAEArchitectureConfig(window_length=63)

    def test_sigma_sq_min_zero(self) -> None:
        with pytest.raises(ValueError, match="sigma_sq_min"):
            VAEArchitectureConfig(sigma_sq_min=0.0)

    def test_sigma_sq_min_gte_max(self) -> None:
        with pytest.raises(ValueError, match="sigma_sq_min"):
            VAEArchitectureConfig(sigma_sq_min=20.0, sigma_sq_max=10.0)

    def test_r_max_zero(self) -> None:
        with pytest.raises(ValueError, match="r_max"):
            VAEArchitectureConfig(r_max=0.0)


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------


class TestLossConfigValidation:
    """Validation tests for LossConfig."""

    def test_default_valid(self) -> None:
        LossConfig()

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            LossConfig(mode="X")

    def test_gamma_below_one(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            LossConfig(gamma=0.5)

    def test_lambda_co_max_negative(self) -> None:
        with pytest.raises(ValueError, match="lambda_co_max"):
            LossConfig(lambda_co_max=-0.1)

    def test_beta_fixed_zero(self) -> None:
        with pytest.raises(ValueError, match="beta_fixed"):
            LossConfig(beta_fixed=0.0)

    def test_warmup_fraction_one(self) -> None:
        with pytest.raises(ValueError, match="warmup_fraction"):
            LossConfig(warmup_fraction=1.0)

    def test_max_pairs_zero(self) -> None:
        with pytest.raises(ValueError, match="max_pairs"):
            LossConfig(max_pairs=0)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfigValidation:
    """Validation tests for TrainingConfig."""

    def test_default_valid(self) -> None:
        TrainingConfig()

    def test_max_epochs_zero(self) -> None:
        with pytest.raises(ValueError, match="max_epochs"):
            TrainingConfig(max_epochs=0)

    def test_batch_size_zero(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_learning_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_lr_factor_zero(self) -> None:
        with pytest.raises(ValueError, match="lr_factor"):
            TrainingConfig(lr_factor=0.0)

    def test_lr_factor_one(self) -> None:
        with pytest.raises(ValueError, match="lr_factor"):
            TrainingConfig(lr_factor=1.0)

    def test_curriculum_sum_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="curriculum_phase1_frac"):
            TrainingConfig(curriculum_phase1_frac=0.6, curriculum_phase2_frac=0.5)

    def test_curriculum_sum_one_valid(self) -> None:
        TrainingConfig(curriculum_phase1_frac=0.5, curriculum_phase2_frac=0.5)


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------


class TestInferenceConfigValidation:
    """Validation tests for InferenceConfig."""

    def test_default_valid(self) -> None:
        InferenceConfig()

    def test_batch_size_zero(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            InferenceConfig(batch_size=0)

    def test_au_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="au_threshold"):
            InferenceConfig(au_threshold=0.0)

    def test_invalid_aggregation(self) -> None:
        with pytest.raises(ValueError, match="aggregation_method"):
            InferenceConfig(aggregation_method="median")


# ---------------------------------------------------------------------------
# RiskModelConfig
# ---------------------------------------------------------------------------


class TestRiskModelConfigValidation:
    """Validation tests for RiskModelConfig."""

    def test_default_valid(self) -> None:
        RiskModelConfig()

    def test_d_eps_floor_zero(self) -> None:
        with pytest.raises(ValueError, match="d_eps_floor"):
            RiskModelConfig(d_eps_floor=0.0)

    def test_winsorize_lo_gte_hi(self) -> None:
        with pytest.raises(ValueError, match="winsorize_lo"):
            RiskModelConfig(winsorize_lo=95.0, winsorize_hi=5.0)

    def test_winsorize_lo_negative(self) -> None:
        with pytest.raises(ValueError, match="winsorize_lo"):
            RiskModelConfig(winsorize_lo=-1.0)

    def test_conditioning_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="conditioning_threshold"):
            RiskModelConfig(conditioning_threshold=0.0)


# ---------------------------------------------------------------------------
# PortfolioConfig
# ---------------------------------------------------------------------------


class TestPortfolioConfigValidation:
    """Validation tests for PortfolioConfig."""

    def test_default_valid(self) -> None:
        PortfolioConfig()

    def test_w_min_zero(self) -> None:
        with pytest.raises(ValueError, match="w_min"):
            PortfolioConfig(w_min=0.0)

    def test_w_max_over_one(self) -> None:
        with pytest.raises(ValueError, match="w_max"):
            PortfolioConfig(w_max=1.5)

    def test_w_min_gte_w_max(self) -> None:
        with pytest.raises(ValueError, match="w_min"):
            PortfolioConfig(w_min=0.10, w_max=0.05)

    def test_tau_max_zero(self) -> None:
        with pytest.raises(ValueError, match="tau_max"):
            PortfolioConfig(tau_max=0.0)

    def test_lambda_risk_zero(self) -> None:
        with pytest.raises(ValueError, match="lambda_risk"):
            PortfolioConfig(lambda_risk=0.0)

    def test_n_starts_zero(self) -> None:
        with pytest.raises(ValueError, match="n_starts"):
            PortfolioConfig(n_starts=0)

    def test_alpha_grid_empty(self) -> None:
        with pytest.raises(ValueError, match="alpha_grid"):
            PortfolioConfig(alpha_grid=[])

    def test_sca_tol_zero(self) -> None:
        with pytest.raises(ValueError, match="sca_tol"):
            PortfolioConfig(sca_tol=0.0)


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------


class TestWalkForwardConfigValidation:
    """Validation tests for WalkForwardConfig."""

    def test_default_valid(self) -> None:
        WalkForwardConfig()

    def test_total_years_zero(self) -> None:
        with pytest.raises(ValueError, match="total_years"):
            WalkForwardConfig(total_years=0)

    def test_min_training_years_zero(self) -> None:
        with pytest.raises(ValueError, match="min_training_years"):
            WalkForwardConfig(min_training_years=0)

    def test_oos_months_zero(self) -> None:
        with pytest.raises(ValueError, match="oos_months"):
            WalkForwardConfig(oos_months=0)

    def test_total_years_too_small(self) -> None:
        with pytest.raises(ValueError, match="total_years"):
            WalkForwardConfig(total_years=5, min_training_years=3, holdout_years=3)

    def test_total_years_equal_sum(self) -> None:
        with pytest.raises(ValueError, match="total_years"):
            WalkForwardConfig(total_years=13, min_training_years=10, holdout_years=3)

    def test_total_years_gt_sum_valid(self) -> None:
        WalkForwardConfig(total_years=14, min_training_years=10, holdout_years=3)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfigValidation:
    """Validation tests for PipelineConfig cross-config consistency."""

    def test_default_valid(self) -> None:
        PipelineConfig()

    def test_window_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="window_length"):
            PipelineConfig(
                data=DataPipelineConfig(window_length=504),
                vae=VAEArchitectureConfig(window_length=252),
            )

    def test_n_features_mismatch(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            PipelineConfig(
                data=DataPipelineConfig(n_features=2),
                vae=VAEArchitectureConfig(n_features=3),
            )

    def test_matching_configs_valid(self) -> None:
        PipelineConfig(
            data=DataPipelineConfig(window_length=252, n_features=3),
            vae=VAEArchitectureConfig(window_length=252, n_features=3),
        )
