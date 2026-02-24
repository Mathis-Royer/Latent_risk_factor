"""
Unit tests for diagnostic experiments module (src/integration/diagnostic_experiments.py).

Covers:
- Experiment data loading (validation of critical keys)
- PCA loadings computation (shape, orthogonality, variance capture)
- Risk model with overrides (default, diagonal Sigma_z, B_A override)
- Portfolio from risk model (equal-weight, min-variance)
- OOS metrics computation (completeness, range checks)
- Factor quality profiling (all metrics, random baseline comparison)
- Parameter sweep (returns valid DataFrame)
- Run all Tier 0 (returns comparison DataFrame)

Reference: docs/strategic_paths_post_v2.md, Section 5 (Experimental Protocol).
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, PortfolioConfig, RiskModelConfig
from src.integration.diagnostic_experiments import (
    build_risk_model_with_overrides,
    compute_factor_quality_profile,
    compute_oos_metrics,
    compute_pca_loadings,
    load_experiment_data,
    run_all_tier0,
    run_parameter_sweep,
    run_portfolio_from_risk_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SEED = 42
N_STOCKS = 30
AU = 8
T_TOTAL = 300
N_FACTORS_TRUE = 5


# ---------------------------------------------------------------------------
# Helper: generate synthetic experiment data
# ---------------------------------------------------------------------------


def _make_synthetic_experiment_data(
    n: int = N_STOCKS,
    au: int = AU,
    T: int = T_TOTAL,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Generate synthetic data for diagnostic experiment tests.

    Creates returns with realistic factor structure: r = B_true @ z + eps,
    where z has mild autocorrelation and eps is heteroscedastic.

    :param n (int): Number of stocks
    :param au (int): Number of active units (VAE latent dims)
    :param T (int): Total time periods (business days)
    :param seed (int): Random seed for reproducibility

    :return data (dict): Keys: B_A, returns, trailing_vol, stock_ids,
            train_start, train_end, returns_oos, config, dates
    """
    rng = np.random.RandomState(seed)

    stock_ids = list(range(1, n + 1))
    dates = pd.bdate_range(start="2010-01-04", periods=T, freq="B")

    # True factor structure for realistic cross-sectional dependence
    B_true = rng.randn(n, N_FACTORS_TRUE) * 0.3

    # Factor returns with mild AR(1) autocorrelation
    z = np.zeros((T, N_FACTORS_TRUE))
    for t in range(1, T):
        z[t] = 0.2 * z[t - 1] + rng.randn(N_FACTORS_TRUE) * 0.01

    # Heteroscedastic idiosyncratic noise
    eps_std = rng.uniform(0.005, 0.015, size=n)
    eps = rng.randn(T, n) * eps_std[np.newaxis, :]

    # Returns = systematic + idiosyncratic
    ret_values = z @ B_true.T + eps

    returns = pd.DataFrame(ret_values, index=dates, columns=stock_ids)

    # Trailing vol: annualized, realistic range [0.15, 0.45]
    vol_values = rng.uniform(0.15, 0.45, size=(T, n)).astype(np.float64)
    trailing_vol = pd.DataFrame(vol_values, index=dates, columns=stock_ids)

    # VAE exposure matrix (n, au) — includes true factor structure + noise
    B_A = np.zeros((n, au), dtype=np.float64)
    B_A[:, :N_FACTORS_TRUE] = B_true + rng.randn(n, N_FACTORS_TRUE) * 0.05
    B_A[:, N_FACTORS_TRUE:] = rng.randn(n, au - N_FACTORS_TRUE) * 0.1

    # Train/OOS split (80/20)
    split_idx = int(T * 0.8)
    train_start = dates[0].strftime("%Y-%m-%d")
    train_end = dates[split_idx - 1].strftime("%Y-%m-%d")
    returns_oos = returns.iloc[split_idx:]

    # Pipeline config with small defaults for fast tests
    config = PipelineConfig(
        risk_model=RiskModelConfig(
            sigma_z_shrinkage="truncation",
            sigma_z_eigenvalue_pct=0.95,
            market_intercept=False,
        ),
        portfolio=PortfolioConfig(
            n_starts=1,
            sca_max_iter=30,
            sca_tol=1e-5,
            alpha_grid=[0.01, 0.1, 1.0],
            frontier_early_stop_patience=1,
            frontier_refine_enabled=False,
        ),
        seed=seed,
    )

    return {
        "B_A": B_A,
        "returns": returns,
        "trailing_vol": trailing_vol,
        "stock_ids": stock_ids,
        "train_start": train_start,
        "train_end": train_end,
        "returns_oos": returns_oos,
        "config": config,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# TestLoadExperimentData
# ---------------------------------------------------------------------------


class TestLoadExperimentData:
    """Tests for load_experiment_data (Tier 0 data loading)."""

    def test_load_experiment_data_missing_keys(self) -> None:
        """Partial checkpoint missing critical keys raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)

            # Create minimal structure with only diagnostics (no B_A, weights)
            diag_path = run_path / "diagnostic_data.json"
            diag_path.write_text(json.dumps({"status": "partial"}))

            # Create stock_ids but NOT B_A or weights
            json_dir = run_path / "json"
            json_dir.mkdir(parents=True)
            (json_dir / "inferred_stock_ids.json").write_text(
                json.dumps([1, 2, 3])
            )

            with pytest.raises(ValueError, match="missing critical data"):
                load_experiment_data(tmpdir)

    def test_load_experiment_data_returns_dict(self) -> None:
        """Complete checkpoint returns dict with expected keys."""
        rng = np.random.RandomState(SEED)
        n, au = 10, 5

        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)

            # Create diagnostics
            diag_path = run_path / "diagnostic_data.json"
            diag_path.write_text(json.dumps({"status": "complete"}))

            # Create arrays
            arrays_inf = run_path / "arrays" / "inference_done"
            arrays_inf.mkdir(parents=True)
            B_A = rng.randn(n, au)
            np.save(arrays_inf / "B_A.npy", B_A)

            arrays_port = run_path / "arrays" / "portfolio_done"
            arrays_port.mkdir(parents=True)
            weights = rng.dirichlet(np.ones(n))
            np.save(arrays_port / "w_vae.npy", weights)

            # Create JSON
            json_dir = run_path / "json"
            json_dir.mkdir(parents=True)
            (json_dir / "inferred_stock_ids.json").write_text(
                json.dumps(list(range(n)))
            )
            (json_dir / "scalars.json").write_text(
                json.dumps({"AU": au})
            )

            data = load_experiment_data(tmpdir)

            assert "B_A" in data, "Must contain B_A"
            assert "weights" in data, "Must contain weights"
            assert "stock_ids" in data, "Must contain stock_ids"
            assert "diagnostics" in data, "Must contain diagnostics"

            # Verify loaded arrays match saved values
            np.testing.assert_array_equal(data["B_A"], B_A)
            np.testing.assert_array_equal(data["weights"], weights)
            assert data["stock_ids"] == list(range(n))
            assert data["AU"] == au


# ---------------------------------------------------------------------------
# TestComputePCALoadings
# ---------------------------------------------------------------------------


class TestComputePCALoadings:
    """Tests for compute_pca_loadings (Tier 0 PCA oracle)."""

    def test_pca_loadings_shape(self) -> None:
        """B_PCA has shape (n, k_star) with k_star > 0."""
        data = _make_synthetic_experiment_data()
        returns = data["returns"]
        stock_ids = data["stock_ids"]

        B_PCA, eigenvalues_pca, k_star = compute_pca_loadings(
            returns, stock_ids, k_max=20,
        )

        n = len(stock_ids)
        assert B_PCA.shape[0] == n, f"Expected n={n} rows, got {B_PCA.shape[0]}"
        assert B_PCA.shape[1] == k_star, "Columns must match k_star"
        assert k_star >= 1, "Must select at least 1 factor"
        assert k_star <= 20, f"k_star={k_star} exceeds k_max=20"
        assert eigenvalues_pca.shape == (k_star,), "Eigenvalues shape mismatch"
        assert np.all(eigenvalues_pca > 0), "All eigenvalues must be positive"

    def test_pca_loadings_orthogonality(self) -> None:
        """PCA loadings columns are orthonormal (V^T V = I)."""
        data = _make_synthetic_experiment_data()
        returns = data["returns"]
        stock_ids = data["stock_ids"]

        B_PCA, _, k_star = compute_pca_loadings(returns, stock_ids, k_max=20)

        # Right singular vectors are orthonormal
        gram = B_PCA.T @ B_PCA  # (k_star, k_star)
        np.testing.assert_allclose(
            gram, np.eye(k_star), atol=1e-10,
            err_msg="PCA loadings columns must be orthonormal (V from SVD)",
        )

    def test_pca_captures_variance(self) -> None:
        """PCA explains more variance than a random matrix baseline."""
        data = _make_synthetic_experiment_data()
        returns = data["returns"]
        stock_ids = data["stock_ids"]
        rng = np.random.RandomState(SEED)

        B_PCA, eigenvalues_pca, k_star = compute_pca_loadings(
            returns, stock_ids, k_max=20,
        )

        # PCA explained variance ratio
        R = returns[stock_ids].values
        R_centered = R - R.mean(axis=0, keepdims=True)
        total_var = np.sum(R_centered ** 2) / R.shape[0]
        pca_explained = float(np.sum(eigenvalues_pca))
        pca_ratio = pca_explained / total_var

        # Random baseline: same k_star random vectors explain less
        random_ratios = []
        for _ in range(20):
            B_rand = rng.randn(len(stock_ids), k_star)
            Q, _ = np.linalg.qr(B_rand)
            # Project and compute explained variance
            proj = R_centered @ Q  # (T, k_star)
            random_explained = float(np.sum(proj ** 2)) / R.shape[0]
            random_ratios.append(random_explained / total_var)

        random_mean = float(np.mean(random_ratios))

        assert pca_ratio > random_mean, (
            f"PCA variance ratio ({pca_ratio:.4f}) must exceed "
            f"random baseline ({random_mean:.4f})"
        )
        assert pca_ratio > 0.05, (
            f"PCA must explain at least 5% of variance, got {pca_ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# TestBuildRiskModelWithOverrides
# ---------------------------------------------------------------------------


class TestBuildRiskModelWithOverrides:
    """Tests for build_risk_model_with_overrides (Tier 0 risk model)."""

    @pytest.fixture()
    def experiment_data(self) -> dict[str, Any]:
        """Shared synthetic data for risk model tests."""
        return _make_synthetic_experiment_data()

    def test_build_risk_model_default(
        self, experiment_data: dict[str, Any],
    ) -> None:
        """Default build returns all expected keys with correct shapes."""
        d = experiment_data
        result = build_risk_model_with_overrides(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
        )

        # Check all expected keys
        expected_keys = {
            "Sigma_assets", "eigenvalues_full", "eigenvalues_signal",
            "B_prime_signal", "B_prime_port", "D_eps_port",
            "n_signal", "z_hat", "condition_number", "B_A_used",
            "n_port", "V",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

        n_port = result["n_port"]
        assert n_port > 0, "Must have at least one stock in portfolio"

        # Sigma_assets: symmetric PSD, correct shape
        Sigma = result["Sigma_assets"]
        assert Sigma.shape == (n_port, n_port), "Sigma shape mismatch"
        np.testing.assert_allclose(
            Sigma, Sigma.T, atol=1e-12,
            err_msg="Sigma_assets must be symmetric",
        )
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10), (
            f"Sigma_assets must be PSD, min eigenvalue = {eigvals.min():.2e}"
        )

        # Condition number finite and positive
        assert np.isfinite(result["condition_number"]), "Condition number must be finite"
        assert result["condition_number"] > 0, "Condition number must be positive"

        # Signal count
        assert result["n_signal"] >= 1, "Must have at least 1 signal factor"

        # B_A_used matches input (no override)
        np.testing.assert_array_equal(result["B_A_used"], d["B_A"])

    def test_build_risk_model_diagonal_sigma_z(
        self, experiment_data: dict[str, Any],
    ) -> None:
        """force_diagonal=True produces lower off-diagonal eigenvalue concentration."""
        d = experiment_data

        # Full Sigma_z
        result_full = build_risk_model_with_overrides(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
        )

        # Diagonal Sigma_z
        result_diag = build_risk_model_with_overrides(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
            force_diagonal_sigma_z=True,
        )

        # Both must produce valid outputs
        assert np.isfinite(result_diag["condition_number"])

        # Diagonal should have different (typically better) conditioning
        # because zeroing cross-factor correlations reduces off-diagonal coupling
        eig_full = result_full["eigenvalues_full"]
        eig_diag = result_diag["eigenvalues_full"]

        assert len(eig_full) == len(eig_diag), "Same number of eigenvalues"

        # Diagonal eigenvalue concentration: top-1 fraction should differ
        top1_full = eig_full[0] / max(eig_full.sum(), 1e-15)
        top1_diag = eig_diag[0] / max(eig_diag.sum(), 1e-15)

        # Not testing direction because it depends on data, just that they differ
        assert not np.isclose(top1_full, top1_diag, atol=1e-8), (
            f"Diagonal override must change eigenvalue distribution "
            f"(full: {top1_full:.4f}, diag: {top1_diag:.4f})"
        )

    def test_build_risk_model_b_a_override(
        self, experiment_data: dict[str, Any],
    ) -> None:
        """B_A_override replaces VAE B_A — verify via B_A_used in output."""
        d = experiment_data
        rng = np.random.RandomState(SEED + 1)

        # Create alternative B_A with different shape (fewer factors)
        n = len(d["stock_ids"])
        B_alt = rng.randn(n, 4) * 0.2

        result = build_risk_model_with_overrides(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
            B_A_override=B_alt,
        )

        # B_A_used must be the override, not the original
        np.testing.assert_array_equal(
            result["B_A_used"], B_alt,
            err_msg="B_A_used must equal B_A_override when provided",
        )

        # Output dimensions should reflect the override
        assert result["V"].shape[0] == 4, (
            f"V rotation matrix width must match override AU=4, got {result['V'].shape[0]}"
        )


# ---------------------------------------------------------------------------
# TestRunPortfolioFromRiskModel
# ---------------------------------------------------------------------------


class TestRunPortfolioFromRiskModel:
    """Tests for run_portfolio_from_risk_model (Tier 0 portfolio)."""

    @pytest.fixture()
    def risk_model(self) -> tuple[dict[str, Any], PipelineConfig]:
        """Pre-built risk model for portfolio tests."""
        d = _make_synthetic_experiment_data()
        rm = build_risk_model_with_overrides(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
        )
        return rm, d["config"]

    def test_portfolio_equal_weight(
        self, risk_model: tuple[dict[str, Any], PipelineConfig],
    ) -> None:
        """Equal-weight returns 1/n weights that sum to 1."""
        rm, config = risk_model
        result = run_portfolio_from_risk_model(
            rm, config, equal_weight_only=True,
        )

        w = result["w_opt"]
        n_port = rm["n_port"]

        # Weights sum to 1
        np.testing.assert_allclose(
            w.sum(), 1.0, atol=1e-12,
            err_msg="Equal weights must sum to 1.0",
        )

        # All weights equal 1/n
        expected_w = np.ones(n_port) / n_port
        np.testing.assert_allclose(
            w, expected_w, atol=1e-12,
            err_msg="All weights must be exactly 1/n",
        )

        # Entropy and variance must be finite
        assert np.isfinite(result["H_opt"]), "Entropy must be finite"
        assert np.isfinite(result["variance_opt"]), "Variance must be finite"
        assert result["variance_opt"] > 0, "Variance must be positive"
        assert result["n_active"] == n_port, "All positions active for EW"
        assert result["solver_stats"]["method"] == "equal_weight"

    def test_portfolio_min_variance(
        self, risk_model: tuple[dict[str, Any], PipelineConfig],
    ) -> None:
        """Min-variance returns valid weights with lower variance than EW."""
        rm, config = risk_model

        # EW baseline
        result_ew = run_portfolio_from_risk_model(
            rm, config, equal_weight_only=True,
        )
        var_ew = result_ew["variance_opt"]

        # Min-variance
        result_mv = run_portfolio_from_risk_model(
            rm, config, min_variance_only=True,
        )
        w_mv = result_mv["w_opt"]
        var_mv = result_mv["variance_opt"]

        # Weights sum to 1, all non-negative
        np.testing.assert_allclose(
            w_mv.sum(), 1.0, atol=1e-8,
            err_msg="Min-var weights must sum to 1.0",
        )
        assert np.all(w_mv >= -1e-10), "Weights must be non-negative"

        # Min-variance must improve on equal-weight (by definition)
        assert var_mv <= var_ew + 1e-12, (
            f"Min-var variance ({var_mv:.6e}) must be <= EW variance ({var_ew:.6e})"
        )

        # Solver stats
        assert result_mv["solver_stats"]["method"] == "min_variance"
        assert result_mv["n_active"] >= 1, "Must have at least 1 active position"


# ---------------------------------------------------------------------------
# TestComputeOOSMetrics
# ---------------------------------------------------------------------------


class TestComputeOOSMetrics:
    """Tests for compute_oos_metrics."""

    def test_oos_metrics_complete(self) -> None:
        """All expected keys present, values in valid ranges."""
        d = _make_synthetic_experiment_data()
        n = len(d["stock_ids"])

        # Equal-weight portfolio
        w = np.ones(n) / n
        metrics = compute_oos_metrics(
            w, d["returns_oos"], d["stock_ids"],
            transaction_cost_bps=10.0,
        )

        # All expected keys present
        expected_keys = {
            "sharpe", "ann_return", "ann_vol", "max_drawdown",
            "calmar", "cumulative_return", "n_active", "herfindahl",
        }
        assert expected_keys == set(metrics.keys()), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

        # Sharpe is finite
        assert np.isfinite(metrics["sharpe"]), (
            f"Sharpe must be finite, got {metrics['sharpe']}"
        )

        # Max drawdown in [0, 1]
        assert 0.0 <= metrics["max_drawdown"] <= 1.0, (
            f"Max drawdown must be in [0, 1], got {metrics['max_drawdown']}"
        )

        # Annualized vol must be positive (non-trivial OOS period)
        assert metrics["ann_vol"] > 0, (
            f"Ann vol must be positive, got {metrics['ann_vol']}"
        )

        # n_active = n for equal-weight
        assert metrics["n_active"] == n, (
            f"EW portfolio should have all {n} stocks active, got {metrics['n_active']}"
        )

        # Herfindahl = 1/n for equal-weight
        expected_herf = 1.0 / n
        assert abs(metrics["herfindahl"] - expected_herf) < 1e-10, (
            f"Herfindahl for EW must be 1/n={expected_herf:.6f}, "
            f"got {metrics['herfindahl']:.6f}"
        )


# ---------------------------------------------------------------------------
# TestFactorQualityProfile
# ---------------------------------------------------------------------------


class TestFactorQualityProfile:
    """Tests for compute_factor_quality_profile (Tier 1)."""

    def test_factor_quality_all_metrics_present(self) -> None:
        """All expected keys in output, values finite."""
        d = _make_synthetic_experiment_data()

        profile = compute_factor_quality_profile(
            d["B_A"], d["returns"], d["stock_ids"],
            n_random_trials=10,  # Reduced for speed
            seed=SEED,
        )

        expected_keys = {
            "cs_r2", "cs_r2_std", "cs_r2_by_date",
            "random_baseline_r2", "random_baseline_std",
            "effective_rank", "condition_number",
            "factor_autocorr", "factor_autocorr_mean",
            "top_1_eigenvalue_pct", "top_3_eigenvalue_pct",
            "singular_values",
        }
        assert expected_keys.issubset(profile.keys()), (
            f"Missing keys: {expected_keys - profile.keys()}"
        )

        # Scalar values must be finite
        for key in ("cs_r2", "cs_r2_std", "random_baseline_r2",
                     "effective_rank", "condition_number",
                     "factor_autocorr_mean",
                     "top_1_eigenvalue_pct", "top_3_eigenvalue_pct"):
            assert np.isfinite(profile[key]), f"{key} must be finite, got {profile[key]}"

        # R-squared in [0, 1] (approximately — can be slightly negative for OLS)
        assert -0.1 <= profile["cs_r2"] <= 1.0, (
            f"CS R-squared must be in [-0.1, 1], got {profile['cs_r2']}"
        )

        # Effective rank >= 1 (at least one non-zero singular value)
        assert profile["effective_rank"] >= 1.0, (
            f"Effective rank must be >= 1, got {profile['effective_rank']}"
        )

        # Singular values shape
        assert profile["singular_values"].shape[0] == d["B_A"].shape[1], (
            "Singular values count must match AU"
        )
        assert np.all(profile["singular_values"] >= 0), "Singular values must be non-negative"

        # Eigenvalue percentages in [0, 1]
        assert 0 < profile["top_1_eigenvalue_pct"] <= 1.0
        assert 0 < profile["top_3_eigenvalue_pct"] <= 1.0
        assert profile["top_3_eigenvalue_pct"] >= profile["top_1_eigenvalue_pct"]

    def test_factor_quality_random_baseline_lower(self) -> None:
        """For structured B_A, random baseline R-squared < VAE R-squared."""
        d = _make_synthetic_experiment_data()

        profile = compute_factor_quality_profile(
            d["B_A"], d["returns"], d["stock_ids"],
            n_random_trials=50,  # Enough trials for stable estimate
            seed=SEED,
        )

        cs_r2 = profile["cs_r2"]
        random_r2 = profile["random_baseline_r2"]

        # VAE B_A encodes true factor structure, so CS R-squared must exceed
        # random baseline by a meaningful margin
        assert cs_r2 > random_r2, (
            f"Structured B_A CS R-squared ({cs_r2:.4f}) must exceed "
            f"random baseline ({random_r2:.4f})"
        )

        # The margin should be at least 1 percentage point for structured data
        assert cs_r2 - random_r2 > 0.01, (
            f"Structured B_A must exceed random by >1pp: "
            f"VAE={cs_r2:.4f}, random={random_r2:.4f}, diff={cs_r2 - random_r2:.4f}"
        )


# ---------------------------------------------------------------------------
# TestParameterSweep
# ---------------------------------------------------------------------------


class TestParameterSweep:
    """Tests for run_parameter_sweep (Tier 2)."""

    @pytest.mark.filterwarnings("ignore")
    def test_parameter_sweep_returns_dataframe(self) -> None:
        """Sweep phi over [0, 5, 10], verify DataFrame with correct shape."""
        d = _make_synthetic_experiment_data()
        phi_values = [0.0, 5.0, 10.0]

        result = run_parameter_sweep(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
            d["returns_oos"],
            param_name="phi",
            param_values=phi_values,
        )

        assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
        assert len(result) == len(phi_values), (
            f"Expected {len(phi_values)} rows, got {len(result)}"
        )

        # Required columns
        expected_cols = {
            "param_value", "sharpe", "ann_return", "ann_vol",
            "max_drawdown", "n_signal", "n_active", "condition_number",
        }
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

        # param_value column matches input
        assert list(result["param_value"]) == phi_values, (
            "param_value column must match input phi_values"
        )

        # At least some Sharpe values should be finite (solver may fail on extreme params)
        finite_sharpes = result["sharpe"].dropna()
        assert len(finite_sharpes) >= 1, (
            "At least one phi value must produce finite Sharpe"
        )


# ---------------------------------------------------------------------------
# TestRunAllTier0
# ---------------------------------------------------------------------------


class TestRunAllTier0:
    """Tests for run_all_tier0 (full Tier 0 comparison)."""

    @pytest.mark.filterwarnings("ignore")
    def test_run_all_tier0_returns_comparison(self) -> None:
        """Returns DataFrame with 7 rows (baseline + 6 experiments)."""
        d = _make_synthetic_experiment_data()

        result = run_all_tier0(
            d["B_A"], d["returns"], d["trailing_vol"],
            d["stock_ids"], d["config"],
            d["train_start"], d["train_end"],
            d["returns_oos"],
        )

        assert isinstance(result, pd.DataFrame), "Must return a DataFrame"

        # 7 experiments: baseline + 6 component substitutions
        assert len(result) == 7, (
            f"Expected 7 rows (baseline + 6 experiments), got {len(result)}"
        )

        # Check experiment names are present
        assert "experiment" in result.columns, "Must have 'experiment' column"
        experiment_names = set(result["experiment"])

        expected_experiments = {
            "T0.0_baseline",
            "T0.1_pca_oracle",
            "T0.2_diagonal_sigma_z",
            "T0.3_equal_weight",
            "T0.4_min_variance",
            "T0.5_phi_zero",
            "T0.6_all_factors",
        }
        assert expected_experiments.issubset(experiment_names), (
            f"Missing experiments: {expected_experiments - experiment_names}"
        )

        # Baseline Sharpe must be finite
        baseline_mask = result["experiment"] == "T0.0_baseline"
        baseline_sharpe = float(result.loc[baseline_mask, "sharpe"].values[0])
        assert np.isfinite(baseline_sharpe), (
            f"Baseline Sharpe must be finite, got {baseline_sharpe}"
        )
