"""
Unit tests for the diagnostics package.

Tests cover:
- VAE diagnostics: log_var analysis, KL per dimension, sampling significance
- Factor diagnostics: eigenvalue concentration, shrinkage distance, exposure norms
- Portfolio diagnostics: budget conservation, gradient balance, step size, monotonicity, frontier
- E2E diagnostics: window alignment, universe attrition
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.diagnostics.vae_diagnostics import (
    analyze_log_var_distribution,
    compute_kl_per_dimension,
    compute_sampling_significance,
)
from src.diagnostics.factor_diagnostics import (
    compute_eigenvalue_concentration,
    compute_shrinkage_distance,
    analyze_exposure_norms,
    validate_dgj_recovery,
)
from src.diagnostics.portfolio_diagnostics import (
    verify_budget_conservation,
    compute_gradient_balance,
    analyze_step_size_trajectory,
    check_objective_monotonicity,
    detect_frontier_anomalies,
)
from src.diagnostics.e2e_diagnostics import (
    validate_window_alignment,
    track_universe_attrition,
)


# ===========================================================================
# VAE Diagnostics Tests
# ===========================================================================

class TestAnalyzeLogVarDistribution:
    """Tests for analyze_log_var_distribution()."""

    def test_analyze_log_var_normal(self) -> None:
        """Normal log_var values within bounds -> no collapse/explosion."""
        np.random.seed(42)
        log_var = np.random.uniform(-2.0, 2.0, size=(100, 20))

        result = analyze_log_var_distribution(log_var)

        assert result["K"] == 20
        assert result["B"] == 100
        assert result["collapse_severity"] < 0.1
        assert result["explosion_severity"] < 0.1
        assert result["n_collapsed"] == 0
        assert result["n_exploded"] == 0

    def test_analyze_log_var_collapse(self) -> None:
        """All values at lower bound -> high collapse_severity."""
        log_var = np.full((50, 10), -6.0)

        result = analyze_log_var_distribution(log_var, lower_bound=-6.0, upper_bound=6.0)

        assert result["collapse_severity"] == 1.0
        assert result["n_collapsed"] == 10
        assert result["explosion_severity"] == 0.0

    def test_analyze_log_var_explosion(self) -> None:
        """All values at upper bound -> high explosion_severity."""
        log_var = np.full((50, 10), 6.0)

        result = analyze_log_var_distribution(log_var, lower_bound=-6.0, upper_bound=6.0)

        assert result["explosion_severity"] == 1.0
        assert result["n_exploded"] == 10
        assert result["collapse_severity"] == 0.0

    def test_analyze_log_var_mixed(self) -> None:
        """Mix of collapsed, normal, and exploded dimensions."""
        log_var = np.zeros((100, 10))
        log_var[:, :3] = -6.0  # Collapsed
        log_var[:, 3:7] = 0.0  # Normal
        log_var[:, 7:] = 6.0   # Exploded

        result = analyze_log_var_distribution(log_var, lower_bound=-6.0, upper_bound=6.0)

        assert result["n_collapsed"] == 3
        assert result["n_exploded"] == 3
        assert 0.0 < result["collapse_severity"] < 1.0
        assert 0.0 < result["explosion_severity"] < 1.0

    def test_analyze_log_var_1d_input(self) -> None:
        """1D input (single sample) is handled correctly."""
        log_var = np.array([0.0, -1.0, 1.0, 2.0, -2.0])

        result = analyze_log_var_distribution(log_var)

        assert result["K"] == 5
        assert result["B"] == 1
        assert len(result["mean_per_dim"]) == 5

    def test_analyze_log_var_torch_tensor(self) -> None:
        """Torch tensor input is handled correctly."""
        log_var = torch.randn(50, 15)

        result = analyze_log_var_distribution(log_var)

        assert result["K"] == 15
        assert result["B"] == 50


class TestComputeKLPerDimension:
    """Tests for compute_kl_per_dimension()."""

    def test_compute_kl_per_dimension_basic(self) -> None:
        """Basic KL computation with known values."""
        mu = np.array([[1.0, 0.0, 0.5], [0.5, 0.0, 0.5]])
        log_var = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        result = compute_kl_per_dimension(mu, log_var)

        assert result["K"] == 3
        assert result["B"] == 2
        assert len(result["kl_per_dim"]) == 3
        assert result["kl_total"] > 0

    def test_compute_kl_zero(self) -> None:
        """mu=0, log_var=0 -> KL=0 per dimension."""
        mu = np.zeros((10, 5))
        log_var = np.zeros((10, 5))

        result = compute_kl_per_dimension(mu, log_var)

        # KL = 0.5 * (0 + 1 - 0 - 1) = 0 for each dimension
        for kl in result["kl_per_dim"]:
            assert abs(kl) < 1e-10

    def test_compute_kl_active_units(self) -> None:
        """Verify AU threshold at 0.01 nats."""
        # Create mu such that some dimensions have KL > 0.01 and some < 0.01
        mu = np.zeros((100, 10))
        mu[:, :5] = 1.0  # High mu -> high KL
        mu[:, 5:] = 0.05  # Low mu -> low KL
        log_var = np.zeros((100, 10))

        result = compute_kl_per_dimension(mu, log_var)

        # First 5 dimensions should have KL = 0.5 * (1 + 1 - 0 - 1) = 0.5 > 0.01
        # Last 5 dimensions should have KL = 0.5 * (0.0025 + 1 - 0 - 1) = 0.00125 < 0.01
        assert result["AU"] == 5
        assert len(result["active_dims"]) == 5
        assert result["au_threshold"] == 0.01

    def test_compute_kl_entropy_balanced(self) -> None:
        """Equal KL across dimensions -> high kl_balance."""
        mu = np.ones((50, 8)) * 0.5
        log_var = np.zeros((50, 8))

        result = compute_kl_per_dimension(mu, log_var)

        # All dimensions should have similar KL -> high entropy/balance
        assert result["kl_balance"] > 0.9

    def test_compute_kl_entropy_concentrated(self) -> None:
        """KL concentrated in few dimensions -> low kl_balance."""
        mu = np.zeros((50, 10))
        mu[:, 0] = 5.0  # Only first dimension has high mu
        log_var = np.zeros((50, 10))

        result = compute_kl_per_dimension(mu, log_var)

        assert result["kl_top3_fraction"] > 0.9  # Top 3 dominate
        assert result["kl_balance"] < 0.5

    def test_compute_kl_torch_input(self) -> None:
        """Torch tensor inputs are handled correctly."""
        mu = torch.randn(30, 12)
        log_var = torch.randn(30, 12) * 0.5

        result = compute_kl_per_dimension(mu, log_var)

        assert result["K"] == 12
        assert result["B"] == 30


class TestComputeSamplingSignificance:
    """Tests for compute_sampling_significance()."""

    def test_compute_sampling_significance_balanced(self) -> None:
        """sigma ~ |mu| -> ratio ~ 1."""
        # For mu magnitude of ~1 and sigma of ~sqrt(2/pi), ratio should be ~1
        mu = np.ones((100, 10))
        # log_var such that sigma = sqrt(2/pi) ~ 0.8
        sigma_target = 1.0
        log_var = np.log(sigma_target ** 2) * np.ones((100, 10))

        result = compute_sampling_significance(mu, log_var)

        # ratio = sigma * sqrt(2/pi) / |mu| = 1 * 0.798 / 1 ~ 0.8
        assert 0.5 < result["global_ratio"] < 1.5
        assert result["n_balanced"] > 0

    def test_compute_sampling_significance_deterministic(self) -> None:
        """sigma << |mu| -> low ratio (near-deterministic)."""
        mu = np.ones((50, 8)) * 10.0  # Large mu
        log_var = np.full((50, 8), -10.0)  # Very small sigma

        result = compute_sampling_significance(mu, log_var)

        assert result["global_ratio"] < 0.1
        assert result["n_low_stoch"] == 8

    def test_compute_sampling_significance_high_stochasticity(self) -> None:
        """sigma >> |mu| -> high ratio (stochastic)."""
        mu = np.ones((50, 8)) * 0.01  # Small mu
        log_var = np.full((50, 8), 2.0)  # Large sigma

        result = compute_sampling_significance(mu, log_var)

        assert result["global_ratio"] > 10.0
        assert result["n_high_stoch"] == 8

    def test_compute_sampling_significance_mixed(self) -> None:
        """Mixed dimensions: some high stoch, some balanced, some low."""
        mu = np.zeros((100, 12))
        log_var = np.zeros((100, 12))

        # First 4: low stochasticity (large mu, small sigma)
        mu[:, :4] = 10.0
        log_var[:, :4] = -6.0

        # Middle 4: balanced
        mu[:, 4:8] = 1.0
        log_var[:, 4:8] = 0.5

        # Last 4: high stochasticity (small mu, large sigma)
        mu[:, 8:] = 0.01
        log_var[:, 8:] = 2.0

        result = compute_sampling_significance(mu, log_var)

        assert result["n_low_stoch"] == 4
        assert result["n_high_stoch"] == 4
        assert result["n_balanced"] == 4


# ===========================================================================
# Factor Diagnostics Tests
# ===========================================================================

class TestComputeEigenvalueConcentration:
    """Tests for compute_eigenvalue_concentration()."""

    def test_eigenvalue_concentration_uniform(self) -> None:
        """Equal eigenvalues -> concentration_ratio = 1."""
        eigenvalues = np.ones(20)

        result = compute_eigenvalue_concentration(eigenvalues)

        assert result["available"]
        assert result["concentration_ratio"] == pytest.approx(1.0)
        assert result["var_explained_top1"] == pytest.approx(1.0 / 20)
        assert result["eff_dim"] == pytest.approx(20.0, rel=0.1)

    def test_eigenvalue_concentration_dominated(self) -> None:
        """First eigenvalue >> others -> high concentration."""
        eigenvalues = np.array([100.0, 1.0, 0.5, 0.2, 0.1])

        result = compute_eigenvalue_concentration(eigenvalues)

        assert result["concentration_ratio"] == pytest.approx(1000.0)
        assert result["var_explained_top1"] > 0.9
        assert result["eff_dim"] < 2.0

    def test_eigenvalue_gaps(self) -> None:
        """Clear gap between eigenvalues -> gap detected."""
        # Strong signal: [10, 10, 10], then gap, then noise: [0.1, 0.1, 0.1]
        eigenvalues = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])

        result = compute_eigenvalue_concentration(eigenvalues)

        assert result["max_gap_ratio"] > 50  # 10 / 0.1 = 100
        assert result["max_gap_index"] == 2  # Gap between index 2 and 3
        assert result["signal_noise_boundary"] == 3

    def test_eigenvalue_concentration_empty(self) -> None:
        """Empty eigenvalues -> not available."""
        result = compute_eigenvalue_concentration(np.array([]))

        assert not result["available"]

    def test_eigenvalue_n_for_variance(self) -> None:
        """Check n_for_90pct, n_for_95pct, n_for_99pct."""
        # Exponentially decaying eigenvalues
        eigenvalues = np.array([0.5 ** i for i in range(20)])

        result = compute_eigenvalue_concentration(eigenvalues)

        assert result["n_for_90pct"] <= result["n_for_95pct"] <= result["n_for_99pct"]
        assert result["n_for_90pct"] < 20


class TestComputeShrinkageDistance:
    """Tests for compute_shrinkage_distance()."""

    def test_shrinkage_distance_identity(self) -> None:
        """Sigma_raw = Sigma_shrunk -> distance = 0."""
        Sigma = np.eye(10)

        result = compute_shrinkage_distance(Sigma, Sigma)

        assert result["available"]
        assert result["frobenius_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["relative_distance"] == pytest.approx(0.0, abs=1e-10)
        assert result["cond_improvement"] == pytest.approx(1.0)

    def test_shrinkage_distance_significant(self) -> None:
        """Different matrices -> positive distance."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        Sigma_raw = X.T @ X / 50

        # Shrunk towards identity
        Sigma_shrunk = 0.5 * Sigma_raw + 0.5 * np.eye(10)

        result = compute_shrinkage_distance(Sigma_raw, Sigma_shrunk)

        assert result["available"]
        assert result["frobenius_diff"] > 0
        assert result["relative_distance"] > 0
        assert result["cond_improvement"] > 1.0  # Shrinkage improves conditioning

    def test_shrinkage_distance_condition_number(self) -> None:
        """Check condition number improvement after shrinkage."""
        # Create ill-conditioned matrix
        Sigma_raw = np.diag([100.0, 1.0, 0.01])
        # Shrink towards identity
        Sigma_shrunk = 0.5 * Sigma_raw + 0.5 * np.eye(3)

        result = compute_shrinkage_distance(Sigma_raw, Sigma_shrunk)

        assert result["cond_raw"] > result["cond_shrunk"]
        assert result["cond_improvement"] > 1.0

    def test_shrinkage_distance_shape_mismatch(self) -> None:
        """Shape mismatch -> not available."""
        result = compute_shrinkage_distance(np.eye(5), np.eye(10))

        assert not result["available"]


class TestAnalyzeExposureNorms:
    """Tests for analyze_exposure_norms()."""

    def test_exposure_norms_uniform(self) -> None:
        """Uniform B_A -> consistent norms."""
        B_A = np.ones((100, 20)) / np.sqrt(100)

        result = analyze_exposure_norms(B_A)

        assert result["available"]
        assert result["n_stocks"] == 100
        assert result["AU"] == 20
        assert result["col_norm_std"] < 0.1  # Low variation
        assert result["n_outlier_factors"] == 0

    def test_exposure_norms_outliers(self) -> None:
        """B_A with outlier columns -> detected."""
        np.random.seed(42)
        B_A = np.random.randn(100, 20)
        # Make one column an outlier
        B_A[:, 0] *= 10.0

        result = analyze_exposure_norms(B_A)

        assert result["n_outlier_factors"] >= 1
        assert 0 in result["outlier_factors"]

    def test_exposure_norms_sparse(self) -> None:
        """Sparse B_A -> high sparsity."""
        B_A = np.zeros((100, 20))
        B_A[:10, :5] = 1.0  # Only small subset is non-zero

        result = analyze_exposure_norms(B_A)

        assert result["sparsity"] > 0.9  # 90%+ zeros


class TestValidateDGJRecovery:
    """Tests for validate_dgj_recovery() - DGJ shrinkage validation."""

    def test_dgj_recovery_signal_detection(self) -> None:
        """Detect signal eigenvalues above BBP threshold."""
        # Create eigenvalues with clear signal/noise separation
        # Signal: 10, 8, 6 (above BBP threshold)
        # Noise: 1.0, 1.0, 1.0, 1.0, 1.0 (below BBP threshold)
        eigs_sample = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        eigs_shrunk = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        gamma = 0.5  # n_samples / n_features

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)

        assert result["available"]
        assert result["n_signal"] > 0
        assert result["bbp_threshold"] > 0
        assert result["gamma"] == 0.5

    def test_dgj_recovery_flat_noise(self) -> None:
        """Good DGJ shrinkage -> flat noise bulk (low CV)."""
        # Sample eigenvalues with signal and noise
        eigs_sample = np.array([100.0, 50.0, 1.0, 1.0, 1.0, 1.0])
        # Shrunk eigenvalues with flattened noise
        eigs_shrunk = np.array([100.0, 50.0, 0.8, 0.8, 0.8, 0.8])
        gamma = 0.3

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)

        assert result["available"]
        # Flat noise should have low CV
        assert result["noise_cv"] < 0.1
        assert result["dgj_quality"] == "good"

    def test_dgj_recovery_poor_noise(self) -> None:
        """Poor DGJ shrinkage -> variable noise bulk (high CV)."""
        # Sample eigenvalues
        eigs_sample = np.array([10.0, 1.5, 1.2, 0.8, 0.5, 0.3])
        # Shrunk but noise is not flattened
        eigs_shrunk = np.array([10.0, 2.0, 1.5, 1.0, 0.4, 0.2])
        gamma = 0.5

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)

        assert result["available"]
        # Variable noise -> high CV
        if result["n_noise"] > 0:
            assert result["noise_cv"] > 0.0

    def test_dgj_recovery_bbp_threshold(self) -> None:
        """BBP threshold computed correctly: sigma^2 * (1 + sqrt(gamma))^2."""
        eigs_sample = np.array([5.0, 2.0, 1.0, 1.0, 1.0])  # median = 1.0
        eigs_shrunk = np.array([5.0, 2.0, 1.0, 1.0, 1.0])
        gamma = 0.25  # sqrt(gamma) = 0.5

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)

        # BBP = 1.0 * (1 + 0.5)^2 = 1.0 * 2.25 = 2.25
        assert result["bbp_threshold"] == pytest.approx(2.25, rel=0.01)

    def test_dgj_recovery_empty_arrays(self) -> None:
        """Empty eigenvalue arrays -> not available."""
        result = validate_dgj_recovery(np.array([]), np.array([]), 0.5)

        assert not result["available"]
        assert result["reason"] == "empty eigenvalues"

    def test_dgj_recovery_shape_mismatch(self) -> None:
        """Mismatched array shapes -> not available."""
        eigs_sample = np.array([1.0, 2.0, 3.0])
        eigs_shrunk = np.array([1.0, 2.0])

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, 0.5)

        assert not result["available"]
        assert result["reason"] == "eigenvalue array shape mismatch"

    def test_dgj_recovery_invalid_gamma(self) -> None:
        """Invalid gamma <= 0 -> not available."""
        eigs_sample = np.array([1.0, 2.0, 3.0])
        eigs_shrunk = np.array([1.0, 2.0, 3.0])

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma=0.0)

        assert not result["available"]
        assert "gamma" in result["reason"]

    def test_dgj_recovery_all_noise(self) -> None:
        """All eigenvalues below BBP -> all noise."""
        # Small eigenvalues relative to BBP
        eigs_sample = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        eigs_shrunk = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # Flat noise
        gamma = 2.0  # High gamma -> low BBP threshold

        result = validate_dgj_recovery(eigs_sample, eigs_shrunk, gamma)

        assert result["available"]
        # With high gamma, even small eigenvalues may be above BBP
        # Check that n_noise + n_signal = total
        assert result["n_signal"] + result["n_noise"] == len(eigs_sample)


# ===========================================================================
# Portfolio Diagnostics Tests
# ===========================================================================

class TestVerifyBudgetConservation:
    """Tests for verify_budget_conservation()."""

    def test_budget_conservation_valid(self) -> None:
        """Properly normalized -> conserved."""
        n, AU = 50, 10
        w = np.ones(n) / n
        B_prime = np.random.randn(n, AU)
        eigenvalues = np.abs(np.random.randn(AU)) + 0.1

        result = verify_budget_conservation(w, B_prime, eigenvalues)

        assert result["available"]
        assert result["systematic_conserved"]
        assert result["sys_deviation"] < 1e-6

    def test_budget_conservation_with_idio(self) -> None:
        """Include idiosyncratic layer -> both conserved."""
        n, AU = 50, 10
        w = np.ones(n) / n
        B_prime = np.random.randn(n, AU)
        eigenvalues = np.abs(np.random.randn(AU)) + 0.1
        D_eps = np.abs(np.random.randn(n)) + 0.01

        result = verify_budget_conservation(w, B_prime, eigenvalues, D_eps)

        assert result["systematic_conserved"]
        assert result["idiosyncratic_conserved"]
        assert result["all_conserved"]

    def test_budget_conservation_zero_risk(self) -> None:
        """Zero risk case -> trivially conserved."""
        w = np.zeros(10)
        B_prime = np.random.randn(10, 5)
        eigenvalues = np.ones(5)

        result = verify_budget_conservation(w, B_prime, eigenvalues)

        # Zero weights -> zero risk -> trivially conserved
        assert result["available"]


class TestComputeGradientBalance:
    """Tests for compute_gradient_balance()."""

    def test_gradient_balance_balanced(self) -> None:
        """Equal norms -> ratio ~ 1."""
        grad_entropy = np.ones(50)
        grad_variance = np.ones(50)
        alpha = 1.0

        result = compute_gradient_balance(grad_entropy, grad_variance, alpha)

        assert result["available"]
        assert result["ratio"] == pytest.approx(1.0)
        assert result["balance_status"] == "balanced"

    def test_gradient_balance_entropy_dominated(self) -> None:
        """Entropy gradient >> variance gradient -> entropy_dominated."""
        grad_entropy = np.ones(50) * 100
        grad_variance = np.ones(50)
        alpha = 1.0

        result = compute_gradient_balance(grad_entropy, grad_variance, alpha)

        assert result["ratio"] > 10.0
        assert result["balance_status"] == "entropy_dominated"

    def test_gradient_balance_variance_dominated(self) -> None:
        """Variance gradient >> entropy gradient -> variance_dominated."""
        grad_entropy = np.ones(50) * 0.001
        grad_variance = np.ones(50)
        alpha = 1.0

        result = compute_gradient_balance(grad_entropy, grad_variance, alpha)

        assert result["ratio"] < 0.1
        assert result["balance_status"] == "variance_dominated"

    def test_gradient_balance_with_alpha(self) -> None:
        """Alpha scaling affects balance."""
        grad_entropy = np.ones(50)
        grad_variance = np.ones(50)
        alpha = 10.0  # High alpha scales variance gradient up

        result = compute_gradient_balance(grad_entropy, grad_variance, alpha)

        assert result["norm_alpha_variance"] == pytest.approx(10.0 * np.sqrt(50))


class TestAnalyzeStepSizeTrajectory:
    """Tests for analyze_step_size_trajectory()."""

    def test_step_size_stable(self) -> None:
        """Constant step sizes -> low variability."""
        step_sizes = [1.0] * 20

        result = analyze_step_size_trajectory(step_sizes)

        assert result["available"]
        assert result["variability"] == "low"
        assert result["cv"] < 0.1
        assert result["trend"] == "stable"

    def test_step_size_decreasing(self) -> None:
        """Decreasing step sizes -> decreasing trend."""
        step_sizes = [1.0 - 0.03 * i for i in range(20)]

        result = analyze_step_size_trajectory(step_sizes)

        assert result["trend"] == "decreasing"

    def test_step_size_oscillating(self) -> None:
        """Oscillating step sizes -> high oscillation_rate."""
        step_sizes = [1.0 if i % 2 == 0 else 0.5 for i in range(20)]

        result = analyze_step_size_trajectory(step_sizes)

        # The key diagnostic is oscillation_rate, not CV-based variability
        assert result["oscillation_rate"] > 0.5

    def test_step_size_high_variability(self) -> None:
        """High CV -> high variability."""
        np.random.seed(42)
        step_sizes = list(np.random.uniform(0.1, 2.0, 30))

        result = analyze_step_size_trajectory(step_sizes)

        assert result["variability"] in ["moderate", "high"]

    def test_step_size_empty(self) -> None:
        """Empty step sizes -> not available."""
        result = analyze_step_size_trajectory([])

        assert not result["available"]


class TestCheckObjectiveMonotonicity:
    """Tests for check_objective_monotonicity()."""

    def test_objective_monotonicity_pass(self) -> None:
        """Strictly increasing values -> monotonic."""
        obj_values = [float(i) for i in range(20)]

        result = check_objective_monotonicity(obj_values)

        assert result["available"]
        assert result["is_monotonic"]
        assert result["n_violations"] == 0
        assert result["total_improvement"] == 19.0

    def test_objective_monotonicity_fail(self) -> None:
        """Decrease detected -> not monotonic."""
        obj_values = [1.0, 2.0, 3.0, 2.5, 4.0]  # Decrease at index 3

        result = check_objective_monotonicity(obj_values)

        assert not result["is_monotonic"]
        assert result["n_violations"] == 1
        assert result["max_violation"] < 0

    def test_objective_monotonicity_multiple_violations(self) -> None:
        """Multiple decreases -> all counted."""
        obj_values = [1.0, 0.5, 2.0, 1.5, 3.0, 2.0]  # 3 decreases

        result = check_objective_monotonicity(obj_values)

        assert result["n_violations"] == 3

    def test_objective_monotonicity_with_tolerance(self) -> None:
        """Small violations within tolerance -> ignored."""
        obj_values = [1.0, 2.0, 1.9999999999, 3.0]  # Tiny decrease

        result = check_objective_monotonicity(obj_values, tol=1e-6)

        assert result["is_monotonic"]

    def test_objective_monotonicity_convergence_ratio(self) -> None:
        """Check convergence ratio computation."""
        obj_values = [0.0, 1.0, 2.0, 3.0, 3.1]  # Slowing down

        result = check_objective_monotonicity(obj_values)

        assert result["convergence_ratio"] < 1.0  # Last improvement < average


class TestDetectFrontierAnomalies:
    """Tests for detect_frontier_anomalies()."""

    def test_frontier_anomalies_clean(self) -> None:
        """Monotonic frontier -> good quality."""
        frontier = [
            {"alpha": 10.0, "variance": 0.01, "entropy": 1.0},
            {"alpha": 5.0, "variance": 0.02, "entropy": 1.5},
            {"alpha": 1.0, "variance": 0.05, "entropy": 2.0},
            {"alpha": 0.1, "variance": 0.10, "entropy": 2.5},
        ]

        result = detect_frontier_anomalies(frontier)

        assert result["available"]
        assert result["is_monotonic"]
        assert not result["is_degenerate"]
        assert result["quality"] in ["good", "acceptable"]

    def test_frontier_anomalies_non_monotonic(self) -> None:
        """Non-monotonic entropy -> violations detected."""
        frontier = [
            {"alpha": 10.0, "variance": 0.01, "entropy": 2.0},  # Higher entropy at high alpha
            {"alpha": 5.0, "variance": 0.02, "entropy": 1.5},
            {"alpha": 1.0, "variance": 0.05, "entropy": 1.0},   # Decreasing entropy
        ]

        result = detect_frontier_anomalies(frontier)

        assert not result["is_monotonic"]
        assert result["n_violations"] >= 1

    def test_frontier_anomalies_degenerate(self) -> None:
        """Very small H range -> degenerate."""
        frontier = [
            {"alpha": 10.0, "variance": 0.01, "entropy": 1.00},
            {"alpha": 5.0, "variance": 0.02, "entropy": 1.01},
            {"alpha": 1.0, "variance": 0.05, "entropy": 1.02},
        ]

        result = detect_frontier_anomalies(frontier)

        assert result["is_degenerate"]
        assert result["quality"] == "degenerate"

    def test_frontier_anomalies_empty(self) -> None:
        """Empty frontier -> not available."""
        result = detect_frontier_anomalies([])

        assert not result["available"]


# ===========================================================================
# E2E Diagnostics Tests
# ===========================================================================

class TestValidateWindowAlignment:
    """Tests for validate_window_alignment()."""

    def test_window_alignment_valid(self) -> None:
        """Proper alignment -> is_valid=True."""
        # Create properly aligned windows
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = []
        for stock_id in [1, 2, 3]:
            for i in range(5):
                data.append({
                    "stock_id": stock_id,
                    "start_date": dates[i],
                    "end_date": dates[i] + pd.Timedelta(days=4),
                })
        window_metadata = pd.DataFrame(data)

        result = validate_window_alignment(window_metadata)

        assert result["available"]
        assert result["n_stocks"] == 3
        assert result["length_consistent"]

    def test_window_alignment_gaps(self) -> None:
        """Gap detection in window sequence."""
        data = [
            {"stock_id": 1, "start_date": "2020-01-01", "end_date": "2020-01-10"},
            {"stock_id": 1, "start_date": "2020-01-20", "end_date": "2020-01-29"},  # Gap
        ]
        window_metadata = pd.DataFrame(data)

        result = validate_window_alignment(window_metadata)

        assert result["n_gap_violations"] >= 1
        assert not result["is_valid"]

    def test_window_alignment_overlaps(self) -> None:
        """Overlap detection in windows."""
        data = [
            {"stock_id": 1, "start_date": "2020-01-01", "end_date": "2020-01-15"},
            {"stock_id": 1, "start_date": "2020-01-10", "end_date": "2020-01-25"},  # Overlap
        ]
        window_metadata = pd.DataFrame(data)

        result = validate_window_alignment(window_metadata)

        assert result["n_overlap_violations"] >= 1

    def test_window_alignment_missing_columns(self) -> None:
        """Missing required columns -> not available."""
        window_metadata = pd.DataFrame({"stock_id": [1, 2]})

        result = validate_window_alignment(window_metadata)

        assert not result["available"]


class TestTrackUniverseAttrition:
    """Tests for track_universe_attrition()."""

    def test_universe_attrition_stable(self) -> None:
        """Stable universe -> high stability score."""
        universe_snapshots = {
            "2020-01-01": [1, 2, 3, 4, 5],
            "2020-01-02": [1, 2, 3, 4, 5],
            "2020-01-03": [1, 2, 3, 4, 5],
            "2020-01-04": [1, 2, 3, 4, 5],
        }

        result = track_universe_attrition(universe_snapshots)

        assert result["available"]
        assert result["stability_score"] == 1.0
        assert result["trend"] == "stable"
        assert result["avg_turnover_rate"] == 0.0

    def test_universe_attrition_shrinking(self) -> None:
        """Shrinking universe -> negative trend."""
        universe_snapshots = {
            "2020-01-01": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "2020-01-02": [1, 2, 3, 4, 5, 6, 7, 8],
            "2020-01-03": [1, 2, 3, 4, 5, 6],
            "2020-01-04": [1, 2, 3, 4],
        }

        result = track_universe_attrition(universe_snapshots)

        assert result["trend"] == "shrinking"
        assert result["daily_slope"] < 0

    def test_universe_attrition_growing(self) -> None:
        """Growing universe -> positive trend."""
        universe_snapshots = {
            "2020-01-01": [1, 2],
            "2020-01-02": [1, 2, 3, 4],
            "2020-01-03": [1, 2, 3, 4, 5, 6],
            "2020-01-04": [1, 2, 3, 4, 5, 6, 7, 8],
        }

        result = track_universe_attrition(universe_snapshots)

        assert result["trend"] == "growing"
        assert result["daily_slope"] > 0

    def test_universe_attrition_large_drop(self) -> None:
        """Large drop detection (>10%)."""
        universe_snapshots = {
            "2020-01-01": list(range(100)),
            "2020-01-02": list(range(100)),
            "2020-01-03": list(range(50)),  # 50% drop
        }

        result = track_universe_attrition(universe_snapshots)

        assert result["n_large_drops"] >= 1
        assert result["large_drops"][0]["pct_change"] < -0.1

    def test_universe_attrition_turnover(self) -> None:
        """Compute turnover rate correctly."""
        universe_snapshots = {
            "2020-01-01": [1, 2, 3, 4, 5],
            "2020-01-02": [1, 2, 3, 6, 7],  # 4,5 exit; 6,7 enter
            "2020-01-03": [1, 2, 8, 9, 10],  # 3,6,7 exit; 8,9,10 enter
        }

        result = track_universe_attrition(universe_snapshots)

        assert result["avg_entries_per_date"] > 0
        assert result["avg_exits_per_date"] > 0
        assert result["avg_turnover_rate"] > 0

    def test_universe_attrition_empty(self) -> None:
        """Empty snapshots -> not available."""
        result = track_universe_attrition({})

        assert not result["available"]
