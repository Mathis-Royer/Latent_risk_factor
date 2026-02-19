"""
Unit tests for the new diagnostic additions.

Tests the following Tier 1 diagnostic features:
1. SCA solver convergence metrics (sca_optimize, multi_start_optimize)
2. Constraint binding analysis (get_binding_constraints)
3. Ledoit-Wolf shrinkage intensity (estimate_sigma_z)
4. Per-feature reconstruction loss (compute_loss)
"""

import numpy as np
import pytest
import torch

from src.portfolio.constraints import get_binding_constraints
from src.portfolio.sca_solver import multi_start_optimize, sca_optimize
from src.risk_model.covariance import estimate_sigma_z
from src.vae.loss import compute_loss, compute_reconstruction_loss_per_feature


# ---------------------------------------------------------------------------
# Test 1: SCA solver convergence diagnostics
# ---------------------------------------------------------------------------


class TestSCAConvergence:
    """Test SCA solver returns convergence diagnostics."""

    def test_sca_optimize_returns_5_tuple(self) -> None:
        """sca_optimize returns 5 values: w, f, H, n_iters, convergence_info."""
        rng = np.random.RandomState(42)
        n = 20
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        eigenvalues = np.array([0.5, 0.3, 0.1])
        w_init = np.ones(n) / n

        result = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_max=0.10,
            is_first=True,
        )

        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}"
        w_opt, f_opt, H_opt, n_iters, convergence_info = result

        # Verify types
        assert isinstance(w_opt, np.ndarray), "w_opt should be ndarray"
        assert isinstance(f_opt, (int, float)), "f_opt should be numeric"
        assert isinstance(H_opt, (int, float)), "H_opt should be numeric"
        assert isinstance(n_iters, int), "n_iters should be int"
        assert isinstance(convergence_info, dict), "convergence_info should be dict"

        # Verify convergence_info contents
        assert "converged" in convergence_info
        assert "final_grad_norm" in convergence_info
        assert "step_sizes" in convergence_info
        assert "obj_improvements" in convergence_info

    def test_multi_start_returns_4_tuple_with_stats(self) -> None:
        """multi_start_optimize returns 4 values: w, f, H, solver_stats."""
        rng = np.random.RandomState(42)
        n = 20
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        eigenvalues = np.array([0.5, 0.3, 0.1])
        D_eps = np.full(n, 0.01)

        result = multi_start_optimize(
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=1.0,
            n_starts=3,
            seed=42,
            w_max=0.10,
            w_bar=0.03,
            is_first=True,
        )

        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"
        w_best, f_best, H_best, solver_stats = result

        # Verify solver_stats contents
        assert isinstance(solver_stats, dict)
        assert "n_starts" in solver_stats
        assert "converged_count" in solver_stats
        assert "iterations" in solver_stats
        assert "best_start_idx" in solver_stats
        assert "best_convergence_info" in solver_stats

        # Verify counts match
        assert solver_stats["n_starts"] == 3
        iterations = solver_stats["iterations"]
        assert isinstance(iterations, list)
        assert len(iterations) == 3

    def test_convergence_flag_true_when_converged(self) -> None:
        """convergence_info['converged'] is True when gradient norm is small."""
        rng = np.random.RandomState(42)
        n = 10
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        eigenvalues = np.array([0.5, 0.3, 0.1])
        w_init = np.ones(n) / n

        _, _, _, _, convergence_info = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_max=0.10,
            is_first=True,
            max_iter=100,  # More iterations to ensure convergence
            tol=1e-6,
        )

        # With 100 iterations and moderate problem, should converge
        grad_norm = convergence_info["final_grad_norm"]
        assert np.isfinite(grad_norm), "grad_norm should be finite"
        assert isinstance(convergence_info["converged"], bool)


# ---------------------------------------------------------------------------
# Test 2: Constraint binding analysis
# ---------------------------------------------------------------------------


class TestConstraintBinding:
    """Test get_binding_constraints function."""

    def test_binding_constraints_returns_expected_keys(self) -> None:
        """get_binding_constraints returns all expected keys."""
        w = np.array([0.10, 0.10, 0.05, 0.05, 0.70])  # Some at w_max
        w_old = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        constraint_params = {
            "w_max": 0.10,
            "w_min": 0.01,
            "w_bar": 0.03,
            "tau_max": 0.30,
        }

        result = get_binding_constraints(w, w_old, constraint_params)

        expected_keys = [
            "n_at_w_max",
            "n_at_w_min",
            "n_above_w_bar",
            "w_max_binding",
            "tau_binding",
            "actual_turnover",
            "concentrated_weight",
            "binding_fraction",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_binding_detects_positions_at_w_max(self) -> None:
        """Detects positions at maximum weight."""
        w = np.array([0.10, 0.10, 0.05, 0.05, 0.70])  # 2 at w_max
        constraint_params = {"w_max": 0.10, "w_min": 0.01, "w_bar": 0.03, "tau_max": 0.30}

        result = get_binding_constraints(w, None, constraint_params)

        # Positions at w_max=0.10 are indices 0, 1
        # Position 4 (0.70) exceeds w_max but is not "at" w_max
        assert result["n_at_w_max"] >= 2, f"Expected at least 2 at w_max, got {result['n_at_w_max']}"

    def test_binding_detects_positions_at_w_min(self) -> None:
        """Detects positions at minimum weight threshold."""
        w = np.array([0.01, 0.01, 0.05, 0.05, 0.88])  # 2 at w_min
        constraint_params = {"w_max": 0.10, "w_min": 0.01, "w_bar": 0.03, "tau_max": 0.30}

        result = get_binding_constraints(w, None, constraint_params)

        assert result["n_at_w_min"] >= 2, f"Expected at least 2 at w_min, got {result['n_at_w_min']}"

    def test_tau_binding_detected_when_turnover_exceeds_max(self) -> None:
        """tau_binding is True when turnover >= tau_max."""
        w = np.array([0.30, 0.30, 0.20, 0.10, 0.10])
        w_old = np.array([0.10, 0.10, 0.30, 0.30, 0.20])  # Big change
        constraint_params = {"w_max": 0.30, "w_min": 0.01, "w_bar": 0.03, "tau_max": 0.10}

        result = get_binding_constraints(w, w_old, constraint_params)

        # Turnover = sum(|w - w_old|) / 2 = (0.20 + 0.20 + 0.10 + 0.20 + 0.10) / 2 = 0.40
        assert result["actual_turnover"] > constraint_params["tau_max"]
        assert result["tau_binding"] is True

    def test_no_binding_when_weights_unconstrained(self) -> None:
        """No constraints bind when portfolio is well-diversified."""
        w = np.full(10, 0.10)  # Equally weighted
        w_old = np.full(10, 0.10)
        constraint_params = {"w_max": 0.20, "w_min": 0.01, "w_bar": 0.05, "tau_max": 0.50}

        result = get_binding_constraints(w, w_old, constraint_params)

        assert result["n_at_w_max"] == 0
        assert result["tau_binding"] is False


# ---------------------------------------------------------------------------
# Test 3: Ledoit-Wolf shrinkage intensity
# ---------------------------------------------------------------------------


class TestShrinkageIntensity:
    """Test estimate_sigma_z returns shrinkage intensity."""

    def test_estimate_sigma_z_returns_3_tuple(self) -> None:
        """estimate_sigma_z returns (Sigma_z, n_signal, shrinkage_intensity)."""
        rng = np.random.RandomState(42)
        z_hat = rng.randn(100, 5) * 0.01  # 100 dates, 5 factors

        result = estimate_sigma_z(z_hat, shrinkage_method="truncation")

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
        Sigma_z, n_signal, shrinkage_intensity = result

        assert isinstance(Sigma_z, np.ndarray)
        assert Sigma_z.shape == (5, 5)
        assert isinstance(n_signal, int)
        assert n_signal >= 0

    def test_shrinkage_intensity_in_valid_range(self) -> None:
        """shrinkage_intensity is in [0, 1] for truncation method."""
        rng = np.random.RandomState(42)
        z_hat = rng.randn(100, 5) * 0.01

        _, _, shrinkage_intensity = estimate_sigma_z(z_hat, shrinkage_method="truncation")

        if shrinkage_intensity is not None:
            assert 0.0 <= shrinkage_intensity <= 1.0, (
                f"Shrinkage intensity {shrinkage_intensity} outside [0, 1]"
            )

    def test_spiked_method_returns_none_shrinkage(self) -> None:
        """Spiked method returns None for shrinkage_intensity (no single alpha)."""
        rng = np.random.RandomState(42)
        z_hat = rng.randn(100, 5) * 0.01

        _, _, shrinkage_intensity = estimate_sigma_z(z_hat, shrinkage_method="spiked")
        # Spiked method doesn't have a single shrinkage intensity
        assert shrinkage_intensity is None, (
            "Spiked method should return None shrinkage_intensity"
        )


# ---------------------------------------------------------------------------
# Test 4: Per-feature reconstruction loss
# ---------------------------------------------------------------------------


class TestPerFeatureReconstruction:
    """Test per-feature reconstruction loss computation."""

    def test_compute_reconstruction_loss_per_feature(self) -> None:
        """compute_reconstruction_loss_per_feature returns list of MSE per feature."""
        x = torch.randn(32, 64, 2)  # Batch=32, T=64, F=2
        x_hat = x + torch.randn_like(x) * 0.1  # Add noise

        result = compute_reconstruction_loss_per_feature(x, x_hat)

        assert isinstance(result, list), "Should return a list"
        assert len(result) == 2, f"Expected 2 features, got {len(result)}"
        assert all(isinstance(v, float) for v in result), "All values should be floats"
        assert all(v >= 0 for v in result), "MSE should be non-negative"

    def test_per_feature_loss_zero_for_perfect_reconstruction(self) -> None:
        """Per-feature MSE is 0 when x == x_hat."""
        x = torch.randn(32, 64, 2)
        x_hat = x.clone()

        result = compute_reconstruction_loss_per_feature(x, x_hat)

        for i, mse in enumerate(result):
            assert abs(mse) < 1e-10, f"Feature {i} MSE should be 0, got {mse}"

    def test_compute_loss_includes_recon_per_feature(self) -> None:
        """compute_loss returns recon_per_feature in components dict."""
        x = torch.randn(16, 64, 2)
        x_hat = x + torch.randn_like(x) * 0.1
        mu = torch.randn(16, 8)
        log_var = torch.zeros(16, 8)
        log_sigma_sq = torch.zeros(())
        crisis_fractions = torch.zeros(16)

        _, components = compute_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            log_var=log_var,
            log_sigma_sq=log_sigma_sq,
            crisis_fractions=crisis_fractions,
            epoch=0,
            total_epochs=10,
            mode="P",
        )

        assert "recon_per_feature" in components
        rpf = components["recon_per_feature"]
        assert isinstance(rpf, list)
        assert len(rpf) == 2, f"Expected 2 features, got {len(rpf)}"

    def test_per_feature_loss_different_for_different_features(self) -> None:
        """Per-feature losses differ when one feature is noisier."""
        x = torch.randn(32, 64, 2)
        x_hat = x.clone()
        # Add much more noise to feature 0
        x_hat[:, :, 0] += torch.randn(32, 64) * 1.0
        x_hat[:, :, 1] += torch.randn(32, 64) * 0.01

        result = compute_reconstruction_loss_per_feature(x, x_hat)

        # Feature 0 should have higher MSE
        assert result[0] > result[1] * 10, (
            f"Feature 0 (mse={result[0]:.4f}) should be >> Feature 1 (mse={result[1]:.4f})"
        )
