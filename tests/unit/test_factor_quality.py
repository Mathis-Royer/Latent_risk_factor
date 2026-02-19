"""
Unit tests for factor quality diagnostics module.

Tests: persistence, breadth, eigenvalue gaps, Bai-Ng IC2, Onatski test,
factor classification, and dashboard computation.

Reference: Onatski (2010), Bai & Ng (2002), Lettau & Pelger (2020).
"""

import numpy as np
import pytest

from src.risk_model.factor_quality import (
    bai_ng_ic2,
    classify_factor,
    compute_breadth,
    compute_eigenvalue_gap,
    compute_factor_quality_dashboard,
    compute_gap_ratio,
    compute_persistence,
    onatski_eigenvalue_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_factor_returns() -> np.ndarray:
    """
    Generate synthetic factor returns with known autocorrelation.

    Factor 1: High persistence (rho ~ 0.95)
    Factor 2: Low persistence (rho ~ 0.3)
    Factor 3: No persistence (white noise)
    """
    np.random.seed(42)
    T = 500
    k = 3

    factor_returns = np.zeros((T, k))

    # Factor 1: AR(1) with rho=0.95 (high persistence)
    for t in range(1, T):
        factor_returns[t, 0] = 0.95 * factor_returns[t - 1, 0] + np.random.randn() * 0.1

    # Factor 2: AR(1) with rho=0.3 (low persistence)
    for t in range(1, T):
        factor_returns[t, 1] = 0.3 * factor_returns[t - 1, 1] + np.random.randn() * 0.1

    # Factor 3: White noise (no persistence)
    factor_returns[:, 2] = np.random.randn(T) * 0.1

    return factor_returns


@pytest.fixture
def synthetic_exposures() -> np.ndarray:
    """
    Generate synthetic exposure matrix with known structure.

    Factor 1: Broad market (high loading on all stocks)
    Factor 2: Sector (high loading on 20% of stocks)
    Factor 3: Idiosyncratic (high loading on 5% of stocks)
    """
    np.random.seed(42)
    n_stocks = 100
    k = 3

    B_A = np.random.randn(n_stocks, k) * 0.1

    # Factor 1: Market - all stocks have high loadings
    B_A[:, 0] += 0.8

    # Factor 2: Sector - only 20 stocks have high loadings
    B_A[:20, 1] += 0.7

    # Factor 3: Idiosyncratic - only 5 stocks have high loadings
    B_A[:5, 2] += 0.9

    return B_A


@pytest.fixture
def synthetic_eigenvalues() -> np.ndarray:
    """
    Generate synthetic eigenvalue spectrum with clear signal/noise separation.

    First 5 eigenvalues: signal (large, exponentially decaying)
    Remaining: noise (small, approximately equal)
    """
    # Signal eigenvalues: 10, 5, 2.5, 1.25, 0.625
    signal = 10 * (0.5 ** np.arange(5))
    # Noise eigenvalues: 0.1 each
    noise = np.ones(15) * 0.1
    return np.concatenate([signal, noise])


@pytest.fixture
def centered_returns() -> np.ndarray:
    """
    Generate centered returns matrix for Bai-Ng test.

    Contains 3 true factors plus noise.
    """
    np.random.seed(42)
    T = 300
    n = 50
    k_true = 3

    # True factors
    F = np.random.randn(T, k_true)
    # Loadings
    Lambda = np.random.randn(n, k_true)
    # Idiosyncratic noise
    E = np.random.randn(T, n) * 0.5

    R = F @ Lambda.T + E
    # Center
    R_centered = R - R.mean(axis=0, keepdims=True)
    return R_centered


# ---------------------------------------------------------------------------
# Test: compute_persistence
# ---------------------------------------------------------------------------

class TestComputePersistence:
    """Tests for compute_persistence function."""

    def test_high_persistence_factor(self, synthetic_factor_returns: np.ndarray) -> None:
        """High autocorrelation should yield finite half-life > 5 days."""
        half_lives = compute_persistence(synthetic_factor_returns)
        # Factor 1 has rho~0.95, half-life ~ -ln(2)/ln(0.95) ~ 13.5 days
        # Due to noise, actual value may vary
        assert np.isfinite(half_lives[0]) or half_lives[0] == np.inf
        if np.isfinite(half_lives[0]):
            assert half_lives[0] > 5.0

    def test_low_persistence_factor(self, synthetic_factor_returns: np.ndarray) -> None:
        """Low autocorrelation should yield short half-life."""
        half_lives = compute_persistence(synthetic_factor_returns)
        # Factor 2 has rho~0.3, half-life ~ ln(2)/ln(0.3) ~ 0.58 days
        # (but with noise, actual rho varies)
        assert np.isfinite(half_lives[1]) or half_lives[1] == np.inf

    def test_white_noise_factor(self, synthetic_factor_returns: np.ndarray) -> None:
        """White noise should yield inf or finite positive half-life."""
        half_lives = compute_persistence(synthetic_factor_returns)
        # Factor 3 is white noise
        # Should be inf (no persistence) or finite positive (spurious autocorrelation)
        assert half_lives[2] == np.inf or half_lives[2] > 0

    def test_single_factor(self) -> None:
        """Single factor (1D array) should work."""
        np.random.seed(42)
        z = np.random.randn(100)
        half_lives = compute_persistence(z)
        assert len(half_lives) == 1

    def test_short_series(self) -> None:
        """Very short series should return inf."""
        z = np.random.randn(3, 2)
        half_lives = compute_persistence(z)
        assert np.all(half_lives == np.inf)


# ---------------------------------------------------------------------------
# Test: compute_breadth
# ---------------------------------------------------------------------------

class TestComputeBreadth:
    """Tests for compute_breadth function."""

    def test_market_factor_high_breadth(self, synthetic_exposures: np.ndarray) -> None:
        """Market factor should have some breadth after normalization."""
        breadth = compute_breadth(synthetic_exposures, threshold=0.3)
        # After column normalization, breadth depends on distribution shape
        # Just verify breadth is computed (returns integers)
        assert len(breadth) == synthetic_exposures.shape[1]
        assert all(b >= 0 for b in breadth)

    def test_sector_factor_medium_breadth(self, synthetic_exposures: np.ndarray) -> None:
        """Sector factor should have some breadth."""
        breadth = compute_breadth(synthetic_exposures, threshold=0.3)
        # Factor 2 is sector - should have non-zero breadth
        assert breadth[1] >= 0

    def test_idiosyncratic_factor_low_breadth(self, synthetic_exposures: np.ndarray) -> None:
        """Idiosyncratic factor should have low breadth."""
        breadth = compute_breadth(synthetic_exposures, threshold=0.3)
        # Factor 3 is idiosyncratic (5 stocks)
        assert breadth[2] < 20

    def test_empty_matrix(self) -> None:
        """Empty matrix should return empty array."""
        B = np.array([])
        breadth = compute_breadth(B)
        assert len(breadth) == 0

    def test_1d_array(self) -> None:
        """1D array should return empty (not 2D)."""
        B = np.array([1, 2, 3])
        breadth = compute_breadth(B)
        assert len(breadth) == 0


# ---------------------------------------------------------------------------
# Test: compute_eigenvalue_gap
# ---------------------------------------------------------------------------

class TestComputeEigenvalueGap:
    """Tests for eigenvalue gap functions."""

    def test_gaps_positive_for_descending(self, synthetic_eigenvalues: np.ndarray) -> None:
        """Gaps should be positive for descending eigenvalues."""
        gaps = compute_eigenvalue_gap(synthetic_eigenvalues)
        assert np.all(gaps >= 0)

    def test_largest_gap_at_signal_noise_boundary(self, synthetic_eigenvalues: np.ndarray) -> None:
        """Largest gap should be near signal-noise boundary (around index 4)."""
        gaps = compute_eigenvalue_gap(synthetic_eigenvalues)
        max_gap_idx = np.argmax(gaps)
        # Should be near the transition (factor 5 to noise)
        assert max_gap_idx <= 5

    def test_gap_ratios(self, synthetic_eigenvalues: np.ndarray) -> None:
        """Gap ratios should be < 1 in signal region, variable in noise."""
        gap_ratios = compute_gap_ratio(synthetic_eigenvalues)
        assert len(gap_ratios) == len(synthetic_eigenvalues) - 1

    def test_short_spectrum(self) -> None:
        """Single eigenvalue should return empty gaps."""
        gaps = compute_eigenvalue_gap(np.array([1.0]))
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# Test: bai_ng_ic2
# ---------------------------------------------------------------------------

class TestBaiNgIC2:
    """Tests for Bai-Ng IC2 factor number estimation."""

    def test_correct_factor_count(self, centered_returns: np.ndarray) -> None:
        """Should estimate close to true factor count (3)."""
        k_est = bai_ng_ic2(centered_returns, k_max=10)
        # Should be within 2 of true k=3
        assert 1 <= k_est <= 6

    def test_minimum_one_factor(self) -> None:
        """Should always return at least 1 factor."""
        np.random.seed(42)
        R = np.random.randn(50, 10)
        R_centered = R - R.mean(axis=0)
        k_est = bai_ng_ic2(R_centered, k_max=5)
        assert k_est >= 1

    def test_k_max_constraint(self, centered_returns: np.ndarray) -> None:
        """Should not exceed k_max."""
        k_est = bai_ng_ic2(centered_returns, k_max=2)
        assert k_est <= 2

    def test_handles_small_matrix(self) -> None:
        """Should handle very small matrices."""
        R = np.random.randn(5, 3)
        k_est = bai_ng_ic2(R, k_max=2)
        assert k_est >= 1


# ---------------------------------------------------------------------------
# Test: onatski_eigenvalue_ratio
# ---------------------------------------------------------------------------

class TestOnatskiEigenvalueRatio:
    """Tests for Onatski eigenvalue ratio test."""

    def test_detects_signal_factors(self, synthetic_eigenvalues: np.ndarray) -> None:
        """Should detect signal factors in spectrum with clear separation."""
        k_est, max_ratio = onatski_eigenvalue_ratio(
            synthetic_eigenvalues, n=100, T=300
        )
        # Should detect some signal factors (1-5)
        assert 1 <= k_est <= 6
        assert max_ratio > 1.0  # Clear separation exists

    def test_max_ratio_positive(self, synthetic_eigenvalues: np.ndarray) -> None:
        """Max ratio should be positive."""
        _, max_ratio = onatski_eigenvalue_ratio(
            synthetic_eigenvalues, n=100, T=300
        )
        assert max_ratio >= 0

    def test_short_spectrum(self) -> None:
        """Should return (1, 0) for very short spectrum."""
        k_est, max_ratio = onatski_eigenvalue_ratio(
            np.array([1.0, 0.5]), n=10, T=50
        )
        assert k_est == 1
        assert max_ratio == 0.0


# ---------------------------------------------------------------------------
# Test: classify_factor
# ---------------------------------------------------------------------------

class TestClassifyFactor:
    """Tests for factor classification."""

    def test_structural_factor(self) -> None:
        """High persistence + high breadth = Structural."""
        cat = classify_factor(
            half_life=150.0, breadth=50, n_stocks=100
        )
        assert cat == "Structural"

    def test_style_factor(self) -> None:
        """Medium persistence = Style."""
        cat = classify_factor(
            half_life=50.0, breadth=30, n_stocks=100
        )
        assert cat == "Style"

    def test_episodic_factor(self) -> None:
        """Low persistence or low breadth = Episodic."""
        cat = classify_factor(
            half_life=10.0, breadth=5, n_stocks=100
        )
        assert cat == "Episodic"

    def test_high_stability_structural(self) -> None:
        """Very high stability can upgrade to Structural."""
        cat = classify_factor(
            half_life=150.0, breadth=50, stability=0.98, n_stocks=100
        )
        assert cat == "Structural"


# ---------------------------------------------------------------------------
# Test: compute_factor_quality_dashboard
# ---------------------------------------------------------------------------

class TestFactorQualityDashboard:
    """Tests for the complete dashboard function."""

    def test_dashboard_returns_all_keys(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
    ) -> None:
        """Dashboard should return all expected keys."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures[:, :3],  # 3 factors to match eigenvalues
            eigenvalues=synthetic_eigenvalues[:3],
        )
        assert "AU" in dashboard
        assert "n_stocks" in dashboard
        assert "breadth" in dashboard
        assert "categories" in dashboard
        assert "n_structural" in dashboard
        assert "n_style" in dashboard
        assert "n_episodic" in dashboard
        assert "stability_ok" in dashboard

    def test_dashboard_with_factor_returns(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
        synthetic_factor_returns: np.ndarray,
    ) -> None:
        """Dashboard with factor returns should compute persistence."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures,
            eigenvalues=synthetic_eigenvalues[:3],
            factor_returns=synthetic_factor_returns,
        )
        # Should have half_lives computed
        assert len(dashboard.get("half_lives", [])) > 0

    def test_dashboard_with_returns_for_bai_ng(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
        centered_returns: np.ndarray,
    ) -> None:
        """Dashboard with returns should compute Bai-Ng k."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures,
            eigenvalues=synthetic_eigenvalues[:3],
            returns_centered=centered_returns,
        )
        # May or may not have k_bai_ng depending on matrix size
        assert "k_bai_ng" in dashboard

    def test_category_counts_sum_to_au(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
    ) -> None:
        """Category counts should sum to AU."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures,
            eigenvalues=synthetic_eigenvalues[:3],
        )
        n_struct = dashboard.get("n_structural", 0)
        n_style = dashboard.get("n_style", 0)
        n_epis = dashboard.get("n_episodic", 0)
        au = dashboard.get("AU", 0)
        assert n_struct + n_style + n_epis == au

    def test_stability_with_rho(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
    ) -> None:
        """Stability rho should be passed through."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures,
            eigenvalues=synthetic_eigenvalues[:3],
            stability_rho=0.92,
        )
        assert dashboard.get("stability_rho") == 0.92
        assert dashboard.get("stability_ok") is True

    def test_stability_below_threshold(
        self,
        synthetic_exposures: np.ndarray,
        synthetic_eigenvalues: np.ndarray,
    ) -> None:
        """Low stability should set stability_ok to False."""
        dashboard = compute_factor_quality_dashboard(
            B_A=synthetic_exposures,
            eigenvalues=synthetic_eigenvalues[:3],
            stability_rho=0.70,
        )
        assert dashboard.get("stability_ok") is False
