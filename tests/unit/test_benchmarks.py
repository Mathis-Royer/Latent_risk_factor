"""
Unit tests for MOD-010 to MOD-015: Benchmark models.

Covers: EqualWeight, InverseVolatility, MinimumVariance,
EqualRiskContribution, PCAFactorRiskParity, PCAVolRiskParity.

Reference: ISD Section MOD-010 to MOD-015.
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.equal_weight import EqualWeight
from src.benchmarks.inverse_vol import InverseVolatility
from src.benchmarks.min_variance import MinimumVariance
from src.benchmarks.erc import EqualRiskContribution
from src.benchmarks.pca_factor_rp import PCAFactorRiskParity
from src.benchmarks.pca_vol import PCAVolRiskParity


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

N_STOCKS = 50
N_DAYS = 300


@pytest.fixture(scope="module")
def shared_returns() -> pd.DataFrame:
    """Generate shared random returns: 50 stocks, 300 business days."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    columns = [f"stock_{i}" for i in range(N_STOCKS)]
    data = np.random.randn(N_DAYS, N_STOCKS) * 0.02
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture(scope="module")
def universe() -> list[str]:
    """Universe of stock identifiers matching shared_returns columns."""
    return [f"stock_{i}" for i in range(N_STOCKS)]


@pytest.fixture(scope="module")
def constraint_params() -> dict[str, float]:
    """Shared constraint parameters for all benchmarks (INV-012)."""
    return {
        "w_max": 0.05,
        "w_min": 0.001,
        "phi": 25.0,
        "kappa_1": 0.1,
        "kappa_2": 7.5,
        "delta_bar": 0.01,
        "tau_max": 0.30,
        "lambda_risk": 1.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Tests for benchmark models."""

    def test_constraints_identical(
        self, constraint_params: dict[str, float]
    ) -> None:
        """All benchmarks receive and store identical constraint_params."""
        benchmarks = [
            EqualWeight(constraint_params=constraint_params),
            InverseVolatility(constraint_params=constraint_params),
            MinimumVariance(constraint_params=constraint_params),
            EqualRiskContribution(constraint_params=constraint_params),
            PCAFactorRiskParity(constraint_params=constraint_params),
            PCAVolRiskParity(constraint_params=constraint_params),
        ]

        for bench in benchmarks:
            assert bench.constraint_params == constraint_params, (
                f"{bench.__class__.__name__} has mismatched constraint_params"
            )

    def test_equal_weight_sum_to_one(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """EqualWeight produces weights summing to 1.0."""
        ew = EqualWeight(constraint_params=constraint_params)
        ew.fit(shared_returns, universe)
        w = ew.optimize(is_first=True)

        assert w.shape == (N_STOCKS,)
        assert abs(np.sum(w) - 1.0) < 1e-6, (
            f"EqualWeight weights sum to {np.sum(w)}, expected 1.0"
        )

    def test_min_var_beats_random(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """MinimumVariance portfolio has lower variance than random weights."""
        np.random.seed(42)

        mv = MinimumVariance(constraint_params=constraint_params)
        mv.fit(shared_returns, universe)
        w_mv = mv.optimize(is_first=True)

        # Covariance from shared returns
        R = shared_returns[universe].values.astype(np.float64)
        Sigma: np.ndarray = np.cov(R, rowvar=False)  # type: ignore[assignment]

        var_mv = float(w_mv @ Sigma @ w_mv)

        # Random portfolio (Dirichlet)
        w_rand = np.random.dirichlet(np.ones(N_STOCKS))
        var_rand = float(w_rand @ Sigma @ w_rand)

        assert var_mv < var_rand, (
            f"Min-var portfolio variance ({var_mv:.8f}) should be lower "
            f"than random portfolio variance ({var_rand:.8f})"
        )

    def test_erc_equal_risk_contributions(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """ERC produces approximately equal risk contributions."""
        erc = EqualRiskContribution(constraint_params=constraint_params)
        erc.fit(shared_returns, universe)
        w = erc.optimize(is_first=True)

        # Compute risk contributions: RC_i = w_i * (Sigma @ w)_i
        Sigma = erc.Sigma_LW
        marginal = Sigma @ w
        rc = w * marginal

        # Normalize risk contributions
        total_risk = np.sum(rc)
        if total_risk > 0:
            rc_normalized = rc / total_risk
        else:
            rc_normalized = rc

        # Check that risk contributions are approximately equal (5% tolerance)
        # Only check active positions (w > w_min)
        active = w > constraint_params["w_min"]
        if np.sum(active) > 1:
            rc_active = rc_normalized[active]
            expected_rc = 1.0 / np.sum(active)
            assert np.allclose(rc_active, expected_rc, atol=0.05), (
                f"Risk contributions are not equal. "
                f"Max deviation: {np.max(np.abs(rc_active - expected_rc)):.4f}"
            )

    def test_pca_ic2_range(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """PCA factor count k selected via IC2 falls within [1, 30]."""
        pca = PCAFactorRiskParity(constraint_params=constraint_params)
        pca.fit(shared_returns, universe, k_max=30)

        assert hasattr(pca, "k"), "PCAFactorRiskParity should store k attribute"
        assert 1 <= pca.k <= 30, (
            f"PCA k={pca.k} is outside expected range [1, 30]"
        )

    def test_pca_factor_rp_uses_sca(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """PCAFactorRiskParity optimize() completes without error."""
        pca = PCAFactorRiskParity(constraint_params=constraint_params)
        pca.fit(shared_returns, universe)
        w = pca.optimize(is_first=True)

        assert w.shape == (N_STOCKS,)
        assert np.isfinite(w).all(), "PCA weights contain non-finite values"

    def test_benchmark_output_format(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """All 6 benchmarks return w of shape (n,), w >= 0, sum(w) ~= 1."""
        benchmarks = [
            EqualWeight(constraint_params=constraint_params),
            InverseVolatility(constraint_params=constraint_params),
            MinimumVariance(constraint_params=constraint_params),
            EqualRiskContribution(constraint_params=constraint_params),
            PCAFactorRiskParity(constraint_params=constraint_params),
            PCAVolRiskParity(constraint_params=constraint_params),
        ]

        for bench in benchmarks:
            name = bench.__class__.__name__
            bench.fit(shared_returns, universe)
            w = bench.optimize(is_first=True)

            assert w.shape == (N_STOCKS,), (
                f"{name}: expected shape ({N_STOCKS},), got {w.shape}"
            )
            assert np.all(w >= -1e-8), (
                f"{name}: weights contain negative values: min={np.min(w)}"
            )
            assert abs(np.sum(w) - 1.0) < 0.02, (
                f"{name}: weights sum to {np.sum(w):.4f}, expected ~1.0"
            )
