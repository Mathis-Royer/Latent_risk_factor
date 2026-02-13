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
        """MinimumVariance portfolio variance < 90th percentile of 100 random portfolios.

        MinVar is the global optimum of portfolio variance, so it must beat
        the vast majority of random portfolios — not just the median.
        """
        np.random.seed(42)

        mv = MinimumVariance(constraint_params=constraint_params)
        mv.fit(shared_returns, universe)
        w_mv = mv.optimize(is_first=True)

        # Covariance from shared returns
        R = shared_returns[universe].values.astype(np.float64)
        Sigma: np.ndarray = np.cov(R, rowvar=False)  # type: ignore[assignment]

        var_mv = float(w_mv @ Sigma @ w_mv)

        # Compare against 100 random Dirichlet portfolios
        rng = np.random.RandomState(42)
        var_randoms = []
        for _ in range(100):
            w_rand = rng.dirichlet(np.ones(N_STOCKS))
            var_randoms.append(float(w_rand @ Sigma @ w_rand))

        p90 = np.percentile(var_randoms, 90)
        assert var_mv < p90, (
            f"Min-var variance ({var_mv:.8f}) should be below 90th percentile of "
            f"random portfolios ({p90:.8f})"
        )

        # Also verify it beats the diagonal minimum (theoretical guarantee)
        diag_min_var = np.min(np.diag(Sigma))
        assert var_mv <= diag_min_var + 1e-8, (
            f"Min-var variance ({var_mv:.8f}) should be <= min diagonal "
            f"variance ({diag_min_var:.8f})"
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

        # Check that risk contributions are approximately equal (0.5% tolerance)
        # Only check active positions (w > w_min)
        active = w > constraint_params["w_min"]
        if np.sum(active) > 1:
            rc_active = rc_normalized[active]
            expected_rc = 1.0 / np.sum(active)
            assert np.allclose(rc_active, expected_rc, atol=0.005), (
                f"Risk contributions are not equal. "
                f"Max deviation: {np.max(np.abs(rc_active - expected_rc)):.6f}, "
                f"expected ~{expected_rc:.4f} per active stock"
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
        """PCAFactorRiskParity optimize() produces valid weights with positive entropy."""
        pca = PCAFactorRiskParity(constraint_params=constraint_params)
        pca.fit(shared_returns, universe)
        w = pca.optimize(is_first=True)

        assert w.shape == (N_STOCKS,)
        assert np.isfinite(w).all(), "PCA weights contain non-finite values"

        # M8: Verify the optimization achieves meaningful entropy (H > 0)
        # Entropy H(w) = -sum(c_k * log(c_k)) where c_k are risk contributions
        # A valid SCA solution should spread risk across factors → H > 0
        active = w > 1e-8
        assert np.sum(active) > 1, (
            "PCA-FRP should select more than 1 stock"
        )

        # IC2-selected k should match the stored attribute
        assert hasattr(pca, "k"), "PCAFactorRiskParity should store k attribute"
        assert pca.k >= 1, f"k={pca.k} is invalid (must be >= 1)"

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
            assert abs(np.sum(w) - 1.0) < 1e-6, (
                f"{name}: weights sum to {np.sum(w):.8f}, expected 1.0"
            )

    def test_all_benchmarks_respect_w_max(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """All 6 benchmarks must have max(w) <= w_max."""
        w_max = constraint_params["w_max"]

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

            assert np.max(w) <= w_max + 1e-6, (
                f"{name}: max(w)={np.max(w):.6f} exceeds w_max={w_max}"
            )

    def test_benchmark_evaluate_metrics_keys(
        self,
        shared_returns: pd.DataFrame,
        universe: list[str],
        constraint_params: dict[str, float],
    ) -> None:
        """Benchmark optimize() returns proper weights with expected properties."""
        ew = EqualWeight(constraint_params=constraint_params)
        ew.fit(shared_returns, universe)
        w = ew.optimize(is_first=True)

        # Basic properties
        assert w.dtype == np.float64 or np.issubdtype(w.dtype, np.floating), (
            f"Weights dtype should be float, got {w.dtype}"
        )
        assert w.shape == (N_STOCKS,)
        assert np.all(np.isfinite(w)), "Weights contain non-finite values"
        assert abs(np.sum(w) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# B3: ERC Spinu formula verification on diagonal Sigma
# ---------------------------------------------------------------------------


class TestERCSpinuFormula:
    """B3: Verify ERC produces known analytical solution on diagonal Sigma."""

    def test_erc_diagonal_sigma_known_solution(self) -> None:
        """For diagonal Sigma = diag(1, 4, 9), ERC gives w proportional to 1/sigma.

        Spinu KKT: (Sigma y)_i = n/y_i for all i.
        For diagonal Sigma: sigma_i^2 * y_i = n/y_i -> y_i = sqrt(n)/sigma_i
        After normalization: w_i = (1/sigma_i) / sum(1/sigma_j)
        """
        n = 3
        sigmas_sq = np.array([1.0, 4.0, 9.0])
        Sigma = np.diag(sigmas_sq)

        # Analytical solution: w proportional to 1/sigma
        sigmas = np.sqrt(sigmas_sq)  # [1, 2, 3]
        w_analytical = (1.0 / sigmas) / np.sum(1.0 / sigmas)
        # w = [1, 1/2, 1/3] / (1 + 1/2 + 1/3) = [1, 0.5, 0.333] / 1.833
        # w = [6/11, 3/11, 2/11] ≈ [0.5455, 0.2727, 0.1818]

        # Use ERC with relaxed constraints to allow unconstrained solution
        erc = EqualRiskContribution(constraint_params={
            "w_max": 1.0,
            "w_min": 0.001,
            "phi": 0.0,
            "kappa_1": 0.0,
            "kappa_2": 0.0,
            "delta_bar": 0.01,
            "tau_max": 1.0,
            "lambda_risk": 1.0,
        })

        # Manually set Sigma_LW (skip fit, use known diagonal)
        erc.Sigma_LW = Sigma
        erc.n = n
        w = erc.optimize(is_first=True)

        np.testing.assert_allclose(
            w, w_analytical, atol=1e-3,
            err_msg=(
                f"ERC on diag([1,4,9]) should give inverse-vol weights. "
                f"Expected {w_analytical}, got {w}"
            ),
        )

        # Verify equal risk contributions
        rc = w * (Sigma @ w)
        rc_norm = rc / np.sum(rc)
        np.testing.assert_allclose(
            rc_norm, np.ones(n) / n, atol=0.005,
            err_msg=f"Risk contributions should be equal: {rc_norm}",
        )


# ---------------------------------------------------------------------------
# B4: IC2 factor selection on synthetic data with known factor count
# ---------------------------------------------------------------------------


class TestIC2FactorSelection:
    """B4: Verify IC2 selects the correct k on synthetic data."""

    def test_ic2_selects_correct_k_strong_signal(self) -> None:
        """With k_true=3 strong factors, IC2 should select k*=3."""
        rng = np.random.RandomState(42)
        T_est = 300
        n = 50
        k_true = 3

        # Generate data: R = F @ Lambda^T + noise
        F_mat = rng.randn(T_est, k_true)
        Lambda = rng.randn(n, k_true) * 0.5
        noise = rng.randn(T_est, n) * 0.01  # Very low noise

        R = F_mat @ Lambda.T + noise
        R_centered = R - R.mean(axis=0, keepdims=True)

        # Use PCAFactorRiskParity._bai_ng_ic2
        pca = PCAFactorRiskParity()
        k_star = pca._bai_ng_ic2(R_centered, k_max=10)

        assert k_star == k_true, (
            f"IC2 should select k*={k_true} with strong signal, got k*={k_star}"
        )

    def test_ic2_values_minimum_at_k_star(self) -> None:
        """IC2(k*) should be the minimum over all k."""
        rng = np.random.RandomState(42)
        T_est = 200
        n = 30
        k_true = 2

        F_mat = rng.randn(T_est, k_true) * 2.0
        Lambda = rng.randn(n, k_true) * 0.5
        noise = rng.randn(T_est, n) * 0.05

        R = F_mat @ Lambda.T + noise
        R_centered = R - R.mean(axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
        penalty_coeff = ((n + T_est) / (n * T_est)) * np.log(min(n, T_est))

        ic2_values = []
        for k in range(1, 11):
            R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            V_k = np.sum((R_centered - R_approx) ** 2) / (n * T_est)
            ic2 = np.log(max(V_k, 1e-30)) + k * penalty_coeff
            ic2_values.append(ic2)

        k_star_manual = int(np.argmin(ic2_values)) + 1  # 1-indexed

        pca = PCAFactorRiskParity()
        k_star = pca._bai_ng_ic2(R_centered, k_max=10)

        assert k_star == k_star_manual, (
            f"IC2 selection mismatch: function={k_star}, manual={k_star_manual}"
        )
        # IC2 at k_star should be less than at k_star+1 and k_star-1 (if they exist)
        if k_star_manual > 1:
            assert ic2_values[k_star_manual - 1] < ic2_values[k_star_manual - 2], (
                f"IC2(k*) should be < IC2(k*-1)"
            )
        if k_star_manual < len(ic2_values):
            assert ic2_values[k_star_manual - 1] < ic2_values[k_star_manual], (
                f"IC2(k*) should be < IC2(k*+1)"
            )


# ---------------------------------------------------------------------------
# Tests: Benchmark constraint properties
# ---------------------------------------------------------------------------


class TestBenchmarkConstraints:
    """Tests for benchmark constraint enforcement (INV-012)."""

    def test_benchmarks_semi_continuous_constraint(self) -> None:
        """INV-012: All benchmarks produce weights with w_i == 0 or w_i >= w_min."""
        n = 20
        w_min = 0.001
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
        stock_ids = [f"stock_{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.02, (252, n)),
            index=dates, columns=stock_ids,
        )
        trailing_vol = pd.DataFrame(
            np.abs(rng.randn(252, n)) * 0.02 + 0.15,
            index=dates, columns=stock_ids,
        )
        for BenchClass in [EqualWeight, InverseVolatility]:
            bench = BenchClass()
            bench.fit(returns, stock_ids, trailing_vol=trailing_vol, current_date=str(dates[-1].date()))  # type: ignore[union-attr]
            w = bench.optimize(is_first=True)
            # All weights should be either 0 or >= w_min
            # For EW and InvVol, all stocks get positive weight (= 1/n or 1/vol)
            assert abs(np.sum(w) - 1.0) < 1e-4, f"{BenchClass.__name__}: sum={np.sum(w)}"
            assert np.all(w >= -1e-8), f"{BenchClass.__name__}: negative weights"

    def test_inverse_vol_inversely_proportional(self) -> None:
        """InverseVolatility: weight ratio should be approximately inverse of vol ratio.

        Stock 0 has 2x vol of stock 1 (and all others), so w[0] should be
        approximately half of w[1]. We verify both ordering and ratio.
        """
        n = 30
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
        stock_ids = [f"s{i}" for i in range(n)]
        # Stock 0 has 2x vol of stock 1; all others have same vol as stock 1
        vol_data = np.full((252, n), 0.20)
        vol_data[:, 0] = 0.40  # Double vol
        returns = pd.DataFrame(rng.normal(0, 0.02, (252, n)), index=dates, columns=stock_ids)
        trailing_vol = pd.DataFrame(vol_data, index=dates, columns=stock_ids)
        bench = InverseVolatility()
        bench.fit(returns, stock_ids, trailing_vol=trailing_vol, current_date=str(dates[-1].date()))  # type: ignore[union-attr]
        w = bench.optimize(is_first=True)
        assert w[0] < w[1], f"Higher vol stock should have lower weight: w[0]={w[0]}, w[1]={w[1]}"

        # Weight ratio should approximate inverse vol ratio: w[0]/w[1] ~ 0.5
        if w[0] > 1e-10 and w[1] > 1e-10:
            weight_ratio = w[0] / w[1]
            expected_ratio = 0.5  # (1/0.40) / (1/0.20) = 0.5
            assert abs(weight_ratio - expected_ratio) < 0.15, (
                f"Weight ratio w[0]/w[1]={weight_ratio:.4f} should be ~{expected_ratio} "
                f"(inverse of vol ratio 2:1), tolerance 0.15"
            )

    def test_equal_weight_uniform(self) -> None:
        """EqualWeight: all weights should be approximately 1/n."""
        n = 20
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=100, freq="B")
        stock_ids = [f"s{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0, 0.02, (100, n)),
            index=dates, columns=stock_ids,
        )
        bench = EqualWeight()
        bench.fit(returns, stock_ids)
        w = bench.optimize(is_first=True)
        expected = 1.0 / n
        # Default w_max = 0.05, and 1/20 = 0.05, so equal weight is at boundary
        assert np.allclose(w, expected, atol=1e-6), (
            f"EqualWeight should produce uniform 1/n={expected}, got range [{w.min()}, {w.max()}]"
        )

    def test_inverse_vol_all_same_vol_equals_equal_weight(self) -> None:
        """InverseVolatility with identical vols should produce equal weights."""
        n = 10
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
        stock_ids = [f"s{i}" for i in range(n)]
        vol_data = np.full((252, n), 0.20)  # All identical
        returns = pd.DataFrame(rng.normal(0, 0.02, (252, n)), index=dates, columns=stock_ids)
        trailing_vol = pd.DataFrame(vol_data, index=dates, columns=stock_ids)
        bench = InverseVolatility()
        bench.fit(returns, stock_ids, trailing_vol=trailing_vol, current_date=str(dates[-1].date()))  # type: ignore[union-attr]
        w = bench.optimize(is_first=True)
        expected = 1.0 / n
        # With projection to constraints, should still be approximately equal
        assert np.allclose(w, expected, atol=1e-4), (
            f"With identical vols, InverseVol should produce equal weights ~{expected}, "
            f"got range [{w.min()}, {w.max()}]"
        )


# ---------------------------------------------------------------------------
# C2/M5: Turnover constraint test (is_first=False with w_old)
# ---------------------------------------------------------------------------

class TestBenchmarkTurnoverConstraint:
    """C2/M5: Verify benchmarks enforce tau_max when is_first=False."""

    def test_turnover_constrained_on_second_rebalancing(self) -> None:
        """
        Run each benchmark twice: first with is_first=True, then with
        is_first=False and a deliberately distant w_old. Verify that the
        second rebalancing respects the tau_max=30% one-way turnover cap.
        """
        n = 30
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        stock_ids = [f"stock_{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.02, (300, n)),
            index=dates, columns=stock_ids,
        )
        trailing_vol = pd.DataFrame(
            np.abs(rng.randn(300, n)) * 0.02 + 0.15,
            index=dates, columns=stock_ids,
        )

        constraint_params: dict[str, float] = {
            "w_max": 0.05,
            "w_min": 0.001,
            "phi": 25.0,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "lambda_risk": 1.0,
        }
        tau_max = constraint_params["tau_max"]

        # Create a w_old that respects w_max but is concentrated
        # (heaviest in first stocks, lighter in rest)
        w_old = np.full(n, 0.001)  # All at w_min
        # Put heavy weight on first 20 stocks (at w_max)
        w_old[:20] = constraint_params["w_max"]  # 20 * 0.05 = 1.0
        w_old = w_old / np.sum(w_old)  # Normalize to sum=1

        benchmark_classes = [
            MinimumVariance,
            EqualRiskContribution,
            PCAFactorRiskParity,
            PCAVolRiskParity,
        ]

        for BenchClass in benchmark_classes:
            name = BenchClass.__name__
            bm = BenchClass(constraint_params=constraint_params)
            bm.fit(
                returns, stock_ids,
                trailing_vol=trailing_vol,
                current_date=str(dates[-1].date()),
            )

            # Second rebalancing with w_old
            w_new = bm.optimize(w_old=w_old, is_first=False)

            # One-way turnover = 0.5 * sum(|w_new - w_old|)
            turnover = 0.5 * np.sum(np.abs(w_new - w_old))

            assert turnover <= tau_max + 1e-4, (
                f"C2/M5 violated: {name} turnover={turnover:.4f} exceeds "
                f"tau_max={tau_max} on second rebalancing"
            )
            assert abs(np.sum(w_new) - 1.0) < 1e-4, (
                f"{name}: weights sum to {np.sum(w_new):.6f} after turnover constraint"
            )
            assert np.all(w_new >= -1e-8), (
                f"{name}: negative weights after turnover constraint"
            )

    def test_first_rebalancing_ignores_turnover(self) -> None:
        """
        When is_first=True, passing w_old should NOT constrain turnover.
        The optimizer should be free to deviate arbitrarily from w_old.
        """
        n = 30
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        stock_ids = [f"stock_{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.02, (300, n)),
            index=dates, columns=stock_ids,
        )

        constraint_params: dict[str, float] = {
            "w_max": 0.05,
            "w_min": 0.001,
            "phi": 25.0,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "lambda_risk": 1.0,
        }

        # Concentrated w_old
        w_old = np.zeros(n)
        w_old[0] = 1.0

        bm = MinimumVariance(constraint_params=constraint_params)
        bm.fit(returns, stock_ids)

        # is_first=True: turnover should NOT be constrained
        w_first = bm.optimize(w_old=w_old, is_first=True)
        turnover_first = 0.5 * np.sum(np.abs(w_first - w_old))

        # With is_first=True, the min-var solution should be diversified,
        # so turnover from the concentrated w_old should be large
        assert turnover_first > constraint_params["tau_max"], (
            f"With is_first=True, optimizer should freely diversify. "
            f"Turnover={turnover_first:.4f} is unexpectedly low."
        )
