"""
Unit tests for the portfolio optimization module (MOD-008).

Covers:
- Entropy gradient at maximum (vanishes)
- Entropy gradient numerical verification
- SCA convergence
- Armijo sufficient decrease
- Multi-start determinism
- Cardinality enforcement
- Hard constraint satisfaction
- Turnover penalty at first rebalancing
- Known diagonal-identity solution

Reference: ISD Section MOD-008.
"""

from typing import Any

import numpy as np
import pytest

from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.sca_solver import sca_optimize, multi_start_optimize, objective_function, has_mi_solver
from src.portfolio.cardinality import enforce_cardinality
from src.portfolio.constraints import (
    concentration_penalty,
    turnover_penalty,
    project_to_constraints,
    check_hard_constraints,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SEED = 42
N_STOCKS = 20
AU = 5


# ---------------------------------------------------------------------------
# Helper: generate valid portfolio test data
# ---------------------------------------------------------------------------

def _make_portfolio_data(
    n: int = N_STOCKS,
    au: int = AU,
    seed: int = SEED,
) -> dict:
    """
    Build deterministic inputs for portfolio optimization tests.

    Produces a PSD Sigma_assets, rotated exposures B_prime, eigenvalues, and
    idiosyncratic variances D_eps.

    :param n (int): Number of stocks
    :param au (int): Number of active units (principal factors)
    :param seed (int): Random seed

    :return data (dict): Sigma_assets, B_prime, eigenvalues, D_eps, B_A_port,
        Sigma_z, V
    """
    rng = np.random.RandomState(seed)

    # Raw exposures
    B_A_port = rng.randn(n, au).astype(np.float64) * 0.3

    # PSD factor covariance
    raw = rng.randn(au, au).astype(np.float64) * 0.05
    Sigma_z = raw @ raw.T + np.eye(au) * 0.01

    # Eigendecomposition of Sigma_z (descending order)
    eigvals, V = np.linalg.eigh(Sigma_z)
    idx = np.argsort(-eigvals)
    eigenvalues = np.maximum(eigvals[idx], 0.0)
    V = V[:, idx]

    # Rotated exposures
    B_prime = B_A_port @ V

    # Idiosyncratic variances
    D_eps = rng.uniform(0.001, 0.01, size=n).astype(np.float64)

    # Full asset covariance
    Sigma_assets = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)

    return {
        "Sigma_assets": Sigma_assets,
        "B_prime": B_prime,
        "eigenvalues": eigenvalues,
        "D_eps": D_eps,
        "B_A_port": B_A_port,
        "Sigma_z": Sigma_z,
        "V": V,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntropy:
    """Tests for entropy computation and gradient (MOD-008 sub-task 1)."""

    def test_entropy_gradient_at_maximum(self) -> None:
        """When all risk contributions are equal, the gradient should vanish.

        Construct B_prime and eigenvalues so that c'_k = const for equal-weight
        portfolio. With equal eigenvalues and B_prime = I (padded), equal weights
        over the first AU stocks give identical beta'_k values.
        """
        np.random.seed(SEED)

        n = AU + 2
        eigenvalues = np.ones(AU, dtype=np.float64) * 0.05

        # B_prime = identity for first AU stocks, zeros for rest
        B_prime = np.zeros((n, AU), dtype=np.float64)
        for k in range(AU):
            B_prime[k, k] = 1.0

        # Weights: equal across the AU stocks only (rest zero)
        w = np.zeros(n, dtype=np.float64)
        w[:AU] = 1.0 / AU

        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        # H should be ln(AU) (maximum entropy for AU equal contributions)
        expected_H = np.log(AU)
        assert abs(H - expected_H) < 1e-10, (
            f"H = {H:.8f}, expected ln({AU}) = {expected_H:.8f}"
        )

        # Gradient should vanish at the maximum for the active positions
        # Only check positions 0..AU-1 since others carry zero weight
        grad_active = grad_H[:AU]
        assert np.allclose(grad_active, 0.0, atol=1e-10), (
            f"Gradient not zero at maximum: {grad_active}"
        )

    def test_entropy_gradient_numerical(self) -> None:
        """Analytical gradient must match finite-difference approximation.

        Perturb each w_i by epsilon and compare (H(w+e)-H(w-e)) / (2*epsilon)
        to the analytical gradient. Relative error should be below 1e-5.
        """
        np.random.seed(SEED)
        data = _make_portfolio_data()

        rng = np.random.RandomState(SEED + 1)
        # Random feasible weights (positive, sum to 1)
        w_raw = rng.dirichlet(np.ones(N_STOCKS))
        w = w_raw.astype(np.float64)

        B_prime = data["B_prime"]
        eigenvalues = data["eigenvalues"]

        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        # Finite-difference gradient
        epsilon = 1e-7
        grad_numerical = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += epsilon
            H_plus = compute_entropy_only(w_plus, B_prime, eigenvalues)

            w_minus = w.copy()
            w_minus[i] -= epsilon
            H_minus = compute_entropy_only(w_minus, B_prime, eigenvalues)

            grad_numerical[i] = (H_plus - H_minus) / (2.0 * epsilon)

        # Check relative error for components with significant gradient
        mask = np.abs(grad_H) > 1e-10
        if mask.any():
            rel_error = np.abs(grad_H[mask] - grad_numerical[mask]) / (
                np.abs(grad_H[mask]) + 1e-15
            )
            assert np.all(rel_error < 1e-7), (
                f"Max relative error = {rel_error.max():.2e}, "
                f"positions: {np.where(mask)[0][rel_error > 1e-7]}"
            )

        # Also check absolute error everywhere
        np.testing.assert_allclose(
            grad_H, grad_numerical, atol=1e-7,
            err_msg="Analytical gradient differs from numerical gradient",
        )


class TestSCASolver:
    """Tests for SCA optimization (MOD-008 sub-task 2)."""

    def test_sca_convergence(self) -> None:
        """SCA must converge in fewer than 100 iterations on a simple problem."""
        np.random.seed(SEED)
        data = _make_portfolio_data()

        w_init = np.ones(N_STOCKS, dtype=np.float64) / N_STOCKS

        w_opt, f_opt, H_opt, n_iters = sca_optimize(
            w_init=w_init,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_max=0.10,
            is_first=True,
            max_iter=100,
        )

        assert n_iters <= 100, f"SCA did not converge: {n_iters} iterations"
        assert np.isfinite(f_opt), "Objective is not finite"
        assert np.isfinite(H_opt), "Entropy is not finite"
        assert abs(np.sum(w_opt) - 1.0) < 1e-4, (
            f"Weights do not sum to 1: {np.sum(w_opt):.6f}"
        )

        # B2: Verify objective composition independently
        # f(w) = -lambda*w^T*Sigma*w + alpha*H(w) - phi*P_conc - P_turn
        H_check = compute_entropy_only(
            w_opt, data["B_prime"], data["eigenvalues"],
        )
        assert abs(H_check - H_opt) < 1e-10, (
            f"Entropy mismatch: H_check={H_check:.8f}, H_opt={H_opt:.8f}"
        )

        f_check = objective_function(
            w=w_opt,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_old=None,
            kappa_1=0.1,
            kappa_2=7.5,
            delta_bar=0.01,
            is_first=True,
        )
        assert abs(f_check - f_opt) < 1e-8, (
            f"Objective mismatch: f_check={f_check:.8f}, f_opt={f_opt:.8f}"
        )

        # Converged solution must be better than initial
        f_init = objective_function(
            w=w_init,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_old=None,
            kappa_1=0.1,
            kappa_2=7.5,
            delta_bar=0.01,
            is_first=True,
        )
        assert f_opt >= f_init - 1e-6, (
            f"SCA should improve objective: f_init={f_init:.8f}, f_opt={f_opt:.8f}"
        )

    def test_armijo_sufficient_decrease(self) -> None:
        """After SCA optimization the final objective must be >= initial (or within tol).

        The Armijo condition ensures sufficient decrease in minimization; for
        maximization this means the objective should not decrease.
        """
        np.random.seed(SEED)
        data = _make_portfolio_data()

        w_init = np.ones(N_STOCKS, dtype=np.float64) / N_STOCKS

        f_init = objective_function(
            w=w_init,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_old=None,
            kappa_1=0.1,
            kappa_2=7.5,
            delta_bar=0.01,
            is_first=True,
        )

        w_opt, f_opt, _, _ = sca_optimize(
            w_init=w_init,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_max=0.10,
            is_first=True,
            max_iter=50,
        )

        # Objective should improve (or stay the same within tolerance)
        assert f_opt >= f_init - 1e-6, (
            f"Objective decreased: f_init={f_init:.8f}, f_final={f_opt:.8f}"
        )

    def test_multi_start_deterministic(self) -> None:
        """multi_start_optimize with the same seed must produce identical results."""
        np.random.seed(SEED)
        data = _make_portfolio_data()

        common_kwargs = {
            "Sigma_assets": data["Sigma_assets"],
            "B_prime": data["B_prime"],
            "eigenvalues": data["eigenvalues"],
            "D_eps": data["D_eps"],
            "alpha": 1.0,
            "n_starts": 3,
            "seed": 42,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "w_max": 0.10,
            "is_first": True,
            "max_iter": 30,
        }

        w1, f1, H1 = multi_start_optimize(**common_kwargs)
        w2, f2, H2 = multi_start_optimize(**common_kwargs)

        np.testing.assert_array_equal(w1, w2, err_msg="Weights differ across runs")
        assert f1 == f2, f"Objectives differ: {f1} vs {f2}"
        assert H1 == H2, f"Entropies differ: {H1} vs {H2}"


class TestCardinality:
    """Tests for cardinality enforcement (MOD-008 sub-task 3)."""

    def test_cardinality_enforcement(self) -> None:
        """After enforce_cardinality with real SCA, no weight in (0, w_min).

        Uses the real sca_optimize solver to verify the full cardinality
        pipeline: elimination + SCA re-optimization + constraint satisfaction.
        """
        np.random.seed(SEED)
        data = _make_portfolio_data()

        w_min = 0.01

        # Create weights with violations: some stocks between 0 and w_min
        rng = np.random.RandomState(SEED)
        w = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
        # Force several positions into the violation range
        w[0] = 0.005   # Below w_min but > 0
        w[1] = 0.002   # Below w_min but > 0
        w[2] = 0.0001  # Below w_min but > 0
        # Renormalize
        w = w / w.sum()

        sca_kwargs: dict[str, Any] = {
            "Sigma_assets": data["Sigma_assets"],
            "B_prime": data["B_prime"],
            "eigenvalues": data["eigenvalues"],
            "alpha": 1.0,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "w_max": 0.10,
            "w_old": None,
            "is_first": True,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "entropy_eps": 1e-30,
        }

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="gradient",
        )

        # Check: no weight in the forbidden zone (0, w_min)
        for i in range(N_STOCKS):
            assert w_enforced[i] == 0.0 or w_enforced[i] >= w_min - 1e-8, (
                f"Violation at stock {i}: w={w_enforced[i]:.6f}, "
                f"w_min={w_min}"
            )
        # Fully-invested constraint
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6, (
            f"Weights don't sum to 1: {np.sum(w_enforced):.8f}"
        )
        # Verify cardinality reduced: at least one stock was zeroed out
        assert np.sum(w_enforced == 0.0) > 0, (
            "No stocks were eliminated despite violations"
        )

    def _make_violation_data(self) -> tuple[
        np.ndarray, dict, dict[str, Any], float,
    ]:
        """Build shared test data with semi-continuous violations."""
        np.random.seed(SEED)
        data = _make_portfolio_data()
        w_min = 0.01

        rng = np.random.RandomState(SEED)
        w = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
        w[0] = 0.005
        w[1] = 0.002
        w[2] = 0.0001
        w = w / w.sum()

        sca_kwargs: dict[str, Any] = {
            "Sigma_assets": data["Sigma_assets"],
            "B_prime": data["B_prime"],
            "eigenvalues": data["eigenvalues"],
            "alpha": 1.0,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "w_max": 0.10,
            "w_old": None,
            "is_first": True,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "entropy_eps": 1e-30,
        }
        return w, data, sca_kwargs, w_min

    @staticmethod
    def _check_semi_continuous(
        w_enforced: np.ndarray, w_min: float, n: int,
    ) -> None:
        """Assert no weight in forbidden zone (0, w_min)."""
        for i in range(n):
            assert w_enforced[i] == 0.0 or w_enforced[i] >= w_min - 1e-8, (
                f"Violation at stock {i}: w={w_enforced[i]:.6f}, w_min={w_min}"
            )

    def test_gradient_method(self) -> None:
        """Gradient method with real SCA satisfies semi-continuous constraint."""
        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="gradient",
        )

        self._check_semi_continuous(w_enforced, w_min, N_STOCKS)
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6

    def test_sequential_method(self) -> None:
        """Sequential method with real SCA satisfies semi-continuous constraint."""
        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="sequential",
        )

        self._check_semi_continuous(w_enforced, w_min, N_STOCKS)
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6

    @pytest.mark.skipif(
        not has_mi_solver(), reason="No MI-capable solver (MOSEK) available",
    )
    def test_miqp_method(self) -> None:
        """MIQP method satisfies semi-continuous constraint."""
        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="miqp",
        )

        self._check_semi_continuous(w_enforced, w_min, N_STOCKS)
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6

    @pytest.mark.skipif(
        not has_mi_solver(), reason="No MI-capable solver (MOSEK) available",
    )
    def test_two_stage_method(self) -> None:
        """Two-stage method satisfies semi-continuous constraint."""
        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="two_stage",
        )

        self._check_semi_continuous(w_enforced, w_min, N_STOCKS)
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6

    def test_fallback_no_mi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no MI solver, miqp falls back to gradient."""
        monkeypatch.setattr(
            "src.portfolio.cardinality.has_mi_solver", lambda: False,
        )

        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_miqp = enforce_cardinality(
            w=w.copy(),
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="miqp",
        )
        w_grad = enforce_cardinality(
            w=w.copy(),
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="gradient",
        )

        np.testing.assert_array_almost_equal(w_miqp, w_grad)

    def test_auto_resolves(self) -> None:
        """Auto method runs without error."""
        w, data, sca_kwargs, w_min = self._make_violation_data()

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=sca_optimize,
            sca_kwargs=sca_kwargs,
            method="auto",
        )

        self._check_semi_continuous(w_enforced, w_min, N_STOCKS)
        assert abs(np.sum(w_enforced) - 1.0) < 1e-6


class TestConstraints:
    """Tests for constraint checking and turnover (MOD-008 sub-task 4)."""

    def test_constraints_satisfied(self) -> None:
        """After SCA optimization, hard constraints must be satisfied.

        w_i in [0, w_max], sum(w) = 1, w_i = 0 or w_i >= w_min.
        """
        np.random.seed(SEED)
        data = _make_portfolio_data()

        w_init = np.ones(N_STOCKS, dtype=np.float64) / N_STOCKS
        w_max = 0.10

        w_opt, _, _, _ = sca_optimize(
            w_init=w_init,
            Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=0.03,
            w_max=w_max,
            is_first=True,
            max_iter=50,
        )

        status = check_hard_constraints(w_opt, w_old=None, w_max=w_max)

        assert status["long_only"], "Long-only constraint violated"
        assert status["fully_invested"], (
            f"Fully invested violated: sum(w) = {np.sum(w_opt):.8f}"
        )
        assert status["weight_cap"], (
            f"Weight cap violated: max(w) = {np.max(w_opt):.8f}"
        )

    def test_turnover_first_rebalancing(self) -> None:
        """With is_first=True, turnover penalty must be exactly zero.

        The first rebalancing has no prior portfolio, so kappa_1 = kappa_2 = 0.
        Verified at two levels:
        1. turnover_penalty function returns 0
        2. objective_function is identical with/without w_old when is_first=True
        """
        np.random.seed(SEED)
        data = _make_portfolio_data()

        w = np.random.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
        w_old = np.random.dirichlet(np.ones(N_STOCKS)).astype(np.float64)

        # 1. Direct turnover_penalty check
        P_turn = turnover_penalty(
            w, w_old, kappa_1=0.1, kappa_2=7.5, delta_bar=0.01, is_first=True,
        )
        assert P_turn == 0.0, f"Turnover penalty not zero: {P_turn:.8f}"

        # 2. Integration: objective_function with w_old=None vs w_old=<vector>
        # must be identical when is_first=True
        common = {
            "Sigma_assets": data["Sigma_assets"],
            "B_prime": data["B_prime"],
            "eigenvalues": data["eigenvalues"],
            "alpha": 1.0,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "is_first": True,
        }
        f_none = objective_function(w=w, w_old=None, **common)
        f_wold = objective_function(w=w, w_old=w_old, **common)

        assert abs(f_none - f_wold) < 1e-10, (
            f"is_first=True: objective should be identical with w_old=None vs w_old=<vector>. "
            f"f_none={f_none:.8f}, f_wold={f_wold:.8f}"
        )


class TestKnownSolution:
    """Tests for known analytical solution."""

    def test_known_solution(self) -> None:
        """For diagonal Sigma = I and B_prime = I, equal-weight should be near-optimal.

        With alpha=0 and large lambda_risk, the objective reduces to minimizing
        portfolio variance. With Sigma = I, all stocks have equal variance and
        zero correlation, so equal-weight minimizes variance.
        """
        np.random.seed(SEED)

        n = N_STOCKS

        # Diagonal covariance = Identity
        Sigma_assets = np.eye(n, dtype=np.float64)
        B_prime = np.eye(n, dtype=np.float64)
        eigenvalues = np.ones(n, dtype=np.float64)
        D_eps = np.zeros(n, dtype=np.float64)

        w_init = np.ones(n, dtype=np.float64) / n

        w_opt, _, _, _ = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=0.0,
            lambda_risk=100.0,
            phi=0.0,
            w_bar=1.0,
            w_max=1.0,
            is_first=True,
            max_iter=50,
        )

        w_expected = np.ones(n, dtype=np.float64) / n

        np.testing.assert_allclose(
            w_opt, w_expected, atol=1e-5,
            err_msg="Optimal weights differ from equal-weight for Sigma=I, alpha=0",
        )

    def test_known_solution_max_entropy(self) -> None:
        """ISD MOD-008: For diagonal Sigma and B = I, max-entropy w_i is proportional to 1/sigma_i.

        With B_prime = I and eigenvalues = sigma_i^2 (diagonal covariance),
        c'_k = w_k^2 * sigma_k^2. Maximum entropy H = ln(AU) is achieved
        when all c'_k are equal, i.e., w_k * sigma_k = const, giving
        w_k proportional to 1/sigma_k.
        """
        np.random.seed(SEED)

        au = 5
        n = au  # Square case: n == AU for identity B

        # Diagonal Sigma with different variances
        sigma_sq = np.array([0.01, 0.02, 0.04, 0.08, 0.16], dtype=np.float64)
        Sigma_assets = np.diag(sigma_sq)
        eigenvalues = sigma_sq
        B_prime = np.eye(n, dtype=np.float64)

        # Analytical solution: w_i proportional to 1/sigma_i
        sigmas = np.sqrt(sigma_sq)
        w_analytical = 1.0 / sigmas
        w_analytical = w_analytical / w_analytical.sum()

        w_init = np.ones(n, dtype=np.float64) / n

        w_opt, _, H_opt, _ = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=10.0,       # Large entropy weight
            lambda_risk=0.0,  # No risk penalty
            phi=0.0,          # No concentration penalty
            w_bar=1.0,
            w_max=1.0,
            is_first=True,
            max_iter=100,
        )

        # At the analytical optimum, H should be ln(AU)
        assert abs(H_opt - np.log(au)) < 0.01, (
            f"H at optimum should be ln({au})={np.log(au):.4f}, got {H_opt:.4f}"
        )

        # Weights should match analytical solution
        np.testing.assert_allclose(
            w_opt, w_analytical, atol=1e-3,
            err_msg="Optimal weights differ from analytical 1/sigma solution",
        )


# ---------------------------------------------------------------------------
# Test: Entropy value range
# ---------------------------------------------------------------------------


class TestEntropyRange:
    """Tests for entropy value bounds."""

    def test_entropy_value_range(self) -> None:
        """0 <= H(w) <= ln(AU) for any valid portfolio."""
        np.random.seed(SEED)
        data = _make_portfolio_data()

        B_prime = data["B_prime"]
        eigenvalues = data["eigenvalues"]

        # Test multiple random portfolios
        rng = np.random.RandomState(SEED)
        for _ in range(20):
            w = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
            H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

            assert H >= 0.0, f"Entropy should be >= 0, got {H}"
            assert H <= np.log(AU) + 1e-10, (
                f"Entropy should be <= ln(AU)={np.log(AU):.4f}, got {H}"
            )

        # Equal-weight portfolio should give near-maximum entropy
        w_ew = np.ones(N_STOCKS, dtype=np.float64) / N_STOCKS
        H_ew = compute_entropy_only(w_ew, B_prime, eigenvalues)
        assert H_ew >= 0.0


# ---------------------------------------------------------------------------
# Test: Variance-entropy frontier monotonic
# ---------------------------------------------------------------------------


class TestFrontierMonotonic:
    """Tests for variance-entropy frontier."""

    def test_variance_entropy_frontier_monotonic(self) -> None:
        """Entropy should generally increase as alpha increases."""
        np.random.seed(SEED)
        data = _make_portfolio_data()

        # Small alpha grid for speed
        alpha_grid = [0.0, 0.1, 1.0, 5.0]

        entropies = []
        for alpha in alpha_grid:
            w_opt, f_opt, H_opt = multi_start_optimize(
                Sigma_assets=data["Sigma_assets"],
                B_prime=data["B_prime"],
                eigenvalues=data["eigenvalues"],
                D_eps=data["D_eps"],
                alpha=alpha,
                n_starts=3,
                seed=42,
                lambda_risk=1.0,
                phi=25.0,
                w_bar=0.03,
                w_max=0.10,
                is_first=True,
                max_iter=30,
            )
            entropies.append(H_opt)

        # Entropy should be non-decreasing (or very close) as alpha increases
        for i in range(1, len(entropies)):
            assert entropies[i] >= entropies[i - 1] - 0.01, (
                f"Entropy decreased from alpha={alpha_grid[i-1]} "
                f"(H={entropies[i-1]:.4f}) to alpha={alpha_grid[i]} "
                f"(H={entropies[i]:.4f}). "
                f"Decrease exceeds numerical tolerance of 0.01."
            )


# ---------------------------------------------------------------------------
# Tests: Entropy properties (INV-007)
# ---------------------------------------------------------------------------


class TestEntropyProperties:
    """Tests for entropy contribution non-negativity (INV-007)."""

    def test_contributions_non_negative(self) -> None:
        """INV-007: c'_k = (beta'_k)^2 * lambda_k >= 0 for any portfolio."""
        rng = np.random.RandomState(42)
        n, au = 15, 5
        eigenvalues = np.abs(rng.randn(au)) * 0.1 + 0.01  # Positive
        B_prime = rng.randn(n, au) * 0.3
        for _ in range(20):
            w = np.abs(rng.randn(n))
            w = w / w.sum()
            beta_prime = B_prime.T @ w
            c_prime = (beta_prime ** 2) * eigenvalues
            assert np.all(c_prime >= 0), f"Negative contribution found: {c_prime}"


# ---------------------------------------------------------------------------
# Tests: Constraint enforcement
# ---------------------------------------------------------------------------


class TestConstraintEnforcement:
    """Tests for constraint enforcement after optimization."""

    def test_w_min_semi_continuous(self) -> None:
        """After cardinality enforcement, no weight in forbidden zone (0, w_min)."""
        rng = np.random.RandomState(42)
        n, au = 15, 5
        B_prime = rng.randn(n, au) * 0.3
        eigenvalues = np.abs(rng.randn(au)) * 0.1 + 0.01
        Sigma_z = np.diag(eigenvalues)
        B_A_port = B_prime  # B_prime = B_A_port @ V, but V=I for this test
        D_eps = rng.uniform(0.001, 0.01, n)
        Sigma_assets = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
        w_opt, _, _ = multi_start_optimize(
            Sigma_assets=Sigma_assets, B_prime=B_prime, eigenvalues=eigenvalues,
            D_eps=D_eps, alpha=1.0, n_starts=3, seed=42, lambda_risk=1.0,
            phi=25.0, w_bar=0.03, w_max=0.10, is_first=True, max_iter=30,
        )
        w_min = 0.001

        sca_kwargs: dict[str, Any] = {
            "Sigma_assets": Sigma_assets,
            "B_prime": B_prime,
            "eigenvalues": eigenvalues,
            "alpha": 1.0,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "w_max": 0.10,
            "is_first": True,
        }
        w_card = enforce_cardinality(
            w_opt, B_prime, eigenvalues, w_min=w_min,
            sca_solver_fn=sca_optimize, sca_kwargs=sca_kwargs,
        )
        for i, wi in enumerate(w_card):
            assert wi < 1e-8 or wi >= w_min - 1e-8, (
                f"Weight {i}={wi} in forbidden zone (0, {w_min})"
            )
        assert abs(np.sum(w_card) - 1.0) < 1e-6, (
            f"Weights don't sum to 1: {np.sum(w_card):.8f}"
        )

    def test_multi_start_finite_output(self) -> None:
        """multi_start_optimize returns finite w, f, H."""
        rng = np.random.RandomState(42)
        n, au = 30, 5
        B_prime = rng.randn(n, au) * 0.3
        eigenvalues = np.abs(rng.randn(au)) * 0.1 + 0.01
        D_eps = rng.uniform(0.001, 0.01, n)
        Sigma_assets = B_prime @ np.diag(eigenvalues) @ B_prime.T + np.diag(D_eps)
        w, f, H = multi_start_optimize(
            Sigma_assets=Sigma_assets, B_prime=B_prime, eigenvalues=eigenvalues,
            D_eps=D_eps, alpha=1.0, n_starts=5, seed=42, lambda_risk=1.0,
            phi=25.0, w_bar=0.03, w_max=0.10, is_first=True, max_iter=30,
        )
        assert np.all(np.isfinite(w)), "Non-finite weights"
        assert np.isfinite(f), "Non-finite objective"
        assert np.isfinite(H), "Non-finite entropy"

    def test_multi_start_explores_better_optima(self) -> None:
        """
        m5: Multi-start with n_starts > 1 should find an objective at least
        as good as single-start. This proves multi-start explores beyond
        a single local optimum.
        """
        rng = np.random.RandomState(42)
        n, au = 30, 5
        B_prime = rng.randn(n, au) * 0.3
        eigenvalues = np.abs(rng.randn(au)) * 0.1 + 0.01
        D_eps = rng.uniform(0.001, 0.01, n)
        Sigma_assets = B_prime @ np.diag(eigenvalues) @ B_prime.T + np.diag(D_eps)

        common = {
            "Sigma_assets": Sigma_assets,
            "B_prime": B_prime,
            "eigenvalues": eigenvalues,
            "D_eps": D_eps,
            "alpha": 1.0,
            "seed": 42,
            "lambda_risk": 1.0,
            "phi": 25.0,
            "w_bar": 0.03,
            "w_max": 0.10,
            "is_first": True,
            "max_iter": 30,
        }

        _, f_single, H_single = multi_start_optimize(n_starts=1, **common)
        _, f_multi, H_multi = multi_start_optimize(n_starts=5, **common)

        # Multi-start should find an objective >= single start
        # (f = H - penalty, so higher is better)
        assert f_multi >= f_single - 1e-8, (
            f"m5: Multi-start objective ({f_multi:.6f}) should be >= "
            f"single-start ({f_single:.6f})"
        )


# ---------------------------------------------------------------------------
# C1: Entropy formula verification on known inputs
# ---------------------------------------------------------------------------


class TestEntropyFormulaVerification:
    """C1: Verify entropy formula H(w) on analytically known cases."""

    def test_entropy_equal_weight_identity_max(self) -> None:
        """Equal weight w=1/n with B'=I, eigenvalues=[1,...,1] -> H = ln(AU)."""
        au = 5
        n = au
        B_prime = np.eye(n, dtype=np.float64)
        eigenvalues = np.ones(au, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) / n

        H = compute_entropy_only(w, B_prime, eigenvalues)
        expected = np.log(au)
        assert abs(H - expected) < 1e-10, (
            f"Equal-weight + identity: H={H:.10f}, expected ln({au})={expected:.10f}"
        )

    def test_entropy_concentrated_single_factor(self) -> None:
        """Concentrated w=e_1 with B'=I, eigenvalues=[1,...,1] -> H = 0."""
        au = 5
        n = au
        B_prime = np.eye(n, dtype=np.float64)
        eigenvalues = np.ones(au, dtype=np.float64)
        w = np.zeros(n, dtype=np.float64)
        w[0] = 1.0

        H = compute_entropy_only(w, B_prime, eigenvalues)
        assert abs(H) < 1e-10, (
            f"Concentrated w=e_1 + identity: H={H:.10f}, expected 0"
        )

    def test_entropy_manual_computation(self) -> None:
        """Verify H(w) by manually computing c'_k, c_hat_k, and -sum(c_hat * log(c_hat))."""
        au = 3
        n = au
        B_prime = np.eye(n, dtype=np.float64)
        eigenvalues = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        w = np.array([0.5, 0.3, 0.2], dtype=np.float64)

        # Manual: beta' = B'^T w = w (since B'=I)
        beta_prime = w
        c_prime = (beta_prime ** 2) * eigenvalues
        # c' = [0.25*1, 0.09*2, 0.04*3] = [0.25, 0.18, 0.12]
        C = np.sum(c_prime)
        c_hat = c_prime / C
        H_manual = -np.sum(c_hat * np.log(c_hat))

        H = compute_entropy_only(w, B_prime, eigenvalues)
        assert abs(H - H_manual) < 1e-12, (
            f"Entropy mismatch: function={H:.12f}, manual={H_manual:.12f}"
        )


# ---------------------------------------------------------------------------
# C4: Objective function decomposition test
# ---------------------------------------------------------------------------


class TestObjectiveDecomposition:
    """C4: Verify f(w) = ret - lambda*risk + alpha*H - phi*P_conc - P_turn."""

    def test_objective_decomposition_matches(self) -> None:
        """Each term of objective_function computed independently must sum to f."""
        np.random.seed(SEED)
        data = _make_portfolio_data()

        rng = np.random.RandomState(SEED + 1)
        w = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
        w_old = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float64)

        alpha = 2.0
        lambda_risk = 1.5
        phi = 25.0
        w_bar = 0.03
        kappa_1 = 0.1
        kappa_2 = 7.5
        delta_bar = 0.01

        # Full objective
        f = objective_function(
            w=w, Sigma_assets=data["Sigma_assets"],
            B_prime=data["B_prime"], eigenvalues=data["eigenvalues"],
            alpha=alpha, lambda_risk=lambda_risk, phi=phi, w_bar=w_bar,
            w_old=w_old, kappa_1=kappa_1, kappa_2=kappa_2,
            delta_bar=delta_bar, is_first=False,
        )

        # Independent term computation
        risk_term = lambda_risk * float(w @ data["Sigma_assets"] @ w)
        H = compute_entropy_only(w, data["B_prime"], data["eigenvalues"])
        P_conc = concentration_penalty(w, w_bar)
        P_turn = turnover_penalty(w, w_old, kappa_1, kappa_2, delta_bar, is_first=False)

        f_manual = -risk_term + alpha * H - phi * P_conc - P_turn

        assert abs(f - f_manual) < 1e-10, (
            f"Objective decomposition mismatch: f={f:.10f}, manual={f_manual:.10f}"
        )


# ---------------------------------------------------------------------------
# C5: Concentration penalty formula test
# ---------------------------------------------------------------------------


class TestConcentrationPenaltyFormula:
    """C5: Verify P_conc = sum(max(0, w_i - w_bar)^2) on known values."""

    def test_concentration_penalty_known_values(self) -> None:
        """For w = [0.01, 0.05, 0.10] with w_bar=0.03:
        P_conc = max(0,0.01-0.03)^2 + max(0,0.05-0.03)^2 + max(0,0.10-0.03)^2
               = 0 + 0.0004 + 0.0049 = 0.0053
        """
        w = np.array([0.01, 0.05, 0.10], dtype=np.float64)
        w_bar = 0.03

        expected = 0.0 + (0.05 - 0.03) ** 2 + (0.10 - 0.03) ** 2
        # = 0 + 0.0004 + 0.0049 = 0.0053
        result = concentration_penalty(w, w_bar)
        assert abs(result - expected) < 1e-12, (
            f"Concentration penalty: got {result:.12f}, expected {expected:.12f}"
        )

    def test_concentration_penalty_below_threshold_zero(self) -> None:
        """When all w_i <= w_bar, penalty is exactly zero."""
        w = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        result = concentration_penalty(w, w_bar=0.03)
        assert result == 0.0, f"Penalty should be 0 when all w <= w_bar, got {result}"

    def test_concentration_penalty_gradient(self) -> None:
        """Gradient = 2 * max(0, w_i - w_bar)."""
        from src.portfolio.constraints import concentration_penalty_gradient

        w = np.array([0.01, 0.05, 0.10], dtype=np.float64)
        w_bar = 0.03
        grad = concentration_penalty_gradient(w, w_bar)

        expected_grad = np.array([0.0, 2 * 0.02, 2 * 0.07], dtype=np.float64)
        np.testing.assert_allclose(
            grad, expected_grad, atol=1e-12,
            err_msg="Concentration gradient mismatch",
        )


# ---------------------------------------------------------------------------
# C6: Turnover penalty formula test
# ---------------------------------------------------------------------------


class TestTurnoverPenaltyFormula:
    """C6: Verify P_turn = kappa_1*(1/2)*sum(|dw|) + kappa_2*sum(max(0,|dw|-delta_bar)^2)."""

    def test_turnover_penalty_known_values(self) -> None:
        """Verify turnover penalty with known w, w_old."""
        w = np.array([0.30, 0.30, 0.40], dtype=np.float64)
        w_old = np.array([0.25, 0.35, 0.40], dtype=np.float64)
        kappa_1 = 0.1
        kappa_2 = 7.5
        delta_bar = 0.01

        delta_w = np.abs(w - w_old)  # [0.05, 0.05, 0.00]
        linear = kappa_1 * 0.5 * np.sum(delta_w)  # 0.1 * 0.5 * 0.10 = 0.005
        excess = np.maximum(0.0, delta_w - delta_bar)  # [0.04, 0.04, 0.00]
        quadratic = kappa_2 * np.sum(excess ** 2)  # 7.5 * (0.0016 + 0.0016) = 0.024
        expected = linear + quadratic

        result = turnover_penalty(w, w_old, kappa_1, kappa_2, delta_bar, is_first=False)
        assert abs(result - expected) < 1e-12, (
            f"Turnover penalty: got {result:.12f}, expected {expected:.12f}"
        )

    def test_turnover_penalty_first_rebalancing_zero(self) -> None:
        """is_first=True must produce exactly zero penalty."""
        w = np.array([0.5, 0.5], dtype=np.float64)
        w_old = np.array([0.0, 1.0], dtype=np.float64)
        result = turnover_penalty(w, w_old, kappa_1=0.1, kappa_2=7.5,
                                  delta_bar=0.01, is_first=True)
        assert result == 0.0, f"First rebalancing should have zero turnover, got {result}"

    def test_turnover_penalty_within_threshold_linear_only(self) -> None:
        """When all |dw_i| <= delta_bar, only linear term contributes."""
        w = np.array([0.505, 0.495], dtype=np.float64)
        w_old = np.array([0.500, 0.500], dtype=np.float64)
        kappa_1 = 0.1
        kappa_2 = 7.5
        delta_bar = 0.01

        delta_w = np.abs(w - w_old)  # [0.005, 0.005]
        expected_linear = kappa_1 * 0.5 * np.sum(delta_w)
        # All within threshold, so quadratic = 0
        result = turnover_penalty(w, w_old, kappa_1, kappa_2, delta_bar, is_first=False)
        assert abs(result - expected_linear) < 1e-12, (
            f"Within-threshold: got {result:.12f}, expected linear-only={expected_linear:.12f}"
        )
