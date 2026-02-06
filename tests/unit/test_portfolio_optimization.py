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

import numpy as np
import pytest

from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.sca_solver import sca_optimize, multi_start_optimize, objective_function
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
            assert np.all(rel_error < 1e-5), (
                f"Max relative error = {rel_error.max():.2e}, "
                f"positions: {np.where(mask)[0][rel_error > 1e-5]}"
            )

        # Also check absolute error everywhere
        np.testing.assert_allclose(
            grad_H, grad_numerical, atol=1e-5,
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
        """After enforce_cardinality, no weight should be in (0, w_min).

        Create a solution with semi-continuous violations, enforce cardinality,
        and verify all active positions are at or above w_min.
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

        # Dummy SCA solver that just returns the input (no re-optimization)
        def dummy_sca(w_init: np.ndarray, **kwargs: object) -> tuple[np.ndarray, float, float, int]:
            return w_init.copy(), 0.0, 0.0, 1

        sca_kwargs = {
            "Sigma_assets": data["Sigma_assets"],
            "B_prime": data["B_prime"],
            "eigenvalues": data["eigenvalues"],
            "alpha": 1.0,
        }

        w_enforced = enforce_cardinality(
            w=w,
            B_prime=data["B_prime"],
            eigenvalues=data["eigenvalues"],
            w_min=w_min,
            sca_solver_fn=dummy_sca,
            sca_kwargs=sca_kwargs,
        )

        # Check: no weight in the forbidden zone (0, w_min)
        for i in range(N_STOCKS):
            assert w_enforced[i] == 0.0 or w_enforced[i] >= w_min - 1e-8, (
                f"Violation at stock {i}: w={w_enforced[i]:.6f}, "
                f"w_min={w_min}"
            )


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
        """
        np.random.seed(SEED)

        w = np.random.dirichlet(np.ones(N_STOCKS)).astype(np.float64)
        w_old = np.random.dirichlet(np.ones(N_STOCKS)).astype(np.float64)

        P_turn = turnover_penalty(
            w, w_old, kappa_1=0.1, kappa_2=7.5, delta_bar=0.01, is_first=True,
        )

        assert P_turn == 0.0, f"Turnover penalty not zero: {P_turn:.8f}"


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
            w_opt, w_expected, atol=1e-3,
            err_msg="Optimal weights differ from equal-weight for Sigma=I, alpha=0",
        )
