"""
Sequential Convex Approximation (SCA) solver with Armijo backtracking.

Multi-start optimization (M=5 initializations):
1. Equal-weight
2. Minimum variance (QP)
3. Approximate ERC (Spinu)
4-5. Random (Dirichlet + projection)

SCA iterations linearize H(w) and solve a convex sub-problem via CVXPY.
Each iteration builds a fresh CVXPY problem with the entropy gradient
baked in as a constant, ensuring DCP compliance across all CVXPY versions
(avoids DPP canonicalization failures with auxiliary variables).

DVT Section 4.7: "CVXPY + MOSEK/ECOS (recommended)".
Reference: ISD Section MOD-008 — Sub-task 2.
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cvxpy as cp

from src.validation import (
    assert_armijo_params_valid,
    assert_cholesky_condition,
    assert_covariance_valid,
    assert_gradient_finite,
    assert_weights_valid,
)
from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.constraints import (
    concentration_penalty,
    turnover_penalty,
    project_to_constraints,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver detection (DVT: "MOSEK/ECOS recommended", MOSEK listed first)
# ---------------------------------------------------------------------------

def _build_solver_chain() -> list[tuple[str, dict[str, object]]]:
    """
    Build solver fallback chain from installed solvers.

    Priority: MOSEK > CLARABEL > ECOS > SCS (DVT Section 4.7).

    :return chain (list): Ordered list of (solver_name, solver_kwargs) tuples
    """
    installed = set(cp.installed_solvers())
    chain: list[tuple[str, dict[str, object]]] = []

    if "MOSEK" in installed:
        chain.append(("MOSEK", {"warm_start": True}))
    if "CLARABEL" in installed:
        chain.append(("CLARABEL", {"warm_start": True}))
    if "ECOS" in installed:
        chain.append(("ECOS", {"warm_start": True, "max_iters": 500}))
    chain.append(("SCS", {"warm_start": True, "max_iters": 5000}))

    return chain


_SOLVER_CHAIN = _build_solver_chain()
logger.debug("SCA solver chain: %s", [s[0] for s in _SOLVER_CHAIN])


def _build_mi_solver_chain() -> list[tuple[str, dict[str, object]]]:
    """
    Build solver chain for mixed-integer problems.

    Only MOSEK supports MI among our solver dependencies.

    :return chain (list): Ordered list of (solver_name, solver_kwargs) tuples
    """
    installed = set(cp.installed_solvers())
    chain: list[tuple[str, dict[str, object]]] = []
    if "MOSEK" in installed:
        chain.append(("MOSEK", {"warm_start": True}))
    return chain


_MI_SOLVER_CHAIN = _build_mi_solver_chain()


def has_mi_solver() -> bool:
    """
    Check whether a mixed-integer-capable solver is available.

    :return available (bool): True if MOSEK is installed
    """
    return len(_MI_SOLVER_CHAIN) > 0


# ---------------------------------------------------------------------------
# Cholesky factorization
# ---------------------------------------------------------------------------

def _safe_cholesky(Sigma: np.ndarray) -> np.ndarray:
    """
    Cholesky factorization with minimal ridge for numerical stability.

    :param Sigma (np.ndarray): Symmetric PSD matrix (n, n)

    :return L (np.ndarray): Lower-triangular factor, Sigma ≈ L @ L.T
    """
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        n = Sigma.shape[0]
        ridge = 1e-8 * np.trace(Sigma) / n
        L = np.linalg.cholesky(Sigma + ridge * np.eye(n))

    # CRITICAL: Validate Cholesky output is finite (diagnostic fix)
    assert np.isfinite(L).all(), "Cholesky produced non-finite values"

    # Validate Cholesky factor is well-conditioned
    assert_cholesky_condition(L, 1e10, "Sigma_Cholesky")

    return L


# ---------------------------------------------------------------------------
# Parametric SCA sub-problem (built once, reused across iterations)
# ---------------------------------------------------------------------------

class _SCASubproblemBuilder:
    """CVXPY sub-problem for one SCA iteration.

    Builds a fresh CVXPY problem each iteration with the current entropy
    gradient baked in as a constant.  This avoids DPP (Disciplined
    Parameterized Programming) entirely, ensuring compatibility across all
    CVXPY versions (some versions raise DCPError during lazy DPP
    canonicalization when auxiliary variables are present).

    Uses Cholesky factor L instead of Sigma for efficient cone formulation:
    lambda ||L^T w||^2 = lambda w^T Sigma w.
    """

    __slots__ = (
        '_n', '_alpha', '_L_sigma', '_lambda_risk', '_phi', '_w_bar',
        '_w_max', '_w_old', '_kappa_1', '_kappa_2', '_delta_bar',
        '_tau_max', '_is_first', '_mu',
    )

    def __init__(
        self,
        n: int,
        L_sigma: np.ndarray,
        alpha: float,
        lambda_risk: float,
        phi: float,
        w_bar: float,
        w_max: float,
        w_old: np.ndarray | None,
        kappa_1: float,
        kappa_2: float,
        delta_bar: float,
        tau_max: float,
        is_first: bool,
        mu: np.ndarray | None = None,
    ) -> None:
        self._n = n
        self._alpha = alpha
        self._L_sigma = L_sigma
        self._lambda_risk = lambda_risk
        self._phi = phi
        self._w_bar = w_bar
        self._w_max = w_max
        self._w_old = w_old
        self._kappa_1 = kappa_1
        self._kappa_2 = kappa_2
        self._delta_bar = delta_bar
        self._tau_max = tau_max
        self._is_first = is_first
        self._mu = mu

    def solve(
        self, grad_H: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        """
        Build and solve a fresh CVXPY problem with the current entropy gradient.

        :param grad_H (np.ndarray): Entropy gradient at current iterate (n,)

        :return w_star (np.ndarray | None): Optimal weights, or None if infeasible
        :return shadow_prices (dict): Constraint shadow prices (dual values):
            - budget (float): Budget constraint dual value
            - w_max_binding_prices (list[float]): Per-asset w_max dual values
            - w_min_binding_prices (list[float]): Per-asset w_min (non-neg) dual values
            - turnover (float | None): Turnover constraint dual value (if active)
        """
        n = self._n
        w = cp.Variable(n)

        # Entropy gradient as constant coefficient (no cp.Parameter)
        grad_coeff = self._alpha * grad_H

        # Return term
        ret_term: cp.Expression | float = (
            self._mu @ w if self._mu is not None else 0.0
        )

        # Risk via Cholesky: lambda ||L^T w||^2
        risk_term = self._lambda_risk * cp.sum_squares(self._L_sigma.T @ w)

        # Linearized entropy: constant @ variable (DCP-affine)
        entropy_term = grad_coeff @ w

        # Concentration penalty via auxiliary variable for DCP compliance.
        conc_penalty: cp.Expression = cp.Constant(0.0)
        conc_constraints: list[cp.Constraint] = []
        if self._phi > 0:
            t_conc = cp.Variable(n, nonneg=True)
            conc_constraints = [t_conc >= w - self._w_bar]  # type: ignore[list-item]
            conc_penalty = self._phi * cp.sum_squares(t_conc)

        # Turnover penalty via auxiliary variable for DCP compliance.
        turn_penalty: cp.Expression = cp.Constant(0.0)
        turn_constraints: list[cp.Constraint] = []
        if self._w_old is not None and not self._is_first:
            delta_w = cp.abs(w - self._w_old)
            linear_turn = self._kappa_1 * 0.5 * cp.sum(delta_w)
            t_turn = cp.Variable(n, nonneg=True)
            turn_constraints = [t_turn >= delta_w - self._delta_bar]  # type: ignore[list-item]
            quad_turn = self._kappa_2 * cp.sum_squares(t_turn)
            turn_penalty = linear_turn + quad_turn  # type: ignore[assignment]

        obj = cp.Maximize(
            ret_term - risk_term + entropy_term - conc_penalty - turn_penalty
        )

        # Named constraints for shadow price extraction (G7)
        # Hard cap uses w_max directly.  w_bar is only for the soft
        # concentration penalty phi * sum(max(0, w_i - w_bar)^2).
        w_min_constraint = w >= 0  # Non-negativity
        w_max_constraint = w <= self._w_max  # Upper bound
        budget_constraint = cp.sum(w) == 1  # Budget

        # Use Any type since CVXPY constraint expressions have complex types
        constraints: list[object] = [
            w_min_constraint,
            w_max_constraint,
            budget_constraint,
        ]
        constraints.extend(conc_constraints)
        constraints.extend(turn_constraints)

        # Turnover constraint (optional)
        turnover_constraint: object | None = None
        has_turnover = self._w_old is not None and not self._is_first
        if has_turnover:
            turnover_constraint = (
                0.5 * cp.sum(cp.abs(w - self._w_old)) <= self._tau_max
            )
            constraints.append(turnover_constraint)

        problem = cp.Problem(obj, constraints)  # type: ignore[arg-type]

        # Default empty shadow prices
        shadow_prices: dict[str, object] = {
            "budget": 0.0,
            "w_max_binding_prices": [],
            "w_min_binding_prices": [],
            "turnover": None,
        }

        for solver_name, solver_kwargs in _SOLVER_CHAIN:
            try:
                problem.solve(
                    solver=solver_name, **solver_kwargs,  # type: ignore[arg-type]
                )
                if (problem.status in ("optimal", "optimal_inaccurate")
                        and w.value is not None):
                    result = np.array(w.value).flatten()
                    if not np.any(np.isnan(result)):
                        # Extract shadow prices (dual values) from constraints
                        # Budget constraint dual value (scalar for equality)
                        budget_dual = getattr(budget_constraint, "dual_value", None)
                        shadow_prices["budget"] = (
                            float(budget_dual) if budget_dual is not None else 0.0
                        )

                        # w_max constraint dual values (array for elementwise <=)
                        w_max_dual = getattr(w_max_constraint, "dual_value", None)
                        if w_max_dual is not None:
                            w_max_arr = np.atleast_1d(np.asarray(w_max_dual))  # type: ignore[call-overload]
                            shadow_prices["w_max_binding_prices"] = [
                                float(d) for d in w_max_arr  # type: ignore[arg-type]
                            ]
                        else:
                            shadow_prices["w_max_binding_prices"] = [0.0] * n

                        # w_min constraint dual values (array for elementwise >=)
                        w_min_dual = getattr(w_min_constraint, "dual_value", None)
                        if w_min_dual is not None:
                            w_min_arr = np.atleast_1d(np.asarray(w_min_dual))  # type: ignore[call-overload]
                            shadow_prices["w_min_binding_prices"] = [
                                float(d) for d in w_min_arr  # type: ignore[arg-type]
                            ]
                        else:
                            shadow_prices["w_min_binding_prices"] = [0.0] * n

                        # Turnover constraint dual value (scalar if exists)
                        if has_turnover and turnover_constraint is not None:
                            turn_dual = getattr(turnover_constraint, "dual_value", None)
                            shadow_prices["turnover"] = (
                                float(turn_dual) if turn_dual is not None else 0.0  # type: ignore[arg-type]
                            )

                        return result, shadow_prices
            except (cp.SolverError, ArithmeticError):
                continue

        return None, shadow_prices


# ---------------------------------------------------------------------------
# Entropy gradient normalization (Michaud 1989, Goldfarb & Iyengar 2003)
# ---------------------------------------------------------------------------

def _compute_alpha_eff(
    alpha: float,
    w_ref: np.ndarray,
    L_sigma: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    lambda_risk: float,
    entropy_eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    budget: np.ndarray | None = None,
) -> float:
    """
    Compute effective alpha by normalizing entropy gradient to risk gradient.

    Without normalization, the entropy gradient dominates the risk gradient
    by ~100-1000x in typical equity settings (lambda_risk=252, daily Sigma),
    making the optimizer effectively a pure entropy maximizer regardless of
    alpha.  This function re-scales alpha so that the two gradient norms
    are commensurate, letting alpha control the *relative* trade-off rather
    than being overwhelmed by scale differences.

    Reference: Michaud (1989) "The Markowitz Optimization Enigma",
               Goldfarb & Iyengar (2003) "Robust Portfolio Selection".

    :param alpha (float): Raw entropy weight from the frontier grid
    :param w_ref (np.ndarray): Reference weights for gradient evaluation (n,)
    :param L_sigma (np.ndarray): Cholesky factor of Sigma_assets (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param lambda_risk (float): Risk aversion coefficient
    :param entropy_eps (float): Numerical stability for entropy
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,)
    :param idio_weight (float): Idiosyncratic entropy weight

    :return alpha_eff (float): Normalized alpha making gradients commensurate
    """
    if alpha <= 0:
        return alpha

    _, grad_H = compute_entropy_and_gradient(
        w_ref, B_prime, eigenvalues, entropy_eps,
        D_eps=D_eps, idio_weight=idio_weight, budget=budget,
    )

    Lw = L_sigma.T @ w_ref
    risk_grad = 2.0 * lambda_risk * (L_sigma @ Lw)

    risk_norm = float(np.linalg.norm(risk_grad))
    entropy_norm = float(np.linalg.norm(grad_H))

    if entropy_norm < 1e-15:
        logger.debug("Entropy gradient near zero — skipping normalization.")
        return alpha

    balance_ratio = risk_norm / entropy_norm
    alpha_eff = alpha * balance_ratio

    logger.info(
        "Entropy gradient normalization: balance_ratio=%.4f, "
        "alpha=%.4g -> alpha_eff=%.6g "
        "(||risk_grad||=%.4e, ||grad_H||=%.4e)",
        balance_ratio, alpha, alpha_eff, risk_norm, entropy_norm,
    )

    return alpha_eff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def objective_function(
    w: np.ndarray,
    Sigma_assets: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    alpha: float,
    lambda_risk: float,
    phi: float,
    w_bar: float,
    w_old: np.ndarray | None,
    kappa_1: float,
    kappa_2: float,
    delta_bar: float,
    is_first: bool,
    mu: np.ndarray | None = None,
    entropy_eps: float = 1e-30,
    _L_sigma: np.ndarray | None = None,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    budget: np.ndarray | None = None,
) -> float:
    """
    Full objective: f(w) = w^T μ - λ w^T Σ w + α H(w) - φ P_conc - P_turn

    Current implementation uses μ=0 (DVT Section 4.7).

    :param w (np.ndarray): Weights (n,)
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param alpha (float): Entropy weight
    :param lambda_risk (float): Risk aversion
    :param phi (float): Concentration penalty weight
    :param w_bar (float): Concentration threshold
    :param w_old (np.ndarray | None): Previous weights
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover threshold
    :param is_first (bool): First rebalancing flag
    :param mu (np.ndarray | None): Expected returns (default: zeros)
    :param entropy_eps (float): Numerical stability for entropy
    :param _L_sigma (np.ndarray | None): Pre-computed Cholesky factor (internal)
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,) for
        entropy computation including idiosyncratic risk contributions.
    :param idio_weight (float): Weight for idiosyncratic entropy layer (0-1).
    :param budget (np.ndarray | None): Target factor risk budget (AU,).
        When provided, entropy uses tilted formulation targeting
        eigenvalue-proportional contributions (Roncalli 2013, Ch. 7).

    :return f (float): Objective value (to maximize)
    """
    n = len(w)

    # Return term (μ=0 by default)
    ret_term = float(w @ mu) if mu is not None else 0.0

    # Risk term — use Cholesky if available
    if _L_sigma is not None:
        v = _L_sigma.T @ w
        risk_term = lambda_risk * float(v @ v)
    else:
        risk_term = lambda_risk * float(w @ Sigma_assets @ w)

    # Entropy term (two-layer: factor + idiosyncratic, tilted if budget given)
    H = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                             idio_weight=idio_weight, budget=budget)

    # Concentration penalty
    P_conc = concentration_penalty(w, w_bar)

    # Turnover penalty
    P_turn = 0.0
    if w_old is not None:
        P_turn = turnover_penalty(w, w_old, kappa_1, kappa_2, delta_bar, is_first)

    return ret_term - risk_term + alpha * H - phi * P_conc - P_turn


def armijo_backtracking(
    w_current: np.ndarray,
    w_star: np.ndarray,
    f_current: float,
    delta_surr: float,
    obj_fn: callable,  # type: ignore[type-arg]
    c: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20,
) -> float:
    """
    Armijo backtracking line search.

    Find η = max{ρ^j} such that f(w + η(w*-w)) ≥ f(w) + c · η · Δ_surr

    :param w_current (np.ndarray): Current iterate (n,)
    :param w_star (np.ndarray): Sub-problem solution (n,)
    :param f_current (float): f(w_current)
    :param delta_surr (float): Surrogate improvement Δ_surr ≥ 0
    :param obj_fn (callable): Objective function f(w) → float
    :param c (float): Armijo sufficient decrease constant
    :param rho (float): Backtracking factor
    :param max_iter (int): Maximum backtracking steps

    :return eta (float): Step size
    """
    eta = 1.0
    direction = w_star - w_current

    for _ in range(max_iter):
        w_trial = w_current + eta * direction
        f_trial = obj_fn(w_trial)

        if f_trial >= f_current + c * eta * delta_surr:
            return eta

        eta *= rho

    return eta


def sca_optimize(
    w_init: np.ndarray,
    Sigma_assets: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    alpha: float,
    lambda_risk: float = 1.0,
    phi: float = 25.0,
    w_bar: float = 0.03,
    w_max: float = 0.05,
    w_old: np.ndarray | None = None,
    kappa_1: float = 0.1,
    kappa_2: float = 7.5,
    delta_bar: float = 0.01,
    tau_max: float = 0.30,
    is_first: bool = False,
    mu: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    armijo_c: float = 1e-4,
    armijo_rho: float = 0.5,
    armijo_max_iter: int = 20,
    entropy_eps: float = 1e-30,
    _L_sigma: np.ndarray | None = None,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    budget: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, int, dict[str, object]]:
    """
    SCA optimization from a single starting point.

    Uses parametric CVXPY problem (built once, reused across iterations)
    and pre-computed Cholesky for fast risk evaluation.

    :param w_init (np.ndarray): Initial weights (n,)
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param alpha (float): Entropy weight
    :param lambda_risk (float): Risk aversion
    :param phi (float): Concentration penalty weight
    :param w_bar (float): Concentration threshold
    :param w_max (float): Maximum weight
    :param w_old (np.ndarray | None): Previous weights
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover threshold
    :param tau_max (float): Maximum one-way turnover
    :param is_first (bool): First rebalancing flag
    :param mu (np.ndarray | None): Expected returns
    :param max_iter (int): Max SCA iterations
    :param tol (float): Convergence tolerance (DVT: 1e-8)
    :param armijo_c (float): Armijo constant
    :param armijo_rho (float): Armijo backtracking factor
    :param armijo_max_iter (int): Max Armijo steps
    :param entropy_eps (float): Numerical stability
    :param _L_sigma (np.ndarray | None): Pre-computed Cholesky (internal)
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,) for
        entropy computation including idiosyncratic risk contributions.
    :param idio_weight (float): Weight for idiosyncratic entropy layer (0-1).
    :param budget (np.ndarray | None): Target factor risk budget (AU,).
        When provided, uses tilted entropy (Roncalli 2013, Ch. 7).

    :return w_final (np.ndarray): Optimized weights (n,)
    :return f_final (float): Final objective value
    :return H_final (float): Final entropy
    :return n_iters (int): Number of SCA iterations used
    :return convergence_info (dict): Solver diagnostics with keys:
        - converged (bool): True if tol reached before max_iter
        - final_grad_norm (float): Norm of smooth gradient at solution
        - step_sizes (list[float]): Armijo step sizes per iteration
        - obj_improvements (list[float]): Objective change per iteration
        - shadow_prices (dict): Constraint dual values from final iteration:
            - budget (float): Budget constraint shadow price
            - w_max_binding_prices (list[float]): Per-asset upper bound prices
            - w_min_binding_prices (list[float]): Per-asset lower bound prices
            - turnover (float | None): Turnover constraint price (if active)
    """
    n = len(w_init)

    # CRITICAL: Validate w_init sums to approximately 1 (diagnostic fix)
    w_init_sum = float(np.sum(w_init))
    assert abs(w_init_sum - 1.0) < 1e-3, (
        f"w_init does not sum to 1: sum={w_init_sum:.6f}"
    )

    # Validate Armijo backtracking parameters
    assert_armijo_params_valid(armijo_c, armijo_rho, "SCA_Armijo")

    # Pre-compute Cholesky once (or reuse from caller)
    L_sigma = _L_sigma if _L_sigma is not None else _safe_cholesky(Sigma_assets)

    # Build SCA sub-problem solver (rebuilds CVXPY problem each iteration
    # with gradient as constant to avoid DPP compatibility issues)
    subproblem = _SCASubproblemBuilder(
        n, L_sigma, alpha, lambda_risk, phi, w_bar, w_max,
        w_old, kappa_1, kappa_2, delta_bar, tau_max, is_first, mu,
    )

    w = w_init.copy()

    def obj_fn(w_eval: np.ndarray) -> float:
        return objective_function(
            w_eval, Sigma_assets, B_prime, eigenvalues, alpha,
            lambda_risk, phi, w_bar, w_old, kappa_1, kappa_2,
            delta_bar, is_first, mu, entropy_eps, _L_sigma=L_sigma,
            D_eps=D_eps, idio_weight=idio_weight, budget=budget,
        )

    f_current = obj_fn(w)
    assert np.isfinite(f_current), f"Initial objective is not finite: {f_current}"

    # Convergence tracking
    step_sizes: list[float] = []
    obj_improvements: list[float] = []
    converged = False
    final_grad_norm = 0.0
    final_iter = 0
    final_shadow_prices: dict[str, object] = {
        "budget": 0.0,
        "w_max_binding_prices": [],
        "w_min_binding_prices": [],
        "turnover": None,
    }

    for it in range(max_iter):
        # Compute entropy and gradient at current point
        _, grad_H = compute_entropy_and_gradient(
            w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
            idio_weight=idio_weight, budget=budget,
        )

        # Validate gradient is finite (prevents NaN propagation in SCA)
        assert_gradient_finite(grad_H, "entropy_gradient")

        # Solve SCA sub-problem (fresh CVXPY problem per iteration)
        w_star, shadow_prices = subproblem.solve(grad_H)

        if w_star is None:
            final_iter = it + 1
            break

        # Update shadow prices (keep last successful iteration's values)
        final_shadow_prices = shadow_prices

        # Directional derivative of the smooth part of the objective:
        #   ∇f_smooth = μ - 2λΣw + α∇H(w)
        # Used as reference descent for Armijo sufficient decrease.
        # Clamped to 0 when SCA direction is not an ascent direction for
        # the true objective (happens when linearization error is large).
        Lw = L_sigma.T @ w
        grad_risk = -2.0 * lambda_risk * (L_sigma @ Lw)
        grad_f_smooth = alpha * grad_H + grad_risk
        if mu is not None:
            grad_f_smooth = grad_f_smooth + mu
        direction = w_star - w
        delta_surr = max(float(grad_f_smooth @ direction), 0.0)

        # Track gradient norm for convergence diagnostics
        final_grad_norm = float(np.linalg.norm(grad_f_smooth))

        # Armijo backtracking
        eta = armijo_backtracking(
            w, w_star, f_current, delta_surr,
            obj_fn, armijo_c, armijo_rho, armijo_max_iter,
        )
        step_sizes.append(eta)

        # Update
        w_new = w + eta * (w_star - w)
        assert np.isfinite(w_new).all(), "w_new contains NaN/Inf after update"

        # Ensure feasibility: clip-renormalize loop to prevent cap
        # violation from renormalization of a clipped vector.
        # Uses w_max (hard cap) — w_bar is only for soft penalty.
        for _ in range(10):
            w_new = np.clip(w_new, 0.0, w_max)
            w_sum = np.sum(w_new)
            if w_sum > 0:
                w_new = w_new / w_sum
            if np.all(w_new <= w_max + 1e-10):
                break

        f_new = obj_fn(w_new)
        obj_improvements.append(f_new - f_current)

        # Convergence check: relative change in objective (scale-independent)
        if abs(f_new - f_current) < tol * max(1e-10, abs(f_current)):
            w = w_new
            f_current = f_new
            converged = True
            final_iter = it + 1
            break

        w = w_new
        f_current = f_new
        final_iter = it + 1

    H_final = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                                    idio_weight=idio_weight, budget=budget)
    assert np.isfinite(H_final), f"H_final is not finite: {H_final}"
    assert_weights_valid(w, "sca_final_weights")

    if final_grad_norm > 1e-4:
        warnings.warn(
            f"SCA final gradient norm {final_grad_norm:.2e} > 1e-4",
            stacklevel=2,
        )

    convergence_info: dict[str, object] = {
        "converged": converged,
        "final_grad_norm": final_grad_norm,
        "step_sizes": step_sizes,
        "obj_improvements": obj_improvements,
        "shadow_prices": final_shadow_prices,
    }

    return w, f_current, H_final, final_iter, convergence_info


def multi_start_optimize(
    Sigma_assets: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray,
    alpha: float,
    n_starts: int = 5,
    seed: int = 42,
    warm_start_w: np.ndarray | None = None,
    **kwargs: float | np.ndarray | bool | None,
) -> tuple[np.ndarray, float, float, dict[str, object]]:
    """
    Multi-start SCA optimization with parallel execution.

    Starts:
    1. Equal-weight
    2. Minimum variance (approximate)
    3. Approximate ERC (inverse vol)
    4+. Random (Dirichlet + projection)
    Optional: warm-start from a previous solution (replaces one random start)

    Pre-computes Cholesky once and shares across all starts.
    Runs starts in parallel via ThreadPoolExecutor (solvers release the GIL).

    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)
    :param alpha (float): Entropy weight
    :param n_starts (int): Number of starts
    :param seed (int): Random seed
    :param warm_start_w (np.ndarray | None): Optional warm-start weights from
        a previous optimization (e.g., adjacent alpha on the frontier).
        Replaces one random start to keep total starts constant.
    :param **kwargs: Additional parameters for sca_optimize

    :return w_best (np.ndarray): Best weights (n,)
    :return f_best (float): Best objective value
    :return H_best (float): Entropy at best solution
    :return solver_stats (dict): Aggregated solver statistics:
        - n_starts (int): Number of starts used
        - best_start_idx (int): Index of winning start
        - converged_count (int): Number of starts that converged
        - iterations (list[int]): Iterations per start
        - best_convergence_info (dict): Detailed info from best start
    """
    n = Sigma_assets.shape[0]
    rng = np.random.RandomState(seed)
    w_max_val = kwargs.get("w_max", 0.05)
    w_max = float(w_max_val) if w_max_val is not None else 0.05
    w_min_val = kwargs.get("w_min", 0.001)
    w_min = float(w_min_val) if w_min_val is not None else 0.001

    # VALIDATION: Verify covariance matrix is valid before optimization
    assert_covariance_valid(Sigma_assets, "Sigma_assets_multi_start")

    # Extract normalization flag (not forwarded to sca_optimize)
    _nflag = kwargs.pop("normalize_entropy_gradient", None)
    normalize_flag = _nflag is not False

    # Pre-compute Cholesky once for all starts
    L_sigma = _safe_cholesky(Sigma_assets)

    # Normalize alpha: rescale so entropy and risk gradient norms are
    # commensurate.  Evaluated at equal-weight (generic, data-independent
    # reference point).  All starts share the same alpha_eff for comparable
    # objective values across multi-start.
    if normalize_flag and alpha > 0:
        w_ew = np.ones(n) / n
        _lr = float(kwargs.get("lambda_risk", 1.0) or 1.0)
        _ee = float(kwargs.get("entropy_eps", 1e-30) or 1e-30)
        _iw = float(kwargs.get("idio_weight", 0.2) or 0.2)
        _budget_raw = kwargs.get("budget", None)
        _budget = _budget_raw if isinstance(_budget_raw, np.ndarray) else None
        alpha_eff = _compute_alpha_eff(
            alpha, w_ew, L_sigma, B_prime, eigenvalues, _lr,
            _ee, D_eps, _iw, budget=_budget,
        )
    else:
        alpha_eff = alpha

    # Generate starting points
    starts: list[np.ndarray] = []

    # 1. Equal-weight
    starts.append(np.ones(n) / n)

    # 2. Minimum variance (approximate: inverse diagonal)
    diag_sigma = np.diag(Sigma_assets)
    diag_sigma = np.maximum(diag_sigma, 1e-10)
    w_minvar = 1.0 / diag_sigma
    w_minvar = w_minvar / w_minvar.sum()
    starts.append(project_to_constraints(w_minvar, w_max, w_min))

    # 3. Approximate ERC (inverse vol)
    vol = np.sqrt(diag_sigma)
    w_erc = 1.0 / np.maximum(vol, 1e-10)
    w_erc = w_erc / w_erc.sum()
    starts.append(project_to_constraints(w_erc, w_max, w_min))

    # 4. Warm-start from previous solution (if available)
    if warm_start_w is not None and warm_start_w.shape[0] == n:
        starts.append(project_to_constraints(warm_start_w.copy(), w_max, w_min))

    # 5+. Random starts (fill remaining slots)
    n_random = max(0, n_starts - len(starts))
    for _ in range(n_random):
        n_active = rng.randint(30, min(300, n) + 1)
        active_idx = rng.choice(n, size=n_active, replace=False)
        w_rand = np.zeros(n)
        w_rand[active_idx] = rng.dirichlet(np.ones(n_active))
        starts.append(project_to_constraints(w_rand, w_max, w_min))

    # w_min is used only for initialization projection in multi_start_optimize,
    # not passed to sca_optimize (which has no w_min parameter — semi-continuous
    # constraints are enforced post-hoc via cardinality enforcement).
    sca_kwargs = {k: v for k, v in kwargs.items() if k != "w_min"}

    active_starts = starts[:n_starts]

    def _run_start(
        w_init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, int, dict[str, object]]:
        return sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=alpha_eff,
            _L_sigma=L_sigma,
            D_eps=D_eps,
            **sca_kwargs,  # type: ignore[arg-type]
        )

    # Run SCA from each start
    best_w = active_starts[0]
    best_f = -np.inf
    best_H = 0.0
    best_convergence_info: dict[str, object] = {}
    best_start_idx = 0
    iterations_list: list[int] = []
    converged_count = 0

    if len(active_starts) <= 1:
        w_opt, f_opt, H_opt, n_iter, conv_info = _run_start(active_starts[0])
        solver_stats: dict[str, object] = {
            "n_starts": 1,
            "best_start_idx": 0,
            "converged_count": 1 if conv_info.get("converged", False) else 0,
            "iterations": [n_iter],
            "best_convergence_info": conv_info,
        }
        return w_opt, f_opt, H_opt, solver_stats

    # Parallel execution: solvers (MOSEK/CLARABEL/ECOS) release the GIL
    with ThreadPoolExecutor(max_workers=len(active_starts)) as executor:
        futures = [
            executor.submit(_run_start, w_init)
            for w_init in active_starts
        ]
        # Iterate in submission order to preserve determinism on ties
        for idx, future in enumerate(futures):
            w_opt, f_opt, H_opt, n_iter, conv_info = future.result()
            iterations_list.append(n_iter)
            if conv_info.get("converged", False):
                converged_count += 1
            if f_opt > best_f:
                best_w = w_opt
                best_f = f_opt
                best_H = H_opt
                best_convergence_info = conv_info
                best_start_idx = idx

    solver_stats = {
        "n_starts": len(active_starts),
        "best_start_idx": best_start_idx,
        "converged_count": converged_count,
        "iterations": iterations_list,
        "best_convergence_info": best_convergence_info,
    }

    return best_w, best_f, best_H, solver_stats
