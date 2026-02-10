"""
Sequential Convex Approximation (SCA) solver with Armijo backtracking.

Multi-start optimization (M=5 initializations):
1. Equal-weight
2. Minimum variance (QP)
3. Approximate ERC (Spinu)
4-5. Random (Dirichlet + projection)

SCA iterations linearize H(w) and solve a convex sub-problem via CVXPY.
Each iteration builds a fresh problem with constant gradient coefficients
to ensure DCP compliance across all CVXPY versions.

DVT Section 4.7: "CVXPY + MOSEK/ECOS (recommended)".
Reference: ISD Section MOD-008 — Sub-task 2.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cvxpy as cp

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
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        n = Sigma.shape[0]
        ridge = 1e-8 * np.trace(Sigma) / n
        return np.linalg.cholesky(Sigma + ridge * np.eye(n))


# ---------------------------------------------------------------------------
# SCA sub-problem builder (fresh CVXPY problem per iteration, DCP-safe)
# ---------------------------------------------------------------------------

class _SCASubproblemBuilder:
    """SCA sub-problem builder with constant gradient coefficients.

    Builds a fresh CVXPY problem each iteration with the entropy gradient
    as a constant numpy array (constant @ variable is always DCP-affine).
    This avoids reliance on DPP canonicalization (cp.Parameter @ cp.Variable)
    which can fail on certain CVXPY versions.

    Uses Cholesky factor L instead of Sigma for efficient cone formulation:
    λ ||L^T w||² = λ w^T Σ w.
    """

    __slots__ = (
        '_n', '_L_sigma', '_alpha', '_lambda_risk', '_phi', '_w_bar',
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
        self._L_sigma = L_sigma
        self._alpha = alpha
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

    def solve(self, grad_H: np.ndarray) -> np.ndarray | None:
        """
        Build a fresh CVXPY problem with grad_H as constant and solve.

        Pre-multiplies alpha * grad_H into a single constant coefficient vector
        so that the entropy term is constant @ variable (always DCP-affine).

        :param grad_H (np.ndarray): Entropy gradient at current iterate (n,)

        :return w_star (np.ndarray | None): Optimal weights, or None if infeasible
        """
        w = cp.Variable(self._n)

        # Return term
        if self._mu is not None:
            ret_term = self._mu @ w
        else:
            ret_term = 0.0

        # Risk via Cholesky: λ ||L^T w||² = λ w^T Σ w
        risk_term = self._lambda_risk * cp.sum_squares(self._L_sigma.T @ w)

        # Linearized entropy: (α · ∇H)^T w — constant numpy vector @ variable
        entropy_coeff = self._alpha * grad_H  # numpy array (constant)
        entropy_term = entropy_coeff @ w

        # Concentration penalty — sum(square(pos(...))) for DCP compliance
        conc_penalty = self._phi * cp.sum(cp.square(cp.pos(w - self._w_bar)))

        # Turnover penalty — same DCP-safe pattern
        turn_penalty: cp.Expression = cp.Constant(0.0)
        if self._w_old is not None and not self._is_first:
            delta_w = cp.abs(w - self._w_old)
            linear_turn = self._kappa_1 * 0.5 * cp.sum(delta_w)
            excess_turn = cp.pos(delta_w - self._delta_bar)
            quad_turn = self._kappa_2 * cp.sum(cp.square(excess_turn))
            turn_penalty = linear_turn + quad_turn  # type: ignore[assignment]

        obj = cp.Maximize(
            ret_term - risk_term + entropy_term - conc_penalty - turn_penalty
        )

        cstr_list = [
            w >= 0,
            w <= self._w_max,
            cp.sum(w) == 1,
        ]
        constraints: list[cp.Constraint] = cstr_list  # type: ignore[assignment]
        if self._w_old is not None and not self._is_first:
            turnover_cstr = 0.5 * cp.sum(cp.abs(w - self._w_old)) <= self._tau_max
            constraints.append(turnover_cstr)  # type: ignore[arg-type]

        problem = cp.Problem(obj, constraints)

        for solver_name, solver_kwargs in _SOLVER_CHAIN:
            try:
                problem.solve(
                    solver=solver_name, **solver_kwargs,  # type: ignore[arg-type]
                )
                if (problem.status in ("optimal", "optimal_inaccurate")
                        and w.value is not None):
                    result = np.array(w.value).flatten()
                    if not np.any(np.isnan(result)):
                        return result
            except cp.SolverError:
                continue

        return None


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

    # Entropy term
    H = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps)

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
) -> tuple[np.ndarray, float, float, int]:
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

    :return w_final (np.ndarray): Optimized weights (n,)
    :return f_final (float): Final objective value
    :return H_final (float): Final entropy
    :return n_iters (int): Number of SCA iterations used
    """
    n = len(w_init)

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
        )

    f_current = obj_fn(w)

    for it in range(max_iter):
        # Compute entropy and gradient at current point
        H_current, grad_H = compute_entropy_and_gradient(
            w, B_prime, eigenvalues, entropy_eps,
        )

        # Solve reusing pre-compiled parametric problem
        w_star = subproblem.solve(grad_H)

        if w_star is None:
            break

        # Surrogate improvement
        f_surr_star = obj_fn(w_star)
        delta_surr = f_surr_star - f_current
        if delta_surr < 0:
            delta_surr = 0.0

        # Armijo backtracking
        eta = armijo_backtracking(
            w, w_star, f_current, max(delta_surr, tol),
            obj_fn, armijo_c, armijo_rho, armijo_max_iter,
        )

        # Update
        w_new = w + eta * (w_star - w)

        # Ensure feasibility
        w_new = np.clip(w_new, 0.0, w_max)
        w_sum = np.sum(w_new)
        if w_sum > 0:
            w_new = w_new / w_sum

        f_new = obj_fn(w_new)

        # Convergence check: |f(w^{t+1}) - f(w^t)| < tol (DVT: 1e-8)
        if abs(f_new - f_current) < tol:
            w = w_new
            f_current = f_new
            break

        w = w_new
        f_current = f_new

    H_final = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps)
    return w, f_current, H_final, it + 1


def multi_start_optimize(
    Sigma_assets: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray,
    alpha: float,
    n_starts: int = 5,
    seed: int = 42,
    **kwargs: float | np.ndarray | bool | None,
) -> tuple[np.ndarray, float, float]:
    """
    Multi-start SCA optimization with parallel execution.

    Starts:
    1. Equal-weight
    2. Minimum variance (approximate)
    3. Approximate ERC (inverse vol)
    4-5. Random (Dirichlet + projection)

    Pre-computes Cholesky once and shares across all starts.
    Runs starts in parallel via ThreadPoolExecutor (solvers release the GIL).

    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)
    :param alpha (float): Entropy weight
    :param n_starts (int): Number of starts
    :param seed (int): Random seed
    :param **kwargs: Additional parameters for sca_optimize

    :return w_best (np.ndarray): Best weights (n,)
    :return f_best (float): Best objective value
    :return H_best (float): Entropy at best solution
    """
    n = Sigma_assets.shape[0]
    rng = np.random.RandomState(seed)
    w_max_val = kwargs.get("w_max", 0.05)
    w_max = float(w_max_val) if w_max_val is not None else 0.05
    w_min_val = kwargs.get("w_min", 0.001)
    w_min = float(w_min_val) if w_min_val is not None else 0.001

    # Pre-compute Cholesky once for all starts
    L_sigma = _safe_cholesky(Sigma_assets)

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

    # 4-5. Random starts
    for _ in range(max(0, n_starts - 3)):
        n_active = rng.randint(30, min(300, n) + 1)
        active_idx = rng.choice(n, size=n_active, replace=False)
        w_rand = np.zeros(n)
        w_rand[active_idx] = rng.dirichlet(np.ones(n_active))
        starts.append(project_to_constraints(w_rand, w_max, w_min))

    # Filter kwargs for sca_optimize (remove w_min which is only for initialization)
    sca_kwargs = {k: v for k, v in kwargs.items() if k != "w_min"}

    active_starts = starts[:n_starts]

    def _run_start(w_init: np.ndarray) -> tuple[np.ndarray, float, float, int]:
        return sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=alpha,
            _L_sigma=L_sigma,
            **sca_kwargs,  # type: ignore[arg-type]
        )

    # Run SCA from each start
    best_w = active_starts[0]
    best_f = -np.inf
    best_H = 0.0

    if len(active_starts) <= 1:
        w_opt, f_opt, H_opt, _ = _run_start(active_starts[0])
        return w_opt, f_opt, H_opt

    # Parallel execution: solvers (MOSEK/CLARABEL/ECOS) release the GIL
    with ThreadPoolExecutor(max_workers=len(active_starts)) as executor:
        futures = [
            executor.submit(_run_start, w_init)
            for w_init in active_starts
        ]
        # Iterate in submission order to preserve determinism on ties
        for future in futures:
            w_opt, f_opt, H_opt, _ = future.result()
            if f_opt > best_f:
                best_w = w_opt
                best_f = f_opt
                best_H = H_opt

    return best_w, best_f, best_H
