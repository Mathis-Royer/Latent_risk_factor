"""
Sequential Convex Approximation (SCA) solver with Armijo backtracking.

Multi-start optimization (M=5 initializations):
1. Equal-weight
2. Minimum variance (QP)
3. Approximate ERC (Spinu)
4-5. Random (Dirichlet + projection)

SCA iterations linearize H(w) and solve a convex sub-problem via CVXPY.

Reference: ISD Section MOD-008 — Sub-task 2.
"""

import numpy as np
import cvxpy as cp

from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.constraints import (
    concentration_penalty,
    turnover_penalty,
    project_to_constraints,
)


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

    :return f (float): Objective value (to maximize)
    """
    n = len(w)

    # Return term (μ=0 by default)
    ret_term = float(w @ mu) if mu is not None else 0.0

    # Risk term
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


def solve_sca_subproblem(
    w_current: np.ndarray,
    grad_H: np.ndarray,
    Sigma_assets: np.ndarray,
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
) -> np.ndarray | None:
    """
    Solve the convex sub-problem using CVXPY.

    max w^T μ - λ w^T Σ w + α ∇H^T w - φ P_conc(w) - P_turn(w)
    s.t. w_i ∈ [0, w_max], 1^T w = 1, turnover ≤ τ_max

    :param w_current (np.ndarray): Current iterate (n,)
    :param grad_H (np.ndarray): Entropy gradient at w_current (n,)
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
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

    :return w_star (np.ndarray | None): Optimal weights, or None if infeasible
    """
    n = len(w_current)
    w = cp.Variable(n)

    # Return term
    if mu is not None:
        ret_term = mu @ w
    else:
        ret_term = 0.0

    # Risk term (quadratic)
    risk_term = lambda_risk * cp.quad_form(w, Sigma_assets)

    # Linearized entropy: α · ∇H^T w
    entropy_term = alpha * (grad_H @ w)

    # Concentration penalty (convex quadratic)
    excess = cp.maximum(0, w - w_bar)
    conc_penalty = phi * cp.sum_squares(excess)

    # Turnover penalty
    turn_penalty: cp.Expression = cp.Constant(0.0)
    if w_old is not None and not is_first:
        delta_w = cp.abs(w - w_old)
        linear_turn = kappa_1 * 0.5 * cp.sum(delta_w)
        excess_turn = cp.maximum(0, delta_w - delta_bar)
        quad_turn = kappa_2 * cp.sum_squares(excess_turn)
        turn_penalty = linear_turn + quad_turn

    # Objective (maximize)
    obj = cp.Maximize(ret_term - risk_term + entropy_term - conc_penalty - turn_penalty)

    # Constraints
    constraints = [
        w >= 0,
        w <= w_max,
        cp.sum(w) == 1,
    ]

    # Turnover hard constraint
    if w_old is not None and not is_first:
        constraints.append(0.5 * cp.sum(cp.abs(w - w_old)) <= tau_max)

    # Solve
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, max_iters=500)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    # Fallback solver
    try:
        prob.solve(solver=cp.SCS, warm_start=True, max_iters=5000)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    return None


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
) -> tuple[np.ndarray, float, float, int]:
    """
    SCA optimization from a single starting point.

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
    :param tol (float): Convergence tolerance
    :param armijo_c (float): Armijo constant
    :param armijo_rho (float): Armijo backtracking factor
    :param armijo_max_iter (int): Max Armijo steps
    :param entropy_eps (float): Numerical stability

    :return w_final (np.ndarray): Optimized weights (n,)
    :return f_final (float): Final objective value
    :return H_final (float): Final entropy
    :return n_iters (int): Number of SCA iterations used
    """
    w = w_init.copy()

    def obj_fn(w_eval: np.ndarray) -> float:
        return objective_function(
            w_eval, Sigma_assets, B_prime, eigenvalues, alpha,
            lambda_risk, phi, w_bar, w_old, kappa_1, kappa_2,
            delta_bar, is_first, mu, entropy_eps,
        )

    f_current = obj_fn(w)

    for it in range(max_iter):
        # Compute entropy and gradient at current point
        H_current, grad_H = compute_entropy_and_gradient(
            w, B_prime, eigenvalues, entropy_eps,
        )

        # Solve convex sub-problem
        w_star = solve_sca_subproblem(
            w, grad_H, Sigma_assets, alpha, lambda_risk, phi, w_bar,
            w_max, w_old, kappa_1, kappa_2, delta_bar, tau_max,
            is_first, mu,
        )

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

        # Convergence check: |f(w^{t+1}) - f(w^t)| < tol
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
    Multi-start SCA optimization.

    Starts:
    1. Equal-weight
    2. Minimum variance (approximate)
    3. Approximate ERC (inverse vol)
    4-5. Random (Dirichlet + projection)

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

    # Run SCA from each start
    best_w = starts[0]
    best_f = -np.inf
    best_H = 0.0

    for w_init in starts[:n_starts]:
        w_opt, f_opt, H_opt, _ = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=alpha,
            **sca_kwargs,  # type: ignore[arg-type]
        )

        if f_opt > best_f:
            best_w = w_opt
            best_f = f_opt
            best_H = H_opt

    return best_w, best_f, best_H
