"""
Cardinality enforcement for semi-continuous portfolio constraints.

Four strategies for enforcing w_i = 0 or w_i >= w_min:
  1. "sequential" — Original per-position entropy evaluation + SCA re-optimization
  2. "gradient"   — First-order Taylor approximation for elimination ranking (fast)
  3. "miqp"       — Single MOSEK mixed-integer QP (binary z_i for semi-continuity)
  4. "two_stage"  — DVT §4.7 decomposition: analytical y* in factor space + MIQP projection

Automatic fallback: two_stage -> miqp -> gradient -> sequential.

Reference: ISD Section MOD-008 — Sub-task 3, DVT Section 4.7 line 846.
"""

import logging
import warnings
from typing import Any

import numpy as np
import cvxpy as cp

from src.validation import assert_weights_sum_to_one, assert_weights_valid
from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.sca_solver import (
    _MI_SOLVER_CHAIN,
    _SOLVER_CHAIN,
    _safe_cholesky,
    has_mi_solver,
)

logger = logging.getLogger(__name__)

MIN_BATCH_SIZE = 3

_MIQP_TIME_LIMIT = 120.0

# Hard cap on binary variables for MI solver tractability
_MAX_BINARY_VARS = 150


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prescreen_active_stocks(
    w: np.ndarray,
    w_min: float,
    label: str = "miqp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-screen stocks for MI formulation: keep only significant positions.

    Uses w_min-aware threshold (w_min * 0.5) and a hard cap on binary
    variables (_MAX_BINARY_VARS) to guarantee MI solver tractability.

    :param w (np.ndarray): Continuous SCA solution (n,)
    :param w_min (float): Minimum active weight
    :param label (str): Strategy label for logging

    :return active_idx (np.ndarray): Indices of stocks to model with binary z_i
    :return fixed_idx (np.ndarray): Indices of stocks fixed at zero
    """
    n = len(w)
    threshold = w_min * 0.5

    # Step 1: filter by w_min-aware threshold
    candidate_mask = w >= threshold
    candidate_idx = np.where(candidate_mask)[0]

    # Step 2: hard cap — if too many, keep top by weight
    if len(candidate_idx) > _MAX_BINARY_VARS:
        weights_at_candidates = w[candidate_idx]
        top_order = np.argsort(-weights_at_candidates)[:_MAX_BINARY_VARS]
        candidate_idx = candidate_idx[top_order]

    active_set = set(candidate_idx.tolist())
    active_idx = np.array(sorted(active_set), dtype=np.intp)
    fixed_idx = np.array([i for i in range(n) if i not in active_set], dtype=np.intp)

    # BUG FIX: Validate no duplicate indices (diagnostic fix)
    assert len(active_idx) == len(set(active_idx)), (
        "Duplicate indices in active_set — pre-screening logic error"
    )

    pct_reduction = 100.0 * (1 - len(active_idx) / max(n, 1))
    logger.info(
        "    [%s] Pre-screening: %d/%d binary variables "
        "(%.0f%% reduction, threshold=%.4f)",
        label, len(active_idx), n, pct_reduction, threshold,
    )

    return active_idx, fixed_idx


def _reduce_sca_kwargs(
    kwargs: dict[str, Any],
    active: np.ndarray,
) -> dict[str, Any]:
    """
    Slice SCA kwargs to active stock indices only.

    Reduces position-dependent matrices (Sigma_assets, B_prime, w_old, mu)
    to the active sub-dimensions. Drops pre-computed Cholesky so
    sca_optimize recomputes it for the reduced matrix.

    :param kwargs (dict): Original SCA keyword arguments
    :param active (np.ndarray): Indices of active (non-eliminated) stocks

    :return reduced (dict): SCA kwargs for the reduced problem
    """
    reduced = dict(kwargs)
    ix = np.ix_(active, active)
    reduced["Sigma_assets"] = kwargs["Sigma_assets"][ix]
    reduced["B_prime"] = kwargs["B_prime"][active]

    # Force Cholesky recomputation on reduced matrix
    reduced["_L_sigma"] = None

    if kwargs.get("w_old") is not None:
        reduced["w_old"] = kwargs["w_old"][active]

    if kwargs.get("mu") is not None:
        reduced["mu"] = kwargs["mu"][active]

    if kwargs.get("D_eps") is not None:
        reduced["D_eps"] = kwargs["D_eps"][active]

    return reduced


def _sca_reoptimize(
    w: np.ndarray,
    n: int,
    eliminated: set[int],
    sca_solver_fn: Any,
    sca_kwargs: dict[str, Any],
) -> np.ndarray:
    """
    Re-optimize via SCA on the reduced active set.

    :param w (np.ndarray): Current weights (n,)
    :param n (int): Total number of stocks
    :param eliminated (set[int]): Indices of eliminated stocks
    :param sca_solver_fn (callable): SCA solver function
    :param sca_kwargs (dict): Arguments for sca_solver_fn

    :return w_new (np.ndarray): Re-optimized weights (n,)
    """
    active_indices = np.array(
        [i for i in range(n) if i not in eliminated and w[i] > 0],
    )
    if len(active_indices) < 2:
        return w

    try:
        reduced_kwargs = _reduce_sca_kwargs(sca_kwargs, active_indices)
        w_init_reduced = w[active_indices].copy()
        w_sum = np.sum(w_init_reduced)
        if w_sum > 0:
            w_init_reduced = w_init_reduced / w_sum

        w_opt_reduced, _, _, _ = sca_solver_fn(
            w_init=w_init_reduced, **reduced_kwargs,
        )
        assert_weights_sum_to_one(w_opt_reduced, "w_reopt")

        w_new = np.zeros(n)
        w_new[active_indices] = w_opt_reduced
        total_w = np.sum(w_new)
        if total_w > 0:
            w_new = w_new / total_w
        return w_new
    except Exception:
        return w


def _eliminate_batch(
    w: np.ndarray,
    costs: list[tuple[int, float]],
    eliminated: set[int],
) -> tuple[np.ndarray, int]:
    """
    Eliminate a batch of positions using adaptive batch sizing.

    :param w (np.ndarray): Current weights (n,) — modified in place
    :param costs (list): Sorted (index, cost) pairs, ascending cost
    :param eliminated (set[int]): Eliminated set — updated in place

    :return w (np.ndarray): Updated weights (renormalized)
    :return batch_size (int): Number of positions eliminated
    """
    n_free = sum(1 for _, c in costs if c <= 0)
    n_positive = len(costs) - n_free
    batch_size = n_free + max(MIN_BATCH_SIZE, n_positive // 2)
    batch_size = min(batch_size, len(costs))
    batch = costs[:batch_size]

    for idx, _ in batch:
        w[idx] = 0.0
        eliminated.add(idx)

    if len(batch) > 0:
        logger.info(
            "    Eliminated %d positions (total: %d), cost range: [%.6f, %.6f]",
            len(batch), len(eliminated), batch[0][1], batch[-1][1],
        )

    total_w = np.sum(w)
    if total_w > 0:
        w = w / total_w

    return w, len(batch)


# ---------------------------------------------------------------------------
# Strategy 1: Sequential (original)
# ---------------------------------------------------------------------------

def _enforce_sequential(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
    sca_solver_fn: Any,
    sca_kwargs: dict[str, Any],
    max_eliminations: int = 100,
    entropy_eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
) -> np.ndarray:
    """
    Original per-position entropy evaluation + SCA re-optimization.

    For each violating position, computes exact H(w^{-i}) by constructing
    a trial weight vector. O(|S_sub| x n x AU) per round.

    :param w (np.ndarray): Initial weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param sca_solver_fn (callable): SCA solver function
    :param sca_kwargs (dict): Arguments for sca_solver_fn
    :param max_eliminations (int): Maximum elimination rounds
    :param entropy_eps (float): Numerical stability
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,)

    :return w_final (np.ndarray): Weights with cardinality enforced
    """
    w = w.copy()
    n = len(w)
    eliminated: set[int] = set()

    for round_idx in range(max_eliminations):
        s_sub = [i for i in range(n)
                 if i not in eliminated and 0 < w[i] < w_min]

        if not s_sub:
            break

        logger.info("    [sequential] Round %d: %d violations",
                     round_idx + 1, len(s_sub))

        H_current = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                                          idio_weight=idio_weight)

        costs: list[tuple[int, float]] = []
        for i in s_sub:
            w_trial = w.copy()
            w_trial[i] = 0.0
            total = np.sum(w_trial)
            if total > 0:
                w_trial = w_trial / total
            H_trial = compute_entropy_only(
                w_trial, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                idio_weight=idio_weight,
            )
            delta_H = H_current - H_trial
            costs.append((i, float(delta_H)))

        costs.sort(key=lambda x: x[1])
        w, _ = _eliminate_batch(w, costs, eliminated)
        w = _sca_reoptimize(w, n, eliminated, sca_solver_fn, sca_kwargs)

    logger.info("    [sequential] Done: %d eliminations, %d active",
                 len(eliminated), int(np.sum(w > 1e-8)))
    return w


# ---------------------------------------------------------------------------
# Strategy 2: Gradient ranking (Piste 3 — quick win)
# ---------------------------------------------------------------------------

def _enforce_gradient(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
    sca_solver_fn: Any,
    sca_kwargs: dict[str, Any],
    max_eliminations: int = 100,
    entropy_eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
) -> np.ndarray:
    """
    Gradient-based cardinality enforcement using first-order Taylor approximation.

    Replaces per-position entropy evaluation with:
        delta_H_i ~ |dH/dw_i| * w_i

    One gradient computation per round instead of |S_sub| entropy evaluations.
    Complexity: O(n x AU) per round instead of O(|S_sub| x n x AU).

    :param w (np.ndarray): Initial weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param sca_solver_fn (callable): SCA solver function
    :param sca_kwargs (dict): Arguments for sca_solver_fn
    :param max_eliminations (int): Maximum elimination rounds
    :param entropy_eps (float): Numerical stability
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,)

    :return w_final (np.ndarray): Weights with cardinality enforced
    """
    w = w.copy()
    n = len(w)
    eliminated: set[int] = set()

    for round_idx in range(max_eliminations):
        s_sub = [i for i in range(n)
                 if i not in eliminated and 0 < w[i] < w_min]

        if not s_sub:
            break

        logger.info("    [gradient] Round %d: %d violations",
                     round_idx + 1, len(s_sub))

        # Single gradient computation replaces |S_sub| entropy evaluations
        _, grad_H = compute_entropy_and_gradient(
            w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
            idio_weight=idio_weight,
        )

        # First-order Taylor: ΔH_i ≈ |∂H/∂w_i| × w_i
        costs: list[tuple[int, float]] = [
            (i, abs(float(grad_H[i])) * w[i]) for i in s_sub
        ]

        costs.sort(key=lambda x: x[1])
        w, _ = _eliminate_batch(w, costs, eliminated)
        w = _sca_reoptimize(w, n, eliminated, sca_solver_fn, sca_kwargs)

    logger.info("    [gradient] Done: %d eliminations, %d active",
                 len(eliminated), int(np.sum(w > 1e-8)))
    return w


# ---------------------------------------------------------------------------
# Strategy 3: MIQP (Piste 2 — single MOSEK solve)
# ---------------------------------------------------------------------------

def _enforce_miqp(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
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
    L_sigma: np.ndarray | None = None,
    entropy_eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """
    MIQP cardinality enforcement in a single solve.

    Linearizes H at the continuous SCA solution w and solves:
        max  mu^T w  +  alpha * grad_H^T w  -  lambda * ||L^T w||^2
             - phi * P_conc  -  P_turn
        s.t. w_min * z_i <= w_i <= w_max * z_i,  z_i in {0,1}
             sum(w) = 1, turnover cap

    Pre-screening: only creates binary z_i for stocks with w_i > 0 in the
    continuous SCA solution. Stocks at zero are fixed, reducing the MIQP
    complexity from O(n) to O(n_active) binary variables.

    Requires a mixed-integer-capable solver (MOSEK).

    :param w (np.ndarray): Continuous SCA solution (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param alpha (float): Entropy weight
    :param lambda_risk (float): Risk aversion
    :param phi (float): Concentration penalty weight
    :param w_bar (float): Concentration threshold
    :param w_max (float): Maximum weight per stock
    :param w_old (np.ndarray | None): Previous weights
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover penalty threshold
    :param tau_max (float): Maximum one-way turnover
    :param is_first (bool): First rebalancing flag
    :param L_sigma (np.ndarray | None): Pre-computed Cholesky factor
    :param entropy_eps (float): Numerical stability
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,)
    :param mu (np.ndarray | None): Return signal (n,) — momentum-weighted z-scores

    :return w_miqp (np.ndarray): MIQP solution (n,)
    """
    n = len(w)
    L = L_sigma if L_sigma is not None else _safe_cholesky(Sigma_assets)

    _, grad_H = compute_entropy_and_gradient(
        w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
        idio_weight=idio_weight,
    )

    # Pre-screening: w_min-aware threshold + hard cap
    active_idx, fixed_idx = _prescreen_active_stocks(w, w_min, "miqp")
    n_active = len(active_idx)

    # Decision variables
    w_var = cp.Variable(n)
    z_active = cp.Variable(n_active, boolean=True)  # type: ignore[call-overload]

    # Objective: same structure as SCA sub-problem but with fixed gradient.
    # Auxiliary variables for DCP compliance (cp.square requires affine arg).
    ret_term: cp.Expression | float = mu @ w_var if mu is not None else 0.0
    entropy_term = alpha * (grad_H @ w_var)
    risk_term = lambda_risk * cp.sum_squares(L.T @ w_var)

    conc_penalty: cp.Expression = cp.Constant(0.0)
    conc_constraints: list[cp.Constraint] = []
    if phi > 0:
        t_conc = cp.Variable(n, nonneg=True)
        conc_constraints = [t_conc >= w_var - w_bar]  # type: ignore[list-item]
        conc_penalty = phi * cp.sum_squares(t_conc)

    turn_penalty: cp.Expression = cp.Constant(0.0)
    turn_constraints: list[cp.Constraint] = []
    if w_old is not None and not is_first:
        delta_w = cp.abs(w_var - w_old)
        linear_turn = kappa_1 * 0.5 * cp.sum(delta_w)
        t_turn = cp.Variable(n, nonneg=True)
        turn_constraints = [t_turn >= delta_w - delta_bar]  # type: ignore[list-item]
        quad_turn = kappa_2 * cp.sum_squares(t_turn)
        turn_penalty = linear_turn + quad_turn  # type: ignore[assignment]

    obj = cp.Maximize(ret_term + entropy_term - risk_term - conc_penalty - turn_penalty)

    # Semi-continuous constraints: binary only for active stocks.
    # Uses w_max (hard cap) — w_bar is only for soft concentration penalty.
    active_list = active_idx.tolist()
    fixed_list = fixed_idx.tolist()
    cstr_list = [
        w_var[active_list] >= w_min * z_active,
        w_var[active_list] <= w_max * z_active,
        cp.sum(w_var) == 1,
    ]
    if len(fixed_list) > 0:
        cstr_list.append(w_var[fixed_list] == 0)
    constraints: list[cp.Constraint] = cstr_list  # type: ignore[assignment]
    constraints.extend(conc_constraints)
    constraints.extend(turn_constraints)

    if w_old is not None and not is_first:
        turnover_cstr = 0.5 * cp.sum(cp.abs(w_var - w_old)) <= tau_max
        constraints.append(turnover_cstr)  # type: ignore[arg-type]

    problem = cp.Problem(obj, constraints)

    # Warm-start from continuous solution
    w_var.value = w
    z_active.value = (w[active_idx] >= w_min * 0.5).astype(float)

    # Solve via MI solver chain
    for solver_name, solver_kwargs in _MI_SOLVER_CHAIN:
        mi_kwargs = dict(solver_kwargs)
        if solver_name == "MOSEK":
            mi_kwargs["mosek_params"] = {
                "MSK_DPAR_MIO_MAX_TIME": _MIQP_TIME_LIMIT,
            }
        try:
            problem.solve(
                solver=solver_name, **mi_kwargs,  # type: ignore[arg-type]
            )
            if (problem.status in ("optimal", "optimal_inaccurate")
                    and w_var.value is not None):
                result = np.array(w_var.value).flatten()
                if not np.any(np.isnan(result)):
                    # Clean numerical residuals from MI solver
                    result[result < w_min * 0.5] = 0.0
                    result = np.clip(result, 0.0, w_max)
                    total = np.sum(result)
                    if total > 0:
                        result = result / total
                    logger.info("    [miqp] Solved via %s: %d active positions",
                                 solver_name, int(np.sum(result > 1e-8)))
                    return result
        except cp.SolverError:
            continue

    raise RuntimeError("MIQP solve failed on all MI-capable solvers")


# ---------------------------------------------------------------------------
# Strategy 4: Two-stage decomposition (Piste 1 — DVT §4.7)
# ---------------------------------------------------------------------------

def _enforce_two_stage(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
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
    L_sigma: np.ndarray | None = None,
    entropy_eps: float = 1e-30,
    use_mi: bool = True,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """
    Two-stage decomposition for cardinality enforcement (DVT Section 4.7).

    Stage 1: Compute target factor exposure y* analytically.
        Equal risk contributions: y*_k = c / sqrt(lambda_k), with sign from
        the continuous SCA solution. Targets H = ln(AU).

    Stage 2: Project to asset space via MIQP (or continuous QP):
        min_w ||B'^T w - y*||^2 + lambda_risk * ||L^T w||^2
              - mu^T w + phi * P_conc + P_turn
        s.t. semi-continuous constraints (MIQP) or w >= 0 (continuous)

    :param w (np.ndarray): Continuous SCA solution (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param Sigma_assets (np.ndarray): Asset covariance (n, n)
    :param alpha (float): Entropy weight (unused directly, included for API)
    :param lambda_risk (float): Risk aversion (regularization in Stage 2)
    :param phi (float): Concentration penalty weight
    :param w_bar (float): Concentration threshold
    :param w_max (float): Maximum weight per stock
    :param w_old (np.ndarray | None): Previous weights
    :param kappa_1 (float): Linear turnover penalty
    :param kappa_2 (float): Quadratic turnover penalty
    :param delta_bar (float): Turnover penalty threshold
    :param tau_max (float): Maximum one-way turnover
    :param is_first (bool): First rebalancing flag
    :param L_sigma (np.ndarray | None): Pre-computed Cholesky factor
    :param entropy_eps (float): Numerical stability
    :param use_mi (bool): Use MI variables for exact semi-continuous constraint
    :param mu (np.ndarray | None): Return signal (n,) — momentum-weighted z-scores

    :return w_final (np.ndarray): Two-stage solution (n,)
    """
    n = len(w)
    au = len(eigenvalues)
    L = L_sigma if L_sigma is not None else _safe_cholesky(Sigma_assets)

    # -----------------------------------------------------------------------
    # Stage 1: Target factor exposure from continuous SCA solution
    # -----------------------------------------------------------------------
    # Use y* = B'^T w_sca (the actual SCA solution's factor exposure) as
    # the tracking target.  The SCA solution balances entropy, risk,
    # concentration, and turnover — using the analytical equal-risk-
    # contribution target y*_k ∝ 1/√λ_k ignores all terms except entropy
    # and produces a target that is inconsistent with the optimizer's actual
    # trade-offs.
    y_star = B_prime.T @ w  # (AU,)

    logger.info("    [two_stage] Stage 1: y* = B'^T w_sca, ||y*||=%.4f",
                 float(np.linalg.norm(y_star)))

    # -----------------------------------------------------------------------
    # Stage 2: Project to asset space via QP/MIQP
    # -----------------------------------------------------------------------
    w_var = cp.Variable(n)

    # Tracking objective: minimize ||B'^T w - y*||^2 + regularization - mu^T w
    # Auxiliary variables for DCP compliance (cp.square requires affine arg).
    tracking_error = cp.sum_squares(B_prime.T @ w_var - y_star)
    risk_reg = lambda_risk * cp.sum_squares(L.T @ w_var)
    ret_term: cp.Expression | float = mu @ w_var if mu is not None else 0.0

    conc_penalty: cp.Expression = cp.Constant(0.0)
    conc_constraints: list[cp.Constraint] = []
    if phi > 0:
        t_conc = cp.Variable(n, nonneg=True)
        conc_constraints = [t_conc >= w_var - w_bar]  # type: ignore[list-item]
        conc_penalty = phi * cp.sum_squares(t_conc)

    turn_penalty: cp.Expression = cp.Constant(0.0)
    turn_constraints: list[cp.Constraint] = []
    if w_old is not None and not is_first:
        delta_w = cp.abs(w_var - w_old)
        linear_turn = kappa_1 * 0.5 * cp.sum(delta_w)
        t_turn = cp.Variable(n, nonneg=True)
        turn_constraints = [t_turn >= delta_w - delta_bar]  # type: ignore[list-item]
        quad_turn = kappa_2 * cp.sum_squares(t_turn)
        turn_penalty = linear_turn + quad_turn  # type: ignore[assignment]

    obj = cp.Minimize(
        tracking_error + risk_reg - ret_term + conc_penalty + turn_penalty
    )

    # Constraints (with pre-screening for MI mode).
    # Uses w_max (hard cap) — w_bar is only for soft concentration penalty.
    if use_mi:
        # Pre-screening: w_min-aware threshold + hard cap
        active_idx, fixed_idx = _prescreen_active_stocks(w, w_min, "two_stage")
        n_active = len(active_idx)

        z_active = cp.Variable(n_active, boolean=True)  # type: ignore[call-overload]
        active_list = active_idx.tolist()
        fixed_list = fixed_idx.tolist()
        cstr_list = [
            w_var[active_list] >= w_min * z_active,
            w_var[active_list] <= w_max * z_active,
            cp.sum(w_var) == 1,
        ]
        if len(fixed_list) > 0:
            cstr_list.append(w_var[fixed_list] == 0)
    else:
        cstr_list = [
            w_var >= 0,
            w_var <= w_max,
            cp.sum(w_var) == 1,
        ]

    constraints: list[cp.Constraint] = cstr_list  # type: ignore[assignment]
    constraints.extend(conc_constraints)
    constraints.extend(turn_constraints)
    if w_old is not None and not is_first:
        turnover_cstr = 0.5 * cp.sum(cp.abs(w_var - w_old)) <= tau_max
        constraints.append(turnover_cstr)  # type: ignore[arg-type]

    problem = cp.Problem(obj, constraints)

    # Warm-start
    w_var.value = w
    if use_mi:
        z_active.value = (w[active_idx] >= w_min * 0.5).astype(float)  # type: ignore[possibly-undefined]

    # Solve
    solver_chain = _MI_SOLVER_CHAIN if use_mi else _SOLVER_CHAIN
    for solver_name, solver_kwargs in solver_chain:
        solve_kwargs = dict(solver_kwargs)
        if use_mi and solver_name == "MOSEK":
            solve_kwargs["mosek_params"] = {
                "MSK_DPAR_MIO_MAX_TIME": _MIQP_TIME_LIMIT,
            }
        try:
            problem.solve(
                solver=solver_name, **solve_kwargs,  # type: ignore[arg-type]
            )
            if (problem.status in ("optimal", "optimal_inaccurate")
                    and w_var.value is not None):
                result = np.array(w_var.value).flatten()
                if not np.any(np.isnan(result)):
                    # Clean numerical residuals from solver
                    if use_mi:
                        result[result < w_min * 0.5] = 0.0
                    result = np.clip(result, 0.0, w_max)
                    total = np.sum(result)
                    if total > 0:
                        result = result / total

                    H_result = compute_entropy_only(
                        result, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                        idio_weight=idio_weight,
                    )
                    H_sca = compute_entropy_only(
                        w, B_prime, eigenvalues, entropy_eps, D_eps=D_eps,
                        idio_weight=idio_weight,
                    )
                    if H_result < H_sca - 1e-6:
                        warnings.warn(
                            f"Two-stage entropy {H_result:.4f} < SCA entropy {H_sca:.4f}",
                            stacklevel=2,
                        )
                    logger.info(
                        "    [two_stage] Stage 2 solved via %s "
                        "(MI=%s): %d active, H=%.4f",
                        solver_name, use_mi,
                        int(np.sum(result > 1e-8)), H_result,
                    )
                    return result
        except cp.SolverError:
            continue

    raise RuntimeError("Two-stage QP solve failed on all solvers")


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def enforce_cardinality(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
    sca_solver_fn: Any,
    sca_kwargs: dict[str, Any],
    max_eliminations: int = 100,
    entropy_eps: float = 1e-30,
    method: str = "auto",
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
) -> np.ndarray:
    """
    Enforce semi-continuous constraint: w_i = 0 or w_i >= w_min.

    Dispatches to the selected strategy with automatic fallback:
        two_stage -> miqp -> gradient -> sequential

    :param w (np.ndarray): Initial weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param sca_solver_fn (callable): SCA solver function
    :param sca_kwargs (dict): Arguments for sca_solver_fn
    :param max_eliminations (int): Maximum elimination rounds
    :param entropy_eps (float): Numerical stability
    :param method (str): Strategy — "auto", "sequential", "gradient",
        "miqp", "two_stage"
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,) for
        entropy computation including idiosyncratic risk contributions.
    :param idio_weight (float): Weight for idiosyncratic entropy layer (0-1).

    :return w_final (np.ndarray): Weights with cardinality enforced
    """
    # Resolve "auto" to best available method
    if method == "auto":
        method = "two_stage" if has_mi_solver() else "gradient"
        logger.info("    Cardinality method auto-resolved to '%s'", method)

    # Extract MIQP parameters from sca_kwargs (needed for miqp + two_stage)
    mi_params = {
        "Sigma_assets": sca_kwargs["Sigma_assets"],
        "alpha": sca_kwargs.get("alpha", 1.0),
        "lambda_risk": sca_kwargs.get("lambda_risk", 1.0),
        "phi": sca_kwargs.get("phi", 25.0),
        "w_bar": sca_kwargs.get("w_bar", 0.03),
        "w_max": sca_kwargs.get("w_max", 0.05),
        "w_old": sca_kwargs.get("w_old"),
        "kappa_1": sca_kwargs.get("kappa_1", 0.1),
        "kappa_2": sca_kwargs.get("kappa_2", 7.5),
        "delta_bar": sca_kwargs.get("delta_bar", 0.01),
        "tau_max": sca_kwargs.get("tau_max", 0.30),
        "is_first": sca_kwargs.get("is_first", True),
        "L_sigma": sca_kwargs.get("_L_sigma"),
        "mu": sca_kwargs.get("mu"),
    }

    # ------------------------------------------------------------------
    # Two-stage (DVT §4.7)
    # ------------------------------------------------------------------
    if method == "two_stage":
        mi_avail = has_mi_solver()
        if mi_avail:
            try:
                return _enforce_two_stage(
                    w, B_prime, eigenvalues, w_min,
                    use_mi=True, entropy_eps=entropy_eps, D_eps=D_eps,
                    idio_weight=idio_weight, **mi_params,
                )
            except Exception as exc:
                logger.warning(
                    "    Two-stage MIQP failed (%s), trying continuous QP", exc,
                )
        # Continuous QP fallback + gradient post-processing
        try:
            w_qp = _enforce_two_stage(
                w, B_prime, eigenvalues, w_min,
                use_mi=False, entropy_eps=entropy_eps, D_eps=D_eps,
                idio_weight=idio_weight, **mi_params,
            )
            # Post-process: continuous QP may leave violations
            return _enforce_gradient(
                w_qp, B_prime, eigenvalues, w_min,
                sca_solver_fn, sca_kwargs, max_eliminations, entropy_eps,
                D_eps=D_eps, idio_weight=idio_weight,
            )
        except Exception as exc:
            logger.warning(
                "    Two-stage continuous failed (%s), falling back to gradient",
                exc,
            )
            method = "gradient"

    # ------------------------------------------------------------------
    # MIQP (single MOSEK solve)
    # ------------------------------------------------------------------
    if method == "miqp":
        if has_mi_solver():
            try:
                return _enforce_miqp(
                    w, B_prime, eigenvalues, w_min,
                    entropy_eps=entropy_eps, D_eps=D_eps,
                    idio_weight=idio_weight, **mi_params,
                )
            except Exception as exc:
                logger.warning(
                    "    MIQP failed (%s), falling back to gradient", exc,
                )
                method = "gradient"
        else:
            logger.info("    No MI solver available, falling back to gradient")
            method = "gradient"

    # ------------------------------------------------------------------
    # Gradient (first-order Taylor approximation)
    # ------------------------------------------------------------------
    if method == "gradient":
        return _enforce_gradient(
            w, B_prime, eigenvalues, w_min,
            sca_solver_fn, sca_kwargs, max_eliminations, entropy_eps,
            D_eps=D_eps, idio_weight=idio_weight,
        )

    # ------------------------------------------------------------------
    # Sequential (original)
    # ------------------------------------------------------------------
    return _enforce_sequential(
        w, B_prime, eigenvalues, w_min,
        sca_solver_fn, sca_kwargs, max_eliminations, entropy_eps,
        D_eps=D_eps, idio_weight=idio_weight,
    )
