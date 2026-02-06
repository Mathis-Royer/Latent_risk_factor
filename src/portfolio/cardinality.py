"""
Sequential entropy-aware cardinality enforcement.

Eliminates semi-continuous violations (0 < w_i < w_min) by removing
the position with the lowest entropy cost, then re-optimizing.

Convergence guarantee: active set strictly decreases at each iteration.

Reference: ISD Section MOD-008 — Sub-task 3.
"""

from typing import Any

import numpy as np

from src.portfolio.entropy import compute_entropy_only


def enforce_cardinality(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    w_min: float,
    sca_solver_fn: Any,
    sca_kwargs: dict[str, Any],
    max_eliminations: int = 100,
    entropy_eps: float = 1e-30,
) -> np.ndarray:
    """
    Enforce semi-continuous constraint: w_i = 0 or w_i ≥ w_min.

    Repeat:
    1. S_sub = {i : 0 < w_i < w_min}. If empty → stop.
    2. For each i ∈ S_sub: ΔH_i = H(w) - H(w^(-i))
    3. Eliminate i* = argmin ΔH_i (least costly)
    4. Re-optimize via SCA on reduced active set
    5. Return to step 1

    :param w (np.ndarray): Initial weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param w_min (float): Minimum active weight
    :param sca_solver_fn (callable): SCA solver function
    :param sca_kwargs (dict): Arguments for sca_solver_fn
    :param max_eliminations (int): Maximum elimination rounds
    :param entropy_eps (float): Numerical stability

    :return w_final (np.ndarray): Weights with cardinality enforced
    """
    w = w.copy()
    eliminated: set[int] = set()

    for _ in range(max_eliminations):
        # Find semi-continuous violations
        s_sub = [i for i in range(len(w))
                 if i not in eliminated and 0 < w[i] < w_min]

        if not s_sub:
            break

        # Compute entropy cost of eliminating each violating position
        H_current = compute_entropy_only(w, B_prime, eigenvalues, entropy_eps)

        min_cost = float("inf")
        elim_idx = s_sub[0]

        for i in s_sub:
            w_trial = w.copy()
            w_trial[i] = 0.0
            # Renormalize
            total = np.sum(w_trial)
            if total > 0:
                w_trial = w_trial / total
            H_trial = compute_entropy_only(w_trial, B_prime, eigenvalues, entropy_eps)
            delta_H = H_current - H_trial

            if delta_H < min_cost:
                min_cost = delta_H
                elim_idx = i

        # Eliminate the least costly position
        w[elim_idx] = 0.0
        eliminated.add(elim_idx)

        # Renormalize
        total = np.sum(w)
        if total > 0:
            w = w / total

        # Re-optimize via SCA on the remaining active set
        active_mask = w > 0
        if active_mask.sum() < 2:
            break

        # Re-run SCA with eliminated positions fixed at zero
        try:
            w_opt, _, _, _ = sca_solver_fn(w_init=w, **sca_kwargs)
            # Zero out eliminated positions
            w_opt[list(eliminated)] = 0.0
            total = np.sum(w_opt)
            if total > 0:
                w_opt = w_opt / total
            w = w_opt
        except Exception:
            # If SCA fails, keep the current solution
            pass

    return w
