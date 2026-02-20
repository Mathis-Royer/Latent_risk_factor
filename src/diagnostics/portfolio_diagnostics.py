"""
Portfolio optimization diagnostics: budget conservation, gradient analysis, frontier quality.

Implements diagnostics from SCA Portfolio Book (2024) and MATLAB optimization:
- Budget conservation verification (Σĉ_k = 1)
- Gradient norm balance (entropy vs risk)
- Step size variability analysis
- Objective monotonicity checking
- Frontier anomaly detection

Reference: ISD diagnostic gaps C.19-C.29.
"""

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# C.19: Budget conservation check
# ---------------------------------------------------------------------------

def verify_budget_conservation(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Verify that factor risk contributions sum to 1 (budget conservation).

    For a valid risk decomposition:
    - Σ_k ĉ_k = 1.0 (systematic layer)
    - Σ_i ĉ^ε_i = 1.0 (idiosyncratic layer, if applicable)

    Violations indicate numerical issues in the entropy computation.

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,)
    :param tol (float): Tolerance for sum = 1 check

    :return analysis (dict): Conservation check results
    """
    if w.size == 0 or B_prime.size == 0:
        return {"available": False, "reason": "empty inputs"}

    # Systematic risk contributions
    beta_prime = B_prime.T @ w  # (AU,)
    c_sys = (beta_prime ** 2) * eigenvalues  # (AU,)
    C_sys = float(np.sum(c_sys))

    if C_sys > 1e-15:
        c_hat_sys = c_sys / C_sys
        sum_c_hat_sys = float(np.sum(c_hat_sys))
        sys_conserved = abs(sum_c_hat_sys - 1.0) < tol
    else:
        c_hat_sys = np.zeros_like(c_sys)
        sum_c_hat_sys = 0.0
        sys_conserved = True  # Trivially conserved if zero

    # Idiosyncratic risk contributions (if applicable)
    if D_eps is not None:
        c_idio = (w ** 2) * D_eps
        C_idio = float(np.sum(c_idio))

        if C_idio > 1e-15:
            c_hat_idio = c_idio / C_idio
            sum_c_hat_idio = float(np.sum(c_hat_idio))
            idio_conserved = abs(sum_c_hat_idio - 1.0) < tol
        else:
            c_hat_idio = np.zeros_like(c_idio)
            sum_c_hat_idio = 0.0
            idio_conserved = True
    else:
        c_hat_idio = None
        sum_c_hat_idio = None
        idio_conserved = True
        C_idio = 0.0

    # Overall conservation
    all_conserved = sys_conserved and idio_conserved

    return {
        "available": True,
        "systematic_conserved": sys_conserved,
        "idiosyncratic_conserved": idio_conserved,
        "all_conserved": all_conserved,
        # Systematic details
        "C_sys": C_sys,
        "sum_c_hat_sys": sum_c_hat_sys,
        "sys_deviation": abs(sum_c_hat_sys - 1.0),
        "c_hat_sys": c_hat_sys.tolist()[:10],  # First 10
        # Idiosyncratic details
        "C_idio": C_idio,
        "sum_c_hat_idio": sum_c_hat_idio,
        "idio_deviation": abs(sum_c_hat_idio - 1.0) if sum_c_hat_idio is not None else 0.0,
        "tolerance": tol,
    }


# ---------------------------------------------------------------------------
# C.20: Gradient norm balance
# ---------------------------------------------------------------------------

def compute_gradient_balance(
    grad_entropy: np.ndarray,
    grad_variance: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """
    Compute balance between entropy and variance gradient contributions.

    In the objective: max H - α * Var
    The gradients should be commensurate for effective optimization.

    Ratio = ||∇H|| / ||α * ∇Var||
    - Ratio >> 1: Entropy dominates (may under-weight risk)
    - Ratio << 1: Variance dominates (may sacrifice diversification)
    - Ratio ≈ 1: Balanced optimization

    :param grad_entropy (np.ndarray): Gradient of entropy (n,)
    :param grad_variance (np.ndarray): Gradient of variance (n,)
    :param alpha (float): Risk aversion parameter

    :return analysis (dict): Gradient balance metrics
    """
    if grad_entropy.size == 0 or grad_variance.size == 0:
        return {"available": False, "reason": "empty gradients"}

    norm_entropy = float(np.linalg.norm(grad_entropy))
    norm_variance = float(np.linalg.norm(grad_variance))
    norm_alpha_var = float(alpha * norm_variance)

    if norm_alpha_var > 1e-15:
        ratio = norm_entropy / norm_alpha_var
    else:
        ratio = float("inf") if norm_entropy > 0 else 0.0

    # Cosine similarity (alignment of gradients)
    if norm_entropy > 1e-15 and norm_variance > 1e-15:
        cos_sim = float(np.dot(grad_entropy, grad_variance) / (norm_entropy * norm_variance))
    else:
        cos_sim = 0.0

    # Classification
    if ratio > 10.0:
        balance_status = "entropy_dominated"
    elif ratio < 0.1:
        balance_status = "variance_dominated"
    else:
        balance_status = "balanced"

    return {
        "available": True,
        "norm_grad_entropy": norm_entropy,
        "norm_grad_variance": norm_variance,
        "norm_alpha_variance": norm_alpha_var,
        "alpha": alpha,
        "ratio": ratio,
        "cosine_similarity": cos_sim,
        "balance_status": balance_status,
        # Per-dimension statistics
        "grad_entropy_mean": float(np.mean(grad_entropy)),
        "grad_entropy_std": float(np.std(grad_entropy)),
        "grad_variance_mean": float(np.mean(grad_variance)),
        "grad_variance_std": float(np.std(grad_variance)),
    }


# ---------------------------------------------------------------------------
# C.21: Two-layer weight balance verification
# ---------------------------------------------------------------------------

def verify_two_layer_balance(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    D_eps: np.ndarray,
    idio_weight: float,
) -> dict[str, Any]:
    """
    Verify that two-layer entropy weighting is properly balanced.

    With idio_weight = 0.2:
    - 80% of entropy gradient from systematic factors
    - 20% from idiosyncratic

    Checks if actual risk contributions match intended weighting.

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)
    :param idio_weight (float): Weight for idiosyncratic layer (0-1)

    :return analysis (dict): Two-layer balance verification
    """
    if w.size == 0:
        return {"available": False, "reason": "empty weights"}

    # Systematic risk
    beta_prime = B_prime.T @ w
    C_sys = float(np.sum((beta_prime ** 2) * eigenvalues))

    # Idiosyncratic risk
    C_idio = float(np.sum((w ** 2) * D_eps))

    # Total risk
    C_total = C_sys + C_idio

    if C_total > 1e-15:
        actual_sys_frac = C_sys / C_total
        actual_idio_frac = C_idio / C_total
    else:
        actual_sys_frac = 0.0
        actual_idio_frac = 0.0

    # Expected fractions based on idio_weight
    expected_sys_frac = 1.0 - idio_weight
    expected_idio_frac = idio_weight

    # Deviation from expected
    sys_deviation = actual_sys_frac - expected_sys_frac
    idio_deviation = actual_idio_frac - expected_idio_frac

    # Balance check (within 20% of target)
    is_balanced = abs(idio_deviation) < 0.2

    return {
        "available": True,
        "C_sys": C_sys,
        "C_idio": C_idio,
        "C_total": C_total,
        "actual_sys_frac": actual_sys_frac,
        "actual_idio_frac": actual_idio_frac,
        "expected_sys_frac": expected_sys_frac,
        "expected_idio_frac": expected_idio_frac,
        "sys_deviation": sys_deviation,
        "idio_deviation": idio_deviation,
        "is_balanced": is_balanced,
        "idio_weight": idio_weight,
    }


# ---------------------------------------------------------------------------
# C.22: Step size variability analysis
# ---------------------------------------------------------------------------

def analyze_step_size_trajectory(
    step_sizes: list[float],
) -> dict[str, Any]:
    """
    Analyze SCA step size trajectory for optimization quality.

    Healthy optimization:
    - Step sizes should be relatively stable
    - No extreme oscillations
    - Gradual decay towards convergence

    :param step_sizes (list[float]): Step sizes per iteration

    :return analysis (dict): Step size statistics and quality indicators
    """
    if not step_sizes:
        return {"available": False, "reason": "no step sizes"}

    n_iters = len(step_sizes)
    steps = np.array(step_sizes)

    # Basic statistics
    mean_step = float(np.mean(steps))
    std_step = float(np.std(steps))
    min_step = float(np.min(steps))
    max_step = float(np.max(steps))

    # Coefficient of variation (std / mean)
    cv = std_step / max(mean_step, 1e-10)

    # Variability indicator
    if cv < 0.5:
        variability = "low"
    elif cv < 1.0:
        variability = "moderate"
    else:
        variability = "high"

    # Trend analysis
    if n_iters >= 3:
        x = np.arange(n_iters)
        slope, _ = np.polyfit(x, steps, 1)
        if slope < -0.01 * mean_step:
            trend = "decreasing"
        elif slope > 0.01 * mean_step:
            trend = "increasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Oscillation detection (sign changes in differences)
    if n_iters >= 3:
        diffs = np.diff(steps)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        oscillation_rate = sign_changes / max(n_iters - 2, 1)
    else:
        oscillation_rate = 0.0

    # Backtracking count (step size reductions)
    if n_iters >= 2:
        n_reductions = sum(1 for i in range(1, n_iters) if steps[i] < steps[i-1] * 0.9)
    else:
        n_reductions = 0

    return {
        "available": True,
        "n_iterations": n_iters,
        # Statistics
        "mean": mean_step,
        "std": std_step,
        "min": min_step,
        "max": max_step,
        "cv": cv,
        # Quality indicators
        "variability": variability,
        "trend": trend,
        "oscillation_rate": oscillation_rate,
        "n_reductions": n_reductions,
        # Raw data (last 20)
        "step_sizes": step_sizes[-20:],
    }


# ---------------------------------------------------------------------------
# C.23: Objective improvement monotonicity
# ---------------------------------------------------------------------------

def check_objective_monotonicity(
    obj_values: list[float],
    tol: float = 1e-8,
) -> dict[str, Any]:
    """
    Check if objective function improvements are monotonic.

    For maximization, objective should be non-decreasing.
    Violations indicate numerical issues or premature termination.

    :param obj_values (list[float]): Objective values per iteration
    :param tol (float): Tolerance for improvement check

    :return analysis (dict): Monotonicity check results
    """
    if not obj_values:
        return {"available": False, "reason": "no objective values"}

    n_iters = len(obj_values)
    objs = np.array(obj_values)

    # Check monotonicity (for maximization: non-decreasing)
    improvements = np.diff(objs)
    n_violations = int(np.sum(improvements < -tol))
    is_monotonic = n_violations == 0

    # Severity of violations
    if n_violations > 0:
        max_violation = float(np.min(improvements))  # Most negative
        violation_indices = np.where(improvements < -tol)[0].tolist()
    else:
        max_violation = 0.0
        violation_indices = []

    # Total improvement
    total_improvement = float(objs[-1] - objs[0])
    avg_improvement = total_improvement / max(n_iters - 1, 1)

    # Convergence quality (ratio of last improvement to average)
    if n_iters >= 2 and avg_improvement > 0:
        last_improvement = float(improvements[-1])
        convergence_ratio = last_improvement / avg_improvement
    else:
        convergence_ratio = 0.0

    return {
        "available": True,
        "n_iterations": n_iters,
        "is_monotonic": is_monotonic,
        "n_violations": n_violations,
        "max_violation": max_violation,
        "violation_indices": violation_indices[:10],  # First 10
        # Improvement statistics
        "total_improvement": total_improvement,
        "avg_improvement": avg_improvement,
        "convergence_ratio": convergence_ratio,
        # Start/end values
        "obj_initial": float(objs[0]),
        "obj_final": float(objs[-1]),
        "improvements": improvements[-20:].tolist(),  # Last 20
        "tolerance": tol,
    }


# ---------------------------------------------------------------------------
# C.24: Per-iteration gradient norm decay
# ---------------------------------------------------------------------------

def analyze_gradient_decay(
    grad_norms: list[float],
) -> dict[str, Any]:
    """
    Analyze gradient norm decay across iterations.

    Healthy convergence:
    - Gradient norms should decrease towards tolerance
    - No plateaus or sudden increases

    :param grad_norms (list[float]): Gradient norms per iteration

    :return analysis (dict): Gradient decay analysis
    """
    if not grad_norms:
        return {"available": False, "reason": "no gradient norms"}

    n_iters = len(grad_norms)
    norms = np.array(grad_norms)

    # Basic statistics
    initial_norm = float(norms[0])
    final_norm = float(norms[-1])
    reduction_ratio = initial_norm / max(final_norm, 1e-15)

    # Decay rate (linear fit on log-norms)
    valid_norms = norms[norms > 1e-15]
    if len(valid_norms) >= 3:
        log_norms = np.log(valid_norms)
        x = np.arange(len(valid_norms))
        slope, _ = np.polyfit(x, log_norms, 1)
        decay_rate = float(-slope)  # Positive = decaying
    else:
        decay_rate = 0.0

    # Plateau detection (no significant change for 5+ iterations)
    if n_iters >= 5:
        rel_changes = np.abs(np.diff(norms)) / np.maximum(norms[:-1], 1e-10)
        plateau_iters = np.sum(rel_changes < 0.01)
        has_plateau = plateau_iters >= 5
    else:
        plateau_iters = 0
        has_plateau = False

    # Convergence status
    if final_norm < 1e-6:
        convergence_status = "converged"
    elif final_norm < 1e-4:
        convergence_status = "near_converged"
    elif final_norm < 1e-2:
        convergence_status = "acceptable"
    else:
        convergence_status = "not_converged"

    return {
        "available": True,
        "n_iterations": n_iters,
        "initial_norm": initial_norm,
        "final_norm": final_norm,
        "reduction_ratio": reduction_ratio,
        "decay_rate": decay_rate,
        "has_plateau": has_plateau,
        "plateau_iters": int(plateau_iters),
        "convergence_status": convergence_status,
        # Raw data (last 20)
        "grad_norms": grad_norms[-20:],
    }


# ---------------------------------------------------------------------------
# C.28-C.29: Frontier anomaly detection
# ---------------------------------------------------------------------------

def detect_frontier_anomalies(
    frontier: list[dict[str, float]],
) -> dict[str, Any]:
    """
    Detect anomalies in the variance-entropy frontier.

    Checks for:
    - Non-monotonicity: H should increase as alpha decreases
    - Degeneracy: Very small H range indicates constrained solution
    - Gaps: Missing alpha values or irregular spacing

    :param frontier (list[dict]): Frontier points with alpha, variance, entropy

    :return analysis (dict): Frontier anomaly detection results
    """
    if not frontier:
        return {"available": False, "reason": "empty frontier"}

    n_points = len(frontier)

    # Extract values (sorted by alpha descending for monotonicity check)
    alphas = [f.get("alpha", 0) for f in frontier]
    entropies = [f.get("entropy", 0) for f in frontier]
    variances = [f.get("variance", 0) for f in frontier]

    # Sort by alpha descending
    sorted_indices = np.argsort(alphas)[::-1]
    alphas_sorted = [alphas[i] for i in sorted_indices]
    entropies_sorted = [entropies[i] for i in sorted_indices]
    _ = [variances[i] for i in sorted_indices]  # Kept for potential future use

    # Non-monotonicity check: as alpha decreases, H should increase
    n_violations = 0
    violation_alphas: list[float] = []
    for i in range(1, n_points):
        # Alpha decreased, H should increase or stay same
        if entropies_sorted[i] < entropies_sorted[i-1] - 1e-6:
            n_violations += 1
            violation_alphas.append(alphas_sorted[i])

    is_monotonic = n_violations == 0

    # Degeneracy check: H range
    h_range = max(entropies) - min(entropies)
    h_max = max(entropies)
    h_relative_range = h_range / max(h_max, 1e-10)
    is_degenerate = h_relative_range < 0.05  # < 5% variation

    # Alpha spacing regularity
    if n_points >= 3:
        alpha_diffs = np.diff(alphas_sorted)
        alpha_spacing_cv = float(np.std(alpha_diffs) / max(np.mean(np.abs(alpha_diffs)), 1e-10))
        has_gaps = alpha_spacing_cv > 1.0
    else:
        alpha_spacing_cv = 0.0
        has_gaps = False

    # Variance-entropy trade-off check
    # As alpha increases (more risk aversion), variance should decrease
    var_h_correlation = float(np.corrcoef(variances, entropies)[0, 1]) if n_points >= 3 else 0.0

    # Overall frontier quality
    if is_monotonic and not is_degenerate and not has_gaps:
        quality = "good"
    elif is_monotonic and not is_degenerate:
        quality = "acceptable"
    elif is_degenerate:
        quality = "degenerate"
    else:
        quality = "problematic"

    return {
        "available": True,
        "n_points": n_points,
        # Monotonicity
        "is_monotonic": is_monotonic,
        "n_violations": n_violations,
        "violation_alphas": violation_alphas[:5],  # First 5
        # Degeneracy
        "h_range": h_range,
        "h_relative_range": h_relative_range,
        "is_degenerate": is_degenerate,
        # Spacing
        "alpha_spacing_cv": alpha_spacing_cv,
        "has_gaps": has_gaps,
        # Trade-off
        "var_h_correlation": var_h_correlation,
        # Overall
        "quality": quality,
        # Summary statistics
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "h_min": min(entropies),
        "h_max": h_max,
        "var_min": min(variances),
        "var_max": max(variances),
    }


# ---------------------------------------------------------------------------
# C.25: Entropy loss per cardinality elimination
# ---------------------------------------------------------------------------

def analyze_cardinality_entropy_loss(
    elimination_rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze entropy loss at each cardinality elimination round.

    Helps diagnose if aggressive elimination is destroying diversification.

    :param elimination_rounds (list[dict]): Each with n_stocks, entropy, eliminated_ids

    :return analysis (dict): Cardinality-entropy trade-off analysis
    """
    if not elimination_rounds:
        return {"available": False, "reason": "no elimination rounds"}

    n_rounds = len(elimination_rounds)

    n_stocks_list = [r.get("n_stocks", 0) for r in elimination_rounds]
    entropy_list = [r.get("entropy", 0) for r in elimination_rounds]

    # Entropy loss per round
    entropy_losses: list[float] = []
    for i in range(1, n_rounds):
        loss = entropy_list[i-1] - entropy_list[i]
        entropy_losses.append(loss)

    # Cumulative entropy loss
    total_loss = entropy_list[0] - entropy_list[-1] if n_rounds >= 2 else 0.0
    relative_loss = total_loss / max(entropy_list[0], 1e-10)

    # Stock reduction
    stocks_removed = n_stocks_list[0] - n_stocks_list[-1] if n_rounds >= 2 else 0
    pct_removed = stocks_removed / max(n_stocks_list[0], 1)

    # Efficiency: entropy retained per stock retained
    if stocks_removed > 0:
        entropy_per_stock_removed = total_loss / stocks_removed
    else:
        entropy_per_stock_removed = 0.0

    # Identify worst rounds (highest entropy loss)
    if entropy_losses:
        worst_round_idx = int(np.argmax(entropy_losses))
        worst_round_loss = float(entropy_losses[worst_round_idx])
    else:
        worst_round_idx = 0
        worst_round_loss = 0.0

    return {
        "available": True,
        "n_rounds": n_rounds,
        # Per-round data
        "n_stocks_trajectory": n_stocks_list,
        "entropy_trajectory": entropy_list,
        "entropy_losses": entropy_losses,
        # Summary
        "total_entropy_loss": total_loss,
        "relative_entropy_loss": relative_loss,
        "stocks_removed": stocks_removed,
        "pct_stocks_removed": pct_removed,
        "entropy_per_stock_removed": entropy_per_stock_removed,
        # Worst round
        "worst_round_idx": worst_round_idx,
        "worst_round_loss": worst_round_loss,
        # Final state
        "final_n_stocks": n_stocks_list[-1] if n_stocks_list else 0,
        "final_entropy": entropy_list[-1] if entropy_list else 0.0,
    }
