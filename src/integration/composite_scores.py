"""
Composite diagnostic scores for the VAE Latent Risk Factor pipeline.

Provides synthetic, interpretable scores (0-100) that summarize raw diagnostic
values into actionable insights. Each function returns a dict with:
- score: 0-100 numeric value
- grade: A/B/C/D/F letter grade
- interpretation: human-readable explanation
- action: recommended action if score < threshold (None if OK)

Reference: docs/diagnostic.md for full interpretation guide.
"""

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Grade mapping
# ---------------------------------------------------------------------------

def _get_grade(score: float) -> str:
    """
    Map numeric score to letter grade.

    :param score (float): Score in [0, 100]

    :return grade (str): Letter grade A/B/C/D/F
    """
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def _get_status(score: float) -> str:
    """
    Map numeric score to status string.

    :param score (float): Score in [0, 100]

    :return status (str): Status label
    """
    if score >= 90:
        return "EXCELLENT"
    elif score >= 75:
        return "GOOD"
    elif score >= 60:
        return "NEEDS ATTENTION"
    elif score >= 40:
        return "MARGINAL"
    else:
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Solver Health Score
# ---------------------------------------------------------------------------

def compute_solver_health_score(solver_stats: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compute solver convergence quality score.

    Evaluates:
    - Final gradient norm (50%): How close to optimum
    - Convergence ratio (30%): Fraction of restarts that converged
    - Iteration efficiency (20%): Did not hit max_iter

    :param solver_stats (dict | None): From solver_diagnostics()

    :return result (dict): score, grade, interpretation, action
    """
    if solver_stats is None or not solver_stats.get("available", False):
        return {
            "available": False,
            "score": 0.0,
            "grade": "F",
            "interpretation": "Solver diagnostics not available",
            "action": "Run pipeline with solver_stats collection enabled",
        }

    grad_norm = solver_stats.get("best_final_grad_norm", float("nan"))
    converged_ratio = solver_stats.get("converged_ratio", 0.0)
    n_iterations = solver_stats.get("best_n_iterations", 0)
    max_iter = solver_stats.get("max_iterations", 100)
    best_converged = solver_stats.get("best_converged", False)

    # Gradient norm score (50 points)
    # log10(1e-10) = -10 -> 50 pts, log10(1e-2) = -2 -> 0 pts
    if np.isnan(grad_norm) or grad_norm <= 0:
        grad_score = 0.0
    else:
        log_grad = np.log10(grad_norm)
        # Map [-10, -2] to [50, 0]
        grad_score = 50.0 * (1.0 - np.clip((log_grad + 10) / 8, 0, 1))

    # Convergence ratio score (30 points)
    conv_score = 30.0 * converged_ratio

    # Iteration efficiency score (20 points)
    if max_iter > 0 and n_iterations < max_iter:
        # Reward finishing early
        iter_score = 20.0 * (1.0 - n_iterations / max_iter)
    else:
        # Hit iteration limit
        iter_score = 5.0

    total_score = grad_score + conv_score + iter_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    if total_score >= 90:
        interpretation = (
            f"Optimum reached with high confidence. "
            f"Gradient norm = {grad_norm:.2e}, "
            f"{converged_ratio:.0%} of starts converged."
        )
    elif total_score >= 75:
        interpretation = (
            f"Reliable solution found. "
            f"Gradient norm = {grad_norm:.2e}, "
            f"{converged_ratio:.0%} of starts converged."
        )
    elif total_score >= 60:
        interpretation = (
            f"Acceptable convergence but verify constraints. "
            f"Gradient norm = {grad_norm:.2e}, "
            f"{converged_ratio:.0%} of starts converged."
        )
    else:
        interpretation = (
            f"Optimization likely incomplete. "
            f"Gradient norm = {grad_norm:.2e}, "
            f"only {converged_ratio:.0%} of starts converged."
        )

    # Generate action if needed
    action = None
    if total_score < 60:
        if grad_norm > 1e-3:
            action = "Increase sca_max_iter or reduce sca_tol"
        elif converged_ratio < 0.5:
            action = "Check objective conditioning; consider more multi-starts"
        else:
            action = "Review SCA parameters (step size, tolerance)"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "grad_norm": grad_norm,
            "converged_ratio": converged_ratio,
            "n_iterations": n_iterations,
            "best_converged": best_converged,
            "component_scores": {
                "gradient": float(grad_score),
                "convergence": float(conv_score),
                "efficiency": float(iter_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Constraint Pressure Score
# ---------------------------------------------------------------------------

def compute_constraint_pressure_score(
    constraints: dict[str, Any] | None,
    n_active: int,
) -> dict[str, Any]:
    """
    Compute constraint binding pressure score.

    Higher score = optimizer has more freedom (interior solution).
    Lower score = many constraints are binding.

    :param constraints (dict | None): From constraint_binding_diagnostics()
    :param n_active (int): Number of active (non-zero) positions

    :return result (dict): score, grade, interpretation, action
    """
    if constraints is None or not constraints.get("available", False):
        return {
            "available": False,
            "score": 50.0,  # Neutral default
            "grade": "C",
            "interpretation": "Constraint binding diagnostics not available",
            "action": None,
        }

    binding_fraction = constraints.get("binding_fraction", 0.0)
    n_at_w_max = constraints.get("n_at_w_max", 0)
    n_at_w_min = constraints.get("n_at_w_min", 0)
    tau_binding = constraints.get("tau_binding", False)
    actual_turnover = constraints.get("actual_turnover", 0.0)
    concentrated_weight = constraints.get("concentrated_weight", 0.0)

    n_active = max(n_active, 1)  # Avoid division by zero

    # w_max binding score (40 points)
    # sqrt decay: 0% binding -> 40, 100% binding -> 0
    wmax_score = 40.0 * np.sqrt(1.0 - np.clip(binding_fraction, 0, 1))

    # w_min binding score (20 points)
    # Penalize if too many at minimum
    wmin_ratio = min(n_at_w_min / n_active * 2, 1.0)
    wmin_score = 20.0 * (1.0 - wmin_ratio)

    # Turnover score (25 points)
    if tau_binding:
        turn_score = 5.0  # Penalty for hitting turnover cap
    else:
        # Reward for being well under the cap
        turn_score = 25.0 * (1.0 - actual_turnover * 0.8)
        turn_score = max(turn_score, 5.0)

    # Concentration score (15 points)
    # eff_n / n_active should be close to 1 for diversified
    eff_n_ratio = 1.0 / (1.0 + concentrated_weight * 10)  # Heuristic
    conc_score = 15.0 * eff_n_ratio

    total_score = wmax_score + wmin_score + turn_score + conc_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    if total_score >= 80:
        interpretation = (
            f"Optimizer is free (interior solution). "
            f"Only {n_at_w_max} positions at w_max ({binding_fraction:.0%})."
        )
    elif total_score >= 65:
        interpretation = (
            f"Normal constraint pressure. "
            f"{n_at_w_max} positions at w_max ({binding_fraction:.0%}), "
            f"turnover = {actual_turnover:.1%}."
        )
    elif total_score >= 50:
        interpretation = (
            f"Moderate constraint pressure. "
            f"{n_at_w_max} positions at w_max ({binding_fraction:.0%}). "
            f"Consider relaxing constraints."
        )
    else:
        interpretation = (
            f"Strong constraint pressure. "
            f"{n_at_w_max} positions at w_max ({binding_fraction:.0%}). "
            f"Constraints may be too restrictive."
        )

    # Generate action if needed
    action = None
    if total_score < 50:
        if binding_fraction > 0.5:
            action = "Consider increasing w_max (e.g., 0.05 -> 0.07)"
        elif tau_binding:
            action = "Consider increasing tau_max or reducing rebalancing frequency"
        else:
            action = "Review constraint configuration"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "binding_fraction": binding_fraction,
            "n_at_w_max": n_at_w_max,
            "n_at_w_min": n_at_w_min,
            "tau_binding": tau_binding,
            "actual_turnover": actual_turnover,
            "component_scores": {
                "wmax": float(wmax_score),
                "wmin": float(wmin_score),
                "turnover": float(turn_score),
                "concentration": float(conc_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Covariance Quality Score
# ---------------------------------------------------------------------------

def compute_covariance_quality_score(risk_model: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compute risk model calibration quality score.

    Evaluates:
    - Condition number (30%): Matrix conditioning
    - Variance ratio (35%): Realized vs predicted accuracy
    - Explanatory power (25%): Factor model fit
    - Shrinkage intensity (10%): Regularization level

    :param risk_model (dict | None): From risk_model_diagnostics()

    :return result (dict): score, grade, interpretation, action
    """
    if risk_model is None:
        return {
            "available": False,
            "score": 0.0,
            "grade": "F",
            "interpretation": "Risk model diagnostics not available",
            "action": "Run pipeline with risk model diagnostics enabled",
        }

    cond_num = risk_model.get("condition_number", 1e10)
    var_ratio = risk_model.get("var_ratio_oos", float("nan"))
    ep = risk_model.get("explanatory_power", 0.0)
    shrinkage = risk_model.get("shrinkage_intensity")

    # Condition number score (30 points)
    # log10(100) = 2 -> 30 pts, log10(1e10) = 10 -> 0 pts
    if cond_num <= 0:
        cond_score = 0.0
    else:
        log_cond = np.log10(cond_num)
        cond_score = 30.0 * (1.0 - np.clip((log_cond - 2) / 8, 0, 1))

    # Variance ratio score (35 points)
    # Target: 1.0, acceptable: [0.5, 2.0]
    if np.isnan(var_ratio):
        var_score = 17.5  # Neutral if unavailable
        var_ratio_status = "unavailable"
    else:
        # Perfect at 1.0, linear decay outside [0.8, 1.2]
        deviation = abs(var_ratio - 1.0)
        if deviation <= 0.2:
            var_score = 35.0
        elif deviation <= 1.0:
            var_score = 35.0 * (1.0 - (deviation - 0.2) / 0.8)
        else:
            var_score = 0.0

        if var_ratio < 0.5:
            var_ratio_status = "overestimates risk"
        elif var_ratio > 2.0:
            var_ratio_status = "underestimates risk"
        elif var_ratio < 0.8:
            var_ratio_status = "slightly conservative"
        elif var_ratio > 1.2:
            var_ratio_status = "slightly aggressive"
        else:
            var_ratio_status = "well-calibrated"

    # Explanatory power score (25 points)
    # EP >= 0.20 is excellent, >= 0.05 is OK
    ep_score = 25.0 * np.clip(ep * 5, 0, 1)

    # Shrinkage intensity score (10 points)
    # Ideal: 0.15-0.65
    if shrinkage is None:
        shrink_score = 5.0  # Neutral if unavailable
    elif 0.15 <= shrinkage <= 0.65:
        shrink_score = 10.0
    elif shrinkage < 0.15:
        # Very low shrinkage might indicate overfitting
        shrink_score = 7.0
    elif shrinkage <= 0.75:
        shrink_score = 5.0
    else:
        # High shrinkage indicates data issues
        shrink_score = 10.0 * (1.0 - (shrinkage - 0.65) / 0.35)
        shrink_score = max(shrink_score, 0.0)

    total_score = cond_score + var_score + ep_score + shrink_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    parts = []
    if not np.isnan(var_ratio):
        parts.append(f"var_ratio={var_ratio:.2f} ({var_ratio_status})")
    if ep > 0:
        parts.append(f"EP={ep:.3f}")
    if shrinkage is not None:
        parts.append(f"shrinkage={shrinkage:.2f}")
    parts.append(f"cond={cond_num:.1e}")

    if total_score >= 80:
        interpretation = f"Risk model well-calibrated. {', '.join(parts)}."
    elif total_score >= 60:
        interpretation = f"Acceptable calibration. {', '.join(parts)}."
    else:
        interpretation = f"Calibration issues detected. {', '.join(parts)}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if not np.isnan(var_ratio) and var_ratio < 0.5:
            action = "Model overestimates risk; check variance targeting scale"
        elif not np.isnan(var_ratio) and var_ratio > 2.0:
            action = "Model underestimates risk; increase shrinkage or history"
        elif shrinkage is not None and shrinkage > 0.75:
            action = "Insufficient data; extend training window or reduce factors"
        elif cond_num > 1e6:
            action = "Ill-conditioned covariance; increase ridge regularization"
        else:
            action = "Review risk model parameters"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "condition_number": cond_num,
            "var_ratio": var_ratio if not np.isnan(var_ratio) else None,
            "var_ratio_status": var_ratio_status if not np.isnan(var_ratio) else None,
            "explanatory_power": ep,
            "shrinkage_intensity": shrinkage,
            "component_scores": {
                "conditioning": float(cond_score),
                "variance": float(var_score),
                "explanatory": float(ep_score),
                "shrinkage": float(shrink_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Reconstruction Balance Score
# ---------------------------------------------------------------------------

def compute_reconstruction_balance_score(
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute VAE reconstruction quality and balance score.

    Evaluates:
    - Overall reconstruction loss (40%)
    - Feature balance (35%): Returns vs volatility ratio
    - Training stability (25%): Overfit ratio

    :param training (dict | None): From training_diagnostics()

    :return result (dict): score, grade, interpretation, action
    """
    if training is None or not training.get("available", False):
        return {
            "available": False,
            "score": 50.0,  # Neutral default
            "grade": "C",
            "interpretation": "Training diagnostics not available",
            "action": None,
        }

    best_recon = training.get("best_recon", 1.0)
    overfit_ratio = training.get("overfit_ratio", 1.0)

    # Per-feature reconstruction (if available)
    recon_per_feature = training.get("recon_per_feature_best", [])
    has_per_feature = len(recon_per_feature) >= 2

    # Reconstruction loss score (40 points)
    # Lower is better; best_recon < 0.2 is excellent
    recon_score = 40.0 * (1.0 - np.clip(best_recon * 5, 0, 1))

    # Feature balance score (35 points)
    if has_per_feature and recon_per_feature[1] > 1e-10:
        ratio = recon_per_feature[0] / recon_per_feature[1]
        # Ideal: 0.8-2.5 (returns slightly harder than volatility)
        if 0.8 <= ratio <= 2.5:
            balance_score = 35.0
        elif ratio < 0.8:
            # Volatility harder than expected
            balance_score = 35.0 * (ratio / 0.8)
        else:
            # Returns much harder
            balance_score = 35.0 * (1.0 - (ratio - 2.5) / 2.5)
            balance_score = max(balance_score, 0.0)
        ratio_status = (
            "balanced" if 0.8 <= ratio <= 2.5
            else "volatility harder" if ratio < 0.8
            else "returns much harder"
        )
    else:
        balance_score = 25.0  # Neutral if unavailable
        ratio = None
        ratio_status = "unavailable"

    # Stability score (25 points)
    # Ideal: 0.9-1.15
    if 0.9 <= overfit_ratio <= 1.15:
        stability_score = 25.0
    elif overfit_ratio < 0.85:
        # Underfitting
        stability_score = 25.0 * (overfit_ratio / 0.85)
    elif overfit_ratio <= 1.3:
        # Mild overfitting
        stability_score = 25.0 * (1.0 - (overfit_ratio - 1.15) / 0.15)
    else:
        # Severe overfitting
        stability_score = max(0.0, 25.0 * (1.0 - (overfit_ratio - 1.3) / 0.5))

    total_score = recon_score + balance_score + stability_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    parts = [f"recon_loss={best_recon:.4f}"]
    if ratio is not None:
        parts.append(f"feature_ratio={ratio:.2f} ({ratio_status})")
    parts.append(f"overfit_ratio={overfit_ratio:.2f}")

    if total_score >= 80:
        interpretation = f"VAE learns both features well. {', '.join(parts)}."
    elif total_score >= 60:
        interpretation = f"Acceptable reconstruction. {', '.join(parts)}."
    else:
        interpretation = f"Reconstruction issues. {', '.join(parts)}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if overfit_ratio > 1.3:
            action = "Overfitting detected; increase dropout or reduce K"
        elif overfit_ratio < 0.85:
            action = "Underfitting; increase epochs or model capacity"
        elif ratio is not None and ratio > 3.0:
            action = "Returns reconstruction very hard; check data preprocessing"
        else:
            action = "Review VAE architecture or training hyperparameters"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "best_recon": best_recon,
            "overfit_ratio": overfit_ratio,
            "feature_ratio": ratio,
            "feature_ratio_status": ratio_status,
            "recon_per_feature": recon_per_feature if has_per_feature else None,
            "component_scores": {
                "reconstruction": float(recon_score),
                "balance": float(balance_score),
                "stability": float(stability_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Overall Score
# ---------------------------------------------------------------------------

def compute_overall_score(
    solver: dict[str, Any],
    constraint: dict[str, Any],
    covariance: dict[str, Any],
    reconstruction: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute weighted overall diagnostic score.

    Weights:
    - Covariance: 35% (risk model accuracy is critical)
    - Solver: 25% (must find actual optimum)
    - Reconstruction: 25% (VAE must learn meaningful factors)
    - Constraints: 15% (some pressure is acceptable)

    :param solver (dict): From compute_solver_health_score()
    :param constraint (dict): From compute_constraint_pressure_score()
    :param covariance (dict): From compute_covariance_quality_score()
    :param reconstruction (dict): From compute_reconstruction_balance_score()

    :return result (dict): score, grade, status, summary, priority_actions
    """
    weights = {
        "solver": 0.25,
        "constraint": 0.15,
        "covariance": 0.35,
        "reconstruction": 0.25,
    }

    components = {
        "solver": solver,
        "constraint": constraint,
        "covariance": covariance,
        "reconstruction": reconstruction,
    }

    # Compute weighted score
    total_weight = 0.0
    weighted_sum = 0.0
    missing_components: list[str] = []

    for name, comp in components.items():
        if comp.get("available", False):
            weighted_sum += weights[name] * comp["score"]
            total_weight += weights[name]
        else:
            missing_components.append(name)

    # Penalty for missing components (2.5 points each)
    penalty = len(missing_components) * 2.5

    if total_weight > 0:
        overall = (weighted_sum / total_weight) - penalty
    else:
        overall = 0.0

    overall = np.clip(overall, 0, 100)
    grade = _get_grade(overall)
    status = _get_status(overall)

    # Generate summary
    if overall >= 80:
        summary = "Pipeline operating normally. All systems healthy."
    elif overall >= 60:
        summary = "Pipeline operating with minor issues. Review recommendations."
    elif overall >= 40:
        summary = "Pipeline needs attention. Multiple components below threshold."
    else:
        summary = "Critical pipeline issues. Immediate investigation required."

    if missing_components:
        summary += f" Missing: {', '.join(missing_components)}."

    # Collect priority actions
    priority_actions: list[dict[str, Any]] = []
    for name, comp in components.items():
        if comp.get("action") is not None:
            priority_actions.append({
                "component": name,
                "score": comp.get("score", 0),
                "action": comp["action"],
            })

    # Sort by score (lowest first = highest priority)
    priority_actions.sort(key=lambda x: x["score"])

    return {
        "score": float(overall),
        "grade": grade,
        "status": status,
        "summary": summary,
        "component_scores": {
            name: {
                "score": comp.get("score", 0),
                "grade": comp.get("grade", "F"),
                "available": comp.get("available", False),
            }
            for name, comp in components.items()
        },
        "priority_actions": priority_actions[:5],  # Top 5 actions
        "missing_components": missing_components,
    }


# ---------------------------------------------------------------------------
# Cross-Fold Comparison
# ---------------------------------------------------------------------------

def compare_across_folds(fold_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare composite scores across walk-forward folds.

    Computes statistics and identifies anomalies.

    :param fold_scores (list[dict]): List of overall score dicts from each fold

    :return comparison (dict): means, stds, trend, anomaly_folds, stability_verdict
    """
    if not fold_scores:
        return {
            "available": False,
            "reason": "No fold scores provided",
        }

    n_folds = len(fold_scores)

    # Extract overall scores
    overall_scores = [f.get("score", 0) for f in fold_scores]

    # Extract component scores
    component_names = ["solver", "constraint", "covariance", "reconstruction"]
    component_scores: dict[str, list[float]] = {name: [] for name in component_names}

    for f in fold_scores:
        comp_dict = f.get("component_scores", {})
        for name in component_names:
            comp_data = comp_dict.get(name, {})
            component_scores[name].append(comp_data.get("score", 0))

    # Compute statistics
    overall_mean = float(np.mean(overall_scores))
    overall_std = float(np.std(overall_scores))

    component_stats: dict[str, dict[str, float]] = {}
    for name in component_names:
        scores = component_scores[name]
        component_stats[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    # Detect trend
    if n_folds >= 3:
        # Simple linear regression for trend
        x = np.arange(n_folds)
        slope, _ = np.polyfit(x, overall_scores, 1)
        if slope > 2:  # Improving by >2 pts per fold
            trend = "improving"
        elif slope < -2:  # Degrading by >2 pts per fold
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Identify anomaly folds (> 2 std from mean)
    anomaly_threshold = 2.0
    anomaly_folds: list[int] = []
    if overall_std > 0:
        for i, score in enumerate(overall_scores):
            if abs(score - overall_mean) > anomaly_threshold * overall_std:
                anomaly_folds.append(i)

    # Generate stability verdict
    if overall_std < 10:
        stability_verdict = "Stable strategy (std < 10)"
    elif overall_std < 20:
        stability_verdict = f"Moderate variation (std = {overall_std:.1f})"
    else:
        stability_verdict = f"Unstable strategy (std = {overall_std:.1f})"

    if anomaly_folds:
        stability_verdict += f". Anomaly folds: {anomaly_folds}"

    return {
        "available": True,
        "n_folds": n_folds,
        "overall": {
            "mean": overall_mean,
            "std": overall_std,
            "min": float(np.min(overall_scores)),
            "max": float(np.max(overall_scores)),
            "scores": overall_scores,
        },
        "components": component_stats,
        "trend": trend,
        "anomaly_folds": anomaly_folds,
        "stability_verdict": stability_verdict,
    }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_all_composite_scores(
    solver_stats: dict[str, Any] | None,
    constraints: dict[str, Any] | None,
    risk_model: dict[str, Any] | None,
    training: dict[str, Any] | None,
    n_active: int,
) -> dict[str, Any]:
    """
    Compute all composite scores in one call.

    :param solver_stats (dict | None): From solver_diagnostics()
    :param constraints (dict | None): From constraint_binding_diagnostics()
    :param risk_model (dict | None): From risk_model_diagnostics()
    :param training (dict | None): From training_diagnostics()
    :param n_active (int): Number of active positions

    :return scores (dict): All composite scores including overall
    """
    solver = compute_solver_health_score(solver_stats)
    constraint = compute_constraint_pressure_score(constraints, n_active)
    covariance = compute_covariance_quality_score(risk_model)
    reconstruction = compute_reconstruction_balance_score(training)
    overall = compute_overall_score(solver, constraint, covariance, reconstruction)

    return {
        "solver": solver,
        "constraint": constraint,
        "covariance": covariance,
        "reconstruction": reconstruction,
        "overall": overall,
    }
