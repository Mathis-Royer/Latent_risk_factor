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
# Score Confidence and Temporal Context
# ---------------------------------------------------------------------------

def compute_score_confidence(
    score: float,
    n_samples: int,
    std: float,
) -> dict[str, Any]:
    """
    Compute confidence interval for a score.

    Uses standard error to estimate confidence bounds and reliability.

    :param score (float): The computed score (0-100)
    :param n_samples (int): Number of samples used to compute the score
    :param std (float): Standard deviation of the underlying metric

    :return result (dict): score, confidence_interval, reliability
    """
    se = std / np.sqrt(max(n_samples, 1))
    ci_lower = max(0.0, score - 1.96 * se)
    ci_upper = min(100.0, score + 1.96 * se)

    if std < 10:
        reliability = "high"
    elif std < 20:
        reliability = "medium"
    else:
        reliability = "low"

    return {
        "score": float(score),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "reliability": reliability,
    }


def encode_temporal_context(
    current: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Add temporal context to current score.

    Computes delta vs previous, trend direction, and anomaly detection.

    :param current (dict): Current score dict with at least "score" key
    :param history (list[dict]): List of historical score dicts

    :return context (dict): current_score, delta_vs_previous, trend_direction, is_anomaly
    """
    current_score = current.get("score", 0.0)

    if not history:
        return {
            "current_score": float(current_score),
            "delta_vs_previous": 0.0,
            "trend_direction": "stable",
            "is_anomaly": False,
        }

    prev = history[-1].get("score", 0.0)
    delta = current_score - prev

    # Compute trend over last 3+ folds
    if len(history) >= 3:
        recent = [h.get("score", 0.0) for h in history[-3:]] + [current_score]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        if slope > 2:
            trend = "improving"
        elif slope < -2:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Anomaly detection
    if len(history) >= 3:
        historical_scores = [h.get("score", 0.0) for h in history]
        mean = np.mean(historical_scores)
        std = np.std(historical_scores)
        z_score = (current_score - mean) / max(std, 1.0)
        is_anomaly = abs(z_score) > 2.0
    else:
        is_anomaly = False

    return {
        "current_score": float(current_score),
        "delta_vs_previous": float(delta),
        "trend_direction": trend,
        "is_anomaly": bool(is_anomaly),
    }


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
# Training Convergence Score
# ---------------------------------------------------------------------------

def compute_training_convergence_score(
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute training convergence quality score.

    Evaluates:
    - Best epoch timing (30%): Was best epoch at optimal fraction of training?
    - Convergence stability (25%): Was val ELBO still decreasing at end?
    - LR scheduling (25%): Did ReduceLROnPlateau work appropriately?
    - Sigma bounds (20%): Did sigma_sq hit its bounds?

    :param training (dict | None): From training_diagnostics()

    :return result (dict): score, grade, interpretation, action, details
    """
    if training is None or not training.get("available", False):
        return {
            "available": False,
            "score": 50.0,  # Neutral default
            "grade": "C",
            "interpretation": "Training diagnostics not available",
            "action": None,
        }

    best_epoch_fraction = training.get("best_epoch_fraction", 0.5)
    still_decreasing = training.get("still_decreasing_at_end", False)
    n_lr_reductions = training.get("n_lr_reductions", 0)
    sigma_min_hit = training.get("sigma_sq_min_hit", False)
    sigma_max_hit = training.get("sigma_sq_max_hit", False)
    overfit_ratio = training.get("overfit_ratio", 1.0)

    # Timing score (30 points)
    # Optimal: best epoch at 30-85% of training
    # Too early (<15%): underfitting
    # Too late (>95%): may still be converging
    if 0.30 <= best_epoch_fraction <= 0.85:
        timing_score = 30.0
    elif best_epoch_fraction < 0.15:
        # Too early = underfitting
        timing_score = 10.0
    elif best_epoch_fraction > 0.95:
        # Very late = may not have converged
        timing_score = 15.0
    elif best_epoch_fraction < 0.30:
        # Early but not too early
        timing_score = 10.0 + 20.0 * (best_epoch_fraction - 0.15) / 0.15
    else:
        # Late but not too late
        timing_score = 30.0 - 15.0 * (best_epoch_fraction - 0.85) / 0.10

    timing_score = np.clip(timing_score, 0, 30)

    # Stability score (25 points)
    # Penalize if val ELBO still decreasing at end
    if still_decreasing:
        stability_score = 15.0  # 40% penalty
    else:
        stability_score = 25.0

    # LR scheduling score (25 points)
    # Ideal: 2-5 reductions (scheduler found plateaus)
    # 0-1: May not need scheduling or didn't trigger
    # >10: Too many reductions = instability
    if 2 <= n_lr_reductions <= 5:
        lr_score = 25.0
    elif n_lr_reductions <= 1:
        lr_score = 15.0
    elif n_lr_reductions <= 7:
        lr_score = 20.0
    elif n_lr_reductions <= 10:
        lr_score = 15.0
    else:
        lr_score = 10.0

    # Sigma bounds score (20 points)
    # Penalize if sigma_sq hit its bounds
    sigma_score = 20.0
    if sigma_min_hit:
        sigma_score -= 10.0
    if sigma_max_hit:
        sigma_score -= 10.0

    total_score = timing_score + stability_score + lr_score + sigma_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    issues = []
    if best_epoch_fraction < 0.15:
        issues.append("best epoch too early (underfitting?)")
    elif best_epoch_fraction > 0.95:
        issues.append("best epoch at end (may need more epochs)")
    if still_decreasing:
        issues.append("val ELBO still decreasing")
    if n_lr_reductions > 10:
        issues.append("many LR reductions (unstable)")
    if sigma_min_hit:
        issues.append("sigma hit lower bound")
    if sigma_max_hit:
        issues.append("sigma hit upper bound")

    if total_score >= 85:
        interpretation = (
            f"Training converged optimally. Best epoch at {best_epoch_fraction:.0%}, "
            f"{n_lr_reductions} LR reductions."
        )
    elif total_score >= 70:
        interpretation = (
            f"Training converged acceptably. Best epoch at {best_epoch_fraction:.0%}."
        )
        if issues:
            interpretation += f" Minor issues: {', '.join(issues[:2])}."
    elif total_score >= 55:
        interpretation = f"Training convergence needs review: {', '.join(issues[:3])}."
    else:
        interpretation = f"Training convergence problems: {', '.join(issues)}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if still_decreasing:
            action = "Increase max_epochs — training did not converge"
        elif best_epoch_fraction < 0.15:
            action = "Model underfitting — increase capacity or reduce regularization"
        elif sigma_min_hit or sigma_max_hit:
            action = "Adjust sigma_sq bounds to allow better fit"
        elif n_lr_reductions > 10:
            action = "Review LR scheduler settings — too many reductions"
        else:
            action = "Review training configuration"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "best_epoch_fraction": best_epoch_fraction,
            "still_decreasing": still_decreasing,
            "n_lr_reductions": n_lr_reductions,
            "sigma_min_hit": sigma_min_hit,
            "sigma_max_hit": sigma_max_hit,
            "overfit_ratio": overfit_ratio,
            "component_scores": {
                "timing": float(timing_score),
                "stability": float(stability_score),
                "lr_scheduling": float(lr_score),
                "sigma_bounds": float(sigma_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Active Unit Score
# ---------------------------------------------------------------------------

def compute_active_unit_score(
    training: dict[str, Any] | None,
    latent: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute active unit utilization score.

    Evaluates:
    - Utilization ratio (35%): AU/K in optimal range
    - AU stability (30%): Final AU vs max during training
    - Effective dimensions (35%): Balanced KL distribution

    :param training (dict | None): From training_diagnostics()
    :param latent (dict | None): From latent_diagnostics()

    :return result (dict): score, grade, interpretation, action, details
    """
    # Get latent metrics
    if latent is None:
        return {
            "available": False,
            "score": 50.0,
            "grade": "C",
            "interpretation": "Latent diagnostics not available",
            "action": None,
        }

    AU = latent.get("AU", 0)
    K = latent.get("K", 1)
    utilization_ratio = latent.get("utilization_ratio", 0.0)
    eff_latent_dims = latent.get("eff_latent_dims", 0.0)

    # Get training AU evolution if available
    au_final = AU
    au_max_during = AU
    if training is not None and training.get("available", False):
        au_final = training.get("au_final", AU)
        au_max_during = training.get("au_max_during_training", AU)

    # Utilization score (35 points)
    # Optimal: 15-60% of capacity used
    # <5%: collapse
    # >80%: saturation
    if 0.15 <= utilization_ratio <= 0.60:
        util_score = 35.0
    elif utilization_ratio < 0.05:
        util_score = 5.0  # Severe collapse
    elif utilization_ratio < 0.15:
        # Partial collapse
        util_score = 5.0 + 30.0 * (utilization_ratio - 0.05) / 0.10
    elif utilization_ratio <= 0.80:
        # Slight oversaturation
        util_score = 35.0 - 15.0 * (utilization_ratio - 0.60) / 0.20
    else:
        # Heavy saturation
        util_score = 20.0

    util_score = np.clip(util_score, 0, 35)

    # AU stability score (30 points)
    # Ratio of final AU to max during training
    # High ratio = stable; low ratio = excessive pruning
    if au_max_during > 0:
        au_retention = au_final / au_max_during
    else:
        au_retention = 1.0

    if 0.8 <= au_retention <= 1.0:
        stability_score = 30.0
    elif au_retention < 0.5:
        stability_score = 10.0  # Excessive pruning
    elif au_retention < 0.8:
        stability_score = 10.0 + 20.0 * (au_retention - 0.5) / 0.3
    else:
        stability_score = 30.0

    stability_score = np.clip(stability_score, 0, 30)

    # Spectrum score (35 points)
    # Ratio of effective dims to AU — balanced distribution
    # Ideal: eff_dims/AU in [0.5, 1.0]
    if AU > 0 and eff_latent_dims > 0:
        spectrum_ratio = eff_latent_dims / AU
    else:
        spectrum_ratio = 0.0

    if 0.5 <= spectrum_ratio <= 1.0:
        spectrum_score = 35.0
    elif spectrum_ratio < 0.3:
        spectrum_score = 15.0  # Few dims dominate
    elif spectrum_ratio < 0.5:
        spectrum_score = 15.0 + 20.0 * (spectrum_ratio - 0.3) / 0.2
    else:
        # ratio > 1 means more "effective" dims than active (shouldn't happen normally)
        spectrum_score = 35.0

    spectrum_score = np.clip(spectrum_score, 0, 35)

    total_score = util_score + stability_score + spectrum_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    if total_score >= 80:
        interpretation = (
            f"Latent space well-utilized. AU={AU}/{K} ({utilization_ratio:.0%}), "
            f"eff_dims={eff_latent_dims:.1f}."
        )
    elif total_score >= 60:
        interpretation = (
            f"Latent space acceptable. AU={AU}/{K} ({utilization_ratio:.0%})."
        )
        if au_retention < 0.8:
            interpretation += f" AU dropped from {au_max_during} to {au_final} during training."
    else:
        issues = []
        if utilization_ratio < 0.05:
            issues.append("severe collapse")
        elif utilization_ratio < 0.15:
            issues.append("low utilization")
        elif utilization_ratio > 0.80:
            issues.append("near saturation")
        if au_retention < 0.5:
            issues.append("excessive pruning")
        if spectrum_ratio < 0.3:
            issues.append("few dims dominate")
        interpretation = f"Latent space issues: {', '.join(issues)}. AU={AU}/{K}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if utilization_ratio < 0.05:
            action = "Posterior collapse detected — reduce beta or slow KL annealing"
        elif utilization_ratio < 0.15:
            action = "Low AU — increase training data or reduce regularization"
        elif utilization_ratio > 0.80:
            action = "Near saturation — consider increasing K"
        elif au_retention < 0.5:
            action = "Excessive AU pruning — review early stopping or KL weight"
        elif spectrum_ratio < 0.3:
            action = "Few dimensions dominate — check for mode collapse"
        else:
            action = "Review VAE architecture"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "AU": AU,
            "K": K,
            "utilization_ratio": utilization_ratio,
            "au_final": au_final,
            "au_max_during": au_max_during,
            "au_retention": au_retention,
            "eff_latent_dims": eff_latent_dims,
            "spectrum_ratio": spectrum_ratio,
            "component_scores": {
                "utilization": float(util_score),
                "stability": float(stability_score),
                "spectrum": float(spectrum_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Portfolio Diversification Score
# ---------------------------------------------------------------------------

def compute_portfolio_diversification_score(
    portfolio: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute portfolio diversification quality score.

    Evaluates:
    - Factor entropy (40%): H_norm_signal normalized entropy
    - ENB ratio (25%): Effective number of bets vs n_signal
    - Position balance (25%): Effective N vs active positions
    - Gini coefficient (10%): Weight equality

    :param portfolio (dict | None): From portfolio_diagnostics()

    :return result (dict): score, grade, interpretation, action, details
    """
    if portfolio is None:
        return {
            "available": False,
            "score": 50.0,
            "grade": "C",
            "interpretation": "Portfolio diagnostics not available",
            "action": None,
        }

    H_norm_signal = portfolio.get("H_norm_signal", 0.0)
    H_norm_eff = portfolio.get("H_norm_eff", 0.0)
    enb = portfolio.get("enb", 0.0)
    n_signal = portfolio.get("n_signal", 1)
    eff_n_positions = portfolio.get("eff_n_positions", 0.0)
    n_active_positions = portfolio.get("n_active_positions", 1)
    gini = portfolio.get("gini_coefficient", 0.0)

    # Use H_norm_signal if available, else fallback to H_norm_eff
    h_norm = H_norm_signal if H_norm_signal > 0 else H_norm_eff

    # Entropy score (40 points)
    # H_norm >= 0.5 is excellent diversification
    if h_norm >= 0.5:
        entropy_score = 40.0
    elif h_norm >= 0.3:
        entropy_score = 40.0 * (h_norm / 0.5)
    else:
        entropy_score = 40.0 * (h_norm / 0.3) * 0.6  # Severe penalty below 0.3

    entropy_score = np.clip(entropy_score, 0, 40)

    # ENB ratio score (25 points)
    # ENB / n_signal close to 1 = perfect diversification
    n_signal_safe = max(n_signal, 1)
    enb_ratio = min(enb / n_signal_safe, 1.0) if enb > 0 else 0.0
    enb_score = 25.0 * enb_ratio

    # Position balance score (25 points)
    # eff_n / n_active close to 1 = balanced weights
    n_active_safe = max(n_active_positions, 1)
    position_ratio = eff_n_positions / n_active_safe if eff_n_positions > 0 else 0.0
    if position_ratio >= 0.7:
        position_score = 25.0
    elif position_ratio >= 0.3:
        position_score = 25.0 * (position_ratio / 0.7)
    else:
        position_score = 10.0  # Very concentrated

    position_score = np.clip(position_score, 0, 25)

    # Gini score (10 points)
    # Gini = 0 (equal) → 10 pts, Gini = 1 (one stock) → 0 pts
    gini_score = 10.0 * (1.0 - np.clip(gini, 0, 1))

    total_score = entropy_score + enb_score + position_score + gini_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    if total_score >= 80:
        interpretation = (
            f"Excellent factor diversification. H_norm={h_norm:.3f}, "
            f"ENB={enb:.1f}/{n_signal}, eff_N={eff_n_positions:.1f}."
        )
    elif total_score >= 60:
        interpretation = (
            f"Acceptable diversification. H_norm={h_norm:.3f}, ENB={enb:.1f}."
        )
    else:
        issues = []
        if h_norm < 0.3:
            issues.append(f"low factor entropy ({h_norm:.2f})")
        if enb_ratio < 0.5:
            issues.append(f"low ENB ({enb:.1f}/{n_signal})")
        if position_ratio < 0.3:
            issues.append("concentrated positions")
        if gini > 0.7:
            issues.append("high Gini")
        interpretation = f"Diversification issues: {', '.join(issues)}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if h_norm < 0.3:
            action = "Factor entropy low — review alpha/entropy tradeoff"
        elif position_ratio < 0.3:
            action = "Positions concentrated — check if w_max is too restrictive"
        elif enb_ratio < 0.5:
            action = "Low ENB — factors may not offer diversification"
        else:
            action = "Review diversification constraints"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "H_norm_signal": H_norm_signal,
            "H_norm_eff": H_norm_eff,
            "h_norm_used": h_norm,
            "enb": enb,
            "n_signal": n_signal,
            "enb_ratio": enb_ratio,
            "eff_n_positions": eff_n_positions,
            "n_active_positions": n_active_positions,
            "position_ratio": position_ratio,
            "gini_coefficient": gini,
            "component_scores": {
                "entropy": float(entropy_score),
                "enb": float(enb_score),
                "position_balance": float(position_score),
                "gini": float(gini_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Factor Stability Score
# ---------------------------------------------------------------------------

def compute_factor_stability_score(
    factor_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute factor stability score.

    Evaluates:
    - Latent stability (50%): Spearman rho between folds
    - Factor composition (30%): Percentage of structural factors
    - AU consistency (20%): Agreement with Bai-Ng IC2

    :param factor_quality (dict | None): From factor_quality_diagnostics()

    :return result (dict): score, grade, interpretation, action, details
    """
    if factor_quality is None or not factor_quality.get("available", False):
        return {
            "available": False,
            "score": 50.0,
            "grade": "C",
            "interpretation": "Factor quality diagnostics not available",
            "action": None,
        }

    stability_rho = factor_quality.get("latent_stability_rho")
    pct_structural = factor_quality.get("pct_structural", 0.0)
    n_structural = factor_quality.get("n_structural", 0)
    au_bai_ng_diff = factor_quality.get("au_bai_ng_diff")
    au_onatski_diff = factor_quality.get("au_onatski_diff")
    AU = factor_quality.get("AU", 0)

    # Latent stability score (50 points)
    # rho >= 0.85: very stable
    # rho < 0.70: unstable
    # None (single fold): neutral 35 pts
    if stability_rho is None or np.isnan(stability_rho):
        stability_score = 35.0  # Neutral for single fold
        stability_status = "N/A (single fold)"
    elif stability_rho >= 0.85:
        stability_score = 50.0
        stability_status = "stable"
    elif stability_rho >= 0.70:
        stability_score = 20.0 + 30.0 * (stability_rho - 0.70) / 0.15
        stability_status = "moderate"
    else:
        stability_score = 20.0 * (stability_rho / 0.70)
        stability_status = "unstable"

    stability_score = np.clip(stability_score, 0, 50)

    # Composition score (30 points)
    # >50% structural factors: persistent factors dominate
    # <20% structural: dominated by episodic
    if pct_structural >= 0.50:
        composition_score = 30.0
    elif pct_structural >= 0.20:
        composition_score = 15.0 + 15.0 * (pct_structural - 0.20) / 0.30
    else:
        composition_score = 15.0 * (pct_structural / 0.20)

    composition_score = np.clip(composition_score, 0, 30)

    # AU consistency score (20 points)
    # |AU - k_bai_ng| < 5: consistent
    # |AU - k_bai_ng| > 20: divergent
    if au_bai_ng_diff is not None:
        abs_diff = abs(au_bai_ng_diff)
        if abs_diff < 5:
            consistency_score = 20.0
        elif abs_diff < 10:
            consistency_score = 15.0
        elif abs_diff < 20:
            consistency_score = 10.0
        else:
            consistency_score = 5.0
    else:
        # Not available, neutral
        consistency_score = 10.0

    total_score = stability_score + composition_score + consistency_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    if total_score >= 80:
        interpretation = (
            f"Factor structure robust. Stability={stability_status}, "
            f"{n_structural} structural factors ({pct_structural:.0%})."
        )
    elif total_score >= 60:
        interpretation = (
            f"Factor structure acceptable. {pct_structural:.0%} structural."
        )
        if stability_status == "moderate":
            interpretation += " Stability moderate."
    else:
        issues = []
        if stability_rho is not None and stability_rho < 0.70:
            issues.append(f"low stability (rho={stability_rho:.2f})")
        if pct_structural < 0.20:
            issues.append("few structural factors")
        if au_bai_ng_diff is not None and abs(au_bai_ng_diff) > 20:
            issues.append(f"AU diverges from Bai-Ng by {au_bai_ng_diff}")
        interpretation = f"Factor stability issues: {', '.join(issues) if issues else 'multiple concerns'}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if stability_rho is not None and stability_rho < 0.70:
            action = "Factor structure unstable between folds — increase training window"
        elif pct_structural < 0.20:
            action = "Most factors are episodic — risk of OOS degradation"
        elif au_bai_ng_diff is not None and abs(au_bai_ng_diff) > 20:
            action = "AU count diverges from statistical tests — review KL threshold"
        else:
            action = "Review factor extraction methodology"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "latent_stability_rho": stability_rho,
            "stability_status": stability_status,
            "pct_structural": pct_structural,
            "n_structural": n_structural,
            "au_bai_ng_diff": au_bai_ng_diff,
            "au_onatski_diff": au_onatski_diff,
            "AU": AU,
            "component_scores": {
                "stability": float(stability_score),
                "composition": float(composition_score),
                "consistency": float(consistency_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# VAE Health Score
# ---------------------------------------------------------------------------

def compute_vae_health_score(vae_diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compute VAE posterior quality score.

    Evaluates:
    - Posterior collapse (35%): Fraction of latent units not collapsed
    - Posterior explosion (25%): Fraction of latent units not exploded
    - KL balance (25%): Balance of KL divergence
    - Reconstruction balance (15%): Balance between features

    :param vae_diagnostics (dict | None): VAE posterior diagnostics

    :return result (dict): score, grade, interpretation, action, details
    """
    if vae_diagnostics is None:
        return {
            "available": False,
            "score": 0.0,
            "grade": "F",
            "interpretation": "VAE diagnostics not available",
            "action": "Run pipeline with VAE diagnostics enabled",
        }

    # Extract values with defaults
    collapse_severity = vae_diagnostics.get("collapse_severity", 0.0)
    explosion_severity = vae_diagnostics.get("explosion_severity", 0.0)
    kl_balance = vae_diagnostics.get("kl_balance", 0.5)
    recon_per_feature = vae_diagnostics.get("recon_per_feature", [])

    # Posterior collapse score (35 points)
    # collapse_severity is fraction at lower bound; lower is better
    collapse_score = 35.0 * (1.0 - np.clip(collapse_severity, 0, 1))

    # Posterior explosion score (25 points)
    # explosion_severity is fraction at upper bound; lower is better
    explosion_score = 25.0 * (1.0 - np.clip(explosion_severity, 0, 1))

    # KL balance score (25 points)
    # kl_balance should ideally be around 0.5 (balanced)
    # Score higher when closer to balanced
    kl_deviation = abs(kl_balance - 0.5)
    kl_score = 25.0 * (1.0 - np.clip(kl_deviation * 2, 0, 1))

    # Reconstruction balance score (15 points)
    if len(recon_per_feature) >= 2:
        max_recon = max(recon_per_feature[0], recon_per_feature[1], 1e-10)
        imbalance = abs(recon_per_feature[0] - recon_per_feature[1]) / max_recon
        balance_score = 15.0 * (1.0 - np.clip(imbalance, 0, 1))
        feature_imbalance = imbalance
    else:
        balance_score = 10.0  # Neutral if unavailable
        feature_imbalance = None

    total_score = collapse_score + explosion_score + kl_score + balance_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    issues = []
    if collapse_severity > 0.2:
        issues.append(f"{collapse_severity:.0%} collapsed")
    if explosion_severity > 0.1:
        issues.append(f"{explosion_severity:.0%} exploded")
    if kl_deviation > 0.3:
        issues.append(f"KL imbalance={kl_balance:.2f}")

    if total_score >= 85:
        interpretation = (
            f"VAE posterior healthy. "
            f"Collapse={collapse_severity:.1%}, explosion={explosion_severity:.1%}."
        )
    elif total_score >= 70:
        interpretation = (
            f"VAE posterior acceptable. "
            f"Collapse={collapse_severity:.1%}, explosion={explosion_severity:.1%}."
        )
    elif total_score >= 50:
        interpretation = f"VAE posterior issues: {', '.join(issues) if issues else 'moderate degradation'}."
    else:
        interpretation = f"VAE posterior critical: {', '.join(issues) if issues else 'severe degradation'}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if collapse_severity > 0.3:
            action = "Posterior collapse detected; reduce beta or KL annealing"
        elif explosion_severity > 0.2:
            action = "Posterior explosion detected; increase regularization"
        elif kl_deviation > 0.3:
            action = "KL imbalance detected; review loss weights"
        else:
            action = "Review VAE training hyperparameters"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "collapse_severity": collapse_severity,
            "explosion_severity": explosion_severity,
            "kl_balance": kl_balance,
            "feature_imbalance": feature_imbalance,
            "recon_per_feature": recon_per_feature if len(recon_per_feature) >= 2 else None,
            "component_scores": {
                "collapse": float(collapse_score),
                "explosion": float(explosion_score),
                "kl_balance": float(kl_score),
                "reconstruction_balance": float(balance_score),
            },
        },
    }


# ---------------------------------------------------------------------------
# Factor Model Score
# ---------------------------------------------------------------------------

def compute_factor_model_score(factor_diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compute factor model quality score.

    Evaluates:
    - Eigenvalue quality (30%): Based on concentration ratio and gap structure
    - Regression stability (30%): 1 - rank deficiency rate
    - R-squared consistency (25%): cs_r2_mean / (1 + cs_r2_std)
    - z_hat normality (15%): 1 - fraction of extreme z_hat values

    :param factor_diagnostics (dict | None): Factor model diagnostics

    :return result (dict): score, grade, interpretation, action, details
    """
    if factor_diagnostics is None:
        return {
            "available": False,
            "score": 0.0,
            "grade": "F",
            "interpretation": "Factor model diagnostics not available",
            "action": "Run pipeline with factor model diagnostics enabled",
        }

    # Extract values with defaults
    concentration_ratio = factor_diagnostics.get("concentration_ratio", 0.0)
    rank_deficiency_rate = factor_diagnostics.get("rank_deficiency_rate", 0.0)
    cs_r2_mean = factor_diagnostics.get("cs_r2_mean", 0.0)
    cs_r2_std = factor_diagnostics.get("cs_r2_std", 0.0)
    n_extreme_z_hat = factor_diagnostics.get("n_extreme_z_hat", 0)
    n_dates = factor_diagnostics.get("n_dates", 1)

    # Eigenvalue quality score (30 points)
    # concentration_ratio > 0.5 is good (factors explain more than residual)
    # Ideal: 0.6-0.9
    if concentration_ratio >= 0.6:
        eigen_score = 30.0 * min(concentration_ratio / 0.9, 1.0)
    elif concentration_ratio >= 0.3:
        eigen_score = 30.0 * (concentration_ratio / 0.6)
    else:
        eigen_score = 30.0 * (concentration_ratio / 0.3) * 0.5

    # Regression stability score (30 points)
    # rank_deficiency_rate should be low (ideally 0)
    stability_score = 30.0 * (1.0 - np.clip(rank_deficiency_rate, 0, 1))

    # R-squared consistency score (25 points)
    # Higher mean and lower std is better
    if cs_r2_mean > 0:
        r2_normalized = cs_r2_mean / (1.0 + cs_r2_std)
        # Scale: 0.1 normalized -> full score
        r2_score = 25.0 * min(r2_normalized / 0.1, 1.0)
    else:
        r2_score = 0.0

    # z_hat normality score (15 points)
    # Fewer extreme values is better
    n_dates = max(n_dates, 1)  # Avoid division by zero
    extreme_fraction = n_extreme_z_hat / n_dates
    normality_score = 15.0 * (1.0 - np.clip(extreme_fraction * 10, 0, 1))

    total_score = eigen_score + stability_score + r2_score + normality_score
    total_score = np.clip(total_score, 0, 100)

    # Generate interpretation
    parts = []
    parts.append(f"conc_ratio={concentration_ratio:.2f}")
    if rank_deficiency_rate > 0:
        parts.append(f"rank_def={rank_deficiency_rate:.1%}")
    parts.append(f"R2={cs_r2_mean:.3f}±{cs_r2_std:.3f}")

    if total_score >= 80:
        interpretation = f"Factor model well-specified. {', '.join(parts)}."
    elif total_score >= 60:
        interpretation = f"Factor model acceptable. {', '.join(parts)}."
    elif total_score >= 40:
        interpretation = f"Factor model issues. {', '.join(parts)}."
    else:
        interpretation = f"Factor model critical issues. {', '.join(parts)}."

    # Generate action if needed
    action = None
    if total_score < 60:
        if rank_deficiency_rate > 0.1:
            action = "High rank deficiency; reduce factor count or increase sample size"
        elif concentration_ratio < 0.3:
            action = "Low factor explanatory power; review factor extraction"
        elif extreme_fraction > 0.1:
            action = "Many extreme z_hat values; check for outliers or misspecification"
        else:
            action = "Review factor model specification"

    return {
        "available": True,
        "score": float(total_score),
        "grade": _get_grade(total_score),
        "interpretation": interpretation,
        "action": action,
        "details": {
            "concentration_ratio": concentration_ratio,
            "rank_deficiency_rate": rank_deficiency_rate,
            "cs_r2_mean": cs_r2_mean,
            "cs_r2_std": cs_r2_std,
            "n_extreme_z_hat": n_extreme_z_hat,
            "n_dates": n_dates,
            "extreme_fraction": extreme_fraction,
            "component_scores": {
                "eigenvalue": float(eigen_score),
                "stability": float(stability_score),
                "r2_consistency": float(r2_score),
                "normality": float(normality_score),
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
    vae_health: dict[str, Any] | None = None,
    factor_model: dict[str, Any] | None = None,
    training_convergence: dict[str, Any] | None = None,
    active_unit: dict[str, Any] | None = None,
    portfolio_diversification: dict[str, Any] | None = None,
    factor_stability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute weighted overall diagnostic score.

    Weights (10 components, total = 100%):
    - Covariance: 20% (risk model accuracy is critical)
    - Solver: 15% (must find actual optimum)
    - Reconstruction: 12% (VAE must learn meaningful factors)
    - VAE Health: 10% (posterior quality)
    - Factor Model: 10% (factor model specification)
    - Constraints: 8% (some pressure is acceptable)
    - Training Convergence: 8% (VAE training quality)
    - Active Unit: 7% (latent space utilization)
    - Portfolio Diversification: 5% (diversification quality)
    - Factor Stability: 5% (factor temporal stability)

    :param solver (dict): From compute_solver_health_score()
    :param constraint (dict): From compute_constraint_pressure_score()
    :param covariance (dict): From compute_covariance_quality_score()
    :param reconstruction (dict): From compute_reconstruction_balance_score()
    :param vae_health (dict | None): From compute_vae_health_score()
    :param factor_model (dict | None): From compute_factor_model_score()
    :param training_convergence (dict | None): From compute_training_convergence_score()
    :param active_unit (dict | None): From compute_active_unit_score()
    :param portfolio_diversification (dict | None): From compute_portfolio_diversification_score()
    :param factor_stability (dict | None): From compute_factor_stability_score()

    :return result (dict): score, grade, status, summary, priority_actions
    """
    weights = {
        "solver": 0.15,
        "constraint": 0.08,
        "covariance": 0.20,
        "reconstruction": 0.12,
        "vae_health": 0.10,
        "factor_model": 0.10,
        "training_convergence": 0.08,
        "active_unit": 0.07,
        "portfolio_diversification": 0.05,
        "factor_stability": 0.05,
    }

    components: dict[str, dict[str, Any]] = {
        "solver": solver,
        "constraint": constraint,
        "covariance": covariance,
        "reconstruction": reconstruction,
    }

    # Add optional components if provided
    if vae_health is not None:
        components["vae_health"] = vae_health
    if factor_model is not None:
        components["factor_model"] = factor_model
    if training_convergence is not None:
        components["training_convergence"] = training_convergence
    if active_unit is not None:
        components["active_unit"] = active_unit
    if portfolio_diversification is not None:
        components["portfolio_diversification"] = portfolio_diversification
    if factor_stability is not None:
        components["factor_stability"] = factor_stability

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
    component_names = [
        "solver", "constraint", "covariance", "reconstruction",
        "vae_health", "factor_model",
        "training_convergence", "active_unit",
        "portfolio_diversification", "factor_stability",
    ]
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
    vae_diagnostics: dict[str, Any] | None = None,
    factor_diagnostics: dict[str, Any] | None = None,
    latent: dict[str, Any] | None = None,
    portfolio: dict[str, Any] | None = None,
    factor_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute all composite scores in one call.

    :param solver_stats (dict | None): From solver_diagnostics()
    :param constraints (dict | None): From constraint_binding_diagnostics()
    :param risk_model (dict | None): From risk_model_diagnostics()
    :param training (dict | None): From training_diagnostics()
    :param n_active (int): Number of active positions
    :param vae_diagnostics (dict | None): VAE posterior diagnostics
    :param factor_diagnostics (dict | None): Factor model diagnostics
    :param latent (dict | None): From latent_diagnostics()
    :param portfolio (dict | None): From portfolio_diagnostics()
    :param factor_quality (dict | None): From factor_quality_diagnostics()

    :return scores (dict): All composite scores including overall
    """
    # Original 6 scores
    solver = compute_solver_health_score(solver_stats)
    constraint = compute_constraint_pressure_score(constraints, n_active)
    covariance = compute_covariance_quality_score(risk_model)
    reconstruction = compute_reconstruction_balance_score(training)
    vae_health = compute_vae_health_score(vae_diagnostics)
    factor_model = compute_factor_model_score(factor_diagnostics)

    # 4 new scores
    training_convergence = compute_training_convergence_score(training)
    active_unit = compute_active_unit_score(training, latent)
    portfolio_diversification = compute_portfolio_diversification_score(portfolio)
    factor_stability = compute_factor_stability_score(factor_quality)

    # Overall score with all 10 components
    overall = compute_overall_score(
        solver=solver,
        constraint=constraint,
        covariance=covariance,
        reconstruction=reconstruction,
        vae_health=vae_health,
        factor_model=factor_model,
        training_convergence=training_convergence,
        active_unit=active_unit,
        portfolio_diversification=portfolio_diversification,
        factor_stability=factor_stability,
    )

    return {
        "solver": solver,
        "constraint": constraint,
        "covariance": covariance,
        "reconstruction": reconstruction,
        "vae_health": vae_health,
        "factor_model": factor_model,
        "training_convergence": training_convergence,
        "active_unit": active_unit,
        "portfolio_diversification": portfolio_diversification,
        "factor_stability": factor_stability,
        "overall": overall,
    }
