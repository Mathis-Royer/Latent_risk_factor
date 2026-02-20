"""
Decision rules engine for automated diagnostic interpretation.

Provides structured decision rules, causal graphs, and metric patterns
to transform composite scores into actionable diagnoses. Used by the
diagnostic report generator to provide root cause analysis.

Reference: docs/diagnostic.md for full interpretation guide.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Decision Rules
# ---------------------------------------------------------------------------

# Each rule defines a scenario with conditions, diagnosis, root causes, and actions.
# Conditions use comparison operators: "<", ">", "<=", ">=", "==", "between"
# "between" format: "between:low:high" (inclusive)

DECISION_RULES: list[dict[str, Any]] = [
    {
        "id": "PURE_OPTIMIZATION",
        "description": "Optimization struggling but constraints are not the issue",
        "condition": {
            "solver_score": {"op": "<", "value": 60},
            "constraint_score": {"op": ">", "value": 80},
        },
        "diagnosis": "Optimization problem, not constraint-related",
        "root_causes": [
            "ill_conditioned_objective",
            "step_size_too_aggressive",
            "local_minima_traps",
        ],
        "actions": [
            "Increase sca_max_iter (e.g., 100 -> 200)",
            "Reduce armijo_rho (e.g., 0.5 -> 0.3)",
            "Increase n_multi_start for better exploration",
        ],
        "confidence": 0.85,
        "severity": "high",
    },
    {
        "id": "CONSTRAINT_DOMINATED",
        "description": "Constraints are too tight, limiting optimization freedom",
        "condition": {
            "solver_score": {"op": ">", "value": 70},
            "constraint_score": {"op": "<", "value": 50},
        },
        "diagnosis": "Constraints limiting optimization freedom",
        "root_causes": [
            "w_max_too_restrictive",
            "turnover_cap_binding",
            "cardinality_constraint_too_tight",
        ],
        "actions": [
            "Increase w_max (e.g., 0.05 -> 0.07)",
            "Increase tau_max or reduce rebalancing frequency",
            "Review cardinality targets vs universe size",
        ],
        "confidence": 0.80,
        "severity": "medium",
    },
    {
        "id": "COVARIANCE_DEGRADATION",
        "description": "Risk model calibration issues affecting portfolio quality",
        "condition": {
            "covariance_score": {"op": "<", "value": 55},
        },
        "diagnosis": "Risk model miscalibrated",
        "root_causes": [
            "insufficient_estimation_history",
            "factor_instability",
            "shrinkage_overcorrection",
            "structural_break_in_data",
        ],
        "actions": [
            "Extend sigma_z_lookback period",
            "Check for regime changes in data",
            "Review Ledoit-Wolf shrinkage intensity",
            "Consider increasing ridge regularization",
        ],
        "confidence": 0.75,
        "severity": "high",
    },
    {
        "id": "VAE_COLLAPSE",
        "description": "VAE posterior collapse with poor reconstruction",
        "condition": {
            "reconstruction_score": {"op": "<", "value": 50},
            "vae_health_score": {"op": "<", "value": 50},
        },
        "diagnosis": "VAE posterior collapse detected",
        "root_causes": [
            "beta_too_high",
            "learning_rate_too_high",
            "insufficient_capacity",
            "kl_annealing_too_aggressive",
        ],
        "actions": [
            "Reduce beta in loss function",
            "Lower learning rate",
            "Increase latent dimension K",
            "Use slower KL annealing schedule",
        ],
        "confidence": 0.90,
        "severity": "critical",
    },
    {
        "id": "FACTOR_MODEL_WEAK",
        "description": "Factor model has low explanatory power",
        "condition": {
            "factor_model_score": {"op": "<", "value": 50},
        },
        "diagnosis": "Factor model specification issues",
        "root_causes": [
            "too_few_active_units",
            "factor_exposure_noise",
            "poor_factor_regression_fit",
            "misaligned_factor_estimation",
        ],
        "actions": [
            "Review AU count vs Bai-Ng optimal factors",
            "Check factor loading distribution for sparsity",
            "Increase training data or extend window",
            "Verify cross-sectional regression quality",
        ],
        "confidence": 0.70,
        "severity": "high",
    },
    {
        "id": "OVERALL_DEGRADATION",
        "description": "Multiple components degraded simultaneously",
        "condition": {
            "solver_score": {"op": "<", "value": 60},
            "covariance_score": {"op": "<", "value": 60},
            "reconstruction_score": {"op": "<", "value": 60},
        },
        "diagnosis": "Systemic pipeline degradation",
        "root_causes": [
            "data_quality_issues",
            "regime_change",
            "configuration_mismatch",
            "insufficient_training_data",
        ],
        "actions": [
            "Check input data quality and preprocessing",
            "Review recent configuration changes",
            "Verify training/validation split",
            "Consider extending training window",
        ],
        "confidence": 0.85,
        "severity": "critical",
    },
    {
        "id": "HEALTHY_PIPELINE",
        "description": "All components operating within normal parameters",
        "condition": {
            "solver_score": {"op": ">", "value": 75},
            "constraint_score": {"op": ">", "value": 60},
            "covariance_score": {"op": ">", "value": 70},
            "reconstruction_score": {"op": ">", "value": 70},
        },
        "diagnosis": "Pipeline operating normally",
        "root_causes": [],
        "actions": ["Continue monitoring", "Consider optimizing for efficiency"],
        "confidence": 0.95,
        "severity": "none",
    },
    {
        "id": "RECONSTRUCTION_IMBALANCE",
        "description": "Feature reconstruction significantly imbalanced",
        "condition": {
            "reconstruction_score": {"op": "between", "low": 40, "high": 70},
            "vae_health_score": {"op": ">", "value": 60},
        },
        "diagnosis": "Feature reconstruction imbalanced (returns vs volatility)",
        "root_causes": [
            "loss_weight_imbalance",
            "feature_scale_mismatch",
            "volatility_preprocessing_issues",
        ],
        "actions": [
            "Verify z-scoring is applied per-feature",
            "Check loss function feature weighting",
            "Review volatility calculation window",
        ],
        "confidence": 0.70,
        "severity": "medium",
    },
    {
        "id": "OVERFITTING_DETECTED",
        "description": "Training shows clear overfitting pattern",
        "condition": {
            "reconstruction_score": {"op": "<", "value": 55},
        },
        "extra_checks": ["overfit_ratio > 1.3"],
        "diagnosis": "VAE overfitting to training data",
        "root_causes": [
            "model_too_complex",
            "insufficient_regularization",
            "training_too_long",
            "small_validation_set",
        ],
        "actions": [
            "Increase dropout rate",
            "Reduce model capacity (smaller K)",
            "Enable/strengthen early stopping",
            "Review train/val split ratio",
        ],
        "confidence": 0.85,
        "severity": "high",
    },
    {
        "id": "SOLVER_CONSTRAINT_CONFLICT",
        "description": "Both solver and constraints struggling",
        "condition": {
            "solver_score": {"op": "<", "value": 60},
            "constraint_score": {"op": "<", "value": 60},
        },
        "diagnosis": "Infeasible or near-infeasible optimization problem",
        "root_causes": [
            "conflicting_constraints",
            "initial_point_far_from_feasible",
            "numerical_precision_issues",
        ],
        "actions": [
            "Check constraint compatibility (w_min + cardinality feasibility)",
            "Verify turnover constraint allows transition from previous weights",
            "Review numerical conditioning of the problem",
            "Consider constraint relaxation hierarchy",
        ],
        "confidence": 0.80,
        "severity": "critical",
    },
]


# ---------------------------------------------------------------------------
# Causal Graph
# ---------------------------------------------------------------------------

# Maps metrics to their upstream causes and downstream effects.
# Used for root cause tracing and impact analysis.

CAUSAL_GRAPH: dict[str, dict[str, list[str]]] = {
    # VAE/Training metrics
    "AU": {
        "upstream": ["posterior_collapse", "beta_schedule", "K_capacity", "training_epochs"],
        "downstream": ["entropy", "factor_quality", "factor_count_mismatch"],
    },
    "posterior_collapse": {
        "upstream": ["beta_too_high", "learning_rate", "decoder_weakness"],
        "downstream": ["AU", "reconstruction_loss", "factor_diversity"],
    },
    "posterior_explosion": {
        "upstream": ["beta_too_low", "insufficient_regularization", "numerical_instability"],
        "downstream": ["AU", "factor_interpretability", "covariance_quality"],
    },
    "reconstruction_loss": {
        "upstream": ["model_capacity", "training_epochs", "data_quality", "encoder_expressiveness"],
        "downstream": ["factor_quality", "latent_informativeness"],
    },
    "kl_divergence": {
        "upstream": ["beta", "posterior_variance", "prior_mismatch"],
        "downstream": ["AU", "latent_regularization"],
    },

    # Risk model metrics
    "shrinkage_intensity": {
        "upstream": ["sample_size", "n_factors", "eigenvalue_spectrum", "noise_level"],
        "downstream": ["var_ratio", "condition_number", "covariance_stability"],
    },
    "condition_number": {
        "upstream": ["shrinkage_intensity", "factor_correlation", "near_singularity"],
        "downstream": ["solver_convergence", "portfolio_stability", "weight_precision"],
    },
    "var_ratio": {
        "upstream": ["covariance_estimation", "factor_model_fit", "regime_change"],
        "downstream": ["risk_forecast_accuracy", "portfolio_risk"],
    },
    "explanatory_power": {
        "upstream": ["n_factors", "factor_relevance", "regression_quality"],
        "downstream": ["idiosyncratic_dominance", "diversification_potential"],
    },

    # Factor model metrics
    "bai_ng_k": {
        "upstream": ["eigenvalue_structure", "sample_size", "true_factor_count"],
        "downstream": ["AU_mismatch", "factor_model_specification"],
    },
    "latent_stability": {
        "upstream": ["regime_change", "training_consistency", "universe_changes"],
        "downstream": ["portfolio_turnover", "strategy_reliability"],
    },
    "factor_persistence": {
        "upstream": ["factor_nature", "market_regime", "data_frequency"],
        "downstream": ["factor_usefulness", "rebalancing_frequency"],
    },

    # Portfolio metrics
    "solver_convergence": {
        "upstream": ["objective_conditioning", "constraint_tightness", "step_size"],
        "downstream": ["portfolio_optimality", "weight_accuracy"],
    },
    "constraint_binding": {
        "upstream": ["constraint_parameters", "optimal_unconstrained", "turnover_path"],
        "downstream": ["portfolio_suboptimality", "diversification_reduction"],
    },
    "entropy": {
        "upstream": ["AU", "factor_exposures", "weight_distribution"],
        "downstream": ["factor_diversification", "strategy_robustness"],
    },
    "turnover": {
        "upstream": ["latent_stability", "regime_change", "rebalancing_frequency"],
        "downstream": ["transaction_costs", "implementation_feasibility"],
    },
}


# ---------------------------------------------------------------------------
# Metric Patterns
# ---------------------------------------------------------------------------

# Cross-metric patterns that indicate specific issues.
# Each pattern has a set of indicators and an interpretation.

METRIC_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "BETA_SCHEDULE_ISSUE",
        "name": "Beta schedule misconfiguration",
        "indicators": {
            "au_trend": "decreasing",
            "kl_divergence": "low",
            "reconstruction_loss": "low",
        },
        "interpretation": "KL term dominated by beta, causing posterior collapse",
        "recommendation": "Reduce beta or use KL annealing warmup",
    },
    {
        "id": "DATA_REGIME_SHIFT",
        "name": "Data regime change detected",
        "indicators": {
            "latent_stability": "low",
            "var_ratio": "off_target",
            "factor_persistence": "low",
        },
        "interpretation": "Market regime changed, factor structure may be outdated",
        "recommendation": "Retrain model or reduce lookback windows",
    },
    {
        "id": "SAMPLE_SIZE_INSUFFICIENT",
        "name": "Insufficient estimation sample",
        "indicators": {
            "shrinkage_intensity": "high",
            "condition_number": "high",
            "explanatory_power": "low",
        },
        "interpretation": "Not enough data for reliable covariance estimation",
        "recommendation": "Extend lookback or reduce factor count",
    },
    {
        "id": "CONSTRAINT_FEASIBILITY",
        "name": "Constraint feasibility issues",
        "indicators": {
            "binding_fraction": "high",
            "solver_convergence": "poor",
            "turnover_binding": True,
        },
        "interpretation": "Constraints may be mutually incompatible or too tight",
        "recommendation": "Review constraint hierarchy and relax secondary constraints",
    },
    {
        "id": "FACTOR_COUNT_MISMATCH",
        "name": "Factor count mismatch",
        "indicators": {
            "au_vs_bai_ng": "large_gap",
            "factor_model_score": "low",
        },
        "interpretation": "VAE active units differ significantly from optimal factor count",
        "recommendation": "Adjust K or review VAE training hyperparameters",
    },
    {
        "id": "IDIOSYNCRATIC_DOMINANCE",
        "name": "Idiosyncratic risk dominates",
        "indicators": {
            "explanatory_power": "very_low",
            "entropy_contribution": "factor_low",
        },
        "interpretation": "Factor model explains little variance; diversification limited",
        "recommendation": "Review factor extraction or increase entropy_idio_weight",
    },
    {
        "id": "NUMERICAL_INSTABILITY",
        "name": "Numerical precision issues",
        "indicators": {
            "condition_number": "very_high",
            "solver_convergence": "erratic",
            "weight_extremes": True,
        },
        "interpretation": "Numerical precision problems affecting optimization",
        "recommendation": "Increase ridge regularization; check input scaling",
    },
    {
        "id": "TRAINING_CONVERGENCE_INCOMPLETE",
        "name": "Training did not converge",
        "indicators": {
            "val_elbo_trend": "still_decreasing",
            "best_epoch_fraction": "late",
            "overfit_ratio": "low",
        },
        "interpretation": "Model may benefit from additional training epochs",
        "recommendation": "Increase max_epochs or adjust early stopping patience",
    },
]


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def _evaluate_condition(condition: dict[str, Any], scores: dict[str, float]) -> bool:  # noqa: ARG001
    """
    Evaluate a single condition against scores.

    :param condition (dict): Condition specification with op and value
    :param scores (dict): Score values by name

    :return result (bool): True if condition is satisfied
    """
    op = condition.get("op", "==")
    value = condition.get("value")
    low = condition.get("low")
    high = condition.get("high")

    # Get the score to compare (key is inferred from parent dict)
    score = condition.get("_score_value")
    if score is None:
        return False

    if op == "<":
        return score < value
    elif op == ">":
        return score > value
    elif op == "<=":
        return score <= value
    elif op == ">=":
        return score >= value
    elif op == "==":
        return score == value
    elif op == "between":
        return low <= score <= high
    else:
        return False


def evaluate_decision_rules(scores: dict[str, float]) -> list[dict[str, Any]]:
    """
    Evaluate all decision rules against provided scores.

    :param scores (dict): Dictionary of scores with keys like:
        - solver_score
        - constraint_score
        - covariance_score
        - reconstruction_score
        - vae_health_score
        - factor_model_score
        - overall_score

    :return matches (list[dict]): List of matching rules with full details
    """
    matches: list[dict[str, Any]] = []

    for rule in DECISION_RULES:
        conditions = rule.get("condition", {})
        all_met = True

        for score_key, condition in conditions.items():
            # Map score key to scores dict
            score_value = scores.get(score_key)
            if score_value is None:
                # Try without _score suffix
                alt_key = score_key.replace("_score", "")
                score_value = scores.get(alt_key)

            if score_value is None:
                all_met = False
                break

            # Add score value to condition for evaluation
            condition_with_value = {**condition, "_score_value": score_value}
            if not _evaluate_condition(condition_with_value, scores):
                all_met = False
                break

        if all_met:
            matches.append({
                "rule_id": rule["id"],
                "description": rule.get("description", ""),
                "diagnosis": rule["diagnosis"],
                "root_causes": rule["root_causes"],
                "actions": rule["actions"],
                "confidence": rule["confidence"],
                "severity": rule["severity"],
            })

    # Sort by severity (critical > high > medium > low > none)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    matches.sort(key=lambda x: severity_order.get(x["severity"], 5))

    return matches


def detect_metric_patterns(
    scores: dict[str, float],
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect cross-metric patterns in the scores and diagnostics.

    :param scores (dict): Composite scores dictionary
    :param diagnostics (dict | None): Optional raw diagnostic values for pattern matching

    :return patterns (list[dict]): List of detected patterns with interpretation
    """
    detected: list[dict[str, Any]] = []
    diagnostics = diagnostics or {}

    for pattern in METRIC_PATTERNS:
        pattern_id = pattern["id"]
        _ = pattern["indicators"]  # Reserved for future pattern-specific logic
        matches_pattern = False

        # Check specific pattern conditions
        if pattern_id == "BETA_SCHEDULE_ISSUE":
            vae_score = scores.get("vae_health_score", 100)
            recon_score = scores.get("reconstruction_score", 100)
            if vae_score < 50 and recon_score > 70:
                matches_pattern = True

        elif pattern_id == "DATA_REGIME_SHIFT":
            cov_score = scores.get("covariance_score", 100)
            latent_stability = diagnostics.get("latent_stability_rho")
            if cov_score < 60 and latent_stability is not None and latent_stability < 0.85:
                matches_pattern = True

        elif pattern_id == "SAMPLE_SIZE_INSUFFICIENT":
            cov_score = scores.get("covariance_score", 100)
            shrinkage = diagnostics.get("shrinkage_intensity")
            if cov_score < 60 and shrinkage is not None and shrinkage > 0.7:
                matches_pattern = True

        elif pattern_id == "CONSTRAINT_FEASIBILITY":
            solver_score = scores.get("solver_score", 100)
            constraint_score = scores.get("constraint_score", 100)
            if solver_score < 60 and constraint_score < 60:
                matches_pattern = True

        elif pattern_id == "FACTOR_COUNT_MISMATCH":
            factor_score = scores.get("factor_model_score", 100)
            au = diagnostics.get("AU")
            bai_ng_k = diagnostics.get("k_bai_ng")
            if factor_score < 60 and au is not None and bai_ng_k is not None:
                if abs(au - bai_ng_k) > 10:
                    matches_pattern = True

        elif pattern_id == "IDIOSYNCRATIC_DOMINANCE":
            factor_score = scores.get("factor_model_score", 100)
            ep = diagnostics.get("explanatory_power")
            if factor_score < 50 and ep is not None and ep < 0.05:
                matches_pattern = True

        elif pattern_id == "NUMERICAL_INSTABILITY":
            solver_score = scores.get("solver_score", 100)
            cond_num = diagnostics.get("condition_number")
            if solver_score < 50 and cond_num is not None and cond_num > 1e8:
                matches_pattern = True

        elif pattern_id == "TRAINING_CONVERGENCE_INCOMPLETE":
            recon_score = scores.get("reconstruction_score", 100)
            overfit_ratio = diagnostics.get("overfit_ratio")
            still_decreasing = diagnostics.get("still_decreasing", False)
            if recon_score < 70 and overfit_ratio is not None and overfit_ratio < 0.95:
                if still_decreasing:
                    matches_pattern = True

        if matches_pattern:
            detected.append({
                "pattern_id": pattern_id,
                "name": pattern["name"],
                "interpretation": pattern["interpretation"],
                "recommendation": pattern["recommendation"],
            })

    return detected


def trace_causal_chain(
    metric: str,
    direction: str = "upstream",
    max_depth: int = 3,
) -> list[str]:
    """
    Trace causal chain from a metric to its causes or effects.

    :param metric (str): Starting metric name
    :param direction (str): "upstream" for causes, "downstream" for effects
    :param max_depth (int): Maximum depth to trace

    :return chain (list[str]): List of related metrics in causal order
    """
    if direction not in ("upstream", "downstream"):
        raise ValueError(f"direction must be 'upstream' or 'downstream', got {direction}")

    visited: set[str] = set()
    chain: list[str] = []

    def _trace(current: str, depth: int) -> None:
        if depth > max_depth or current in visited:
            return
        visited.add(current)

        if current in CAUSAL_GRAPH:
            related = CAUSAL_GRAPH[current].get(direction, [])
            for r in related:
                if r not in visited:
                    chain.append(r)
                    _trace(r, depth + 1)

    _trace(metric, 0)
    return chain


def get_root_cause_analysis(
    scores: dict[str, float],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform comprehensive root cause analysis.

    Combines decision rules, pattern detection, and causal tracing
    to produce actionable insights.

    :param scores (dict): Composite scores dictionary
    :param diagnostics (dict | None): Optional raw diagnostic values

    :return analysis (dict): Complete root cause analysis
    """
    # Evaluate decision rules
    matching_rules = evaluate_decision_rules(scores)

    # Detect metric patterns
    patterns = detect_metric_patterns(scores, diagnostics)

    # Identify lowest-scoring component for causal analysis
    component_scores: dict[str, float] = {
        "solver": float(scores.get("solver_score", 100)),
        "constraint": float(scores.get("constraint_score", 100)),
        "covariance": float(scores.get("covariance_score", 100)),
        "reconstruction": float(scores.get("reconstruction_score", 100)),
        "vae_health": float(scores.get("vae_health_score", 100)),
        "factor_model": float(scores.get("factor_model_score", 100)),
    }
    weakest_component: str = min(component_scores, key=lambda k: component_scores[k])
    weakest_score = component_scores[weakest_component]

    # Map component to causal graph metric
    component_to_metric: dict[str, str] = {
        "solver": "solver_convergence",
        "constraint": "constraint_binding",
        "covariance": "var_ratio",
        "reconstruction": "reconstruction_loss",
        "vae_health": "posterior_collapse",
        "factor_model": "explanatory_power",
    }
    causal_metric: str = component_to_metric.get(weakest_component, weakest_component)
    upstream_causes = trace_causal_chain(causal_metric, "upstream")
    downstream_effects = trace_causal_chain(causal_metric, "downstream")

    # Aggregate all actions
    all_actions: list[str] = []
    for rule in matching_rules[:3]:  # Top 3 rules
        all_actions.extend(rule["actions"][:2])  # Top 2 actions per rule
    for pattern in patterns[:2]:  # Top 2 patterns
        all_actions.append(pattern["recommendation"])

    # Deduplicate actions
    seen_actions: set[str] = set()
    unique_actions: list[str] = []
    for action in all_actions:
        if action not in seen_actions:
            seen_actions.add(action)
            unique_actions.append(action)

    # Determine overall severity
    if matching_rules:
        overall_severity = matching_rules[0]["severity"]
    else:
        overall_severity = "none"

    return {
        "matching_rules": matching_rules,
        "detected_patterns": patterns,
        "weakest_component": weakest_component,
        "weakest_score": weakest_score,
        "causal_analysis": {
            "metric": causal_metric,
            "upstream_causes": upstream_causes,
            "downstream_effects": downstream_effects,
        },
        "priority_actions": unique_actions[:5],
        "overall_severity": overall_severity,
    }


def format_diagnosis_summary(analysis: dict[str, Any]) -> str:
    """
    Format root cause analysis into readable summary.

    :param analysis (dict): Output from get_root_cause_analysis()

    :return summary (str): Human-readable diagnosis summary
    """
    lines: list[str] = []

    # Header with severity
    severity = analysis.get("overall_severity", "none")
    severity_label = {
        "critical": "CRITICAL",
        "high": "HIGH",
        "medium": "MEDIUM",
        "low": "LOW",
        "none": "OK",
    }.get(severity, "UNKNOWN")

    lines.append(f"Diagnostic Summary [{severity_label}]")
    lines.append("=" * 40)

    # Weakest component
    weakest = analysis.get("weakest_component", "unknown")
    weakest_score = analysis.get("weakest_score", 0)
    lines.append(f"\nWeakest Component: {weakest} (score: {weakest_score:.1f})")

    # Matching rules
    rules = analysis.get("matching_rules", [])
    if rules:
        lines.append(f"\nDiagnosis ({len(rules)} rules matched):")
        for rule in rules[:3]:
            confidence = rule.get("confidence", 0) * 100
            lines.append(f"  - {rule['diagnosis']} ({confidence:.0f}% confidence)")
            for cause in rule.get("root_causes", [])[:2]:
                lines.append(f"    * Possible cause: {cause}")

    # Patterns
    patterns = analysis.get("detected_patterns", [])
    if patterns:
        lines.append(f"\nDetected Patterns:")
        for pattern in patterns[:2]:
            lines.append(f"  - {pattern['name']}: {pattern['interpretation']}")

    # Priority actions
    actions = analysis.get("priority_actions", [])
    if actions:
        lines.append(f"\nPriority Actions:")
        for i, action in enumerate(actions[:5], 1):
            lines.append(f"  {i}. {action}")

    # Causal chain
    causal = analysis.get("causal_analysis", {})
    if causal.get("upstream_causes"):
        lines.append(f"\nCausal Chain (upstream of {causal.get('metric', '?')}):")
        lines.append(f"  {' -> '.join(causal['upstream_causes'][:4])}")

    return "\n".join(lines)
