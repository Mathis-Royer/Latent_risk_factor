"""
JSON Schema for diagnostic output validation.

Provides a formal schema definition for diagnostic output, enabling:
- Validation of diagnostic reports
- Documentation of expected format
- Interoperability with external tools

Usage:
    from src.integration.diagnostic_schema import validate_diagnostic_output, get_schema

    # Validate diagnostic output
    is_valid, errors = validate_diagnostic_output(diagnostic_dict)
    if not is_valid:
        print(f"Validation errors: {errors}")
"""

from typing import Any


# ---------------------------------------------------------------------------
# JSON Schema Definition
# ---------------------------------------------------------------------------

DIAGNOSTIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://latent-risk-factor/diagnostic-output.schema.json",
    "title": "Diagnostic Output Schema",
    "description": "Schema for VAE Latent Risk Factor pipeline diagnostic output",
    "type": "object",
    "properties": {
        # Core fields
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Weighted overall diagnostic score (0-100)",
        },
        "severity": {
            "type": "string",
            "enum": ["NOMINAL", "MODERATE", "CRITICAL"],
            "description": "Overall severity classification",
        },
        "verdict": {
            "type": "string",
            "maxLength": 200,
            "description": "One-line summary of diagnostic findings",
        },
        "grade": {
            "type": "string",
            "enum": ["A", "B", "C", "D", "F"],
            "description": "Letter grade based on overall score",
        },
        "status": {
            "type": "string",
            "enum": ["EXCELLENT", "GOOD", "NEEDS ATTENTION", "MARGINAL", "CRITICAL"],
            "description": "Status label based on overall score",
        },

        # Component scores
        "component_scores": {
            "type": "object",
            "properties": {
                "solver": {"$ref": "#/$defs/componentScore"},
                "constraint": {"$ref": "#/$defs/componentScore"},
                "covariance": {"$ref": "#/$defs/componentScore"},
                "reconstruction": {"$ref": "#/$defs/componentScore"},
                "vae_health": {"$ref": "#/$defs/componentScore"},
                "factor_model": {"$ref": "#/$defs/componentScore"},
            },
            "description": "Per-component diagnostic scores",
        },

        # Key findings
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Name of the diagnostic metric",
                    },
                    "value": {
                        "type": ["number", "string", "null"],
                        "description": "Observed value of the metric",
                    },
                    "interpretation": {
                        "type": "string",
                        "description": "Human-readable interpretation",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence level (0-1)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Reference threshold for comparison",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["OK", "WARN", "CRIT"],
                        "description": "Status based on threshold comparison",
                    },
                },
                "required": ["metric", "interpretation"],
            },
            "description": "List of key diagnostic findings",
        },

        # Priority actions
        "priority_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Component generating this action",
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Component score (lower = higher priority)",
                    },
                    "action": {
                        "type": "string",
                        "description": "Recommended action",
                    },
                    "config_key": {
                        "type": "string",
                        "description": "Config parameter to change (if applicable)",
                    },
                    "suggested_value": {
                        "type": ["number", "string", "array"],
                        "description": "Suggested new value (if applicable)",
                    },
                },
                "required": ["component", "action"],
            },
            "maxItems": 10,
            "description": "Prioritized list of recommended actions",
        },

        # Temporal context
        "temporal_context": {
            "type": "object",
            "properties": {
                "fold_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Walk-forward fold index",
                },
                "n_folds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Total number of folds",
                },
                "training_start": {
                    "type": "string",
                    "format": "date",
                    "description": "Training period start date",
                },
                "training_end": {
                    "type": "string",
                    "format": "date",
                    "description": "Training period end date",
                },
                "oos_start": {
                    "type": "string",
                    "format": "date",
                    "description": "Out-of-sample period start date",
                },
                "oos_end": {
                    "type": "string",
                    "format": "date",
                    "description": "Out-of-sample period end date",
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Diagnostic run timestamp",
                },
            },
            "description": "Temporal context for the diagnostic",
        },

        # Cross-fold comparison (optional)
        "cross_fold_comparison": {
            "type": "object",
            "properties": {
                "available": {"type": "boolean"},
                "n_folds": {"type": "integer", "minimum": 1},
                "overall": {
                    "type": "object",
                    "properties": {
                        "mean": {"type": "number"},
                        "std": {"type": "number"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                    },
                },
                "trend": {
                    "type": "string",
                    "enum": ["improving", "stable", "degrading", "insufficient_data"],
                },
                "anomaly_folds": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "stability_verdict": {"type": "string"},
            },
            "description": "Comparison across walk-forward folds",
        },

        # Metadata
        "metadata": {
            "type": "object",
            "properties": {
                "schema_version": {
                    "type": "string",
                    "description": "Schema version identifier",
                },
                "pipeline_version": {
                    "type": "string",
                    "description": "Pipeline version",
                },
                "config_hash": {
                    "type": "string",
                    "description": "Hash of config for reproducibility",
                },
            },
        },
    },
    "required": ["overall_score", "severity", "verdict"],

    # Definitions for reusable schemas
    "$defs": {
        "componentScore": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "grade": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D", "F"],
                },
                "available": {
                    "type": "boolean",
                },
                "interpretation": {
                    "type": "string",
                },
                "action": {
                    "type": ["string", "null"],
                },
                "details": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["score", "grade", "available"],
        },
    },
}


# ---------------------------------------------------------------------------
# Example Output
# ---------------------------------------------------------------------------

EXAMPLE_OUTPUT: dict[str, Any] = {
    "overall_score": 67.3,
    "severity": "MODERATE",
    "verdict": "INVESTIGATE - Covariance calibration issue detected",
    "grade": "C",
    "status": "NEEDS ATTENTION",

    "component_scores": {
        "solver": {
            "score": 82.5,
            "grade": "B",
            "available": True,
            "interpretation": (
                "Reliable solution found. Gradient norm = 1.23e-06, 80% converged."
            ),
            "action": None,
            "details": {
                "grad_norm": 1.23e-6,
                "converged_ratio": 0.8,
                "n_iterations": 45,
            },
        },
        "constraint": {
            "score": 71.2,
            "grade": "B",
            "available": True,
            "interpretation": (
                "Normal constraint pressure. 12 positions at w_max (24%), turnover = 15.2%."
            ),
            "action": None,
            "details": {
                "binding_fraction": 0.24,
                "n_at_w_max": 12,
                "tau_binding": False,
            },
        },
        "covariance": {
            "score": 48.7,
            "grade": "D",
            "available": True,
            "interpretation": (
                "Calibration issues detected. var_ratio=0.42 (overestimates risk), "
                "EP=0.085, shrinkage=0.71, cond=2.3e+05."
            ),
            "action": "Model overestimates risk; check variance targeting scale",
            "details": {
                "condition_number": 2.3e5,
                "var_ratio": 0.42,
                "explanatory_power": 0.085,
                "shrinkage_intensity": 0.71,
            },
        },
        "reconstruction": {
            "score": 75.8,
            "grade": "B",
            "available": True,
            "interpretation": (
                "Acceptable reconstruction. recon_loss=0.0234, "
                "feature_ratio=1.45 (balanced), overfit_ratio=1.08."
            ),
            "action": None,
            "details": {
                "best_recon": 0.0234,
                "overfit_ratio": 1.08,
                "feature_ratio": 1.45,
            },
        },
        "vae_health": {
            "score": 78.4,
            "grade": "B",
            "available": True,
            "interpretation": "VAE posterior acceptable. Collapse=8.2%, explosion=2.1%.",
            "action": None,
            "details": {
                "collapse_severity": 0.082,
                "explosion_severity": 0.021,
                "kl_balance": 0.48,
            },
        },
        "factor_model": {
            "score": 69.5,
            "grade": "C",
            "available": True,
            "interpretation": "Factor model acceptable. conc_ratio=0.58, R2=0.045±0.012.",
            "action": None,
            "details": {
                "concentration_ratio": 0.58,
                "cs_r2_mean": 0.045,
                "cs_r2_std": 0.012,
            },
        },
    },

    "key_findings": [
        {
            "metric": "var_ratio",
            "value": 0.42,
            "interpretation": "Risk model overestimates portfolio variance by 58%",
            "confidence": 0.85,
            "threshold": 1.0,
            "status": "CRIT",
        },
        {
            "metric": "shrinkage_intensity",
            "value": 0.71,
            "interpretation": "High shrinkage indicates limited sample size",
            "confidence": 0.90,
            "threshold": 0.65,
            "status": "WARN",
        },
        {
            "metric": "AU",
            "value": 32,
            "interpretation": "32 active latent dimensions (64% of K=50)",
            "confidence": 0.95,
            "threshold": None,
            "status": "OK",
        },
    ],

    "priority_actions": [
        {
            "component": "covariance",
            "score": 48.7,
            "action": "Model overestimates risk; check variance targeting scale",
            "config_key": "risk_model.sigma_z_shrinkage",
            "suggested_value": "spiked",
        },
        {
            "component": "factor_model",
            "score": 69.5,
            "action": "Review factor model specification",
        },
    ],

    "temporal_context": {
        "fold_index": 5,
        "n_folds": 34,
        "training_start": "2000-01-03",
        "training_end": "2010-12-31",
        "oos_start": "2011-01-03",
        "oos_end": "2011-06-30",
        "timestamp": "2026-02-20T10:30:00Z",
    },

    "metadata": {
        "schema_version": "1.0.0",
        "pipeline_version": "0.1.0",
        "config_hash": "a1b2c3d4e5f6",
    },
}


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_diagnostic_output(output: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate output against schema.

    Performs structural validation without external dependencies.
    For full JSON Schema validation, use jsonschema library.

    :param output (dict): Diagnostic output to validate

    :return tuple: (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Check required fields
    required_fields = ["overall_score", "severity", "verdict"]
    for field in required_fields:
        if field not in output:
            errors.append(f"Missing required field: {field}")

    # Validate overall_score
    if "overall_score" in output:
        score = output["overall_score"]
        if not isinstance(score, (int, float)):
            errors.append(f"overall_score must be a number, got {type(score).__name__}")
        elif score < 0 or score > 100:
            errors.append(f"overall_score must be in [0, 100], got {score}")

    # Validate severity
    if "severity" in output:
        severity = output["severity"]
        valid_severities = {"NOMINAL", "MODERATE", "CRITICAL"}
        if severity not in valid_severities:
            errors.append(f"severity must be one of {valid_severities}, got {severity}")

    # Validate verdict
    if "verdict" in output:
        verdict = output["verdict"]
        if not isinstance(verdict, str):
            errors.append(f"verdict must be a string, got {type(verdict).__name__}")
        elif len(verdict) > 200:
            errors.append(f"verdict exceeds 200 characters: {len(verdict)}")

    # Validate grade if present
    if "grade" in output:
        grade = output["grade"]
        valid_grades = {"A", "B", "C", "D", "F"}
        if grade not in valid_grades:
            errors.append(f"grade must be one of {valid_grades}, got {grade}")

    # Validate component_scores if present
    if "component_scores" in output:
        comp_scores = output["component_scores"]
        if not isinstance(comp_scores, dict):
            errors.append("component_scores must be a dict")
        else:
            for comp_name, comp_data in comp_scores.items():
                if not isinstance(comp_data, dict):
                    errors.append(f"component_scores.{comp_name} must be a dict")
                    continue

                # Check required component fields
                if "score" in comp_data:
                    cs = comp_data["score"]
                    if not isinstance(cs, (int, float)):
                        errors.append(
                            f"component_scores.{comp_name}.score must be a number"
                        )
                    elif cs < 0 or cs > 100:
                        errors.append(
                            f"component_scores.{comp_name}.score must be in [0, 100]"
                        )

                if "grade" in comp_data:
                    cg = comp_data["grade"]
                    if cg not in valid_grades:
                        errors.append(
                            f"component_scores.{comp_name}.grade must be one of {valid_grades}"
                        )

                if "available" in comp_data:
                    if not isinstance(comp_data["available"], bool):
                        errors.append(
                            f"component_scores.{comp_name}.available must be a boolean"
                        )

    # Validate priority_actions if present
    if "priority_actions" in output:
        pa = output["priority_actions"]
        if not isinstance(pa, list):
            errors.append("priority_actions must be a list")
        elif len(pa) > 10:
            errors.append(f"priority_actions exceeds 10 items: {len(pa)}")
        else:
            for i, action in enumerate(pa):
                if not isinstance(action, dict):
                    errors.append(f"priority_actions[{i}] must be a dict")
                    continue
                if "component" not in action:
                    errors.append(f"priority_actions[{i}] missing 'component'")
                if "action" not in action:
                    errors.append(f"priority_actions[{i}] missing 'action'")

    # Validate key_findings if present
    if "key_findings" in output:
        kf = output["key_findings"]
        if not isinstance(kf, list):
            errors.append("key_findings must be a list")
        else:
            for i, finding in enumerate(kf):
                if not isinstance(finding, dict):
                    errors.append(f"key_findings[{i}] must be a dict")
                    continue
                if "metric" not in finding:
                    errors.append(f"key_findings[{i}] missing 'metric'")
                if "interpretation" not in finding:
                    errors.append(f"key_findings[{i}] missing 'interpretation'")

    return (len(errors) == 0, errors)


def get_schema() -> dict[str, Any]:
    """
    Return the JSON schema.

    :return schema (dict): Full JSON Schema definition
    """
    return DIAGNOSTIC_SCHEMA.copy()


def get_example_output() -> dict[str, Any]:
    """
    Return an example valid diagnostic output.

    :return example (dict): Example output conforming to schema
    """
    return EXAMPLE_OUTPUT.copy()


def get_schema_version() -> str:
    """
    Return the current schema version.

    :return version (str): Semantic version string
    """
    return "1.0.0"


def create_minimal_output(
    overall_score: float,
    severity: str,
    verdict: str,
) -> dict[str, Any]:
    """
    Create a minimal valid diagnostic output.

    :param overall_score (float): Score in [0, 100]
    :param severity (str): NOMINAL, MODERATE, or CRITICAL
    :param verdict (str): Summary string (max 200 chars)

    :return output (dict): Minimal valid diagnostic output
    """
    # Map score to grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 75:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    elif overall_score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Map score to status
    if overall_score >= 90:
        status = "EXCELLENT"
    elif overall_score >= 75:
        status = "GOOD"
    elif overall_score >= 60:
        status = "NEEDS ATTENTION"
    elif overall_score >= 40:
        status = "MARGINAL"
    else:
        status = "CRITICAL"

    return {
        "overall_score": float(overall_score),
        "severity": severity,
        "verdict": verdict[:200],
        "grade": grade,
        "status": status,
    }


def severity_from_score(score: float) -> str:
    """
    Derive severity from overall score.

    :param score (float): Overall diagnostic score in [0, 100]

    :return severity (str): NOMINAL, MODERATE, or CRITICAL
    """
    if score >= 75:
        return "NOMINAL"
    if score >= 50:
        return "MODERATE"
    return "CRITICAL"
