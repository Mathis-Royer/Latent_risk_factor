"""
Machine-readable action specifications for diagnostic recommendations.

Provides executable specifications that map diagnostic action strings to
concrete config changes, including validation ranges and rationale.

Usage:
    from src.integration.action_specs import get_executable_actions, get_action_spec

    # Get specifications for priority actions from diagnostic output
    exec_actions = get_executable_actions(priority_actions)
    for action in exec_actions:
        print(f"Change {action['config_key']} to {action['suggested_value']}")
"""

from typing import Any


# ---------------------------------------------------------------------------
# Action Specifications Registry
# ---------------------------------------------------------------------------

ACTION_SPECS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Solver / SCA Optimization Actions
    # -------------------------------------------------------------------------
    "increase_sca_max_iter": {
        "config_key": "portfolio.sca_max_iter",
        "current_value_path": "config.portfolio.sca_max_iter",
        "suggested_multiplier": 2.0,
        "max_value": 500,
        "min_value": 50,
        "rationale": "More iterations for difficult optimization landscapes",
        "category": "solver",
        "triggers": ["Gradient norm > 1e-3", "Low convergence ratio"],
    },
    "reduce_sca_tol": {
        "config_key": "portfolio.sca_tol",
        "current_value_path": "config.portfolio.sca_tol",
        "suggested_multiplier": 0.1,
        "min_value": 1e-12,
        "max_value": 1e-6,
        "rationale": "Tighter tolerance for higher precision at iteration cost",
        "category": "solver",
        "triggers": ["Gradient norm between 1e-8 and 1e-3"],
    },
    "increase_n_starts": {
        "config_key": "portfolio.n_starts",
        "current_value_path": "config.portfolio.n_starts",
        "suggested_multiplier": 2.0,
        "min_value": 3,
        "max_value": 20,
        "rationale": "More random restarts to escape local optima",
        "category": "solver",
        "triggers": ["Low convergence ratio", "High objective variance across starts"],
    },
    "reduce_armijo_rho": {
        "config_key": "portfolio.armijo_rho",
        "current_value_path": "config.portfolio.armijo_rho",
        "suggested_multiplier": 0.5,
        "min_value": 0.1,
        "max_value": 0.9,
        "rationale": "Smaller step size for stability in difficult landscapes",
        "category": "solver",
        "triggers": ["Armijo failures", "Oscillating convergence"],
    },
    "increase_armijo_max_iter": {
        "config_key": "portfolio.armijo_max_iter",
        "current_value_path": "config.portfolio.armijo_max_iter",
        "suggested_multiplier": 1.5,
        "min_value": 10,
        "max_value": 50,
        "rationale": "More backtracking steps to find valid step size",
        "category": "solver",
        "triggers": ["Armijo line search failures"],
    },

    # -------------------------------------------------------------------------
    # Constraint Relaxation Actions
    # -------------------------------------------------------------------------
    "increase_w_max": {
        "config_key": "portfolio.w_max",
        "current_value_path": "config.portfolio.w_max",
        "suggested_values": [0.07, 0.10, 0.15],
        "min_value": 0.02,
        "max_value": 0.25,
        "rationale": "Reduce constraint pressure from too-tight position limits",
        "category": "constraints",
        "triggers": [">50% positions at w_max", "Binding fraction > 0.5"],
    },
    "reduce_w_min": {
        "config_key": "portfolio.w_min",
        "current_value_path": "config.portfolio.w_min",
        "suggested_multiplier": 0.5,
        "min_value": 1e-5,
        "max_value": 0.005,
        "rationale": "Allow smaller positions for better diversification",
        "category": "constraints",
        "triggers": ["Many positions at minimum", "Cardinality too restrictive"],
    },
    "increase_tau_max": {
        "config_key": "portfolio.tau_max",
        "current_value_path": "config.portfolio.tau_max",
        "suggested_values": [0.40, 0.50],
        "min_value": 0.10,
        "max_value": 0.80,
        "rationale": "Allow more turnover when market conditions change",
        "category": "constraints",
        "triggers": ["Turnover constraint binding", "tau_binding = True"],
    },
    "reduce_kappa_1": {
        "config_key": "portfolio.kappa_1",
        "current_value_path": "config.portfolio.kappa_1",
        "suggested_multiplier": 0.5,
        "min_value": 0.0,
        "max_value": 1.0,
        "rationale": "Lower linear turnover penalty for faster adaptation",
        "category": "constraints",
        "triggers": ["High turnover penalty impact"],
    },
    "reduce_kappa_2": {
        "config_key": "portfolio.kappa_2",
        "current_value_path": "config.portfolio.kappa_2",
        "suggested_multiplier": 0.5,
        "min_value": 0.0,
        "max_value": 15.0,
        "rationale": "Lower quadratic turnover penalty",
        "category": "constraints",
        "triggers": ["Excessive turnover smoothing"],
    },
    "increase_w_bar": {
        "config_key": "portfolio.w_bar",
        "current_value_path": "config.portfolio.w_bar",
        "suggested_values": [0.04, 0.05],
        "min_value": 0.01,
        "max_value": 0.10,
        "rationale": "Relax soft concentration threshold",
        "category": "constraints",
        "triggers": ["High concentration penalty", "Many positions > w_bar"],
    },

    # -------------------------------------------------------------------------
    # VAE Architecture Actions
    # -------------------------------------------------------------------------
    "increase_dropout": {
        "config_key": "vae.dropout",
        "current_value_path": "config.vae.dropout",
        "suggested_values": [0.2, 0.3, 0.4],
        "min_value": 0.0,
        "max_value": 0.5,
        "rationale": "Reduce overfitting in VAE encoder/decoder",
        "category": "vae",
        "triggers": ["Overfit ratio > 1.3", "Train loss << Val loss"],
    },
    "reduce_dropout": {
        "config_key": "vae.dropout",
        "current_value_path": "config.vae.dropout",
        "suggested_values": [0.1, 0.05],
        "min_value": 0.0,
        "max_value": 0.5,
        "rationale": "Increase model capacity for underfitting scenarios",
        "category": "vae",
        "triggers": ["Overfit ratio < 0.85", "High reconstruction loss"],
    },
    "reduce_K": {
        "config_key": "vae.K",
        "current_value_path": "config.vae.K",
        "suggested_multiplier": 0.75,
        "min_value": 20,
        "max_value": 200,
        "rationale": "Reduce latent capacity if many dimensions collapsed",
        "category": "vae",
        "triggers": ["AU < 0.3 * K", "High posterior collapse"],
    },
    "increase_K": {
        "config_key": "vae.K",
        "current_value_path": "config.vae.K",
        "suggested_multiplier": 1.5,
        "min_value": 20,
        "max_value": 200,
        "rationale": "Increase latent capacity if all dimensions active",
        "category": "vae",
        "triggers": ["AU = K", "Reconstruction loss still high"],
    },

    # -------------------------------------------------------------------------
    # Training Actions
    # -------------------------------------------------------------------------
    "increase_epochs": {
        "config_key": "training.max_epochs",
        "current_value_path": "config.training.max_epochs",
        "suggested_multiplier": 1.5,
        "min_value": 50,
        "max_value": 500,
        "rationale": "More training time for convergence",
        "category": "training",
        "triggers": ["Early stopping not triggered", "Loss still decreasing"],
    },
    "reduce_learning_rate": {
        "config_key": "training.learning_rate",
        "current_value_path": "config.training.learning_rate",
        "suggested_multiplier": 0.5,
        "min_value": 1e-5,
        "max_value": 1e-2,
        "rationale": "Slower learning for stability",
        "category": "training",
        "triggers": ["Loss spikes", "Unstable training"],
    },
    "increase_learning_rate": {
        "config_key": "training.learning_rate",
        "current_value_path": "config.training.learning_rate",
        "suggested_multiplier": 2.0,
        "min_value": 1e-5,
        "max_value": 1e-2,
        "rationale": "Faster learning if stuck in flat region",
        "category": "training",
        "triggers": ["Very slow convergence", "Loss plateau"],
    },
    "increase_patience": {
        "config_key": "training.patience",
        "current_value_path": "config.training.patience",
        "suggested_multiplier": 1.5,
        "min_value": 5,
        "max_value": 50,
        "rationale": "Wait longer for improvement before stopping",
        "category": "training",
        "triggers": ["Early stopping too aggressive"],
    },
    "increase_batch_size": {
        "config_key": "training.batch_size",
        "current_value_path": "config.training.batch_size",
        "suggested_multiplier": 2.0,
        "min_value": 32,
        "max_value": 2048,
        "rationale": "Larger batches for more stable gradients",
        "category": "training",
        "triggers": ["High gradient variance", "Noisy loss curves"],
    },

    # -------------------------------------------------------------------------
    # Loss Function Actions
    # -------------------------------------------------------------------------
    "reduce_beta": {
        "config_key": "loss.beta_fixed",
        "current_value_path": "config.loss.beta_fixed",
        "suggested_multiplier": 0.5,
        "min_value": 0.1,
        "max_value": 5.0,
        "rationale": "Lower KL weight to reduce posterior collapse",
        "category": "loss",
        "triggers": ["High collapse severity", ">30% dimensions collapsed"],
    },
    "increase_beta": {
        "config_key": "loss.beta_fixed",
        "current_value_path": "config.loss.beta_fixed",
        "suggested_multiplier": 2.0,
        "min_value": 0.1,
        "max_value": 5.0,
        "rationale": "Higher KL weight to prevent posterior explosion",
        "category": "loss",
        "triggers": ["Posterior explosion", "Unbounded variance"],
    },
    "reduce_gamma": {
        "config_key": "loss.gamma",
        "current_value_path": "config.loss.gamma",
        "suggested_multiplier": 0.5,
        "min_value": 1.0,
        "max_value": 10.0,
        "rationale": "Lower crisis overweighting if causing instability",
        "category": "loss",
        "triggers": ["Unstable crisis period training"],
    },
    "reduce_lambda_co_max": {
        "config_key": "loss.lambda_co_max",
        "current_value_path": "config.loss.lambda_co_max",
        "suggested_multiplier": 0.5,
        "min_value": 0.0,
        "max_value": 0.5,
        "rationale": "Reduce co-movement loss weight if dominating",
        "category": "loss",
        "triggers": ["Co-movement loss >> reconstruction loss"],
    },

    # -------------------------------------------------------------------------
    # Risk Model Actions
    # -------------------------------------------------------------------------
    "increase_ridge_scale": {
        "config_key": "risk_model.ridge_scale",
        "current_value_path": "config.risk_model.ridge_scale",
        "suggested_multiplier": 10.0,
        "min_value": 1e-8,
        "max_value": 1e-3,
        "rationale": "Increase regularization for ill-conditioned covariance",
        "category": "risk_model",
        "triggers": ["Condition number > 1e6", "Near-singular covariance"],
    },
    "increase_d_eps_floor": {
        "config_key": "risk_model.d_eps_floor",
        "current_value_path": "config.risk_model.d_eps_floor",
        "suggested_multiplier": 10.0,
        "min_value": 1e-8,
        "max_value": 1e-4,
        "rationale": "Higher floor for idiosyncratic variance stability",
        "category": "risk_model",
        "triggers": ["Very small idiosyncratic variances", "Numerical issues"],
    },
    "switch_to_spiked_shrinkage": {
        "config_key": "risk_model.sigma_z_shrinkage",
        "current_value_path": "config.risk_model.sigma_z_shrinkage",
        "suggested_values": ["spiked"],
        "rationale": "Use optimal spiked covariance estimator",
        "category": "risk_model",
        "triggers": ["Few dominant eigenvalues", "Factor model structure"],
    },
    "increase_sigma_z_ewma": {
        "config_key": "risk_model.sigma_z_ewma_half_life",
        "current_value_path": "config.risk_model.sigma_z_ewma_half_life",
        "suggested_values": [63, 126, 252],
        "min_value": 0,
        "max_value": 504,
        "rationale": "More weight on recent observations for time-varying covariance",
        "category": "risk_model",
        "triggers": ["Regime changes", "Time-varying factor covariance"],
    },
    "reduce_b_a_clip_threshold": {
        "config_key": "risk_model.b_a_clip_threshold",
        "current_value_path": "config.risk_model.b_a_clip_threshold",
        "suggested_values": [3.0, 2.5],
        "min_value": 2.0,
        "max_value": 5.0,
        "rationale": "Tighter clipping for extreme exposure loadings",
        "category": "risk_model",
        "triggers": ["Extreme z_hat values", "Factor loading outliers"],
    },

    # -------------------------------------------------------------------------
    # Inference Actions
    # -------------------------------------------------------------------------
    "increase_au_threshold": {
        "config_key": "inference.au_threshold",
        "current_value_path": "config.inference.au_threshold",
        "suggested_values": [0.02, 0.05],
        "min_value": 0.001,
        "max_value": 0.1,
        "rationale": "Stricter active unit threshold for cleaner factors",
        "category": "inference",
        "triggers": ["Many near-inactive dimensions", "AU close to K"],
    },
    "reduce_au_threshold": {
        "config_key": "inference.au_threshold",
        "current_value_path": "config.inference.au_threshold",
        "suggested_values": [0.005, 0.001],
        "min_value": 0.001,
        "max_value": 0.1,
        "rationale": "Lower threshold to capture more factors",
        "category": "inference",
        "triggers": ["AU too low", "Missing useful factors"],
    },

    # -------------------------------------------------------------------------
    # Data Pipeline Actions
    # -------------------------------------------------------------------------
    "increase_window_length": {
        "config_key": "data.window_length",
        "current_value_path": "config.data.window_length",
        "suggested_values": [756, 1008],
        "min_value": 252,
        "max_value": 1260,
        "rationale": "Longer history for more stable factor extraction",
        "category": "data",
        "triggers": ["Unstable factors", "High latent stability variance"],
        "note": "Must also update vae.window_length",
    },
    "increase_min_valid_fraction": {
        "config_key": "data.min_valid_fraction",
        "current_value_path": "config.data.min_valid_fraction",
        "suggested_values": [0.85, 0.90, 0.95],
        "min_value": 0.50,
        "max_value": 0.99,
        "rationale": "Stricter data quality filter",
        "category": "data",
        "triggers": ["Many NaN values", "Data quality issues"],
    },
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_action_spec(action_name: str) -> dict[str, Any] | None:
    """
    Get specification for a named action.

    :param action_name (str): Action identifier (e.g., "increase_sca_max_iter")

    :return spec (dict | None): Action specification or None if not found
    """
    return ACTION_SPECS.get(action_name)


def get_all_action_names() -> list[str]:
    """
    Get list of all available action names.

    :return names (list[str]): List of action identifiers
    """
    return list(ACTION_SPECS.keys())


def get_actions_by_category(category: str) -> dict[str, dict[str, Any]]:
    """
    Get all actions for a specific category.

    :param category (str): Category name (solver, constraints, vae, training, etc.)

    :return actions (dict): Subset of ACTION_SPECS matching the category
    """
    return {
        name: spec
        for name, spec in ACTION_SPECS.items()
        if spec.get("category") == category
    }


def _parse_action_string(action_string: str) -> str | None:
    """
    Parse an action string to extract the action name.

    Handles various formats:
    - "Increase sca_max_iter or reduce sca_tol" -> "increase_sca_max_iter"
    - "Consider increasing w_max (e.g., 0.05 -> 0.07)" -> "increase_w_max"
    - "Overfitting detected; increase dropout or reduce K" -> "increase_dropout"

    :param action_string (str): Human-readable action recommendation

    :return action_name (str | None): Matching action name or None
    """
    action_lower = action_string.lower()

    # Direct keyword matching
    keyword_to_action: dict[str, str] = {
        "increase sca_max_iter": "increase_sca_max_iter",
        "reduce sca_tol": "reduce_sca_tol",
        "more multi-starts": "increase_n_starts",
        "increase w_max": "increase_w_max",
        "increasing w_max": "increase_w_max",
        "increase tau_max": "increase_tau_max",
        "increasing tau_max": "increase_tau_max",
        "increase dropout": "increase_dropout",
        "reduce dropout": "reduce_dropout",
        "reduce k": "reduce_K",
        "increase k": "increase_K",
        "increase ridge": "increase_ridge_scale",
        "increase regularization": "increase_ridge_scale",
        "reduce beta": "reduce_beta",
        "reduce kl": "reduce_beta",
        "increase shrinkage": "increase_ridge_scale",
        "increase history": "increase_window_length",
        "extend training window": "increase_window_length",
        "reduce factors": "reduce_K",
        "variance targeting": "increase_d_eps_floor",
        "increase epochs": "increase_epochs",
        "reduce learning rate": "reduce_learning_rate",
        "lower learning rate": "reduce_learning_rate",
        "increase patience": "increase_patience",
        "armijo": "reduce_armijo_rho",
        "step size": "reduce_armijo_rho",
    }

    for keyword, action_name in keyword_to_action.items():
        if keyword in action_lower:
            return action_name

    return None


def get_executable_actions(priority_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert action strings to executable specifications.

    Takes priority_actions from compute_overall_score() and returns
    machine-readable specifications with config keys and suggested values.

    :param priority_actions (list[dict]): List with 'component', 'score', 'action' keys

    :return exec_actions (list[dict]): List of executable action specs
    """
    exec_actions: list[dict[str, Any]] = []

    for pa in priority_actions:
        action_string = pa.get("action", "")
        if not action_string:
            continue

        action_name = _parse_action_string(action_string)
        if action_name is None:
            # Include as unrecognized but keep the original recommendation
            exec_actions.append({
                "recognized": False,
                "original_action": action_string,
                "component": pa.get("component"),
                "score": pa.get("score"),
            })
            continue

        spec = ACTION_SPECS.get(action_name)
        if spec is None:
            continue

        exec_action: dict[str, Any] = {
            "recognized": True,
            "action_name": action_name,
            "config_key": spec["config_key"],
            "category": spec.get("category"),
            "rationale": spec.get("rationale"),
            "component": pa.get("component"),
            "score": pa.get("score"),
            "original_action": action_string,
        }

        # Add suggested value(s)
        if "suggested_values" in spec:
            exec_action["suggested_values"] = spec["suggested_values"]
            exec_action["suggested_value"] = spec["suggested_values"][0]
        elif "suggested_multiplier" in spec:
            exec_action["suggested_multiplier"] = spec["suggested_multiplier"]
            exec_action["min_value"] = spec.get("min_value")
            exec_action["max_value"] = spec.get("max_value")

        if "note" in spec:
            exec_action["note"] = spec["note"]

        exec_actions.append(exec_action)

    return exec_actions


def suggest_config_change(
    action_name: str,
    current_value: float | int | str | None = None,
) -> dict[str, Any]:
    """
    Suggest a concrete config change for a named action.

    :param action_name (str): Action identifier
    :param current_value (float | int | str | None): Current config value

    :return suggestion (dict): Suggested change with new value and rationale
    """
    spec = ACTION_SPECS.get(action_name)
    if spec is None:
        return {
            "valid": False,
            "error": f"Unknown action: {action_name}",
        }

    result: dict[str, Any] = {
        "valid": True,
        "action_name": action_name,
        "config_key": spec["config_key"],
        "current_value": current_value,
        "rationale": spec.get("rationale"),
        "category": spec.get("category"),
    }

    # Compute suggested value
    if "suggested_values" in spec:
        # Fixed suggested values
        result["suggested_values"] = spec["suggested_values"]
        result["suggested_value"] = spec["suggested_values"][0]
    elif "suggested_multiplier" in spec and current_value is not None:
        # Multiplier-based suggestion
        try:
            new_value = float(current_value) * spec["suggested_multiplier"]

            # Apply bounds
            if "min_value" in spec:
                new_value = max(new_value, spec["min_value"])
            if "max_value" in spec:
                new_value = min(new_value, spec["max_value"])

            # Keep as int if original was int
            if isinstance(current_value, int):
                new_value = int(round(new_value))

            result["suggested_value"] = new_value
            result["multiplier_used"] = spec["suggested_multiplier"]
        except (TypeError, ValueError):
            result["error"] = "Cannot apply multiplier to current value"

    return result


def get_action_triggers(action_name: str) -> list[str]:
    """
    Get diagnostic triggers for a specific action.

    :param action_name (str): Action identifier

    :return triggers (list[str]): List of diagnostic conditions that trigger this action
    """
    spec = ACTION_SPECS.get(action_name)
    if spec is None:
        return []
    return spec.get("triggers", [])
