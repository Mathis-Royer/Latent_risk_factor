"""
Comprehensive diagnostic modules for the VAE Latent Risk Factor pipeline.

This package provides granular diagnostic metrics from literature review:
- VAE diagnostics: Posterior quality, KL per-dimension, sampling significance
- Factor model diagnostics: Eigenvalue concentration, shrinkage distance, regression quality
- Portfolio diagnostics: Budget conservation, gradient balance, frontier analysis
- E2E diagnostics: Inference determinism, window alignment, universe attrition

Reference: Scale-VAE (LREC 2024), PCF-VAE (Nature 2025), Ledoit-Wolf (JFEc 2020),
SCA Portfolio Book (2024).
"""

from src.diagnostics.vae_diagnostics import (
    analyze_log_var_distribution,
    compute_kl_per_dimension,
    compute_sampling_significance,
)
from src.diagnostics.factor_diagnostics import (
    compute_eigenvalue_concentration,
    compute_shrinkage_distance,
    track_regression_quality,
)
from src.diagnostics.portfolio_diagnostics import (
    verify_budget_conservation,
    compute_gradient_balance,
    analyze_step_size_trajectory,
    check_objective_monotonicity,
    detect_frontier_anomalies,
)
from src.diagnostics.e2e_diagnostics import (
    verify_inference_determinism,
    validate_window_alignment,
    track_universe_attrition,
)

__all__ = [
    # VAE diagnostics
    "analyze_log_var_distribution",
    "compute_kl_per_dimension",
    "compute_sampling_significance",
    # Factor model diagnostics
    "compute_eigenvalue_concentration",
    "compute_shrinkage_distance",
    "track_regression_quality",
    # Portfolio diagnostics
    "verify_budget_conservation",
    "compute_gradient_balance",
    "analyze_step_size_trajectory",
    "check_objective_monotonicity",
    "detect_frontier_anomalies",
    # E2E diagnostics
    "verify_inference_determinism",
    "validate_window_alignment",
    "track_universe_attrition",
]
