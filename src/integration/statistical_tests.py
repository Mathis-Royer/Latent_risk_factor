"""
Statistical tests for walk-forward comparison.

- Wilcoxon signed-rank test (per-fold paired differences)
- Bootstrap effect size with confidence intervals
- Holm-Bonferroni correction (6 benchmarks × 4 metrics = 24 tests)
- Regime decomposition (crisis vs calm folds)

Reference: ISD Section MOD-016 — Sub-task 2.
"""

from typing import Any

import numpy as np
from scipy import stats


def wilcoxon_paired_test(
    vae_scores: np.ndarray,
    benchmark_scores: np.ndarray,
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test on per-fold differences.

    H0: median(Δ_k) = 0.

    :param vae_scores (np.ndarray): VAE metric values per fold (n_folds,)
    :param benchmark_scores (np.ndarray): Benchmark values per fold (n_folds,)

    :return result (dict): statistic, p_value, median_diff
    """
    diffs = vae_scores - benchmark_scores

    # Remove zero differences (Wilcoxon requires non-zero)
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 5:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "median_diff": float(np.median(diffs)),
            "n_pairs": len(non_zero),
        }

    result = stats.wilcoxon(non_zero, alternative="two-sided")

    return {
        "statistic": float(result.statistic),  # type: ignore[union-attr]
        "p_value": float(result.pvalue),  # type: ignore[union-attr]
        "median_diff": float(np.median(diffs)),
        "n_pairs": len(non_zero),
    }


def bootstrap_effect_size(
    vae_scores: np.ndarray,
    benchmark_scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap median effect size with confidence interval (percentile method).

    :param vae_scores (np.ndarray): VAE values per fold
    :param benchmark_scores (np.ndarray): Benchmark values per fold
    :param n_bootstrap (int): Number of bootstrap samples
    :param ci_level (float): Confidence level
    :param seed (int): Random seed

    :return result (dict): median_effect, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    diffs = vae_scores - benchmark_scores
    n = len(diffs)

    boot_medians = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_medians[b] = np.median(sample)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))

    return {
        "median_effect": float(np.median(diffs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": bool(ci_lower > 0 or ci_upper < 0),
    }


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    """
    Holm-Bonferroni step-down correction for multiple testing.

    For 6 benchmarks × 4 metrics = 24 tests.

    :param p_values (list[float]): Raw p-values
    :param alpha (float): Family-wise error rate

    :return results (list[dict]): Corrected results with reject flag
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [{"original_index": 0, "p_value": 0.0, "adjusted_alpha": 0.0, "reject": False}] * m

    reject_all = True
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        reject = reject_all and (pval <= adjusted_alpha)
        if not reject:
            reject_all = False

        results[orig_idx] = {
            "original_index": orig_idx,
            "p_value": pval,
            "adjusted_alpha": adjusted_alpha,
            "rank": rank + 1,
            "reject": reject,
        }

    return results


def regime_decomposition(
    fold_metrics: list[dict[str, float]],
    fold_crisis_fractions: list[float],
    crisis_threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Separate folds into "crisis" (> 20% days VIX > P80) and "calm".

    :param fold_metrics (list[dict]): Per-fold metrics
    :param fold_crisis_fractions (list[float]): Fraction of crisis days per fold
    :param crisis_threshold (float): Threshold for crisis classification

    :return decomposition (dict): "crisis" and "calm" fold metrics
    """
    crisis_folds: list[dict[str, float]] = []
    calm_folds: list[dict[str, float]] = []

    for metrics, cf in zip(fold_metrics, fold_crisis_fractions):
        if cf > crisis_threshold:
            crisis_folds.append(metrics)
        else:
            calm_folds.append(metrics)

    return {
        "crisis": crisis_folds,
        "calm": calm_folds,
        "n_crisis": len(crisis_folds),
        "n_calm": len(calm_folds),
    }
