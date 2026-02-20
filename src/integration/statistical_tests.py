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


def _compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio from daily returns.

    :param returns (np.ndarray): Daily returns
    :param annualization (float): Annualization factor (252 for daily)

    :return sharpe (float): Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))


def compute_pbo(
    fold_returns: list[np.ndarray],
    strategy_returns: list[np.ndarray],
    max_combinations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute Probability of Backtest Overfitting (PBO) using CSCV.

    Combinatorial Symmetric Cross-Validation splits N folds into
    C(N, N/2) combinations of in-sample/out-of-sample sets. For each
    combination, compares whether the in-sample Sharpe winner is also
    the out-of-sample winner.

    Reference: Bailey, Borwein, Lopez de Prado, Zhu (2017)
    "The Probability of Backtest Overfitting", Journal of Computational Finance.

    :param fold_returns (np.ndarray): Benchmark returns per fold (list of arrays)
    :param strategy_returns (np.ndarray): Strategy returns per fold (list of arrays)
    :param max_combinations (int): Maximum combinations to sample (for large N)
    :param seed (int): Random seed for sampling

    :return result (dict): pbo_probability, n_combinations, logit_statistic, is_overfitted
    """
    n_folds = len(fold_returns)

    # Validate inputs
    if n_folds != len(strategy_returns):
        raise ValueError(
            f"fold_returns and strategy_returns must have same length: "
            f"{n_folds} != {len(strategy_returns)}"
        )

    if n_folds < 4:
        # Not enough folds for meaningful CSCV
        return {
            "pbo_probability": np.nan,
            "n_combinations": 0,
            "logit_statistic": np.nan,
            "is_overfitted": False,
            "warning": "Insufficient folds for CSCV (need >= 4)",
        }

    # Use even number of folds
    effective_n = n_folds if n_folds % 2 == 0 else n_folds - 1
    half_n = effective_n // 2

    # Compute Sharpe ratios for each fold
    benchmark_sharpes = np.array([_compute_sharpe(r) for r in fold_returns[:effective_n]])
    strategy_sharpes = np.array([_compute_sharpe(r) for r in strategy_returns[:effective_n]])

    # Generate combinations
    from itertools import combinations
    from math import comb

    total_combinations = comb(effective_n, half_n)
    all_indices = list(range(effective_n))

    rng = np.random.RandomState(seed)

    # Determine if we need to sample
    if total_combinations <= max_combinations:
        # Use all combinations
        is_combinations = list(combinations(all_indices, half_n))
        sampled = False
    else:
        # Sample combinations randomly
        is_combinations = []
        seen: set[tuple[int, ...]] = set()
        while len(is_combinations) < max_combinations:
            perm = rng.permutation(effective_n)
            is_set = tuple(sorted(perm[:half_n]))
            if is_set not in seen:
                seen.add(is_set)
                is_combinations.append(is_set)
        sampled = True

    n_overfit = 0
    n_valid = 0

    for is_indices in is_combinations:
        is_set = set(is_indices)
        oos_indices = [i for i in all_indices if i not in is_set]

        # Compute IS Sharpe (average across IS folds)
        is_benchmark_sharpe = np.mean(benchmark_sharpes[list(is_indices)])
        is_strategy_sharpe = np.mean(strategy_sharpes[list(is_indices)])

        # Compute OOS Sharpe (average across OOS folds)
        oos_benchmark_sharpe = np.mean(benchmark_sharpes[oos_indices])
        oos_strategy_sharpe = np.mean(strategy_sharpes[oos_indices])

        # IS winner: strategy beats benchmark?
        is_strategy_wins = is_strategy_sharpe > is_benchmark_sharpe

        # OOS winner: strategy beats benchmark?
        oos_strategy_wins = oos_strategy_sharpe > oos_benchmark_sharpe

        # Check consistency
        if is_strategy_wins != oos_strategy_wins:
            n_overfit += 1

        n_valid += 1

    # Compute PBO
    pbo = n_overfit / n_valid if n_valid > 0 else np.nan

    # Compute logit statistic: log(PBO / (1 - PBO))
    # Clamp to avoid log(0) or log(inf)
    pbo_clamped = np.clip(pbo, 1e-10, 1.0 - 1e-10)
    logit = float(np.log(pbo_clamped / (1.0 - pbo_clamped)))

    result: dict[str, Any] = {
        "pbo_probability": float(pbo) if not np.isnan(pbo) else pbo,
        "n_combinations": n_valid,
        "logit_statistic": logit,
        "is_overfitted": bool(pbo > 0.5),
    }

    if sampled:
        result["sampled"] = True
        result["total_possible_combinations"] = total_combinations

    return result
