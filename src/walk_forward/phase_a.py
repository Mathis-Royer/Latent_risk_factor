"""
Phase A: Hyperparameter selection via nested validation.

For each HP config, train on [start, train_end - 2yr], validate on
[train_end - 2yr, train_end], evaluate on OOS. Select best config
via composite score.

Reference: ISD Section MOD-009 — Sub-task 2.
"""

from typing import Any

import numpy as np


def composite_score(
    H_oos: float,
    AU: int,
    mdd_oos: float,
    n_obs: int,
    mdd_threshold: float = 0.20,
    lambda_pen: float = 5.0,
    lambda_est: float = 2.0,
) -> float:
    """
    Composite scoring for HP selection.

    Score = Ĥ_OOS - λ_pen · max(0, MDD_OOS - MDD_threshold) - λ_est · max(0, 1 - R_Σ)

    Where:
      Ĥ = H(w) / ln(AU) (normalized entropy, ∈ [0, 1])
      R_Σ = N_obs / (AU(AU+1)/2)

    :param H_oos (float): OOS factor entropy
    :param AU (int): Number of active units
    :param mdd_oos (float): OOS maximum drawdown (fraction)
    :param n_obs (int): Number of observations
    :param mdd_threshold (float): MDD threshold
    :param lambda_pen (float): MDD penalty weight
    :param lambda_est (float): Estimation quality penalty

    :return score (float): Composite score (higher is better)
    """
    # Normalized entropy
    H_norm = H_oos / max(np.log(max(AU, 1)), 1e-10)
    H_norm = min(H_norm, 1.0)

    # Estimation quality ratio
    R_sigma = n_obs / max(AU * (AU + 1) / 2, 1)

    # Composite score
    score = (
        H_norm
        - lambda_pen * max(0.0, mdd_oos - mdd_threshold)
        - lambda_est * max(0.0, 1.0 - R_sigma)
    )

    return float(score)


def eliminate_configs(
    config_results: list[dict[str, Any]],
    K: int,
    AU_PCA: int = 0,
    EP_PCA: float = 0.0,
    n_stocks: int | None = None,
) -> list[dict[str, Any]]:
    """
    Eliminate poor configs before scoring.

    Criteria:
      - AU < au_min → eliminated  (au_min = max(0.15K, AU_PCA), capped at n_stocks/2)
      - EP < max(0.40, EP_PCA + 0.10) → eliminated
      - OOS/train MSE > 3.0 → eliminated

    :param config_results (list[dict]): Results per config
    :param K (int): Latent capacity
    :param AU_PCA (int): PCA benchmark AU
    :param EP_PCA (float): PCA explanatory power
    :param n_stocks (int | None): Universe size, used to cap AU threshold

    :return surviving (list[dict]): Non-eliminated configs
    """
    au_min = max(int(0.15 * K), AU_PCA)
    # Cap AU threshold for small universes: can't expect more AU than n/2
    if n_stocks is not None and n_stocks > 0:
        au_min = min(au_min, max(n_stocks // 2, 2))
    ep_min = max(0.40, EP_PCA + 0.10)

    surviving: list[dict[str, Any]] = []
    for result in config_results:
        au = result.get("AU", 0)
        ep = result.get("explanatory_power", 0.0)
        mse_ratio = result.get("oos_train_mse_ratio", 1.0)

        if au < au_min:
            continue
        if ep < ep_min:
            continue
        if mse_ratio > 3.0:
            continue

        surviving.append(result)

    return surviving


def select_best_config(
    config_results: list[dict[str, Any]],
    K: int,
    AU_PCA: int = 0,
    EP_PCA: float = 0.0,
    mdd_threshold: float = 0.20,
    lambda_pen: float = 5.0,
    lambda_est: float = 2.0,
    n_stocks: int | None = None,
) -> dict[str, Any] | None:
    """
    Select best HP config via elimination + composite scoring.

    :param config_results (list[dict]): Results per config
    :param K (int): Latent capacity
    :param AU_PCA (int): PCA benchmark AU
    :param EP_PCA (float): PCA explanatory power
    :param mdd_threshold (float): MDD threshold
    :param lambda_pen (float): MDD penalty
    :param lambda_est (float): Estimation penalty
    :param n_stocks (int | None): Universe size for threshold scaling

    :return best (dict | None): Best config, or None if all eliminated
    """
    surviving = eliminate_configs(config_results, K, AU_PCA, EP_PCA, n_stocks=n_stocks)

    if not surviving:
        return None

    assert len(surviving) > 0, "No surviving configs after elimination"

    best_score = -float("inf")
    best_config: dict[str, Any] | None = None

    for result in surviving:
        au_val = result.get("AU", 1)
        assert np.isfinite(au_val) and au_val > 0, (
            f"Invalid AU={au_val} in surviving config"
        )
        score = composite_score(
            H_oos=result.get("H_oos", 0.0),
            AU=result.get("AU", 1),
            mdd_oos=result.get("mdd_oos", 0.0),
            n_obs=result.get("n_obs", 1),
            mdd_threshold=mdd_threshold,
            lambda_pen=lambda_pen,
            lambda_est=lambda_est,
        )
        result["composite_score"] = score

        if score > best_score:
            best_score = score
            best_config = result

    return best_config
