"""
Abstract base class for all benchmark models.

Each benchmark receives identical inputs (universe, returns, constraints)
and produces identical output format (weights). INV-012: constraints are
shared between VAE and all benchmarks.

Reference: ISD Section MOD-010–015 — Common Abstract Class.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.portfolio.constraints import (
    concentration_penalty,
    turnover_penalty,
    project_to_constraints,
)


class BenchmarkModel(ABC):
    """
    Common interface for all benchmarks.

    Attributes:
        constraint_params: dict — shared constraint parameters (INV-012)
        n: int — universe size (set during fit)
    """

    def __init__(self, constraint_params: dict[str, float] | None = None) -> None:
        """
        :param constraint_params (dict | None): Shared constraint parameters.
            Keys: w_max, w_min, phi, kappa_1, kappa_2, delta_bar, tau_max, lambda_risk
        """
        self.constraint_params = constraint_params or {
            "w_max": 0.05,
            "w_min": 0.001,
            "phi": 25.0,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "lambda_risk": 1.0,
        }
        self.n = 0

    @abstractmethod
    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        Estimate the risk model (if applicable).

        :param returns (pd.DataFrame): Historical returns (dates × stocks)
        :param universe (list[str]): Active stock identifiers
        """

    @abstractmethod
    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        Produce optimal weights under shared constraints.

        :param w_old (np.ndarray | None): Previous weights
        :param is_first (bool): True for first rebalancing (κ₁=κ₂=0)

        :return w (np.ndarray): Optimal weights (n,)
        """

    def _project_to_constraints(
        self,
        w: np.ndarray,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        Project weights onto hard constraints. Iterative clip + renormalize.

        :param w (np.ndarray): Raw weights (n,)
        :param w_old (np.ndarray | None): Previous weights
        :param is_first (bool): First rebalancing flag

        :return w_proj (np.ndarray): Projected weights (n,)
        """
        w_max = self.constraint_params["w_max"]
        w_min = self.constraint_params["w_min"]
        tau_max = self.constraint_params["tau_max"]

        w = project_to_constraints(w, w_max, w_min)

        # Enforce turnover constraint if not first
        if w_old is not None and not is_first:
            one_way = 0.5 * np.sum(np.abs(w - w_old))
            if one_way > tau_max:
                # Scale the change to fit within tau_max
                alpha = tau_max / max(one_way, 1e-10)
                w = w_old + alpha * (w - w_old)
                w = np.clip(w, 0.0, w_max)
                total = np.sum(w)
                if total > 0:
                    w = w / total

        return w

    def evaluate(
        self,
        w: np.ndarray,
        returns_oos: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, float]:
        """
        Compute standard OOS metrics (shared across all models).

        :param w (np.ndarray): Portfolio weights (n,)
        :param returns_oos (pd.DataFrame): OOS returns (dates × stocks)
        :param universe (list[str]): Stock identifiers matching w

        :return metrics (dict): OOS portfolio metrics
        """
        available = [s for s in universe if s in returns_oos.columns]
        R_oos = returns_oos[available].values
        w_active = w[:len(available)]

        # Portfolio returns
        port_returns = R_oos @ w_active
        n_days = len(port_returns)

        # Annualized return
        ann_return = float(np.mean(port_returns) * 252)

        # Annualized volatility
        ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252))

        # Sharpe ratio
        sharpe = ann_return / max(ann_vol, 1e-10)

        # Maximum drawdown
        cum_returns = np.cumsum(port_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - running_max
        max_drawdown = float(-np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        calmar = ann_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0.0

        # Effective number of positions
        eff_n = float(1.0 / np.sum(w_active ** 2)) if np.sum(w_active ** 2) > 0 else 0.0

        return {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "eff_n_positions": eff_n,
            "n_days_oos": n_days,
        }
