"""
MOD-010: Equal-weight (1/N) benchmark.

No risk model, no optimization. w_i = 1/n for all stocks.
Hard cap at w_max is non-binding for n ≥ 20.

Reference: ISD Section MOD-010.
"""

import numpy as np
import pandas as pd

from src.benchmarks.base import BenchmarkModel


class EqualWeight(BenchmarkModel):
    """Equal-weight benchmark: w_i = 1/n."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        :param returns (pd.DataFrame): Not used
        :param universe (list[str]): Active stock identifiers
        """
        self.n = len(universe)

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        :return w (np.ndarray): Equal weights (n,), clipped to w_max
        """
        w = np.ones(self.n) / self.n
        w_max = self.constraint_params["w_max"]
        return np.clip(w, 0.0, w_max)
