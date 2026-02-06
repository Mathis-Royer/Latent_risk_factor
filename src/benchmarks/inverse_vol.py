"""
MOD-011: Inverse volatility benchmark.

w_i ∝ 1/σ_i where σ_i = trailing 252d annualized vol.
Projection onto shared constraints after normalization.

Reference: ISD Section MOD-011.
"""

import numpy as np
import pandas as pd

from src.benchmarks.base import BenchmarkModel


class InverseVolatility(BenchmarkModel):
    """Inverse-volatility weighted benchmark."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        :param returns (pd.DataFrame): Historical returns
        :param universe (list[str]): Active stock identifiers
        :param trailing_vol (pd.DataFrame): Trailing vols (in kwargs)
        :param current_date (str): Current date (in kwargs)
        """
        self.n = len(universe)

        trailing_vol = kwargs.get("trailing_vol")
        current_date = kwargs.get("current_date")

        if trailing_vol is not None and current_date is not None:
            vol_df: pd.DataFrame = trailing_vol  # type: ignore[assignment]
            available = [s for s in universe if s in vol_df.columns]
            self.sigma = vol_df.loc[current_date, available].values.astype(np.float64)  # type: ignore[union-attr]
        else:
            # Fallback: compute from returns
            R = returns[universe].dropna(how="all")
            self.sigma = np.asarray(R.std(axis=0)) * np.sqrt(252)

        # Ensure positive
        self.sigma = np.maximum(self.sigma, 1e-10)

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        :return w (np.ndarray): Inverse-vol weights (n,), projected
        """
        w = 1.0 / self.sigma
        w = w / w.sum()
        return self._project_to_constraints(w, w_old, is_first)
