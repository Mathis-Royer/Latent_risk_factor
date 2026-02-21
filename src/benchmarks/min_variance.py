"""
MOD-012: Minimum variance benchmark (Ledoit-Wolf shrinkage).

min w^T Σ_LW w   s.t. shared constraints.

Σ_LW estimated via Ledoit-Wolf (2004) shrinkage toward scaled identity,
computed on the FULL training window (expanding, anti-cyclical).

Reference: ISD Section MOD-012.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from src.benchmarks.base import BenchmarkModel


class MinimumVariance(BenchmarkModel):
    """Minimum variance benchmark with Ledoit-Wolf covariance."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        Estimate Σ_LW via Ledoit-Wolf shrinkage.

        :param returns (pd.DataFrame): Historical returns
        :param universe (list[str]): Active stock identifiers
        """
        self.n = len(universe)
        self.universe = universe

        available = [s for s in universe if s in returns.columns]
        R_df = returns[available]

        # Filter stocks with >50% NaN, then fill residual NaN with 0.
        # dropna() on 600+ stocks eliminates nearly all rows due to scattered NaN.
        valid_frac: pd.Series = R_df.notna().mean()  # type: ignore[assignment]
        keep: list[str] = valid_frac[valid_frac > 0.5].index.tolist()  # type: ignore[assignment]
        if len(keep) < 2:
            keep = available
        R: np.ndarray = R_df[keep].fillna(0.0).values  # type: ignore[assignment]

        lw = LedoitWolf()
        lw.fit(R)
        self.Sigma_LW: np.ndarray = lw.covariance_  # type: ignore[assignment]

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        Solve min-variance QP with shared constraints via CVXPY.

        :return w (np.ndarray): Min-variance weights (n,)
        """
        n = self.Sigma_LW.shape[0]
        w_max = self.constraint_params["w_max"]
        tau_max = self.constraint_params["tau_max"]

        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, self.Sigma_LW))

        # Uses w_max (hard cap) — w_bar is only for soft penalty in SCA
        constraints = [
            w >= 0,
            w <= w_max,
            cp.sum(w) == 1,
        ]

        if w_old is not None and not is_first:
            constraints.append(0.5 * cp.sum(cp.abs(w - w_old)) <= tau_max)

        prob = cp.Problem(objective, constraints)

        # Solver chain: MOSEK > ECOS > SCS (DVT Section 4.7)
        for solver, kwargs in [
            (cp.MOSEK, {"warm_start": True}),
            (cp.ECOS, {"warm_start": True}),
            (cp.SCS, {"warm_start": True, "max_iters": 5000}),
        ]:
            try:
                prob.solve(solver=solver, **kwargs)
                if w.value is not None:
                    return np.array(w.value).flatten()
            except (cp.SolverError, Exception):
                continue

        # Last resort: equal weight
        return np.ones(n) / n

    def rebalance(
        self,
        returns_trailing: pd.DataFrame,
        trailing_vol: pd.DataFrame | None,
        w_old: np.ndarray,
        universe: list[str],
        current_date: str,
    ) -> np.ndarray:
        """
        Re-estimate Ledoit-Wolf covariance and re-optimize min-variance.

        :return w (np.ndarray): Min-variance weights on new universe
        """
        self.fit(returns_trailing, universe)
        return self.optimize(w_old=w_old, is_first=False)
