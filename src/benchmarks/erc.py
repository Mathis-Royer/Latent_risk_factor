"""
MOD-013: Equal Risk Contribution (ERC) benchmark via Spinu (2013).

Convex log-barrier formulation:
  min (1/2) w^T Σ w - c · Σ_i ln(w_i)

where c scales to enforce ERC. Post-hoc projection onto hard caps.

Reference: ISD Section MOD-013.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from src.benchmarks.base import BenchmarkModel


class EqualRiskContribution(BenchmarkModel):
    """Equal Risk Contribution benchmark (Spinu log-barrier)."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        Estimate Σ_LW via Ledoit-Wolf.

        :param returns (pd.DataFrame): Historical returns
        :param universe (list[str]): Active stock identifiers
        """
        self.n = len(universe)

        available = [s for s in universe if s in returns.columns]
        R = returns[available].dropna().values

        lw = LedoitWolf()
        lw.fit(R)
        self.Sigma_LW: np.ndarray = lw.covariance_  # type: ignore[assignment]

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        ERC via Spinu's convex formulation + projection.

        :return w (np.ndarray): ERC weights (n,)
        """
        n = self.Sigma_LW.shape[0]

        # Spinu log-barrier: min (1/2) y^T Σ y - Σ ln(y_i), then normalize
        y = cp.Variable(n)
        budget_scale = n  # scale factor for log barrier

        objective = cp.Minimize(
            0.5 * cp.quad_form(y, self.Sigma_LW) - budget_scale * cp.sum(cp.log(y))
        )
        constraints: list[cp.Constraint] = [y >= 1e-6]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, max_iters=500)
        except cp.SolverError:
            try:
                prob.solve(solver=cp.SCS, max_iters=5000)
            except cp.SolverError:
                return np.ones(n) / n

        if y.value is None:
            return np.ones(n) / n

        w = np.array(y.value).flatten()
        w = np.maximum(w, 0.0)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            return np.ones(n) / n

        return self._project_to_constraints(w, w_old, is_first)
