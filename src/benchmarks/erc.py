"""
MOD-013: Equal Risk Contribution (ERC) benchmark via Spinu (2013).

Convex log-barrier formulation:
  min (1/2) y^T Σ y - c · Σ_i ln(y_i)

where c is chosen so the optimality condition yields equal risk
contributions: RC_i = w_i * (Σw)_i = const.  Spinu (2013) shows
c = 1 (the theoretical barrier weight) is sufficient — the scaling
drops out at normalization.  Using c = n (as some implementations do)
makes the barrier dominate the quadratic term for large n, pushing
the solution toward 1/N regardless of Σ.

Post-hoc projection onto hard caps.

Reference: ISD Section MOD-013.
"""

import logging

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from src.benchmarks.base import BenchmarkModel

logger = logging.getLogger(__name__)


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

        # Spinu log-barrier: min (1/2) y^T Σ y - c * Σ ln(y_i), then normalize
        # c = 1 (theory): barrier weight is scale-invariant after normalization.
        # c = n was used before but makes barrier dominate for large n → EW.
        y = cp.Variable(n)
        budget_scale = 1.0

        objective = cp.Minimize(
            0.5 * cp.quad_form(y, self.Sigma_LW) - budget_scale * cp.sum(cp.log(y))
        )
        constraints: list[cp.Constraint] = [y >= 1e-6]

        prob = cp.Problem(objective, constraints)

        # Solver chain: MOSEK > ECOS > SCS (DVT Section 4.7)
        solved = False
        solver_used = "none"
        for solver, slv_kwargs in [
            (cp.MOSEK, {"warm_start": True}),
            (cp.ECOS, {"warm_start": True, "max_iters": 500}),
            (cp.SCS, {"warm_start": True, "max_iters": 5000}),
        ]:
            try:
                prob.solve(solver=solver, **slv_kwargs)
                if y.value is not None:
                    solved = True
                    solver_used = str(solver)
                    break
            except (cp.SolverError, Exception):
                continue

        if not solved or y.value is None:
            logger.warning("  [ERC] All solvers failed — falling back to 1/N")
            return np.ones(n) / n

        logger.info(
            "  [ERC] Solved via %s, status=%s",
            solver_used, prob.status,
        )

        w = np.array(y.value).flatten()
        w = np.maximum(w, 0.0)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            logger.warning("  [ERC] Zero-sum solution — falling back to 1/N")
            return np.ones(n) / n

        # ERC condition check: RC_i = w_i * (Σw)_i should be equal
        Sigma_w = self.Sigma_LW @ w
        rc = w * Sigma_w
        rc_total = float(np.sum(rc))
        if rc_total > 1e-12:
            rc_norm = rc / rc_total
            rc_std = float(np.std(rc_norm))
            rc_max = float(np.max(rc_norm))
            rc_min = float(np.min(rc_norm[rc_norm > 1e-10]))
            logger.info(
                "  [ERC] Risk contributions: std=%.6f, max/min=%.2f, "
                "eff_n=%.1f/%d",
                rc_std, rc_max / max(rc_min, 1e-10),
                1.0 / max(float(np.sum(rc_norm ** 2)), 1e-10), n,
            )

        return self._project_to_constraints(w, w_old, is_first)
