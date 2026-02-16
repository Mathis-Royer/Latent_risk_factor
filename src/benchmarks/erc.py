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

When CVXPY solvers fail (common for n > 500), a direct Newton-based
solver (Roncalli 2013, Ch. 11) is used as fallback.

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


def _newton_erc(
    Sigma: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray | None:
    """
    Newton-based ERC solver (Roncalli 2013, Ch. 11, Algorithm 11.1).

    Solves the fixed-point equation (Σy)_i · y_i = c for all i,
    then normalizes w = y / sum(y).

    More numerically stable than CVXPY log-barrier for large n because
    it operates on the n-dimensional system directly without interior-point
    overhead.

    :param Sigma (np.ndarray): Covariance matrix (n, n), must be PD
    :param max_iter (int): Maximum Newton iterations
    :param tol (float): Convergence tolerance on ||F(y)||

    :return w (np.ndarray | None): ERC weights (n,), or None if failed
    """
    n = Sigma.shape[0]

    # Initialize with inverse-volatility (good starting point for ERC)
    diag_sigma = np.diag(Sigma)
    if np.any(diag_sigma <= 0):
        return None
    y = 1.0 / np.sqrt(diag_sigma)

    for it in range(max_iter):
        Sy = Sigma @ y  # (n,)
        # Residual: F_i = y_i * (Σy)_i - 1
        F = y * Sy - 1.0
        norm_F = float(np.linalg.norm(F))

        if norm_F < tol:
            logger.info(
                "  [ERC/Newton] Converged in %d iterations (||F||=%.2e)",
                it + 1, norm_F,
            )
            break

        # Jacobian: J_ij = δ_ij (Σy)_i + y_i Σ_ij
        # J = diag(Sy) + diag(y) @ Σ = diag(Sy) + Y Σ
        J = np.diag(Sy) + np.diag(y) @ Sigma

        # Newton step: y_new = y - J^{-1} F
        try:
            delta = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            logger.warning(
                "  [ERC/Newton] Singular Jacobian at iteration %d", it + 1,
            )
            return None

        # Damped Newton: ensure y stays positive
        step = 1.0
        for _ in range(20):
            y_trial = y - step * delta
            if np.all(y_trial > 0):
                break
            step *= 0.5
        else:
            logger.warning(
                "  [ERC/Newton] Cannot maintain positive y at iteration %d",
                it + 1,
            )
            return None

        y = y_trial
    else:
        logger.warning(
            "  [ERC/Newton] Did not converge after %d iterations (||F||=%.2e)",
            max_iter, norm_F,
        )
        # Return best effort if residual is reasonable
        if norm_F > 1.0:
            return None

    # Normalize to portfolio weights
    w = y / np.sum(y)
    return w


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
        R_df = returns[available]

        # Filter stocks with >50% NaN, then fill residual NaN with 0.
        # dropna() on 600+ stocks eliminates nearly all rows; this
        # matches the MinVar benchmark NaN handling in pipeline.py.
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
        ERC via Spinu's convex formulation + Newton fallback + projection.

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

        if solved and y.value is not None:
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
                solved = False

        # Newton fallback when CVXPY solvers fail (common for n > 500)
        if not solved:
            logger.warning(
                "  [ERC] CVXPY solvers failed (n=%d) — trying Newton solver "
                "(Roncalli 2013, Ch. 11)",
                n,
            )
            w_newton = _newton_erc(self.Sigma_LW)
            if w_newton is not None:
                w = w_newton
                solved = True
                solver_used = "Newton"
            else:
                logger.warning(
                    "  [ERC] All solvers failed (CVXPY + Newton) — "
                    "falling back to 1/N"
                )
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
            rc_ratio = rc_max / max(rc_min, 1e-10)
            eff_n_erc = 1.0 / max(float(np.sum(rc_norm ** 2)), 1e-10)
            logger.info(
                "  [ERC] Risk contributions (%s): std=%.6f, max/min=%.2f, "
                "eff_n=%.1f/%d",
                solver_used, rc_std, rc_ratio, eff_n_erc, n,
            )

        return self._project_to_constraints(w, w_old, is_first)
