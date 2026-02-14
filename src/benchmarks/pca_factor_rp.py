"""
MOD-014: PCA factor risk parity benchmark.

The most important benchmark — isolates VAE added value vs linear PCA.

1. PCA on returns matrix (T_est × n)
2. k factors via Bai & Ng (2002) IC₂
3. Factor risk model: Σ_assets = B_PCA Λ_k B_PCA^T + D_ε_PCA
4. Portfolio optimization: SAME SCA solver as VAE (MOD-008)
5. Constraints identical (INV-012)

Reference: ISD Section MOD-014.
"""

import numpy as np
import pandas as pd

from src.benchmarks.base import BenchmarkModel
from src.portfolio.entropy import compute_entropy_and_gradient, compute_entropy_only
from src.portfolio.sca_solver import multi_start_optimize
from src.risk_model.covariance import estimate_d_eps


class PCAFactorRiskParity(BenchmarkModel):
    """PCA factor risk parity using Bai-Ng IC₂ for factor selection."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        PCA decomposition + IC₂ factor selection.

        :param returns (pd.DataFrame): Historical returns
        :param universe (list[str]): Active stock identifiers
        :param k_max (int): Maximum factors to test (in kwargs, default 30)
        """
        self.n = len(universe)
        self.universe = universe

        k_max = int(kwargs.get("k_max", 30))  # type: ignore[arg-type]

        available = [s for s in universe if s in returns.columns]
        R = returns[available].dropna()
        R_mat = R.values.astype(np.float64)

        # Demean
        R_centered = R_mat - R_mat.mean(axis=0, keepdims=True)

        T_est, n = R_centered.shape

        # PCA via SVD
        U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)

        # Bai-Ng IC₂ for factor selection
        k_star = self._bai_ng_ic2(R_centered, k_max=min(k_max, min(T_est, n) - 1))
        k_star = max(1, k_star)

        # PCA loadings: B_PCA (n × k) — unscaled eigenvectors
        # Variance is carried entirely by eigenvalues = S²/T, NOT by B.
        # Using B = Vt.T * S would double-count S in B @ diag(S²/T) @ B^T.
        self.k = k_star
        self.B_PCA = Vt[:k_star].T  # (n, k) — pure eigenvector directions

        # Factor covariance: diagonal (principal components are orthogonal)
        # eigenvalues = S²/T
        self.eigenvalues = (S[:k_star] ** 2) / T_est  # (k,)

        # Rotated exposures B' — since Σ_z is already diagonal, V=I
        # So B_prime = B_PCA (already in principal basis)
        self.B_prime = self.B_PCA.copy()

        # Idiosyncratic variances
        # Reconstruction: R_approx = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        R_approx = U[:, :k_star] @ np.diag(S[:k_star]) @ Vt[:k_star, :]
        residuals = R_centered - R_approx

        self.D_eps = np.maximum(
            np.var(residuals, axis=0, ddof=1),
            1e-6,
        )

        # Asset covariance
        self.Sigma_assets = (
            self.B_PCA @ np.diag(self.eigenvalues) @ self.B_PCA.T
            + np.diag(self.D_eps)
        )

    def _bai_ng_ic2(
        self,
        R_centered: np.ndarray,
        k_max: int = 30,
    ) -> int:
        """
        Bai & Ng (2002) Information Criterion IC₂.

        IC₂(k) = ln(V(k)) + k · ((n+T)/(n·T)) · ln(min(n,T))
        V(k) = (1/(n·T)) · ||R - F_k Λ_k^T||²_F

        :param R_centered (np.ndarray): Centered returns (T, n)
        :param k_max (int): Maximum factors to consider

        :return k_star (int): Optimal number of factors
        """
        T_est, n = R_centered.shape
        U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)

        penalty_coeff = ((n + T_est) / (n * T_est)) * np.log(min(n, T_est))

        best_ic = float("inf")
        best_k = 1

        for k in range(1, min(k_max + 1, len(S))):
            # Reconstruction error
            R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            V_k = np.sum((R_centered - R_approx) ** 2) / (n * T_est)

            ic2 = np.log(max(V_k, 1e-30)) + k * penalty_coeff

            if ic2 < best_ic:
                best_ic = ic2
                best_k = k

        return best_k

    def optimize(
        self,
        w_old: np.ndarray | None = None,
        is_first: bool = False,
    ) -> np.ndarray:
        """
        SAME SCA solver as VAE. H(w) in principal PCA factor basis.

        Since Σ_z_PCA = Λ_k is diagonal, V=I, so B_prime = B_PCA.

        :return w (np.ndarray): Optimized weights (n,)
        """
        n = self.Sigma_assets.shape[0]

        w_opt, _, _ = multi_start_optimize(
            Sigma_assets=self.Sigma_assets,
            B_prime=self.B_prime,
            eigenvalues=self.eigenvalues,
            D_eps=self.D_eps,
            alpha=0.1,  # Default α, can be overridden via frontier
            n_starts=5,
            seed=42,
            lambda_risk=self.constraint_params["lambda_risk"],
            w_max=self.constraint_params["w_max"],
            w_min=self.constraint_params["w_min"],
            phi=self.constraint_params["phi"],
            w_bar=0.03,
            w_old=w_old,
            kappa_1=self.constraint_params["kappa_1"],
            kappa_2=self.constraint_params["kappa_2"],
            delta_bar=self.constraint_params["delta_bar"],
            tau_max=self.constraint_params["tau_max"],
            is_first=is_first,
        )

        return w_opt
