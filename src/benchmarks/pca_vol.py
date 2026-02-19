"""
MOD-015: PCA + realized volatility variant.

Variant of MOD-014 with augmented matrix (T × 2n):
  [z-scored returns | z-scored 21d realized vol]

PCA on this augmented matrix → factor risk model → same SCA solver.
Isolates VAE non-linearity independently of feature enrichment.

Reference: ISD Section MOD-015.
"""

import numpy as np
import pandas as pd

from src.benchmarks.pca_factor_rp import PCAFactorRiskParity


class PCAVolRiskParity(PCAFactorRiskParity):
    """PCA factor risk parity on augmented returns + volatility matrix."""

    def fit(
        self,
        returns: pd.DataFrame,
        universe: list[str],
        **kwargs: object,
    ) -> None:
        """
        PCA on augmented matrix [returns | vol], both z-scored.

        :param returns (pd.DataFrame): Historical returns
        :param universe (list[str]): Active stock identifiers
        :param trailing_vol (pd.DataFrame): 21d rolling vol (in kwargs)
        :param k_max (int): Maximum factors (in kwargs, default 30)
        """
        self.n = len(universe)
        self.universe = universe

        k_max = int(kwargs.get("k_max", 30))  # type: ignore[arg-type]

        available = [s for s in universe if s in returns.columns]

        # Returns matrix
        R = returns[available].dropna()
        R_mat = R.values.astype(np.float64)

        # Z-score returns
        R_mean = R_mat.mean(axis=0, keepdims=True)
        R_std = R_mat.std(axis=0, keepdims=True)
        R_std = np.maximum(R_std, 1e-10)
        R_z = (R_mat - R_mean) / R_std

        # Volatility matrix
        trailing_vol = kwargs.get("trailing_vol")
        if trailing_vol is not None:
            vol_df: pd.DataFrame = trailing_vol  # type: ignore[assignment]
            vol_available = [s for s in available if s in vol_df.columns]
            V_mat = vol_df.loc[R.index, vol_available].values.astype(np.float64)  # type: ignore[union-attr]

            # Handle NaN in vol
            V_mat = np.nan_to_num(V_mat, nan=0.0)

            # Z-score vol
            V_mean = V_mat.mean(axis=0, keepdims=True)
            V_std = V_mat.std(axis=0, keepdims=True)
            V_std = np.maximum(V_std, 1e-10)
            V_z = (V_mat - V_mean) / V_std
        else:
            # Fallback: compute 21d rolling vol from returns
            rolling_vol = returns[available].rolling(21).std()
            V_mat = rolling_vol.loc[R.index].values.astype(np.float64)
            V_mat = np.nan_to_num(V_mat, nan=0.0)
            V_mean = V_mat.mean(axis=0, keepdims=True)
            V_std = V_mat.std(axis=0, keepdims=True)
            V_std = np.maximum(V_std, 1e-10)
            V_z = (V_mat - V_mean) / V_std

        # Augmented matrix: (T, 2n)
        R_augmented = np.hstack([R_z, V_z[:R_z.shape[0], :R_z.shape[1]]])

        # Center
        R_centered = R_augmented - R_augmented.mean(axis=0, keepdims=True)

        T_est, n_aug = R_centered.shape
        n_orig = len(available)

        # PCA via SVD on augmented matrix
        U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)

        # Bai-Ng IC₂ on returns-only matrix (not augmented).
        # Using augmented dimensions (2n) changes the penalty coefficient
        # in ways not calibrated by the original Bai-Ng (2002) paper.
        R_z_centered = R_z - R_z.mean(axis=0, keepdims=True)
        k_star = self._bai_ng_ic2(
            R_z_centered, k_max=min(k_max, min(T_est, n_orig) - 1),
        )
        k_star = max(1, k_star)
        self.k = k_star

        # Extract loadings for the original n stocks only (first n columns)
        # Unscaled eigenvectors in z-scored space, then rescale to original
        # return space by multiplying each row i by the stock's std.
        # This ensures B @ diag(eigenvalues) @ B^T is in original variance units.
        B_z = Vt[:k_star, :n_orig].T  # (n, k) — z-scored-space eigenvectors
        self.B_PCA = B_z * R_std.reshape(-1, 1)  # (n, k) — original-scale loadings

        # Eigenvalues: S²/T (variance carried here, NOT in B)
        self.eigenvalues = (S[:k_star] ** 2) / T_est

        # B_prime = B_PCA (Σ_z already diagonal)
        self.B_prime = self.B_PCA.copy()

        # Idiosyncratic variances in original return space
        # Reconstruct z-scored returns, then convert to original scale
        R_z_approx = U[:, :k_star] @ np.diag(S[:k_star]) @ Vt[:k_star, :n_orig]
        R_orig_approx = R_z_approx * R_std  # back to original scale
        R_orig_centered = R_mat - R_mean
        residuals = R_orig_centered - R_orig_approx

        self.D_eps = np.maximum(
            np.var(residuals, axis=0, ddof=1),
            1e-6,
        )

        # Asset covariance
        self.Sigma_assets = (
            self.B_PCA @ np.diag(self.eigenvalues) @ self.B_PCA.T
            + np.diag(self.D_eps)
        )

    def rebalance(
        self,
        returns_trailing: "pd.DataFrame",
        trailing_vol: "pd.DataFrame | None",
        w_old: np.ndarray,
        universe: list[str],
        current_date: str,
    ) -> np.ndarray:
        """
        Re-run PCA on trailing returns + vol and re-optimize.

        :return w (np.ndarray): PCA-Vol-RP weights on new universe
        """
        self.fit(returns_trailing, universe, trailing_vol=trailing_vol)
        return self.optimize(w_old=w_old, is_first=False)
