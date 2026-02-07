"""
Covariance estimation: Σ_z (Ledoit-Wolf), D_ε (idiosyncratic), Σ_assets assembly.

- Σ_z: empirical covariance of ẑ_t over FULL history + Ledoit-Wolf shrinkage
- D_ε: per-stock variance of residuals, floored at 1e-6
- Σ_assets = B̃^port Σ_z (B̃^port)^T + D_ε
- Eigendecomposition: Σ_z = V Λ V^T → rotated B' = B̃^port V

Reference: ISD Section MOD-007 — Sub-tasks 3-4.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_sigma_z(
    z_hat: np.ndarray,
) -> np.ndarray:
    """
    Estimate factor covariance Σ_z using Ledoit-Wolf shrinkage.

    Σ̂_z = (1-δ*) S_emp + δ* · (tr(S_emp)/AU) · I_AU

    Uses the FULL history (anti-cyclical principle).

    :param z_hat (np.ndarray): Factor returns (n_dates, AU)

    :return Sigma_z (np.ndarray): Shrunk factor covariance (AU, AU)
    """
    lw = LedoitWolf()
    lw.fit(z_hat)
    return lw.covariance_  # type: ignore[return-value]


def estimate_d_eps(
    residuals_by_stock: dict[int, list[float]],
    stock_ids: list[int],
    d_eps_floor: float = 1e-6,
) -> np.ndarray:
    """
    Estimate idiosyncratic variances D_ε from residuals.

    D_{ε,i} = Var(ε_{i,·}) over all dates where stock i was active.
    Floor: D_{ε,i} ≥ d_eps_floor.

    :param residuals_by_stock (dict): stock_id → list of residuals
    :param stock_ids (list[int]): Ordered stock IDs (permnos)
    :param d_eps_floor (float): Minimum idiosyncratic variance

    :return D_eps (np.ndarray): Idiosyncratic variances (n,)
    """
    D_eps = np.full(len(stock_ids), d_eps_floor, dtype=np.float64)

    for i, sid in enumerate(stock_ids):
        resids = residuals_by_stock.get(sid, [])
        if len(resids) >= 2:
            var_eps = np.var(resids, ddof=1)
            D_eps[i] = max(var_eps, d_eps_floor)

    return D_eps


def assemble_risk_model(
    B_A_port: np.ndarray,
    Sigma_z: np.ndarray,
    D_eps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Assemble full asset covariance and compute eigendecomposition.

    Σ_assets = B̃^port · Σ_z · (B̃^port)^T + diag(D_ε)

    Eigendecomposition: Σ_z = V Λ V^T
    Rotated exposures: B' = B̃^port · V

    :param B_A_port (np.ndarray): Portfolio-rescaled exposures (n, AU)
    :param Sigma_z (np.ndarray): Factor covariance (AU, AU)
    :param D_eps (np.ndarray): Idiosyncratic variances (n,)

    :return risk_model (dict): Contains Sigma_assets, eigenvalues, V, B_prime_port
    """
    n, AU = B_A_port.shape

    # Σ_assets = B Σ_z B^T + D_ε
    # For large n, compute as factor model (don't form full n×n if n is huge)
    Sigma_assets = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)

    # Eigendecomposition of Σ_z
    eigenvalues, V = np.linalg.eigh(Sigma_z)

    # Sort descending
    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    V = V[:, sort_idx]

    # INV-007: eigenvalues ≥ 0 (PSD)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Rotated exposures in principal factor basis
    B_prime_port = B_A_port @ V

    return {
        "Sigma_assets": Sigma_assets,
        "eigenvalues": eigenvalues,
        "V": V,
        "B_prime_port": B_prime_port,
    }
