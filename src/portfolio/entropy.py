"""
Shannon entropy on principal factor risk contributions and its gradient.

H(w) = -Σ_k ĉ'_k ln(ĉ'_k)

Where:
- B' = rotated exposures (n × AU), β' = B'^T w (AU,)
- c'_k = (β'_k)² · λ_k (risk contribution, ≥ 0)
- C = Σ_k c'_k (total systematic risk)
- ĉ'_k = c'_k / C (normalized)

Gradient: ∇_w H = -(2/C) · B' · φ, where φ_k = λ_k β'_k (ln ĉ'_k + H)

WARNING: Must be computed in the PRINCIPAL factor basis (after V rotation),
NOT in the raw latent basis where contributions can be negative.

Reference: ISD Section MOD-008 — Sub-task 1.
"""

import numpy as np


def compute_entropy_and_gradient(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    eps: float = 1e-30,
) -> tuple[float, np.ndarray]:
    """
    Compute Shannon entropy H(w) and its gradient ∇_w H.

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param eps (float): Numerical stability for log

    :return H (float): Entropy value
    :return grad_H (np.ndarray): Gradient (n,)
    """
    # Portfolio exposure in principal factor basis
    beta_prime = B_prime.T @ w  # (AU,)

    # Risk contributions: c'_k = (β'_k)² · λ_k ≥ 0
    c_prime = (beta_prime ** 2) * eigenvalues  # (AU,)

    # Total systematic risk
    C = np.sum(c_prime)
    if C < eps:
        # Degenerate case: zero systematic risk
        return 0.0, np.zeros_like(w)

    # Normalized contributions
    c_hat = c_prime / C  # (AU,)

    # Entropy: H = -Σ_k ĉ'_k ln(ĉ'_k)
    # Protect log with eps
    log_c_hat = np.log(np.maximum(c_hat, eps))
    H = -np.sum(c_hat * log_c_hat)

    # Gradient: ∇_w H = -(2/C) · B' · φ
    # φ_k = λ_k · β'_k · (ln ĉ'_k + H)
    phi = eigenvalues * beta_prime * (log_c_hat + H)  # (AU,)
    grad_H = -(2.0 / C) * (B_prime @ phi)  # (n,)

    return float(H), grad_H


def compute_entropy_only(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    eps: float = 1e-30,
) -> float:
    """
    Compute entropy H(w) without gradient (faster for evaluation).

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param eps (float): Numerical stability

    :return H (float): Entropy value
    """
    beta_prime = B_prime.T @ w
    c_prime = (beta_prime ** 2) * eigenvalues
    C = np.sum(c_prime)

    if C < eps:
        return 0.0

    c_hat = c_prime / C
    log_c_hat = np.log(np.maximum(c_hat, eps))
    return float(-np.sum(c_hat * log_c_hat))
