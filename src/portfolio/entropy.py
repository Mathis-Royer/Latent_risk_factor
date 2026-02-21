"""
Two-layer Shannon entropy on factor risk contributions and its gradient.

H(w) = (1 - w_idio) · H_factor(w) + w_idio · H_idio(w)

Where:
- H_factor = -Σ_k ĉ'_k ln(ĉ'_k)   (entropy over AU systematic contributions)
- H_idio  = -Σ_i ĉ^ε_i ln(ĉ^ε_i)  (entropy over n idiosyncratic contributions)
- c'_k = (β'_k)² · λ_k (systematic risk contribution, ≥ 0)
- c^ε_i = w_i² · D_{ε,i} (idiosyncratic risk contribution, ≥ 0)

Two-layer formulation prevents the n >> AU idiosyncratic terms from drowning
out the systematic factor diversification signal.  With a flat (single-layer)
entropy over AU + n bins, maximizing H is approximately equivalent to a 1/N
objective because the idiosyncratic block has ~12× more terms than the factor
block.

When D_eps is None, only H_factor is computed (factor-only mode).
When idio_weight=0.0, this reduces to factor-only entropy regardless of D_eps.

Gradient:
  Factor part:   (1-w_idio) · -(2/C_sys) · B' · φ
  Idio part:     w_idio · -(2/C_idio) · w · D_eps · (ln ĉ^ε + H_idio)

WARNING: Must be computed in the PRINCIPAL factor basis (after V rotation),
NOT in the raw latent basis where contributions can be negative.

Reference: ISD Section MOD-008 — Sub-task 1.
Literature: Roncalli (2013) for risk budgeting, Meucci (2009) for entropy
diversification.  Two-layer approach avoids dimensional imbalance between
systematic (AU terms) and idiosyncratic (n terms) populations.
"""

import warnings

import numpy as np


def _entropy_on_contributions(
    contributions: np.ndarray,
    eps: float = 1e-30,
    budget: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """
    Compute Shannon entropy on a vector of risk contributions.

    When budget is None (default): standard Shannon entropy H = -Σ ĉ ln(ĉ).
    When budget is provided: tilted entropy H = -Σ ĉ ln(ĉ/b), which targets
    risk contributions proportional to budget b instead of equal.  Maximizing
    tilted entropy pushes ĉ toward b (minimizes KL(ĉ||b)).  With concentrated
    eigenvalue spectra (λ₁/λ₂ >> 1), this avoids under-weighting dominant
    factors that standard entropy imposes.

    Reference: Roncalli (2013, Ch. 7, Proposition 7.3) — budget risk parity.

    :param contributions (np.ndarray): Risk contributions (m,), all >= 0
    :param eps (float): Numerical stability for log
    :param budget (np.ndarray | None): Target budget distribution (m,),
        must sum to 1.  None = uniform budget (standard Shannon entropy).

    :return H (float): Entropy value (or tilted entropy if budget given)
    :return log_term (np.ndarray): ln(ĉ) or ln(ĉ/b) used in gradient (m,)
    :return C (float): Total risk (sum of contributions)
    """
    C = float(np.sum(contributions))
    if C < eps:
        return 0.0, np.full_like(contributions, np.log(eps)), C

    c_hat = contributions / C
    log_c_hat = np.log(np.maximum(c_hat, eps))

    if budget is not None:
        log_budget = np.log(np.maximum(budget, eps))
        log_ratio = log_c_hat - log_budget
        H = float(-np.sum(c_hat * log_ratio))
        return H, log_ratio, C

    H = float(-np.sum(c_hat * log_c_hat))
    return H, log_c_hat, C


def compute_entropy_and_gradient(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    budget: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute two-layer Shannon entropy H(w) and its gradient.

    H = (1 - idio_weight) * H_factor + idio_weight * H_idio

    where H_factor is entropy over AU systematic risk contributions and
    H_idio is entropy over n idiosyncratic risk contributions.  This
    prevents the n >> AU idiosyncratic terms from drowning out the
    factor diversification signal.

    When D_eps is None or idio_weight=0.0, returns factor-only entropy.

    When budget is provided, H_factor uses tilted entropy (negative KL
    divergence) targeting risk contributions proportional to budget instead
    of equal.  Budget only applies to the factor layer, not idiosyncratic.

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param eps (float): Numerical stability for log
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,).
        When None, only systematic contributions are used.
    :param idio_weight (float): Weight for idiosyncratic entropy layer.
        0.0 = factor-only, 1.0 = idio-only.  Default 0.2.
    :param budget (np.ndarray | None): Target factor risk budget (AU,),
        must sum to 1.  None = uniform (standard Shannon entropy).
        Recommended: eigenvalues / eigenvalues.sum().

    :return H (float): Combined entropy value
    :return grad_H (np.ndarray): Gradient (n,)
    """
    # Validate idio_weight in [0, 1]
    assert 0.0 <= idio_weight <= 1.0, (
        f"idio_weight must be in [0, 1], got {idio_weight}"
    )

    # Shape validation: B_prime rows must match w length
    assert B_prime.shape[0] == len(w), (
        f"B_prime rows {B_prime.shape[0]} != w length {len(w)}"
    )

    # Shape validation: eigenvalues must match B_prime columns
    assert len(eigenvalues) == B_prime.shape[1], (
        f"eigenvalues length {len(eigenvalues)} != B_prime cols {B_prime.shape[1]}"
    )

    # Shape validation
    if D_eps is not None:
        assert len(D_eps) == len(w), (
            f"D_eps length {len(D_eps)} != w length {len(w)}"
        )

    # CRITICAL: Validate eigenvalues are non-negative (diagnostic fix)
    assert np.all(eigenvalues >= -1e-10), (
        f"Negative eigenvalues detected: min={eigenvalues.min():.2e}"
    )

    # Portfolio exposure in principal factor basis
    beta_prime = B_prime.T @ w  # (AU,)

    # Systematic risk contributions: c'_k = (β'_k)² · λ_k ≥ 0
    c_sys = (beta_prime ** 2) * eigenvalues  # (AU,)

    # Factor entropy (tilted when budget is provided)
    H_factor, log_c_hat_sys, C_sys = _entropy_on_contributions(c_sys, eps, budget=budget)

    # Factor gradient: -(2/C_sys) · B' · φ
    # Two-layer weights only apply when idiosyncratic layer is active
    two_layer = D_eps is not None and idio_weight > 0.0
    w_factor = (1.0 - idio_weight) if two_layer else 1.0

    if C_sys > eps:
        phi = eigenvalues * beta_prime * (log_c_hat_sys + H_factor)  # (AU,)
        grad_factor = -(2.0 / C_sys) * (B_prime @ phi)  # (n,)
    else:
        grad_factor = np.zeros_like(w)

    assert np.isfinite(grad_factor).all(), "grad_factor contains NaN/Inf"

    grad_H = w_factor * grad_factor

    # Idiosyncratic entropy (separate layer)
    if two_layer:
        assert D_eps is not None  # for type narrowing
        c_idio = (w ** 2) * D_eps  # (n,)
        H_idio, log_c_hat_idio, C_idio = _entropy_on_contributions(c_idio, eps)

        if C_idio > eps:
            grad_idio = -(2.0 / C_idio) * w * D_eps * (log_c_hat_idio + H_idio)
        else:
            grad_idio = np.zeros_like(w)

        grad_H = grad_H + idio_weight * grad_idio
        H = w_factor * H_factor + idio_weight * H_idio
    else:
        H = H_factor

    # Note: Tilted entropy (with budget) can legitimately be negative
    # because H_tilted = -KL(ĉ||b). Only standard entropy (budget=None) is ≥ 0.
    # Reference: Roncalli (2013, Ch. 7, Prop 7.3)
    if budget is None:
        # Standard entropy must be non-negative
        assert H >= -1e-10, f"Standard entropy is negative: {H}"
        H = max(H, 0.0)

    return float(H), grad_H


def compute_entropy_only(
    w: np.ndarray,
    B_prime: np.ndarray,
    eigenvalues: np.ndarray,
    eps: float = 1e-30,
    D_eps: np.ndarray | None = None,
    idio_weight: float = 0.2,
    budget: np.ndarray | None = None,
) -> float:
    """
    Compute two-layer entropy H(w) without gradient (faster for evaluation).

    H = (1 - idio_weight) * H_factor + idio_weight * H_idio

    When D_eps is None or idio_weight=0.0, returns factor-only entropy.
    When budget is provided, H_factor uses tilted entropy (factor layer only).

    :param w (np.ndarray): Portfolio weights (n,)
    :param B_prime (np.ndarray): Rotated exposures (n, AU)
    :param eigenvalues (np.ndarray): Principal eigenvalues (AU,)
    :param eps (float): Numerical stability
    :param D_eps (np.ndarray | None): Idiosyncratic variances (n,).
        When None, only systematic contributions are used.
    :param idio_weight (float): Weight for idiosyncratic entropy layer.
    :param budget (np.ndarray | None): Target factor risk budget (AU,).

    :return H (float): Combined entropy value
    """
    # Shape validation
    if D_eps is not None:
        assert len(D_eps) == len(w), (
            f"D_eps length {len(D_eps)} != w length {len(w)}"
        )

    beta_prime = B_prime.T @ w
    c_sys = (beta_prime ** 2) * eigenvalues

    H_factor, _, _ = _entropy_on_contributions(c_sys, eps, budget=budget)

    if D_eps is not None and idio_weight > 0.0:
        w_factor = 1.0 - idio_weight
        c_idio = (w ** 2) * D_eps
        H_idio, _, _ = _entropy_on_contributions(c_idio, eps)
        return float(w_factor * H_factor + idio_weight * H_idio)

    return float(H_factor)
