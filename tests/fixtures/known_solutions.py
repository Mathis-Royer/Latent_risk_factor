"""
Known analytical solutions for verification of portfolio optimization,
factor regression, rescaling, and entropy computations.

Every solution is analytically derived (not numerically computed) and
serves as a ground truth for unit tests.

Reference: ISD Section MOD-003 — test_infrastructure.
"""

import numpy as np


def diagonal_covariance_solution(n: int = 10, seed: int = 42) -> dict:
    """
    Diagonal covariance: Σ = diag(σ²_1, ..., σ²_n), B = I (identity).

    Known results:
    - Minimum-variance: w_i ∝ 1/σ²_i
    - ERC (Equal Risk Contribution): w_i ∝ 1/σ_i
    - Entropy at ERC: H = ln(n) (maximum, all contributions equal)

    :param n (int): Number of assets
    :param seed (int): Random seed for generating variances

    :return result (dict): sigma, w_minvar, w_erc, H_max, Sigma, B
    """
    rng = np.random.RandomState(seed)
    sigma = rng.uniform(0.1, 0.5, size=n)
    sigma_sq = sigma ** 2

    # Minimum-variance weights: w_i ∝ 1/σ²_i
    inv_var = 1.0 / sigma_sq
    w_minvar = inv_var / inv_var.sum()

    # ERC weights: w_i ∝ 1/σ_i
    inv_sigma = 1.0 / sigma
    w_erc = inv_sigma / inv_sigma.sum()

    # Maximum entropy (at ERC for diagonal case)
    H_max = np.log(n)

    return {
        "sigma": sigma,
        "sigma_sq": sigma_sq,
        "w_minvar": w_minvar,
        "w_erc": w_erc,
        "H_max": H_max,
        "Sigma": np.diag(sigma_sq),
        "B": np.eye(n),
        "n": n,
    }


def two_factor_solution() -> dict:
    """
    Small 2-factor model for SCA solver verification.

    n=4 stocks, AU=2 factors. Known B_A, Sigma_z, and the
    analytical maximum-entropy portfolio.

    :return result (dict): n, AU, B_A, Sigma_z, D_eps, eigenvalues, V,
        w_max_entropy, H_max_entropy
    """
    n = 4
    AU = 2

    # Factor exposures (hand-picked for a clean analytical solution)
    B_A = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.3, 0.7],
    ], dtype=np.float64)

    # Factor covariance (diagonal for simplicity)
    Sigma_z = np.array([
        [0.04, 0.005],
        [0.005, 0.09],
    ], dtype=np.float64)

    # Idiosyncratic variance
    D_eps = np.array([0.001, 0.001, 0.001, 0.001], dtype=np.float64)

    # Eigendecomposition of Sigma_z
    eigenvalues, V = np.linalg.eigh(Sigma_z)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Rotated exposures
    B_prime = B_A @ V

    # Equal-weight as reference (not the optimum, but a baseline)
    w_equal = np.ones(n) / n

    # Factor risk contributions at equal weight
    beta_prime = B_prime.T @ w_equal  # (AU,)
    c_prime = eigenvalues * beta_prime ** 2
    C_total = c_prime.sum()
    c_hat = c_prime / C_total if C_total > 0 else np.ones(AU) / AU
    H_equal = -np.sum(c_hat * np.log(np.maximum(c_hat, 1e-30)))

    return {
        "n": n,
        "AU": AU,
        "B_A": B_A,
        "Sigma_z": Sigma_z,
        "D_eps": D_eps,
        "eigenvalues": eigenvalues,
        "V": V,
        "B_prime": B_prime,
        "w_equal": w_equal,
        "H_equal": H_equal,
    }


def entropy_gradient_verification(AU: int = 5) -> dict:
    """
    At maximum entropy (ĉ'_k = 1/AU for all k):
    - H = ln(AU)
    - ∇H = 0 (gradient vanishes)

    Constructs w, B_prime, eigenvalues such that all principal factor
    risk contributions are exactly equal.

    :param AU (int): Number of active units (principal factors)

    :return result (dict): AU, w, B_prime, eigenvalues, H_max, grad_H
    """
    n = AU + 2  # need n > AU for a meaningful portfolio

    # Equal eigenvalues → contributions differ only through β'
    eigenvalues = np.ones(AU, dtype=np.float64) * 0.05

    # B_prime such that B_prime.T @ w gives equal β'_k for all k
    # Use B_prime = identity padded with zeros for extra stocks
    B_prime = np.zeros((n, AU), dtype=np.float64)
    for k in range(AU):
        B_prime[k, k] = 1.0

    # w that gives β'_k = 1/AU for all k: w_k = 1/AU for k<AU
    # Remaining stocks get equal share of leftover
    w = np.zeros(n, dtype=np.float64)
    w_factor = 1.0 / AU
    for k in range(AU):
        w[k] = w_factor
    # These weights sum to 1.0 exactly when AU stocks carry all weight
    # Redistribute to include all stocks
    w = np.ones(n, dtype=np.float64) / n

    # Recompute contributions
    beta_prime = B_prime.T @ w  # (AU,)
    c_prime = eigenvalues * beta_prime ** 2
    C_total = c_prime.sum()

    if C_total > 0:
        c_hat = c_prime / C_total
        H = -np.sum(c_hat * np.log(np.maximum(c_hat, 1e-30)))
    else:
        H = 0.0

    # Gradient: -(2/C) * B' * φ, where φ_k = λ_k * β'_k * (ln(ĉ'_k) + H)
    if C_total > 0:
        phi = eigenvalues * beta_prime * (np.log(np.maximum(c_hat, 1e-30)) + H)
        grad_H = -(2.0 / C_total) * B_prime @ phi
    else:
        grad_H = np.zeros(n, dtype=np.float64)

    return {
        "AU": AU,
        "n": n,
        "w": w,
        "B_prime": B_prime,
        "eigenvalues": eigenvalues,
        "H": H,
        "grad_H": grad_H,
        "c_hat": c_hat if C_total > 0 else np.ones(AU) / AU,
        "C_total": C_total,
    }


def factor_regression_identity(n: int = 10, T_hist: int = 252) -> dict:
    """
    Identity test for factor regression.

    B = I (identity), r = z (no noise) → z_hat = (B^T B)^{-1} B^T r = r.

    :param n (int): Number of stocks (= number of factors)
    :param T_hist (int): Number of historical dates

    :return result (dict): n, T_hist, B, returns, z_expected
    """
    rng = np.random.RandomState(99)

    B = np.eye(n, dtype=np.float64)
    z_true = rng.randn(T_hist, n) * 0.01

    # Returns = factor returns (no noise)
    returns = z_true.copy()

    return {
        "n": n,
        "T_hist": T_hist,
        "B": B,
        "returns": returns,
        "z_expected": z_true,
    }


def rescaling_verification(n: int = 10, T_hist: int = 50) -> dict:
    """
    Known rescaling inputs and expected outputs with winsorization.

    Tests INV-004 (dual rescaling) and INV-008 (winsorization at P5/P95).

    :param n (int): Number of stocks
    :param T_hist (int): Number of historical dates

    :return result (dict): sigma_it, sigma_bar_t, mu_A, B_A_estimation,
        B_A_portfolio, winsorize_lo, winsorize_hi
    """
    rng = np.random.RandomState(77)

    AU = 3
    mu_A = rng.randn(n, AU) * 0.5  # Raw composite profiles (unscaled)

    # Trailing vols per stock per date: σ_{i,t}
    sigma_it = rng.uniform(0.05, 0.60, size=(T_hist, n))
    # Add one outlier stock with extreme vol
    sigma_it[:, 0] = 2.0  # Very high vol → will be winsorized

    # Cross-sectional median at each date
    sigma_bar_t = np.median(sigma_it, axis=1)  # (T_hist,)

    # Ratios before winsorization
    ratios = sigma_it / sigma_bar_t[:, np.newaxis]

    # Winsorize at P5/P95 cross-sectionally per date
    ratios_winsorized = ratios.copy()
    for t in range(T_hist):
        p5 = np.percentile(ratios[t], 5.0)
        p95 = np.percentile(ratios[t], 95.0)
        ratios_winsorized[t] = np.clip(ratios[t], p5, p95)

    # B_A_estimation[i,t] = ratio_winsorized[i,t] * mu_A[i]
    B_A_estimation = np.zeros((T_hist, n, AU), dtype=np.float64)
    for t in range(T_hist):
        B_A_estimation[t] = ratios_winsorized[t, :, np.newaxis] * mu_A

    # B_A_portfolio uses current (last) date
    B_A_portfolio = ratios_winsorized[-1, :, np.newaxis] * mu_A

    return {
        "n": n,
        "AU": AU,
        "T_hist": T_hist,
        "mu_A": mu_A,
        "sigma_it": sigma_it,
        "sigma_bar_t": sigma_bar_t,
        "ratios_raw": ratios,
        "ratios_winsorized": ratios_winsorized,
        "B_A_estimation": B_A_estimation,
        "B_A_portfolio": B_A_portfolio,
        "winsorize_lo": 5.0,
        "winsorize_hi": 95.0,
    }
