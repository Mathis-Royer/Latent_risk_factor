"""
Deterministic synthetic data generators for all unit and integration tests.

Every test in the project should be reproducible with fixed seeds and
independent of external data (CRSP, VIX files, EODHD).

Reference: ISD Section MOD-003 — test_infrastructure.
"""

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

TEST_SEED: int = 42
TEST_N_STOCKS: int = 50
TEST_N_DAYS: int = 2520  # 10 years
TEST_T: int = 504
TEST_F: int = 2
TEST_K: int = 20  # small K for fast tests
TEST_BATCH_SIZE: int = 32


def set_deterministic(seed: int = TEST_SEED) -> None:
    """
    Set all random seeds for reproducibility.

    :param seed (int): Random seed to use across all generators
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Synthetic returns from a known factor model
# ---------------------------------------------------------------------------

def generate_synthetic_returns(
    n_stocks: int = TEST_N_STOCKS,
    n_days: int = TEST_N_DAYS,
    n_factors: int = 5,
    noise_std: float = 0.02,
    seed: int = TEST_SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate returns from a known factor model for verification.

    r_{i,t} = B_true[i, :] @ z_true[t, :] + ε_{i,t}

    :param n_stocks (int): Number of stocks
    :param n_days (int): Number of trading days
    :param n_factors (int): Number of latent factors
    :param noise_std (float): Idiosyncratic noise standard deviation
    :param seed (int): Random seed

    :return returns_df (pd.DataFrame): Log-returns (n_days, n_stocks)
    :return B_true (np.ndarray): True factor loadings (n_stocks, n_factors)
    :return z_true (np.ndarray): True factor returns (n_days, n_factors)
    """
    rng = np.random.RandomState(seed)

    B_true = rng.randn(n_stocks, n_factors).astype(np.float64)
    z_true = rng.randn(n_days, n_factors).astype(np.float64) * 0.01
    noise = rng.randn(n_days, n_stocks).astype(np.float64) * noise_std

    returns = z_true @ B_true.T + noise

    dates = pd.bdate_range(start="2000-01-03", periods=n_days, freq="B")
    stock_ids = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    returns_df = pd.DataFrame(returns, index=dates, columns=stock_ids)

    return returns_df, B_true, z_true


# ---------------------------------------------------------------------------
# Synthetic pre-z-scored windows with known latent structure
# ---------------------------------------------------------------------------

def generate_synthetic_windows(
    n_windows: int = 1000,
    T: int = TEST_T,
    F: int = TEST_F,
    K_true: int = 10,
    seed: int = TEST_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pre-z-scored windows with known latent structure.

    CONV-02: each window has mean ≈ 0, std ≈ 1 per feature.

    :param n_windows (int): Number of windows to generate
    :param T (int): Window length in trading days
    :param F (int): Number of features per timestep
    :param K_true (int): Number of true latent factors
    :param seed (int): Random seed

    :return windows (torch.Tensor): Shape (N, T, F), z-scored
    :return true_latents (torch.Tensor): Shape (N, K_true)
    """
    rng = np.random.RandomState(seed)

    true_latents = rng.randn(n_windows, K_true).astype(np.float32)

    # Generate time-varying basis functions (shared across windows)
    basis = rng.randn(K_true, T, F).astype(np.float32) * 0.1

    # Construct windows as linear combination of basis + noise
    # windows[i] = sum_k latents[i,k] * basis[k] + noise
    windows_np = np.einsum("nk,ktf->ntf", true_latents, basis)
    windows_np += rng.randn(n_windows, T, F).astype(np.float32) * 0.5

    # Z-score per window, per feature (CONV-02)
    for i in range(n_windows):
        for f in range(F):
            feat = windows_np[i, :, f]
            mu = feat.mean()
            sigma = feat.std()
            if sigma < 1e-8:
                sigma = 1e-8
            windows_np[i, :, f] = (feat - mu) / sigma

    windows = torch.from_numpy(windows_np)
    true_latents_tensor = torch.from_numpy(true_latents)

    return windows, true_latents_tensor


# ---------------------------------------------------------------------------
# Synthetic crisis labels
# ---------------------------------------------------------------------------

def generate_crisis_labels(
    n_windows: int = 1000,
    crisis_fraction: float = 0.20,
    seed: int = TEST_SEED,
) -> torch.Tensor:
    """
    Generate realistic crisis fractions with bimodal distribution.

    ~80% of windows have f_c ≈ 0 (calm), ~20% have f_c ∈ [0.5, 1.0] (crisis).
    VIX is autocorrelated, so crisis windows cluster together.

    :param n_windows (int): Number of windows
    :param crisis_fraction (float): Fraction of windows in crisis
    :param seed (int): Random seed

    :return crisis_fractions (torch.Tensor): Shape (N,) with values in [0, 1]
    """
    rng = np.random.RandomState(seed)

    fractions = np.zeros(n_windows, dtype=np.float32)

    n_crisis = int(n_windows * crisis_fraction)
    crisis_indices = rng.choice(n_windows, size=n_crisis, replace=False)

    # Crisis windows get f_c drawn uniformly from [0.5, 1.0]
    fractions[crisis_indices] = rng.uniform(0.5, 1.0, size=n_crisis).astype(
        np.float32
    )

    # Calm windows get f_c drawn from [0.0, 0.05] (small residual)
    calm_mask = np.ones(n_windows, dtype=bool)
    calm_mask[crisis_indices] = False
    n_calm = calm_mask.sum()
    fractions[calm_mask] = rng.uniform(0.0, 0.05, size=n_calm).astype(np.float32)

    return torch.from_numpy(fractions)


# ---------------------------------------------------------------------------
# Synthetic universe with delistings
# ---------------------------------------------------------------------------

def generate_synthetic_universe(
    n_stocks: int = TEST_N_STOCKS,
    n_days: int = TEST_N_DAYS,
    n_delistings: int = 5,
    seed: int = TEST_SEED,
) -> dict:
    """
    Generate point-in-time universe with realistic delistings.

    :param n_stocks (int): Number of stocks
    :param n_days (int): Number of trading days
    :param n_delistings (int): Number of stocks that get delisted
    :param seed (int): Random seed

    :return result (dict): Keys: prices, market_caps, volumes, exchange_codes,
        share_codes, delisting_dates, delisting_returns, stock_ids, dates
    """
    rng = np.random.RandomState(seed)

    dates = pd.bdate_range(start="2000-01-03", periods=n_days, freq="B")
    stock_ids = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    # GBM prices: P_{t+1} = P_t * exp(mu_i + sigma_i * eps)
    mu = rng.uniform(0.0001, 0.0005, size=n_stocks)
    sigma = rng.uniform(0.005, 0.03, size=n_stocks)
    p0 = rng.uniform(10.0, 200.0, size=n_stocks)

    log_returns = mu[np.newaxis, :] + sigma[np.newaxis, :] * rng.randn(
        n_days, n_stocks
    )
    log_prices = np.log(p0)[np.newaxis, :] + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    # Shares outstanding and market cap
    shares_outstanding = rng.uniform(10e6, 500e6, size=n_stocks)
    market_caps = prices * shares_outstanding[np.newaxis, :]

    # Volume: LogNormal correlated with market cap
    mu_vol = np.log(shares_outstanding * 0.005)
    volumes = np.zeros((n_days, n_stocks), dtype=np.float64)
    for i in range(n_stocks):
        volumes[:, i] = rng.lognormal(mu_vol[i], 1.0, size=n_days)
    volumes = volumes.astype(np.int64)

    # Exchange codes: 60% NYSE (1), 10% AMEX (2), 30% NASDAQ (3)
    exchange_probs = [0.60, 0.10, 0.30]
    exchange_codes = rng.choice([1, 2, 3], size=n_stocks, p=exchange_probs)

    # Share codes: all common equity (10 or 11)
    share_codes = rng.choice([10, 11], size=n_stocks)

    # Delistings in the second half of history
    delisting_dates = {}
    delisting_returns = {}
    if n_delistings > 0:
        delist_stocks = rng.choice(n_stocks, size=n_delistings, replace=False)
        half_point = n_days // 2
        for stock_idx in delist_stocks:
            delist_day = rng.randint(half_point, n_days - 1)
            delisting_dates[stock_ids[stock_idx]] = dates[delist_day]

            # ~50% have NaN delisting return, rest get imputed
            if rng.random() < 0.5:
                delisting_returns[stock_ids[stock_idx]] = np.nan
            else:
                if exchange_codes[stock_idx] in (1, 2):
                    delisting_returns[stock_ids[stock_idx]] = -0.30
                else:
                    delisting_returns[stock_ids[stock_idx]] = -0.55

            # Zero out prices after delisting
            prices[delist_day + 1:, stock_idx] = np.nan
            market_caps[delist_day + 1:, stock_idx] = np.nan
            volumes[delist_day + 1:, stock_idx] = 0

    # Add ~2% missing data (random NaN gaps)
    missing_mask = rng.random((n_days, n_stocks)) < 0.02
    # Don't add NaN to the first day (need P_0)
    missing_mask[0, :] = False
    prices_with_gaps = prices.copy()
    prices_with_gaps[missing_mask] = np.nan

    prices_df = pd.DataFrame(prices_with_gaps, index=dates, columns=stock_ids)
    market_caps_df = pd.DataFrame(market_caps, index=dates, columns=stock_ids)
    volumes_df = pd.DataFrame(volumes, index=dates, columns=stock_ids)

    return {
        "prices": prices_df,
        "market_caps": market_caps_df,
        "volumes": volumes_df,
        "exchange_codes": dict(zip(stock_ids, exchange_codes)),
        "share_codes": dict(zip(stock_ids, share_codes)),
        "delisting_dates": delisting_dates,
        "delisting_returns": delisting_returns,
        "stock_ids": stock_ids,
        "dates": dates,
    }
