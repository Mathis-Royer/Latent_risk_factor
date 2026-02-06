"""
Unit tests for the risk model module (MOD-007).

Covers:
- Dual rescaling (estimation vs portfolio)
- Winsorization bounds
- Cross-sectional OLS factor regression
- Factor covariance estimation (Ledoit-Wolf)
- Idiosyncratic variance with floor
- Full covariance reconstruction
- Conditioning guard
- Rotation invariance of assembled risk model

Reference: ISD Section MOD-007.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk_model.rescaling import (
    _compute_winsorized_ratios,
    rescale_estimation,
    rescale_portfolio,
)
from src.risk_model.factor_regression import estimate_factor_returns, compute_residuals
from src.risk_model.covariance import estimate_sigma_z, estimate_d_eps, assemble_risk_model
from src.risk_model.conditioning import safe_solve


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SEED = 42
N_STOCKS = 20
AU = 5


# ---------------------------------------------------------------------------
# Helper: generate valid risk-model test data
# ---------------------------------------------------------------------------

def _make_risk_model_data(
    n: int = N_STOCKS,
    au: int = AU,
    n_dates: int = 60,
    seed: int = SEED,
) -> dict:
    """
    Build deterministic inputs for risk-model tests.

    :param n (int): Number of stocks
    :param au (int): Number of active units
    :param n_dates (int): Number of historical dates
    :param seed (int): Random seed

    :return data (dict): B_A, trailing_vol, universe_snapshots, stock_ids,
        returns, dates
    """
    rng = np.random.RandomState(seed)

    stock_ids = [f"S{i:03d}" for i in range(n)]
    dates = pd.bdate_range(start="2020-01-02", periods=n_dates, freq="B")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Exposure matrix (n, AU)
    B_A = rng.randn(n, au).astype(np.float64) * 0.3

    # Trailing vol DataFrame (dates x stocks)
    vol_values = rng.uniform(0.10, 0.50, size=(n_dates, n)).astype(np.float64)
    trailing_vol = pd.DataFrame(vol_values, index=date_strs, columns=stock_ids)

    # Universe snapshots: all stocks active at every date
    universe_snapshots: dict[str, list[str]] = {
        d: stock_ids[:] for d in date_strs
    }

    # Returns DataFrame (dates x stocks) ~ small Gaussian
    ret_values = rng.randn(n_dates, n).astype(np.float64) * 0.01
    returns = pd.DataFrame(ret_values, index=date_strs, columns=stock_ids)

    return {
        "B_A": B_A,
        "trailing_vol": trailing_vol,
        "universe_snapshots": universe_snapshots,
        "stock_ids": stock_ids,
        "returns": returns,
        "dates": date_strs,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRescaling:
    """Tests for dual rescaling (MOD-007 sub-task 1)."""

    def test_rescaling_dual(self) -> None:
        """Estimation and portfolio rescaling must produce different results.

        Estimation rescaling returns per-date matrices using historical vols,
        while portfolio rescaling returns a single matrix at the current date.
        The two should differ because cross-sectional vol distributions vary
        over time.
        """
        np.random.seed(SEED)
        data = _make_risk_model_data()

        B_A = data["B_A"]
        trailing_vol = data["trailing_vol"]
        universe_snapshots = data["universe_snapshots"]
        stock_ids = data["stock_ids"]
        date_strs = data["dates"]

        # Estimation rescaling (per-date)
        B_A_by_date = rescale_estimation(
            B_A, trailing_vol, universe_snapshots, stock_ids,
        )

        # Portfolio rescaling (current date = last date)
        current_date = date_strs[-1]
        B_A_port = rescale_portfolio(
            B_A, trailing_vol, current_date, stock_ids, stock_ids,
        )

        # Both should produce valid arrays
        assert len(B_A_by_date) > 0, "Estimation rescaling produced no dates"
        assert B_A_port.shape == (N_STOCKS, AU)

        # The estimation matrix at the last date should generally differ from
        # the portfolio matrix because estimation and portfolio rescaling
        # use different code paths (but with the same date the values can match).
        # Instead, verify a non-last date differs from portfolio rescaling.
        first_date = date_strs[0]
        B_A_first = B_A_by_date[first_date]
        assert not np.allclose(B_A_first, B_A_port), (
            "Estimation (first date) and portfolio (last date) rescaling "
            "should produce different results"
        )

    def test_winsorization(self) -> None:
        """Winsorized ratios must be bounded to [P5, P95]."""
        np.random.seed(SEED)

        # Cross-section with one extreme outlier
        vol_cross_section = np.array(
            [0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.23, 0.17, 5.0, 0.001],
            dtype=np.float64,
        )

        ratios = _compute_winsorized_ratios(vol_cross_section, 5.0, 95.0)

        # The raw ratio for the extreme values would be far outside [P5, P95]
        lo = np.percentile(vol_cross_section / np.median(vol_cross_section), 5.0)
        hi = np.percentile(vol_cross_section / np.median(vol_cross_section), 95.0)

        assert np.all(ratios >= lo - 1e-10), (
            f"Ratios below P5 bound: min ratio={ratios.min():.6f}, lo={lo:.6f}"
        )
        assert np.all(ratios <= hi + 1e-10), (
            f"Ratios above P95 bound: max ratio={ratios.max():.6f}, hi={hi:.6f}"
        )


class TestFactorRegression:
    """Tests for cross-sectional OLS (MOD-007 sub-task 2)."""

    def test_factor_regression_identity(self) -> None:
        """For B_A = I and r = z (no noise), z_hat should approximate z.

        When the exposure matrix is the identity and returns equal the factor
        returns, the OLS solution z_hat = (I^T I)^{-1} I^T r = r exactly.
        """
        np.random.seed(SEED)

        n = 10
        n_dates = 50
        rng = np.random.RandomState(SEED)

        stock_ids = [f"S{i:03d}" for i in range(n)]
        dates = pd.bdate_range(start="2020-01-02", periods=n_dates, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Identity exposure at each date
        B_identity = np.eye(n, dtype=np.float64)
        B_A_by_date: dict[str, np.ndarray] = {d: B_identity.copy() for d in date_strs}

        # Factor returns = stock returns (no noise)
        z_true = rng.randn(n_dates, n).astype(np.float64) * 0.01
        returns = pd.DataFrame(z_true, index=date_strs, columns=stock_ids)

        universe_snapshots: dict[str, list[str]] = {
            d: stock_ids[:] for d in date_strs
        }

        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns, universe_snapshots,
        )

        assert z_hat.shape[0] == len(valid_dates)
        assert z_hat.shape[0] > 0, "No dates produced"

        # Match dates and compare
        for t_idx, d in enumerate(valid_dates):
            d_idx = date_strs.index(d)
            np.testing.assert_allclose(
                z_hat[t_idx], z_true[d_idx], atol=1e-10,
                err_msg=f"z_hat != z_true at date {d}",
            )


class TestCovariance:
    """Tests for covariance estimation (MOD-007 sub-tasks 3-4)."""

    def test_Sigma_z_psd(self) -> None:
        """Ledoit-Wolf Sigma_z must be positive semi-definite."""
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        z_hat = rng.randn(100, AU).astype(np.float64) * 0.01
        Sigma_z = estimate_sigma_z(z_hat)

        eigenvalues = np.linalg.eigvalsh(Sigma_z)
        assert np.all(eigenvalues >= -1e-12), (
            f"Sigma_z not PSD: min eigenvalue = {eigenvalues.min():.2e}"
        )

    def test_D_eps_floor(self) -> None:
        """min(D_eps) must be >= 1e-6 even when residuals have near-zero variance."""
        np.random.seed(SEED)
        stock_ids = [f"S{i:03d}" for i in range(N_STOCKS)]

        # Create residuals where some stocks have near-zero variance
        residuals_by_stock: dict[str, list[float]] = {}
        for i, sid in enumerate(stock_ids):
            if i < 5:
                # Near-zero variance: all residuals the same
                residuals_by_stock[sid] = [1e-10] * 50
            else:
                residuals_by_stock[sid] = list(
                    np.random.RandomState(SEED + i).randn(50) * 0.01
                )

        D_eps = estimate_d_eps(residuals_by_stock, stock_ids, d_eps_floor=1e-6)

        assert D_eps.shape == (N_STOCKS,)
        assert np.all(D_eps >= 1e-6), (
            f"D_eps floor violated: min = {D_eps.min():.2e}"
        )

    def test_covariance_reconstruction(self) -> None:
        """Sigma_assets must equal B_A_port @ Sigma_z @ B_A_port.T + diag(D_eps)."""
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        n = N_STOCKS
        au = AU

        B_A_port = rng.randn(n, au).astype(np.float64) * 0.3

        # Build a valid PSD Sigma_z
        raw = rng.randn(au, au).astype(np.float64) * 0.01
        Sigma_z = raw @ raw.T + np.eye(au) * 0.001

        D_eps = rng.uniform(0.001, 0.01, size=n).astype(np.float64)

        risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
        Sigma_assets = risk_model["Sigma_assets"]

        # Reconstruct manually
        Sigma_expected = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)

        np.testing.assert_allclose(
            Sigma_assets, Sigma_expected, atol=1e-12,
            err_msg="Sigma_assets != B @ Sigma_z @ B.T + diag(D_eps)",
        )

    def test_rotation_preserves_covariance(self) -> None:
        """Sigma_assets is the same whether computed via rotated or direct basis.

        B' = B V, Lambda = diag(eigenvalues).
        B' Lambda B'^T = B V Lambda V^T B^T = B Sigma_z B^T.
        So Sigma_assets must be identical both ways.
        """
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        n = N_STOCKS
        au = AU

        B_A_port = rng.randn(n, au).astype(np.float64) * 0.3

        # Build a valid PSD Sigma_z
        raw = rng.randn(au, au).astype(np.float64) * 0.01
        Sigma_z = raw @ raw.T + np.eye(au) * 0.001

        D_eps = rng.uniform(0.001, 0.01, size=n).astype(np.float64)

        risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
        eigenvalues = risk_model["eigenvalues"]
        B_prime = risk_model["B_prime_port"]

        # Via rotated basis: B' diag(lambda) B'^T + diag(D_eps)
        Sigma_rotated = B_prime @ np.diag(eigenvalues) @ B_prime.T + np.diag(D_eps)

        # Via direct: B Sigma_z B^T + diag(D_eps)
        Sigma_direct = risk_model["Sigma_assets"]

        np.testing.assert_allclose(
            Sigma_rotated, Sigma_direct, atol=1e-10,
            err_msg="Rotation does not preserve Sigma_assets",
        )


class TestConditioning:
    """Tests for conditioning guard (MOD-007 sub-task 2)."""

    def test_conditioning_guard(self) -> None:
        """safe_solve must not crash on a nearly singular B matrix.

        When columns of B are nearly collinear, the condition number of
        B^T B exceeds the threshold and ridge regularization is applied.
        """
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        n_active = 30
        au = AU

        # Build a nearly singular B: first two columns are almost identical
        B_t = rng.randn(n_active, au).astype(np.float64)
        B_t[:, 1] = B_t[:, 0] + rng.randn(n_active) * 1e-10  # near collinear

        r_t = rng.randn(n_active).astype(np.float64) * 0.01

        # This should not raise thanks to the ridge fallback
        z_hat = safe_solve(B_t, r_t, conditioning_threshold=1e6, ridge_scale=1e-6)

        assert z_hat.shape == (au,), f"Unexpected shape: {z_hat.shape}"
        assert np.all(np.isfinite(z_hat)), "z_hat contains non-finite values"
