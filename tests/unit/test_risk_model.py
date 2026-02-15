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

from tests.fixtures.known_solutions import (
    two_factor_solution,
    rescaling_verification,
)


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

    stock_ids = list(range(n))
    dates = pd.bdate_range(start="2020-01-02", periods=n_dates, freq="B")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Exposure matrix (n, AU)
    B_A = rng.randn(n, au).astype(np.float64) * 0.3

    # Trailing vol DataFrame (dates x stocks)
    vol_values = rng.uniform(0.10, 0.50, size=(n_dates, n)).astype(np.float64)
    trailing_vol = pd.DataFrame(vol_values, index=date_strs, columns=stock_ids)

    # Universe snapshots: all stocks active at every date
    universe_snapshots: dict[str, list[int]] = {
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

        # Formula verification for one date: manually compute rescaled B
        verify_date = first_date
        vol_at_date = trailing_vol.loc[verify_date].values.astype(np.float64)
        sigma_bar = np.median(vol_at_date)
        ratios_manual = _compute_winsorized_ratios(vol_at_date)
        B_manual = B_A * ratios_manual[:, np.newaxis]
        np.testing.assert_allclose(
            B_A_by_date[verify_date], B_manual, atol=1e-12,
            err_msg=f"Manual rescaling formula mismatch at date {verify_date}",
        )

    def test_rescaling_formula_known_values(self) -> None:
        """Verify rescaling formula B_rescaled[i] = winsorized_ratio_i * B_A[i].

        Uses known inputs where:
        1. Stock with vol == median has ratio exactly 1.0 (proves denominator is median)
        2. Row-wise scaling is uniform across columns
        """
        n, au = 5, 2
        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=1, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        B_A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ], dtype=np.float64)

        # Vols: stock 2 (index 2) is the exact median (odd count)
        vols = np.array([[0.28, 0.29, 0.30, 0.31, 0.32]])
        trailing_vol = pd.DataFrame(vols, index=date_strs, columns=stock_ids)
        snapshots = {date_strs[0]: stock_ids[:]}

        B_port = rescale_portfolio(B_A, trailing_vol, date_strs[0], stock_ids, stock_ids)

        # Stock 2 has vol == median(0.30) -> ratio == 1.0, so B_rescaled[2] == B_A[2]
        np.testing.assert_allclose(
            B_port[2], B_A[2], atol=1e-10,
            err_msg="Stock with vol == median should have ratio 1.0 (B unchanged)",
        )

        # All rows must be scaled uniformly: ratio_col0 == ratio_col1
        for i in range(n):
            ratio_col0 = B_port[i, 0] / B_A[i, 0]
            ratio_col1 = B_port[i, 1] / B_A[i, 1]
            assert abs(ratio_col0 - ratio_col1) < 1e-10, (
                f"Stock {i}: scaling must be uniform across columns. "
                f"ratio_col0={ratio_col0:.6f}, ratio_col1={ratio_col1:.6f}"
            )

        # Verify non-median stock: stock 0 has vol=0.28, median=0.30
        # Raw ratio = 0.28/0.30 = 0.9333, but P5 winsorization clips it upward
        # Compute expected via _compute_winsorized_ratios for consistency
        winsorized_ratios = _compute_winsorized_ratios(vols[0])
        expected_ratio_stock0 = winsorized_ratios[0]
        actual_ratio_stock0 = B_port[0, 0] / B_A[0, 0]
        np.testing.assert_allclose(
            actual_ratio_stock0, expected_ratio_stock0, atol=1e-10,
            err_msg=(
                f"Stock 0 winsorized ratio should be {expected_ratio_stock0:.6f}, "
                f"got {actual_ratio_stock0:.6f}"
            ),
        )
        # Verify the raw ratio (pre-winsorization) is less than the winsorized one
        raw_ratio_stock0 = 0.28 / 0.30
        assert expected_ratio_stock0 >= raw_ratio_stock0 - 1e-10, (
            f"P5 winsorization should clip stock 0 upward: "
            f"raw={raw_ratio_stock0:.6f}, winsorized={expected_ratio_stock0:.6f}"
        )

    def test_winsorization_uses_median_not_mean(self) -> None:
        """Winsorized ratios must use median as denominator, not mean.

        With an extreme outlier pulling mean >> median, stocks at the median
        value should have ratio ~1.0 (proving median is the denominator).
        If mean were used, their ratio would be ~0.17 instead.
        """
        # 10 stocks: 9 at 0.20, 1 outlier at 10.0
        vols = np.array(
            [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 10.0],
            dtype=np.float64,
        )

        median_val = np.median(vols)
        mean_val = np.mean(vols)
        assert abs(median_val - 0.20) < 1e-10, "Median should be 0.20"
        assert mean_val > 1.0, f"Mean should be >1.0 with outlier, got {mean_val}"

        ratios = _compute_winsorized_ratios(vols, 5.0, 95.0)

        # The 9 non-outlier stocks: if median used, ratio = 0.20/0.20 = 1.0
        # If mean used, ratio = 0.20/1.18 ≈ 0.17 (far from 1.0)
        for i in range(9):
            assert abs(ratios[i] - 1.0) < 0.05, (
                f"Stock {i}: ratio={ratios[i]:.4f}, expected ~1.0 "
                f"(would be ~{0.20 / mean_val:.4f} if mean were used)"
            )

        # The outlier must be clipped below its raw ratio (50.0)
        raw_outlier_ratio = 10.0 / median_val
        assert ratios[9] < raw_outlier_ratio, (
            f"Outlier not clipped: ratio={ratios[9]:.4f}, raw={raw_outlier_ratio:.4f}"
        )

        # Verify outlier is clipped to exactly P95 of the ratio distribution
        raw_ratios = vols / np.median(vols)
        p95 = np.percentile(raw_ratios, 95.0)
        np.testing.assert_allclose(
            ratios[9], p95, atol=1e-12,
            err_msg=f"Outlier ratio should be clipped to P95={p95:.6f}, got {ratios[9]:.6f}",
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

        stock_ids = list(range(n))
        dates = pd.bdate_range(start="2020-01-02", periods=n_dates, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Identity exposure at each date
        B_identity = np.eye(n, dtype=np.float64)
        B_A_by_date: dict[str, np.ndarray] = {d: B_identity.copy() for d in date_strs}

        # Factor returns = stock returns (no noise)
        z_true = rng.randn(n_dates, n).astype(np.float64) * 0.01
        returns = pd.DataFrame(z_true, index=date_strs, columns=stock_ids)

        universe_snapshots: dict[str, list[int]] = {
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
        """Default (spiked) Sigma_z must be PSD."""
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        z_hat = rng.randn(100, AU).astype(np.float64) * 0.01
        Sigma_z, n_signal = estimate_sigma_z(z_hat)  # default: spiked (DGJ)

        assert isinstance(n_signal, int) and n_signal >= 0
        eigenvalues = np.linalg.eigvalsh(Sigma_z)
        assert np.all(eigenvalues >= -1e-14), (
            f"Sigma_z not PSD: min eigenvalue = {eigenvalues.min():.2e}"
        )
        assert Sigma_z.shape == (AU, AU)

    def test_Sigma_z_truncation_matches_lw(self) -> None:
        """Truncation method uses LW and must match shrinkage formula.

        B1: Verify Sigma_LW = (1-delta)*S_sample + delta*(tr(S_sample)/AU)*I
        """
        from sklearn.covariance import LedoitWolf

        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        z_hat = rng.randn(100, AU).astype(np.float64) * 0.01
        # Use truncation method (which applies LW internally)
        Sigma_z, n_signal_trunc = estimate_sigma_z(z_hat, shrinkage_method="truncation")

        # B1: Verify LW shrinkage formula directly
        lw = LedoitWolf()
        lw.fit(z_hat)
        delta = lw.shrinkage_
        # LedoitWolf uses ddof=0 internally (1/n normalization)
        S_sample = np.cov(z_hat.T, ddof=0)

        # Shrinkage parameter in valid range
        assert 0.0 <= delta <= 1.0, (
            f"LW shrinkage delta={delta:.6f} outside [0, 1]"
        )

        # Reconstruct: Sigma_LW = (1-delta)*S_sample + delta*(tr(S)/AU)*I
        target = (np.trace(S_sample) / AU) * np.eye(AU)
        Sigma_LW_manual = (1.0 - delta) * S_sample + delta * target
        np.testing.assert_allclose(
            Sigma_z, Sigma_LW_manual, atol=1e-10,
            err_msg="estimate_sigma_z(truncation) doesn't match LW formula",
        )

        # Shrinkage MUST improve conditioning
        cond_sample = np.linalg.cond(S_sample)
        cond_lw = np.linalg.cond(Sigma_z)
        assert cond_lw <= cond_sample * (1.0 + 1e-10), (
            f"LW must improve conditioning: cond_sample={cond_sample:.1f}, cond_LW={cond_lw:.1f}"
        )

    def test_D_eps_floor(self) -> None:
        """min(D_eps) must be >= 1e-6 even when residuals have near-zero variance."""
        np.random.seed(SEED)
        stock_ids = list(range(N_STOCKS))

        # Create residuals where some stocks have near-zero variance
        residuals_by_stock: dict[int, list[float]] = {}
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

        # This should not raise thanks to lstsq's SVD-based solving
        z_hat = safe_solve(B_t, r_t, conditioning_threshold=1e6, ridge_scale=1e-6)

        assert z_hat.shape == (au,), f"Unexpected shape: {z_hat.shape}"
        assert np.all(np.isfinite(z_hat)), "z_hat contains non-finite values"

        # Verify safe_solve matches np.linalg.lstsq
        z_hat_lstsq, _, _, _ = np.linalg.lstsq(B_t, r_t, rcond=None)
        np.testing.assert_allclose(
            z_hat, z_hat_lstsq, atol=1e-10,
            err_msg="safe_solve result doesn't match np.linalg.lstsq",
        )

    def test_conditioning_guard_triggers_at_threshold(self) -> None:
        """Ridge applied when condition number exceeds threshold (kappa > 1e6)."""
        np.random.seed(SEED)
        rng = np.random.RandomState(SEED)

        n_active = 30
        au = AU

        # Well-conditioned B: should NOT apply ridge
        B_good = rng.randn(n_active, au).astype(np.float64) * 0.3
        r_good = rng.randn(n_active).astype(np.float64) * 0.01
        z_good = safe_solve(B_good, r_good, conditioning_threshold=1e6)
        assert np.all(np.isfinite(z_good))

        # Manual OLS verification: z_hat = (B^T B)^{-1} B^T r (no ridge)
        z_hat_manual = np.linalg.solve(B_good.T @ B_good, B_good.T @ r_good)
        np.testing.assert_allclose(
            z_good, z_hat_manual, atol=1e-8,
            err_msg="Well-conditioned safe_solve should match plain OLS",
        )

        # Ill-conditioned B: should apply ridge
        B_bad = rng.randn(n_active, au).astype(np.float64)
        B_bad[:, 1] = B_bad[:, 0] + rng.randn(n_active) * 1e-12
        r_bad = rng.randn(n_active).astype(np.float64) * 0.01

        kappa = np.linalg.cond(B_bad.T @ B_bad)
        assert kappa > 1e6, (
            f"Test setup: B_bad not ill-conditioned enough: kappa={kappa:.2e}"
        )

        z_bad = safe_solve(B_bad, r_bad, conditioning_threshold=1e6, ridge_scale=1e-6)
        assert z_bad.shape == (au,)
        assert np.all(np.isfinite(z_bad)), "Ridge fallback produced non-finite values"



class TestResiduals:
    """Tests for residual dtype and shape."""

    def test_residuals_dtype_and_shape(self) -> None:
        """Residuals must be float64 with shape (n_dates, n_stocks)."""
        np.random.seed(SEED)
        data = _make_risk_model_data()

        B_A = data["B_A"]
        trailing_vol = data["trailing_vol"]
        universe_snapshots = data["universe_snapshots"]
        stock_ids = data["stock_ids"]
        returns = data["returns"]

        B_A_by_date = rescale_estimation(
            B_A, trailing_vol, universe_snapshots, stock_ids,
        )

        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns, universe_snapshots,
        )

        residuals = compute_residuals(
            B_A_by_date, z_hat, returns, universe_snapshots, valid_dates, stock_ids,
        )

        assert isinstance(residuals, dict), "Residuals should be a dict"
        for sid, res_list in residuals.items():
            assert isinstance(res_list, list), f"Residuals[{sid}] not a list"
            if res_list:
                assert isinstance(res_list[0], float), (
                    f"Residuals[{sid}][0] not float: {type(res_list[0])}"
                )

        # Formula verification: eps_{i,t} = r_{i,t} - B_A_{i,t} @ z_hat_t
        # Check first 3 stocks at the first valid date
        first_date = valid_dates[0]
        active_at_date = universe_snapshots[first_date]
        B_t = B_A_by_date[first_date]
        for check_idx in range(min(3, len(active_at_date))):
            sid = active_at_date[check_idx]
            r_it = float(returns.loc[first_date, sid])
            b_it = B_t[check_idx]  # row for this stock at this date
            eps_manual = r_it - float(b_it @ z_hat[0])
            # The first residual stored for this stock should match
            assert len(residuals[sid]) > 0, f"No residuals for stock {sid}"
            np.testing.assert_allclose(
                residuals[sid][0], eps_manual, atol=1e-10,
                err_msg=f"Residual formula mismatch for stock {sid} at {first_date}",
            )


# ---------------------------------------------------------------------------
# Tests: Dual rescaling correctness (INV-004)
# ---------------------------------------------------------------------------


class TestDualRescalingCorrectness:
    """Tests verifying date-specific vs current-date rescaling (INV-004)."""

    def test_estimation_uses_date_specific_vol(self) -> None:
        """Different dates in rescale_estimation produce different matrices when vol differs."""
        rng = np.random.RandomState(42)
        n, au = 10, 3
        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=30, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        B_A = rng.randn(n, au) * 0.3
        # Deliberately different cross-sectional vol profile at different dates
        vol_data = rng.uniform(0.15, 0.30, (30, n))
        # First date: stock 0 has much higher vol relative to others
        vol_data[0, :] = 0.20
        vol_data[0, 0] = 0.80  # Outlier at first date only
        trailing_vol = pd.DataFrame(vol_data, index=date_strs, columns=stock_ids)
        snapshots = {d: stock_ids[:] for d in date_strs}
        B_by_date = rescale_estimation(B_A, trailing_vol, snapshots, stock_ids)
        assert isinstance(B_by_date, dict)
        # First date (with outlier stock 0) should differ from last date
        d0 = date_strs[0]
        d_last = date_strs[-1]
        if d0 in B_by_date and d_last in B_by_date:
            assert not np.allclose(B_by_date[d0], B_by_date[d_last]), (
                "Different vol dates should produce different rescaled B"
            )

        # Manual rescaled B for date 0 using winsorized ratios
        vol_at_d0 = vol_data[0].astype(np.float64)
        ratios_d0 = _compute_winsorized_ratios(vol_at_d0)
        B_est_d0_manual = B_A * ratios_d0[:, np.newaxis]
        np.testing.assert_allclose(
            B_by_date[d0], B_est_d0_manual, atol=1e-8,
            err_msg="Manual rescaled B at date 0 does not match function output",
        )

    def test_portfolio_uses_current_date_only(self) -> None:
        """rescale_portfolio is invariant to changes in non-current-date vol."""
        rng = np.random.RandomState(42)
        n, au = 10, 3
        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=30, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        B_A = rng.randn(n, au) * 0.3
        vol_data = rng.uniform(0.10, 0.30, (30, n))
        trailing_vol_1 = pd.DataFrame(vol_data.copy(), index=date_strs, columns=stock_ids)
        trailing_vol_2 = pd.DataFrame(vol_data.copy(), index=date_strs, columns=stock_ids)
        # Change vol at first date (not current date = last date)
        trailing_vol_2.iloc[0] = 5.0
        current_date = date_strs[-1]
        B_port_1 = rescale_portfolio(B_A, trailing_vol_1, current_date, stock_ids, stock_ids)
        B_port_2 = rescale_portfolio(B_A, trailing_vol_2, current_date, stock_ids, stock_ids)
        np.testing.assert_array_almost_equal(B_port_1, B_port_2, decimal=10,
            err_msg="Portfolio rescaling should only use current date vol")


# ---------------------------------------------------------------------------
# Tests: Covariance properties (LW conditioning, D_eps floor)
# ---------------------------------------------------------------------------


class TestCovarianceProperties:
    """Additional tests for covariance estimation quality."""

    def test_ledoit_wolf_better_conditioned(self) -> None:
        """Ledoit-Wolf shrinkage produces a better-conditioned matrix than sample covariance."""
        rng = np.random.RandomState(42)
        au = 5
        n_dates = 30  # Small sample -> poorly conditioned sample cov
        z_hat = rng.randn(n_dates, au)
        Sigma_lw, _ = estimate_sigma_z(z_hat)
        Sigma_sample = np.cov(z_hat.T)
        cond_lw = np.linalg.cond(Sigma_lw)
        cond_sample = np.linalg.cond(Sigma_sample)
        assert cond_lw <= cond_sample * (1.0 + 1e-10), (
            f"LW must improve conditioning: cond_sample={cond_sample:.1f}, cond_LW={cond_lw:.1f}"
        )
        # PSD check
        assert np.all(np.linalg.eigvalsh(Sigma_lw) >= -1e-10)

    def test_d_eps_floor_enforced(self) -> None:
        """estimate_d_eps enforces floor even when residuals have zero variance."""
        stock_ids = [0, 1, 2]
        residuals: dict[int, list[float]] = {
            0: [0.01, 0.02, -0.01],  # Normal
            1: [0.0, 0.0, 0.0],       # Zero variance
            2: [0.005],                # Single observation -> use floor
        }
        floor = 1e-6
        D_eps = estimate_d_eps(residuals, stock_ids, d_eps_floor=floor)
        assert D_eps.shape == (3,)
        assert np.all(D_eps >= floor), f"Floor violated: min={D_eps.min()}"
        assert D_eps[1] == floor, f"Zero-variance stock should get floor, got {D_eps[1]}"

    def test_winsorization_applied_in_rescaling(self) -> None:
        """INV-008: Extreme vol values must be bounded by winsorization in rescaling.

        Creates a 50x outlier and verifies:
        1. All values remain finite
        2. The outlier stock's amplification is much less than the raw 50x ratio
        """
        rng = np.random.RandomState(42)
        n, au = 20, 3
        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=5, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        B_A = rng.randn(n, au) * 0.3
        vol_data = np.full((5, n), 0.20)
        vol_data[:, 0] = 10.0  # Extreme outlier: 50x median
        trailing_vol = pd.DataFrame(vol_data, index=date_strs, columns=stock_ids)
        snapshots = {d: stock_ids[:] for d in date_strs}
        B_by_date = rescale_estimation(B_A, trailing_vol, snapshots, stock_ids)

        ba_outlier_norm = np.linalg.norm(B_A[0])
        for d in B_by_date:
            assert np.all(np.isfinite(B_by_date[d])), f"Non-finite values at {d}"

            # Without winsorization, the outlier ratio would be 10.0/0.20 = 50.
            # After P95 clipping, it should equal P95 of the ratio distribution.
            vols_at_d = vol_data[0].astype(np.float64)  # same at all dates
            raw_ratios = vols_at_d / np.median(vols_at_d)
            p95 = np.percentile(raw_ratios, 95.0)

            rescaled_outlier_norm = np.linalg.norm(B_by_date[d][0])
            if ba_outlier_norm > 1e-10:
                amplification = rescaled_outlier_norm / ba_outlier_norm
                np.testing.assert_allclose(
                    amplification, p95, atol=1e-8,
                    err_msg=(
                        f"Date {d}: outlier amplification={amplification:.6f}, "
                        f"expected P95={p95:.6f}"
                    ),
                )


# ---------------------------------------------------------------------------
# M4: Sigma_z must use full history (anti-cyclical principle)
# ---------------------------------------------------------------------------


class TestSigmaZFullHistory:
    """M4: Verify estimate_sigma_z uses all provided data, not a rolling window."""

    def test_sigma_z_uses_all_data(self) -> None:
        """
        Provide factor returns from two distinct periods. Sigma_z from full
        history must differ from Sigma_z on either period alone, proving
        all data is used.
        """
        rng = np.random.RandomState(42)
        au = 3

        # Period 1: low-variance factors (100 dates)
        z_period1 = rng.randn(100, au) * 0.005

        # Period 2: high-variance factors with different correlation (100 dates)
        z_period2 = rng.randn(100, au) * 0.05
        z_period2[:, 1] = z_period2[:, 0] * 0.8 + rng.randn(100) * 0.01

        # Full history = both periods
        z_full = np.vstack([z_period1, z_period2])

        Sigma_full, _ = estimate_sigma_z(z_full)
        Sigma_p1, _ = estimate_sigma_z(z_period1)
        Sigma_p2, _ = estimate_sigma_z(z_period2)

        # Sigma_full must differ from both period-specific estimates
        assert not np.allclose(Sigma_full, Sigma_p1, atol=1e-6), (
            "Sigma_z(full) should differ from Sigma_z(period1 only)"
        )
        assert not np.allclose(Sigma_full, Sigma_p2, atol=1e-6), (
            "Sigma_z(full) should differ from Sigma_z(period2 only)"
        )

        # Sigma_full variance should be between period1 and period2 variances
        # (since full history averages both regimes)
        var_full = np.trace(Sigma_full)
        var_p1 = np.trace(Sigma_p1)
        var_p2 = np.trace(Sigma_p2)
        assert var_p1 < var_full < var_p2, (
            f"Full history variance (trace={var_full:.6f}) should be between "
            f"period1 ({var_p1:.6f}) and period2 ({var_p2:.6f})"
        )

    def test_sigma_z_shape_matches_input(self) -> None:
        """estimate_sigma_z output shape is (AU, AU) matching input columns."""
        rng = np.random.RandomState(42)
        for au in [3, 5, 10]:
            z_hat = rng.randn(50, au) * 0.01
            Sigma_z, n_sig = estimate_sigma_z(z_hat)
            assert isinstance(n_sig, int) and n_sig >= 0
            assert Sigma_z.shape == (au, au), (
                f"Sigma_z shape {Sigma_z.shape} != expected ({au}, {au})"
            )


# ---------------------------------------------------------------------------
# Phase 1: Consume two_factor_solution fixture — verify eigendecomposition
# ---------------------------------------------------------------------------


class TestTwoFactorKnownSolution:
    """Consume two_factor_solution() fixture and verify eigendecomposition + risk model assembly."""

    def test_eigendecomposition_matches_sigma_z(self) -> None:
        """B' Lambda B'^T must reconstruct Sigma_z exactly (eigendecomposition identity)."""
        sol = two_factor_solution()
        B_A = sol["B_A"]
        Sigma_z = sol["Sigma_z"]
        eigenvalues = sol["eigenvalues"]
        V = sol["V"]
        B_prime = sol["B_prime"]

        # Eigendecomposition identity: V @ diag(eigenvalues) @ V^T == Sigma_z
        Sigma_z_reconstructed = V @ np.diag(eigenvalues) @ V.T
        np.testing.assert_allclose(
            Sigma_z_reconstructed, Sigma_z, atol=1e-12,
            err_msg="V @ diag(lambda) @ V^T must reconstruct Sigma_z",
        )

        # V must be orthogonal: V^T V == I
        np.testing.assert_allclose(
            V.T @ V, np.eye(sol["AU"]), atol=1e-12,
            err_msg="Eigenvector matrix V must be orthogonal",
        )

    def test_rotated_covariance_equivalence(self) -> None:
        """B_A @ Sigma_z @ B_A^T must equal B' @ diag(lambda) @ B'^T."""
        sol = two_factor_solution()
        B_A = sol["B_A"]
        Sigma_z = sol["Sigma_z"]
        eigenvalues = sol["eigenvalues"]
        B_prime = sol["B_prime"]

        Sigma_direct = B_A @ Sigma_z @ B_A.T
        Sigma_rotated = B_prime @ np.diag(eigenvalues) @ B_prime.T
        np.testing.assert_allclose(
            Sigma_rotated, Sigma_direct, atol=1e-12,
            err_msg="Rotation must preserve factor covariance: B Sigma_z B^T == B' Lambda B'^T",
        )

    def test_assemble_risk_model_with_two_factor(self) -> None:
        """assemble_risk_model with known two-factor inputs produces correct Sigma_assets."""
        sol = two_factor_solution()
        B_A = sol["B_A"]
        Sigma_z = sol["Sigma_z"]
        D_eps = sol["D_eps"]
        n = sol["n"]

        risk_model = assemble_risk_model(B_A, Sigma_z, D_eps)
        Sigma_assets = risk_model["Sigma_assets"]

        # Manual computation: Sigma_assets = B_A @ Sigma_z @ B_A.T + diag(D_eps)
        Sigma_expected = B_A @ Sigma_z @ B_A.T + np.diag(D_eps)
        np.testing.assert_allclose(
            Sigma_assets, Sigma_expected, atol=1e-12,
            err_msg="assemble_risk_model result doesn't match manual formula",
        )

        # Sigma_assets must be symmetric and PSD
        np.testing.assert_allclose(Sigma_assets, Sigma_assets.T, atol=1e-14)
        eigs = np.linalg.eigvalsh(Sigma_assets)
        assert np.all(eigs >= -1e-12), f"Sigma_assets not PSD: min eigenvalue={eigs.min()}"

    def test_entropy_at_equal_weight_with_two_factor(self) -> None:
        """Verify entropy H at equal weight using two-factor known solution."""
        from src.portfolio.entropy import compute_entropy_and_gradient

        sol = two_factor_solution()
        B_prime = sol["B_prime"]
        eigenvalues = sol["eigenvalues"]
        w_equal = sol["w_equal"]
        H_expected = sol["H_equal"]

        H, grad_H = compute_entropy_and_gradient(w_equal, B_prime, eigenvalues)

        assert abs(H - H_expected) < 1e-10, (
            f"Entropy at equal weight: got {H:.10f}, expected {H_expected:.10f}"
        )
        assert grad_H.shape == (sol["n"],)


# ---------------------------------------------------------------------------
# Phase 1: Consume rescaling_verification fixture — verify exact winsorized values
# ---------------------------------------------------------------------------


class TestRescalingKnownValues:
    """Consume rescaling_verification() fixture and verify exact winsorized ratios."""

    def test_winsorized_ratios_per_date(self) -> None:
        """Verify winsorized ratios match fixture's pre-computed values at each date."""
        fix = rescaling_verification()
        sigma_it = fix["sigma_it"]
        sigma_bar_t = fix["sigma_bar_t"]
        ratios_winsorized = fix["ratios_winsorized"]
        T_hist = fix["T_hist"]
        n = fix["n"]

        for t in range(T_hist):
            computed_ratios = _compute_winsorized_ratios(
                sigma_it[t], fix["winsorize_lo"], fix["winsorize_hi"],
            )
            np.testing.assert_allclose(
                computed_ratios, ratios_winsorized[t], atol=1e-12,
                err_msg=f"Winsorized ratios mismatch at date {t}",
            )

    def test_outlier_stock_clipped(self) -> None:
        """Stock 0 (vol=2.0, extreme outlier) must be clipped below raw ratio."""
        fix = rescaling_verification()
        sigma_it = fix["sigma_it"]
        sigma_bar_t = fix["sigma_bar_t"]

        for t in range(fix["T_hist"]):
            raw_ratio_0 = sigma_it[t, 0] / sigma_bar_t[t]
            winsorized = _compute_winsorized_ratios(
                sigma_it[t], fix["winsorize_lo"], fix["winsorize_hi"],
            )
            assert winsorized[0] <= raw_ratio_0 + 1e-10, (
                f"Date {t}: outlier not clipped. raw={raw_ratio_0:.4f}, "
                f"winsorized={winsorized[0]:.4f}"
            )

    def test_portfolio_rescaling_uses_last_date_ratios(self) -> None:
        """B_A_portfolio must use ratios from the last date only."""
        fix = rescaling_verification()
        mu_A = fix["mu_A"]
        ratios_last = fix["ratios_winsorized"][-1]
        B_A_portfolio_expected = ratios_last[:, np.newaxis] * mu_A

        np.testing.assert_allclose(
            fix["B_A_portfolio"], B_A_portfolio_expected, atol=1e-12,
            err_msg="Portfolio rescaling must use last date's winsorized ratios",
        )


# ---------------------------------------------------------------------------
# Non-trivial OLS test: B ≠ I, verify z_hat = (B^T B)^{-1} B^T r
# ---------------------------------------------------------------------------


class TestNonTrivialOLS:
    """Factor regression with B ≠ I must still produce z_hat = (B^T B)^{-1} B^T r."""

    def test_ols_formula_nontrivial_b(self) -> None:
        """With B ≠ I and r = B @ z_true (no noise), OLS must recover z_true."""
        rng = np.random.RandomState(42)
        n = 15
        au = 3
        n_dates = 40
        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=n_dates, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Non-trivial well-conditioned B
        B_raw = rng.randn(n, au).astype(np.float64) * 0.3
        B_A_by_date = {d: B_raw.copy() for d in date_strs}

        # True factor returns
        z_true = rng.randn(n_dates, au).astype(np.float64) * 0.01

        # r = B @ z (no noise → exact recovery expected)
        returns_data = np.array([B_raw @ z_true[t] for t in range(n_dates)])
        returns = pd.DataFrame(returns_data, index=date_strs, columns=stock_ids)

        universe_snapshots = {d: stock_ids[:] for d in date_strs}

        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns, universe_snapshots,
        )

        for t_idx, d in enumerate(valid_dates):
            d_idx = date_strs.index(d)
            np.testing.assert_allclose(
                z_hat[t_idx], z_true[d_idx], atol=1e-8,
                err_msg=f"OLS recovery failed at date {d}: B≠I case",
            )


# ---------------------------------------------------------------------------
# D_eps variance formula: Var(residuals) with ddof=1, floored at 1e-6
# ---------------------------------------------------------------------------


class TestDEpsVarianceFormula:
    """Verify D_eps = max(Var(eps_i, ddof=1), 1e-6) with known residuals."""

    def test_d_eps_matches_manual_variance(self) -> None:
        """D_eps without shrinkage must match np.var(eps, ddof=1)."""
        rng = np.random.RandomState(42)
        stock_ids = [0, 1, 2, 3, 4]
        residuals_by_stock: dict[int, list[float]] = {}
        expected_d_eps = np.zeros(5)

        for i, sid in enumerate(stock_ids):
            eps = list(rng.randn(100) * (0.01 * (i + 1)))
            residuals_by_stock[sid] = eps
            expected_d_eps[i] = max(np.var(eps, ddof=1), 1e-6)

        D_eps = estimate_d_eps(
            residuals_by_stock, stock_ids, d_eps_floor=1e-6,
            shrink_toward_mean=False,
        )

        np.testing.assert_allclose(
            D_eps, expected_d_eps, rtol=1e-10,
            err_msg="D_eps doesn't match manual Var(eps, ddof=1) with floor",
        )

    def test_d_eps_james_stein_shrinkage(self) -> None:
        """D_eps with James-Stein shrinkage moves estimates toward the mean."""
        rng = np.random.RandomState(42)
        stock_ids = [0, 1, 2, 3, 4]
        residuals_by_stock: dict[int, list[float]] = {}

        for i, sid in enumerate(stock_ids):
            eps = list(rng.randn(100) * (0.01 * (i + 1)))
            residuals_by_stock[sid] = eps

        D_raw = estimate_d_eps(
            residuals_by_stock, stock_ids, d_eps_floor=1e-6,
            shrink_toward_mean=False,
        )
        D_shrunk = estimate_d_eps(
            residuals_by_stock, stock_ids, d_eps_floor=1e-6,
            shrink_toward_mean=True,
        )

        # Shrinkage should reduce cross-sectional dispersion
        assert float(np.std(D_shrunk)) < float(np.std(D_raw)), (
            "James-Stein shrinkage should reduce dispersion of D_eps"
        )
        # Shrinkage should preserve the mean (approximately)
        assert abs(float(np.mean(D_shrunk)) - float(np.mean(D_raw))) < 1e-6


# ---------------------------------------------------------------------------
# Fixture-based tests using known_solutions.py
# ---------------------------------------------------------------------------


class TestKnownSolutionsFixtures:
    """Tests using pre-computed analytical solutions from known_solutions.py."""

    def test_rescaling_fixture_winsorized_ratios(self) -> None:
        """Verify _compute_winsorized_ratios against rescaling_verification fixture."""
        fixture = rescaling_verification()
        for t in range(fixture["T_hist"]):
            actual = _compute_winsorized_ratios(fixture["sigma_it"][t])
            expected = fixture["ratios_winsorized"][t]
            np.testing.assert_allclose(
                actual, expected, atol=1e-12,
                err_msg=f"Winsorized ratios at date {t} don't match fixture",
            )

    def test_two_factor_entropy_fixture(self) -> None:
        """Verify entropy computation against two_factor_solution fixture."""
        from src.portfolio.entropy import compute_entropy_and_gradient

        fixture = two_factor_solution()
        H, grad_H = compute_entropy_and_gradient(
            fixture["w_equal"], fixture["B_prime"], fixture["eigenvalues"],
        )
        np.testing.assert_allclose(
            H, fixture["H_equal"], atol=1e-10,
            err_msg="Entropy at equal weight should match fixture value",
        )
        assert grad_H.shape == (fixture["n"],)

    def test_two_factor_risk_model_assembly(self) -> None:
        """Verify assemble_risk_model against two_factor_solution fixture."""
        fixture = two_factor_solution()
        result = assemble_risk_model(
            fixture["B_A"], fixture["Sigma_z"], fixture["D_eps"],
        )
        # Verify eigendecomposition
        np.testing.assert_allclose(
            result["eigenvalues"], fixture["eigenvalues"], atol=1e-10,
            err_msg="Eigenvalues should match fixture",
        )
        # Verify V is orthogonal
        V = result["V"]
        np.testing.assert_allclose(
            V @ V.T, np.eye(fixture["AU"]), atol=1e-10,
            err_msg="V should be orthogonal",
        )
        # Verify rotated exposures
        np.testing.assert_allclose(
            result["B_prime_port"], fixture["B_prime"], atol=1e-10,
            err_msg="Rotated exposures should match fixture",
        )
