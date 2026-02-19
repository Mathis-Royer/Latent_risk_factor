"""
Edge case and defensive tests for pipeline robustness.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.windowing import create_windows
from src.data_pipeline.features import compute_rolling_realized_vol
from src.vae.build_vae import build_vae
from src.inference.active_units import measure_active_units, filter_exposure_matrix
from src.risk_model.rescaling import rescale_estimation
from src.risk_model.covariance import estimate_sigma_z, assemble_risk_model
from src.portfolio.entropy import compute_entropy_and_gradient
from src.portfolio.sca_solver import multi_start_optimize


# ---------------------------------------------------------------------------
# 1. Single-stock portfolio
# ---------------------------------------------------------------------------


class TestSingleStockPortfolio:
    """With n=1, the optimizer should return w=[1.0] or handle gracefully."""

    def test_single_stock_portfolio(self) -> None:
        """multi_start_optimize with n=1 returns w=[1.0]."""
        rng = np.random.RandomState(42)
        n = 1
        AU = 2

        B_prime = rng.randn(n, AU)
        eigenvalues = np.array([0.5, 0.3])
        Sigma_assets = np.array([[0.04]])  # 20% vol, single stock
        D_eps = np.array([0.01])

        w_best, f_best, H_best, _ = multi_start_optimize(
            Sigma_assets=Sigma_assets,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=1.0,
            n_starts=1,
            seed=42,
            w_max=1.0,
            w_bar=1.0,
            is_first=True,
        )

        # With a single stock, the only feasible solution is w=[1.0]
        np.testing.assert_allclose(
            w_best, np.array([1.0]), atol=1e-4,
            err_msg="Single stock portfolio should be w=[1.0]",
        )


# ---------------------------------------------------------------------------
# 2. AU=0 fallback
# ---------------------------------------------------------------------------


class TestAUZeroFallback:
    """Untrained VAE may yield AU=0; filter_exposure_matrix must handle it."""

    def test_au_zero_fallback(self) -> None:
        """
        With an untrained VAE, AU could be 0. active_dims should be empty.
        filter_exposure_matrix with empty dims should return shape (n, 0).
        """
        K = 8
        T = 64

        model, info = build_vae(
            n=25, T=T, T_annee=5, F=2, K=K, r_max=200.0, c_min=144,
        )

        # Use very small windows to keep test fast
        windows = torch.randn(5, T, 2)

        AU, kl_per_dim, active_dims = measure_active_units(
            model, windows, batch_size=5, au_threshold=0.01,
        )

        if AU == 0:
            # active_dims should be empty
            assert active_dims == [], (
                f"Expected empty active_dims when AU=0, got {active_dims}"
            )

            # filter_exposure_matrix with empty dims
            rng = np.random.RandomState(42)
            B = rng.randn(10, K)
            B_A = filter_exposure_matrix(B, active_dims)
            assert B_A.shape == (10, 0), (
                f"Expected shape (10, 0), got {B_A.shape}"
            )
        else:
            # Even if AU > 0 for this random init, verify filter works correctly
            rng = np.random.RandomState(42)
            B = rng.randn(10, K)

            # Force empty active_dims scenario
            B_empty = filter_exposure_matrix(B, [])
            assert B_empty.shape == (10, 0), (
                f"Expected shape (10, 0) for empty dims, got {B_empty.shape}"
            )


# ---------------------------------------------------------------------------
# 3. All-NaN returns excluded
# ---------------------------------------------------------------------------


class TestAllNaNReturnsExcluded:
    """Stock with all-NaN returns should produce no windows and no NaN output."""

    def test_all_nan_returns_excluded(self) -> None:
        """
        A stock with all NaN returns should not appear in the windowed output.
        The output should contain no NaN.
        """
        T = 64
        n_dates = T + 30
        rng = np.random.RandomState(42)

        dates = pd.bdate_range("2000-01-01", periods=n_dates, freq="B")
        permnos = [10001, 10002, 10003]

        # Stock 10001: normal returns
        ret_1 = rng.randn(n_dates) * 0.01
        # Stock 10002: ALL NaN
        ret_2 = np.full(n_dates, np.nan)
        # Stock 10003: normal returns
        ret_3 = rng.randn(n_dates) * 0.01

        returns_df = pd.DataFrame(
            {10001: ret_1, 10002: ret_2, 10003: ret_3},
            index=dates,
        )
        vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

        windows, metadata, raw_returns = create_windows(
            returns_df, vol_df, permnos, T=T, stride=10,
        )

        # Stock 10002 should not appear in metadata
        if len(metadata) > 0:
            stock_ids_in_output = set(metadata["stock_id"].unique())
            assert 10002 not in stock_ids_in_output, (
                "Stock 10002 (all NaN) should not appear in windows"
            )

        # No NaN in windows tensor
        assert not torch.any(torch.isnan(windows)), (
            "Output windows should contain no NaN"
        )

        # No NaN in raw_returns tensor
        assert not torch.any(torch.isnan(raw_returns)), (
            "Raw returns output should contain no NaN"
        )


# ---------------------------------------------------------------------------
# 4. Extreme vol ratios winsorized
# ---------------------------------------------------------------------------


class TestExtremeVolRatiosWinsorized:
    """Extreme volatility outliers should be clipped by winsorization in rescaling."""

    def test_extreme_vol_ratios_winsorized(self) -> None:
        """
        One stock with vol=5.0 among stocks at ~0.20 should be winsorized
        so that the rescaled exposure does not blow up.
        """
        rng = np.random.RandomState(42)
        n = 20
        AU = 3
        n_dates = 100

        stock_ids = list(range(10001, 10001 + n))
        dates = pd.bdate_range("2020-01-01", periods=n_dates, freq="B")

        # Normal vols around 0.20, except stock 0 has extreme vol=5.0
        vol_data = rng.uniform(0.18, 0.22, size=(n_dates, n))
        vol_data[:, 0] = 5.0  # Extreme outlier

        vol_df = pd.DataFrame(vol_data, index=dates, columns=stock_ids)
        B_A = rng.randn(n, AU) * 0.1

        target_ts: pd.Timestamp = dates[50]  # type: ignore[assignment]
        target_date_str = str(target_ts.date())
        universe_snapshots = {target_date_str: stock_ids}

        B_est = rescale_estimation(
            B_A, vol_df, universe_snapshots, stock_ids,
        )

        assert target_date_str in B_est
        B_rescaled = B_est[target_date_str]

        # Without winsorization, stock 0 ratio would be ~25x (5.0/0.20).
        # With P5/P95 winsorization, it should be capped.
        # The rescaled value for stock 0 should not be much larger than others.
        ratio_stock0 = np.linalg.norm(B_rescaled[0]) / np.median(
            np.linalg.norm(B_rescaled[1:], axis=1)
        )

        # With winsorization, ratio should be bounded (not 25x)
        assert ratio_stock0 < 10.0, (
            f"Stock 0 rescaled norm ratio={ratio_stock0:.2f} is too large; "
            "winsorization may not be working"
        )


# ---------------------------------------------------------------------------
# 5. Near-singular covariance
# ---------------------------------------------------------------------------


class TestNearSingularCovariance:
    """Nearly collinear factor returns should still yield PSD covariance (via Ledoit-Wolf)."""

    def test_near_singular_covariance(self) -> None:
        """
        z_hat with nearly collinear columns should still produce a PSD
        covariance via Ledoit-Wolf shrinkage, and assemble_risk_model
        should produce Sigma_assets = B @ Sigma_z @ B.T + diag(D_eps).
        """
        rng = np.random.RandomState(42)
        n_dates = 100
        AU = 4
        n = 15

        # Build z_hat with nearly collinear columns
        base = rng.randn(n_dates, 1)
        z_hat = np.hstack([
            base,
            base + 1e-8 * rng.randn(n_dates, 1),  # Nearly identical to base
            rng.randn(n_dates, 1),
            rng.randn(n_dates, 1),
        ])

        # Ledoit-Wolf should handle this
        Sigma_z, _, _ = estimate_sigma_z(z_hat)

        # Verify PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(Sigma_z)
        assert np.all(eigvals >= -1e-10), (
            f"Sigma_z has negative eigenvalues: min={eigvals.min():.2e}"
        )

        # Assemble full risk model
        B_A_port = rng.randn(n, AU) * 0.1
        D_eps = np.full(n, 0.001)

        risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)

        # Sigma_assets should be PSD
        Sigma_assets = risk_model["Sigma_assets"]
        eigvals_assets = np.linalg.eigvalsh(Sigma_assets)
        assert np.all(eigvals_assets >= -1e-10), (
            f"Sigma_assets has negative eigenvalues: min={eigvals_assets.min():.2e}"
        )

        # All expected keys present
        assert "eigenvalues" in risk_model
        assert "V" in risk_model
        assert "B_prime_port" in risk_model

        # Formula: Sigma_assets = B @ Sigma_z @ B.T + diag(D_eps)
        Sigma_manual = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
        np.testing.assert_allclose(
            Sigma_assets, Sigma_manual, atol=1e-10,
            err_msg="Sigma_assets != B @ Sigma_z @ B.T + diag(D_eps)",
        )

        # Eigendecomposition: B'_port = B @ V, eigenvalues from Sigma_z
        V = risk_model["V"]
        eigs = risk_model["eigenvalues"]
        B_prime_port = risk_model["B_prime_port"]
        np.testing.assert_allclose(
            B_prime_port, B_A_port @ V, atol=1e-10,
            err_msg="B_prime_port != B_A_port @ V",
        )


# ---------------------------------------------------------------------------
# 6. Entropy with single active factor
# ---------------------------------------------------------------------------


class TestEntropySingleActiveFactor:
    """With AU=1, entropy should be 0 and gradient should be all zeros."""

    def test_entropy_single_active_factor(self) -> None:
        """
        With a single active factor (AU=1), there is only one risk contribution
        which is 100% of total. H = -1*ln(1) = 0, gradient = 0.
        """
        rng = np.random.RandomState(42)
        n = 10

        B_prime = rng.randn(n, 1)  # AU = 1
        eigenvalues = np.array([0.5])
        w = np.ones(n) / n

        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        # With a single factor, normalized contribution c_hat = [1.0]
        # H = -1.0 * ln(1.0) = 0.0
        assert abs(H) < 1e-10, f"Expected H=0 with AU=1, got H={H}"

        # Gradient should be all zeros
        np.testing.assert_allclose(
            grad_H, np.zeros(n), atol=1e-10,
            err_msg="Expected grad_H = zeros with AU=1",
        )


# ---------------------------------------------------------------------------
# 7. Non-zero drift returns: z-scoring and OLS must handle non-zero mean
# ---------------------------------------------------------------------------


class TestNonZeroDriftReturns:
    """Real returns have non-zero mean (drift). Verify z-scoring and OLS work."""

    def test_zscore_removes_drift(self) -> None:
        """
        Z-scoring per window must normalize mean to ~0 even when the raw
        returns have a significant positive drift (mu=0.05/252 per day).
        """
        T = 64
        n_stocks = 3
        n_days = T + 50
        rng = np.random.RandomState(42)

        dates = pd.bdate_range("2000-01-01", periods=n_days, freq="B")
        drift = 0.05 / 252  # ~20bp/day annualizing to ~5%
        stock_ids = list(range(n_stocks))

        returns_df = pd.DataFrame(
            rng.randn(n_days, n_stocks) * 0.01 + drift,
            index=dates,
            columns=stock_ids,
        )
        vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

        windows, metadata, raw_returns = create_windows(
            returns_df, vol_df, stock_ids, T=T, stride=10,
        )

        # CONV-02: z-scored windows must have mean≈0, std≈1 per window per feature
        for i in range(min(5, windows.shape[0])):
            for f_idx in range(windows.shape[2]):
                feat = windows[i, :, f_idx].numpy()
                assert abs(feat.mean()) < 1e-3, (
                    f"Window {i}, feat {f_idx}: z-scored mean={feat.mean():.6f} "
                    f"(expected < 1e-3 despite drift)"
                )
                assert abs(feat.std() - 1.0) < 0.05, (
                    f"Window {i}, feat {f_idx}: z-scored std={feat.std():.6f} "
                    f"(expected ~1.0)"
                )

        # Raw returns must NOT be z-scored (needed for co-movement loss)
        raw_means = raw_returns.mean(dim=1)
        # With drift, raw means should be significantly non-zero
        assert torch.any(torch.abs(raw_means) > 1e-4), (
            "Raw returns with drift should have non-zero per-window mean"
        )

    def test_ols_regression_with_drift(self) -> None:
        """
        Factor regression z_hat = (B^T B)^{-1} B^T r must work correctly
        when returns have a non-zero mean (drift component).
        """
        from src.risk_model.factor_regression import estimate_factor_returns

        rng = np.random.RandomState(42)
        n = 10
        au = 3
        n_dates = 50
        drift = 0.05 / 252

        stock_ids = list(range(n))
        dates = pd.bdate_range("2020-01-02", periods=n_dates, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Non-trivial B matrix
        B_raw = rng.randn(n, au).astype(np.float64) * 0.3
        B_A_by_date: dict[str, np.ndarray] = {d: B_raw.copy() for d in date_strs}

        # True factor returns with drift
        z_true = rng.randn(n_dates, au).astype(np.float64) * 0.01 + drift

        # r = B @ z (no noise -> exact recovery expected)
        returns_data = np.array([B_raw @ z_true[t] for t in range(n_dates)])
        returns = pd.DataFrame(returns_data, index=date_strs, columns=stock_ids)
        universe_snapshots: dict[str, list[int]] = {d: stock_ids[:] for d in date_strs}

        z_hat, valid_dates = estimate_factor_returns(
            B_A_by_date, returns, universe_snapshots,
        )

        # OLS must recover z_true exactly even with drift
        for t_idx, d in enumerate(valid_dates):
            d_idx = date_strs.index(d)
            np.testing.assert_allclose(
                z_hat[t_idx], z_true[d_idx], atol=1e-8,
                err_msg=f"OLS recovery failed at {d} with drifted returns",
            )
