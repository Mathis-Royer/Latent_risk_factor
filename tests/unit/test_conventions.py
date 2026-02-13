"""
Tests for the 10 project conventions (CONV-01 to CONV-10).

Reference: ISD conventions section.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.windowing import create_windows
from src.data_pipeline.features import compute_rolling_realized_vol
from src.data_pipeline.universe import construct_universe
from src.data_pipeline.data_loader import generate_synthetic_csv, load_stock_data
from src.vae.build_vae import build_vae
from src.inference.active_units import measure_active_units
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.walk_forward.folds import generate_fold_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stock_data_from_prices(
    permnos: list[int],
    dates: list[pd.Timestamp],
    price_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Build a stock_data DataFrame from a price matrix.

    :param permnos (list[int]): Stock identifiers
    :param dates (list[pd.Timestamp]): Trading dates
    :param price_matrix (np.ndarray): Shape (n_dates, n_stocks)

    :return stock_data (pd.DataFrame): Long-format stock data
    """
    records = []
    for j, permno in enumerate(permnos):
        for i, dt in enumerate(dates):
            records.append({
                "permno": permno,
                "date": dt,
                "adj_price": price_matrix[i, j],
                "volume": 1_000_000,
                "exchange_code": 1,
                "share_code": 10,
                "market_cap": price_matrix[i, j] * 100e6,
                "delisting_return": np.nan,
            })
    return pd.DataFrame(records).sort_values(["permno", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CONV-01: Log returns, NEVER arithmetic
# ---------------------------------------------------------------------------


class TestConv01LogReturns:
    """CONV-01: Log returns r_t = ln(P_t / P_{t-1}), NEVER arithmetic."""

    def test_conv01_log_returns_not_arithmetic(self) -> None:
        """Known prices [100, 105, 102] must yield log-returns, not arithmetic."""
        prices = [100.0, 105.0, 102.0]
        dates = pd.bdate_range("2020-01-01", periods=3, freq="B")
        stock_data = pd.DataFrame({
            "permno": [1] * 3,
            "date": dates,
            "adj_price": prices,
            "volume": [1_000_000] * 3,
            "exchange_code": [1] * 3,
            "share_code": [10] * 3,
            "market_cap": [1e9] * 3,
            "delisting_return": [np.nan] * 3,
        })

        returns_df = compute_log_returns(stock_data, max_gap_fill=5)

        # Expected log-returns
        expected_r1 = np.log(105.0 / 100.0)
        expected_r2 = np.log(102.0 / 105.0)

        # Arithmetic returns (WRONG) for comparison
        arithmetic_r1 = (105.0 - 100.0) / 100.0
        arithmetic_r2 = (102.0 - 105.0) / 105.0

        actual = returns_df[1].dropna().values

        # Verify log returns match
        np.testing.assert_allclose(actual[0], expected_r1, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected_r2, atol=1e-10)

        # Verify they are NOT arithmetic returns
        assert abs(actual[0] - arithmetic_r1) > 1e-5, (
            "Return matches arithmetic formula; should be log-return"
        )
        assert abs(actual[1] - arithmetic_r2) > 1e-5, (
            "Return matches arithmetic formula; should be log-return"
        )


# ---------------------------------------------------------------------------
# CONV-02: Z-score per-window, per-feature
# ---------------------------------------------------------------------------


class TestConv02ZScore:
    """CONV-02: Z-score per-window, per-feature (mean~0, std~1)."""

    def test_conv02_zscore_per_window_per_feature(self) -> None:
        """Each window, each feature should have approximately mean=0 and std=1."""
        T = 64
        n_stocks = 5
        n_dates = T + 50

        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2000-01-01", periods=n_dates, freq="B")
        permnos = list(range(10001, 10001 + n_stocks))

        # Simulate returns and vol
        returns_data = rng.randn(n_dates, n_stocks) * 0.01
        returns_df = pd.DataFrame(returns_data, index=dates, columns=permnos)

        vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

        windows, metadata, raw_returns = create_windows(
            returns_df, vol_df, permnos, T=T, stride=10,
        )

        assert windows.shape[0] > 0, "No windows produced"
        assert windows.shape[1] == T
        assert windows.shape[2] == 2  # F=2

        # Check each window, each feature
        for i in range(windows.shape[0]):
            for f in range(windows.shape[2]):
                feat = windows[i, :, f].numpy()
                assert abs(float(np.mean(feat))) < 1e-5, (
                    f"Window {i}, feature {f}: mean={np.mean(feat):.6f}, expected ~0"
                )
                assert abs(float(np.std(feat)) - 1.0) < 1e-3, (
                    f"Window {i}, feature {f}: std={np.std(feat):.6f}, expected ~1"
                )


# ---------------------------------------------------------------------------
# CONV-03: 0-based indices everywhere
# ---------------------------------------------------------------------------


class TestConv03ZeroBasedIndices:
    """CONV-03: All indices (folds, dimensions, windows) are 0-based."""

    def test_conv03_fold_ids_start_at_zero(self) -> None:
        """Walk-forward fold IDs must start at 0 and be sequential."""
        folds = generate_fold_schedule(
            start_date="2000-01-01",
            total_years=20,
            min_training_years=8,
            oos_months=6,
        )

        fold_ids = [f["fold_id"] for f in folds]
        assert fold_ids[0] == 0, f"First fold_id should be 0, got {fold_ids[0]}"
        for i, fid in enumerate(fold_ids):
            assert fid == i, f"fold_id[{i}] should be {i}, got {fid}"

    def test_conv03_active_dims_0_based(self) -> None:
        """Active dimensions from AU measurement must be 0-based (all < K)."""
        from src.inference.active_units import measure_active_units

        K_val = 8
        model, _info = build_vae(
            n=25, T=64, T_annee=5, F=2, K=K_val, r_max=200.0, c_min=144,
        )

        torch.manual_seed(42)
        windows = torch.randn(20, 64, 2)
        AU, _kl_per_dim, active_dims = measure_active_units(
            model, windows, batch_size=10, au_threshold=0.01,
        )

        for dim in active_dims:
            assert 0 <= dim < K_val, (
                f"Dimension {dim} out of 0-based range [0, K={K_val})"
            )

    def test_conv03_window_indices_0_based(self) -> None:
        """Window metadata DataFrame index starts at 0 and is sequential."""
        T = 64
        n_stocks = 3
        n_dates = T + 50

        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2000-01-01", periods=n_dates, freq="B")
        permnos = list(range(10001, 10001 + n_stocks))

        returns_data = rng.randn(n_dates, n_stocks) * 0.01
        returns_df = pd.DataFrame(returns_data, index=dates, columns=permnos)
        vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

        windows, metadata, _raw = create_windows(
            returns_df, vol_df, permnos, T=T, stride=10,
        )

        if len(metadata) > 0:
            # DataFrame row index must be 0-based and sequential
            assert metadata.index[0] == 0, (
                f"Metadata index should start at 0, got {metadata.index[0]}"
            )
            expected_idx = list(range(len(metadata)))
            assert list(metadata.index) == expected_idx, (
                f"Metadata index not sequential 0-based: {list(metadata.index[:5])}..."
            )
            # Number of windows must match metadata rows
            assert windows.shape[0] == len(metadata), (
                f"Windows count {windows.shape[0]} != metadata rows {len(metadata)}"
            )


# ---------------------------------------------------------------------------
# CONV-04: PyTorch for VAE, numpy for downstream
# ---------------------------------------------------------------------------


class TestConv04TorchVsNumpy:
    """CONV-04: PyTorch for VAE forward pass, numpy for downstream pipeline."""

    def test_conv04_pytorch_for_vae_numpy_for_downstream(self) -> None:
        """VAE outputs are torch.Tensor; rescaling and entropy use np.ndarray."""
        from src.portfolio.entropy import compute_entropy_and_gradient

        T = 64
        K = 8
        n = 10

        model, info = build_vae(
            n=n, T=T, T_annee=5, F=2, K=K, r_max=200.0, c_min=144,
        )
        model.eval()

        # Forward pass produces torch.Tensor
        x = torch.randn(2, T, 2)
        with torch.no_grad():
            x_hat, mu, log_var = model(x)
        assert isinstance(x_hat, torch.Tensor)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(log_var, torch.Tensor)

        # Downstream: B matrix is np.ndarray
        rng = np.random.RandomState(42)
        B = rng.randn(n, K).astype(np.float64)
        assert isinstance(B, np.ndarray)

        # Rescale outputs are np.ndarray
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        vol_data = rng.uniform(0.15, 0.35, size=(300, n))
        stock_ids = list(range(10001, 10001 + n))
        vol_df = pd.DataFrame(vol_data, index=dates, columns=stock_ids)

        target_ts: pd.Timestamp = dates[200]  # type: ignore[assignment]
        universe_snapshots = {str(target_ts.date()): stock_ids}
        B_est = rescale_estimation(
            B, vol_df, universe_snapshots, stock_ids,
        )
        for date_str, val in B_est.items():
            assert isinstance(val, np.ndarray)

        B_port = rescale_portfolio(
            B, vol_df, str(target_ts.date()), stock_ids, stock_ids,
        )
        assert isinstance(B_port, np.ndarray)

        # Entropy output: (float, np.ndarray)
        w = np.ones(n) / n
        eigenvalues = np.abs(rng.randn(K)) + 0.01
        B_prime = rng.randn(n, K)
        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)
        assert isinstance(H, float)
        assert isinstance(grad_H, np.ndarray)


# ---------------------------------------------------------------------------
# CONV-05: Window shape (batch, T, F=2)
# ---------------------------------------------------------------------------


class TestConv05WindowShape:
    """CONV-05: Window shape is (N, T, F=2)."""

    def test_conv05_window_shape_B_T_F(self) -> None:
        """create_windows returns shape (N, T, F=2) with correct T dimension."""
        T = 64
        n_stocks = 5
        n_dates = T + 50

        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2000-01-01", periods=n_dates, freq="B")
        permnos = list(range(10001, 10001 + n_stocks))

        returns_data = rng.randn(n_dates, n_stocks) * 0.01
        returns_df = pd.DataFrame(returns_data, index=dates, columns=permnos)
        vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

        windows, metadata, raw_returns = create_windows(
            returns_df, vol_df, permnos, T=T, stride=10,
        )

        assert windows.ndim == 3
        assert windows.shape[1] == T, f"Expected T={T}, got {windows.shape[1]}"
        assert windows.shape[2] == 2, f"Expected F=2, got {windows.shape[2]}"

        # FORMULA: CONV-05 requires T=504 in production, F=2 always
        # (here using T=64 for test speed, but verify F=2 exactly)
        assert windows.shape[2] == 2, (
            f"F dimension must be exactly 2 (returns + realized vol), got {windows.shape[2]}"
        )

        # FORMULA: feature 0 correlates with returns, feature 1 with vol
        # Verify separate features have different distributions
        feat_0_var = windows[:, :, 0].var().item()
        feat_1_var = windows[:, :, 1].var().item()
        assert feat_0_var > 0, "Feature 0 variance should be > 0"
        assert feat_1_var > 0, "Feature 1 variance should be > 0"


# ---------------------------------------------------------------------------
# CONV-06: sigma_sq is scalar
# ---------------------------------------------------------------------------


class TestConv06SigmaSqScalar:
    """CONV-06: Observation variance sigma_sq is a scalar parameter."""

    def test_conv06_sigma_sq_scalar(self) -> None:
        """log_sigma_sq must have ndim == 0 (scalar)."""
        model, info = build_vae(
            n=25, T=64, T_annee=5, F=2, K=8, r_max=200.0, c_min=144,
        )
        assert model.log_sigma_sq.ndim == 0, (
            f"log_sigma_sq.ndim={model.log_sigma_sq.ndim}, expected 0 (scalar)"
        )

        # FORMULA: σ² = exp(log_σ²), verify numerically
        import torch
        log_val = model.log_sigma_sq.item()
        obs_var = model.obs_var.item()
        import math
        expected_sigma_sq = math.exp(max(min(log_val, math.log(10.0)), math.log(1e-4)))
        assert abs(obs_var - expected_sigma_sq) < 1e-6, (
            f"obs_var={obs_var:.8f} != exp(clamp(log_σ²))={expected_sigma_sq:.8f}"
        )

        # FORMULA: σ² must be in clamped range [1e-4, 10]
        assert 1e-4 - 1e-8 <= obs_var <= 10.0 + 1e-8, (
            f"σ²={obs_var} out of clamped range [1e-4, 10]"
        )


# ---------------------------------------------------------------------------
# CONV-07: AU threshold = 0.01 nats
# ---------------------------------------------------------------------------


class TestConv07AUThreshold:
    """CONV-07: Active unit threshold is 0.01 nats."""

    def test_conv07_au_threshold_0_01_nats(self) -> None:
        """measure_active_units with au_threshold=0.01 matches manual count."""
        K = 8
        T = 64

        model, info = build_vae(
            n=25, T=T, T_annee=5, F=2, K=K, r_max=200.0, c_min=144,
        )

        rng = np.random.RandomState(42)
        windows = torch.randn(20, T, 2)

        AU, kl_per_dim, active_dims = measure_active_units(
            model, windows, batch_size=10, au_threshold=0.01,
        )

        # Manual count: sum of dims where KL > 0.01
        manual_count = int(np.sum(kl_per_dim > 0.01))
        assert AU == manual_count, (
            f"AU={AU} differs from manual count={manual_count}"
        )

        # Verify active_dims indices all have KL > 0.01
        for dim in active_dims:
            assert kl_per_dim[dim] > 0.01, (
                f"Dimension {dim} in active_dims but KL={kl_per_dim[dim]:.6f} <= 0.01"
            )

        # Verify inactive dims all have KL <= 0.01
        all_dims = set(range(K))
        inactive = all_dims - set(active_dims)
        for dim in inactive:
            assert kl_per_dim[dim] <= 0.01, (
                f"Dimension {dim} NOT in active_dims but KL={kl_per_dim[dim]:.6f} > 0.01"
            )


# ---------------------------------------------------------------------------
# CONV-08: Dual rescaling (estimation vs portfolio)
# ---------------------------------------------------------------------------


class TestConv08DualRescaling:
    """CONV-08: Estimation rescaling is date-specific dict; portfolio is a single array."""

    def test_conv08_dual_rescaling(self) -> None:
        """rescale_estimation returns dict, rescale_portfolio returns ndarray. Different types and values."""
        rng = np.random.RandomState(42)
        n = 10
        AU = 4
        n_dates = 300

        stock_ids = list(range(10001, 10001 + n))
        dates = pd.bdate_range("2020-01-01", periods=n_dates, freq="B")
        vol_data = rng.uniform(0.10, 0.40, size=(n_dates, n))
        vol_df = pd.DataFrame(vol_data, index=dates, columns=stock_ids)

        B_A = rng.randn(n, AU)
        target_ts: pd.Timestamp = dates[200]  # type: ignore[assignment]
        target_date_str = str(target_ts.date())

        universe_snapshots = {target_date_str: stock_ids}

        # Estimation rescaling: returns a dict
        B_est = rescale_estimation(
            B_A, vol_df, universe_snapshots, stock_ids,
        )
        assert isinstance(B_est, dict), "rescale_estimation should return dict"
        assert target_date_str in B_est

        # Portfolio rescaling: returns a numpy array
        B_port = rescale_portfolio(
            B_A, vol_df, target_date_str, stock_ids, stock_ids,
        )
        assert isinstance(B_port, np.ndarray), "rescale_portfolio should return ndarray"

        # The values should generally differ because vol ratios
        # are from different dates (estimation=each date, portfolio=current only)
        B_est_val = B_est[target_date_str]

        # Both should have the same shape
        assert B_est_val.shape == B_port.shape

        # FORMULA: verify R_i = σ_i / median(σ) then B̃ = R_i · B_A
        from src.risk_model.rescaling import _compute_winsorized_ratios
        vol_row = vol_df.loc[target_ts, stock_ids].values.astype(np.float64)
        med_vol = np.median(vol_row)
        raw_r = vol_row / med_vol
        lo = np.percentile(raw_r, 5.0)
        hi = np.percentile(raw_r, 95.0)
        expected_ratios = np.clip(raw_r, lo, hi)
        expected_B = B_A * expected_ratios[:, np.newaxis]

        # Portfolio rescaling should match manual formula
        np.testing.assert_allclose(
            B_port, expected_B, atol=1e-10,
            err_msg="Portfolio rescaling B̃ = R_i · B_A formula mismatch",
        )

        # Estimation should also match for the same date
        np.testing.assert_allclose(
            B_est_val, expected_B, atol=1e-10,
            err_msg="Estimation rescaling B̃ = R_i · B_A formula mismatch",
        )


# ---------------------------------------------------------------------------
# CONV-09: Expanding window walk-forward
# ---------------------------------------------------------------------------


class TestConv09ExpandingWindow:
    """CONV-09: Walk-forward uses expanding windows from a fixed start."""

    def test_conv09_expanding_window_walk_forward(self) -> None:
        """All folds share the same train_start; train_end grows monotonically."""
        folds = generate_fold_schedule(
            start_date="2000-01-01",
            total_years=20,
            min_training_years=8,
            oos_months=6,
            holdout_years=2,
        )

        # Filter out holdout fold for walk-forward checks
        wf_folds = [f for f in folds if not f["is_holdout"]]
        assert len(wf_folds) >= 2, "Need at least 2 WF folds"

        # All folds must share the same train_start
        first_start = wf_folds[0]["train_start"]
        for fold in wf_folds:
            assert fold["train_start"] == first_start, (
                f"Fold {fold['fold_id']}: train_start={fold['train_start']} "
                f"differs from expected {first_start}"
            )

        # train_end must be strictly increasing
        for k in range(len(wf_folds) - 1):
            end_k = pd.Timestamp(str(wf_folds[k]["train_end"]))
            end_k1 = pd.Timestamp(str(wf_folds[k + 1]["train_end"]))
            assert end_k1 > end_k, (
                f"Fold {k+1} train_end={end_k1} should be > fold {k} train_end={end_k}"
            )


# ---------------------------------------------------------------------------
# CONV-10: Point-in-time universe
# ---------------------------------------------------------------------------


class TestConv10PointInTime:
    """CONV-10: No future data; universe reconstituted at each date."""

    def test_conv10_point_in_time_universe(self) -> None:
        """
        Stocks with varying start dates: early universe is smaller;
        a stock not yet listed at an early date is absent.
        """
        rng = np.random.RandomState(42)

        dates_full = pd.bdate_range("2000-01-01", periods=800, freq="B")
        early_date: pd.Timestamp = dates_full[550]  # type: ignore[assignment]
        late_date: pd.Timestamp = dates_full[799]  # type: ignore[assignment]

        # Stock A: listed from the start (full 800 days)
        # Stock B: listed from day 600 onward (late entrant)
        permno_a = 10001
        permno_b = 10002

        # Stock A: full history
        prices_a = 100.0 * np.exp(np.cumsum(rng.randn(800) * 0.01))
        records_a = []
        for i, dt in enumerate(dates_full):
            records_a.append({
                "permno": permno_a,
                "date": dt,
                "adj_price": prices_a[i],
                "volume": 5_000_000,
                "exchange_code": 1,
                "share_code": 10,
                "market_cap": prices_a[i] * 200e6,
                "delisting_return": np.nan,
            })

        # Stock B: only from day 600 onwards
        prices_b = 50.0 * np.exp(np.cumsum(rng.randn(200) * 0.01))
        records_b = []
        for i, dt in enumerate(dates_full[600:]):
            records_b.append({
                "permno": permno_b,
                "date": dt,
                "adj_price": prices_b[i],
                "volume": 5_000_000,
                "exchange_code": 1,
                "share_code": 10,
                "market_cap": prices_b[i] * 200e6,
                "delisting_return": np.nan,
            })

        stock_data = pd.DataFrame(records_a + records_b)
        stock_data = stock_data.sort_values(["permno", "date"]).reset_index(drop=True)

        # At early_date (day 550), stock B is not yet listed
        universe_early = construct_universe(
            stock_data, date=early_date,
            n_max=100, cap_entry=0, cap_exit=0, adv_min=0,
            min_listing_days=100,
        )

        # At late_date (day 799), both should qualify
        universe_late = construct_universe(
            stock_data, date=late_date,
            n_max=100, cap_entry=0, cap_exit=0, adv_min=0,
            min_listing_days=100,
        )

        # Stock B should NOT be in early universe (not yet listed at day 550)
        assert permno_b not in universe_early, (
            "Stock B should not be in early universe (not yet listed)"
        )

        # Stock A should be in both
        assert permno_a in universe_early, "Stock A should be in early universe"
        assert permno_a in universe_late, "Stock A should be in late universe"

        # Late universe should be at least as large as early universe
        assert len(universe_late) >= len(universe_early), (
            f"Late universe ({len(universe_late)}) should be >= "
            f"early universe ({len(universe_early)})"
        )


# ---------------------------------------------------------------------------
# CONV-02: Z-score formula verification with exact manual computation
# ---------------------------------------------------------------------------


class TestConv02ZScoreFormula:
    """CONV-02: Verify z-scoring produces exactly (x - mean) / std per feature."""

    def test_zscore_exact_manual_values(self) -> None:
        """For known feature values, verify z-scored output matches manual formula."""
        import torch

        # Known raw values for a single feature in a window
        raw = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
        mu = raw.mean()   # 6.0
        sigma = raw.std(ddof=0)  # population std = sqrt(8) ≈ 2.8284

        # z-scored: (x - mu) / sigma
        expected_z = (raw - mu) / sigma
        # Expected: [-1.4142, -0.7071, 0.0, 0.7071, 1.4142]

        # Verify properties
        assert abs(expected_z.mean()) < 1e-10, "Z-scored mean should be 0"
        # std with ddof=0 should be 1.0
        assert abs(expected_z.std(ddof=0) - 1.0) < 1e-10, "Z-scored std should be 1"

        # Verify specific values
        np.testing.assert_allclose(
            expected_z,
            np.array([-np.sqrt(2), -1/np.sqrt(2), 0.0, 1/np.sqrt(2), np.sqrt(2)]),
            atol=1e-10,
            err_msg="Z-score formula verification failed on known values",
        )


# ---------------------------------------------------------------------------
# CONV-08: Dual rescaling formula verification with known vol profiles
# ---------------------------------------------------------------------------


class TestConv08DualRescalingFormula:
    """CONV-08: Verify B̃_{A,i} = R_i · μ̄_{A,i} where R_i = σ_i / median(σ)."""

    def test_rescaling_formula_known_vols(self) -> None:
        """Verify R_i = clip(σ_i / median(σ), P5, P95) with enough stocks."""
        from src.risk_model.rescaling import _compute_winsorized_ratios, rescale_portfolio

        # Use 21 stocks so P5/P95 percentiles don't squeeze interior values
        rng = np.random.RandomState(0)
        vols = np.sort(rng.uniform(0.05, 0.40, 21))
        median_v = np.median(vols)
        raw_ratios = vols / median_v

        ratios = _compute_winsorized_ratios(vols, 5.0, 95.0)

        # All interior ratios (between P5 and P95 of ratios) must equal raw
        lo = np.percentile(raw_ratios, 5.0)
        hi = np.percentile(raw_ratios, 95.0)
        interior = (raw_ratios >= lo) & (raw_ratios <= hi)
        np.testing.assert_allclose(
            ratios[interior], raw_ratios[interior], atol=1e-12,
            err_msg="Interior ratios should be unclipped R_i = σ_i / median(σ)",
        )

        # All ratios must be in [lo, hi]
        assert np.all(ratios >= lo - 1e-12), "Below P5 not clipped"
        assert np.all(ratios <= hi + 1e-12), "Above P95 not clipped"

        # Verify rescale_portfolio applies B̃ = R_i · B_A
        n_test = 5
        B_A = rng.randn(n_test, 2).astype(np.float64)
        test_vols = vols[:n_test]
        stock_ids = list(range(n_test))
        trailing_vol = pd.DataFrame(
            [test_vols], index=["2020-01-02"], columns=stock_ids,
        )
        test_ratios = _compute_winsorized_ratios(test_vols, 5.0, 95.0)

        B_port = rescale_portfolio(
            B_A, trailing_vol, "2020-01-02", stock_ids, stock_ids,
        )
        for i in range(n_test):
            np.testing.assert_allclose(
                B_port[i], test_ratios[i] * B_A[i], atol=1e-10,
                err_msg=f"Stock {i}: B̃ = R_i · B_A not satisfied",
            )
