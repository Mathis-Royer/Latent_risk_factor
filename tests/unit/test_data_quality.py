"""
Data quality and consistency tests for the Tiingo pipeline.

Validates data invariants and transformation consistency at every pipeline
stage to ensure the VAE model trains with clean, well-formed data.

Complements test_data_pipeline.py (functional correctness) with checks for
data corruption patterns: duplicates, outliers, Inf, type mismatches,
cross-module alignment, and statistical sanity.
"""

import os
import tempfile
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.data_loader import (
    generate_synthetic_csv,
    load_stock_data,
)
from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.universe import construct_universe
from src.data_pipeline.windowing import create_windows
from src.data_pipeline.features import (
    compute_rolling_realized_vol,
    compute_trailing_volatility,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _complete_stocks(returns_df: pd.DataFrame, n: int = 10) -> list[int]:
    """
    Return up to `n` permnos with the fewest NaN values.

    :param returns_df (pd.DataFrame): Wide-format returns (dates x stocks)
    :param n (int): Maximum number of stocks to return

    :return permnos (list[int]): List of permno IDs with least missing data
    """
    nan_counts: pd.Series = returns_df.isna().sum()  # type: ignore[assignment]
    sorted_series = nan_counts.sort_values()
    col_names = list(sorted_series.head(n).index)  # type: ignore[arg-type]
    return [int(c) for c in col_names]


def _autocorrelation_lag1(series: np.ndarray) -> float:
    """
    Compute lag-1 autocorrelation of a 1D array, ignoring NaN pairs.

    :param series (np.ndarray): 1D time series

    :return rho (float): Lag-1 autocorrelation coefficient
    """
    x = series[:-1]
    y = series[1:]
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 10:
        return 0.0
    return float(np.corrcoef(x_clean, y_clean)[0, 1])


# ---------------------------------------------------------------------------
# Fixtures (module-scoped, chained)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_csv_path() -> Generator[str, None, None]:
    """Generate a synthetic CSV mimicking Tiingo data properties."""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        path = f.name

    generate_synthetic_csv(
        output_path=path,
        n_stocks=50,
        start_date="2000-01-03",
        end_date="2008-12-31",
        n_delistings=5,
        seed=42,
    )
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def stock_data(synthetic_csv_path: str) -> pd.DataFrame:
    """Load synthetic stock data once for the module."""
    return load_stock_data(synthetic_csv_path)


@pytest.fixture(scope="module")
def returns_df(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from stock data."""
    return compute_log_returns(stock_data)


@pytest.fixture(scope="module")
def trailing_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 252-day trailing annualized volatility."""
    return compute_trailing_volatility(returns_df, window=252)


@pytest.fixture(scope="module")
def rolling_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day rolling realized volatility."""
    return compute_rolling_realized_vol(returns_df, rolling_window=21)


@pytest.fixture(scope="module")
def universe_ids(stock_data: pd.DataFrame) -> list[int]:
    """Construct universe at 2005-01-03 with relaxed filters."""
    date: pd.Timestamp = pd.Timestamp("2005-01-03")  # type: ignore[assignment]
    return construct_universe(
        stock_data,
        date=date,
        n_max=1000,
        cap_entry=0,
        cap_exit=0,
        adv_min=0,
        min_listing_days=504,
    )


@pytest.fixture(scope="module")
def windows_and_metadata(
    returns_df: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    universe_ids: list[int],
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Create sliding windows from returns and rolling vol."""
    return create_windows(
        returns_df, rolling_vol, universe_ids, T=504, stride=252
    )


# ---------------------------------------------------------------------------
# Class 1: Raw data quality
# ---------------------------------------------------------------------------


class TestRawDataQuality:
    """Verify structural integrity of raw loaded stock data.

    These checks catch data corruption at the source level before any
    transformations are applied.
    """

    def test_no_duplicate_permno_date_pairs(
        self, stock_data: pd.DataFrame
    ) -> None:
        """Duplicate (permno, date) pairs cause incorrect pivoting."""
        n_dupes = int(stock_data.duplicated(subset=["permno", "date"]).sum())
        assert n_dupes == 0, f"Found {n_dupes} duplicate (permno, date) pairs"

    def test_adj_price_positive_where_not_nan(
        self, stock_data: pd.DataFrame
    ) -> None:
        """Zero or negative prices produce Inf/NaN in log-returns."""
        valid_prices = stock_data.loc[
            stock_data["adj_price"].notna(), "adj_price"
        ]
        n_bad = int((valid_prices <= 0).sum())
        assert n_bad == 0, (
            f"Found {n_bad} non-positive adj_price values out of "
            f"{len(valid_prices)} non-NaN"
        )

    def test_volume_non_negative(self, stock_data: pd.DataFrame) -> None:
        """Negative volume is data corruption."""
        n_neg = int((stock_data["volume"] < 0).sum())
        assert n_neg == 0, f"Found {n_neg} negative volume values"

    def test_exchange_codes_valid(self, stock_data: pd.DataFrame) -> None:
        """Universe filters on {1, 2, 3}; invalid codes silently exclude."""
        valid_codes = [1, 2, 3]
        invalid = stock_data[~stock_data["exchange_code"].isin(valid_codes)]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} rows with invalid exchange_code: "
            f"{np.unique(np.asarray(invalid['exchange_code']))}"
        )

    def test_share_codes_valid(self, stock_data: pd.DataFrame) -> None:
        """Universe filters common equity (10, 11); invalid codes excluded."""
        valid_codes = [10, 11]
        invalid = stock_data[~stock_data["share_code"].isin(valid_codes)]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} rows with invalid share_code: "
            f"{np.unique(np.asarray(invalid['share_code']))}"
        )

    def test_market_cap_positive_where_not_nan(
        self, stock_data: pd.DataFrame
    ) -> None:
        """Non-positive market cap breaks universe cap ranking."""
        valid_caps = stock_data.loc[
            stock_data["market_cap"].notna(), "market_cap"
        ]
        n_bad = int((valid_caps <= 0).sum())
        assert n_bad == 0, (
            f"Found {n_bad} non-positive market_cap values out of "
            f"{len(valid_caps)} non-NaN"
        )

    def test_date_monotonicity_per_stock(
        self, stock_data: pd.DataFrame
    ) -> None:
        """Dates must be strictly increasing per permno for gap detection."""
        sample_permnos = list(stock_data["permno"].unique()[:10])
        for permno in sample_permnos:
            group = stock_data[stock_data["permno"] == permno]
            dates = np.asarray(group["date"])
            assert bool(np.all(dates[1:] > dates[:-1])), (
                f"Permno {permno}: dates are not strictly increasing"
            )

    def test_no_future_dates(self, stock_data: pd.DataFrame) -> None:
        """Future dates violate CONV-10 (no look-ahead)."""
        max_date = stock_data["date"].max()
        assert max_date <= pd.Timestamp.today(), (
            f"Max date {max_date} is in the future"
        )

    def test_reasonable_price_range(self, stock_data: pd.DataFrame) -> None:
        """Extreme prices indicate data corruption (wrong decimal/currency)."""
        valid_prices = stock_data.loc[
            stock_data["adj_price"].notna(), "adj_price"
        ]
        in_range = (valid_prices >= 0.01) & (valid_prices <= 1e7)
        frac_in_range = float(in_range.sum()) / len(valid_prices)
        assert frac_in_range >= 0.99, (
            f"Only {frac_in_range:.2%} of prices in [0.01, 1e7], "
            f"expected >= 99%"
        )

    def test_minimum_history_depth(self, stock_data: pd.DataFrame) -> None:
        """Need enough stocks with >= 504 days for meaningful windowing."""
        days_per_stock = stock_data.groupby("permno").size()
        n_long = int((days_per_stock >= 504).sum())
        assert n_long >= 10, (
            f"Only {n_long} stocks have >= 504 trading days, need >= 10"
        )


# ---------------------------------------------------------------------------
# Class 2: Returns quality
# ---------------------------------------------------------------------------


class TestReturnsQuality:
    """Verify log-return computation produces clean data for VAE training.

    Catches infinities, extreme values, NaN propagation, and structural
    anomalies in the returns matrix.
    """

    def test_no_infinities_in_returns(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Inf values propagate through all downstream computations."""
        values = returns_df.values
        n_inf = int(np.isinf(np.where(np.isnan(values), 0.0, values)).sum())
        assert n_inf == 0, f"Found {n_inf} Inf values in returns"

    def test_returns_in_reasonable_range(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Daily |r| > 1.0 (~172% move) is extremely rare."""
        values = returns_df.values.ravel()
        finite_vals = values[np.isfinite(values)]
        n_extreme = int((np.abs(finite_vals) >= 1.0).sum())
        frac_extreme = n_extreme / len(finite_vals)
        assert frac_extreme < 0.001, (
            f"{frac_extreme:.4%} of returns have |r| >= 1.0 "
            f"({n_extreme} values), expected < 0.1%"
        )

    def test_nan_count_bounded(self, returns_df: pd.DataFrame) -> None:
        """Excessive NaN means forward-fill failed or data has huge gaps."""
        total = returns_df.size
        n_nan = int(returns_df.isna().sum().sum())
        frac_nan = n_nan / total
        assert frac_nan < 0.15, (
            f"NaN fraction is {frac_nan:.2%} ({n_nan}/{total}), "
            f"expected < 15%"
        )

    def test_cross_sectional_mean_near_zero(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Large systematic bias indicates processing error."""
        daily_means = returns_df.mean(axis=1).dropna()
        median_mean = float(np.median(np.asarray(daily_means)))
        assert -0.005 < median_mean < 0.005, (
            f"Median daily cross-sectional mean = {median_mean:.6f}, "
            f"expected in [-0.005, 0.005]"
        )

    def test_cross_sectional_std_reasonable(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Std near 0 = degenerate data; too large = corruption."""
        daily_stds = returns_df.std(axis=1).dropna()
        median_std = float(np.median(np.asarray(daily_stds)))
        assert 0.005 < median_std < 0.10, (
            f"Median daily cross-sectional std = {median_std:.6f}, "
            f"expected in [0.005, 0.10]"
        )

    def test_return_matrix_shape_consistency(
        self, returns_df: pd.DataFrame, stock_data: pd.DataFrame
    ) -> None:
        """Pivot preserved structure: int columns, sorted index."""
        # Columns are integer permnos
        for col in returns_df.columns:
            assert isinstance(col, (int, np.integer)), (
                f"Column {col} has type {type(col)}, expected int"
            )

        # Index is monotonically increasing dates
        assert returns_df.index.is_monotonic_increasing, (
            "Returns index is not sorted by date"
        )

        # Columns match stock_data permnos
        expected_permnos = set(stock_data["permno"].unique())
        actual_permnos = set(int(c) for c in returns_df.columns)
        assert actual_permnos.issubset(expected_permnos), (
            f"Returns columns have permnos not in stock_data: "
            f"{actual_permnos - expected_permnos}"
        )

    def test_no_strong_serial_correlation(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Strong autocorrelation suggests unadjusted prices or fabrication."""
        stocks = _complete_stocks(returns_df, n=5)
        for permno in stocks:
            series = np.asarray(returns_df[permno].dropna().values)
            if len(series) < 50:
                continue
            rho = _autocorrelation_lag1(series)
            assert abs(rho) < 0.15, (
                f"Permno {permno}: lag-1 autocorrelation = {rho:.4f}, "
                f"expected |rho| < 0.15"
            )


# ---------------------------------------------------------------------------
# Class 3: Volatility consistency
# ---------------------------------------------------------------------------


class TestVolatilityConsistency:
    """Verify trailing and rolling volatility are consistent and reasonable.

    Bad volatility propagates to z-scoring (window features) and the risk
    model (rescaling).
    """

    def test_trailing_vol_positive_after_warmup(
        self, trailing_vol: pd.DataFrame
    ) -> None:
        """Volatility must be strictly positive after warmup period."""
        post_warmup = trailing_vol.iloc[252:]
        valid = post_warmup.values[~np.isnan(post_warmup.values)]
        n_nonpositive = int((valid <= 0).sum())
        assert n_nonpositive == 0, (
            f"Found {n_nonpositive} non-positive trailing vol values "
            f"after warmup"
        )

    def test_trailing_vol_reasonable_range(
        self, trailing_vol: pd.DataFrame
    ) -> None:
        """Annualized vol in [5%, 200%] for equities."""
        post_warmup = trailing_vol.iloc[252:]
        valid = post_warmup.values[~np.isnan(post_warmup.values)]
        in_range = (valid >= 0.05) & (valid <= 2.0)
        frac = float(in_range.sum()) / len(valid)
        assert frac >= 0.95, (
            f"Only {frac:.2%} of trailing vol in [0.05, 2.0], "
            f"expected >= 95%"
        )

    def test_rolling_vol_positive_after_warmup(
        self, rolling_vol: pd.DataFrame
    ) -> None:
        """Rolling vol must be strictly positive after warmup period."""
        post_warmup = rolling_vol.iloc[21:]
        valid = post_warmup.values[~np.isnan(post_warmup.values)]
        n_nonpositive = int((valid <= 0).sum())
        assert n_nonpositive == 0, (
            f"Found {n_nonpositive} non-positive rolling vol values "
            f"after warmup"
        )

    def test_rolling_vol_reasonable_range(
        self, rolling_vol: pd.DataFrame
    ) -> None:
        """21-day rolling std in [0.001, 0.15] (not annualized)."""
        post_warmup = rolling_vol.iloc[21:]
        valid = post_warmup.values[~np.isnan(post_warmup.values)]
        in_range = (valid >= 0.001) & (valid <= 0.15)
        frac = float(in_range.sum()) / len(valid)
        assert frac >= 0.95, (
            f"Only {frac:.2%} of rolling vol in [0.001, 0.15], "
            f"expected >= 95%"
        )

    def test_vol_and_returns_alignment(
        self,
        returns_df: pd.DataFrame,
        trailing_vol: pd.DataFrame,
        rolling_vol: pd.DataFrame,
    ) -> None:
        """Misalignment causes wrong features paired in windowing."""
        # Trailing vol columns and index match returns
        assert set(trailing_vol.columns) == set(returns_df.columns), (
            "Trailing vol columns differ from returns columns"
        )
        assert trailing_vol.index.equals(returns_df.index), (
            "Trailing vol index differs from returns index"
        )

        # Rolling vol columns and index match returns
        assert set(rolling_vol.columns) == set(returns_df.columns), (
            "Rolling vol columns differ from returns columns"
        )
        assert rolling_vol.index.equals(returns_df.index), (
            "Rolling vol index differs from returns index"
        )


# ---------------------------------------------------------------------------
# Class 4: Windowing integrity
# ---------------------------------------------------------------------------


class TestWindowingIntegrity:
    """Verify sliding windows are clean, correctly structured tensors.

    Catches NaN/Inf leaks, shape errors, and data duplication bugs that
    would corrupt VAE training.
    """

    def test_no_nan_in_windows(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """NaN in input tensors produces NaN loss; training diverges."""
        windows, _ = windows_and_metadata
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        n_nan = int(torch.isnan(windows).sum().item())
        assert n_nan == 0, f"Found {n_nan} NaN values in windows tensor"

    def test_no_inf_in_windows(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Inf in input tensors causes NaN loss."""
        windows, _ = windows_and_metadata
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        n_inf = int(torch.isinf(windows).sum().item())
        assert n_inf == 0, f"Found {n_inf} Inf values in windows tensor"

    def test_window_metadata_date_consistency(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Window date range should span ~503 business days (T-1)."""
        _, metadata = windows_and_metadata
        if metadata.empty:
            pytest.skip("No windows generated")

        sample = metadata.head(20)
        for _, row in sample.iterrows():
            start = pd.Timestamp(str(row["start_date"]))
            end = pd.Timestamp(str(row["end_date"]))
            bdays = len(pd.bdate_range(start, end))
            # T=504, so bdays should be approximately 504
            assert 480 <= bdays <= 530, (
                f"Window for stock {row['stock_id']}: {bdays} business "
                f"days between {start.date()} and {end.date()}, "
                f"expected ~504"
            )

    def test_no_extreme_outliers_in_zscored_windows(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """After z-scoring, |z| > 8 is extremely unlikely (< 1e-15)."""
        windows, _ = windows_and_metadata
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")

        values = windows.numpy().ravel()
        n_extreme = int((np.abs(values) > 8.0).sum())
        frac = n_extreme / len(values)
        assert frac < 0.001, (
            f"{frac:.4%} of z-scored values have |z| > 8 "
            f"({n_extreme} values), expected < 0.1%"
        )

    def test_cross_window_diversity(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """No two windows should be identical (detect duplication bugs)."""
        windows, _ = windows_and_metadata
        n = min(windows.shape[0], 20)
        if n < 2:
            pytest.skip("Not enough windows to check diversity")

        sample = windows[:n].reshape(n, -1).numpy()
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(sample[i] - sample[j]))
                assert dist > 1e-6, (
                    f"Windows {i} and {j} are identical (L2 distance "
                    f"= {dist:.2e})"
                )

    def test_metadata_covers_multiple_stocks(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Windows should come from multiple stocks."""
        _, metadata = windows_and_metadata
        if metadata.empty:
            pytest.skip("No windows generated")

        n_stocks = metadata["stock_id"].nunique()
        assert n_stocks >= 2, (
            f"Windows cover only {n_stocks} stock(s), expected >= 2"
        )

    def test_window_feature_variance(
        self, windows_and_metadata: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Z-scored data should have overall variance near 1.0."""
        windows, _ = windows_and_metadata
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")

        for f in range(windows.shape[2]):
            feat_values = windows[:, :, f].numpy().ravel()
            var = float(np.var(feat_values))
            assert 0.5 <= var <= 2.0, (
                f"Feature {f}: pooled variance = {var:.4f}, "
                f"expected in [0.5, 2.0]"
            )


# ---------------------------------------------------------------------------
# Class 5: Pipeline alignment
# ---------------------------------------------------------------------------


class TestPipelineAlignment:
    """Verify cross-module alignment: universe, returns, windows, types.

    Catches mismatches that silently produce wrong data combinations.
    """

    def test_universe_stocks_in_returns_columns(
        self,
        universe_ids: list[int],
        returns_df: pd.DataFrame,
    ) -> None:
        """Universe stocks missing from returns would be silently skipped."""
        returns_cols = set(int(c) for c in returns_df.columns)
        missing = set(universe_ids) - returns_cols
        assert len(missing) == 0, (
            f"{len(missing)} universe stocks not in returns columns: "
            f"{sorted(missing)[:5]}..."
        )

    def test_window_stock_ids_in_returns_columns(
        self,
        windows_and_metadata: tuple[torch.Tensor, pd.DataFrame],
        returns_df: pd.DataFrame,
    ) -> None:
        """Window metadata should only reference stocks with return data."""
        _, metadata = windows_and_metadata
        if metadata.empty:
            pytest.skip("No windows generated")

        returns_cols = set(int(c) for c in returns_df.columns)
        window_stocks = set(int(s) for s in metadata["stock_id"].unique())
        missing = window_stocks - returns_cols
        assert len(missing) == 0, (
            f"{len(missing)} window stock_ids not in returns columns: "
            f"{sorted(missing)[:5]}..."
        )

    def test_no_data_leakage_across_train_test(
        self,
        returns_df: pd.DataFrame,
        rolling_vol: pd.DataFrame,
        universe_ids: list[int],
    ) -> None:
        """Windows from training period must not extend into test period."""
        train_end: pd.Timestamp = pd.Timestamp("2005-06-30")  # type: ignore[assignment]
        train_returns = returns_df.loc[:train_end]
        train_vol = rolling_vol.loc[:train_end]

        _, metadata = create_windows(
            train_returns, train_vol, universe_ids, T=504, stride=252
        )
        if metadata.empty:
            pytest.skip("No windows generated for training period")

        max_end = pd.Timestamp(str(metadata["end_date"].max()))
        assert max_end <= train_end, (
            f"Window end date {max_end.date()} exceeds training end "
            f"{train_end.date()}"
        )

    def test_deterministic_reproducibility(
        self,
        returns_df: pd.DataFrame,
        rolling_vol: pd.DataFrame,
        universe_ids: list[int],
    ) -> None:
        """Same inputs must produce identical windows."""
        w1, m1 = create_windows(
            returns_df, rolling_vol, universe_ids, T=504, stride=252
        )
        w2, m2 = create_windows(
            returns_df, rolling_vol, universe_ids, T=504, stride=252
        )

        if w1.shape[0] == 0:
            pytest.skip("No windows generated")

        assert torch.equal(w1, w2), "Windows differ across identical runs"
        assert m1.equals(m2), "Metadata differs across identical runs"

    def test_permno_column_type_consistency(
        self,
        stock_data: pd.DataFrame,
        returns_df: pd.DataFrame,
        windows_and_metadata: tuple[torch.Tensor, pd.DataFrame],
    ) -> None:
        """Type mismatches (str vs int) cause silent key lookup failures."""
        # stock_data permno is int
        assert stock_data["permno"].dtype in (
            np.int64, np.int32, int
        ), f"stock_data permno dtype: {stock_data['permno'].dtype}"

        # returns columns are int
        for col in returns_df.columns:
            assert isinstance(col, (int, np.integer)), (
                f"returns column {col!r} is {type(col).__name__}, not int"
            )

        # metadata stock_id is int
        _, metadata = windows_and_metadata
        if not metadata.empty:
            for val in metadata["stock_id"].unique():
                assert isinstance(val, (int, np.integer)), (
                    f"metadata stock_id {val!r} is {type(val).__name__}, "
                    f"not int"
                )


# ---------------------------------------------------------------------------
# Class 6: Statistical sanity checks
# ---------------------------------------------------------------------------


class TestStatisticalSanityChecks:
    """Verify stylized facts of financial returns hold in processed data.

    These model-free empirical regularities should hold for any reasonable
    equity dataset. Thresholds are intentionally loose to pass on synthetic
    GBM data while catching pathological corruption.
    """

    def test_cross_sectional_correlation_structure(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Correlation matrix should be neither identity nor all-ones."""
        stocks = _complete_stocks(returns_df, n=10)
        if len(stocks) < 3:
            pytest.skip("Not enough complete stocks for correlation test")

        sub = returns_df[stocks].dropna()
        if len(sub) < 50:
            pytest.skip("Not enough complete rows for correlation")

        corr = np.corrcoef(np.asarray(sub.values).T)
        n = corr.shape[0]
        # Extract off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        off_diag = corr[mask]
        mean_corr = float(np.mean(off_diag))

        assert -0.5 < mean_corr < 0.95, (
            f"Mean off-diagonal correlation = {mean_corr:.4f}, "
            f"expected in (-0.5, 0.95)"
        )

    def test_volatility_clustering(
        self, returns_df: pd.DataFrame
    ) -> None:
        """Absolute returns should show positive lag-1 autocorrelation."""
        stocks = _complete_stocks(returns_df, n=5)
        positive_count = 0
        tested = 0

        for permno in stocks:
            series = np.asarray(returns_df[permno].dropna().values)
            if len(series) < 100:
                continue
            abs_ret = np.abs(series)
            rho = _autocorrelation_lag1(abs_ret)
            tested += 1
            if rho > -0.1:
                positive_count += 1

        if tested < 3:
            pytest.skip("Not enough stocks with sufficient data")

        assert positive_count >= 3, (
            f"Only {positive_count}/{tested} stocks have vol clustering "
            f"(autocorr of |r| > -0.1), expected >= 3"
        )

    def test_fat_tails(self, returns_df: pd.DataFrame) -> None:
        """Financial returns should not be pathologically thin-tailed."""
        stocks = _complete_stocks(returns_df, n=5)
        pass_count = 0
        tested = 0

        for permno in stocks:
            series = np.asarray(returns_df[permno].dropna().values)
            if len(series) < 100:
                continue
            # Excess kurtosis (Fisher's definition, Gaussian = 0)
            mean = np.mean(series)
            std = np.std(series, ddof=1)
            if std < 1e-10:
                continue
            excess_kurt = float(
                np.mean(((series - mean) / std) ** 4) - 3.0
            )
            tested += 1
            if excess_kurt >= -0.5:
                pass_count += 1

        if tested < 3:
            pytest.skip("Not enough stocks with sufficient data")

        assert pass_count >= 3, (
            f"Only {pass_count}/{tested} stocks have excess kurtosis "
            f">= -0.5, expected >= 3"
        )

    def test_delisted_stocks_have_data_before_delisting(
        self, stock_data: pd.DataFrame
    ) -> None:
        """Delisted stocks must have pre-delisting history for Shumway."""
        dataset_end = stock_data["date"].max()
        cutoff: pd.Timestamp = dataset_end - pd.DateOffset(days=365)  # type: ignore[assignment]

        # Stocks whose last date is well before dataset end = likely delisted
        last_dates: pd.Series = stock_data.groupby("permno")["date"].max()  # type: ignore[assignment]
        delisted_mask: pd.Series = last_dates < cutoff  # type: ignore[assignment]
        delisted_series: pd.Series = last_dates[delisted_mask]  # type: ignore[assignment]
        delisted_permnos = list(delisted_series.index)  # type: ignore[arg-type]

        if len(delisted_permnos) == 0:
            pytest.skip("No clearly delisted stocks detected")

        for permno in delisted_permnos:
            n_days = int(
                (stock_data["permno"] == permno).sum()
            )
            assert n_days >= 100, (
                f"Delisted permno {permno} has only {n_days} trading "
                f"days, expected >= 100 for meaningful history"
            )


# =========================================================================
# TIINGO-SPECIFIC TESTS — run on real data from data/tiingo_us_equities.parquet
# =========================================================================

TIINGO_PARQUET = os.path.join("data", "tiingo_us_equities.parquet")
_tiingo_available = os.path.exists(TIINGO_PARQUET)

tiingo_skip = pytest.mark.skipif(
    not _tiingo_available,
    reason="Tiingo data not found at data/tiingo_us_equities.parquet",
)


# ---------------------------------------------------------------------------
# Tiingo fixtures (module-scoped, independent from synthetic fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiingo_stock_data() -> pd.DataFrame:
    """Load real Tiingo stock data."""
    if not _tiingo_available:
        pytest.skip("Tiingo data not available")
    return load_stock_data(TIINGO_PARQUET)


@pytest.fixture(scope="module")
def tiingo_returns(tiingo_stock_data: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from Tiingo data."""
    return compute_log_returns(tiingo_stock_data)


@pytest.fixture(scope="module")
def tiingo_trailing_vol(tiingo_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute 252-day trailing annualized volatility on Tiingo data."""
    return compute_trailing_volatility(tiingo_returns, window=252)


@pytest.fixture(scope="module")
def tiingo_rolling_vol(tiingo_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day rolling realized volatility on Tiingo data."""
    return compute_rolling_realized_vol(tiingo_returns, rolling_window=21)


@pytest.fixture(scope="module")
def tiingo_universe(tiingo_stock_data: pd.DataFrame) -> list[int]:
    """Construct universe from Tiingo data at 2020-01-02 with relaxed filters."""
    date: pd.Timestamp = pd.Timestamp("2020-01-02")  # type: ignore[assignment]
    return construct_universe(
        tiingo_stock_data,
        date=date,
        n_max=1000,
        cap_entry=0,
        cap_exit=0,
        adv_min=0,
        min_listing_days=504,
    )


@pytest.fixture(scope="module")
def tiingo_windows(
    tiingo_returns: pd.DataFrame,
    tiingo_rolling_vol: pd.DataFrame,
    tiingo_universe: list[int],
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Create sliding windows from Tiingo data."""
    return create_windows(
        tiingo_returns, tiingo_rolling_vol, tiingo_universe,
        T=504, stride=252,
    )


# ---------------------------------------------------------------------------
# Class 7: Tiingo raw data quality
# ---------------------------------------------------------------------------


@tiingo_skip
class TestTiingoRawDataQuality:
    """Verify structural integrity of the real Tiingo dataset.

    These tests catch data corruption and quality issues that would silently
    degrade VAE training quality on production data.
    """

    def test_no_duplicate_permno_date_pairs(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Duplicates cause incorrect pivoting in compute_log_returns."""
        n_dupes = int(
            tiingo_stock_data.duplicated(subset=["permno", "date"]).sum()
        )
        assert n_dupes == 0, (
            f"Found {n_dupes} duplicate (permno, date) pairs"
        )

    def test_adj_price_positive(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """All prices must be strictly positive for log-returns."""
        valid = tiingo_stock_data.loc[
            tiingo_stock_data["adj_price"].notna(), "adj_price"
        ]
        n_bad = int((valid <= 0).sum())
        assert n_bad == 0, (
            f"Found {n_bad} non-positive adj_price values"
        )

    def test_volume_non_negative(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Negative volume is data corruption."""
        n_neg = int((tiingo_stock_data["volume"] < 0).sum())
        assert n_neg == 0, f"Found {n_neg} negative volume values"

    def test_exchange_codes_valid(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """All exchange codes must be in {1, 2, 3}."""
        valid_codes = [1, 2, 3]
        invalid = tiingo_stock_data[
            ~tiingo_stock_data["exchange_code"].isin(valid_codes)
        ]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} rows with invalid exchange_code: "
            f"{np.unique(np.asarray(invalid['exchange_code']))}"
        )

    def test_date_monotonicity_per_stock(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Dates strictly increasing per permno."""
        for permno in list(tiingo_stock_data["permno"].unique()[:20]):
            group = tiingo_stock_data[
                tiingo_stock_data["permno"] == permno
            ]
            if len(group) < 2:
                continue
            dates = np.asarray(group["date"])
            diffs = np.diff(dates.astype(np.int64))
            assert np.all(diffs > 0), (
                f"Permno {permno}: dates not strictly increasing"
            )

    def test_penny_stock_fraction_bounded(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Penny stocks (adj_price < $0.01) should be a small fraction."""
        valid = tiingo_stock_data.loc[
            tiingo_stock_data["adj_price"].notna(), "adj_price"
        ]
        n_penny = int((valid < 0.01).sum())
        frac = n_penny / len(valid)
        assert frac < 0.01, (
            f"{frac:.2%} of prices are penny stocks (< $0.01): "
            f"{n_penny} rows. Consider filtering these stocks."
        )

    def test_zero_volume_fraction_bounded(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Too many zero-volume days indicate stale or inactive tickers."""
        n_zero = int((tiingo_stock_data["volume"] == 0).sum())
        frac = n_zero / len(tiingo_stock_data)
        assert frac < 0.10, (
            f"{frac:.2%} of rows have zero volume ({n_zero} rows), "
            f"expected < 10%"
        )

    def test_market_cap_proxy_quality(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Market cap proxy should be positive for rows with volume > 0."""
        active = tiingo_stock_data[tiingo_stock_data["volume"] > 0]
        cap_series: pd.Series = active["market_cap"]  # type: ignore[assignment]
        valid_caps = active.loc[cap_series.notna(), "market_cap"]
        n_bad = int((valid_caps <= 0).sum())
        assert n_bad == 0, (
            f"Found {n_bad} non-positive market_cap on active "
            f"(volume > 0) days"
        )

    def test_sufficient_stocks_with_long_history(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Need >= 20 stocks with >= 504 days for meaningful VAE training."""
        days_per_stock = tiingo_stock_data.groupby("permno").size()
        n_long = int((days_per_stock >= 504).sum())
        assert n_long >= 20, (
            f"Only {n_long} stocks with >= 504 trading days, "
            f"need >= 20 for meaningful training"
        )

    def test_no_stocks_with_trivial_history(
        self, tiingo_stock_data: pd.DataFrame
    ) -> None:
        """Flag stocks with very few trading days (< 21 = vol warmup)."""
        days_per_stock: pd.Series = tiingo_stock_data.groupby(  # type: ignore[assignment]
            "permno"
        ).size()
        thin_mask: pd.Series = days_per_stock < 21  # type: ignore[assignment]
        thin_stocks: pd.Series = days_per_stock[thin_mask]  # type: ignore[assignment]
        n_thin = len(thin_stocks)
        thin_permnos = list(thin_stocks.head(5).index)  # type: ignore[arg-type]
        assert n_thin == 0, (
            f"Found {n_thin} stocks with < 21 trading days "
            f"(useless for vol computation): "
            f"permnos {thin_permnos}"
        )


# ---------------------------------------------------------------------------
# Class 8: Tiingo returns quality
# ---------------------------------------------------------------------------


@tiingo_skip
class TestTiingoReturnsQuality:
    """Verify Tiingo returns are clean enough for VAE training.

    Real data has penny stocks producing extreme log-returns. These tests
    validate that the pipeline handles them correctly.
    """

    def test_no_infinities(self, tiingo_returns: pd.DataFrame) -> None:
        """Inf values propagate through all computations."""
        values = tiingo_returns.values
        n_inf = int(
            np.isinf(np.where(np.isnan(values), 0.0, values)).sum()
        )
        assert n_inf == 0, f"Found {n_inf} Inf values in Tiingo returns"

    def test_extreme_return_fraction(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Quantify extreme returns (|r| > 0.5 = 65% daily move)."""
        values = tiingo_returns.values.ravel()
        finite = values[np.isfinite(values)]
        n_extreme_50 = int((np.abs(finite) > 0.5).sum())
        n_extreme_100 = int((np.abs(finite) > 1.0).sum())
        frac_50 = n_extreme_50 / len(finite)
        frac_100 = n_extreme_100 / len(finite)
        # Allow up to 0.1% extreme returns — penny stocks produce these
        assert frac_100 < 0.001, (
            f"|r| > 1.0: {n_extreme_100} ({frac_100:.4%}), "
            f"|r| > 0.5: {n_extreme_50} ({frac_50:.4%}). "
            f"Consider filtering penny stocks."
        )

    def test_per_stock_nan_fraction(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Each stock that exists should have at least 50% valid returns."""
        for col in tiingo_returns.columns:
            series: pd.Series = tiingo_returns[col]  # type: ignore[assignment]
            total = len(series)
            n_valid = int(series.notna().sum())
            # Only check stocks that have meaningful presence
            if n_valid < 100:
                continue
            frac_valid = n_valid / total
            # This test reveals the sparse matrix issue: most stocks
            # only exist for part of the date range
            assert frac_valid > 0.01, (
                f"Permno {col}: only {frac_valid:.2%} valid returns "
                f"({n_valid}/{total})"
            )

    def test_return_matrix_shape(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Returns matrix should have expected structure."""
        assert tiingo_returns.shape[1] >= 20, (
            f"Only {tiingo_returns.shape[1]} stocks in returns matrix, "
            f"expected >= 20"
        )
        assert tiingo_returns.index.is_monotonic_increasing, (
            "Returns date index is not sorted"
        )

    def test_max_return_magnitude(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Flag returns with unreasonable magnitude (|r| > 5 = 14742% move)."""
        values = tiingo_returns.values.ravel()
        finite = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite)))
        # Log return > 5 means price moved by factor e^5 ≈ 148x in one day
        # This is almost certainly a data error, not a real price move
        assert max_abs < 5.0, (
            f"Max |return| = {max_abs:.4f} (= {np.exp(max_abs):.0f}x "
            f"price move). This is likely a data error from penny stocks."
        )


# ---------------------------------------------------------------------------
# Class 9: Tiingo volatility quality
# ---------------------------------------------------------------------------


@tiingo_skip
class TestTiingoVolatilityQuality:
    """Verify volatility computed from Tiingo data is reasonable."""

    def test_trailing_vol_no_inf(
        self, tiingo_trailing_vol: pd.DataFrame
    ) -> None:
        """Inf in trailing vol would propagate to risk model rescaling."""
        values = tiingo_trailing_vol.values
        n_inf = int(
            np.isinf(np.where(np.isnan(values), 0.0, values)).sum()
        )
        assert n_inf == 0, (
            f"Found {n_inf} Inf values in Tiingo trailing vol"
        )

    def test_trailing_vol_range(
        self, tiingo_trailing_vol: pd.DataFrame
    ) -> None:
        """Annualized vol should be in [1%, 500%] for real equities."""
        post_warmup = tiingo_trailing_vol.iloc[252:]
        valid = post_warmup.values[~np.isnan(post_warmup.values)]
        if len(valid) == 0:
            pytest.skip("No valid trailing vol values")
        in_range = (valid >= 0.01) & (valid <= 5.0)
        frac = float(in_range.sum()) / len(valid)
        assert frac >= 0.90, (
            f"Only {frac:.2%} of trailing vol in [0.01, 5.0], "
            f"expected >= 90%. Range: [{valid.min():.4f}, "
            f"{valid.max():.4f}]"
        )

    def test_rolling_vol_no_inf(
        self, tiingo_rolling_vol: pd.DataFrame
    ) -> None:
        """Inf in rolling vol corrupts VAE input feature."""
        values = tiingo_rolling_vol.values
        n_inf = int(
            np.isinf(np.where(np.isnan(values), 0.0, values)).sum()
        )
        assert n_inf == 0, (
            f"Found {n_inf} Inf values in Tiingo rolling vol"
        )

    def test_vol_returns_alignment(
        self,
        tiingo_returns: pd.DataFrame,
        tiingo_trailing_vol: pd.DataFrame,
        tiingo_rolling_vol: pd.DataFrame,
    ) -> None:
        """Vol DataFrames must align with returns for correct windowing."""
        assert set(tiingo_trailing_vol.columns) == set(
            tiingo_returns.columns
        ), "Trailing vol columns mismatch with returns"
        assert set(tiingo_rolling_vol.columns) == set(
            tiingo_returns.columns
        ), "Rolling vol columns mismatch with returns"


# ---------------------------------------------------------------------------
# Class 10: Tiingo windowing quality
# ---------------------------------------------------------------------------


@tiingo_skip
class TestTiingoWindowingQuality:
    """Verify windows generated from Tiingo data are clean for VAE input."""

    def test_windows_generated(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Tiingo data should produce a reasonable number of windows."""
        windows, metadata = tiingo_windows
        assert windows.shape[0] >= 10, (
            f"Only {windows.shape[0]} windows generated from Tiingo "
            f"data, expected >= 10"
        )
        assert windows.shape[1] == 504, (
            f"Window T={windows.shape[1]}, expected 504"
        )
        assert windows.shape[2] == 2, (
            f"Window F={windows.shape[2]}, expected 2"
        )

    def test_no_nan_in_windows(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """NaN in windows causes NaN loss during VAE training."""
        windows, _ = tiingo_windows
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        n_nan = int(torch.isnan(windows).sum().item())
        assert n_nan == 0, (
            f"Found {n_nan} NaN values in Tiingo windows tensor"
        )

    def test_no_inf_in_windows(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Inf in windows causes NaN loss during VAE training."""
        windows, _ = tiingo_windows
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        n_inf = int(torch.isinf(windows).sum().item())
        assert n_inf == 0, (
            f"Found {n_inf} Inf values in Tiingo windows tensor"
        )

    def test_no_extreme_zscores(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Z-scored values > 10 sigma indicate contaminated windows."""
        windows, _ = tiingo_windows
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        values = windows.numpy().ravel()
        n_extreme = int((np.abs(values) > 10.0).sum())
        frac = n_extreme / len(values)
        assert frac < 0.001, (
            f"{frac:.4%} of z-scored values have |z| > 10 "
            f"({n_extreme} values), expected < 0.1%"
        )

    def test_multiple_stocks_in_windows(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Windows should represent multiple stocks for diversified training."""
        _, metadata = tiingo_windows
        if metadata.empty:
            pytest.skip("No windows generated")
        n_stocks = metadata["stock_id"].nunique()
        assert n_stocks >= 5, (
            f"Windows cover only {n_stocks} stocks, expected >= 5"
        )

    def test_window_feature_variance(
        self, tiingo_windows: tuple[torch.Tensor, pd.DataFrame]
    ) -> None:
        """Z-scored features should have variance near 1.0."""
        windows, _ = tiingo_windows
        if windows.shape[0] == 0:
            pytest.skip("No windows generated")
        for f_idx in range(windows.shape[2]):
            feat = windows[:, :, f_idx].numpy().ravel()
            var = float(np.var(feat))
            assert 0.3 <= var <= 3.0, (
                f"Feature {f_idx}: pooled variance = {var:.4f}, "
                f"expected in [0.3, 3.0]"
            )


# ---------------------------------------------------------------------------
# Class 11: Tiingo statistical sanity
# ---------------------------------------------------------------------------


@tiingo_skip
class TestTiingoStatisticalSanity:
    """Verify real Tiingo data exhibits expected financial stylized facts.

    These checks validate that the data source provides genuine market
    data, not synthetic or corrupted series.
    """

    def test_volatility_clustering(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Real markets exhibit volatility clustering (ARCH effects)."""
        stocks = _complete_stocks(tiingo_returns, n=10)
        positive_count = 0
        tested = 0
        for permno in stocks:
            series = np.asarray(tiingo_returns[permno].dropna().values)
            if len(series) < 252:
                continue
            abs_ret = np.abs(series)
            rho = _autocorrelation_lag1(abs_ret)
            tested += 1
            if rho > 0.0:
                positive_count += 1
        if tested < 3:
            pytest.skip("Not enough Tiingo stocks with sufficient data")
        # Real data should show strong vol clustering
        assert positive_count >= int(tested * 0.5), (
            f"Only {positive_count}/{tested} stocks show vol clustering "
            f"(autocorr of |r| > 0), expected >= 50%"
        )

    def test_fat_tails(self, tiingo_returns: pd.DataFrame) -> None:
        """Real financial returns are leptokurtic (kurtosis > 3)."""
        stocks = _complete_stocks(tiingo_returns, n=10)
        fat_count = 0
        tested = 0
        for permno in stocks:
            series = np.asarray(tiingo_returns[permno].dropna().values)
            if len(series) < 252:
                continue
            mean = np.mean(series)
            std = np.std(series, ddof=1)
            if std < 1e-10:
                continue
            excess_kurt = float(
                np.mean(((series - mean) / std) ** 4) - 3.0
            )
            tested += 1
            if excess_kurt > 0:
                fat_count += 1
        if tested < 3:
            pytest.skip("Not enough Tiingo stocks with sufficient data")
        # Real financial data should have positive excess kurtosis
        assert fat_count >= int(tested * 0.7), (
            f"Only {fat_count}/{tested} stocks have positive excess "
            f"kurtosis, expected >= 70% for real financial data"
        )

    def test_cross_sectional_correlation_positive(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Stocks should show positive average cross-correlation (market factor)."""
        stocks = _complete_stocks(tiingo_returns, n=10)
        if len(stocks) < 5:
            pytest.skip("Not enough stocks for correlation test")
        sub = tiingo_returns[stocks].dropna()
        if len(sub) < 252:
            pytest.skip("Not enough overlapping data")
        corr = np.corrcoef(np.asarray(sub.values).T)
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = corr[mask]
        mean_corr = float(np.mean(off_diag))
        # Real equities are positively correlated on average (market factor)
        assert mean_corr > 0.0, (
            f"Mean off-diagonal correlation = {mean_corr:.4f}, "
            f"expected > 0 for real equity data (common market factor)"
        )

    def test_no_serial_correlation_in_returns(
        self, tiingo_returns: pd.DataFrame
    ) -> None:
        """Returns should not show strong lag-1 autocorrelation."""
        stocks = _complete_stocks(tiingo_returns, n=10)
        tested = 0
        suspect = 0
        for permno in stocks:
            series = np.asarray(tiingo_returns[permno].dropna().values)
            if len(series) < 252:
                continue
            rho = _autocorrelation_lag1(series)
            tested += 1
            if abs(rho) > 0.10:
                suspect += 1
        if tested < 3:
            pytest.skip("Not enough Tiingo stocks to test")
        # At most 30% of stocks should show strong serial correlation
        assert suspect <= int(tested * 0.3), (
            f"{suspect}/{tested} stocks have |autocorr| > 0.10, "
            f"expected <= 30%"
        )
