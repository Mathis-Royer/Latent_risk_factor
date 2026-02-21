"""
Unit tests for centralized validation utilities.

Tests assert_* functions from src/validation.py.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.validation import (
    assert_finite_2d,
    assert_positive_semidefinite,
    assert_no_nan_tensor,
    assert_no_inf_tensor,
    assert_finite_tensor,
    assert_no_duplicate_ids,
    assert_shape_match,
    assert_column_exists,
    assert_weights_sum_to_one,
    assert_non_negative_eigenvalues,
    assert_bounds,
    assert_monotonic_dates,
    assert_condition_number,
    assert_date_alignment,
    assert_stock_id_alignment,
    assert_alignment_by_id,
    assert_matrix_square,
    assert_covariance_valid,
    assert_positive_definite,
    assert_returns_valid,
    assert_weights_valid,
    assert_tensor_shape,
    assert_fold_consistency,
    assert_no_lookahead,
    warn_if_nan_fraction_exceeds,
    # Phase 2 validation functions
    assert_non_empty_dataframe,
    assert_tensor_bounds,
    warn_if_tensor_extreme,
    assert_embargo_date_in_index,
    assert_growth_finite,
    warn_loss_explosion,
    assert_active_units_valid,
    assert_ann_vol_positive,
)


# ---------------------------------------------------------------------------
# assert_finite_2d tests
# ---------------------------------------------------------------------------

class TestAssertFinite2D:
    """Tests for assert_finite_2d."""

    def test_valid_2d_array_passes(self):
        """Valid 2D finite array should pass."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_finite_2d(arr, "test_array")  # Should not raise

    def test_1d_array_fails(self):
        """1D array should fail."""
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError, match="must be 2D"):
            assert_finite_2d(arr, "test_array")

    def test_3d_array_fails(self):
        """3D array should fail."""
        arr = np.ones((2, 3, 4))
        with pytest.raises(AssertionError, match="must be 2D"):
            assert_finite_2d(arr, "test_array")

    def test_nan_fails(self):
        """Array with NaN should fail."""
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(AssertionError, match="contains NaN/Inf"):
            assert_finite_2d(arr, "test_array")

    def test_inf_fails(self):
        """Array with Inf should fail."""
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(AssertionError, match="contains NaN/Inf"):
            assert_finite_2d(arr, "test_array")

    def test_negative_inf_fails(self):
        """Array with -Inf should fail."""
        arr = np.array([[1.0, -np.inf], [3.0, 4.0]])
        with pytest.raises(AssertionError, match="contains NaN/Inf"):
            assert_finite_2d(arr, "test_array")


# ---------------------------------------------------------------------------
# assert_positive_semidefinite tests
# ---------------------------------------------------------------------------

class TestAssertPositiveSemidefinite:
    """Tests for assert_positive_semidefinite."""

    def test_identity_matrix_passes(self):
        """Identity matrix is PSD."""
        Sigma = np.eye(3)
        assert_positive_semidefinite(Sigma, "covariance")  # Should not raise

    def test_correlation_matrix_passes(self):
        """Valid correlation matrix is PSD."""
        Sigma = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        assert_positive_semidefinite(Sigma, "correlation")  # Should not raise

    def test_negative_definite_fails(self):
        """Matrix with negative eigenvalues should fail."""
        # -I has all eigenvalues = -1
        Sigma = -np.eye(3)
        with pytest.raises(AssertionError, match="not PSD"):
            assert_positive_semidefinite(Sigma, "bad_matrix")

    def test_nearly_singular_passes(self):
        """Nearly singular but PSD matrix should pass."""
        Sigma = np.array([
            [1.0, 0.9999],
            [0.9999, 1.0],
        ])
        # This has eigenvalues ~0.0001 and ~1.9999
        assert_positive_semidefinite(Sigma, "near_singular")  # Should not raise

    def test_custom_tolerance(self):
        """Custom tolerance should work."""
        # Matrix with small negative eigenvalue
        Sigma = np.array([
            [1.0, 1.0001],
            [1.0001, 1.0],
        ])
        # This has a small negative eigenvalue
        with pytest.raises(AssertionError, match="not PSD"):
            assert_positive_semidefinite(Sigma, "test", tol=0.0)


# ---------------------------------------------------------------------------
# assert_no_nan_tensor tests
# ---------------------------------------------------------------------------

class TestAssertNoNanTensor:
    """Tests for assert_no_nan_tensor."""

    def test_valid_tensor_passes(self):
        """Tensor without NaN should pass."""
        t = torch.tensor([1.0, 2.0, 3.0])
        assert_no_nan_tensor(t, "test_tensor")  # Should not raise

    def test_nan_tensor_fails(self):
        """Tensor with NaN should fail."""
        t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="contains NaN"):
            assert_no_nan_tensor(t, "test_tensor")

    def test_inf_tensor_passes(self):
        """Tensor with Inf (but no NaN) should pass."""
        t = torch.tensor([1.0, float("inf"), 3.0])
        assert_no_nan_tensor(t, "test_tensor")  # Should not raise (no NaN check)

    def test_multidimensional_tensor(self):
        """Test with multidimensional tensor."""
        t = torch.ones((3, 4, 5))
        assert_no_nan_tensor(t, "test_tensor")  # Should not raise


# ---------------------------------------------------------------------------
# assert_no_inf_tensor tests
# ---------------------------------------------------------------------------

class TestAssertNoInfTensor:
    """Tests for assert_no_inf_tensor."""

    def test_valid_tensor_passes(self):
        """Tensor without Inf should pass."""
        t = torch.tensor([1.0, 2.0, 3.0])
        assert_no_inf_tensor(t, "test_tensor")  # Should not raise

    def test_inf_tensor_fails(self):
        """Tensor with Inf should fail."""
        t = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(AssertionError, match="contains Inf"):
            assert_no_inf_tensor(t, "test_tensor")

    def test_negative_inf_fails(self):
        """Tensor with -Inf should fail."""
        t = torch.tensor([1.0, float("-inf"), 3.0])
        with pytest.raises(AssertionError, match="contains Inf"):
            assert_no_inf_tensor(t, "test_tensor")


# ---------------------------------------------------------------------------
# assert_finite_tensor tests
# ---------------------------------------------------------------------------

class TestAssertFiniteTensor:
    """Tests for assert_finite_tensor."""

    def test_valid_tensor_passes(self):
        """Finite tensor should pass."""
        t = torch.tensor([1.0, 2.0, 3.0])
        assert_finite_tensor(t, "test_tensor")  # Should not raise

    def test_nan_fails(self):
        """Tensor with NaN should fail."""
        t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="contains NaN/Inf"):
            assert_finite_tensor(t, "test_tensor")

    def test_inf_fails(self):
        """Tensor with Inf should fail."""
        t = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(AssertionError, match="contains NaN/Inf"):
            assert_finite_tensor(t, "test_tensor")


# ---------------------------------------------------------------------------
# assert_no_duplicate_ids tests
# ---------------------------------------------------------------------------

class TestAssertNoDuplicateIds:
    """Tests for assert_no_duplicate_ids."""

    def test_unique_list_passes(self):
        """List with unique IDs should pass."""
        ids = [1, 2, 3, 4, 5]
        assert_no_duplicate_ids(ids, "stock_ids")  # Should not raise

    def test_duplicate_list_fails(self):
        """List with duplicates should fail."""
        ids = [1, 2, 3, 2, 5]  # 2 appears twice
        with pytest.raises(AssertionError, match="Duplicate IDs"):
            assert_no_duplicate_ids(ids, "stock_ids")

    def test_unique_array_passes(self):
        """NumPy array with unique IDs should pass."""
        ids = np.array([10, 20, 30, 40])
        assert_no_duplicate_ids(ids, "permnos")  # Should not raise

    def test_duplicate_array_fails(self):
        """NumPy array with duplicates should fail."""
        ids = np.array([10, 20, 30, 20])  # 20 appears twice
        with pytest.raises(AssertionError, match="Duplicate IDs"):
            assert_no_duplicate_ids(ids, "permnos")

    def test_empty_list_passes(self):
        """Empty list should pass."""
        ids: list[int] = []
        assert_no_duplicate_ids(ids, "empty")  # Should not raise


# ---------------------------------------------------------------------------
# assert_shape_match tests
# ---------------------------------------------------------------------------

class TestAssertShapeMatch:
    """Tests for assert_shape_match."""

    def test_matching_shapes_pass(self):
        """Arrays with same shape should pass."""
        arr1 = np.ones((3, 4))
        arr2 = np.zeros((3, 4))
        assert_shape_match(arr1, arr2, "returns", "volatility")  # Should not raise

    def test_mismatched_shapes_fail(self):
        """Arrays with different shapes should fail."""
        arr1 = np.ones((3, 4))
        arr2 = np.zeros((3, 5))
        with pytest.raises(AssertionError, match="shape"):
            assert_shape_match(arr1, arr2, "returns", "volatility")

    def test_1d_shapes(self):
        """1D arrays with same shape should pass."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert_shape_match(arr1, arr2, "a", "b")  # Should not raise


# ---------------------------------------------------------------------------
# assert_column_exists tests
# ---------------------------------------------------------------------------

class TestAssertColumnExists:
    """Tests for assert_column_exists."""

    def test_existing_column_passes(self):
        """Existing column should pass."""
        df = pd.DataFrame({"date": [1, 2], "stock_id": [10, 20]})
        assert_column_exists(df, "stock_id", "metadata")  # Should not raise

    def test_missing_column_fails(self):
        """Missing column should fail."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 200]})
        with pytest.raises(AssertionError, match="must have 'stock_id'"):
            assert_column_exists(df, "stock_id", "metadata")


# ---------------------------------------------------------------------------
# assert_weights_sum_to_one tests
# ---------------------------------------------------------------------------

class TestAssertWeightsSumToOne:
    """Tests for assert_weights_sum_to_one."""

    def test_equal_weights_pass(self):
        """Equal weights summing to 1 should pass."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert_weights_sum_to_one(w, "portfolio")  # Should not raise

    def test_arbitrary_weights_pass(self):
        """Arbitrary weights summing to 1 should pass."""
        w = np.array([0.5, 0.3, 0.15, 0.05])
        assert_weights_sum_to_one(w, "portfolio")  # Should not raise

    def test_weights_not_summing_to_one_fail(self):
        """Weights not summing to 1 should fail."""
        w = np.array([0.5, 0.3, 0.1])  # sum = 0.9
        with pytest.raises(AssertionError, match="do not sum to 1"):
            assert_weights_sum_to_one(w, "portfolio")

    def test_tolerance(self):
        """Weights within tolerance should pass."""
        w = np.array([0.5, 0.5001])  # sum = 1.0001
        assert_weights_sum_to_one(w, "portfolio", tol=1e-3)  # Should pass

        w2 = np.array([0.5, 0.51])  # sum = 1.01
        with pytest.raises(AssertionError):
            assert_weights_sum_to_one(w2, "portfolio", tol=1e-3)


# ---------------------------------------------------------------------------
# assert_non_negative_eigenvalues tests
# ---------------------------------------------------------------------------

class TestAssertNonNegativeEigenvalues:
    """Tests for assert_non_negative_eigenvalues."""

    def test_positive_eigenvalues_pass(self):
        """Positive eigenvalues should pass."""
        eigenvalues = np.array([3.0, 2.0, 1.0, 0.5])
        assert_non_negative_eigenvalues(eigenvalues, "Sigma_z")  # Should not raise

    def test_zero_eigenvalue_passes(self):
        """Zero eigenvalue should pass."""
        eigenvalues = np.array([3.0, 2.0, 1.0, 0.0])
        assert_non_negative_eigenvalues(eigenvalues, "Sigma_z")  # Should not raise

    def test_negative_eigenvalue_fails(self):
        """Negative eigenvalue should fail."""
        eigenvalues = np.array([3.0, 2.0, -0.1])
        with pytest.raises(AssertionError, match="Negative eigenvalues"):
            assert_non_negative_eigenvalues(eigenvalues, "Sigma_z")

    def test_tolerance(self):
        """Small negative within tolerance should pass."""
        eigenvalues = np.array([3.0, 2.0, -1e-12])
        assert_non_negative_eigenvalues(eigenvalues, "Sigma_z", tol=-1e-10)  # Should pass


# ---------------------------------------------------------------------------
# assert_bounds tests
# ---------------------------------------------------------------------------

class TestAssertBounds:
    """Tests for assert_bounds."""

    def test_values_within_bounds_pass(self):
        """Values within bounds should pass."""
        arr = np.array([0.5, 1.0, 1.5])
        assert_bounds(arr, 0.0, 2.0, "test")

    def test_values_below_low_fail(self):
        """Values below low bound should fail."""
        arr = np.array([-0.5, 1.0, 1.5])
        with pytest.raises(AssertionError, match="out of bounds"):
            assert_bounds(arr, 0.0, 2.0, "test")

    def test_values_above_high_fail(self):
        """Values above high bound should fail."""
        arr = np.array([0.5, 1.0, 2.5])
        with pytest.raises(AssertionError, match="out of bounds"):
            assert_bounds(arr, 0.0, 2.0, "test")

    def test_exact_boundary_passes(self):
        """Values exactly at bounds should pass."""
        arr = np.array([0.0, 1.0, 2.0])
        assert_bounds(arr, 0.0, 2.0, "test")

    def test_nan_values_ignored(self):
        """NaN values should be ignored by nanmin/nanmax."""
        arr = np.array([0.5, np.nan, 1.5])
        assert_bounds(arr, 0.0, 2.0, "test")


# ---------------------------------------------------------------------------
# assert_monotonic_dates tests
# ---------------------------------------------------------------------------

class TestAssertMonotonicDates:
    """Tests for assert_monotonic_dates."""

    def test_sorted_dates_pass(self):
        """Monotonically increasing dates should pass."""
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
        assert_monotonic_dates(dates, "trade_dates")

    def test_unsorted_dates_fail(self):
        """Non-monotonic dates should fail."""
        dates = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
        with pytest.raises(AssertionError, match="not monotonically increasing"):
            assert_monotonic_dates(dates, "trade_dates")

    def test_duplicate_dates_pass_non_strict(self):
        """Duplicate dates should pass in non-strict mode."""
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        assert_monotonic_dates(dates, "trade_dates", strict=False)

    def test_duplicate_dates_fail_strict(self):
        """Duplicate dates should fail in strict mode."""
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        with pytest.raises(AssertionError, match="not strictly monotonic"):
            assert_monotonic_dates(dates, "trade_dates", strict=True)

    def test_single_date_passes(self):
        """Single date should pass."""
        dates = pd.DatetimeIndex(["2020-01-01"])
        assert_monotonic_dates(dates, "trade_dates", strict=True)


# ---------------------------------------------------------------------------
# assert_condition_number tests
# ---------------------------------------------------------------------------

class TestAssertConditionNumber:
    """Tests for assert_condition_number."""

    def test_identity_low_condition(self):
        """Identity matrix has condition number 1."""
        Sigma = np.eye(3)
        assert_condition_number(Sigma, 10.0, "Sigma")

    def test_high_condition_fails(self):
        """Ill-conditioned matrix should fail."""
        Sigma = np.diag([1e6, 1e-6])
        with pytest.raises(AssertionError, match="condition number"):
            assert_condition_number(Sigma, 1e6, "Sigma")

    def test_singular_matrix_fails(self):
        """Singular matrix has infinite condition number."""
        Sigma = np.array([[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(AssertionError, match="condition number"):
            assert_condition_number(Sigma, 1e10, "Sigma")

    def test_well_conditioned_passes(self):
        """Well-conditioned diagonal matrix should pass."""
        Sigma = np.diag([2.0, 1.0, 1.5])
        assert_condition_number(Sigma, 10.0, "Sigma")


# ---------------------------------------------------------------------------
# assert_date_alignment tests
# ---------------------------------------------------------------------------

class TestAssertDateAlignment:
    """Tests for assert_date_alignment."""

    def test_aligned_dates_pass(self):
        """DataFrames with same date index should pass."""
        idx = pd.date_range("2020-01-01", periods=3)
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        df2 = pd.DataFrame({"b": [4, 5, 6]}, index=idx)
        assert_date_alignment(df1, df2, "returns", "vol")

    def test_misaligned_dates_fail(self):
        """DataFrames with different date indices should fail."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
        df2 = pd.DataFrame({"b": [3, 4]}, index=pd.date_range("2020-01-03", periods=2))
        with pytest.raises(AssertionError, match="misaligned dates"):
            assert_date_alignment(df1, df2, "returns", "vol")

    def test_different_length_fails(self):
        """DataFrames with different lengths should fail."""
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        df2 = pd.DataFrame({"b": [4, 5]}, index=pd.date_range("2020-01-01", periods=2))
        with pytest.raises(AssertionError, match="misaligned dates"):
            assert_date_alignment(df1, df2, "returns", "vol")

    def test_empty_dataframes_pass(self):
        """Empty DataFrames with same empty index should pass."""
        df1 = pd.DataFrame(index=pd.DatetimeIndex([]))
        df2 = pd.DataFrame(index=pd.DatetimeIndex([]))
        assert_date_alignment(df1, df2, "a", "b")


# ---------------------------------------------------------------------------
# assert_stock_id_alignment tests
# ---------------------------------------------------------------------------

class TestAssertStockIdAlignment:
    """Tests for assert_stock_id_alignment."""

    def test_matching_ids_pass(self):
        """Identical ID sets should pass."""
        ids1 = [10, 20, 30]
        ids2 = [30, 10, 20]
        assert_stock_id_alignment(ids1, ids2, "B", "returns")

    def test_mismatched_ids_fail(self):
        """Different ID sets should fail."""
        ids1 = [10, 20, 30]
        ids2 = [10, 20, 40]
        with pytest.raises(AssertionError, match="misaligned stock IDs"):
            assert_stock_id_alignment(ids1, ids2, "B", "returns")

    def test_subset_fails(self):
        """Subset relationship should still fail."""
        ids1 = [10, 20]
        ids2 = [10, 20, 30]
        with pytest.raises(AssertionError, match="misaligned stock IDs"):
            assert_stock_id_alignment(ids1, ids2, "B", "returns")

    def test_numpy_arrays_pass(self):
        """NumPy arrays with matching IDs should pass."""
        ids1 = np.array([1, 2, 3])
        ids2 = np.array([3, 2, 1])
        assert_stock_id_alignment(ids1, ids2, "a", "b")


# ---------------------------------------------------------------------------
# assert_alignment_by_id tests
# ---------------------------------------------------------------------------

class TestAssertAlignmentById:
    """Tests for assert_alignment_by_id."""

    def test_valid_subset_passes(self):
        """Array IDs that are subset of ref should pass."""
        arr = np.ones((3, 5))
        ids = [10, 20, 30]
        ref_ids = [10, 20, 30, 40, 50]
        assert_alignment_by_id(arr, ids, ref_ids, "B_A")

    def test_ids_not_in_ref_fail(self):
        """IDs not in reference set should fail."""
        arr = np.ones((3, 5))
        ids = [10, 20, 99]
        ref_ids = [10, 20, 30]
        with pytest.raises(AssertionError, match="not in reference set"):
            assert_alignment_by_id(arr, ids, ref_ids, "B_A")

    def test_length_mismatch_fails(self):
        """IDs length not matching array rows should fail."""
        arr = np.ones((3, 5))
        ids = [10, 20]
        ref_ids = [10, 20, 30]
        with pytest.raises(AssertionError, match="IDs length"):
            assert_alignment_by_id(arr, ids, ref_ids, "B_A")

    def test_exact_match_passes(self):
        """Exact match of IDs and ref should pass."""
        arr = np.ones((2, 3))
        ids = [1, 2]
        ref_ids = [1, 2]
        assert_alignment_by_id(arr, ids, ref_ids, "test")


# ---------------------------------------------------------------------------
# assert_matrix_square tests
# ---------------------------------------------------------------------------

class TestAssertMatrixSquare:
    """Tests for assert_matrix_square."""

    def test_square_matrix_passes(self):
        """Square matrix should pass."""
        arr = np.eye(4)
        assert_matrix_square(arr, "Sigma")

    def test_rectangular_fails(self):
        """Rectangular matrix should fail."""
        arr = np.ones((3, 5))
        with pytest.raises(AssertionError, match="must be square"):
            assert_matrix_square(arr, "Sigma")

    def test_1d_fails(self):
        """1D array should fail."""
        arr = np.array([1, 2, 3])
        with pytest.raises(AssertionError, match="must be 2D"):
            assert_matrix_square(arr, "Sigma")

    def test_1x1_passes(self):
        """1x1 matrix should pass."""
        arr = np.array([[5.0]])
        assert_matrix_square(arr, "Sigma")


# ---------------------------------------------------------------------------
# assert_covariance_valid tests
# ---------------------------------------------------------------------------

class TestAssertCovarianceValid:
    """Tests for assert_covariance_valid."""

    def test_valid_covariance_passes(self):
        """Valid covariance matrix should pass."""
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert_covariance_valid(Sigma, "cov")

    def test_non_symmetric_fails(self):
        """Non-symmetric matrix should fail."""
        Sigma = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(AssertionError, match="not symmetric"):
            assert_covariance_valid(Sigma, "cov")

    def test_not_psd_fails(self):
        """Symmetric but non-PSD matrix should fail."""
        Sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(AssertionError, match="not PSD"):
            assert_covariance_valid(Sigma, "cov")

    def test_contains_nan_fails(self):
        """Covariance with NaN should fail."""
        Sigma = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with pytest.raises(AssertionError, match="NaN/Inf"):
            assert_covariance_valid(Sigma, "cov")

    def test_not_square_fails(self):
        """Non-square matrix should fail."""
        Sigma = np.ones((2, 3))
        with pytest.raises(AssertionError, match="must be square"):
            assert_covariance_valid(Sigma, "cov")


# ---------------------------------------------------------------------------
# assert_positive_definite tests
# ---------------------------------------------------------------------------

class TestAssertPositiveDefinite:
    """Tests for assert_positive_definite."""

    def test_identity_passes(self):
        """Identity matrix is positive definite."""
        Sigma = np.eye(3)
        assert_positive_definite(Sigma, "Sigma")

    def test_singular_fails(self):
        """Singular PSD matrix is not positive definite."""
        Sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(AssertionError, match="not positive definite"):
            assert_positive_definite(Sigma, "Sigma")

    def test_negative_definite_fails(self):
        """Negative definite matrix should fail."""
        Sigma = -np.eye(2)
        with pytest.raises(AssertionError, match="not positive definite"):
            assert_positive_definite(Sigma, "Sigma")

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        Sigma = np.diag([1.0, 0.5, 1e-8])
        assert_positive_definite(Sigma, "Sigma", tol=1e-9)
        with pytest.raises(AssertionError, match="not positive definite"):
            assert_positive_definite(Sigma, "Sigma", tol=1e-7)


# ---------------------------------------------------------------------------
# assert_returns_valid tests
# ---------------------------------------------------------------------------

class TestAssertReturnsValid:
    """Tests for assert_returns_valid."""

    def test_normal_returns_pass(self):
        """Normal returns should pass."""
        df = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.005, -0.01, 0.02]})
        assert_returns_valid(df, "log_returns")

    def test_extreme_returns_fail(self):
        """Returns exceeding max_abs_return should fail."""
        df = pd.DataFrame({"A": [0.01, 3.0, -0.02]})
        with pytest.raises(AssertionError, match="returns exceeding"):
            assert_returns_valid(df, "returns", max_abs_return=2.0)

    def test_inf_returns_fail(self):
        """Infinite returns should fail."""
        df = pd.DataFrame({"A": [0.01, np.inf, -0.02]})
        with pytest.raises(AssertionError, match="infinite values"):
            assert_returns_valid(df, "returns")

    def test_nan_allowed(self):
        """NaN in sparse returns should pass (not infinite, and NaN excluded from bounds)."""
        df = pd.DataFrame({"A": [0.01, np.nan, -0.02]})
        assert_returns_valid(df, "returns")

    def test_custom_threshold(self):
        """Custom max_abs_return threshold should work."""
        df = pd.DataFrame({"A": [0.5, -0.5]})
        assert_returns_valid(df, "returns", max_abs_return=1.0)
        with pytest.raises(AssertionError, match="returns exceeding"):
            assert_returns_valid(df, "returns", max_abs_return=0.3)


# ---------------------------------------------------------------------------
# assert_weights_valid tests
# ---------------------------------------------------------------------------

class TestAssertWeightsValid:
    """Tests for assert_weights_valid."""

    def test_valid_weights_pass(self):
        """Valid equal weights should pass."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert_weights_valid(w, "portfolio")

    def test_not_summing_to_one_fails(self):
        """Weights not summing to 1 should fail."""
        w = np.array([0.3, 0.3, 0.3])
        with pytest.raises(AssertionError, match="do not sum to 1"):
            assert_weights_valid(w, "portfolio")

    def test_nan_weight_fails(self):
        """NaN in weights should fail."""
        w = np.array([0.5, np.nan, 0.5])
        with pytest.raises(AssertionError, match="NaN/Inf"):
            assert_weights_valid(w, "portfolio")

    def test_below_w_min_fails(self):
        """Weight below w_min should fail."""
        w = np.array([0.01, 0.49, 0.50])
        with pytest.raises(AssertionError, match="below w_min"):
            assert_weights_valid(w, "portfolio", w_min=0.05)

    def test_above_w_max_fails(self):
        """Weight above w_max should fail."""
        w = np.array([0.6, 0.2, 0.2])
        with pytest.raises(AssertionError, match="above w_max"):
            assert_weights_valid(w, "portfolio", w_max=0.5)


# ---------------------------------------------------------------------------
# assert_tensor_shape tests
# ---------------------------------------------------------------------------

class TestAssertTensorShape:
    """Tests for assert_tensor_shape."""

    def test_exact_shape_passes(self):
        """Tensor with exact expected shape should pass."""
        t = torch.ones(3, 4, 5)
        assert_tensor_shape(t, (3, 4, 5), "input")

    def test_wrong_shape_fails(self):
        """Tensor with wrong shape should fail."""
        t = torch.ones(3, 4, 5)
        with pytest.raises(AssertionError, match="dim"):
            assert_tensor_shape(t, (3, 4, 6), "input")

    def test_wrong_ndim_fails(self):
        """Tensor with wrong number of dimensions should fail."""
        t = torch.ones(3, 4)
        with pytest.raises(AssertionError, match="dims"):
            assert_tensor_shape(t, (3, 4, 5), "input")

    def test_wildcard_dimension(self):
        """Using -1 as wildcard should match any size."""
        t = torch.ones(3, 4, 5)
        assert_tensor_shape(t, (3, -1, 5), "input")

    def test_all_wildcards(self):
        """All wildcards should match any 3D tensor."""
        t = torch.ones(7, 8, 9)
        assert_tensor_shape(t, (-1, -1, -1), "input")


# ---------------------------------------------------------------------------
# assert_fold_consistency tests
# ---------------------------------------------------------------------------

class TestAssertFoldConsistency:
    """Tests for assert_fold_consistency."""

    def test_valid_fold_passes(self):
        """Valid fold with correct ordering should pass."""
        fold = {
            "train_start": "2010-01-01",
            "train_end": "2015-12-31",
            "oos_start": "2016-01-01",
            "oos_end": "2016-12-31",
        }
        assert_fold_consistency(fold, "fold_1")

    def test_missing_key_fails(self):
        """Fold missing required key should fail."""
        fold = {"train_start": "2010-01-01", "train_end": "2015-12-31"}
        with pytest.raises(AssertionError, match="missing required key"):
            assert_fold_consistency(fold, "fold_1")

    def test_train_end_after_oos_start_fails(self):
        """train_end > oos_start (overlap) should fail."""
        fold = {
            "train_start": "2010-01-01",
            "train_end": "2016-06-01",
            "oos_start": "2016-01-01",
            "oos_end": "2016-12-31",
        }
        with pytest.raises(AssertionError, match="train_end"):
            assert_fold_consistency(fold, "fold_1")

    def test_inverted_train_dates_fail(self):
        """train_start >= train_end should fail."""
        fold = {
            "train_start": "2016-01-01",
            "train_end": "2010-01-01",
            "oos_start": "2017-01-01",
            "oos_end": "2017-12-31",
        }
        with pytest.raises(AssertionError, match="train_start >= train_end"):
            assert_fold_consistency(fold, "fold_1")

    def test_embargo_gap_passes(self):
        """Fold with embargo gap between train_end and oos_start should pass."""
        fold = {
            "train_start": "2010-01-01",
            "train_end": "2015-12-31",
            "oos_start": "2016-03-01",
            "oos_end": "2016-12-31",
        }
        assert_fold_consistency(fold, "fold_1")


# ---------------------------------------------------------------------------
# assert_no_lookahead tests
# ---------------------------------------------------------------------------

class TestAssertNoLookahead:
    """Tests for assert_no_lookahead."""

    def test_valid_gap_passes(self):
        """Sufficient gap should pass."""
        assert_no_lookahead("2020-01-01", "2020-02-01", "fold_1", min_embargo_days=5)

    def test_zero_embargo_same_day_passes(self):
        """Same day with zero embargo should pass."""
        assert_no_lookahead("2020-01-01", "2020-01-01", "fold_1", min_embargo_days=0)

    def test_insufficient_gap_fails(self):
        """Gap less than min_embargo should fail."""
        with pytest.raises(AssertionError, match="look-ahead bias"):
            assert_no_lookahead("2020-01-01", "2020-01-03", "fold_1", min_embargo_days=5)

    def test_overlap_fails(self):
        """oos_start before train_end should fail."""
        with pytest.raises(AssertionError, match="look-ahead bias"):
            assert_no_lookahead("2020-06-01", "2020-01-01", "fold_1", min_embargo_days=0)

    def test_string_and_timestamp_inputs(self):
        """Both string and Timestamp inputs should work."""
        ts1: pd.Timestamp = pd.Timestamp("2020-01-01")  # type: ignore[assignment]
        ts2: pd.Timestamp = pd.Timestamp("2020-02-01")  # type: ignore[assignment]
        assert_no_lookahead(ts1, ts2, "fold_1")


# ---------------------------------------------------------------------------
# warn_if_nan_fraction_exceeds tests
# ---------------------------------------------------------------------------

class TestWarnIfNanFractionExceeds:
    """Tests for warn_if_nan_fraction_exceeds."""

    def test_no_nans_returns_false(self):
        """Array without NaN should return False."""
        arr = np.array([1.0, 2.0, 3.0])
        result = warn_if_nan_fraction_exceeds(arr, 0.1, "test")
        assert result is False

    def test_high_nan_fraction_warns(self):
        """Array with NaN fraction above threshold should warn."""
        arr = np.array([np.nan, np.nan, 1.0])
        with pytest.warns(UserWarning, match="NaN values"):
            result = warn_if_nan_fraction_exceeds(arr, 0.5, "test")
        assert result is True

    def test_nan_fraction_below_threshold_returns_false(self):
        """NaN fraction below threshold should not warn."""
        arr = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        result = warn_if_nan_fraction_exceeds(arr, 0.5, "test")
        assert result is False

    def test_dataframe_input(self):
        """DataFrame input should work."""
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, np.nan]})
        with pytest.warns(UserWarning, match="NaN values"):
            result = warn_if_nan_fraction_exceeds(df, 0.5, "test")
        assert result is True

    def test_empty_array_returns_false(self):
        """Empty array should return False."""
        arr = np.array([])
        result = warn_if_nan_fraction_exceeds(arr, 0.1, "test")
        assert result is False


# ===========================================================================
# Phase 2 validation function tests
# ===========================================================================


# ---------------------------------------------------------------------------
# assert_non_empty_dataframe tests
# ---------------------------------------------------------------------------

class TestAssertNonEmptyDataframe:
    """Tests for assert_non_empty_dataframe."""

    def test_non_empty_passes(self):
        """DataFrame with rows should pass."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert_non_empty_dataframe(df, "test")

    def test_empty_fails(self):
        """Empty DataFrame should fail."""
        df = pd.DataFrame(columns=["a", "b"])
        with pytest.raises(AssertionError, match="empty"):
            assert_non_empty_dataframe(df, "test")

    def test_single_row_passes(self):
        """DataFrame with single row should pass."""
        df = pd.DataFrame({"a": [1]})
        assert_non_empty_dataframe(df, "test")

    def test_empty_with_index_fails(self):
        """DataFrame with index but no data should fail."""
        df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=0))
        with pytest.raises(AssertionError, match="empty"):
            assert_non_empty_dataframe(df, "test")


# ---------------------------------------------------------------------------
# assert_tensor_bounds tests
# ---------------------------------------------------------------------------

class TestAssertTensorBounds:
    """Tests for assert_tensor_bounds."""

    def test_within_bounds_passes(self):
        """Tensor with values within bounds should pass."""
        t = torch.tensor([0.5, 1.0, 1.5])
        assert_tensor_bounds(t, 0.0, 2.0, "test")

    def test_below_low_fails(self):
        """Tensor with value below low bound should fail."""
        t = torch.tensor([-0.5, 1.0, 1.5])
        with pytest.raises(AssertionError, match="out of bounds"):
            assert_tensor_bounds(t, 0.0, 2.0, "test")

    def test_above_high_fails(self):
        """Tensor with value above high bound should fail."""
        t = torch.tensor([0.5, 1.0, 2.5])
        with pytest.raises(AssertionError, match="out of bounds"):
            assert_tensor_bounds(t, 0.0, 2.0, "test")

    def test_exact_bounds_passes(self):
        """Tensor with values exactly at bounds should pass."""
        t = torch.tensor([0.0, 1.0, 2.0])
        assert_tensor_bounds(t, 0.0, 2.0, "test")

    def test_negative_bounds(self):
        """Negative bounds should work."""
        t = torch.tensor([-2.0, -1.0, 0.0])
        assert_tensor_bounds(t, -3.0, 1.0, "test")

    def test_multidimensional_tensor(self):
        """Multidimensional tensor should be checked."""
        t = torch.ones(3, 4, 5) * 0.5
        assert_tensor_bounds(t, 0.0, 1.0, "test")


# ---------------------------------------------------------------------------
# warn_if_tensor_extreme tests
# ---------------------------------------------------------------------------

class TestWarnIfTensorExtreme:
    """Tests for warn_if_tensor_extreme."""

    def test_normal_values_returns_false(self):
        """Tensor with normal values should return False."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = warn_if_tensor_extreme(t, 10.0, "test")
        assert result is False

    def test_extreme_values_warns(self):
        """Tensor with extreme values should warn."""
        t = torch.tensor([1.0, 100.0, 3.0])
        with pytest.warns(UserWarning, match="extreme values"):
            result = warn_if_tensor_extreme(t, 50.0, "test")
        assert result is True

    def test_negative_extreme_warns(self):
        """Tensor with negative extreme values should warn."""
        t = torch.tensor([1.0, -100.0, 3.0])
        with pytest.warns(UserWarning, match="extreme values"):
            result = warn_if_tensor_extreme(t, 50.0, "test")
        assert result is True

    def test_at_threshold_no_warn(self):
        """Values exactly at threshold should not warn."""
        t = torch.tensor([10.0, -10.0])
        result = warn_if_tensor_extreme(t, 10.0, "test")
        assert result is False


# ---------------------------------------------------------------------------
# assert_embargo_date_in_index tests
# ---------------------------------------------------------------------------

class TestAssertEmbargoDateInIndex:
    """Tests for assert_embargo_date_in_index."""

    def test_date_in_index_passes(self):
        """Date in index should pass."""
        date: pd.Timestamp = pd.Timestamp("2020-01-05")  # type: ignore[assignment]
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        assert_embargo_date_in_index(date, index, "embargo")

    def test_date_not_in_index_fails(self):
        """Date not in index should fail."""
        date: pd.Timestamp = pd.Timestamp("2020-01-03")  # type: ignore[assignment]
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        with pytest.raises(AssertionError, match="not in OOS date index"):
            assert_embargo_date_in_index(date, index, "embargo")

    def test_first_date_passes(self):
        """First date in index should pass."""
        date: pd.Timestamp = pd.Timestamp("2020-01-01")  # type: ignore[assignment]
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        assert_embargo_date_in_index(date, index, "embargo")

    def test_last_date_passes(self):
        """Last date in index should pass."""
        date: pd.Timestamp = pd.Timestamp("2020-01-10")  # type: ignore[assignment]
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        assert_embargo_date_in_index(date, index, "embargo")


# ---------------------------------------------------------------------------
# assert_growth_finite tests
# ---------------------------------------------------------------------------

class TestAssertGrowthFinite:
    """Tests for assert_growth_finite."""

    def test_finite_positive_passes(self):
        """Finite positive growth factors should pass."""
        growth = np.array([1.01, 0.99, 1.05, 0.95])
        assert_growth_finite(growth, "growth")

    def test_nan_fails(self):
        """NaN growth factor should fail."""
        growth = np.array([1.01, np.nan, 1.05])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_growth_finite(growth, "growth")

    def test_inf_fails(self):
        """Inf growth factor should fail."""
        growth = np.array([1.01, np.inf, 1.05])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_growth_finite(growth, "growth")

    def test_negative_inf_fails(self):
        """-Inf growth factor should fail."""
        growth = np.array([1.01, -np.inf, 1.05])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_growth_finite(growth, "growth")

    def test_negative_fails(self):
        """Negative growth factor should fail."""
        growth = np.array([1.01, -0.5, 1.05])
        with pytest.raises(AssertionError, match="negative"):
            assert_growth_finite(growth, "growth")

    def test_zero_passes(self):
        """Zero growth factor should pass (not negative)."""
        growth = np.array([1.01, 0.0, 1.05])
        assert_growth_finite(growth, "growth")


# ---------------------------------------------------------------------------
# warn_loss_explosion tests
# ---------------------------------------------------------------------------

class TestWarnLossExplosion:
    """Tests for warn_loss_explosion."""

    def test_normal_loss_no_warn(self):
        """Loss below threshold should not warn."""
        result = warn_loss_explosion(
            current_loss=5.0,
            initial_loss=4.0,
            multiplier=10.0,
            epoch=5,
        )
        assert result is False

    def test_exploded_loss_warns(self):
        """Loss exceeding threshold should warn."""
        with pytest.warns(UserWarning, match="Loss explosion"):
            result = warn_loss_explosion(
                current_loss=50.0,
                initial_loss=4.0,
                multiplier=10.0,
                epoch=5,
            )
        assert result is True

    def test_exactly_at_threshold_no_warn(self):
        """Loss exactly at threshold should not warn."""
        result = warn_loss_explosion(
            current_loss=40.0,
            initial_loss=4.0,
            multiplier=10.0,
            epoch=5,
        )
        assert result is False

    def test_zero_initial_no_warn(self):
        """Zero initial loss should not warn (division guard)."""
        result = warn_loss_explosion(
            current_loss=100.0,
            initial_loss=0.0,
            multiplier=10.0,
            epoch=5,
        )
        assert result is False

    def test_decreasing_loss_no_warn(self):
        """Decreasing loss should not warn."""
        result = warn_loss_explosion(
            current_loss=2.0,
            initial_loss=10.0,
            multiplier=10.0,
            epoch=5,
        )
        assert result is False


# ---------------------------------------------------------------------------
# assert_active_units_valid tests
# ---------------------------------------------------------------------------

class TestAssertActiveUnitsValid:
    """Tests for assert_active_units_valid."""

    def test_valid_range_passes(self):
        """AU within valid range should pass."""
        assert_active_units_valid(50, 100, "train")

    def test_zero_au_passes(self):
        """Zero active units should pass."""
        assert_active_units_valid(0, 100, "train")

    def test_max_au_passes(self):
        """AU equal to K should pass."""
        assert_active_units_valid(100, 100, "train")

    def test_negative_au_fails(self):
        """Negative AU should fail."""
        with pytest.raises(AssertionError, match="outside valid range"):
            assert_active_units_valid(-1, 100, "train")

    def test_au_exceeds_k_fails(self):
        """AU exceeding K should fail."""
        with pytest.raises(AssertionError, match="outside valid range"):
            assert_active_units_valid(150, 100, "train")


# ---------------------------------------------------------------------------
# assert_ann_vol_positive tests
# ---------------------------------------------------------------------------

class TestAssertAnnVolPositive:
    """Tests for assert_ann_vol_positive."""

    def test_positive_vol_passes(self):
        """Positive volatility should pass."""
        assert_ann_vol_positive(0.2, "portfolio_vol")

    def test_small_positive_passes(self):
        """Small positive volatility should pass."""
        assert_ann_vol_positive(1e-8, "portfolio_vol")

    def test_zero_fails(self):
        """Zero volatility should fail."""
        with pytest.raises(AssertionError, match="invalid"):
            assert_ann_vol_positive(0.0, "portfolio_vol")

    def test_negative_fails(self):
        """Negative volatility should fail."""
        with pytest.raises(AssertionError, match="invalid"):
            assert_ann_vol_positive(-0.1, "portfolio_vol")

    def test_nan_fails(self):
        """NaN volatility should fail."""
        with pytest.raises(AssertionError, match="invalid"):
            assert_ann_vol_positive(float("nan"), "portfolio_vol")

    def test_inf_fails(self):
        """Inf volatility should fail."""
        with pytest.raises(AssertionError, match="invalid"):
            assert_ann_vol_positive(float("inf"), "portfolio_vol")

    def test_custom_min_vol(self):
        """Custom min_vol threshold should work."""
        assert_ann_vol_positive(0.001, "vol", min_vol=0.0001)
        with pytest.raises(AssertionError, match="invalid"):
            assert_ann_vol_positive(0.0001, "vol", min_vol=0.001)


# ===========================================================================
# Phase 1 Extension: Data Integrity Validators
# ===========================================================================

from src.validation import (
    assert_positive_prices,
    assert_log_input_positive,
    assert_volume_non_negative,
    assert_z_score_normalized,
    assert_crisis_fraction_bounds,
    warn_if_price_discontinuity,
)


class TestAssertPositivePrices:
    """Tests for assert_positive_prices."""

    def test_positive_prices_pass(self):
        """Positive prices should pass."""
        prices = np.array([10.0, 20.0, 30.0])
        assert_positive_prices(prices, "stock_prices")

    def test_zero_price_fails(self):
        """Zero price should fail."""
        prices = np.array([10.0, 0.0, 30.0])
        with pytest.raises(AssertionError, match="non-positive"):
            assert_positive_prices(prices, "stock_prices")

    def test_negative_price_fails(self):
        """Negative price should fail."""
        prices = np.array([10.0, -5.0, 30.0])
        with pytest.raises(AssertionError, match="non-positive"):
            assert_positive_prices(prices, "stock_prices")

    def test_nan_ignored(self):
        """NaN prices should be ignored."""
        prices = np.array([10.0, np.nan, 30.0])
        assert_positive_prices(prices, "stock_prices")

    def test_dataframe_input(self):
        """DataFrame input should work."""
        df = pd.DataFrame({"A": [10.0, 20.0], "B": [30.0, 40.0]})
        assert_positive_prices(df, "stock_prices")

    def test_series_input(self):
        """Series input should work."""
        series = pd.Series([10.0, 20.0, 30.0])
        assert_positive_prices(series, "stock_prices")


class TestAssertLogInputPositive:
    """Tests for assert_log_input_positive."""

    def test_positive_values_pass(self):
        """Positive values should pass."""
        values = np.array([0.1, 1.0, 10.0])
        assert_log_input_positive(values, "prices")

    def test_zero_fails(self):
        """Zero value should fail."""
        values = np.array([0.1, 0.0, 10.0])
        with pytest.raises(AssertionError, match="non-positive"):
            assert_log_input_positive(values, "prices")

    def test_negative_fails(self):
        """Negative value should fail."""
        values = np.array([0.1, -1.0, 10.0])
        with pytest.raises(AssertionError, match="non-positive"):
            assert_log_input_positive(values, "prices")

    def test_nan_ignored(self):
        """NaN values should be ignored."""
        values = np.array([0.1, np.nan, 10.0])
        assert_log_input_positive(values, "prices")


class TestAssertVolumeNonNegative:
    """Tests for assert_volume_non_negative."""

    def test_positive_volume_passes(self):
        """Positive volume should pass."""
        volume = np.array([1000, 2000, 3000])
        assert_volume_non_negative(volume, "volume")

    def test_zero_volume_passes(self):
        """Zero volume should pass."""
        volume = np.array([1000, 0, 3000])
        assert_volume_non_negative(volume, "volume")

    def test_negative_volume_fails(self):
        """Negative volume should fail."""
        volume = np.array([1000, -500, 3000])
        with pytest.raises(AssertionError, match="negative"):
            assert_volume_non_negative(volume, "volume")

    def test_dataframe_input(self):
        """DataFrame input should work."""
        df = pd.DataFrame({"vol": [1000, 2000, 3000]})
        assert_volume_non_negative(df, "volume")


class TestAssertZScoreNormalized:
    """Tests for assert_z_score_normalized."""

    def test_normalized_passes(self):
        """Properly z-scored data should pass."""
        np.random.seed(42)
        arr = np.random.randn(1000)
        assert_z_score_normalized(arr, "features")

    def test_non_zero_mean_fails(self):
        """Data with non-zero mean should fail."""
        np.random.seed(42)
        arr = np.random.randn(1000) + 5.0  # Shift mean
        with pytest.raises(AssertionError, match="z-score normalization"):
            assert_z_score_normalized(arr, "features")

    def test_non_unit_std_fails(self):
        """Data with non-unit std should fail."""
        np.random.seed(42)
        arr = np.random.randn(1000) * 3.0  # Scale std
        with pytest.raises(AssertionError, match="z-score normalization"):
            assert_z_score_normalized(arr, "features")

    def test_small_sample_skipped(self):
        """Small samples should be skipped."""
        arr = np.array([1.0, 2.0, 3.0])  # Only 3 values
        assert_z_score_normalized(arr, "features")  # Should not raise


class TestAssertCrisisFractionBounds:
    """Tests for assert_crisis_fraction_bounds."""

    def test_valid_fractions_pass(self):
        """Fractions in [0, 1] should pass."""
        fractions = np.array([0.0, 0.5, 1.0])
        assert_crisis_fraction_bounds(fractions, "crisis")

    def test_negative_fraction_fails(self):
        """Negative fraction should fail."""
        fractions = np.array([0.0, -0.1, 0.5])
        with pytest.raises(AssertionError, match="out of"):
            assert_crisis_fraction_bounds(fractions, "crisis")

    def test_greater_than_one_fails(self):
        """Fraction > 1 should fail."""
        fractions = np.array([0.0, 0.5, 1.5])
        with pytest.raises(AssertionError, match="out of"):
            assert_crisis_fraction_bounds(fractions, "crisis")

    def test_tensor_input(self):
        """Tensor input should work."""
        fractions = torch.tensor([0.0, 0.5, 1.0])
        assert_crisis_fraction_bounds(fractions, "crisis")


class TestWarnIfPriceDiscontinuity:
    """Tests for warn_if_price_discontinuity."""

    def test_normal_returns_no_warn(self):
        """Normal returns should not warn."""
        returns = np.array([0.01, -0.02, 0.015])
        result = warn_if_price_discontinuity(returns, 0.5, "returns")
        assert result is False

    def test_extreme_returns_warns(self):
        """Extreme returns should warn."""
        returns = np.array([0.01, 0.8, 0.015])  # 80% return
        with pytest.warns(UserWarning, match="returns exceed"):
            result = warn_if_price_discontinuity(returns, 0.5, "returns")
        assert result is True

    def test_dataframe_input(self):
        """DataFrame input should work."""
        df = pd.DataFrame({"A": [0.01, 0.02, 0.03]})
        result = warn_if_price_discontinuity(df, 0.5, "returns")
        assert result is False


# ===========================================================================
# Phase 2 Extension: Numerical Stability Validators
# ===========================================================================

from src.validation import (
    assert_kl_non_negative,
    assert_reconstruction_bounded,
    assert_gradient_finite,
    assert_eigenvalue_spectrum_valid,
    assert_cholesky_condition,
    assert_armijo_params_valid,
    warn_if_loss_component_imbalance,
)


class TestAssertKLNonNegative:
    """Tests for assert_kl_non_negative."""

    def test_positive_kl_passes(self):
        """Positive KL should pass."""
        assert_kl_non_negative(1.5, "kl")

    def test_zero_kl_passes(self):
        """Zero KL should pass."""
        assert_kl_non_negative(0.0, "kl")

    def test_negative_kl_fails(self):
        """Negative KL should fail."""
        with pytest.raises(AssertionError, match="negative"):
            assert_kl_non_negative(-0.1, "kl")

    def test_tensor_input(self):
        """Tensor input should work."""
        kl = torch.tensor(1.5)
        assert_kl_non_negative(kl, "kl")


class TestAssertReconstructionBounded:
    """Tests for assert_reconstruction_bounded."""

    def test_normal_loss_passes(self):
        """Normal loss should pass."""
        assert_reconstruction_bounded(100.0, 1e6, "recon")

    def test_exploded_loss_fails(self):
        """Exploded loss should fail."""
        with pytest.raises(AssertionError, match="exploding"):
            assert_reconstruction_bounded(1e7, 1e6, "recon")

    def test_nan_fails(self):
        """NaN loss should fail."""
        with pytest.raises(AssertionError, match="non-finite"):
            assert_reconstruction_bounded(float("nan"), 1e6, "recon")

    def test_inf_fails(self):
        """Inf loss should fail."""
        with pytest.raises(AssertionError, match="non-finite"):
            assert_reconstruction_bounded(float("inf"), 1e6, "recon")

    def test_tensor_input(self):
        """Tensor input should work."""
        recon = torch.tensor(100.0)
        assert_reconstruction_bounded(recon, 1e6, "recon")


class TestAssertGradientFinite:
    """Tests for assert_gradient_finite."""

    def test_finite_gradient_passes(self):
        """Finite gradient should pass."""
        grad = np.array([0.1, -0.2, 0.3])
        assert_gradient_finite(grad, "grad")

    def test_nan_gradient_fails(self):
        """Gradient with NaN should fail."""
        grad = np.array([0.1, np.nan, 0.3])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_gradient_finite(grad, "grad")

    def test_inf_gradient_fails(self):
        """Gradient with Inf should fail."""
        grad = np.array([0.1, np.inf, 0.3])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_gradient_finite(grad, "grad")

    def test_tensor_input(self):
        """Tensor input should work."""
        grad = torch.tensor([0.1, -0.2, 0.3])
        assert_gradient_finite(grad, "grad")


class TestAssertEigenvalueSpectrumValid:
    """Tests for assert_eigenvalue_spectrum_valid."""

    def test_positive_eigenvalues_pass(self):
        """Positive eigenvalues should pass."""
        eigenvalues = np.array([3.0, 2.0, 1.0, 0.5])
        assert_eigenvalue_spectrum_valid(eigenvalues, "Sigma")

    def test_negative_eigenvalue_fails(self):
        """Negative eigenvalue below threshold should fail."""
        eigenvalues = np.array([3.0, 2.0, -0.1])
        with pytest.raises(AssertionError, match="below threshold"):
            assert_eigenvalue_spectrum_valid(eigenvalues, "Sigma")

    def test_small_negative_within_tolerance(self):
        """Small negative within tolerance should pass."""
        eigenvalues = np.array([3.0, 2.0, -1e-12])
        assert_eigenvalue_spectrum_valid(eigenvalues, "Sigma", min_val=-1e-10)

    def test_nan_eigenvalue_fails(self):
        """NaN eigenvalue should fail."""
        eigenvalues = np.array([3.0, np.nan, 1.0])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_eigenvalue_spectrum_valid(eigenvalues, "Sigma")


class TestAssertCholeskyCondition:
    """Tests for assert_cholesky_condition."""

    def test_well_conditioned_passes(self):
        """Well-conditioned Cholesky should pass."""
        L = np.diag([2.0, 1.5, 1.0])
        assert_cholesky_condition(L, 1e8, "L")

    def test_ill_conditioned_fails(self):
        """Ill-conditioned Cholesky should fail."""
        L = np.diag([1e6, 1e-6, 1.0])
        with pytest.raises(AssertionError, match="condition number"):
            assert_cholesky_condition(L, 1e6, "L")

    def test_zero_diagonal_fails(self):
        """Zero diagonal element should fail (infinite condition)."""
        L = np.diag([1.0, 0.0, 1.0])
        with pytest.raises(AssertionError, match="condition number"):
            assert_cholesky_condition(L, 1e8, "L")


class TestAssertArmijoParamsValid:
    """Tests for assert_armijo_params_valid."""

    def test_valid_params_pass(self):
        """Valid Armijo parameters should pass."""
        assert_armijo_params_valid(0.1, 0.5, "Armijo")

    def test_c_out_of_range_fails(self):
        """c outside (0, 0.5) should fail."""
        with pytest.raises(AssertionError, match="invalid parameters"):
            assert_armijo_params_valid(0.6, 0.5, "Armijo")

    def test_rho_out_of_range_fails(self):
        """rho outside (0, 1) should fail."""
        with pytest.raises(AssertionError, match="invalid parameters"):
            assert_armijo_params_valid(0.1, 1.5, "Armijo")

    def test_negative_c_fails(self):
        """Negative c should fail."""
        with pytest.raises(AssertionError, match="invalid parameters"):
            assert_armijo_params_valid(-0.1, 0.5, "Armijo")


class TestWarnIfLossComponentImbalance:
    """Tests for warn_if_loss_component_imbalance."""

    def test_balanced_no_warn(self):
        """Balanced components should not warn."""
        result = warn_if_loss_component_imbalance(10.0, 15.0, 100.0, "loss")
        assert result is False

    def test_imbalanced_warns(self):
        """Imbalanced components should warn."""
        with pytest.warns(UserWarning, match="imbalance"):
            result = warn_if_loss_component_imbalance(1000.0, 1.0, 100.0, "loss")
        assert result is True

    def test_zero_component_no_warn(self):
        """Zero component should not warn (can't compute ratio)."""
        result = warn_if_loss_component_imbalance(0.0, 10.0, 100.0, "loss")
        assert result is False


# ===========================================================================
# Phase 3 Extension: Cross-Module Alignment Validators
# ===========================================================================

from src.validation import (
    assert_universe_snapshot_consistency,
    assert_factor_idio_trace_match,
    assert_embargo_in_calendar,
    assert_exposure_matrix_alignment,
    warn_if_universe_drift,
)


class TestAssertUniverseSnapshotConsistency:
    """Tests for assert_universe_snapshot_consistency."""

    def test_high_overlap_passes(self):
        """High universe overlap should pass."""
        train_ids = [1, 2, 3, 4, 5]
        test_ids = [1, 2, 3, 4, 6]
        assert_universe_snapshot_consistency(train_ids, test_ids, 0.6, "universe")

    def test_low_overlap_fails(self):
        """Low universe overlap should fail."""
        train_ids = [1, 2, 3]
        test_ids = [4, 5, 6]
        with pytest.raises(AssertionError, match="insufficient universe overlap"):
            assert_universe_snapshot_consistency(train_ids, test_ids, 0.5, "universe")

    def test_identical_passes(self):
        """Identical universes should pass."""
        ids = [1, 2, 3, 4, 5]
        assert_universe_snapshot_consistency(ids, ids, 0.99, "universe")


class TestAssertFactorIdioTraceMatch:
    """Tests for assert_factor_idio_trace_match."""

    def test_matching_traces_pass(self):
        """Matching trace decomposition should pass."""
        Sigma_f = np.diag([1.0, 1.0, 1.0])  # trace = 3
        D_eps = np.array([0.5, 0.5, 0.5])   # trace = 1.5
        Sigma_total = np.diag([1.5, 1.5, 1.5])  # trace = 4.5
        assert_factor_idio_trace_match(Sigma_f, D_eps, Sigma_total, 0.01, "cov")

    def test_mismatched_traces_fail(self):
        """Mismatched trace decomposition should fail."""
        Sigma_f = np.diag([1.0, 1.0, 1.0])  # trace = 3
        D_eps = np.array([0.5, 0.5, 0.5])   # trace = 1.5
        Sigma_total = np.diag([3.0, 3.0, 3.0])  # trace = 9 (doesn't match 3+1.5)
        with pytest.raises(AssertionError, match="trace decomposition"):
            assert_factor_idio_trace_match(Sigma_f, D_eps, Sigma_total, 0.01, "cov")


class TestAssertEmbargoInCalendar:
    """Tests for assert_embargo_in_calendar."""

    def test_date_in_calendar_passes(self):
        """Date in calendar should pass."""
        embargo = pd.Timestamp("2020-01-05")
        calendar = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        assert_embargo_in_calendar(embargo, calendar, "embargo")

    def test_date_not_in_calendar_fails(self):
        """Date not in calendar should fail."""
        embargo = pd.Timestamp("2020-01-03")
        calendar = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
        with pytest.raises(AssertionError, match="not in trading calendar"):
            assert_embargo_in_calendar(embargo, calendar, "embargo")

    def test_string_input(self):
        """String date input should work."""
        calendar = pd.DatetimeIndex(["2020-01-01", "2020-01-05"])
        assert_embargo_in_calendar("2020-01-01", calendar, "embargo")


class TestAssertExposureMatrixAlignment:
    """Tests for assert_exposure_matrix_alignment."""

    def test_aligned_passes(self):
        """Aligned B and universe should pass."""
        B = np.ones((3, 5))
        stock_ids = [10, 20, 30]
        universe_ids = [10, 20, 30]
        assert_exposure_matrix_alignment(B, stock_ids, universe_ids, "B")

    def test_misaligned_fails(self):
        """Misaligned B and universe should fail."""
        B = np.ones((3, 5))
        stock_ids = [10, 20, 30]
        universe_ids = [10, 20, 40]
        with pytest.raises(AssertionError, match="don't match universe"):
            assert_exposure_matrix_alignment(B, stock_ids, universe_ids, "B")

    def test_row_count_mismatch_fails(self):
        """B rows not matching stock_ids should fail."""
        B = np.ones((3, 5))
        stock_ids = [10, 20]  # Only 2 IDs for 3 rows
        universe_ids = [10, 20]
        with pytest.raises(AssertionError, match="B rows"):
            assert_exposure_matrix_alignment(B, stock_ids, universe_ids, "B")


class TestWarnIfUniverseDrift:
    """Tests for warn_if_universe_drift."""

    def test_low_drift_no_warn(self):
        """Low universe drift should not warn."""
        current = [1, 2, 3, 4, 5]
        previous = [1, 2, 3, 4, 6]
        result = warn_if_universe_drift(current, previous, 0.3, "universe")
        assert result is False

    def test_high_drift_warns(self):
        """High universe drift should warn."""
        current = [1, 2, 3]
        previous = [4, 5, 6, 7, 8]  # No overlap
        with pytest.warns(UserWarning, match="universe drift"):
            result = warn_if_universe_drift(current, previous, 0.3, "universe")
        assert result is True

    def test_empty_previous_no_warn(self):
        """Empty previous universe should not warn."""
        current = [1, 2, 3]
        previous: list[int] = []
        result = warn_if_universe_drift(current, previous, 0.3, "universe")
        assert result is False


# ===========================================================================
# Phase 4 Extension: Training Dynamics & Portfolio Validators
# ===========================================================================

from src.validation import (
    warn_if_au_collapsed,
    warn_if_sigma_bounds_stuck,
    assert_sharpe_denominator_valid,
    assert_sortino_observations_sufficient,
    assert_calmar_denominator_valid,
    assert_turnover_constraint_satisfied,
)


class TestWarnIfAuCollapsed:
    """Tests for warn_if_au_collapsed."""

    def test_normal_ratio_no_warn(self):
        """Normal AU ratio should not warn."""
        result = warn_if_au_collapsed(30, 100, 0.05, 0.80, "AU")
        assert result is False

    def test_low_ratio_warns(self):
        """Low AU ratio should warn."""
        with pytest.warns(UserWarning, match="collapsed"):
            result = warn_if_au_collapsed(2, 100, 0.05, 0.80, "AU")
        assert result is True

    def test_high_ratio_warns(self):
        """High AU ratio should warn."""
        with pytest.warns(UserWarning, match="high"):
            result = warn_if_au_collapsed(90, 100, 0.05, 0.80, "AU")
        assert result is True

    def test_exact_bounds_no_warn(self):
        """AU at exact bounds should not warn."""
        result = warn_if_au_collapsed(5, 100, 0.05, 0.80, "AU")
        assert result is False
        result = warn_if_au_collapsed(80, 100, 0.05, 0.80, "AU")
        assert result is False


class TestWarnIfSigmaBoundsStuck:
    """Tests for warn_if_sigma_bounds_stuck."""

    def test_short_streak_no_warn(self):
        """Short streak should not warn."""
        result = warn_if_sigma_bounds_stuck(3, 5, "sigma")
        assert result is False

    def test_long_streak_warns(self):
        """Long streak should warn."""
        with pytest.warns(UserWarning, match="bounds hit"):
            result = warn_if_sigma_bounds_stuck(10, 5, "sigma")
        assert result is True

    def test_at_max_no_warn(self):
        """Streak at max should not warn."""
        result = warn_if_sigma_bounds_stuck(5, 5, "sigma")
        assert result is False


class TestAssertSharpeDenominatorValid:
    """Tests for assert_sharpe_denominator_valid."""

    def test_normal_vol_passes(self):
        """Normal volatility should pass."""
        assert_sharpe_denominator_valid(0.2, 1e-6, "Sharpe")

    def test_zero_vol_fails(self):
        """Zero volatility should fail."""
        with pytest.raises(AssertionError, match="invalid volatility"):
            assert_sharpe_denominator_valid(0.0, 1e-6, "Sharpe")

    def test_tiny_vol_fails(self):
        """Tiny volatility should fail."""
        with pytest.raises(AssertionError, match="invalid volatility"):
            assert_sharpe_denominator_valid(1e-8, 1e-6, "Sharpe")

    def test_nan_vol_fails(self):
        """NaN volatility should fail."""
        with pytest.raises(AssertionError, match="invalid volatility"):
            assert_sharpe_denominator_valid(float("nan"), 1e-6, "Sharpe")


class TestAssertSortinoObservationsSufficient:
    """Tests for assert_sortino_observations_sufficient."""

    def test_sufficient_passes(self):
        """Sufficient observations should pass."""
        assert_sortino_observations_sufficient(10, 2, "Sortino")

    def test_insufficient_fails(self):
        """Insufficient observations should fail."""
        with pytest.raises(AssertionError, match="insufficient"):
            assert_sortino_observations_sufficient(1, 2, "Sortino")

    def test_at_minimum_passes(self):
        """Exactly at minimum should pass."""
        assert_sortino_observations_sufficient(2, 2, "Sortino")


class TestAssertCalmarDenominatorValid:
    """Tests for assert_calmar_denominator_valid."""

    def test_positive_drawdown_passes(self):
        """Positive max drawdown should pass."""
        assert_calmar_denominator_valid(0.2, "Calmar")

    def test_zero_drawdown_fails(self):
        """Zero max drawdown should fail."""
        with pytest.raises(AssertionError, match="invalid max drawdown"):
            assert_calmar_denominator_valid(0.0, "Calmar")

    def test_nan_drawdown_fails(self):
        """NaN max drawdown should fail."""
        with pytest.raises(AssertionError, match="invalid max drawdown"):
            assert_calmar_denominator_valid(float("nan"), "Calmar")

    def test_negative_drawdown_passes(self):
        """Negative max drawdown should pass (it's non-zero)."""
        assert_calmar_denominator_valid(-0.2, "Calmar")


class TestAssertTurnoverConstraintSatisfied:
    """Tests for assert_turnover_constraint_satisfied."""

    def test_satisfied_passes(self):
        """Satisfied constraint should pass."""
        assert_turnover_constraint_satisfied(0.3, 0.5, 0.01, "turnover")

    def test_exactly_at_limit_passes(self):
        """Turnover exactly at limit should pass."""
        assert_turnover_constraint_satisfied(0.5, 0.5, 0.01, "turnover")

    def test_violated_fails(self):
        """Violated constraint should fail."""
        with pytest.raises(AssertionError, match="violated"):
            assert_turnover_constraint_satisfied(0.6, 0.5, 0.01, "turnover")

    def test_within_tolerance_passes(self):
        """Turnover within tolerance should pass."""
        assert_turnover_constraint_satisfied(0.505, 0.5, 0.01, "turnover")
