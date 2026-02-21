"""
Centralized validation utilities for runtime checks.

Provides reusable assertion functions for validating array properties
(dimensionality, finiteness, positive semidefiniteness) across modules.

Reference: data_preprocessing_diagnostic.md Section 5 (CRITICAL validations).

Validation level can be configured via environment variable:
- VALIDATION_LEVEL="strict" (default): Assertions + exceptions
- VALIDATION_LEVEL="warn": Log warnings but continue
- VALIDATION_LEVEL="off": Disabled (production high-performance)
"""

import logging
import os
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _get_validation_level() -> str:
    """Get current validation level from environment (checked at call time)."""
    return os.environ.get("VALIDATION_LEVEL", "strict")


def _should_validate() -> bool:
    """Check if validation is enabled."""
    return _get_validation_level() != "off"


def _is_strict() -> bool:
    """Check if validation is in strict mode."""
    return _get_validation_level() == "strict"


def assert_finite_2d(arr: np.ndarray, name: str) -> None:
    """
    Assert array is 2D and contains only finite values (no NaN/Inf).

    :param arr (np.ndarray): Array to validate
    :param name (str): Name for error messages

    :raises AssertionError: If array is not 2D or contains NaN/Inf
    """
    assert arr.ndim == 2, f"{name} must be 2D, got {arr.ndim}D"
    assert np.isfinite(arr).all(), f"{name} contains NaN/Inf values"


def assert_positive_semidefinite(
    Sigma: np.ndarray,
    name: str,
    tol: float = -1e-10,
) -> None:
    """
    Assert matrix is positive semi-definite (all eigenvalues >= tol).

    :param Sigma (np.ndarray): Symmetric matrix to validate
    :param name (str): Name for error messages
    :param tol (float): Minimum eigenvalue threshold (default: -1e-10 for numerical tolerance)

    :raises AssertionError: If any eigenvalue is below tolerance
    """
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eig = float(eigenvalues.min())
    assert min_eig >= tol, (
        f"{name} is not PSD (min eigenvalue: {min_eig:.2e}, threshold: {tol:.2e})"
    )


def assert_no_nan_tensor(t: torch.Tensor, name: str) -> None:
    """
    Assert PyTorch tensor contains no NaN values.

    :param t (torch.Tensor): Tensor to validate
    :param name (str): Name for error messages

    :raises AssertionError: If tensor contains NaN values
    """
    assert not torch.isnan(t).any(), f"{name} contains NaN values"


def assert_no_inf_tensor(t: torch.Tensor, name: str) -> None:
    """
    Assert PyTorch tensor contains no Inf values.

    :param t (torch.Tensor): Tensor to validate
    :param name (str): Name for error messages

    :raises AssertionError: If tensor contains Inf values
    """
    assert not torch.isinf(t).any(), f"{name} contains Inf values"


def assert_finite_tensor(t: torch.Tensor, name: str) -> None:
    """
    Assert PyTorch tensor contains only finite values (no NaN/Inf).

    :param t (torch.Tensor): Tensor to validate
    :param name (str): Name for error messages

    :raises AssertionError: If tensor contains NaN or Inf values
    """
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf values"


def assert_no_duplicate_ids(
    ids: list[int] | np.ndarray,
    name: str,
) -> None:
    """
    Assert list/array of IDs contains no duplicates.

    :param ids (list[int] | np.ndarray): IDs to validate
    :param name (str): Name for error messages

    :raises AssertionError: If duplicate IDs found
    """
    ids_list = list(ids) if isinstance(ids, np.ndarray) else ids
    assert len(ids_list) == len(set(ids_list)), (
        f"Duplicate IDs found in {name}: {len(ids_list)} total, "
        f"{len(set(ids_list))} unique"
    )


def assert_shape_match(
    arr1: np.ndarray,
    arr2: np.ndarray,
    name1: str,
    name2: str,
) -> None:
    """
    Assert two arrays have matching shapes.

    :param arr1 (np.ndarray): First array
    :param arr2 (np.ndarray): Second array
    :param name1 (str): Name of first array for error messages
    :param name2 (str): Name of second array for error messages

    :raises AssertionError: If shapes don't match
    """
    assert arr1.shape == arr2.shape, (
        f"{name1} shape {arr1.shape} != {name2} shape {arr2.shape}"
    )


def assert_column_exists(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    column: str,
    name: str,
) -> None:
    """
    Assert DataFrame contains required column.

    :param df (pd.DataFrame): DataFrame to validate
    :param column (str): Required column name
    :param name (str): DataFrame name for error messages

    :raises AssertionError: If column not found
    """
    assert column in df.columns, f"{name} must have '{column}' column"


def assert_weights_sum_to_one(
    w: np.ndarray,
    name: str = "weights",
    tol: float = 1e-3,
) -> None:
    """
    Assert portfolio weights sum to approximately 1.

    :param w (np.ndarray): Weight vector
    :param name (str): Name for error messages
    :param tol (float): Tolerance for sum deviation from 1.0

    :raises AssertionError: If sum deviates from 1.0 by more than tol
    """
    w_sum = float(np.sum(w))
    assert abs(w_sum - 1.0) < tol, (
        f"{name} do not sum to 1: sum={w_sum:.6f}, expected 1.0 +/- {tol}"
    )


def assert_non_negative_eigenvalues(
    eigenvalues: np.ndarray,
    name: str,
    tol: float = -1e-10,
) -> None:
    """
    Assert all eigenvalues are non-negative (within tolerance).

    :param eigenvalues (np.ndarray): Eigenvalue array
    :param name (str): Name for error messages
    :param tol (float): Minimum allowed eigenvalue

    :raises AssertionError: If any eigenvalue is below tolerance
    """
    min_eig = float(eigenvalues.min())
    assert min_eig >= tol, (
        f"Negative eigenvalues detected in {name}: min={min_eig:.2e}"
    )


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 1)
# ===========================================================================


def assert_bounds(
    arr: np.ndarray,
    low: float,
    high: float,
    name: str,
) -> None:
    """
    Assert all values in array are within [low, high].

    :param arr (np.ndarray): Array to validate
    :param low (float): Minimum allowed value
    :param high (float): Maximum allowed value
    :param name (str): Name for error messages

    :raises AssertionError: If any value is outside bounds
    """
    if not _should_validate():
        return
    arr_min = float(np.nanmin(arr))
    arr_max = float(np.nanmax(arr))
    msg = f"{name} values out of bounds [{low}, {high}]: min={arr_min:.4g}, max={arr_max:.4g}"
    if _is_strict():
        assert arr_min >= low and arr_max <= high, msg
    elif arr_min < low or arr_max > high:
        logger.warning(msg)


def assert_monotonic_dates(
    dates: pd.DatetimeIndex | pd.Series | np.ndarray,
    name: str,
    strict: bool = False,
) -> None:
    """
    Assert dates are monotonically increasing (no duplicates if strict=True).

    :param dates (pd.DatetimeIndex | pd.Series | np.ndarray): Date sequence
    :param name (str): Name for error messages
    :param strict (bool): If True, require strictly increasing (no ties)

    :raises AssertionError: If dates are not monotonically increasing
    """
    if not _should_validate():
        return
    dates_arr = pd.to_datetime(dates)
    if strict:
        is_sorted = bool(dates_arr.is_monotonic_increasing)
        has_no_dups = len(dates_arr) == len(dates_arr.unique())
        condition = is_sorted and has_no_dups
        msg = f"{name} dates are not strictly monotonic increasing"
    else:
        condition = bool(dates_arr.is_monotonic_increasing)
        msg = f"{name} dates are not monotonically increasing"

    if _is_strict():
        assert condition, msg
    elif not condition:
        logger.warning(msg)


def assert_condition_number(
    Sigma: np.ndarray,
    threshold: float,
    name: str,
) -> None:
    """
    Assert matrix condition number is below threshold.

    :param Sigma (np.ndarray): Matrix to validate (typically covariance)
    :param threshold (float): Maximum allowed condition number
    :param name (str): Name for error messages

    :raises AssertionError: If condition number exceeds threshold
    """
    if not _should_validate():
        return
    eigenvalues = np.linalg.eigvalsh(Sigma)
    max_eig = float(np.max(np.abs(eigenvalues)))
    min_eig = float(np.min(np.abs(eigenvalues)))
    if min_eig < 1e-14:
        cond_num = np.inf
    else:
        cond_num = max_eig / min_eig

    msg = f"{name} condition number {cond_num:.2e} exceeds threshold {threshold:.2e}"
    if _is_strict():
        assert cond_num <= threshold, msg
    elif cond_num > threshold:
        logger.warning(msg)


def assert_date_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str,
    name2: str,
) -> None:
    """
    Assert two DataFrames have aligned date indices.

    :param df1 (pd.DataFrame): First DataFrame
    :param df2 (pd.DataFrame): Second DataFrame
    :param name1 (str): Name of first DataFrame
    :param name2 (str): Name of second DataFrame

    :raises AssertionError: If date indices don't match
    """
    if not _should_validate():
        return
    idx1 = pd.DatetimeIndex(df1.index)
    idx2 = pd.DatetimeIndex(df2.index)
    if not idx1.equals(idx2):
        n_common = len(idx1.intersection(idx2))
        msg = (
            f"{name1} and {name2} have misaligned dates: "
            f"{len(idx1)} vs {len(idx2)} rows, {n_common} common"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_stock_id_alignment(
    ids1: Sequence[int] | np.ndarray,
    ids2: Sequence[int] | np.ndarray,
    name1: str,
    name2: str,
) -> None:
    """
    Assert two sets of stock IDs are identical (same order not required).

    :param ids1 (Sequence[int] | np.ndarray): First ID set
    :param ids2 (Sequence[int] | np.ndarray): Second ID set
    :param name1 (str): Name of first ID set
    :param name2 (str): Name of second ID set

    :raises AssertionError: If ID sets don't match
    """
    if not _should_validate():
        return
    set1 = set(ids1)
    set2 = set(ids2)
    if set1 != set2:
        n_only1 = len(set1 - set2)
        n_only2 = len(set2 - set1)
        msg = (
            f"{name1} and {name2} have misaligned stock IDs: "
            f"{n_only1} only in {name1}, {n_only2} only in {name2}"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_alignment_by_id(
    arr: np.ndarray,
    ids: Sequence[int] | np.ndarray,
    ref_ids: Sequence[int] | np.ndarray,
    name: str,
) -> None:
    """
    Assert array rows align with reference IDs (subset check).

    :param arr (np.ndarray): Array whose rows should align with ids
    :param ids (Sequence[int] | np.ndarray): IDs corresponding to arr rows
    :param ref_ids (Sequence[int] | np.ndarray): Reference IDs (superset)
    :param name (str): Name for error messages

    :raises AssertionError: If arr/ids rows don't form subset of ref_ids
    """
    if not _should_validate():
        return
    ids_set = set(ids)
    ref_set = set(ref_ids)
    missing = ids_set - ref_set
    if missing:
        msg = f"{name}: {len(missing)} IDs not in reference set"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    if len(ids) != arr.shape[0]:
        msg = f"{name}: IDs length {len(ids)} != array rows {arr.shape[0]}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_matrix_square(arr: np.ndarray, name: str) -> None:
    """
    Assert matrix is square.

    :param arr (np.ndarray): Matrix to validate
    :param name (str): Name for error messages

    :raises AssertionError: If matrix is not square
    """
    if not _should_validate():
        return
    if arr.ndim != 2:
        msg = f"{name} must be 2D, got {arr.ndim}D"
    elif arr.shape[0] != arr.shape[1]:
        msg = f"{name} must be square, got shape {arr.shape}"
    else:
        return

    if _is_strict():
        raise AssertionError(msg)
    else:
        logger.warning(msg)


def assert_covariance_valid(Sigma: np.ndarray, name: str) -> None:
    """
    Assert matrix is a valid covariance (2D, square, symmetric, PSD, finite).

    :param Sigma (np.ndarray): Covariance matrix to validate
    :param name (str): Name for error messages

    :raises AssertionError: If any covariance property is violated
    """
    if not _should_validate():
        return
    # Check 2D
    if Sigma.ndim != 2:
        msg = f"{name} must be 2D, got {Sigma.ndim}D"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    # Check square
    if Sigma.shape[0] != Sigma.shape[1]:
        msg = f"{name} must be square, got shape {Sigma.shape}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    # Check finite
    if not np.isfinite(Sigma).all():
        msg = f"{name} contains NaN/Inf"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    # Check symmetric
    if not np.allclose(Sigma, Sigma.T, rtol=1e-8, atol=1e-10):
        max_diff = float(np.max(np.abs(Sigma - Sigma.T)))
        msg = f"{name} is not symmetric (max diff: {max_diff:.2e})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    # Check PSD
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eig = float(eigenvalues.min())
    if min_eig < -1e-10:
        msg = f"{name} is not PSD (min eigenvalue: {min_eig:.2e})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_positive_definite(
    Sigma: np.ndarray,
    name: str,
    tol: float = 1e-10,
) -> None:
    """
    Assert matrix is positive definite (eigenvalues > tol, stricter than PSD).

    :param Sigma (np.ndarray): Matrix to validate
    :param name (str): Name for error messages
    :param tol (float): Minimum eigenvalue (must be > 0)

    :raises AssertionError: If any eigenvalue is <= tol
    """
    if not _should_validate():
        return
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eig = float(eigenvalues.min())
    if min_eig <= tol:
        msg = f"{name} is not positive definite (min eigenvalue: {min_eig:.2e}, tol: {tol:.2e})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_returns_valid(
    returns_df: pd.DataFrame,
    name: str,
    max_abs_return: float = 2.0,
) -> None:
    """
    Assert returns DataFrame is valid (finite and bounded).

    :param returns_df (pd.DataFrame): Returns DataFrame
    :param name (str): Name for error messages
    :param max_abs_return (float): Maximum absolute return (default 2.0 = 200%)

    :raises AssertionError: If returns contain NaN/Inf or exceed bounds
    """
    if not _should_validate():
        return
    values = returns_df.values
    # Check for infinite values (NaN allowed in sparse returns)
    if np.isinf(values).any():
        n_inf = int(np.isinf(values).sum())
        msg = f"{name} contains {n_inf} infinite values"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    # Check bounds on finite values
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) > 0:
        abs_max = float(np.max(np.abs(finite_vals)))
        if abs_max > max_abs_return:
            msg = f"{name} has returns exceeding {max_abs_return}: max_abs={abs_max:.4f}"
            if _is_strict():
                raise AssertionError(msg)
            else:
                logger.warning(msg)


def assert_weights_valid(
    w: np.ndarray,
    name: str,
    w_min: float | None = None,
    w_max: float | None = None,
) -> None:
    """
    Assert portfolio weights are valid (sum=1, finite, within bounds).

    :param w (np.ndarray): Weight vector
    :param name (str): Name for error messages
    :param w_min (float | None): Minimum weight (optional)
    :param w_max (float | None): Maximum weight (optional)

    :raises AssertionError: If weights are invalid
    """
    if not _should_validate():
        return
    # Check finite
    if not np.isfinite(w).all():
        msg = f"{name} contains NaN/Inf"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    # Check sum to 1
    w_sum = float(np.sum(w))
    if abs(w_sum - 1.0) > 1e-3:
        msg = f"{name} do not sum to 1: sum={w_sum:.6f}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    # Check bounds
    if w_min is not None:
        w_actual_min = float(np.min(w))
        if w_actual_min < w_min - 1e-8:
            msg = f"{name} has weight {w_actual_min:.6f} below w_min={w_min}"
            if _is_strict():
                raise AssertionError(msg)
            else:
                logger.warning(msg)
    if w_max is not None:
        w_actual_max = float(np.max(w))
        if w_actual_max > w_max + 1e-8:
            msg = f"{name} has weight {w_actual_max:.6f} above w_max={w_max}"
            if _is_strict():
                raise AssertionError(msg)
            else:
                logger.warning(msg)


def assert_tensor_shape(
    t: torch.Tensor,
    expected: tuple[int, ...],
    name: str,
) -> None:
    """
    Assert PyTorch tensor has expected shape.

    :param t (torch.Tensor): Tensor to validate
    :param expected (tuple[int, ...]): Expected shape (use -1 for any)
    :param name (str): Name for error messages

    :raises AssertionError: If shape doesn't match
    """
    if not _should_validate():
        return
    actual = tuple(t.shape)
    if len(actual) != len(expected):
        msg = f"{name} has {len(actual)} dims, expected {len(expected)}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e != -1 and a != e:
            msg = f"{name} dim {i} is {a}, expected {e}"
            if _is_strict():
                raise AssertionError(msg)
            else:
                logger.warning(msg)
                return


def assert_fold_consistency(
    fold: dict[str, Any],
    name: str,
) -> None:
    """
    Assert walk-forward fold has consistent dates.

    :param fold (dict): Fold dictionary with train_start, train_end, val_start, val_end, oos_start, oos_end
    :param name (str): Name for error messages

    :raises AssertionError: If fold dates are inconsistent
    """
    if not _should_validate():
        return
    required_keys = ["train_start", "train_end", "oos_start", "oos_end"]
    for key in required_keys:
        if key not in fold:
            msg = f"{name} missing required key: {key}"
            if _is_strict():
                raise AssertionError(msg)
            else:
                logger.warning(msg)
                return

    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    oos_start = pd.Timestamp(fold["oos_start"])
    oos_end = pd.Timestamp(fold["oos_end"])

    # Check ordering
    if train_start >= train_end:
        msg = f"{name}: train_start >= train_end"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    if oos_start >= oos_end:
        msg = f"{name}: oos_start >= oos_end"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    # Check no overlap (allows gap for embargo)
    if train_end > oos_start:
        msg = f"{name}: train_end {train_end} > oos_start {oos_start}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_no_lookahead(
    train_end: pd.Timestamp | str,
    oos_start: pd.Timestamp | str,
    name: str,
    min_embargo_days: int = 0,
) -> None:
    """
    Assert no look-ahead bias between training end and OOS start.

    :param train_end (pd.Timestamp | str): Training period end date
    :param oos_start (pd.Timestamp | str): OOS period start date
    :param name (str): Name for error messages
    :param min_embargo_days (int): Minimum gap in days (default 0)

    :raises AssertionError: If train_end + embargo > oos_start
    """
    if not _should_validate():
        return
    te = pd.Timestamp(train_end)
    os = pd.Timestamp(oos_start)
    gap = (os - te).days
    if gap < min_embargo_days:
        msg = (
            f"{name}: potential look-ahead bias, train_end={te.date()}, "
            f"oos_start={os.date()}, gap={gap} days < min_embargo={min_embargo_days}"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_if_nan_fraction_exceeds(
    arr: np.ndarray | pd.DataFrame,
    threshold: float,
    name: str,
) -> bool:
    """
    Warn if NaN fraction exceeds threshold (non-blocking).

    :param arr (np.ndarray | pd.DataFrame): Array to check
    :param threshold (float): Maximum acceptable NaN fraction (0-1)
    :param name (str): Name for warning messages

    :return (bool): True if NaN fraction exceeds threshold
    """
    if not _should_validate():
        return False
    if isinstance(arr, pd.DataFrame):
        values = arr.values
    else:
        values = arr
    nan_frac = float(np.isnan(values).sum() / values.size) if values.size > 0 else 0.0
    if nan_frac > threshold:
        warnings.warn(
            f"{name} has {nan_frac:.1%} NaN values, exceeds threshold {threshold:.1%}",
            stacklevel=2,
        )
        return True
    return False


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 2 - Training Audit)
# ===========================================================================


def assert_non_empty_dataframe(
    df: pd.DataFrame,
    name: str,
) -> None:
    """
    Assert DataFrame has at least one row.

    :param df (pd.DataFrame): DataFrame to validate
    :param name (str): Name for error messages

    :raises AssertionError: If DataFrame is empty
    """
    if not _should_validate():
        return
    if len(df) == 0:
        msg = f"{name} is empty (0 rows)"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_tensor_bounds(
    t: torch.Tensor,
    low: float,
    high: float,
    name: str,
) -> None:
    """
    Assert PyTorch tensor values within [low, high].

    :param t (torch.Tensor): Tensor to validate
    :param low (float): Minimum allowed value
    :param high (float): Maximum allowed value
    :param name (str): Name for error messages

    :raises AssertionError: If any value is outside bounds
    """
    if not _should_validate():
        return
    t_cpu = t.detach().cpu()
    t_min = float(t_cpu.min())
    t_max = float(t_cpu.max())
    if t_min < low or t_max > high:
        msg = f"{name} values out of bounds [{low}, {high}]: min={t_min:.4g}, max={t_max:.4g}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_if_tensor_extreme(
    t: torch.Tensor,
    threshold: float,
    name: str,
) -> bool:
    """
    Warn if tensor has values exceeding threshold (non-blocking).

    :param t (torch.Tensor): Tensor to check
    :param threshold (float): Maximum acceptable absolute value
    :param name (str): Name for warning messages

    :return (bool): True if any value exceeds threshold
    """
    if not _should_validate():
        return False
    t_cpu = t.detach().cpu()
    max_abs = float(t_cpu.abs().max())
    if max_abs > threshold:
        warnings.warn(
            f"{name} has extreme values: max_abs={max_abs:.4g} > threshold={threshold}",
            stacklevel=2,
        )
        return True
    return False


def assert_embargo_date_in_index(
    date: pd.Timestamp,
    index: pd.DatetimeIndex,
    name: str,
) -> None:
    """
    Assert embargo/rebalancing date exists in the OOS date index.

    :param date (pd.Timestamp): Date to check
    :param index (pd.DatetimeIndex): Available dates
    :param name (str): Name for error messages

    :raises AssertionError: If date not in index
    """
    if not _should_validate():
        return
    if date not in index:
        msg = f"{name}: date {date.date()} not in OOS date index"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_growth_finite(
    growth: np.ndarray,
    name: str,
) -> None:
    """
    Assert portfolio growth factors are finite and positive.

    :param growth (np.ndarray): Growth factors (w * exp(r))
    :param name (str): Name for error messages

    :raises AssertionError: If any growth factor is non-finite or negative
    """
    if not _should_validate():
        return
    if not np.isfinite(growth).all():
        n_nonfinite = int((~np.isfinite(growth)).sum())
        msg = f"{name}: {n_nonfinite} non-finite growth factors"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    if np.any(growth < 0):
        n_neg = int((growth < 0).sum())
        msg = f"{name}: {n_neg} negative growth factors"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_loss_explosion(
    current_loss: float,
    initial_loss: float,
    multiplier: float,
    epoch: int,
) -> bool:
    """
    Warn if loss has exploded relative to initial value.

    :param current_loss (float): Current epoch loss
    :param initial_loss (float): Loss from first epoch
    :param multiplier (float): Maximum acceptable loss multiplier
    :param epoch (int): Current epoch number

    :return (bool): True if loss has exploded
    """
    if not _should_validate():
        return False
    if initial_loss > 0 and current_loss > multiplier * initial_loss:
        warnings.warn(
            f"Loss explosion at epoch {epoch}: {current_loss:.4f} > "
            f"{multiplier}x initial ({initial_loss:.4f})",
            stacklevel=2,
        )
        return True
    return False


def assert_active_units_valid(
    au: int,
    k: int,
    name: str,
) -> None:
    """
    Assert number of active units is within valid range.

    :param au (int): Number of active units
    :param k (int): Maximum latent dimension K
    :param name (str): Name for error messages

    :raises AssertionError: If AU > K or AU < 0
    """
    if not _should_validate():
        return
    if au < 0 or au > k:
        msg = f"{name}: AU={au} outside valid range [0, K={k}]"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_ann_vol_positive(
    ann_vol: float,
    name: str,
    min_vol: float = 1e-10,
) -> None:
    """
    Assert annualized volatility is positive.

    :param ann_vol (float): Annualized volatility
    :param name (str): Name for error messages
    :param min_vol (float): Minimum acceptable volatility

    :raises AssertionError: If volatility is non-positive or non-finite
    """
    if not _should_validate():
        return
    if not np.isfinite(ann_vol) or ann_vol < min_vol:
        msg = f"{name}: ann_vol={ann_vol:.6g} is invalid (must be >= {min_vol})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 1 Extension - Data Integrity)
# ===========================================================================


def assert_positive_prices(
    prices: np.ndarray | pd.Series | pd.DataFrame,
    name: str,
    min_price: float = 0.0,
) -> None:
    """
    Assert all prices are strictly positive (required before log()).

    :param prices (np.ndarray | pd.Series | pd.DataFrame): Price data
    :param name (str): Name for error messages
    :param min_price (float): Minimum price threshold (default 0, prices must be > min_price)

    :raises AssertionError: If any price is <= min_price
    """
    if not _should_validate():
        return
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        values = prices.values
    else:
        values = prices
    # Ignore NaN values (sparse data)
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) == 0:
        return
    min_val = float(np.min(finite_vals))
    if min_val <= min_price:
        msg = f"{name}: non-positive price detected (min={min_val:.6g}, must be > {min_price})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_log_input_positive(
    values: np.ndarray | pd.Series | pd.DataFrame,
    name: str,
) -> None:
    """
    Assert all values are strictly positive before applying log().

    Guards against log(0) or log(negative) which produce -Inf or NaN.

    :param values (np.ndarray | pd.Series | pd.DataFrame): Values to check
    :param name (str): Name for error messages

    :raises AssertionError: If any finite value is <= 0
    """
    if not _should_validate():
        return
    if isinstance(values, (pd.Series, pd.DataFrame)):
        arr = values.values
    else:
        arr = values
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    if len(finite_vals) == 0:
        return
    if np.any(finite_vals <= 0):
        n_non_positive = int(np.sum(finite_vals <= 0))
        min_val = float(np.min(finite_vals))
        msg = f"{name}: {n_non_positive} non-positive values before log() (min={min_val:.6g})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_volume_non_negative(
    volume: np.ndarray | pd.Series | pd.DataFrame,
    name: str,
) -> None:
    """
    Assert trading volume is non-negative.

    :param volume (np.ndarray | pd.Series | pd.DataFrame): Volume data
    :param name (str): Name for error messages

    :raises AssertionError: If any volume is negative
    """
    if not _should_validate():
        return
    if isinstance(volume, (pd.Series, pd.DataFrame)):
        arr = volume.values
    else:
        arr = volume
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return
    min_vol = float(np.min(finite_vals))
    if min_vol < 0:
        n_negative = int(np.sum(finite_vals < 0))
        msg = f"{name}: {n_negative} negative volume values (min={min_vol:.6g})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_z_score_normalized(
    arr: np.ndarray,
    name: str,
    mean_tol: float = 0.1,
    std_tol: float = 0.1,
) -> None:
    """
    Assert array is z-score normalized (mean ≈ 0, std ≈ 1).

    :param arr (np.ndarray): Z-scored data
    :param name (str): Name for error messages
    :param mean_tol (float): Maximum acceptable |mean|
    :param std_tol (float): Maximum acceptable |std - 1|

    :raises AssertionError: If mean or std deviate too much from 0, 1
    """
    if not _should_validate():
        return
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 10:
        return  # Too few values to assess normalization
    mean_val = float(np.mean(finite_vals))
    std_val = float(np.std(finite_vals))
    issues = []
    if abs(mean_val) > mean_tol:
        issues.append(f"mean={mean_val:.4f} (expected ~0)")
    if abs(std_val - 1.0) > std_tol:
        issues.append(f"std={std_val:.4f} (expected ~1)")
    if issues:
        msg = f"{name}: z-score normalization issue: {', '.join(issues)}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_crisis_fraction_bounds(
    fractions: np.ndarray | torch.Tensor,
    name: str,
) -> None:
    """
    Assert crisis fractions are within [0, 1].

    :param fractions (np.ndarray | torch.Tensor): Crisis label fractions
    :param name (str): Name for error messages

    :raises AssertionError: If any fraction is outside [0, 1]
    """
    if not _should_validate():
        return
    if isinstance(fractions, torch.Tensor):
        arr = fractions.detach().cpu().numpy()
    else:
        arr = fractions
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return
    min_val = float(np.min(finite_vals))
    max_val = float(np.max(finite_vals))
    if min_val < 0 or max_val > 1:
        msg = f"{name}: crisis fractions out of [0, 1]: min={min_val:.4f}, max={max_val:.4f}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_if_price_discontinuity(
    returns: np.ndarray | pd.DataFrame,
    threshold: float = 0.5,
    name: str = "returns",
) -> bool:
    """
    Warn if returns contain suspicious jumps (possible data errors).

    Detects single-day returns exceeding threshold, which may indicate
    stock splits not adjusted, corporate actions, or data errors.

    :param returns (np.ndarray | pd.DataFrame): Log-returns data
    :param threshold (float): Maximum expected single-day return (default 0.5 = 50%)
    :param name (str): Name for warning messages

    :return (bool): True if discontinuities detected
    """
    if not _should_validate():
        return False
    if isinstance(returns, pd.DataFrame):
        arr = returns.values
    else:
        arr = returns
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return False
    n_extreme = int(np.sum(np.abs(finite_vals) > threshold))
    if n_extreme > 0:
        pct_extreme = 100 * n_extreme / len(finite_vals)
        warnings.warn(
            f"{name}: {n_extreme} returns exceed |{threshold:.0%}| "
            f"({pct_extreme:.3f}% of observations) - check for data errors",
            stacklevel=2,
        )
        return True
    return False


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 2 Extension - Numerical Stability)
# ===========================================================================


def assert_kl_non_negative(
    kl: torch.Tensor | float,
    name: str,
    tol: float = -1e-6,
) -> None:
    """
    Assert KL divergence is non-negative (theory guarantee).

    KL divergence is always >= 0 by definition. Negative values indicate
    numerical issues or implementation bugs.

    :param kl (torch.Tensor | float): KL divergence value
    :param name (str): Name for error messages
    :param tol (float): Tolerance for small numerical errors

    :raises AssertionError: If KL < tol
    """
    if not _should_validate():
        return
    if isinstance(kl, torch.Tensor):
        kl_val = float(kl.detach().cpu().item())
    else:
        kl_val = float(kl)
    if kl_val < tol:
        msg = f"{name}: KL divergence is negative ({kl_val:.6g} < {tol})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_reconstruction_bounded(
    recon: torch.Tensor | float,
    max_val: float = 1e6,
    name: str = "reconstruction_loss",
) -> None:
    """
    Assert reconstruction loss is bounded (not exploding).

    :param recon (torch.Tensor | float): Reconstruction loss value
    :param max_val (float): Maximum acceptable value
    :param name (str): Name for error messages

    :raises AssertionError: If reconstruction loss exceeds max_val
    """
    if not _should_validate():
        return
    if isinstance(recon, torch.Tensor):
        recon_val = float(recon.detach().cpu().item())
    else:
        recon_val = float(recon)
    if not np.isfinite(recon_val):
        msg = f"{name}: non-finite value ({recon_val})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
    elif recon_val > max_val:
        msg = f"{name}: loss exploding ({recon_val:.4g} > {max_val:.4g})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_gradient_finite(
    grad: torch.Tensor | np.ndarray,
    name: str,
) -> None:
    """
    Assert gradient contains no NaN/Inf values before backward pass.

    :param grad (torch.Tensor | np.ndarray): Gradient tensor/array
    :param name (str): Name for error messages

    :raises AssertionError: If gradient contains NaN or Inf
    """
    if not _should_validate():
        return
    if isinstance(grad, torch.Tensor):
        is_finite = bool(torch.isfinite(grad).all().item())
        n_nonfinite = int((~torch.isfinite(grad)).sum().item())
    else:
        is_finite = bool(np.isfinite(grad).all())
        n_nonfinite = int((~np.isfinite(grad)).sum())

    if not is_finite:
        msg = f"{name}: gradient contains {n_nonfinite} non-finite values"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_eigenvalue_spectrum_valid(
    eigenvalues: np.ndarray,
    name: str,
    min_val: float = -1e-10,
) -> None:
    """
    Assert eigenvalue spectrum is valid (all >= min_val, finite).

    :param eigenvalues (np.ndarray): Eigenvalue array (sorted or unsorted)
    :param name (str): Name for error messages
    :param min_val (float): Minimum allowed eigenvalue

    :raises AssertionError: If any eigenvalue is below min_val or non-finite
    """
    if not _should_validate():
        return
    if not np.isfinite(eigenvalues).all():
        n_nonfinite = int((~np.isfinite(eigenvalues)).sum())
        msg = f"{name}: {n_nonfinite} non-finite eigenvalues"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    min_eig = float(eigenvalues.min())
    if min_eig < min_val:
        msg = f"{name}: eigenvalue {min_eig:.6g} below threshold {min_val:.6g}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_cholesky_condition(
    L: np.ndarray,
    max_cond: float = 1e8,
    name: str = "Cholesky",
) -> None:
    """
    Assert Cholesky factor is well-conditioned.

    :param L (np.ndarray): Lower-triangular Cholesky factor
    :param max_cond (float): Maximum acceptable condition number
    :param name (str): Name for error messages

    :raises AssertionError: If condition number exceeds max_cond
    """
    if not _should_validate():
        return
    diag = np.abs(np.diag(L))
    if len(diag) == 0 or np.min(diag) < 1e-14:
        cond_num = np.inf
    else:
        cond_num = float(np.max(diag) / np.min(diag))
    if cond_num > max_cond:
        msg = f"{name}: Cholesky condition number {cond_num:.2e} exceeds {max_cond:.2e}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_armijo_params_valid(
    c: float,
    rho: float,
    name: str = "Armijo",
) -> None:
    """
    Assert Armijo backtracking parameters are valid.

    :param c (float): Armijo sufficient decrease parameter (must be in (0, 0.5))
    :param rho (float): Backtracking contraction factor (must be in (0, 1))
    :param name (str): Name for error messages

    :raises AssertionError: If parameters are outside valid ranges
    """
    if not _should_validate():
        return
    issues = []
    if not (0 < c < 0.5):
        issues.append(f"c={c} not in (0, 0.5)")
    if not (0 < rho < 1):
        issues.append(f"rho={rho} not in (0, 1)")
    if issues:
        msg = f"{name}: invalid parameters: {', '.join(issues)}"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_if_loss_component_imbalance(
    kl: float,
    recon: float,
    max_ratio: float = 100.0,
    name: str = "loss",
) -> bool:
    """
    Warn if KL and reconstruction loss are severely imbalanced.

    Severe imbalance (KL >> recon or recon >> KL) indicates training issues.

    :param kl (float): KL divergence term value
    :param recon (float): Reconstruction loss term value
    :param max_ratio (float): Maximum acceptable ratio between components
    :param name (str): Name for warning messages

    :return (bool): True if imbalance detected
    """
    if not _should_validate():
        return False
    if kl <= 0 or recon <= 0:
        return False  # Can't compute ratio with zero/negative
    ratio = max(kl / recon, recon / kl)
    if ratio > max_ratio:
        warnings.warn(
            f"{name}: loss component imbalance (KL={kl:.4g}, recon={recon:.4g}, "
            f"ratio={ratio:.1f} > {max_ratio})",
            stacklevel=2,
        )
        return True
    return False


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 3 Extension - Cross-Module Alignment)
# ===========================================================================


def assert_universe_snapshot_consistency(
    train_ids: Sequence[int] | np.ndarray,
    test_ids: Sequence[int] | np.ndarray,
    min_overlap: float = 0.8,
    name: str = "universe",
) -> None:
    """
    Assert universe overlap between train and test is sufficient.

    Prevents excessive universe drift between fold segments.

    :param train_ids (Sequence[int] | np.ndarray): Training universe stock IDs
    :param test_ids (Sequence[int] | np.ndarray): Test universe stock IDs
    :param min_overlap (float): Minimum required Jaccard similarity (0-1)
    :param name (str): Name for error messages

    :raises AssertionError: If overlap is below min_overlap
    """
    if not _should_validate():
        return
    set_train = set(train_ids)
    set_test = set(test_ids)
    if len(set_train) == 0 or len(set_test) == 0:
        return  # Empty universe, can't compute overlap
    intersection = len(set_train & set_test)
    union = len(set_train | set_test)
    jaccard = intersection / union if union > 0 else 0.0
    if jaccard < min_overlap:
        msg = (
            f"{name}: insufficient universe overlap (Jaccard={jaccard:.2%} < {min_overlap:.0%}), "
            f"train={len(set_train)}, test={len(set_test)}, common={intersection}"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_factor_idio_trace_match(
    Sigma_f: np.ndarray,
    D_eps: np.ndarray,
    Sigma_total: np.ndarray,
    tol: float = 0.01,
    name: str = "covariance",
) -> None:
    """
    Assert trace(factor) + trace(idio) ≈ trace(total) decomposition holds.

    Validates that factor and idiosyncratic covariances properly decompose
    the total covariance.

    :param Sigma_f (np.ndarray): Factor covariance contribution (n x n)
    :param D_eps (np.ndarray): Idiosyncratic variances (diagonal, n)
    :param Sigma_total (np.ndarray): Total covariance matrix (n x n)
    :param tol (float): Relative tolerance for trace mismatch
    :param name (str): Name for error messages

    :raises AssertionError: If trace decomposition doesn't hold
    """
    if not _should_validate():
        return
    trace_f = float(np.trace(Sigma_f))
    trace_idio = float(np.sum(D_eps)) if D_eps.ndim == 1 else float(np.trace(D_eps))
    trace_total = float(np.trace(Sigma_total))
    if trace_total == 0:
        return  # Can't check relative error with zero trace
    trace_sum = trace_f + trace_idio
    rel_error = abs(trace_sum - trace_total) / abs(trace_total)
    if rel_error > tol:
        msg = (
            f"{name}: trace decomposition mismatch "
            f"(factor={trace_f:.4g} + idio={trace_idio:.4g} = {trace_sum:.4g} vs "
            f"total={trace_total:.4g}, rel_error={rel_error:.2%})"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_embargo_in_calendar(
    embargo_date: pd.Timestamp | str,
    trading_dates: pd.DatetimeIndex,
    name: str = "embargo",
) -> None:
    """
    Assert embargo date falls on an actual trading day.

    :param embargo_date (pd.Timestamp | str): Embargo/rebalancing date
    :param trading_dates (pd.DatetimeIndex): Trading calendar dates
    :param name (str): Name for error messages

    :raises AssertionError: If embargo_date is not in trading calendar
    """
    if not _should_validate():
        return
    date: pd.Timestamp = pd.Timestamp(embargo_date)  # type: ignore[assignment]
    if date not in trading_dates:
        # Find nearest trading day for helpful error message
        if len(trading_dates) > 0:
            idx = int(np.searchsorted(np.asarray(trading_dates), np.datetime64(date)))
            if idx < len(trading_dates):
                nearest: pd.Timestamp = pd.Timestamp(trading_dates[idx])  # type: ignore[assignment]
            else:
                nearest = pd.Timestamp(trading_dates[-1])  # type: ignore[assignment]
            msg = f"{name}: date {date.date()} not in trading calendar (nearest: {nearest.date()})"
        else:
            msg = f"{name}: date {date.date()} not in trading calendar (empty calendar)"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_exposure_matrix_alignment(
    B: np.ndarray,
    stock_ids: Sequence[int] | np.ndarray,
    universe_ids: Sequence[int] | np.ndarray,
    name: str = "exposure_matrix",
) -> None:
    """
    Assert exposure matrix B rows align with universe.

    :param B (np.ndarray): Exposure matrix (n_stocks x K)
    :param stock_ids (Sequence[int] | np.ndarray): Stock IDs for B rows
    :param universe_ids (Sequence[int] | np.ndarray): Expected universe IDs
    :param name (str): Name for error messages

    :raises AssertionError: If B rows don't match universe
    """
    if not _should_validate():
        return
    if B.shape[0] != len(stock_ids):
        msg = f"{name}: B rows ({B.shape[0]}) != stock_ids length ({len(stock_ids)})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
            return
    set_B = set(stock_ids)
    set_universe = set(universe_ids)
    if set_B != set_universe:
        missing_in_B = set_universe - set_B
        extra_in_B = set_B - set_universe
        msg = (
            f"{name}: B stock IDs don't match universe "
            f"(missing: {len(missing_in_B)}, extra: {len(extra_in_B)})"
        )
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def warn_if_universe_drift(
    current_ids: Sequence[int] | np.ndarray,
    previous_ids: Sequence[int] | np.ndarray,
    threshold: float = 0.3,
    name: str = "universe",
) -> bool:
    """
    Warn if universe turnover exceeds threshold between periods.

    :param current_ids (Sequence[int] | np.ndarray): Current period stock IDs
    :param previous_ids (Sequence[int] | np.ndarray): Previous period stock IDs
    :param threshold (float): Maximum acceptable turnover (0-1)
    :param name (str): Name for warning messages

    :return (bool): True if drift exceeds threshold
    """
    if not _should_validate():
        return False
    set_current = set(current_ids)
    set_previous = set(previous_ids)
    if len(set_previous) == 0:
        return False  # No previous period to compare
    # Turnover = fraction of previous universe that changed
    unchanged = len(set_current & set_previous)
    turnover = 1 - (unchanged / len(set_previous))
    if turnover > threshold:
        warnings.warn(
            f"{name}: universe drift {turnover:.1%} exceeds {threshold:.0%} "
            f"(prev={len(set_previous)}, curr={len(set_current)}, unchanged={unchanged})",
            stacklevel=2,
        )
        return True
    return False


# ===========================================================================
# NEW VALIDATION FUNCTIONS (Phase 4 Extension - Training Dynamics & Portfolio)
# ===========================================================================


def warn_if_au_collapsed(
    au: int,
    k: int,
    min_ratio: float = 0.05,
    max_ratio: float = 0.80,
    name: str = "AU",
) -> bool:
    """
    Warn if active unit ratio is too low (collapsed) or too high (no regularization).

    :param au (int): Number of active units
    :param k (int): Total latent dimensions K
    :param min_ratio (float): Minimum acceptable AU/K ratio
    :param max_ratio (float): Maximum acceptable AU/K ratio
    :param name (str): Name for warning messages

    :return (bool): True if AU ratio is outside acceptable range
    """
    if not _should_validate():
        return False
    if k <= 0:
        return False
    ratio = au / k
    if ratio < min_ratio:
        warnings.warn(
            f"{name}: AU collapsed ({au}/{k} = {ratio:.1%} < {min_ratio:.0%}) - "
            "latent space may be underutilized",
            stacklevel=2,
        )
        return True
    if ratio > max_ratio:
        warnings.warn(
            f"{name}: AU ratio high ({au}/{k} = {ratio:.1%} > {max_ratio:.0%}) - "
            "may indicate insufficient regularization",
            stacklevel=2,
        )
        return True
    return False


def warn_if_sigma_bounds_stuck(
    streak: int,
    max_streak: int = 5,
    name: str = "sigma_sq",
) -> bool:
    """
    Warn if sigma bounds are hit repeatedly (stuck at boundary).

    :param streak (int): Number of consecutive epochs hitting bounds
    :param max_streak (int): Maximum acceptable streak before warning
    :param name (str): Name for warning messages

    :return (bool): True if streak exceeds max_streak
    """
    if not _should_validate():
        return False
    if streak > max_streak:
        warnings.warn(
            f"{name}: bounds hit for {streak} consecutive epochs (> {max_streak}) - "
            "consider adjusting bounds or learning rate",
            stacklevel=2,
        )
        return True
    return False


def assert_sharpe_denominator_valid(
    vol: float,
    min_vol: float = 1e-6,
    name: str = "Sharpe",
) -> None:
    """
    Assert volatility is valid for Sharpe ratio computation.

    :param vol (float): Volatility (denominator)
    :param min_vol (float): Minimum acceptable volatility
    :param name (str): Name for error messages

    :raises AssertionError: If volatility is too small or non-finite
    """
    if not _should_validate():
        return
    if not np.isfinite(vol) or vol < min_vol:
        msg = f"{name}: invalid volatility {vol:.6g} for Sharpe computation (min={min_vol})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_sortino_observations_sufficient(
    n_downside: int,
    min_obs: int = 2,
    name: str = "Sortino",
) -> None:
    """
    Assert sufficient downside observations for Sortino ratio.

    :param n_downside (int): Number of downside observations
    :param min_obs (int): Minimum required observations
    :param name (str): Name for error messages

    :raises AssertionError: If n_downside < min_obs
    """
    if not _should_validate():
        return
    if n_downside < min_obs:
        msg = f"{name}: insufficient downside observations ({n_downside} < {min_obs})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_calmar_denominator_valid(
    max_dd: float,
    name: str = "Calmar",
) -> None:
    """
    Assert max drawdown is valid for Calmar ratio computation.

    :param max_dd (float): Maximum drawdown (should be > 0)
    :param name (str): Name for error messages

    :raises AssertionError: If max_dd is zero or non-finite
    """
    if not _should_validate():
        return
    if not np.isfinite(max_dd) or max_dd == 0:
        msg = f"{name}: invalid max drawdown {max_dd:.6g} for Calmar computation"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)


def assert_turnover_constraint_satisfied(
    turnover: float,
    tau_max: float,
    tol: float = 0.01,
    name: str = "turnover",
) -> None:
    """
    Assert turnover constraint is satisfied post-rebalance.

    :param turnover (float): Actual turnover
    :param tau_max (float): Maximum allowed turnover
    :param tol (float): Tolerance for constraint violation
    :param name (str): Name for error messages

    :raises AssertionError: If turnover exceeds tau_max + tol
    """
    if not _should_validate():
        return
    if turnover > tau_max + tol:
        msg = f"{name}: constraint violated ({turnover:.4f} > {tau_max} + {tol})"
        if _is_strict():
            raise AssertionError(msg)
        else:
            logger.warning(msg)
