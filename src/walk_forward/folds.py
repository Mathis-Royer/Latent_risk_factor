"""
Walk-forward fold scheduling.

~34 folds from Year 10 to Year 27 (6-month steps), plus holdout (Year 27-30).
Each fold specifies training, validation, embargo, and OOS periods.

Reference: ISD Section MOD-009 — Sub-task 1.
"""

import pandas as pd

from src.validation import assert_fold_consistency, assert_no_lookahead


def generate_fold_schedule(
    start_date: str,
    total_years: int = 30,
    min_training_years: int = 10,
    oos_months: int = 6,
    embargo_days: int = 21,
    holdout_years: int = 3,
    val_years: int = 2,
) -> list[dict[str, object]]:
    """
    Generate the walk-forward fold schedule.

    Fold k:
      - Training: [start, start + 10yr + k×0.5yr]
      - Validation (nested): [train_end - 2yr, train_end]
      - Embargo: 21 trading days after train_end
      - OOS: [embargo_end + 1, embargo_end + 6mo]

    Holdout: last ~3 years (Year total-3 to Year total).

    :param start_date (str): Data start date (YYYY-MM-DD)
    :param total_years (int): Total history length
    :param min_training_years (int): Minimum training window
    :param oos_months (int): OOS period length in months
    :param embargo_days (int): Embargo between training and OOS (trading days)
    :param holdout_years (int): Final holdout period in years
    :param val_years (int): Nested validation window in years

    :return folds (list[dict]): List of fold specifications
    """
    data_start = pd.Timestamp(start_date)
    data_end: pd.Timestamp = data_start + pd.DateOffset(years=total_years)  # type: ignore[assignment]
    holdout_start: pd.Timestamp = data_end - pd.DateOffset(years=holdout_years)  # type: ignore[assignment]

    folds: list[dict[str, object]] = []
    fold_id = 0

    # Walk-forward folds: 6-month steps from min_training_years to holdout
    current_train_end = data_start + pd.DateOffset(years=min_training_years)

    while current_train_end < holdout_start:
        # Training period
        train_start = data_start
        train_end = current_train_end

        # Nested validation
        val_start: pd.Timestamp = train_end - pd.DateOffset(years=val_years)  # type: ignore[assignment]
        val_end = train_end

        # Embargo
        embargo_start = train_end + pd.DateOffset(days=1)
        embargo_end = train_end + pd.offsets.BDay(embargo_days)

        # OOS
        oos_start = embargo_end + pd.DateOffset(days=1)
        oos_end = oos_start + pd.DateOffset(months=oos_months)

        # Ensure OOS doesn't exceed holdout
        if oos_end > holdout_start:
            oos_end = holdout_start

        fold_dict: dict[str, object] = {
            "fold_id": fold_id,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "embargo_start": str(embargo_start.date()),
            "embargo_end": str(embargo_end.date()),
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "is_holdout": False,
        }

        # Validate fold consistency and no look-ahead bias
        assert_fold_consistency(fold_dict, f"fold_{fold_id}")  # type: ignore[arg-type]
        assert_no_lookahead(
            str(train_end.date()), str(oos_start.date()), f"fold_{fold_id}",
        )

        folds.append(fold_dict)

        fold_id += 1
        current_train_end += pd.DateOffset(months=oos_months)

    # Holdout fold
    holdout_fold: dict[str, object] = {
        "fold_id": fold_id,
        "train_start": str(data_start.date()),
        "train_end": str(holdout_start.date()),
        "val_start": str((holdout_start - pd.DateOffset(years=val_years)).date()),
        "val_end": str(holdout_start.date()),
        "embargo_start": str((holdout_start + pd.DateOffset(days=1)).date()),
        "embargo_end": str((holdout_start + pd.offsets.BDay(embargo_days)).date()),
        "oos_start": str((holdout_start + pd.offsets.BDay(embargo_days) + pd.DateOffset(days=1)).date()),
        "oos_end": str(data_end.date()),
        "is_holdout": True,
    }
    assert_fold_consistency(holdout_fold, f"fold_{fold_id}_holdout")  # type: ignore[arg-type]
    assert_no_lookahead(
        str(holdout_start.date()),
        str((holdout_start + pd.offsets.BDay(embargo_days) + pd.DateOffset(days=1)).date()),
        f"fold_{fold_id}_holdout",
    )
    folds.append(holdout_fold)

    return folds


def validate_fold_schedule(
    folds: list[dict[str, object]],
) -> dict[str, bool]:
    """
    Validate fold schedule for common issues.

    :param folds (list[dict]): Fold specifications

    :return checks (dict): Validation results
    """
    checks: dict[str, bool] = {}

    # No training/OOS overlap
    no_overlap = True
    for fold in folds:
        train_end = pd.Timestamp(str(fold["train_end"]))
        oos_start = pd.Timestamp(str(fold["oos_start"]))
        if oos_start <= train_end:
            no_overlap = False
    checks["no_train_oos_overlap"] = no_overlap

    # Folds chronologically ordered
    sequential = True
    for i in range(1, len(folds)):
        if folds[i]["is_holdout"] or folds[i - 1]["is_holdout"]:
            continue
        prev_end = pd.Timestamp(str(folds[i - 1]["oos_end"]))
        curr_start = pd.Timestamp(str(folds[i]["oos_start"]))
        if curr_start < prev_end - pd.DateOffset(days=31):  # type: ignore[operator]
            sequential = False
    checks["chronological_order"] = sequential

    # Holdout untouched by walk-forward
    holdout_folds = [f for f in folds if f["is_holdout"]]
    wf_folds = [f for f in folds if not f["is_holdout"]]
    holdout_ok = True
    if holdout_folds and wf_folds:
        holdout_start = pd.Timestamp(str(holdout_folds[0]["oos_start"]))
        for wf in wf_folds:
            wf_oos_end = pd.Timestamp(str(wf["oos_end"]))
            if wf_oos_end > holdout_start:
                holdout_ok = False
    checks["holdout_untouched"] = holdout_ok

    return checks
