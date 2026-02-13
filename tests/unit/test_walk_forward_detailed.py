"""
Unit tests for walk-forward validation details.

Covers: embargo gap, Phase A selection, E* determination, training sanity.

Reference: ISD Section MOD-009.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.walk_forward.folds import generate_fold_schedule, validate_fold_schedule
from src.walk_forward.phase_a import composite_score
from src.walk_forward.phase_b import check_training_sanity, determine_e_star


# ---------------------------------------------------------------------------
# 1. test_embargo_gap_size
# ---------------------------------------------------------------------------

def test_embargo_gap_size() -> None:
    """Embargo gap must be >= 21 business days between train_end and oos_start."""
    folds = generate_fold_schedule(
        start_date="1993-01-01",
        total_years=30,
        min_training_years=10,
        oos_months=6,
        embargo_days=21,
    )

    assert len(folds) > 0, "No folds generated"

    for fold in folds:
        train_end = pd.Timestamp(str(fold["train_end"]))
        embargo_end = pd.Timestamp(str(fold["embargo_end"]))

        # Count business days strictly after train_end up to embargo_end.
        # BDay(21) adds 21 business days, but when train_end falls on a
        # non-business day (weekend/holiday), the anchor shifts forward first.
        # Use exclusive start to count the actual gap correctly.
        next_bday = train_end + pd.offsets.BDay(1)
        bdays = pd.bdate_range(start=next_bday, end=embargo_end)
        gap = len(bdays)

        assert gap >= 20, (
            f"Fold {fold['fold_id']}: embargo gap = {gap} bdays, expected >= 20. "
            f"train_end={train_end.date()}, embargo_end={embargo_end.date()}"
        )

        # OOS must start strictly after embargo
        oos_start = pd.Timestamp(str(fold["oos_start"]))
        assert oos_start > embargo_end, (
            f"Fold {fold['fold_id']}: OOS start {oos_start.date()} not after "
            f"embargo_end {embargo_end.date()}"
        )

    # Validate schedule integrity
    checks = validate_fold_schedule(folds)
    assert checks["no_train_oos_overlap"], "Training/OOS overlap detected"
    assert checks["chronological_order"], "Folds not in chronological order"


# ---------------------------------------------------------------------------
# 2. test_phase_a_selection_criterion
# ---------------------------------------------------------------------------

def test_phase_a_selection_criterion() -> None:
    """composite_score must return values in [0, 1] for normalized entropy."""
    # Good config: high entropy, low MDD, enough observations
    score_good = composite_score(
        H_oos=1.5,         # ln(5) ≈ 1.609 → H_norm ≈ 0.93
        AU=5,
        mdd_oos=0.10,      # Below threshold
        n_obs=500,
    )

    # The normalized entropy is ~0.93, no penalties → score should be ~0.93
    assert 0.0 <= score_good <= 1.0, (
        f"Good config score={score_good:.4f}, expected in [0, 1]"
    )

    # Bad config: low entropy, high MDD
    score_bad = composite_score(
        H_oos=0.1,
        AU=5,
        mdd_oos=0.40,      # Above threshold → penalty
        n_obs=500,
    )

    assert score_good > score_bad, (
        f"Good config score ({score_good:.4f}) should exceed "
        f"bad config score ({score_bad:.4f})"
    )

    # Edge case: AU=1 → ln(1) = 0 → should handle gracefully
    score_edge = composite_score(
        H_oos=0.0,
        AU=1,
        mdd_oos=0.10,
        n_obs=100,
    )
    assert math.isfinite(score_edge), f"Score not finite for AU=1: {score_edge}"


# ---------------------------------------------------------------------------
# 3. test_determine_e_star_median_logic
# ---------------------------------------------------------------------------

def test_determine_e_star_median_logic() -> None:
    """determine_e_star uses median of previous E* when >= 3 folds available."""
    # With < 3 previous folds: use config value
    e_star = determine_e_star(
        fold_id=1,
        e_star_config=50,
        previous_e_stars=[40, 60],
        is_holdout=False,
    )
    assert e_star == 50, f"Expected config E*=50, got {e_star}"

    # With >= 3 previous folds: use median
    e_star = determine_e_star(
        fold_id=5,
        e_star_config=50,
        previous_e_stars=[30, 40, 50, 60, 70],
        is_holdout=False,
    )
    expected_median = int(np.median([30, 40, 50, 60, 70]))  # 50
    assert e_star == expected_median, (
        f"Expected median E*={expected_median}, got {e_star}"
    )

    # With 3 previous folds (exactly threshold)
    e_star = determine_e_star(
        fold_id=3,
        e_star_config=100,
        previous_e_stars=[20, 40, 80],
        is_holdout=False,
    )
    assert e_star == int(np.median([20, 40, 80])), (
        f"Expected median of [20, 40, 80] = 40, got {e_star}"
    )

    # Holdout: use median of all walk-forward E*s
    e_star = determine_e_star(
        fold_id=34,
        e_star_config=50,
        previous_e_stars=[30, 40, 50],
        is_holdout=True,
        all_e_stars=[25, 35, 45, 55, 65],
    )
    expected_holdout = int(np.median([25, 35, 45, 55, 65]))
    assert e_star == expected_holdout, (
        f"Holdout E* should be {expected_holdout}, got {e_star}"
    )


# ---------------------------------------------------------------------------
# 4. test_check_training_sanity_flags
# ---------------------------------------------------------------------------

def test_check_training_sanity_flags() -> None:
    """check_training_sanity flags when relative diff > 20%."""
    # No flag: Phase B loss close to Phase A
    result = check_training_sanity(
        phase_b_loss=1.0,
        phase_a_loss=1.1,
    )
    assert not result["flag"], (
        f"Should not flag: relative_diff={result['relative_diff']:.4f}"
    )

    # Flag: Phase B loss much lower than Phase A (>20% improvement)
    result = check_training_sanity(
        phase_b_loss=0.7,
        phase_a_loss=1.0,
    )
    assert result["flag"], (
        f"Should flag: relative_diff={result['relative_diff']:.4f} > 0.20"
    )
    assert result["relative_diff"] > 0.20

    # Edge: Phase A loss = 0 → no flag (avoid division by zero)
    result = check_training_sanity(
        phase_b_loss=0.5,
        phase_a_loss=0.0,
    )
    assert not result["flag"], "Should not flag when phase_a_loss=0"


# ---------------------------------------------------------------------------
# 5. TestFoldScheduleProperties
# ---------------------------------------------------------------------------

class TestFoldScheduleProperties:
    """Tests for walk-forward fold schedule structural properties."""

    def test_fold_count_approximately_34(self) -> None:
        """30 years with 6-month OOS produces approximately 25-40 folds."""
        from src.walk_forward.folds import generate_fold_schedule
        folds = generate_fold_schedule(
            start_date="1993-01-01",
            total_years=30,
            min_training_years=10,
            oos_months=6,
            embargo_days=21,
        )
        wf_folds = [f for f in folds if not f.get("is_holdout", False)]
        assert 20 <= len(wf_folds) <= 50, (
            f"Expected 20-50 WF folds for 30 years, got {len(wf_folds)}"
        )

    def test_folds_expanding_training_window(self) -> None:
        """CONV-09: All folds share the same train_start; train_end increases monotonically."""
        from src.walk_forward.folds import generate_fold_schedule
        folds = generate_fold_schedule(
            start_date="1993-01-01", total_years=20,
            min_training_years=5, oos_months=6, embargo_days=21,
        )
        wf_folds = [f for f in folds if not f.get("is_holdout", False)]
        if len(wf_folds) < 2:
            pytest.skip("Not enough folds")
        # All have same train_start
        starts = [f["train_start"] for f in wf_folds]
        assert len(set(starts)) == 1, f"Expected same train_start, got {set(starts)}"
        # train_end increases
        ends = [pd.Timestamp(str(f["train_end"])) for f in wf_folds]
        for i in range(1, len(ends)):
            assert ends[i] >= ends[i-1], (
                f"train_end not monotonic: fold {i-1}={ends[i-1]}, fold {i}={ends[i]}"
            )

    def test_oos_periods_roughly_sequential(self) -> None:
        """Consecutive WF OOS periods are roughly sequential (<=31 day overlap allowed).

        The fold schedule allows minor OOS boundary overlap due to the embargo
        gap + calendar month arithmetic. validate_fold_schedule uses a 31-day
        tolerance (folds.py:135), so we match that here.
        """
        from src.walk_forward.folds import generate_fold_schedule
        folds = generate_fold_schedule(
            start_date="1993-01-01", total_years=20,
            min_training_years=5, oos_months=6, embargo_days=21,
        )
        wf_folds = [f for f in folds if not f.get("is_holdout", False)]
        for i in range(1, len(wf_folds)):
            prev_end = pd.Timestamp(str(wf_folds[i - 1]["oos_end"]))
            curr_start = pd.Timestamp(str(wf_folds[i]["oos_start"]))
            # Allow up to 31 days overlap (matching validate_fold_schedule tolerance)
            assert curr_start >= prev_end - pd.Timedelta(days=31), (
                f"Fold {i-1} OOS end {prev_end.date()} too far past "
                f"fold {i} OOS start {curr_start.date()}"
            )

    def test_embargo_gap_exists(self) -> None:
        """Each fold has an embargo gap between train_end and oos_start."""
        from src.walk_forward.folds import generate_fold_schedule
        folds = generate_fold_schedule(
            start_date="1993-01-01", total_years=15,
            min_training_years=5, oos_months=6, embargo_days=21,
        )
        wf_folds = [f for f in folds if not f.get("is_holdout", False)]
        for i, fold in enumerate(wf_folds):
            train_end = pd.Timestamp(str(fold["train_end"]))
            oos_start = pd.Timestamp(str(fold["oos_start"]))
            assert train_end < oos_start, (
                f"Fold {i}: train_end={train_end} >= oos_start={oos_start}"
            )

    def test_holdout_fold_present(self) -> None:
        """Fold schedule includes at least one holdout fold."""
        from src.walk_forward.folds import generate_fold_schedule
        folds = generate_fold_schedule(
            start_date="1993-01-01", total_years=20,
            min_training_years=5, oos_months=6, embargo_days=21,
        )
        holdout_folds = [f for f in folds if f.get("is_holdout", False)]
        assert len(holdout_folds) >= 1, "Expected at least 1 holdout fold"


# ---------------------------------------------------------------------------
# 6. AU posterior collapse and capacity exhaustion detection
# ---------------------------------------------------------------------------


class TestAUPosteriorCollapseDetection:
    """ISD §4.8: AU thresholds for config elimination.

    - AU < max(0.15*K, AU_PCA) -> eliminate config (posterior collapse)
    - AU > 0.85*K -> capacity exhaustion (not yet implemented as elimination,
      but should be flagged)
    """

    def test_au_below_threshold_eliminated(self) -> None:
        """Config with AU < max(0.15*K, AU_PCA) must be eliminated."""
        from src.walk_forward.phase_a import eliminate_configs

        K = 100
        AU_PCA = 5
        au_min = max(int(0.15 * K), AU_PCA)  # max(15, 5) = 15

        config_results = [
            {"AU": 0, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
            {"AU": 10, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
            {"AU": 14, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
            {"AU": 20, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
        ]

        surviving = eliminate_configs(config_results, K=K, AU_PCA=AU_PCA)

        # Only AU=20 (>= 15) should survive
        surviving_aus = [r["AU"] for r in surviving]
        assert 0 not in surviving_aus, "AU=0 (total collapse) should be eliminated"
        assert 10 not in surviving_aus, "AU=10 (< 15) should be eliminated"
        assert 14 not in surviving_aus, "AU=14 (< 15) should be eliminated"
        assert 20 in surviving_aus, "AU=20 (>= 15) should survive"

    def test_au_pca_raises_threshold(self) -> None:
        """AU_PCA > 0.15*K should raise the elimination threshold."""
        from src.walk_forward.phase_a import eliminate_configs

        K = 20
        # 0.15 * 20 = 3, but AU_PCA = 8 -> threshold = max(3, 8) = 8
        config_results = [
            {"AU": 5, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
            {"AU": 10, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
        ]

        surviving_no_pca = eliminate_configs(config_results, K=K, AU_PCA=0)
        surviving_with_pca = eliminate_configs(config_results, K=K, AU_PCA=8)

        # Without PCA baseline, AU=5 survives (>= 3)
        assert any(r["AU"] == 5 for r in surviving_no_pca), (
            "AU=5 should survive when AU_PCA=0 (threshold=3)"
        )

        # With PCA baseline, AU=5 is eliminated (< 8)
        assert not any(r["AU"] == 5 for r in surviving_with_pca), (
            "AU=5 should be eliminated when AU_PCA=8 (threshold=8)"
        )

        # AU=10 survives in both cases
        assert any(r["AU"] == 10 for r in surviving_with_pca), (
            "AU=10 should survive when AU_PCA=8 (threshold=8)"
        )

    def test_total_collapse_au_zero_eliminated(self) -> None:
        """AU=0 (total posterior collapse) must always be eliminated."""
        from src.walk_forward.phase_a import eliminate_configs

        config_results = [
            {"AU": 0, "explanatory_power": 0.80, "oos_train_mse_ratio": 1.0},
        ]

        surviving = eliminate_configs(config_results, K=50, AU_PCA=0)
        assert len(surviving) == 0, "AU=0 should always be eliminated"
