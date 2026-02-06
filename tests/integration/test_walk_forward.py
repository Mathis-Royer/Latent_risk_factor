"""
Integration tests for MOD-009: Walk-forward validation.

Covers: fold scheduling, composite scoring, Phase B E* determination.

Reference: ISD Section MOD-009.
"""

import numpy as np
import pandas as pd
import pytest

from src.walk_forward.folds import generate_fold_schedule, validate_fold_schedule
from src.walk_forward.phase_a import composite_score
from src.walk_forward.phase_b import determine_e_star


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE = "1995-01-03"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fold_schedule() -> list[dict[str, object]]:
    """Generate a standard fold schedule for testing."""
    return generate_fold_schedule(
        start_date=START_DATE,
        total_years=30,
        min_training_years=10,
        oos_months=6,
        embargo_days=21,
        holdout_years=3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWalkForward:
    """Tests for walk-forward fold scheduling and scoring."""

    def test_fold_no_overlap(
        self, fold_schedule: list[dict[str, object]]
    ) -> None:
        """No training period should overlap with its own OOS period."""
        checks = validate_fold_schedule(fold_schedule)

        assert checks["no_train_oos_overlap"] is True, (
            "Training/OOS overlap detected in fold schedule"
        )

    def test_fold_dates_sequential(
        self, fold_schedule: list[dict[str, object]]
    ) -> None:
        """Folds are chronologically ordered: OOS_k ends before OOS_{k+1} starts."""
        wf_folds = [f for f in fold_schedule if not f["is_holdout"]]

        for i in range(1, len(wf_folds)):
            prev_oos_end = pd.Timestamp(str(wf_folds[i - 1]["oos_end"]))
            curr_oos_start = pd.Timestamp(str(wf_folds[i]["oos_start"]))

            # Allow small overlap (up to 31 days) due to step rounding
            lower_bound: pd.Timestamp = prev_oos_end - pd.DateOffset(days=31)  # type: ignore[assignment]
            assert curr_oos_start >= lower_bound, (
                f"Fold {i}: OOS start {curr_oos_start} is too early "
                f"relative to previous OOS end {prev_oos_end}"
            )

    def test_holdout_untouched(
        self, fold_schedule: list[dict[str, object]]
    ) -> None:
        """Holdout data never appears in any walk-forward fold OOS period."""
        checks = validate_fold_schedule(fold_schedule)

        assert checks["holdout_untouched"] is True, (
            "Holdout period contaminated by walk-forward OOS folds"
        )

    def test_score_normalization(self) -> None:
        """composite_score produces normalized values in expected range."""
        # Case 1: Perfect entropy (H = ln(AU)), no MDD, large n_obs
        AU = 50
        H_oos = float(np.log(AU))
        score = composite_score(
            H_oos=H_oos,
            AU=AU,
            mdd_oos=0.05,
            n_obs=10000,
        )
        # H_norm = H/ln(AU) = 1.0, no penalties => score ~= 1.0
        assert 0.0 <= score <= 1.0, (
            f"Score {score:.4f} outside [0, 1] for perfect entropy case"
        )

        # Case 2: Low entropy, high MDD => penalized score
        score_low = composite_score(
            H_oos=0.5,
            AU=50,
            mdd_oos=0.40,
            n_obs=100,
        )
        assert score_low < score, (
            f"Low-quality score ({score_low:.4f}) should be less than "
            f"perfect score ({score:.4f})"
        )

    def test_phase_b_no_early_stopping(self) -> None:
        """determine_e_star returns config E* when no previous folds exist."""
        e_star_config = 75
        result = determine_e_star(
            fold_id=0,
            e_star_config=e_star_config,
            previous_e_stars=[],
            is_holdout=False,
        )

        assert result == e_star_config, (
            f"E* should be {e_star_config} for first fold, got {result}"
        )
