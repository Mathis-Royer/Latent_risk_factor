"""
Unit tests for Google Drive persistence in Colab.

Tests cover:
- is_colab() detection
- cleanup_old_runs() rotation logic
- list_runs() metadata extraction
- get_latest_run() selection
- setup_drive_persistence() local mode (no-op when not in Colab)
"""

import json
from pathlib import Path
import tempfile

from src.integration.colab_drive import (
    is_colab,
    cleanup_old_runs,
    list_runs,
    get_latest_run,
    setup_drive_persistence,
    _is_valid_run_name,
)


class TestIsColab:
    """Tests for is_colab() detection."""

    def test_returns_false_locally(self) -> None:
        """is_colab() returns False when not in Colab environment."""
        # When running tests locally, google.colab module is not available
        assert is_colab() is False


class TestIsValidRunName:
    """Tests for _is_valid_run_name() helper."""

    def test_valid_timestamp_format(self) -> None:
        """Valid YYYY-MM-DD_HHMMSS format returns True."""
        assert _is_valid_run_name("2026-02-22_143052") is True
        assert _is_valid_run_name("2025-01-01_000000") is True
        assert _is_valid_run_name("2030-12-31_235959") is True

    def test_invalid_formats(self) -> None:
        """Invalid formats return False."""
        # Wrong length
        assert _is_valid_run_name("2026-02-22") is False
        assert _is_valid_run_name("2026-02-22_14305") is False
        assert _is_valid_run_name("2026-02-22_1430520") is False

        # Wrong format
        assert _is_valid_run_name("not_a_timestamp") is False
        assert _is_valid_run_name("20260222_143052") is False
        assert _is_valid_run_name("2026/02/22_143052") is False

        # Invalid date/time values
        assert _is_valid_run_name("2026-13-22_143052") is False  # Month 13
        assert _is_valid_run_name("2026-02-32_143052") is False  # Day 32
        assert _is_valid_run_name("2026-02-22_250000") is False  # Hour 25


class TestCleanupOldRuns:
    """Tests for cleanup_old_runs() rotation."""

    def test_no_cleanup_when_under_limit(self) -> None:
        """No runs deleted when count <= keep_n."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            # Create 3 runs
            (runs_dir / "2026-02-20_100000").mkdir()
            (runs_dir / "2026-02-21_100000").mkdir()
            (runs_dir / "2026-02-22_100000").mkdir()

            deleted = cleanup_old_runs(runs_dir, keep_n=5)

            assert deleted == []
            assert len(list(runs_dir.iterdir())) == 3

    def test_cleanup_oldest_beyond_limit(self) -> None:
        """Oldest runs are deleted when count > keep_n."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            # Create 7 runs
            run_names = [
                "2026-02-15_100000",  # Oldest (to delete)
                "2026-02-16_100000",  # 2nd oldest (to delete)
                "2026-02-17_100000",  # Keep
                "2026-02-18_100000",  # Keep
                "2026-02-19_100000",  # Keep
                "2026-02-20_100000",  # Keep
                "2026-02-21_100000",  # Keep (newest)
            ]
            for name in run_names:
                (runs_dir / name).mkdir()

            deleted = cleanup_old_runs(runs_dir, keep_n=5)

            # Check 2 oldest were deleted
            assert len(deleted) == 2
            assert runs_dir / "2026-02-15_100000" in deleted
            assert runs_dir / "2026-02-16_100000" in deleted

            # Check 5 newest remain
            remaining = [d.name for d in runs_dir.iterdir()]
            assert len(remaining) == 5
            assert "2026-02-15_100000" not in remaining
            assert "2026-02-16_100000" not in remaining
            assert "2026-02-21_100000" in remaining

    def test_ignores_invalid_directories(self) -> None:
        """Non-timestamp directories are ignored in cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            # Create mixed directories
            (runs_dir / "2026-02-20_100000").mkdir()
            (runs_dir / "2026-02-21_100000").mkdir()
            (runs_dir / "not_a_run").mkdir()
            (runs_dir / "random_folder").mkdir()

            deleted = cleanup_old_runs(runs_dir, keep_n=1)

            # Only 1 valid run deleted
            assert len(deleted) == 1
            assert deleted[0].name == "2026-02-20_100000"

            # Invalid dirs still exist
            assert (runs_dir / "not_a_run").exists()
            assert (runs_dir / "random_folder").exists()

    def test_empty_directory(self) -> None:
        """Empty directory returns no deletions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            deleted = cleanup_old_runs(runs_dir, keep_n=5)
            assert deleted == []

    def test_nonexistent_directory(self) -> None:
        """Nonexistent directory returns no deletions."""
        deleted = cleanup_old_runs(Path("/nonexistent/path"), keep_n=5)
        assert deleted == []


class TestListRuns:
    """Tests for list_runs() metadata extraction."""

    def test_lists_valid_runs(self) -> None:
        """Lists all valid run directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            # Create 2 runs with some content
            run1 = runs_dir / "2026-02-20_100000"
            run2 = runs_dir / "2026-02-21_150000"
            run1.mkdir()
            run2.mkdir()

            # Add some files to run1
            (run1 / "test.npy").write_bytes(b"0" * 1024)  # 1KB

            runs = list_runs(runs_dir)

            assert len(runs) == 2
            # Sorted newest first
            assert runs[0]["name"] == "2026-02-21_150000"
            assert runs[1]["name"] == "2026-02-20_100000"
            assert runs[1]["size_mb"] > 0

    def test_extracts_stage_from_state_file(self) -> None:
        """Extracts last_stage from state JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            run = runs_dir / "2026-02-22_120000"
            run.mkdir()

            # Create state file
            state = {"current_stage": "COVARIANCE_DONE"}
            (run / "fold_00_state.json").write_text(json.dumps(state))

            runs = list_runs(runs_dir)

            assert len(runs) == 1
            assert runs[0]["last_stage"] == "COVARIANCE_DONE"

    def test_handles_missing_state_file(self) -> None:
        """Returns UNKNOWN when state file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            run = runs_dir / "2026-02-22_120000"
            run.mkdir()

            runs = list_runs(runs_dir)

            assert runs[0]["last_stage"] == "UNKNOWN"

    def test_ignores_invalid_directories(self) -> None:
        """Ignores directories that don't match timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            (runs_dir / "2026-02-22_120000").mkdir()
            (runs_dir / "invalid_folder").mkdir()
            (runs_dir / "test").mkdir()

            runs = list_runs(runs_dir)

            assert len(runs) == 1
            assert runs[0]["name"] == "2026-02-22_120000"


class TestGetLatestRun:
    """Tests for get_latest_run() selection."""

    def test_returns_most_recent(self) -> None:
        """Returns the most recent run by timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            (runs_dir / "2026-02-20_100000").mkdir()
            (runs_dir / "2026-02-22_150000").mkdir()  # Latest
            (runs_dir / "2026-02-21_120000").mkdir()

            latest = get_latest_run(runs_dir)

            assert latest is not None
            assert latest.name == "2026-02-22_150000"

    def test_returns_none_for_empty_directory(self) -> None:
        """Returns None when no runs exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            latest = get_latest_run(runs_dir)
            assert latest is None

    def test_returns_none_for_nonexistent_directory(self) -> None:
        """Returns None for nonexistent directory."""
        latest = get_latest_run(Path("/nonexistent/path"))
        assert latest is None


class TestSetupDrivePersistence:
    """Tests for setup_drive_persistence() in local mode."""

    def test_noop_when_not_in_colab(self) -> None:
        """Returns local path when not in Colab."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "results" / "diagnostic_runs"

            result = setup_drive_persistence(
                local_results_dir=local_dir,
                keep_n_runs=5
            )

            # Returns local path (not Drive)
            assert result == local_dir
            # Creates local directory
            assert local_dir.exists()

    def test_creates_nested_directories(self) -> None:
        """Creates nested directory structure when not in Colab."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "deep" / "nested" / "path"

            result = setup_drive_persistence(local_results_dir=local_dir)

            assert result == local_dir
            assert local_dir.exists()
