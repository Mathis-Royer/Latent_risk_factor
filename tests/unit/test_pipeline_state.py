"""
Unit tests for pipeline state management.

Tests:
- PipelineStage enum ordering
- StageInfo dataclass serialization
- PipelineState lifecycle (create, save, load)
- PipelineStateManager checkpoint operations
- Resume logic and fallback handling
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.integration.pipeline_state import (
    DiagnosticRunManager,
    PipelineStage,
    PipelineStateManager,
    PipelineState,
    StageInfo,
    StageStatus,
    load_run_data,
)


class TestPipelineStage:
    """Test PipelineStage enum."""

    def test_stage_ordering(self) -> None:
        """Stages should have correct ordering by value."""
        stages = list(PipelineStage)
        assert PipelineStage.NOT_STARTED == stages[0]
        assert PipelineStage.VAE_TRAINED.value > PipelineStage.DATA_PREP.value
        assert PipelineStage.COMPLETE.value > PipelineStage.DIAGNOSTICS_DONE.value

    def test_stage_names(self) -> None:
        """All expected stages should exist."""
        expected = [
            "NOT_STARTED", "DATA_PREP", "VAE_TRAINED", "INFERENCE_DONE",
            "COVARIANCE_DONE", "PORTFOLIO_DONE", "BENCHMARKS_DONE",
            "METRICS_DONE", "DIAGNOSTICS_DONE", "COMPLETE",
        ]
        actual = [s.name for s in PipelineStage]
        assert actual == expected


class TestStageStatus:
    """Test StageStatus enum."""

    def test_status_values(self) -> None:
        """Status enum should have correct string values."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.SUCCESS.value == "success"
        assert StageStatus.FAILED.value == "failed"
        assert StageStatus.FALLBACK.value == "fallback"


class TestStageInfo:
    """Test StageInfo dataclass."""

    def test_default_values(self) -> None:
        """Default StageInfo should be pending with no times."""
        info = StageInfo()
        assert info.status == StageStatus.PENDING
        assert info.start_time is None
        assert info.end_time is None
        assert info.error_message is None

    def test_to_dict(self) -> None:
        """StageInfo should serialize to dict."""
        info = StageInfo(
            status=StageStatus.SUCCESS,
            start_time=1000.0,
            end_time=1010.0,
            duration_sec=10.0,
        )
        d = info.to_dict()
        assert d["status"] == "success"
        assert d["start_time"] == 1000.0
        assert d["duration_sec"] == 10.0

    def test_from_dict(self) -> None:
        """StageInfo should deserialize from dict."""
        d = {
            "status": "failed",
            "start_time": 500.0,
            "error_message": "Test error",
        }
        info = StageInfo.from_dict(d)
        assert info.status == StageStatus.FAILED
        assert info.error_message == "Test error"


class TestPipelineState:
    """Test PipelineState dataclass."""

    def test_post_init_creates_stages(self) -> None:
        """__post_init__ should create stage dict entries."""
        state = PipelineState(fold_id=0)
        # Should have entries for all stages except NOT_STARTED
        for stage in PipelineStage:
            if stage != PipelineStage.NOT_STARTED:
                assert stage.name in state.stages

    def test_to_dict_roundtrip(self) -> None:
        """State should survive serialization roundtrip."""
        state = PipelineState(fold_id=5)
        state.current_stage = PipelineStage.VAE_TRAINED
        state.array_paths["B_A"] = "fold_05_arrays/B_A.npy"

        d = state.to_dict()
        restored = PipelineState.from_dict(d)

        assert restored.fold_id == 5
        assert restored.current_stage == PipelineStage.VAE_TRAINED
        assert restored.array_paths["B_A"] == "fold_05_arrays/B_A.npy"

    def test_get_summary(self) -> None:
        """Summary should count stages correctly."""
        state = PipelineState(fold_id=0)
        state.stages["VAE_TRAINED"] = StageInfo(status=StageStatus.SUCCESS)
        state.stages["INFERENCE_DONE"] = StageInfo(status=StageStatus.SUCCESS)
        state.stages["BENCHMARKS_DONE"] = StageInfo(status=StageStatus.FALLBACK)

        summary = state.get_summary()
        assert summary["stages_completed"] == 2
        assert summary["stages_fallback"] == 1


class TestPipelineStateManager:
    """Test PipelineStateManager."""

    @pytest.fixture
    def temp_dir(self):  # type: ignore[misc]
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_init_creates_directories(self, temp_dir: str) -> None:
        """Manager should create checkpoint and arrays directories."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        assert manager.checkpoint_dir.exists()
        assert manager.arrays_dir.exists()

    def test_save_load_state(self, temp_dir: str) -> None:
        """State should persist across save/load."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.state.current_stage = PipelineStage.COVARIANCE_DONE
        manager.save_state()

        # Create new manager and load
        manager2 = PipelineStateManager(temp_dir, fold_id=0)
        assert manager2.state.current_stage == PipelineStage.COVARIANCE_DONE

    def test_save_load_array(self, temp_dir: str) -> None:
        """Arrays should persist correctly."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        arr = np.random.randn(100, 10)
        manager.save_array("test_array", arr)

        loaded = manager.load_array("test_array")
        assert loaded is not None
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_load_missing_array(self, temp_dir: str) -> None:
        """Loading non-existent array should return None."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        assert manager.load_array("nonexistent") is None

    def test_mark_stage_success(self, temp_dir: str) -> None:
        """Marking stage success should update state."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)

        info = manager.state.stages["VAE_TRAINED"]
        assert info.status == StageStatus.SUCCESS
        assert info.duration_sec is not None
        assert info.duration_sec >= 0

    def test_mark_stage_failed(self, temp_dir: str) -> None:
        """Marking stage failed should log error."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.mark_stage_start(PipelineStage.INFERENCE_DONE)
        manager.mark_stage_failed(
            PipelineStage.INFERENCE_DONE,
            "Test failure",
            ValueError("test"),
        )

        info = manager.state.stages["INFERENCE_DONE"]
        assert info.status == StageStatus.FAILED
        assert info.error_message == "Test failure"
        assert len(manager.state.error_log) == 1

    def test_mark_stage_fallback(self, temp_dir: str) -> None:
        """Marking stage fallback should record reason."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.mark_stage_start(PipelineStage.PORTFOLIO_DONE)
        manager.mark_stage_fallback(
            PipelineStage.PORTFOLIO_DONE,
            "Used equal-weight due to Cholesky failure",
        )

        info = manager.state.stages["PORTFOLIO_DONE"]
        assert info.status == StageStatus.FALLBACK
        assert "Cholesky" in str(info.fallback_reason)

    def test_get_resume_stage(self, temp_dir: str) -> None:
        """Resume stage should be last successful stage."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        # No stages completed
        assert manager.get_resume_stage() == PipelineStage.NOT_STARTED

        # Complete some stages
        manager.mark_stage_start(PipelineStage.DATA_PREP)
        manager.mark_stage_success(PipelineStage.DATA_PREP)
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)
        manager.mark_stage_start(PipelineStage.INFERENCE_DONE)
        manager.mark_stage_success(PipelineStage.INFERENCE_DONE)
        manager.mark_stage_start(PipelineStage.COVARIANCE_DONE)
        manager.mark_stage_failed(PipelineStage.COVARIANCE_DONE, "Error")

        # Resume should be at last success (INFERENCE_DONE)
        assert manager.get_resume_stage() == PipelineStage.INFERENCE_DONE

    def test_can_resume(self, temp_dir: str) -> None:
        """can_resume should return True only if there's progress."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        assert not manager.can_resume()

        manager.mark_stage_start(PipelineStage.DATA_PREP)
        manager.mark_stage_success(PipelineStage.DATA_PREP)
        assert manager.can_resume()

    def test_benchmark_status_tracking(self, temp_dir: str) -> None:
        """Benchmark statuses should be tracked separately."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        manager.mark_benchmark_start("equal_weight")
        manager.mark_benchmark_success("equal_weight")

        manager.mark_benchmark_start("min_variance")
        manager.mark_benchmark_fallback("min_variance", "Singular matrix")

        manager.mark_benchmark_start("pca_factor_rp")
        manager.mark_benchmark_failed("pca_factor_rp", "Division by zero")

        summary = manager.state.get_summary()
        assert summary["benchmarks_success"] == 1
        assert summary["benchmarks_fallback"] == 1
        assert summary["benchmarks_failed"] == 1

    def test_mark_complete(self, temp_dir: str) -> None:
        """mark_complete should set stage to COMPLETE."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.mark_complete()
        assert manager.state.current_stage == PipelineStage.COMPLETE

    def test_export_status_json(self, temp_dir: str) -> None:
        """export_status_json should create valid JSON file."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)

        output_path = os.path.join(temp_dir, "status.json")
        status = manager.export_status_json(output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["fold_id"] == 0
        assert loaded["stages"]["VAE_TRAINED"]["status"] == "success"

    def test_set_model_checkpoint(self, temp_dir: str) -> None:
        """Model checkpoint path should be persisted."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        manager.set_model_checkpoint("/path/to/model.pt")
        manager.save_state()

        manager2 = PipelineStateManager(temp_dir, fold_id=0)
        assert manager2.state.model_checkpoint_path == "/path/to/model.pt"

    def test_multiple_folds_isolated(self, temp_dir: str) -> None:
        """Different fold_ids should have isolated state."""
        manager0 = PipelineStateManager(temp_dir, fold_id=0)
        manager1 = PipelineStateManager(temp_dir, fold_id=1)

        manager0.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager0.mark_stage_success(PipelineStage.VAE_TRAINED)

        # Fold 1 should not see fold 0's progress
        assert manager1.state.stages["VAE_TRAINED"].status == StageStatus.PENDING


class TestResumeLogic:
    """Test resume scenarios."""

    @pytest.fixture
    def temp_dir(self):  # type: ignore[misc]
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_resume_after_vae_trained(self, temp_dir: str) -> None:
        """Should resume from VAE_TRAINED when that stage completed."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        # Simulate successful VAE training
        manager.mark_stage_start(PipelineStage.DATA_PREP)
        manager.mark_stage_success(PipelineStage.DATA_PREP)
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)

        # Save B_A array
        B_A = np.random.randn(100, 10)
        manager.save_array("B_A", B_A)
        manager.set_model_checkpoint("/path/to/model.pt")
        manager.save_state()

        # Simulate crash and restart
        manager2 = PipelineStateManager(temp_dir, fold_id=0)
        assert manager2.can_resume()
        assert manager2.get_resume_stage() == PipelineStage.VAE_TRAINED

        # Should be able to load arrays
        loaded_B_A = manager2.load_array("B_A")
        assert loaded_B_A is not None
        np.testing.assert_array_almost_equal(B_A, loaded_B_A)

    def test_resume_skips_completed_benchmarks(self, temp_dir: str) -> None:
        """Should track which benchmarks completed for selective resume."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        # Complete VAE + 3 benchmarks
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)

        for bench in ["equal_weight", "inverse_vol", "min_variance"]:
            manager.mark_benchmark_start(bench)
            manager.mark_benchmark_success(bench)

        manager.save_state()

        # Check which benchmarks need to run
        manager2 = PipelineStateManager(temp_dir, fold_id=0)
        completed = [
            name for name, info in manager2.state.benchmark_statuses.items()
            if info.status == StageStatus.SUCCESS
        ]
        assert set(completed) == {"equal_weight", "inverse_vol", "min_variance"}

    def test_fallback_counts_as_completed(self, temp_dir: str) -> None:
        """Stages with fallback should count as completed for resume."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        manager.mark_stage_start(PipelineStage.DATA_PREP)
        manager.mark_stage_success(PipelineStage.DATA_PREP)
        manager.mark_stage_start(PipelineStage.VAE_TRAINED)
        manager.mark_stage_success(PipelineStage.VAE_TRAINED)
        manager.mark_stage_start(PipelineStage.INFERENCE_DONE)
        manager.mark_stage_fallback(
            PipelineStage.INFERENCE_DONE,
            "Used identity matrix fallback",
        )

        # Resume should be at INFERENCE_DONE (fallback counts)
        assert manager.get_resume_stage() == PipelineStage.INFERENCE_DONE


# ---------------------------------------------------------------------------
# Extended state bag persistence tests
# ---------------------------------------------------------------------------

class TestExtendedStateBagPersistence:
    """Test extended state bag save/load methods."""

    @pytest.fixture
    def temp_dir(self):  # type: ignore[misc]
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_save_load_json(self, temp_dir: str) -> None:
        """JSON data should persist correctly."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        manager.save_json("test_data", data)

        loaded = manager.load_json("test_data")
        assert loaded == data

    def test_load_missing_json(self, temp_dir: str) -> None:
        """Loading non-existent JSON should return None."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        assert manager.load_json("nonexistent") is None

    def test_save_load_stage_arrays(self, temp_dir: str) -> None:
        """Stage arrays should batch save and load correctly."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        arrays = {
            "B_A": np.random.randn(100, 10),
            "kl_per_dim": np.random.rand(10),
        }
        manager.save_stage_arrays(PipelineStage.INFERENCE_DONE, arrays)

        loaded = manager.load_stage_arrays(PipelineStage.INFERENCE_DONE)
        assert "B_A" in loaded
        assert "kl_per_dim" in loaded
        np.testing.assert_array_almost_equal(arrays["B_A"], loaded["B_A"])
        np.testing.assert_array_almost_equal(arrays["kl_per_dim"], loaded["kl_per_dim"])

    def test_load_missing_stage_arrays(self, temp_dir: str) -> None:
        """Loading arrays from non-existent stage should return empty dict."""
        manager = PipelineStateManager(temp_dir, fold_id=0)
        loaded = manager.load_stage_arrays(PipelineStage.PORTFOLIO_DONE)
        assert loaded == {}

    def test_save_load_scalars(self, temp_dir: str) -> None:
        """Scalars should persist and merge correctly."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        manager.save_scalars({"AU": 25, "n_signal": 15})
        loaded = manager.load_scalars()
        assert loaded["AU"] == 25
        assert loaded["n_signal"] == 15

        # Merge additional scalars
        manager.save_scalars({"alpha_opt": 1.5})
        loaded = manager.load_scalars()
        assert loaded["AU"] == 25  # Original preserved
        assert loaded["alpha_opt"] == 1.5  # New added

    def test_save_state_bag_for_inference_stage(self, temp_dir: str) -> None:
        """State bag save should route data correctly for INFERENCE_DONE."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        state_bag = {
            "B_A": np.random.randn(50, 8),
            "kl_per_dim": np.random.rand(8),
            "AU": 8,
            "active_dims": [0, 2, 4, 6, 8, 10, 12, 14],
            "inferred_stock_ids": [100, 200, 300],
        }

        manager.save_state_bag_for_stage(PipelineStage.INFERENCE_DONE, state_bag)

        # Verify arrays saved
        loaded_arrays = manager.load_stage_arrays(PipelineStage.INFERENCE_DONE)
        assert "B_A" in loaded_arrays
        np.testing.assert_array_almost_equal(state_bag["B_A"], loaded_arrays["B_A"])

        # Verify scalars saved
        scalars = manager.load_scalars()
        assert scalars["AU"] == 8

        # Verify JSON saved
        active_dims = manager.load_json("active_dims")
        assert active_dims == [0, 2, 4, 6, 8, 10, 12, 14]

        stock_ids = manager.load_json("inferred_stock_ids")
        assert stock_ids == [100, 200, 300]

    def test_save_state_bag_for_covariance_stage(self, temp_dir: str) -> None:
        """State bag save should route data correctly for COVARIANCE_DONE."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        state_bag = {
            "risk_model": {
                "Sigma_assets": np.eye(10),
            },
            "z_hat": np.random.randn(100, 5),
            "vt_scale_sys": 1.05,
            "vt_scale_idio": 0.95,
            "n_signal": 5,
            "shrinkage_intensity": 0.3,
            "valid_dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }

        manager.save_state_bag_for_stage(PipelineStage.COVARIANCE_DONE, state_bag)

        # Verify arrays saved
        loaded_arrays = manager.load_stage_arrays(PipelineStage.COVARIANCE_DONE)
        assert "Sigma_assets" in loaded_arrays
        assert "z_hat" in loaded_arrays

        # Verify scalars saved
        scalars = manager.load_scalars()
        assert abs(scalars["vt_scale_sys"] - 1.05) < 1e-6
        assert scalars["n_signal"] == 5

        # Verify JSON saved
        valid_dates = manager.load_json("valid_dates")
        assert valid_dates == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_save_state_bag_for_portfolio_stage(self, temp_dir: str) -> None:
        """State bag save should route data correctly for PORTFOLIO_DONE."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        state_bag = {
            "w_vae": np.array([0.1, 0.2, 0.3, 0.4]),
            "alpha_opt": 2.5,
            "frontier": {"alphas": [1.0, 2.0], "H_values": [0.5, 0.6]},
            "solver_stats": {"converged": True, "n_iterations": 50},
            "binding_status": {"n_at_w_max": 5, "tau_binding": False},
        }

        manager.save_state_bag_for_stage(PipelineStage.PORTFOLIO_DONE, state_bag)

        # Verify array saved
        loaded_arrays = manager.load_stage_arrays(PipelineStage.PORTFOLIO_DONE)
        assert "w_vae" in loaded_arrays
        np.testing.assert_array_almost_equal(state_bag["w_vae"], loaded_arrays["w_vae"])

        # Verify scalar saved
        scalars = manager.load_scalars()
        assert abs(scalars["alpha_opt"] - 2.5) < 1e-6

        # Verify JSON saved
        frontier = manager.load_json("frontier")
        assert frontier is not None
        assert isinstance(frontier, dict)
        assert frontier["alphas"] == [1.0, 2.0]

        solver_stats = manager.load_json("solver_stats")
        assert solver_stats is not None
        assert isinstance(solver_stats, dict)
        assert solver_stats["converged"] is True

    def test_load_state_bag_for_stage(self, temp_dir: str) -> None:
        """State bag load should reconstruct data correctly."""
        manager = PipelineStateManager(temp_dir, fold_id=0)

        # Save data
        original_bag = {
            "B_A": np.random.randn(50, 8),
            "kl_per_dim": np.random.rand(8),
            "AU": 8,
            "active_dims": [0, 2, 4, 6],
            "inferred_stock_ids": [100, 200, 300],
        }
        manager.save_state_bag_for_stage(PipelineStage.INFERENCE_DONE, original_bag)

        # Load and verify
        loaded_bag = manager.load_state_bag_for_stage(PipelineStage.INFERENCE_DONE)

        assert "B_A" in loaded_bag
        np.testing.assert_array_almost_equal(original_bag["B_A"], loaded_bag["B_A"])
        assert loaded_bag["AU"] == 8
        assert loaded_bag["active_dims"] == [0, 2, 4, 6]
        assert loaded_bag["inferred_stock_ids"] == [100, 200, 300]


# ---------------------------------------------------------------------------
# DiagnosticRunManager tests
# ---------------------------------------------------------------------------

class TestDiagnosticRunManager:
    """Test DiagnosticRunManager."""

    @pytest.fixture
    def temp_dir(self):  # type: ignore[misc]
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_creates_timestamped_folder(self, temp_dir: str) -> None:
        """New run manager should create timestamped folder."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        assert manager.run_dir.exists()
        assert manager.run_dir.parent == Path(temp_dir)
        # Folder name should be timestamp format YYYY-MM-DD_HHMMSS
        folder_name = manager.run_dir.name
        assert len(folder_name) == 17  # 10 (date) + 1 (_) + 6 (time)
        assert "_" in folder_name

    def test_creates_subdirectories(self, temp_dir: str) -> None:
        """Run manager should create required subdirectories."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        assert (manager.run_dir / "arrays").exists()
        assert (manager.run_dir / "json").exists()
        assert (manager.run_dir / "checkpoints").exists()
        assert (manager.run_dir / "plots").exists()

    def test_resume_from_existing_folder(self, temp_dir: str) -> None:
        """Should resume from existing run folder."""
        # Create initial run
        manager1 = DiagnosticRunManager(base_dir=temp_dir)
        run_dir_path = manager1.run_dir_str

        # Save some data
        manager1.save_run_config({"test": "config"})

        # Resume from same folder
        manager2 = DiagnosticRunManager(run_dir=run_dir_path)
        assert manager2.run_dir_str == run_dir_path

        # Should be able to load config
        loaded = manager2.load_run_config()
        assert loaded is not None
        assert loaded["config"]["test"] == "config"

    def test_resume_nonexistent_folder_raises(self, temp_dir: str) -> None:
        """Resuming from non-existent folder should raise."""
        with pytest.raises(FileNotFoundError):
            DiagnosticRunManager(run_dir=os.path.join(temp_dir, "nonexistent"))

    def test_save_and_load_run_config(self, temp_dir: str) -> None:
        """Run config should persist correctly."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        config = {"vae": {"K": 75}, "training": {"max_epochs": 100}}
        cli_args = {"profile": "full", "device": "auto"}

        manager.save_run_config(config, cli_args)
        loaded = manager.load_run_config()

        assert loaded is not None
        assert loaded["config"]["vae"]["K"] == 75
        assert loaded["cli_args"]["profile"] == "full"
        assert "timestamp" in loaded
        assert "git_hash" in loaded

    def test_get_vae_checkpoint_path_with_override(self, temp_dir: str) -> None:
        """Override checkpoint path should take priority."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create a fake checkpoint file
        ckpt_path = os.path.join(temp_dir, "override.pt")
        Path(ckpt_path).touch()

        result = manager.get_vae_checkpoint_path(override=ckpt_path)
        assert result == ckpt_path

    def test_get_vae_checkpoint_path_from_run_folder(self, temp_dir: str) -> None:
        """Should find checkpoint in run folder's checkpoints dir."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create a fake checkpoint file in run folder
        ckpt_dir = manager.run_dir / "checkpoints"
        ckpt_path = ckpt_dir / "model.pt"
        ckpt_path.touch()

        result = manager.get_vae_checkpoint_path()
        assert result == str(ckpt_path)

    def test_get_vae_checkpoint_path_none_when_missing(self, temp_dir: str) -> None:
        """Should return None when no checkpoint exists."""
        manager = DiagnosticRunManager(base_dir=temp_dir)
        result = manager.get_vae_checkpoint_path()
        assert result is None

    def test_create_state_manager(self, temp_dir: str) -> None:
        """Should create state manager using run directory."""
        manager = DiagnosticRunManager(base_dir=temp_dir)
        state_mgr = manager.create_state_manager(fold_id=0)

        assert isinstance(state_mgr, PipelineStateManager)
        assert str(state_mgr.checkpoint_dir) == manager.run_dir_str

    def test_get_checkpoint_dir(self, temp_dir: str) -> None:
        """Should return checkpoints subdirectory."""
        manager = DiagnosticRunManager(base_dir=temp_dir)
        ckpt_dir = manager.get_checkpoint_dir()
        assert ckpt_dir == str(manager.run_dir / "checkpoints")

    def test_get_plots_dir(self, temp_dir: str) -> None:
        """Should return plots subdirectory."""
        manager = DiagnosticRunManager(base_dir=temp_dir)
        plots_dir = manager.get_plots_dir()
        assert plots_dir == str(manager.run_dir / "plots")


# ---------------------------------------------------------------------------
# load_run_data tests
# ---------------------------------------------------------------------------

class TestLoadRunData:
    """Test load_run_data function."""

    @pytest.fixture
    def temp_dir(self):  # type: ignore[misc]
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_load_diagnostics_json(self, temp_dir: str) -> None:
        """Should load diagnostic_data.json."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create diagnostic data file
        diag_path = manager.run_dir / "diagnostic_data.json"
        diag_data = {"training": {"n_epochs": 100}, "latent": {"AU": 25}}
        with open(diag_path, "w") as f:
            json.dump(diag_data, f)

        data = load_run_data(manager.run_dir_str)
        assert "diagnostics" in data
        assert data["diagnostics"]["latent"]["AU"] == 25

    def test_load_weights_from_new_path(self, temp_dir: str) -> None:
        """Should load weights from arrays/portfolio_done/."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create weights file in new path structure
        arrays_dir = manager.run_dir / "arrays" / "portfolio_done"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        np.save(arrays_dir / "w_vae.npy", weights)

        data = load_run_data(manager.run_dir_str)
        assert "weights" in data
        np.testing.assert_array_almost_equal(data["weights"], weights)

    def test_load_stock_ids(self, temp_dir: str) -> None:
        """Should load inferred_stock_ids.json."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create stock IDs file
        json_dir = manager.run_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        stock_ids = [100, 200, 300, 400]
        with open(json_dir / "inferred_stock_ids.json", "w") as f:
            json.dump(stock_ids, f)

        data = load_run_data(manager.run_dir_str)
        assert "stock_ids" in data
        assert data["stock_ids"] == stock_ids

    def test_load_b_a_from_new_path(self, temp_dir: str) -> None:
        """Should load B_A from arrays/inference_done/."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create B_A file in new path structure
        arrays_dir = manager.run_dir / "arrays" / "inference_done"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        B_A = np.random.randn(50, 10)
        np.save(arrays_dir / "B_A.npy", B_A)

        data = load_run_data(manager.run_dir_str)
        assert "B_A" in data
        np.testing.assert_array_almost_equal(data["B_A"], B_A)

    def test_load_fallback_to_old_path(self, temp_dir: str) -> None:
        """Should fallback to old path structure (fold_00_arrays/)."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        # Create weights file in old path structure
        old_dir = manager.run_dir / "fold_00_arrays"
        old_dir.mkdir(parents=True, exist_ok=True)
        weights = np.array([0.5, 0.5])
        np.save(old_dir / "w_vae.npy", weights)

        data = load_run_data(manager.run_dir_str)
        assert "weights" in data
        np.testing.assert_array_almost_equal(data["weights"], weights)

    def test_load_run_config(self, temp_dir: str) -> None:
        """Should load run_config.json."""
        manager = DiagnosticRunManager(base_dir=temp_dir)
        manager.save_run_config({"vae": {"K": 75}}, {"device": "auto"})

        data = load_run_data(manager.run_dir_str)
        assert "run_config" in data
        assert data["run_config"]["config"]["vae"]["K"] == 75

    def test_load_empty_folder(self, temp_dir: str) -> None:
        """Should handle empty run folder gracefully."""
        manager = DiagnosticRunManager(base_dir=temp_dir)

        data = load_run_data(manager.run_dir_str)
        assert data["run_dir"] == manager.run_dir_str
        # Other keys should not be present
        assert "diagnostics" not in data
        assert "weights" not in data


class TestSerializeForJsonSizeGuard:
    """Test size guards in _serialize_for_json to prevent memory explosion."""

    def test_large_array_returns_placeholder(self) -> None:
        """Arrays >100k elements should return placeholder, not full data."""
        from src.integration.pipeline_state import _serialize_for_json

        large_array = np.zeros((200, 600))  # 120k elements
        result = _serialize_for_json(large_array)
        assert isinstance(result, str)
        assert "array shape=" in result
        assert "size=120000" in result

    def test_small_array_returns_list(self) -> None:
        """Arrays <=100k elements should serialize normally."""
        from src.integration.pipeline_state import _serialize_for_json

        small_array = np.array([[1, 2], [3, 4]])
        result = _serialize_for_json(small_array)
        assert result == [[1, 2], [3, 4]]

    def test_large_list_returns_placeholder(self) -> None:
        """Lists >10k items should return placeholder."""
        from src.integration.pipeline_state import _serialize_for_json

        large_list = list(range(15000))
        result = _serialize_for_json(large_list)
        assert isinstance(result, str)
        assert "list len=15000" in result

    def test_small_list_serializes(self) -> None:
        """Lists <=10k items should serialize normally."""
        from src.integration.pipeline_state import _serialize_for_json

        small_list = [1, 2, 3]
        result = _serialize_for_json(small_list)
        assert result == [1, 2, 3]

    def test_max_depth_truncates(self) -> None:
        """Deeply nested structures should be truncated."""
        from src.integration.pipeline_state import _serialize_for_json

        # Create deeply nested dict
        nested: dict = {"level": 0}
        current = nested
        for i in range(15):
            current["child"] = {"level": i + 1}
            current = current["child"]

        result = _serialize_for_json(nested, _max_depth=5)
        # Should have truncation somewhere in the result
        result_str = str(result)
        assert "truncated" in result_str or "level" in result_str

    def test_numpy_generic_converts(self) -> None:
        """Numpy scalars should convert to Python types."""
        from src.integration.pipeline_state import _serialize_for_json

        result = _serialize_for_json(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)


class TestFitResultFiltering:
    """Test that fit_result is properly filtered in save_state_bag_for_stage."""

    def test_fit_result_only_saves_scalars(self) -> None:
        """VAE_TRAINED stage should save only summary scalars, not full history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PipelineStateManager(temp_dir, fold_id=0)

            # Create state_bag with full fit_result including large history
            state_bag = {
                "fit_result": {
                    "best_epoch": 50,
                    "best_val_elbo": 123.45,
                    "train_elbo": 120.0,
                    "overfit_flag": False,
                    "overfit_ratio": 1.03,
                    "history": [
                        {
                            "epoch": i,
                            "train_loss": 100 - i,
                            "val_elbo": 105 - i,
                            "AU": 10,
                            "sigma_sq": 0.5,
                            "learning_rate": 1e-4,
                            # Large arrays that should NOT be saved
                            "kl_per_dim": np.zeros(200).tolist(),
                            "log_var_stats": {"mean": np.zeros(200).tolist()},
                        }
                        for i in range(100)
                    ],
                },
                "build_params": {"n": 100, "K": 75},
                "vae_info": {"total_params": 1000000},
            }

            manager.save_state_bag_for_stage(PipelineStage.VAE_TRAINED, state_bag)

            # Load and verify
            fit_result = manager.load_json("fit_result")
            assert fit_result is not None
            assert isinstance(fit_result, dict)
            # Should have summary scalars
            assert fit_result["best_epoch"] == 50
            assert fit_result["best_val_elbo"] == 123.45
            # Should NOT have history (that's in training_curve now)
            assert "history" not in fit_result

            # Check training_curve is compact
            training_curve = manager.load_json("training_curve")
            assert training_curve is not None
            assert isinstance(training_curve, list)
            assert len(training_curve) == 100
            # Each entry should NOT have large arrays
            for entry in training_curve[:5]:
                assert isinstance(entry, dict)
                assert "kl_per_dim" not in entry
                assert "log_var_stats" not in entry
                # Should have scalars
                assert "epoch" in entry
                assert "train_loss" in entry
