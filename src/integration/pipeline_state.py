"""
Pipeline state management for incremental checkpointing and resume.

Enables:
- Saving state after each major pipeline stage
- Resuming from last successful checkpoint after crash
- Tracking per-stage and per-benchmark status
- Fallback handling for partial failures
- Timestamped run folders with full diagnostic data persistence

Reference: DVT Section 4.6 (resilience requirements).
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types and other non-JSON-serializable objects.

    :param obj (Any): Object to serialize

    :return serialized (Any): JSON-safe version
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):  # datetime-like
        return obj.isoformat()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


class PipelineStage(Enum):
    """
    Pipeline execution stages in order.

    Each stage corresponds to a major computation that can be checkpointed.
    """
    NOT_STARTED = auto()
    DATA_PREP = auto()           # Window creation, universe filtering
    VAE_TRAINED = auto()         # After VAE training (existing checkpoint)
    INFERENCE_DONE = auto()      # After inference + AU measurement
    COVARIANCE_DONE = auto()     # After dual rescaling + risk model
    PORTFOLIO_DONE = auto()      # After portfolio optimization
    BENCHMARKS_DONE = auto()     # After all benchmarks complete
    METRICS_DONE = auto()        # After OOS metrics computed
    DIAGNOSTICS_DONE = auto()    # After diagnostics collected
    COMPLETE = auto()            # All done


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"  # Completed with degraded behavior


@dataclass
class StageInfo:
    """Status info for a single pipeline stage."""
    status: StageStatus = StageStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    duration_sec: float | None = None
    error_message: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "error_message": self.error_message,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageInfo":
        """Deserialize from dict."""
        return cls(
            status=StageStatus(d.get("status", "pending")),
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
            duration_sec=d.get("duration_sec"),
            error_message=d.get("error_message"),
            fallback_reason=d.get("fallback_reason"),
        )


@dataclass
class PipelineState:
    """
    Complete pipeline state for a single fold.

    Tracks:
    - Current stage and overall status
    - Per-stage status and timing
    - Per-benchmark status
    - Paths to saved arrays (NPY files)
    - Error log
    """
    fold_id: int = 0
    current_stage: PipelineStage = PipelineStage.NOT_STARTED
    stages: dict[str, StageInfo] = field(default_factory=dict)
    benchmark_statuses: dict[str, StageInfo] = field(default_factory=dict)

    # Paths to saved arrays (relative to checkpoint_dir)
    array_paths: dict[str, str] = field(default_factory=dict)

    # Model checkpoint path
    model_checkpoint_path: str | None = None

    # Error log for debugging
    error_log: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self) -> None:
        """Initialize stage dict if empty."""
        if not self.stages:
            for stage in PipelineStage:
                if stage != PipelineStage.NOT_STARTED:
                    self.stages[stage.name] = StageInfo()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "fold_id": self.fold_id,
            "current_stage": self.current_stage.name,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "benchmark_statuses": {k: v.to_dict() for k, v in self.benchmark_statuses.items()},
            "array_paths": self.array_paths,
            "model_checkpoint_path": self.model_checkpoint_path,
            "error_log": self.error_log,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineState":
        """Deserialize from dict."""
        state = cls(
            fold_id=d.get("fold_id", 0),
            current_stage=PipelineStage[d.get("current_stage", "NOT_STARTED")],
            array_paths=d.get("array_paths", {}),
            model_checkpoint_path=d.get("model_checkpoint_path"),
            error_log=d.get("error_log", []),
            created_at=d.get("created_at", ""),
            last_updated=d.get("last_updated", ""),
        )
        # Restore stages
        for stage_name, stage_dict in d.get("stages", {}).items():
            state.stages[stage_name] = StageInfo.from_dict(stage_dict)
        # Restore benchmark statuses
        for bench_name, bench_dict in d.get("benchmark_statuses", {}).items():
            state.benchmark_statuses[bench_name] = StageInfo.from_dict(bench_dict)
        return state

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for reporting."""
        stages_success = sum(
            1 for s in self.stages.values() if s.status == StageStatus.SUCCESS
        )
        stages_failed = sum(
            1 for s in self.stages.values() if s.status == StageStatus.FAILED
        )
        stages_fallback = sum(
            1 for s in self.stages.values() if s.status == StageStatus.FALLBACK
        )

        bench_success = sum(
            1 for s in self.benchmark_statuses.values() if s.status == StageStatus.SUCCESS
        )
        bench_failed = sum(
            1 for s in self.benchmark_statuses.values() if s.status == StageStatus.FAILED
        )
        bench_fallback = sum(
            1 for s in self.benchmark_statuses.values() if s.status == StageStatus.FALLBACK
        )

        overall_status = "complete"
        if stages_failed > 0 or self.current_stage != PipelineStage.COMPLETE:
            overall_status = "partial" if stages_success > 0 else "failed"

        return {
            "fold_id": self.fold_id,
            "overall_status": overall_status,
            "current_stage": self.current_stage.name,
            "stages_completed": stages_success,
            "stages_fallback": stages_fallback,
            "stages_failed": stages_failed,
            "benchmarks_success": bench_success,
            "benchmarks_fallback": bench_fallback,
            "benchmarks_failed": bench_failed,
            "n_errors": len(self.error_log),
        }


class PipelineStateManager:
    """
    Manages pipeline state persistence and resume logic.

    Handles:
    - Saving/loading state to JSON
    - Saving/loading numpy arrays to NPY files
    - Determining resume point after crash
    - Stage transition tracking
    """

    def __init__(self, checkpoint_dir: str, fold_id: int = 0) -> None:
        """
        Initialize state manager.

        :param checkpoint_dir (str): Directory for checkpoints
        :param fold_id (int): Fold identifier
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fold_id = fold_id
        self.arrays_dir = self.checkpoint_dir / f"fold_{fold_id:02d}_arrays"
        self.state_path = self.checkpoint_dir / f"fold_{fold_id:02d}_state.json"

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state or create new
        if self.state_path.exists():
            self.state = self.load_state()
        else:
            self.state = PipelineState(fold_id=fold_id)

    def save_state(self) -> str:
        """
        Save current state to JSON.

        :return path (str): Path to saved state file
        """
        self.state.last_updated = datetime.now().isoformat()
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug("State saved: %s", self.state_path)
        return str(self.state_path)

    def load_state(self) -> PipelineState:
        """
        Load state from JSON.

        :return state (PipelineState): Loaded state
        """
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}")

        with open(self.state_path, "r") as f:
            data = json.load(f)

        state = PipelineState.from_dict(data)
        logger.info(
            "State loaded: fold=%d, stage=%s, summary=%s",
            state.fold_id, state.current_stage.name, state.get_summary(),
        )
        return state

    def save_array(self, name: str, arr: np.ndarray) -> str:
        """
        Save numpy array to NPY file.

        :param name (str): Array name (e.g., "B_A", "Sigma_assets")
        :param arr (np.ndarray): Array to save

        :return path (str): Path to saved file
        """
        path = self.arrays_dir / f"{name}.npy"
        np.save(path, arr)
        self.state.array_paths[name] = str(path.relative_to(self.checkpoint_dir))
        self.save_state()
        logger.debug("Array saved: %s (%s)", name, arr.shape)
        return str(path)

    def load_array(self, name: str) -> np.ndarray | None:
        """
        Load numpy array from NPY file.

        :param name (str): Array name

        :return arr (np.ndarray | None): Loaded array or None if not found
        """
        rel_path = self.state.array_paths.get(name)
        if rel_path is None:
            return None

        path = self.checkpoint_dir / rel_path
        if not path.exists():
            logger.warning("Array file not found: %s", path)
            return None

        arr = np.load(path)
        logger.debug("Array loaded: %s (%s)", name, arr.shape)
        return arr

    def mark_stage_start(self, stage: PipelineStage) -> None:
        """
        Mark a stage as in progress.

        :param stage (PipelineStage): Stage being started
        """
        info = self.state.stages.get(stage.name, StageInfo())
        info.status = StageStatus.IN_PROGRESS
        info.start_time = time.time()
        info.end_time = None
        info.duration_sec = None
        info.error_message = None
        self.state.stages[stage.name] = info
        self.state.current_stage = stage
        self.save_state()
        logger.info("Stage started: %s", stage.name)

    def mark_stage_success(self, stage: PipelineStage) -> None:
        """
        Mark a stage as successfully completed.

        :param stage (PipelineStage): Stage completed
        """
        info = self.state.stages.get(stage.name, StageInfo())
        info.status = StageStatus.SUCCESS
        info.end_time = time.time()
        if info.start_time is not None:
            info.duration_sec = info.end_time - info.start_time
        self.state.stages[stage.name] = info
        self.save_state()
        logger.info("Stage success: %s (%.1fs)", stage.name, info.duration_sec or 0)

    def mark_stage_failed(
        self,
        stage: PipelineStage,
        error_message: str,
        exc: Exception | None = None,
    ) -> None:
        """
        Mark a stage as failed.

        :param stage (PipelineStage): Stage that failed
        :param error_message (str): Error description
        :param exc (Exception | None): Exception if available
        """
        info = self.state.stages.get(stage.name, StageInfo())
        info.status = StageStatus.FAILED
        info.end_time = time.time()
        if info.start_time is not None:
            info.duration_sec = info.end_time - info.start_time
        info.error_message = error_message
        self.state.stages[stage.name] = info

        # Log error
        self.state.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage.name,
            "error": error_message,
            "exception_type": type(exc).__name__ if exc else None,
        })

        self.save_state()
        logger.error("Stage failed: %s — %s", stage.name, error_message)

    def mark_stage_fallback(
        self,
        stage: PipelineStage,
        reason: str,
    ) -> None:
        """
        Mark a stage as completed with fallback behavior.

        :param stage (PipelineStage): Stage with fallback
        :param reason (str): Why fallback was used
        """
        info = self.state.stages.get(stage.name, StageInfo())
        info.status = StageStatus.FALLBACK
        info.end_time = time.time()
        if info.start_time is not None:
            info.duration_sec = info.end_time - info.start_time
        info.fallback_reason = reason
        self.state.stages[stage.name] = info
        self.save_state()
        logger.warning("Stage fallback: %s — %s", stage.name, reason)

    def mark_benchmark_start(self, benchmark_name: str) -> None:
        """Mark a benchmark as in progress."""
        info = StageInfo(
            status=StageStatus.IN_PROGRESS,
            start_time=time.time(),
        )
        self.state.benchmark_statuses[benchmark_name] = info
        self.save_state()

    def mark_benchmark_success(self, benchmark_name: str) -> None:
        """Mark a benchmark as successful."""
        info = self.state.benchmark_statuses.get(benchmark_name, StageInfo())
        info.status = StageStatus.SUCCESS
        info.end_time = time.time()
        if info.start_time is not None:
            info.duration_sec = info.end_time - info.start_time
        self.state.benchmark_statuses[benchmark_name] = info
        self.save_state()
        logger.info("Benchmark success: %s", benchmark_name)

    def mark_benchmark_failed(
        self,
        benchmark_name: str,
        error_message: str,
    ) -> None:
        """Mark a benchmark as failed."""
        info = self.state.benchmark_statuses.get(benchmark_name, StageInfo())
        info.status = StageStatus.FAILED
        info.end_time = time.time()
        info.error_message = error_message
        self.state.benchmark_statuses[benchmark_name] = info

        self.state.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "benchmark": benchmark_name,
            "error": error_message,
        })

        self.save_state()
        logger.error("Benchmark failed: %s — %s", benchmark_name, error_message)

    def mark_benchmark_fallback(
        self,
        benchmark_name: str,
        reason: str,
    ) -> None:
        """Mark a benchmark as using fallback (equal-weight)."""
        info = self.state.benchmark_statuses.get(benchmark_name, StageInfo())
        info.status = StageStatus.FALLBACK
        info.end_time = time.time()
        info.fallback_reason = reason
        self.state.benchmark_statuses[benchmark_name] = info
        self.save_state()
        logger.warning("Benchmark fallback: %s — %s", benchmark_name, reason)

    def get_resume_stage(self) -> PipelineStage:
        """
        Determine the stage to resume from after a crash.

        Returns the last successfully completed stage, so the pipeline
        can restart from the next one.

        :return stage (PipelineStage): Stage to resume from
        """
        # Find the last successful stage
        last_success = PipelineStage.NOT_STARTED
        for stage in PipelineStage:
            if stage == PipelineStage.NOT_STARTED:
                continue
            info = self.state.stages.get(stage.name)
            if info is not None and info.status in (StageStatus.SUCCESS, StageStatus.FALLBACK):
                last_success = stage
            else:
                break  # Stop at first non-success

        return last_success

    def can_resume(self) -> bool:
        """
        Check if there is a valid checkpoint to resume from.

        :return can_resume (bool): True if resume is possible
        """
        return self.get_resume_stage() != PipelineStage.NOT_STARTED

    def set_model_checkpoint(self, path: str) -> None:
        """
        Record the model checkpoint path.

        :param path (str): Path to model checkpoint file
        """
        self.state.model_checkpoint_path = path
        self.save_state()

    def mark_complete(self) -> None:
        """Mark the entire pipeline as complete."""
        self.state.current_stage = PipelineStage.COMPLETE
        self.save_state()
        logger.info(
            "Pipeline complete for fold %d: %s",
            self.fold_id, self.state.get_summary(),
        )

    def export_status_json(self, output_path: str | None = None) -> dict[str, Any]:
        """
        Export detailed status as JSON for reporting.

        :param output_path (str | None): Optional path to save JSON

        :return status (dict): Complete status dict
        """
        status = {
            "fold_id": self.fold_id,
            "overall_status": self.state.get_summary()["overall_status"],
            "stages": {
                name: info.to_dict()
                for name, info in self.state.stages.items()
            },
            "benchmarks": {
                name: info.to_dict()
                for name, info in self.state.benchmark_statuses.items()
            },
            "summary": self.state.get_summary(),
            "error_log": self.state.error_log,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(status, f, indent=2)
            logger.info("Status exported: %s", output_path)

        return status

    # -------------------------------------------------------------------
    # Extended state bag persistence
    # -------------------------------------------------------------------

    def save_json(self, name: str, data: dict | list) -> str:
        """
        Save JSON-serializable data to json/{name}.json.

        :param name (str): File name (without .json extension)
        :param data (dict | list): Data to save

        :return path (str): Path to saved file
        """
        json_dir = self.checkpoint_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        path = json_dir / f"{name}.json"

        serialized = _serialize_for_json(data)
        with open(path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)

        logger.debug("JSON saved: %s", path)
        return str(path)

    def load_json(self, name: str) -> dict | list | None:
        """
        Load JSON data from json/{name}.json.

        :param name (str): File name (without .json extension)

        :return data (dict | list | None): Loaded data or None if not found
        """
        path = self.checkpoint_dir / "json" / f"{name}.json"
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        logger.debug("JSON loaded: %s", path)
        return data

    def save_stage_arrays(
        self,
        stage: PipelineStage,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, str]:
        """
        Batch save arrays to arrays/{stage_name}/.

        :param stage (PipelineStage): Stage these arrays belong to
        :param arrays (dict): name -> array mapping

        :return paths (dict): name -> saved path mapping
        """
        stage_dir = self.checkpoint_dir / "arrays" / stage.name.lower()
        stage_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}
        for name, arr in arrays.items():
            path = stage_dir / f"{name}.npy"
            np.save(path, arr)
            rel_path = str(path.relative_to(self.checkpoint_dir))
            self.state.array_paths[name] = rel_path
            paths[name] = str(path)
            logger.debug("Stage array saved: %s/%s (%s)", stage.name, name, arr.shape)

        self.save_state()
        return paths

    def load_stage_arrays(self, stage: PipelineStage) -> dict[str, np.ndarray]:
        """
        Load all arrays from arrays/{stage_name}/.

        :param stage (PipelineStage): Stage to load arrays for

        :return arrays (dict): name -> array mapping
        """
        stage_dir = self.checkpoint_dir / "arrays" / stage.name.lower()
        if not stage_dir.exists():
            return {}

        arrays: dict[str, np.ndarray] = {}
        for path in stage_dir.glob("*.npy"):
            name = path.stem
            arrays[name] = np.load(path)
            logger.debug("Stage array loaded: %s/%s (%s)", stage.name, name, arrays[name].shape)

        return arrays

    def save_scalars(self, scalars: dict[str, float | int | str]) -> None:
        """
        Merge scalar values into json/scalars.json.

        :param scalars (dict): Scalar values to merge
        """
        existing = self.load_json("scalars")
        if existing is None:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}

        existing.update(scalars)
        self.save_json("scalars", existing)

    def load_scalars(self) -> dict[str, Any]:
        """
        Load json/scalars.json.

        :return scalars (dict): Loaded scalars or empty dict
        """
        data = self.load_json("scalars")
        if data is None or not isinstance(data, dict):
            return {}
        return data

    def save_state_bag_for_stage(
        self,
        stage: PipelineStage,
        state_bag: dict[str, Any],
    ) -> None:
        """
        Save all relevant state_bag data for a given stage.

        Routes data to appropriate storage (arrays for numpy, JSON for dicts).

        :param stage (PipelineStage): Stage being checkpointed
        :param state_bag (dict): State bag with mixed data types
        """
        # Stage-specific keys to save
        stage_arrays: dict[str, np.ndarray] = {}
        stage_json: dict[str, Any] = {}
        stage_scalars: dict[str, float | int | str] = {}

        if stage == PipelineStage.VAE_TRAINED:
            # Training results
            fit_result = state_bag.get("fit_result")
            if fit_result is not None:
                # Filter out non-serializable model reference
                fr_clean = {k: v for k, v in fit_result.items() if k != "model"}
                stage_json["fit_result"] = fr_clean

            # Build params and VAE info
            if "build_params" in state_bag:
                stage_json["build_params"] = state_bag["build_params"]
            if "vae_info" in state_bag:
                stage_json["vae_info"] = state_bag["vae_info"]

        elif stage == PipelineStage.INFERENCE_DONE:
            # Exposure matrix and KL per dim
            if "B_A" in state_bag:
                stage_arrays["B_A"] = state_bag["B_A"]
            if "kl_per_dim" in state_bag:
                stage_arrays["kl_per_dim"] = state_bag["kl_per_dim"]
            if "active_dims" in state_bag:
                stage_json["active_dims"] = state_bag["active_dims"]
            if "inferred_stock_ids" in state_bag:
                stage_json["inferred_stock_ids"] = state_bag["inferred_stock_ids"]
            if "AU" in state_bag:
                stage_scalars["AU"] = state_bag["AU"]

        elif stage == PipelineStage.COVARIANCE_DONE:
            # Risk model components
            risk_model = state_bag.get("risk_model", {})
            if "Sigma_assets" in risk_model:
                stage_arrays["Sigma_assets"] = risk_model["Sigma_assets"]
            if "z_hat" in state_bag:
                stage_arrays["z_hat"] = state_bag["z_hat"]
            if "eigenvalues_signal" in state_bag:
                stage_arrays["eigenvalues_signal"] = state_bag["eigenvalues_signal"]
            if "B_prime_signal" in state_bag:
                stage_arrays["B_prime_signal"] = state_bag["B_prime_signal"]

            # Scalars
            for key in ["vt_scale_sys", "vt_scale_idio", "n_signal", "shrinkage_intensity"]:
                if key in state_bag:
                    stage_scalars[key] = state_bag[key]

            # B_A by date (dict of arrays) and valid_dates
            if "B_A_by_date" in state_bag:
                # Save each date's B_A separately
                b_a_by_date = state_bag["B_A_by_date"]
                stage_json["B_A_by_date_keys"] = list(b_a_by_date.keys())
                for date_str, arr in b_a_by_date.items():
                    safe_name = date_str.replace("-", "_")
                    stage_arrays[f"B_A_date_{safe_name}"] = arr

            if "valid_dates" in state_bag:
                stage_json["valid_dates"] = state_bag["valid_dates"]

        elif stage == PipelineStage.PORTFOLIO_DONE:
            # Portfolio weights and optimization results
            if "w_vae" in state_bag:
                stage_arrays["w_vae"] = state_bag["w_vae"]

            if "frontier" in state_bag:
                stage_json["frontier"] = state_bag["frontier"]
            if "solver_stats" in state_bag:
                stage_json["solver_stats"] = state_bag["solver_stats"]
            if "binding_status" in state_bag:
                stage_json["binding_status"] = state_bag["binding_status"]

            if "alpha_opt" in state_bag:
                stage_scalars["alpha_opt"] = state_bag["alpha_opt"]

        # Save arrays
        if stage_arrays:
            self.save_stage_arrays(stage, stage_arrays)

        # Save JSON data
        for name, data in stage_json.items():
            self.save_json(name, data)

        # Save scalars
        if stage_scalars:
            self.save_scalars(stage_scalars)

        logger.info(
            "State bag saved for %s: %d arrays, %d JSON, %d scalars",
            stage.name, len(stage_arrays), len(stage_json), len(stage_scalars),
        )

    def load_state_bag_for_stage(self, stage: PipelineStage) -> dict[str, Any]:
        """
        Reconstruct state_bag data from saved checkpoints for a given stage.

        :param stage (PipelineStage): Stage to load data for

        :return state_bag (dict): Reconstructed state bag entries
        """
        state_bag: dict[str, Any] = {}

        # Load stage arrays
        arrays = self.load_stage_arrays(stage)
        state_bag.update(arrays)

        # Load scalars
        scalars = self.load_scalars()
        state_bag.update(scalars)

        # Stage-specific JSON loading
        if stage == PipelineStage.VAE_TRAINED:
            fit_result = self.load_json("fit_result")
            if fit_result:
                state_bag["fit_result"] = fit_result
            build_params = self.load_json("build_params")
            if build_params:
                state_bag["build_params"] = build_params
            vae_info = self.load_json("vae_info")
            if vae_info:
                state_bag["vae_info"] = vae_info

        elif stage == PipelineStage.INFERENCE_DONE:
            active_dims = self.load_json("active_dims")
            if active_dims:
                state_bag["active_dims"] = active_dims
            stock_ids = self.load_json("inferred_stock_ids")
            if stock_ids:
                state_bag["inferred_stock_ids"] = stock_ids

        elif stage == PipelineStage.COVARIANCE_DONE:
            valid_dates = self.load_json("valid_dates")
            if valid_dates:
                state_bag["valid_dates"] = valid_dates

            # Reconstruct B_A_by_date
            b_a_keys = self.load_json("B_A_by_date_keys")
            if b_a_keys:
                b_a_by_date: dict[str, np.ndarray] = {}
                for date_str in b_a_keys:
                    safe_name = date_str.replace("-", "_")
                    arr_key = f"B_A_date_{safe_name}"
                    if arr_key in arrays:
                        b_a_by_date[date_str] = arrays[arr_key]
                        del state_bag[arr_key]  # Remove individual entries
                if b_a_by_date:
                    state_bag["B_A_by_date"] = b_a_by_date

            # Wrap risk model components
            risk_model: dict[str, Any] = {}
            if "Sigma_assets" in state_bag:
                risk_model["Sigma_assets"] = state_bag.pop("Sigma_assets")
            if risk_model:
                state_bag["risk_model"] = risk_model

        elif stage == PipelineStage.PORTFOLIO_DONE:
            frontier = self.load_json("frontier")
            if frontier:
                state_bag["frontier"] = frontier
            solver_stats = self.load_json("solver_stats")
            if solver_stats:
                state_bag["solver_stats"] = solver_stats
            binding_status = self.load_json("binding_status")
            if binding_status:
                state_bag["binding_status"] = binding_status

        logger.info(
            "State bag loaded for %s: %d keys",
            stage.name, len(state_bag),
        )
        return state_bag


# ---------------------------------------------------------------------------
# DiagnosticRunManager — Timestamped run folder management
# ---------------------------------------------------------------------------

def _get_git_hash() -> str | None:
    """
    Get current git commit hash (short form).

    :return hash (str | None): Git hash or None if not in a repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class DiagnosticRunManager:
    """
    Manages timestamped diagnostic run folders.

    Creates folder structure:
        results/diagnostic_runs/
        └── 2026-02-21_143052/
            ├── run_config.json       # Config + CLI args + git hash
            ├── state.json            # Pipeline state
            ├── arrays/               # NPY files by stage
            ├── json/                 # JSON-serializable data
            ├── checkpoints/          # VAE model
            ├── plots/                # PNG files
            └── diagnostic_report.md

    Supports resuming from existing runs.
    """

    def __init__(
        self,
        base_dir: str = "results/diagnostic_runs",
        run_dir: str | None = None,
    ) -> None:
        """
        Initialize run manager.

        :param base_dir (str): Base directory for runs (default: results/diagnostic_runs)
        :param run_dir (str | None): Existing run folder for resume. If None, creates new.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_dir is not None:
            # Resume from existing
            self.run_dir = Path(run_dir)
            if not self.run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
            logger.info("Using existing run directory: %s", self.run_dir)
        else:
            # Create new timestamped folder
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.run_dir = self.base_dir / timestamp
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created new run directory: %s", self.run_dir)

        # Create subdirectories
        (self.run_dir / "arrays").mkdir(exist_ok=True)
        (self.run_dir / "json").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

    @property
    def run_dir_str(self) -> str:
        """Get run directory as string."""
        return str(self.run_dir)

    def get_vae_checkpoint_path(self, override: str | None = None) -> str | None:
        """
        Get VAE checkpoint path with priority: override > run_dir/checkpoints/ > None.

        :param override (str | None): User-specified checkpoint path

        :return path (str | None): Checkpoint path or None if not found
        """
        if override is not None:
            if Path(override).exists():
                return override
            logger.warning("Override checkpoint not found: %s", override)

        # Look in run's checkpoints folder
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        if ckpt_files:
            # Return the most recently modified
            latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            logger.info("Found VAE checkpoint: %s", latest)
            return str(latest)

        return None

    def save_run_config(
        self,
        config: dict[str, Any],
        cli_args: dict[str, Any] | None = None,
    ) -> str:
        """
        Save run configuration to run_config.json.

        :param config (dict): Pipeline configuration
        :param cli_args (dict | None): CLI arguments

        :return path (str): Path to saved config
        """
        run_config = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": _get_git_hash(),
            "config": _serialize_for_json(config),
            "cli_args": cli_args or {},
        }

        path = self.run_dir / "run_config.json"
        with open(path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)

        logger.info("Run config saved: %s", path)
        return str(path)

    def load_run_config(self) -> dict[str, Any] | None:
        """
        Load run configuration from run_config.json.

        :return config (dict | None): Run config or None if not found
        """
        path = self.run_dir / "run_config.json"
        if not path.exists():
            return None

        with open(path, "r") as f:
            return json.load(f)

    def create_state_manager(self, fold_id: int = 0) -> PipelineStateManager:
        """
        Create a PipelineStateManager using this run's directory.

        :param fold_id (int): Fold identifier

        :return manager (PipelineStateManager): State manager
        """
        return PipelineStateManager(str(self.run_dir), fold_id=fold_id)

    def get_checkpoint_dir(self) -> str:
        """Get path to checkpoints subdirectory."""
        return str(self.run_dir / "checkpoints")

    def get_plots_dir(self) -> str:
        """Get path to plots subdirectory."""
        return str(self.run_dir / "plots")

    def get_output_dir(self) -> str:
        """Get run directory as output_dir (for report saving)."""
        return str(self.run_dir)


def load_run_data(run_dir: str) -> dict[str, Any]:
    """
    Load all data from a diagnostic run folder.

    Returns a dict compatible with notebook sections 9-10.

    :param run_dir (str): Path to run folder

    :return data (dict): Loaded run data
    """
    run_path = Path(run_dir)
    data: dict[str, Any] = {}
    data["run_dir"] = run_dir

    # diagnostic_data.json (main diagnostics)
    diag_path = run_path / "diagnostic_data.json"
    if diag_path.exists():
        with open(diag_path, "r") as f:
            data["diagnostics"] = json.load(f)

    # w_vae.npy (portfolio weights)
    weights_path = run_path / "arrays" / "portfolio_done" / "w_vae.npy"
    if weights_path.exists():
        data["weights"] = np.load(weights_path)
    else:
        # Fallback to older path structure
        alt_weights = run_path / "fold_00_arrays" / "w_vae.npy"
        if alt_weights.exists():
            data["weights"] = np.load(alt_weights)

    # inferred_stock_ids.json
    ids_path = run_path / "json" / "inferred_stock_ids.json"
    if ids_path.exists():
        with open(ids_path, "r") as f:
            data["stock_ids"] = json.load(f)

    # B_A.npy (exposures for visualization)
    ba_path = run_path / "arrays" / "inference_done" / "B_A.npy"
    if ba_path.exists():
        data["B_A"] = np.load(ba_path)
    else:
        # Fallback
        alt_ba = run_path / "fold_00_arrays" / "B_A.npy"
        if alt_ba.exists():
            data["B_A"] = np.load(alt_ba)

    # run_config.json
    config_path = run_path / "run_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            data["run_config"] = json.load(f)

    # state.json (pipeline state)
    state_path = run_path / "fold_00_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            data["pipeline_state"] = json.load(f)

    logger.info(
        "Loaded run data from %s: %s",
        run_dir, list(data.keys()),
    )
    return data
