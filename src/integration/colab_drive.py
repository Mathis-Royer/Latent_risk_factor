"""
Google Drive integration for Colab persistence.

This module provides automatic checkpointing to Google Drive when running
in Google Colab, enabling crash recovery and persistent storage across
runtime disconnections.

Architecture:
- Local `results/diagnostic_runs/` becomes a symlink to Drive
- All existing code writes transparently to Drive via the symlink
- No modifications needed to PipelineStateManager or other modules
"""

from pathlib import Path
import shutil
from datetime import datetime


def is_colab() -> bool:
    """
    Detect if running in Google Colab.

    :return is_colab (bool): True if running in Google Colab environment
    """
    try:
        # Import and check module exists (import itself is the test)
        import importlib.util
        return importlib.util.find_spec("google.colab") is not None
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """
    Mount Google Drive. No-op if already mounted.

    :param mount_point (str): Mount point for Google Drive

    :return mount_path (Path): Path to mounted Drive root
    """
    mount_path = Path(mount_point)

    if not is_colab():
        raise RuntimeError("mount_drive() can only be called in Google Colab")

    if mount_path.exists() and (mount_path / "MyDrive").exists():
        # Already mounted
        return mount_path

    from google.colab import drive  # type: ignore[import-not-found]
    drive.mount(mount_point)

    return mount_path


def setup_drive_persistence(
    local_results_dir: Path = Path("results/diagnostic_runs"),
    drive_project_name: str = "latent_risk_factor",
    keep_n_runs: int = 5,
    mount_point: str = "/content/drive"
) -> Path:
    """
    Main entry point. Creates symlink from local to Drive.

    Steps:
    1. Mount Drive
    2. Create Drive directory structure
    3. Cleanup old runs (keep N most recent)
    4. Create symlink: local_results_dir -> Drive path
    5. Return Drive path

    :param local_results_dir (Path): Local path to redirect (will become symlink)
    :param drive_project_name (str): Project folder name on Drive
    :param keep_n_runs (int): Number of old runs to keep (rotation)
    :param mount_point (str): Google Drive mount point

    :return drive_runs_dir (Path): Path to runs directory on Drive
    """
    if not is_colab():
        print("Not running in Colab - Drive persistence disabled")
        # Ensure local directory exists
        local_results_dir.mkdir(parents=True, exist_ok=True)
        return local_results_dir

    # Mount Drive
    drive_root = mount_drive(mount_point)
    my_drive = drive_root / "MyDrive"

    # Create Drive directory structure
    drive_runs_dir = my_drive / drive_project_name / "results" / "diagnostic_runs"
    drive_runs_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup old runs
    deleted = cleanup_old_runs(drive_runs_dir, keep_n=keep_n_runs)
    if deleted:
        print(f"Cleaned up {len(deleted)} old run(s)")

    # Ensure parent of local_results_dir exists
    local_results_dir = Path(local_results_dir).resolve()
    local_results_dir.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing local directory/symlink if present
    if local_results_dir.is_symlink():
        local_results_dir.unlink()
    elif local_results_dir.exists():
        # Move existing local data to Drive before creating symlink
        print(f"Moving existing local data to Drive...")
        for item in local_results_dir.iterdir():
            dest = drive_runs_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(local_results_dir)

    # Create symlink: local -> Drive
    local_results_dir.symlink_to(drive_runs_dir)
    print(f"Symlink created: {local_results_dir} -> {drive_runs_dir}")

    return drive_runs_dir


def cleanup_old_runs(runs_dir: Path, keep_n: int = 5) -> list[Path]:
    """
    Delete oldest runs beyond keep_n.

    :param runs_dir (Path): Directory containing run folders
    :param keep_n (int): Number of most recent runs to keep

    :return deleted (list[Path]): List of deleted directory paths
    """
    if not runs_dir.exists():
        return []

    # Get all run directories (format: YYYY-MM-DD_HHMMSS)
    run_dirs = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and _is_valid_run_name(d.name)
    ]

    if len(run_dirs) <= keep_n:
        return []

    # Sort by name (timestamp format sorts chronologically)
    run_dirs.sort(key=lambda d: d.name, reverse=True)

    # Delete oldest beyond keep_n
    deleted: list[Path] = []
    for old_run in run_dirs[keep_n:]:
        try:
            shutil.rmtree(old_run)
            deleted.append(old_run)
        except OSError as e:
            print(f"Warning: Could not delete {old_run}: {e}")

    return deleted


def _is_valid_run_name(name: str) -> bool:
    """
    Check if directory name matches run timestamp format (YYYY-MM-DD_HHMMSS).

    :param name (str): Directory name to check

    :return is_valid (bool): True if matches expected format
    """
    if len(name) != 17:  # YYYY-MM-DD_HHMMSS
        return False
    try:
        datetime.strptime(name, "%Y-%m-%d_%H%M%S")
        return True
    except ValueError:
        return False


def list_runs(runs_dir: Path) -> list[dict]:
    """
    List all runs with metadata.

    :param runs_dir (Path): Directory containing run folders

    :return runs (list[dict]): List of dicts with keys:
        - name: Run directory name
        - path: Full path to run
        - size_mb: Total size in MB
        - last_stage: Last completed stage (from state file)
        - created: Creation timestamp
    """
    if not runs_dir.exists():
        return []

    runs: list[dict] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not _is_valid_run_name(run_dir.name):
            continue

        # Calculate size
        size_bytes = sum(
            f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
        )
        size_mb = size_bytes / (1024 * 1024)

        # Get last stage from state file
        last_stage = "UNKNOWN"
        state_files = list(run_dir.glob("*_state.json"))
        if state_files:
            import json
            try:
                with open(state_files[0], "r") as f:
                    state = json.load(f)
                    last_stage = state.get("current_stage", "UNKNOWN")
            except (json.JSONDecodeError, OSError):
                pass

        # Parse creation time from name
        try:
            created = datetime.strptime(run_dir.name, "%Y-%m-%d_%H%M%S")
        except ValueError:
            created = datetime.min

        runs.append({
            "name": run_dir.name,
            "path": run_dir,
            "size_mb": size_mb,
            "last_stage": last_stage,
            "created": created
        })

    # Sort by creation time (newest first)
    runs.sort(key=lambda r: r["created"], reverse=True)
    return runs


def get_latest_run(runs_dir: Path) -> Path | None:
    """
    Get most recent run directory for resume.

    :param runs_dir (Path): Directory containing run folders

    :return latest_run (Path | None): Path to most recent run, or None if empty
    """
    runs = list_runs(runs_dir)
    if not runs:
        return None
    return runs[0]["path"]
