"""
End-to-end pipeline diagnostics: determinism, alignment, universe tracking.

Implements critical validation checks:
- Inference determinism (encode() reproducibility)
- Window alignment validation
- Universe attrition tracking per date

Reference: ISD diagnostic gaps D.30-D.32.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# D.30: Inference determinism verification
# ---------------------------------------------------------------------------

def verify_inference_determinism(
    model: torch.nn.Module,
    windows: torch.Tensor,
    n_samples: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Verify that VAE encoder produces deterministic outputs in eval mode.

    Runs encode() twice on the same inputs and checks if μ differs by < tol.
    In eval mode, no dropout/batchnorm randomness should affect the encoder.

    :param model (torch.nn.Module): VAE model with encode() method
    :param windows (torch.Tensor): Input windows (N, T, F)
    :param n_samples (int): Number of windows to test
    :param tol (float): Tolerance for difference check
    :param seed (int): Random seed for sample selection

    :return analysis (dict): Determinism verification results
    """
    if windows.size(0) == 0:
        return {"available": False, "reason": "empty windows"}

    # Ensure model is in eval mode
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    n_test = min(n_samples, windows.size(0))

    # Select random subset
    rng = np.random.default_rng(seed)
    indices = rng.choice(windows.size(0), size=n_test, replace=False)
    test_windows = windows[indices].to(device)

    max_diff = 0.0
    mean_diff = 0.0
    n_violations = 0
    violation_indices: list[int] = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x = test_windows[i:i+1]

            # First encode
            encode_fn = getattr(model, "encode", None)
            if encode_fn is not None:
                mu1, _ = encode_fn(x)
            else:
                # Fall back to forward pass
                _, mu1, _ = model(x)

            # Second encode (should be identical in eval mode)
            if encode_fn is not None:
                mu2, _ = encode_fn(x)
            else:
                _, mu2, _ = model(x)

            # Check difference
            diff = float(torch.max(torch.abs(mu1 - mu2)).item())
            mean_diff += diff
            max_diff = max(max_diff, diff)

            if diff > tol:
                n_violations += 1
                violation_indices.append(int(idx))

    mean_diff = mean_diff / n_test

    # Restore training mode if needed
    if was_training:
        model.train()

    is_deterministic = n_violations == 0

    return {
        "available": True,
        "n_tested": n_test,
        "is_deterministic": is_deterministic,
        "n_violations": n_violations,
        "violation_indices": violation_indices[:10],  # First 10
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tol,
        "eval_mode_was_set": not was_training,
    }


# ---------------------------------------------------------------------------
# D.31: Window alignment validation
# ---------------------------------------------------------------------------

def validate_window_alignment(
    window_metadata: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate that sliding windows are properly aligned.

    Checks:
    - No gaps in date sequences per stock
    - Chronological ordering
    - Consistent window lengths
    - No overlapping windows (for same stock)

    :param window_metadata (pd.DataFrame): Window metadata with columns:
        stock_id, start_date, end_date, window_length

    :return analysis (dict): Alignment validation results
    """
    required_cols = ["stock_id", "start_date", "end_date"]
    if not all(col in window_metadata.columns for col in required_cols):
        return {
            "available": False,
            "reason": f"missing columns, need {required_cols}",
        }

    if len(window_metadata) == 0:
        return {"available": False, "reason": "empty metadata"}

    n_windows = len(window_metadata)
    n_stocks = window_metadata["stock_id"].nunique()

    # Convert dates to comparable format
    df = window_metadata.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    # Check window lengths
    df["length"] = (df["end_date"] - df["start_date"]).dt.days + 1
    length_arr = np.asarray(df["length"].tolist(), dtype=np.float64)
    length_stats = {
        "mean": float(np.mean(length_arr)),
        "std": float(np.std(length_arr)),
        "min": int(np.min(length_arr)),
        "max": int(np.max(length_arr)),
    }
    length_consistent = df["length"].std() < 1.0  # Within 1 day

    # Per-stock validation
    gap_violations: list[dict[str, Any]] = []
    order_violations: list[dict[str, Any]] = []
    overlap_violations: list[dict[str, Any]] = []

    for stock_id, group in df.groupby("stock_id"):
        # Sort by start date
        sorted_group = group.sort_values("start_date")

        for i in range(1, len(sorted_group)):
            prev = sorted_group.iloc[i-1]
            curr = sorted_group.iloc[i]

            # Check chronological order
            if curr["start_date"] < prev["start_date"]:
                order_violations.append({
                    "stock_id": stock_id,
                    "window_idx": i,
                    "prev_start": str(prev["start_date"]),
                    "curr_start": str(curr["start_date"]),
                })

            # Check for gaps (expected: curr_start = prev_end + 1 day for rolling)
            expected_start = prev["end_date"] + pd.Timedelta(days=1)
            gap_days = (curr["start_date"] - expected_start).days

            if gap_days > 1:  # More than 1 day gap
                gap_violations.append({
                    "stock_id": stock_id,
                    "window_idx": i,
                    "gap_days": gap_days,
                    "prev_end": str(prev["end_date"]),
                    "curr_start": str(curr["start_date"]),
                })

            # Check for overlaps
            if curr["start_date"] < prev["end_date"]:
                overlap_days = (prev["end_date"] - curr["start_date"]).days
                overlap_violations.append({
                    "stock_id": stock_id,
                    "window_idx": i,
                    "overlap_days": overlap_days,
                })

    # Summary
    is_valid = (
        len(gap_violations) == 0 and
        len(order_violations) == 0 and
        len(overlap_violations) == 0 and
        length_consistent
    )

    return {
        "available": True,
        "n_windows": n_windows,
        "n_stocks": n_stocks,
        "is_valid": is_valid,
        # Length analysis
        "length_stats": length_stats,
        "length_consistent": length_consistent,
        # Violations
        "n_gap_violations": len(gap_violations),
        "n_order_violations": len(order_violations),
        "n_overlap_violations": len(overlap_violations),
        "gap_violations": gap_violations[:10],  # First 10
        "order_violations": order_violations[:10],
        "overlap_violations": overlap_violations[:10],
    }


# ---------------------------------------------------------------------------
# D.32: Universe attrition tracking
# ---------------------------------------------------------------------------

def track_universe_attrition(
    universe_snapshots: dict[str, list[int]],
) -> dict[str, Any]:
    """
    Track universe size changes across dates.

    Identifies:
    - Stocks entering/exiting the universe
    - Sudden drops in universe size
    - Trend in universe size over time

    :param universe_snapshots (dict): date_str -> list of active stock_ids

    :return analysis (dict): Universe attrition metrics
    """
    if not universe_snapshots:
        return {"available": False, "reason": "empty universe snapshots"}

    # Sort dates chronologically
    sorted_dates = sorted(universe_snapshots.keys())
    n_dates = len(sorted_dates)

    # Universe size per date
    sizes = [len(universe_snapshots[d]) for d in sorted_dates]
    sizes_arr = np.array(sizes)

    # Basic statistics
    size_mean = float(np.mean(sizes_arr))
    size_std = float(np.std(sizes_arr))
    size_min = int(np.min(sizes_arr))
    size_max = int(np.max(sizes_arr))

    # Entries and exits
    entries_per_date: list[int] = []
    exits_per_date: list[int] = []
    large_drops: list[dict[str, Any]] = []

    prev_set: set[int] = set()
    for i, date in enumerate(sorted_dates):
        curr_set = set(universe_snapshots[date])

        if i > 0:
            entries = len(curr_set - prev_set)
            exits = len(prev_set - curr_set)
            entries_per_date.append(entries)
            exits_per_date.append(exits)

            # Large drop detection (> 10% of previous)
            size_change = sizes[i] - sizes[i-1]
            if size_change < -0.1 * sizes[i-1]:
                large_drops.append({
                    "date": date,
                    "size_change": size_change,
                    "prev_size": sizes[i-1],
                    "curr_size": sizes[i],
                    "pct_change": size_change / sizes[i-1],
                })

        prev_set = curr_set

    # Turnover rate (average % of stocks changing per date)
    if entries_per_date:
        avg_entries = float(np.mean(entries_per_date))
        avg_exits = float(np.mean(exits_per_date))
        avg_turnover = (avg_entries + avg_exits) / (2 * max(size_mean, 1))
    else:
        avg_entries = 0.0
        avg_exits = 0.0
        avg_turnover = 0.0

    # Trend analysis
    if n_dates >= 3:
        x = np.arange(n_dates)
        slope, _ = np.polyfit(x, sizes_arr, 1)
        if slope > 1.0:
            trend = "growing"
        elif slope < -1.0:
            trend = "shrinking"
        else:
            trend = "stable"
        daily_slope = float(slope)
    else:
        trend = "insufficient_data"
        daily_slope = 0.0

    # Stability score (1 - CV)
    stability = 1.0 - min(size_std / max(size_mean, 1), 1.0)

    return {
        "available": True,
        "n_dates": n_dates,
        # Size statistics
        "size_mean": size_mean,
        "size_std": size_std,
        "size_min": size_min,
        "size_max": size_max,
        "size_range": size_max - size_min,
        # Entry/exit statistics
        "avg_entries_per_date": avg_entries,
        "avg_exits_per_date": avg_exits,
        "avg_turnover_rate": avg_turnover,
        # Large drops
        "n_large_drops": len(large_drops),
        "large_drops": large_drops[:10],  # First 10
        # Trend
        "trend": trend,
        "daily_slope": daily_slope,
        "stability_score": stability,
        # Time series (last 50)
        "sizes": sizes[-50:],
        "dates": sorted_dates[-50:],
    }


# ---------------------------------------------------------------------------
# Combined E2E validation
# ---------------------------------------------------------------------------

def run_e2e_validation(
    model: torch.nn.Module | None = None,
    windows: torch.Tensor | None = None,
    window_metadata: pd.DataFrame | None = None,
    universe_snapshots: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """
    Run all E2E validation checks.

    :param model (torch.nn.Module | None): VAE model for determinism check
    :param windows (torch.Tensor | None): Input windows
    :param window_metadata (pd.DataFrame | None): Window metadata
    :param universe_snapshots (dict | None): Universe snapshots

    :return results (dict): Combined E2E validation results
    """
    results: dict[str, Any] = {"available": True}

    # Determinism check
    if model is not None and windows is not None:
        results["determinism"] = verify_inference_determinism(model, windows)
    else:
        results["determinism"] = {"available": False, "reason": "model or windows not provided"}

    # Window alignment
    if window_metadata is not None:
        results["alignment"] = validate_window_alignment(window_metadata)
    else:
        results["alignment"] = {"available": False, "reason": "window_metadata not provided"}

    # Universe attrition
    if universe_snapshots is not None:
        results["attrition"] = track_universe_attrition(universe_snapshots)
    else:
        results["attrition"] = {"available": False, "reason": "universe_snapshots not provided"}

    # Overall status
    checks_passed = 0
    checks_total = 0

    for key in ["determinism", "alignment", "attrition"]:
        check = results.get(key, {})
        if check.get("available", False):
            checks_total += 1
            if key == "determinism" and check.get("is_deterministic", False):
                checks_passed += 1
            elif key == "alignment" and check.get("is_valid", False):
                checks_passed += 1
            elif key == "attrition" and check.get("stability_score", 0) > 0.8:
                checks_passed += 1

    results["summary"] = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "all_passed": checks_passed == checks_total and checks_total > 0,
    }

    return results
