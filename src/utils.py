"""
Hardware adaptation utilities: device detection, backend config, DataLoader kwargs, AMP config.

Centralizes all platform-specific logic so that the rest of the codebase
stays hardware-agnostic. Auto-detects the best configuration for the
current system (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).
"""

import logging
import os
import platform

import torch

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """
    Auto-detect the best available PyTorch device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU.

    :return device (torch.device): Best available device
    """
    if torch.backends.mps.is_available():
        logger.info("Device auto-detected: MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Device auto-detected: CUDA (%s)", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("Device auto-detected: CPU")
    return torch.device("cpu")


def configure_backend(device: torch.device) -> None:
    """
    Configure global PyTorch backend flags for optimal performance on the given device.

    - CUDA: enable cuDNN auto-tuning, TF32 for matmuls and convolutions
    - MPS / CPU: no-op (no equivalent flags)

    Call once at pipeline start, before creating models or optimizers.

    :param device (torch.device): Target compute device
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        # New TF32 API (PyTorch >= 2.9); fall back to legacy if unavailable
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        else:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        logger.info(
            "CUDA backend configured: cudnn.benchmark=True, TF32 enabled"
        )


def clear_device_cache(device: torch.device) -> None:
    """
    Release cached GPU memory for the given device.

    - CUDA: torch.cuda.empty_cache()
    - MPS: torch.mps.empty_cache() (if available)
    - CPU: no-op

    :param device (torch.device): Target compute device
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _shm_available() -> bool:
    """
    Check if PyTorch shared memory manager is executable.

    Workers need torch_shm_manager to share tensor data across processes.
    If the binary is missing or lacks execute permission, fall back to 0 workers.

    :return available (bool): True if shared memory is usable
    """
    shm_path = os.path.join(os.path.dirname(torch.__file__), "bin", "torch_shm_manager")
    return os.path.isfile(shm_path) and os.access(shm_path, os.X_OK)


def get_dataloader_kwargs(device: torch.device) -> dict[str, object]:
    """
    Compute optimal DataLoader keyword arguments for the current platform.

    - macOS (spawn multiprocessing): max 2 workers (high spawn overhead)
    - Linux/other (fork multiprocessing): max 4 workers
    - pin_memory: True on CUDA only (DMA transfer); False on MPS (unified memory)
    - persistent_workers: True when num_workers > 0 (avoids respawn per epoch)
    - Falls back to 0 workers if torch_shm_manager is not accessible

    :param device (torch.device): Target compute device

    :return kwargs (dict): Keys: num_workers, pin_memory, persistent_workers
    """
    is_macos = platform.system() == "Darwin"
    cpu_count = os.cpu_count() or 0

    if is_macos:
        num_workers = min(2, cpu_count)
    else:
        num_workers = min(4, cpu_count)

    # Fall back to 0 workers if shared memory manager is not available
    if num_workers > 0 and not _shm_available():
        logger.warning("torch_shm_manager not accessible — falling back to num_workers=0")
        num_workers = 0

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    # Prefetch next batches to overlap CPU→GPU transfer with compute
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return kwargs


def get_amp_config(device: torch.device) -> dict[str, object]:
    """
    Determine Automatic Mixed Precision (AMP) configuration for the given device.

    - CUDA (NVIDIA GPU): autocast float16, GradScaler enabled
    - MPS (Apple Silicon): AMP disabled — MPS float16 autocast is unreliable
      and causes NaN in intermediate computations (reductions, exp, log).
      Apple Silicon unified memory makes AMP less beneficial anyway.
    - CPU: AMP disabled (full float32)

    :param device (torch.device): Target compute device

    :return config (dict): Keys: use_amp, device_type, dtype, use_scaler
    """
    device_type = device.type

    if device_type == "cuda":
        # Prefer bfloat16: same exponent range as float32 (max ~3.4e38)
        # so no overflow risk in intermediate Conv1d activations or exp().
        # float16 max is only 65504 — easily overflows with deep CNNs.
        if torch.cuda.is_bf16_supported():
            return {
                "use_amp": True,
                "device_type": device_type,
                "dtype": torch.bfloat16,
                "use_scaler": False,  # bfloat16 doesn't need loss scaling
            }
        return {
            "use_amp": True,
            "device_type": device_type,
            "dtype": torch.float16,
            "use_scaler": True,
        }

    # MPS and CPU: full float32 (MPS float16 autocast causes NaN)
    return {
        "use_amp": False,
        "device_type": device_type if device_type == "mps" else "cpu",
        "dtype": torch.float32,
        "use_scaler": False,
    }
