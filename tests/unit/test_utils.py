"""
Unit tests for hardware adaptation utilities (src/utils.py).
"""

import platform
from unittest.mock import patch

import torch

from src.utils import get_amp_config, get_dataloader_kwargs, get_optimal_device


class TestGetOptimalDevice:
    """Tests for get_optimal_device()."""

    def test_returns_torch_device(self) -> None:
        """Return type must be torch.device regardless of platform."""
        device = get_optimal_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_valid(self) -> None:
        """Device type must be one of cpu, cuda, mps."""
        device = get_optimal_device()
        assert device.type in ("cpu", "cuda", "mps")

    @patch("src.utils.torch")
    def test_prefers_mps_over_cuda(self, mock_torch: object) -> None:
        """When both MPS and CUDA are available, prefer MPS."""
        import src.utils as utils_mod
        mock = mock_torch  # type: ignore[assignment]
        mock.backends.mps.is_available.return_value = True
        mock.cuda.is_available.return_value = True
        mock.device = torch.device
        device = utils_mod.get_optimal_device()
        assert device.type == "mps"

    @patch("src.utils.torch")
    def test_falls_back_to_cpu(self, mock_torch: object) -> None:
        """When no GPU is available, return CPU."""
        import src.utils as utils_mod
        mock = mock_torch  # type: ignore[assignment]
        mock.backends.mps.is_available.return_value = False
        mock.cuda.is_available.return_value = False
        mock.device = torch.device
        device = utils_mod.get_optimal_device()
        assert device.type == "cpu"


class TestGetDataloaderKwargs:
    """Tests for get_dataloader_kwargs()."""

    def test_cpu_no_pin_memory(self) -> None:
        """CPU device should not use pin_memory."""
        kwargs = get_dataloader_kwargs(torch.device("cpu"))
        assert kwargs["pin_memory"] is False

    def test_num_workers_non_negative(self) -> None:
        """num_workers must be >= 0."""
        kwargs = get_dataloader_kwargs(torch.device("cpu"))
        assert isinstance(kwargs["num_workers"], int)
        assert kwargs["num_workers"] >= 0  # type: ignore[operator]

    def test_persistent_workers_consistent(self) -> None:
        """persistent_workers must be True iff num_workers > 0."""
        kwargs = get_dataloader_kwargs(torch.device("cpu"))
        n_workers = kwargs["num_workers"]
        expected = int(n_workers) > 0  # type: ignore[arg-type]
        assert kwargs["persistent_workers"] is expected

    def test_mps_no_pin_memory(self) -> None:
        """MPS (unified memory) should not use pin_memory."""
        kwargs = get_dataloader_kwargs(torch.device("mps"))
        assert kwargs["pin_memory"] is False

    @patch("src.utils.os.cpu_count", return_value=0)
    def test_zero_cpus_no_workers(self, _mock: object) -> None:
        """When cpu_count returns 0, num_workers=0 and persistent_workers=False."""
        kwargs = get_dataloader_kwargs(torch.device("cpu"))
        assert kwargs["num_workers"] == 0
        assert kwargs["persistent_workers"] is False

    def test_macos_max_two_workers(self) -> None:
        """On macOS, num_workers should be at most 2."""
        if platform.system() != "Darwin":
            return  # Skip on non-macOS
        kwargs = get_dataloader_kwargs(torch.device("cpu"))
        assert kwargs["num_workers"] <= 2  # type: ignore[operator]


class TestGetAmpConfig:
    """Tests for get_amp_config()."""

    def test_cpu_amp_disabled(self) -> None:
        """CPU should have AMP disabled."""
        cfg = get_amp_config(torch.device("cpu"))
        assert cfg["use_amp"] is False
        assert cfg["use_scaler"] is False
        assert cfg["dtype"] is torch.float32

    def test_mps_amp_disabled(self) -> None:
        """MPS should have AMP disabled (float16 unreliable on MPS)."""
        cfg = get_amp_config(torch.device("mps"))
        assert cfg["use_amp"] is False
        assert cfg["device_type"] == "mps"
        assert cfg["dtype"] is torch.float32
        assert cfg["use_scaler"] is False

    def test_cuda_amp_enabled(self) -> None:
        """CUDA should have AMP enabled (bfloat16 preferred, float16 fallback)."""
        cfg = get_amp_config(torch.device("cuda"))
        assert cfg["use_amp"] is True
        assert cfg["device_type"] == "cuda"
        if torch.cuda.is_bf16_supported():
            assert cfg["dtype"] is torch.bfloat16
            assert cfg["use_scaler"] is False
        else:
            assert cfg["dtype"] is torch.float16
            assert cfg["use_scaler"] is True
