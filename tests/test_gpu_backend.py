"""Tests for GPU backend abstraction (gpu_backend.py).

Covers NvidiaBackend, RocmBackend, detect_backend, singleton, and backward-compat
callable injection for subprocess_mgr.get_gpu_memory_used_mib.
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.gpu_backend import (
    NvidiaBackend,
    RocmBackend,
    detect_backend,
    get_backend,
    reset_backend,
    set_backend,
)


class TestNvidiaBackend:
    def test_get_gpu_memory_used_mib(self):
        b = NvidiaBackend()
        with patch.object(subprocess, "check_output", return_value="1234\n"):
            assert b.get_gpu_used_mib() == 1234

    def test_get_gpu_memory_used_mib_none_on_error(self):
        b = NvidiaBackend()
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert b.get_gpu_used_mib() is None

    def test_get_gpu_free_mib(self):
        b = NvidiaBackend()
        with patch.object(subprocess, "check_output", return_value="4096\n8192\n"):
            assert b.get_gpu_free_mib() == [4096, 8192]

    def test_get_gpu_free_mib_none_on_error(self):
        b = NvidiaBackend()
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert b.get_gpu_free_mib() is None

    def test_scan_compute_apps(self):
        b = NvidiaBackend()
        with patch.object(
            subprocess,
            "check_output",
            return_value="1234,512\n5678,1024\n",
        ):
            apps = b.scan_compute_apps()
            assert len(apps) == 2
            assert apps[0]["pid"] == 1234
            assert apps[0]["used_memory_mib"] == 512

    def test_scan_compute_apps_empty(self):
        b = NvidiaBackend()
        with patch.object(subprocess, "check_output", return_value=""):
            assert b.scan_compute_apps() == []

    def test_scan_compute_apps_error(self):
        b = NvidiaBackend()
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert b.scan_compute_apps() == []


def _mock_run(stdout="", returncode=0):
    """Create a mock CompletedProcess for subprocess.run."""
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


class TestRocmBackend:
    def test_get_gpu_memory_used_mib(self):
        b = RocmBackend()
        json_out = '{"card0": {"vram": {"total": 16777216000, "used": 4294967296}}}'
        with patch.object(subprocess, "run", return_value=_mock_run(json_out)):
            assert b.get_gpu_used_mib() == 4294967296 // (1024 * 1024)

    def test_get_gpu_memory_used_mib_none_on_error(self):
        b = RocmBackend()
        with patch.object(
            subprocess, "run", side_effect=FileNotFoundError
        ):
            assert b.get_gpu_used_mib() is None

    def test_get_gpu_memory_used_mib_none_on_bad_json(self):
        b = RocmBackend()
        with patch.object(subprocess, "run", return_value=_mock_run("not json")):
            assert b.get_gpu_used_mib() is None

    def test_get_gpu_free_mib(self):
        b = RocmBackend()
        json_out = '{"card0": {"vram": {"total": 16777216000, "used": 4294967296}}}'
        with patch.object(subprocess, "run", return_value=_mock_run(json_out)):
            free = b.get_gpu_free_mib()
            assert free is not None
            assert free[0] == (16777216000 - 4294967296) // (1024 * 1024)

    def test_get_gpu_free_mib_none_on_error(self):
        b = RocmBackend()
        with patch.object(
            subprocess, "run", side_effect=FileNotFoundError
        ):
            assert b.get_gpu_free_mib() is None

    def test_scan_compute_apps(self):
        b = RocmBackend()
        with patch.object(
            subprocess, "run", return_value=_mock_run("PID  MEM\n1234 512\n")
        ):
            apps = b.scan_compute_apps()
            assert len(apps) == 1
            assert apps[0]["pid"] == 1234

    def test_scan_compute_apps_empty(self):
        b = RocmBackend()
        with patch.object(subprocess, "run", return_value=_mock_run("")):
            assert b.scan_compute_apps() == []

    def test_scan_compute_apps_error(self):
        b = RocmBackend()
        with patch.object(
            subprocess, "run", side_effect=FileNotFoundError
        ):
            assert b.scan_compute_apps() == []

    def test_scan_compute_apps_nonzero_exit(self):
        b = RocmBackend()
        with patch.object(
            subprocess, "run", return_value=_mock_run("error output", returncode=2)
        ):
            assert b.scan_compute_apps() == []

    def test_scan_compute_apps_ambiguous_error(self):
        b = RocmBackend()
        err_text = "rocm-smi: error: ambiguous option: --showpid could match --showpids, --showpidgpus"
        with patch.object(
            subprocess, "run", return_value=_mock_run(err_text, returncode=2)
        ):
            assert b.scan_compute_apps() == []


class TestDetectBackend:
    def test_detect_backend_nvidia(self):
        with patch.object(subprocess, "check_output", return_value="1000\n"):
            b = detect_backend("nvidia")
            assert isinstance(b, NvidiaBackend)

    def test_detect_backend_rocm(self):
        json_out = '{"card0": {"vram": {"total": 16777216000, "used": 4294967296}}}'
        with patch.object(subprocess, "run", return_value=_mock_run(json_out)):
            b = detect_backend("rocm")
            assert isinstance(b, RocmBackend)

    def test_detect_backend_none_available(self):
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert detect_backend("auto") is None

    def test_detect_backend_forced_nvidia(self):
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert detect_backend("nvidia") is None

    def test_detect_backend_forced_rocm(self):
        with patch.object(
            subprocess, "check_output", side_effect=FileNotFoundError
        ):
            assert detect_backend("rocm") is None


class TestBackendSingleton:
    def test_get_backend_initializes(self):
        reset_backend()
        with patch("turbohaul.gpu_backend.detect_backend") as mock:
            mock.return_value = MagicMock()
            b = get_backend()
            assert b is not None
            mock.assert_called_once()

    def test_get_backend_cached(self):
        mock_backend = MagicMock()
        set_backend(mock_backend)
        b = get_backend()
        assert b is mock_backend

    def test_set_backend_overrides(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        set_backend(mock1)
        set_backend(mock2)
        assert get_backend() is mock2

    def test_reset_backend(self):
        set_backend(MagicMock())
        reset_backend()
        assert get_backend() is None


class TestSubprocessMgrBackwardCompat:
    def test_get_gpu_memory_used_mib_with_runner(self):
        from turbohaul.subprocess_mgr import get_gpu_memory_used_mib

        runner = MagicMock(return_value="500\n")
        assert get_gpu_memory_used_mib(nvidia_smi_runner=runner) == 500

    def test_get_gpu_memory_used_mib_with_runner_error(self):
        from turbohaul.subprocess_mgr import get_gpu_memory_used_mib

        runner = MagicMock(side_effect=FileNotFoundError)
        assert get_gpu_memory_used_mib(nvidia_smi_runner=runner) is None

    def test_get_gpu_memory_used_mib_delegates_to_backend(self):
        from turbohaul.subprocess_mgr import get_gpu_memory_used_mib

        mock_backend = MagicMock()
        mock_backend.get_gpu_used_mib.return_value = 2048
        set_backend(mock_backend)
        result = get_gpu_memory_used_mib()
        assert result == 2048
        mock_backend.get_gpu_used_mib.assert_called_once()
        reset_backend()


class TestSafetyBackendCompat:
    def test_read_free_vram_mib_delegates_to_backend(self):
        from turbohaul.safety import _read_free_vram_mib

        mock_backend = MagicMock()
        mock_backend.get_gpu_free_mib.return_value = [8192]
        set_backend(mock_backend)
        result = _read_free_vram_mib()
        assert result == 8192
        reset_backend()

    def test_check_free_vram_uses_backend(self):
        from turbohaul.safety import check_free_vram

        mock_backend = MagicMock()
        mock_backend.get_gpu_free_mib.return_value = [8192]
        set_backend(mock_backend)
        result = check_free_vram(min_free_mib=512)
        assert result.ok is True
        reset_backend()

    def test_check_free_vram_refuses_when_low(self):
        from turbohaul.safety import check_free_vram

        mock_backend = MagicMock()
        mock_backend.get_gpu_free_mib.return_value = [100]
        set_backend(mock_backend)
        result = check_free_vram(min_free_mib=512)
        assert result.ok is False
        reset_backend()
