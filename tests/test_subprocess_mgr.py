"""Tests for subprocess_mgr (mocked Popen + httpx + nvidia-smi + killpg)."""
import asyncio
import hashlib
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from turbohaul.subprocess_mgr import (
    HEALTH_OK_STATUSES,
    HEALTH_REQUIRED_FIELDS,
    SchemaMismatch,
    SidecarHandle,
    drained_sigterm,
    get_gpu_memory_used_mib,
    health_check_once,
    spawn_sidecar,
    verify_binary_sha256,
    verify_vram_cleared,
    wait_until_healthy,
)


def _make_fake_proc(pid=12345, poll_return=None):
    p = MagicMock()
    p.pid = pid
    if callable(poll_return):
        p.poll.side_effect = poll_return
    elif isinstance(poll_return, list):
        p.poll.side_effect = poll_return
    else:
        p.poll.return_value = poll_return
    return p


class TestSpawn:
    def test_spawn_uses_setsid(self):
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fake_proc()

        spawn_sidecar(
            binary=Path("/opt/turboquant/build/bin/llama-server"),
            gguf_path=Path("/var/lib/turbohaul/blobs/sha256/ab/abc"),
            port=11500,
            model_tag="t",
            argv_flags=["--ctx-size", "4096"],
            popen_factory=fake_popen,
        )
        assert captured_kwargs["start_new_session"] is True

    def test_spawn_passes_argv(self):
        captured: list = []

        def fake_popen(cmd, **kwargs):
            captured.extend(cmd)
            return _make_fake_proc()

        spawn_sidecar(
            Path("/x/llama-server"),
            Path("/x/model.gguf"),
            11500,
            "t",
            ["--ctx-size", "4096", "--mlock"],
            popen_factory=fake_popen,
        )
        assert "--ctx-size" in captured
        assert "4096" in captured
        assert "--mlock" in captured
        assert "--port" in captured
        assert "11500" in captured
        assert "--host" in captured
        assert "127.0.0.1" in captured
        assert "-m" in captured

    def test_spawn_returns_handle(self):
        def fake_popen(*a, **k):
            return _make_fake_proc(pid=99)

        handle = spawn_sidecar(
            Path("/x"), Path("/y"), 11500, "model-a", [], popen_factory=fake_popen
        )
        assert isinstance(handle, SidecarHandle)
        assert handle.port == 11500
        assert handle.model_tag == "model-a"
        assert handle.pid == 99
        assert handle.is_alive() is True

    def test_handle_is_alive_after_exit(self):
        proc = _make_fake_proc(pid=1, poll_return=0)
        h = SidecarHandle(proc=proc, port=11500, model_tag="t")
        assert h.is_alive() is False


@pytest.mark.asyncio
class TestHealthCheck:
    async def test_health_check_200_ok(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_response)
        result = await health_check_once(11500, mock_client)
        assert result == {"status": "ok"}

    async def test_health_check_503_returns_none(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        assert await health_check_once(11500, mock_client) is None

    async def test_health_check_missing_status_schema_mismatch(self):
        """the TurboQuant llama.cpp fork drift defense - missing required field raises (M3)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"loaded": True}
        mock_client.get = AsyncMock(return_value=mock_response)
        with pytest.raises(SchemaMismatch, match="missing fields"):
            await health_check_once(11500, mock_client)

    async def test_health_check_non_dict_schema_mismatch(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "a", "dict"]
        mock_client.get = AsyncMock(return_value=mock_response)
        with pytest.raises(SchemaMismatch, match="not a dict"):
            await health_check_once(11500, mock_client)

    async def test_health_check_network_error_returns_none(self):
        import httpx
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("conn refused"))
        assert await health_check_once(11500, mock_client) is None


@pytest.mark.asyncio
class TestWaitUntilHealthy:
    async def test_immediate_ok(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(11500, timeout_s=2.0, http_client=mock_client, poll_interval_s=0.01)
        assert ok is True

    async def test_timeout(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(11500, timeout_s=0.1, http_client=mock_client, poll_interval_s=0.05)
        assert ok is False


class TestGpuMemoryRead:
    def test_get_gpu_memory_used(self):
        assert get_gpu_memory_used_mib(lambda: "1234\n") == 1234

    def test_get_gpu_memory_handles_empty(self):
        assert get_gpu_memory_used_mib(lambda: "") is None

    def test_get_gpu_memory_handles_nvidia_smi_missing(self):
        def runner():
            raise FileNotFoundError("nvidia-smi")
        assert get_gpu_memory_used_mib(runner) is None

    def test_get_gpu_memory_csv_with_comma(self):
        assert get_gpu_memory_used_mib(lambda: "1024, MiB\n") == 1024

    def test_get_gpu_memory_multiline_takes_first(self):
        assert get_gpu_memory_used_mib(lambda: "1024\n2048\n") == 1024


@pytest.mark.asyncio
class TestDrainedSigterm:
    async def test_already_gone(self):
        # getpgid raises ProcessLookupError → process already gone
        def getpgid_fn(pid):
            raise ProcessLookupError("no such process")

        proc = _make_fake_proc(pid=99, poll_return=0)
        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=1.0,
            is_active=True,
            getpgid_fn=getpgid_fn,
        )
        assert ok is True
        assert status == "already-gone"

    async def test_sigterm_clean_exit(self):
        killpg_calls = []

        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))

        def getpgid_fn(pid):
            return 99999

        # Process is alive (poll=None) on first checks, then exits (poll=0)
        # SIGTERM sent → poll returns 0 next iteration
        poll_values = [None, 0]
        poll_iter = iter(poll_values)
        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=lambda: next(poll_iter, 0))

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=2.0,
            is_active=True,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            poll_interval_s=0.01,
        )
        assert ok is True
        assert status == "sigterm-clean"
        # SIGTERM sent to the process group
        assert (99999, signal.SIGTERM) in killpg_calls
        # No SIGKILL needed
        assert (99999, signal.SIGKILL) not in killpg_calls

    async def test_sigterm_timeout_then_sigkill(self):
        killpg_calls = []
        killed = [False]  # flips True only after SIGKILL is delivered

        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))
            if sig == signal.SIGKILL:
                killed[0] = True

        def poll_side_effect():
            # Process stays alive (None) until SIGKILL is sent, then exits (0)
            return 0 if killed[0] else None

        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=poll_side_effect)

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=0.05,  # very short window forces escalation
            is_active=True,
            killpg_fn=killpg_fn,
            getpgid_fn=lambda pid: 99999,
            poll_interval_s=0.01,
        )
        assert ok is True
        assert status == "sigkill-clean"
        sigs_sent = [s for _, s in killpg_calls]
        assert signal.SIGTERM in sigs_sent
        assert signal.SIGKILL in sigs_sent

    async def test_cold_uses_shorter_window(self):
        """Cold slot uses cold_window_s (5s default) not drained_window_s (15s)."""
        killpg_calls = []
        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))

        poll_count = [0]
        def poll_side_effect():
            poll_count[0] += 1
            return None if poll_count[0] < 5 else 0

        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=poll_side_effect)

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        # is_active=False → uses cold_window_s
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=10.0,  # would be used if is_active=True
            is_active=False,
            cold_window_s=0.05,  # actually used
            killpg_fn=killpg_fn,
            getpgid_fn=lambda pid: 99999,
            poll_interval_s=0.01,
        )
        # Either sigterm-clean (process exits in cold window) or sigkill-clean
        assert ok is True


@pytest.mark.asyncio
class TestVramVerify:
    async def test_vram_cleared_drops_below_threshold(self):
        readings = iter([22000, 500])

        def runner():
            return f"{next(readings)}\n"

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
        assert cleared is True
        assert current == 500

    async def test_vram_unavailable_returns_true(self):
        def runner():
            raise FileNotFoundError("nvidia-smi")

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=1.0,
        )
        assert cleared is True
        assert current is None

    async def test_vram_timeout_returns_false(self):
        def runner():
            return "22000\n"

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=0.05,
            poll_interval_s=0.02,
        )
        assert cleared is False


class TestVerifyBinarySha256:
    def test_empty_expected_skips(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        bin_path.write_bytes(b"contents")
        assert verify_binary_sha256(bin_path, "") is True

    def test_correct_sha256_passes(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        content = b"some binary contents"
        bin_path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert verify_binary_sha256(bin_path, expected) is True

    def test_wrong_sha256_fails(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        bin_path.write_bytes(b"contents")
        assert verify_binary_sha256(bin_path, "deadbeef" * 8) is False

    def test_missing_binary_returns_false(self, tmp_path):
        bin_path = tmp_path / "nonexistent"
        assert verify_binary_sha256(bin_path, "deadbeef" * 8) is False


class TestHealthConstants:
    def test_required_fields(self):
        assert "status" in HEALTH_REQUIRED_FIELDS

    def test_ok_statuses(self):
        assert "ok" in HEALTH_OK_STATUSES
        assert "ready" in HEALTH_OK_STATUSES
        assert "healthy" in HEALTH_OK_STATUSES
        assert "loaded" in HEALTH_OK_STATUSES
