"""llama.cpp (llama-server) backend.

Wraps the TurboQuant llama.cpp fork as a supervised subprocess.
Spawns ``llama-server`` with flags derived from the model manifest.
"""
import asyncio
import logging
import os
import signal
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from turbohaul.backends.base import BackendInterface, SidecarHandle, SpawnRequest


log = logging.getLogger(__name__)


# Resolve nvidia-smi at module load (defense-in-depth against PATH poisoning).
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


class LlamaCppBackend(BackendInterface):
    """llama-server subprocess backend."""

    @property
    def name(self) -> str:
        return "llama.cpp"

    def spawn(self, req: SpawnRequest) -> SidecarHandle:
        """Spawn llama-server child in its own process group (setsid)."""
        binary = req.binary
        gguf_path = req.gguf_path
        if binary is None:
            raise RuntimeError("llama.cpp backend: binary path required")
        if gguf_path is None:
            raise RuntimeError("llama.cpp backend: gguf_path required")

        # Exec path: pinned fd (/proc/self/fd/<N>) or direct path.
        if req.binary_fd is not None:
            exec_path = f"/proc/self/fd/{req.binary_fd}"
            pass_fds: tuple[int, ...] = (req.binary_fd,)
        else:
            exec_path = str(binary)
            pass_fds = ()

        cmd = [
            exec_path,
            "--port", str(req.port),
            "--host", "127.0.0.1",
            "-m", str(gguf_path),
            *req.argv_flags,
        ]

        log.info(
            "spawning llama-server pid=? port=%d model=%s pinned_fd=%s",
            req.port, req.model_tag, "yes" if req.binary_fd is not None else "no",
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=pass_fds,
        )

        return SidecarHandle(proc=proc, port=req.port, model_tag=req.model_tag)

    async def wait_healthy(
        self,
        port: int,
        timeout_s: float,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Poll /health until 200+ok or timeout."""
        async with httpx.AsyncClient() as client:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    r = await client.get(
                        f"http://127.0.0.1:{port}/health", timeout=2.0
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, dict):
                            status = (data.get("status") or "").lower()
                            if status in {"ok", "ready", "healthy", "loaded"}:
                                return True
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(poll_interval_s)
            return False

    async def teardown(
        self,
        handle: SidecarHandle,
        drained_window_s: float,
        is_active: bool,
        cold_window_s: float = 5.0,
    ) -> tuple[bool, str]:
        """SIGTERM process group → wait → SIGKILL on timeout."""
        wait_window = drained_window_s if is_active else cold_window_s

        try:
            pgid = os.getpgid(handle.pid)
        except ProcessLookupError:
            return True, "already-gone"

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                await asyncio.to_thread(handle.proc.wait, timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            return True, "already-gone-during-sigterm"
        except PermissionError:
            return False, "sigterm-failed-permission-denied"

        deadline = time.monotonic() + wait_window
        while time.monotonic() < deadline:
            if handle.proc.poll() is not None:
                try:
                    await asyncio.to_thread(handle.proc.wait, timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                return True, "sigterm-clean"
            await asyncio.sleep(0.2)

        # Escalate to SIGKILL
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True, "sigkill-already-gone"
        except PermissionError:
            return False, "sigkill-permission-denied"

        await asyncio.sleep(0.5)
        if handle.proc.poll() is None:
            return False, "sigkill-failed-still-alive"
        try:
            await asyncio.to_thread(handle.proc.wait, timeout=5.0)
        except subprocess.TimeoutExpired:
            return False, "wait-timeout-after-sigkill"
        return True, "sigkill-clean"

    def make_completion_fn(self, base_url: str):
        """Build httpx-based completion function for llama-server."""
        async def complete_fn(slot: Any, handle: SidecarHandle) -> dict | None:
            """POST to llama-server /v1/chat/completions."""
            # This is a placeholder — the actual implementation lives in
            # api/chat_completion.py's make_llama_server_complete_fn().
            # We keep this method on the interface for consistency.
            await asyncio.sleep(0.001)
            return None

        return complete_fn

    def health_endpoint(self) -> str:
        return "/health"


# Re-export for backward compatibility (existing code imports from subprocess_mgr).
async def health_check_once(port: int, http_client: httpx.AsyncClient) -> dict | None:
    """One health probe. Returns parsed JSON on 200, None on non-200 / network error."""
    try:
        r = await http_client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
    except (httpx.HTTPError, OSError):
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return data


async def wait_until_healthy(
    port: int,
    timeout_s: float,
    http_client: httpx.AsyncClient | None = None,
    poll_interval_s: float = 2.0,
) -> bool:
    """Poll /health until 200+ok or timeout."""
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data = await health_check_once(port, http_client)
            except Exception:
                raise
            if data is not None:
                status = (data.get("status") or "").lower()
                if status in {"ok", "ready", "healthy", "loaded"}:
                    return True
            await asyncio.sleep(poll_interval_s)
        return False
    finally:
        if own_client:
            await http_client.aclose()


def get_gpu_memory_used_mib(
    nvidia_smi_runner: Callable[..., str] | None = None,
) -> int | None:
    """Return GPU 0 memory.used in MiB. None if nvidia-smi unavailable."""
    if nvidia_smi_runner is None:
        try:
            out = subprocess.check_output(
                [
                    _NVIDIA_SMI_PATH,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i", "0",
                ],
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
    else:
        try:
            out = nvidia_smi_runner()
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    try:
        return int(line.strip().split(",")[0].strip())
    except (ValueError, IndexError):
        return None
