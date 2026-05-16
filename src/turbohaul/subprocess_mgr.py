"""Supervised subprocess management for llama-server children.

Per v0.2 ARCHITECTURE.md §10 - addresses:
- Brainstormer #12 F3 (SIGTERM 5s too short on 21GB; no orphan reaper)
- Failure Predictor #16 M1 (orphan Popen on parent death)
- Failure Predictor #16 M2 (GRACE→POPPED race + drained-SIGTERM)
- Failure Predictor #16 M3 (Tom's Fork health-contract drift)

Spawn: subprocess.Popen with start_new_session=True (setsid - process group isolation).
Health: poll /health every poll_interval_s, default 600s cold-load tolerance.
Pop: drained-SIGTERM on the whole process group; killpg(SIGKILL) on timeout.
VRAM verify: nvidia-smi cross-check after POPPED before next stage.
Binary integrity: sha256 verify at boot (defense-in-depth, v0.2 §7.1).
"""
import asyncio
import contextlib
import hashlib
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx


log = logging.getLogger(__name__)


class HealthCheckFailed(RuntimeError):
    """llama-server failed to become healthy within loading_health_timeout_s."""


class SchemaMismatch(RuntimeError):
    """Tom's Fork /health response shape changed unexpectedly (FP M3 defense)."""


class SidecarHandle:
    """Wraps a running llama-server subprocess + its identity."""

    def __init__(self, proc: subprocess.Popen, port: int, model_tag: str) -> None:
        self.proc = proc
        self.port = port
        self.model_tag = model_tag
        self.spawned_at = time.monotonic()
        self.activated_at: float | None = None

    @property
    def pid(self) -> int:
        return self.proc.pid

    def is_alive(self) -> bool:
        return self.proc.poll() is None


def spawn_sidecar(
    binary: Path,
    gguf_path: Path,
    port: int,
    model_tag: str,
    argv_flags: list[str],
    popen_factory: Callable[..., subprocess.Popen] | None = None,
) -> SidecarHandle:
    """Spawn a llama-server child in its own process group (setsid).

    popen_factory exists for test injection. Default = subprocess.Popen.
    """
    factory = popen_factory or subprocess.Popen
    cmd = [
        str(binary),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-m", str(gguf_path),
        *argv_flags,
    ]
    log.info("spawning llama-server pid=? port=%d model=%s", port, model_tag)
    proc = factory(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # setsid - own process group → killpg works
    )
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


# Defense-in-depth schema check for Tom's Fork /health (FP M3).
# If Tom's Fork changes the response shape, we want a loud failure not silent
# health-pass.
HEALTH_REQUIRED_FIELDS: set[str] = {"status"}
HEALTH_OK_STATUSES: set[str] = {"ok", "ready", "healthy", "loaded"}


async def health_check_once(port: int, http_client: httpx.AsyncClient) -> dict | None:
    """One health probe. Returns parsed JSON on 200, None on non-200 / network error.

    Raises SchemaMismatch if the response shape is unexpectedly different from
    HEALTH_REQUIRED_FIELDS - this is intentional load-bearing visibility for
    Tom's Fork contract drift (FP M3).
    """
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
        raise SchemaMismatch(f"health response not a dict: {type(data).__name__}")
    missing = HEALTH_REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise SchemaMismatch(f"health response missing fields: {missing}")
    return data


async def wait_until_healthy(
    port: int,
    timeout_s: float,
    http_client: httpx.AsyncClient | None = None,
    poll_interval_s: float = 2.0,
) -> bool:
    """Poll /health until 200+ok or timeout. Returns True on healthy, False on timeout."""
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data = await health_check_once(port, http_client)
            except SchemaMismatch:
                # Re-raise so the manager can surface this as a schema-drift event
                raise
            if data is not None:
                status = (data.get("status") or "").lower()
                if status in HEALTH_OK_STATUSES:
                    return True
            await asyncio.sleep(poll_interval_s)
        return False
    finally:
        if own_client:
            await http_client.aclose()


def _default_nvidia_smi_runner() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ],
        text=True,
        timeout=5,
    )


def get_gpu_memory_used_mib(
    nvidia_smi_runner: Callable[..., str] | None = None,
) -> int | None:
    """Return GPU 0 `memory.used` in MiB. None if nvidia-smi unavailable."""
    runner = nvidia_smi_runner or _default_nvidia_smi_runner
    try:
        out = runner()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    try:
        return int(line.strip().split(",")[0].strip())
    except (ValueError, IndexError):
        return None


async def drained_sigterm(
    handle: SidecarHandle,
    drained_window_s: float,
    is_active: bool,
    cold_window_s: float = 5.0,
    killpg_fn: Callable[[int, int], None] | None = None,
    getpgid_fn: Callable[[int], int] | None = None,
    poll_interval_s: float = 0.2,
) -> tuple[bool, str]:
    """SIGTERM the whole process group → wait → SIGKILL on timeout.

    For active slots (is_active=True), use drained_window_s (default 15s in v0.2
    to allow in-flight decode to complete cleanly on 21GB-resident llama-server).
    For cold/IDLE_HOT slots, use cold_window_s (default 5s).

    Returns (success, status_str). Status strings:
      - "already-gone"   process already exited before SIGTERM
      - "sigterm-clean"  exited during drained window
      - "sigkill-clean"  needed SIGKILL escalation
      - "sigterm-failed-*", "sigkill-permission-denied", "sigkill-failed-still-alive"

    killpg_fn / getpgid_fn allow test injection.
    """
    killpg = killpg_fn or os.killpg
    getpgid = getpgid_fn or os.getpgid

    wait_window = drained_window_s if is_active else cold_window_s

    try:
        pgid = getpgid(handle.pid)
    except ProcessLookupError:
        return True, "already-gone"

    try:
        killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "already-gone-during-sigterm"
    except PermissionError as e:
        return False, f"sigterm-failed-permission-denied"

    deadline = time.monotonic() + wait_window
    while time.monotonic() < deadline:
        if handle.proc.poll() is not None:
            return True, "sigterm-clean"
        await asyncio.sleep(poll_interval_s)

    # Escalate to SIGKILL
    try:
        killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True, "sigkill-already-gone"
    except PermissionError:
        return False, "sigkill-permission-denied"

    # Brief final settle
    await asyncio.sleep(0.5)
    if handle.proc.poll() is None:
        return False, "sigkill-failed-still-alive"
    return True, "sigkill-clean"


async def verify_vram_cleared(
    expected_drop_mib: int,
    nvidia_smi_runner: Callable[..., str] | None = None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
) -> tuple[bool, int | None]:
    """After POPPED, poll until VRAM drops by ≥90% of expected.

    Defends against the CUDA-allocator-stuck failure mode flagged by Brainstormer F3.

    Returns (cleared_ok, current_used_mib).
    nvidia_smi_runner=None and unavailable → returns (True, None) (dev tolerance).
    """
    initial = get_gpu_memory_used_mib(nvidia_smi_runner)
    if initial is None:
        return True, None  # nvidia-smi unavailable — trust the kill (dev mode)
    target = max(0, initial - int(expected_drop_mib * 0.9))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = get_gpu_memory_used_mib(nvidia_smi_runner)
        if current is None:
            return True, None
        if current <= target:
            return True, current
        await asyncio.sleep(poll_interval_s)
    return False, get_gpu_memory_used_mib(nvidia_smi_runner)


def verify_binary_sha256(binary_path: Path, expected_sha256: str) -> bool:
    """Verify llama-server binary sha256 matches the pinned value (v0.2 §7.1).

    Empty expected_sha256 = skip verify (dev mode).
    """
    if not expected_sha256:
        return True
    if not binary_path.exists():
        return False
    h = hashlib.sha256()
    with binary_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256
