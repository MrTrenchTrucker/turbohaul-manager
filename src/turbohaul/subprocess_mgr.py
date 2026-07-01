"""Supervised subprocess management for llama-server children.

Per v0.2 ARCHITECTURE.md §10 - addresses:
- SIGTERM 5s too short on 21GB-resident models; no orphan reaper
- orphan Popen on parent death
- GRACE→POPPED race + drained-SIGTERM
- upstream llama-server health-contract drift

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

from turbohaul.gpu_backend import get_backend


log = logging.getLogger(__name__)


class HealthCheckFailed(RuntimeError):
    """llama-server failed to become healthy within loading_health_timeout_s."""


class SchemaMismatch(RuntimeError):
    """Upstream /health response shape changed unexpectedly (contract-drift defense)."""


class SidecarHandle:
    """Wraps a running llama-server subprocess + its identity."""

    def __init__(
        self,
        proc: subprocess.Popen,
        port: int,
        model_tag: str,
        parallel: int = 1,
    ) -> None:
        self.proc = proc
        self.port = port
        self.model_tag = model_tag
        # Live concurrency width pinned at spawn from the actual --parallel argv.
        # The manager derives its per-model in-flight admission cap from THIS,
        # never a later manifest read (which can drift across warm-inherit reuse).
        self.parallel = parallel
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
    binary_fd: int | None = None,
) -> SidecarHandle:
    """Spawn a llama-server child in its own process group (setsid).

    popen_factory exists for test injection. Default = subprocess.Popen.
    """
    factory = popen_factory or subprocess.Popen
    # If a pinned fd is provided, exec via /proc/self/fd/<fd>
    # so the inode we hashed at boot is exactly what we exec; the path could
    # have been swapped after verify, but the fd still points to the right
    # inode. Falls back to path-based exec when binary_fd is None (dev mode
    # / empty sha256).
    if binary_fd is not None:
        exec_path = f"/proc/self/fd/{binary_fd}"
        pass_fds: tuple[int, ...] = (binary_fd,)
    else:
        exec_path = str(binary)
        pass_fds = ()
    cmd = [
        exec_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-m", str(gguf_path),
        *argv_flags,
    ]
    log.info(
        "spawning llama-server pid=? port=%d model=%s pinned_fd=%s",
        port, model_tag, "yes" if binary_fd is not None else "no",
    )
    # stdout/stderr to DEVNULL — PIPE without an active drainer
    # fills the 64KB OS pipe buffer once llama-server emits enough log lines
    # (model load + slot ops + per-token perf), at which point write(2) blocks
    # inside the child and the drained-SIGTERM contract no longer holds.
    # llama-server has its own --log-file argv option if structured log capture
    # is required; wire it via argv_flags rather than re-introducing PIPE here.
    proc = factory(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # setsid - own process group → killpg works
        pass_fds=pass_fds,
    )
    # Pin the live --parallel width from the spawn argv (default 1 when absent).
    # Single source of truth for the manager's in-flight admission cap.
    parallel = 1
    for i, tok in enumerate(argv_flags):
        if tok == "--parallel" and i + 1 < len(argv_flags):
            try:
                parallel = max(1, int(argv_flags[i + 1]))
            except (TypeError, ValueError):
                parallel = 1
            break
    return SidecarHandle(
        proc=proc, port=port, model_tag=model_tag, parallel=parallel
    )


# Defense-in-depth schema check for the upstream /health endpoint.
# If the upstream server changes the response shape, we want a loud failure not
# silent health-pass.
HEALTH_REQUIRED_FIELDS: set[str] = {"status"}
HEALTH_OK_STATUSES: set[str] = {"ok", "ready", "healthy", "loaded"}


async def health_check_once(port: int, http_client: httpx.AsyncClient) -> dict | None:
    """One health probe. Returns parsed JSON on 200, None on non-200 / network error.

    Raises SchemaMismatch if the response shape is unexpectedly different from
    HEALTH_REQUIRED_FIELDS - this is intentional load-bearing visibility for
    upstream health-contract drift.
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
    is_alive: Callable[[], bool] | None = None,
) -> bool:
    """Poll /health until 200+ok or timeout. Returns True on healthy, False on timeout.

    is_alive: optional liveness probe for the spawned child (SidecarHandle.is_alive).
    When supplied, a child that has EXITED (is_alive() is False) fails fast with
    False instead of burning the full timeout_s (the FSM-wedge fix). LIVENESS-ONLY:
    a slow-but-alive cold load (poll() is None) never trips this, so a legitimate
    large-context load is not killed. Default None keeps existing callers unchanged."""
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
            # FSM-wedge fix: bail fast if the spawned child has already exited
            # (crash / OOM / bad-flag). Checked AFTER the health probe so a child
            # that turns healthy in the same tick still wins; runs every iteration.
            if is_alive is not None and not is_alive():
                log.warning(
                    "wait_until_healthy: child for port %d exited during load "
                    "(is_alive=False) - failing fast instead of waiting out timeout",
                    port,
                )
                return False
            await asyncio.sleep(poll_interval_s)
        return False
    finally:
        if own_client:
            await http_client.aclose()


def _default_nvidia_smi_runner() -> str:
    backend = get_backend()
    if backend is None:
        raise FileNotFoundError("no GPU backend available")
    used = backend.get_gpu_used_mib()
    if used is None:
        raise FileNotFoundError("GPU backend returned no data")
    return str(used)


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
        # Best-effort waitpid on already-gone — reaps zombie
        # if process exited between getpgid and killpg.
        try:
            await asyncio.to_thread(handle.proc.wait, timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return True, "already-gone-during-sigterm"
    except PermissionError as e:
        return False, f"sigterm-failed-permission-denied"

    deadline = time.monotonic() + wait_window
    while time.monotonic() < deadline:
        if handle.proc.poll() is not None:
            # Explicit waitpid reap. poll() already reaps if
            # exited, but be defensive — the zombie-free invariant that
            # the orphan reaper depends on must be guaranteed here.
            try:
                await asyncio.to_thread(handle.proc.wait, timeout=2.0)
            except subprocess.TimeoutExpired:
                pass  # benign — process exited per poll(), waitpid race
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
    # Explicit waitpid reap. Without this, kernel keeps the
    # PID slot occupied as <defunct>; any helper the llama-server spawned that
    # reparents inherits the zombie as PPid, slipping the PPid==1 reaper
    # filter (false-negative).
    try:
        await asyncio.to_thread(handle.proc.wait, timeout=5.0)
    except subprocess.TimeoutExpired:
        return False, "wait-timeout-after-sigkill"
    return True, "sigkill-clean"


async def verify_vram_cleared(
    expected_drop_mib: int,
    nvidia_smi_runner: Callable[..., str] | None = None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
) -> tuple[bool, int | None]:
    """After POPPED, poll until VRAM drops by >= 90% of expected.

    Defends against the CUDA-allocator-stuck failure mode (VRAM not released).

    Returns (cleared_ok, current_used_mib).
    GPU backend unavailable -> returns (True, None) (dev tolerance).
    """
    initial = get_gpu_memory_used_mib(nvidia_smi_runner)
    if initial is None:
        return True, None  # GPU probe unavailable -- trust the kill (dev mode)
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


def open_and_verify_binary(binary_path: Path, expected_sha256: str) -> int | None:
    """Open llama-server binary fd at boot + verify sha256 via fd.

    Returns:
      - the open fd (caller must keep it alive for process lifetime) on hash match
      - None on empty expected_sha256 (dev mode, no pinning needed)
      - None on hash mismatch or path missing

    Pairs with ``spawn_sidecar(..., binary_fd=fd)`` which execs via
    ``/proc/self/fd/<fd>``. Because the fd points at a specific inode that has
    already been hashed, any later attacker-write to ``binary_path`` cannot
    redirect the spawn to a different binary -- even if the path is overwritten,
    renamed, or replaced. The TOCTOU window between hash-check and exec is
    closed at the kernel level (inode pin via open fd).

    O_CLOEXEC is intentionally NOT set: the fd must survive fork+exec so the
    child can resolve ``/proc/self/fd/<fd>`` at exec time. subprocess.Popen
    ``pass_fds`` keeps the fd inheritable across its close-fds sweep.
    """
    if not expected_sha256:
        return None  # dev mode -- no pinning needed
    if not binary_path.exists():
        return None
    fd = os.open(str(binary_path), os.O_RDONLY)
    try:
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        if h.hexdigest() != expected_sha256:
            os.close(fd)
            return None
        # Reset offset (defensive; exec doesn't care).
        os.lseek(fd, 0, 0)
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
