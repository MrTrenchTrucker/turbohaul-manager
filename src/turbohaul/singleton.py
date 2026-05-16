"""Singleton invariant enforcement per v0.2 ARCHITECTURE.md §3.1.

Brainstormer #12 F1 must-fix: turbohaul-manager MUST be the only writer to GPU 0
on a given host. Without this, the cross-process race we are fixing can simply
be re-introduced by a second turbohaul instance on the same box.

Three enforcement layers:
  1. fcntl.flock on state.sqlite - second instance refuses to start
  2. Boot-time nvidia-smi scan - refuse to start if foreign llama-server processes
     are using GPU 0
  3. Boot-time orphan reaper - find llama-server children with PPid=1 (orphaned to
     init) and ports in our runtime.default_port_base range; SIGTERM then SIGKILL
"""
import contextlib
import errno
import fcntl
import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)


class SingletonViolation(RuntimeError):
    """Another turbohaul-manager instance holds the singleton lock."""


@contextlib.contextmanager
def acquire_state_lock(state_db_path: Path) -> Iterator[int]:
    """Acquire exclusive flock on state.sqlite; yield fd; release on exit.

    Raises SingletonViolation if another process already holds it.
    """
    state_db_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_db_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise SingletonViolation(
                    f"another turbohaul-manager already holds flock on {state_db_path}. "
                    "refusing to start (singleton invariant per v0.2 §3.1)"
                ) from e
            raise
        yield fd
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def scan_gpu_compute_apps() -> list[dict]:
    """Return list of {pid, used_memory_mib} from nvidia-smi.

    Returns [] silently if nvidia-smi is unavailable (dev / test environments).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        log.warning("nvidia-smi unavailable; skipping GPU compute-apps scan (dev mode)")
        return []
    apps: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                apps.append({"pid": int(parts[0]), "used_memory_mib": int(parts[1])})
            except ValueError:
                continue
    return apps


def _read_proc_cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_text(errors="ignore")
            .replace("\x00", " ")
            .strip()
        )
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _read_proc_ppid(pid: int) -> int | None:
    try:
        status_text = Path(f"/proc/{pid}/status").read_text(errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for line in status_text.splitlines():
        if line.startswith("PPid:"):
            with contextlib.suppress(ValueError, IndexError):
                return int(line.split()[1])
    return None


def find_orphan_llama_servers(port_base: int, port_range_size: int = 100) -> list[dict]:
    """Find llama-server processes with PPid=1 and a port in our range.

    Returns list of {pid, port, cmdline}.
    """
    orphans: list[dict] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []  # non-Linux dev env

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline = _read_proc_cmdline(pid)
        if not cmdline or "llama-server" not in cmdline:
            continue
        if _read_proc_ppid(pid) != 1:
            continue
        # Try to find --port in cmdline
        port: int | None = None
        tokens = cmdline.split()
        for i, tok in enumerate(tokens):
            if tok in ("--port", "-p") and i + 1 < len(tokens):
                with contextlib.suppress(ValueError):
                    port = int(tokens[i + 1])
                break
        if port is None or not (port_base <= port < port_base + port_range_size):
            continue
        orphans.append({"pid": pid, "port": port, "cmdline": cmdline})
    return orphans


def reap_orphan(pid: int, sigterm_wait_s: float = 5.0) -> tuple[bool, str]:
    """SIGTERM the orphan; wait; SIGKILL on timeout. Returns (success, status_str)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "already-gone"
    except PermissionError:
        return False, f"permission-denied-sigterm-pid-{pid}"

    deadline = time.time() + sigterm_wait_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            return True, "sigterm-clean"

    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        os.kill(pid, 0)
        return False, "sigkill-failed-still-alive"
    except ProcessLookupError:
        return True, "sigkill-clean"
    except PermissionError:
        return False, f"permission-denied-sigkill-pid-{pid}"


def boot_orphan_reaper(port_base: int, known_pids: set[int] | None = None) -> dict:
    """Boot-time orphan reaper.

    Finds llama-server orphans (PPid=1) on our port range, kills those not in
    known_pids (state.sqlite reconciliation set).
    """
    known = known_pids or set()
    orphans = find_orphan_llama_servers(port_base)
    reaped = 0
    failed = 0
    details: list[dict] = []
    for orph in orphans:
        if orph["pid"] in known:
            details.append({**orph, "action": "skipped-known"})
            continue
        ok, status = reap_orphan(orph["pid"])
        details.append({**orph, "action": "reap", "status": status, "ok": ok})
        if ok:
            reaped += 1
        else:
            failed += 1
    return {
        "scanned": len(orphans),
        "orphans_found": len(orphans),
        "reaped": reaped,
        "failed": failed,
        "details": details,
    }


def detect_foreign_gpu_apps(known_pids: set[int] | None = None) -> list[dict]:
    """Detect GPU 0 compute processes that are NOT in our known_pids set."""
    known = known_pids or set()
    apps = scan_gpu_compute_apps()
    foreign: list[dict] = []
    for app in apps:
        if app["pid"] in known:
            continue
        foreign.append({**app, "cmdline": _read_proc_cmdline(app["pid"]) or "<unknown>"})
    return foreign
