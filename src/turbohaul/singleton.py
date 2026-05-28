"""Singleton invariant enforcement per v0.2 ARCHITECTURE.md §3.1.

F1 must-fix: turbohaul-manager MUST be the only writer to GPU 0
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
import re
import signal
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)


# Note: absolute path for nvidia-smi (PATH-injection-resistant).
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


def _detect_subreaper_pid() -> int | None:
    """Note: detect a sub-reaper PID for orphan-detection (containers
    using tini / systemd Restart=always / PR_SET_CHILD_SUBREAPER).

    Returns the PID of the manager process's OLDEST ancestor that is
    NOT pid 1, or None if the manager IS pid 1. Anything reparented to
    this subreaper (or to pid 1) is a candidate orphan.
    """
    try:
        ppid = os.getppid()
    except Exception:
        return None
    if ppid == 1:
        return None
    # Walk parent chain until we hit pid 1.
    seen: set[int] = set()
    current = ppid
    for _ in range(50):  # bounded
        if current in seen or current <= 1:
            break
        seen.add(current)
        try:
            status_text = Path(f"/proc/{current}/status").read_text(
                errors="ignore"
            )
        except (FileNotFoundError, PermissionError, OSError):
            break
        next_ppid: int | None = None
        for line in status_text.splitlines():
            if line.startswith("PPid:"):
                with contextlib.suppress(ValueError, IndexError):
                    next_ppid = int(line.split()[1])
                break
        if next_ppid is None or next_ppid <= 1:
            break
        current = next_ppid
    return current if current > 1 else None


_SUBREAPER_PID: int | None = _detect_subreaper_pid()


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
                _NVIDIA_SMI_PATH,
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
        ppid = _read_proc_ppid(pid)
        # Note: accept PPid in {1, subreaper} so we catch orphans
        # reparented to a sub-reaper (tini / systemd) rather than init.
        allowed_reapers = {1}
        if _SUBREAPER_PID is not None:
            allowed_reapers.add(_SUBREAPER_PID)
        if ppid not in allowed_reapers:
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


def _read_proc_starttime(pid: int) -> int | None:
    """Note: read /proc/<pid>/stat field 22 (starttime, jiffies-since-boot).

    Used to distinguish original process from a PID-reused replacement on
    busy systems where the kernel can recycle a freed pid within seconds.
    """
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    # Field 22 is starttime. Fields 1-2 may include spaces in the comm,
    # so split off the comm parenthesized region first.
    rp = stat_text.rfind(")")
    if rp == -1:
        return None
    rest = stat_text[rp + 1:].split()
    if len(rest) < 20:  # field 3..22 -> indices 0..19 in rest
        return None
    with contextlib.suppress(ValueError):
        return int(rest[19])
    return None

def reap_orphan(pid: int, sigterm_wait_s: float = 5.0) -> tuple[bool, str]:
    """SIGTERM the orphan; wait; SIGKILL on timeout. Returns (success, status_str).

    Note: capture starttime BEFORE signaling; compare in final check to
    distinguish original process from a PID-reused replacement.
    """
    original_starttime = _read_proc_starttime(pid)
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
        # Note: verify the alive pid is still our original process by
        # comparing starttime; if different the original is gone and a
        # new process re-used the pid.
        current_starttime = _read_proc_starttime(pid)
        if (
            original_starttime is not None
            and current_starttime is not None
            and current_starttime != original_starttime
        ):
            return True, "sigkill-clean-pid-reused"
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


def intra_lifetime_orphan_scan(
    port_base: int,
    known_handle_pids: set[int],
    port_range: int = 100,
) -> dict:
    """Fix: detect llama-server processes bound to our port
    range whose PID is NOT in the live-handle set.

    These are orphans from lost-handle bugs (CancelledError unwind,
    exception inside finally, ``_active_handle = None`` without prior
    sigterm) — invisible to boot_orphan_reaper because they are STILL
    parented to the running manager (PPid != 1). Without this scan the
    leak only resolves at manager restart.

    Walks /proc/*/cmdline for llama-server processes; extracts --port
    flag; checks against [port_base, port_base + port_range); SIGTERMs
    any whose PID is not in known_handle_pids.

    Returns ``{"scanned": N, "matched": M, "reaped": K, "errors": E}``.
    """
    stats = {"scanned": 0, "matched": 0, "reaped": 0, "errors": 0}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return stats
    port_re = re.compile(r"--port\s+(\d+)")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return stats
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stats["scanned"] += 1
        cmdline = _read_proc_cmdline(int(entry.name))
        if "llama-server" not in cmdline:
            continue
        m = port_re.search(cmdline)
        if not m:
            continue
        try:
            port = int(m.group(1))
        except ValueError:
            continue
        if not (port_base <= port < port_base + port_range):
            continue
        stats["matched"] += 1
        pid = int(entry.name)
        if pid in known_handle_pids:
            continue
        log.warning(
            "intra_lifetime_orphan_scan: orphan llama-server pid=%d port=%d "
            "(not in known_handle_pids); sending SIGTERM",
            pid, port,
        )
        try:
            os.kill(pid, signal.SIGTERM)
            stats["reaped"] += 1
        except ProcessLookupError:
            pass  # raced; benign
        except (PermissionError, OSError) as e:
            log.warning(
                "intra_lifetime_orphan_scan: failed SIGTERM pid %d: %s",
                pid, e,
            )
            stats["errors"] += 1
    return stats
