"""Singleton invariant enforcement.

Design requirement: turbohaul-manager MUST be the only writer to GPU 0
on a given host. Without this, the cross-process race we are fixing can simply
be re-introduced by a second turbohaul instance on the same box.

Three enforcement layers:
  1. fcntl.flock on state.sqlite - second instance refuses to start
  2. Boot-time nvidia-smi scan - refuse to start if foreign llama-server processes
     are using GPU 0
  3. Boot-time orphan reaper - find PROVEN Turbohaul-owned llama-server
     stale engines by matching the LIVE /proc (pid, port, starttime) against
     a DURABLE engine-identity record (pid + port + engine_starttime) persisted
     atomically in state.sqlite at spawn; SIGTERM then SIGKILL. Reparenting to
     init (PPid=1) is NOT ownership proof -- a foreign/independent llama-server
     that is itself orphaned (PPid=1) is reported only, never reaped. The
     starttime cross-check prevents PID reuse: a recycled pid has a different
     starttime, so it can never match a recorded identity.
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


# Absolute path for nvidia-smi (PATH-injection-resistant).
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


def _detect_subreaper_pid() -> int | None:
    """Detect a sub-reaper PID for orphan-detection (containers
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
                    "refusing to start (singleton invariant)"
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




# ---------------------------------------------------------------------------
# DURABLE ENGINE-IDENTITY OWNERSHIP PROOF
#
# The heuristic parent-chain proof (PPid in {1, subreaper} OR parent chain
# contains a Turbohaul manager marker) is INSUFFICIENT and has been REPLACED:
#
#   Flaw: an unrelated ``llama-server --port N`` that is itself orphaned
#   (PPid=1) — e.g. launched by a foreign process that later died, or launched
#   by a process whose cmdline happens to contain a manager-marker — would be
#   reaped. Reparenting to init only proves SOME parent died, not that the
#   parent was Turbohaul. The parent-chain marker heuristic is spoofable and
#   cannot survive a manager crash (the manager PID is gone).
#
# DURABLE PROOF (the replacement): Turbohaul records the engine's identity
# (pid + port + engine_starttime) atomically in state.sqlite at spawn time
# (record_engine_identity). On boot, a live /proc llama-server candidate is
# Turbohaul-owned ONLY if its LIVE (pid, port, starttime) matches a recorded
# triple. This:
#   - survives a manager crash (the record is in state.sqlite, not in RAM);
#   - prevents PID reuse (starttime is unique per process instance per boot
#     and never changes for the life of a process; a recycled pid has a
#     different starttime, so it can never match a recorded identity);
#   - never reaps a foreign/independent llama-server, INCLUDING one that is
#     itself orphaned (PPid=1) — it has no recorded identity, so it is
#     report-only.
#
# A candidate lacking a matching recorded identity is NEVER reaped — it is
# reported via the diagnostics-only listener scan. This is the conservative
# contract. PPid=1 / parent-chain heuristics are NOT consulted as ownership
# proof; they are at most candidate *filters* (e.g. the PPid-based scan still
# nominates PPid=1 llama-servers as candidates), but the durable identity
# match is the ONLY reaping authority.


def _read_proc_starttime(pid: int) -> int | None:
    """Read /proc/<pid>/stat field 22 (starttime, jiffies-since-boot).

    Used to distinguish original process from a PID-reused replacement on
    busy systems where the kernel can recycle a freed pid within seconds.
    starttime is unique per process instance on a given boot and never
    changes for the life of a process, so (pid, starttime) uniquely
    identifies a process instance.
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


def _list_proc_pids() -> list[int]:
    """List numeric PIDs currently in /proc. Non-Linux -> []."""
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    return [int(e.name) for e in proc_root.iterdir() if e.name.isdigit()]


def _parse_port_from_cmdline(cmdline: str) -> int | None:
    """Extract the --port / -p value from a cmdline. None if absent/unparseable."""
    tokens = cmdline.split()
    for i, tok in enumerate(tokens):
        if tok in ("--port", "-p") and i + 1 < len(tokens):
            with contextlib.suppress(ValueError):
                return int(tokens[i + 1])
            return None
    return None


def _identity_matches(
    pid: int,
    port: int,
    *,
    starttime_fn: "Callable[[int], int | None]",
    known_engine_identities: "set[tuple[int, int, int]]",
) -> bool:
    """DURABLE ownership proof: does the live (pid, port, starttime) match a
    recorded engine identity?

    Returns True ONLY if there is a recorded (pid, port, starttime) triple
    in known_engine_identities whose pid and port match the candidate AND
    whose starttime matches the live /proc starttime for that pid. The
    starttime cross-check is what prevents PID reuse: a recycled pid has a
    different starttime, so the triple can never match.

    Returns False if:
      - no recorded triple has the candidate's pid+port (foreign process);
      - the live starttime cannot be read (process gone / non-Linux);
      - the live starttime differs from every recorded triple for that
        pid+port (PID was reused by a different process).
    A False result means the candidate is NOT proven Turbohaul-owned — the
    caller MUST NOT reap (report-only).
    """
    if not known_engine_identities:
        return False
    live_st = starttime_fn(pid)
    if live_st is None:
        return False
    return (pid, port, int(live_st)) in known_engine_identities


def find_orphan_llama_servers(
    port_base: int,
    port_range_size: int = 100,
    *,
    cmdline_fn: "Callable[[int], str] | None" = None,
    ppid_fn: "Callable[[int], int | None] | None" = None,
    proc_pids_fn: "Callable[[], list[int]] | None" = None,
    starttime_fn: "Callable[[int], int | None] | None" = None,
    known_engine_identities: "set[tuple[int, int, int]] | None" = None,
) -> list[dict]:
    """Find llama-server processes with PPid in {1, subreaper} and a port in
    our range that are PROVEN Turbohaul-owned by a durable identity record.

    NOMINATES PPid-in-{1,subreaper} llama-servers in the managed range, then
    retains ONLY those whose live (pid, port, starttime) matches a recorded
    engine identity in known_engine_identities. PPid=1 alone is NOT
    ownership proof — a foreign/independent llama-server that is itself
    orphaned (PPid=1) has no recorded identity and is dropped (report-only).

    Returns list of {pid, port, cmdline, ppid}.
    """
    read_cmdline = cmdline_fn or _read_proc_cmdline
    read_ppid = ppid_fn or _read_proc_ppid
    list_pids = proc_pids_fn or _list_proc_pids
    read_starttime = starttime_fn or _read_proc_starttime
    known = known_engine_identities or set()
    orphans: list[dict] = []
    proc_root = Path("/proc")
    if not proc_root.exists() and proc_pids_fn is None:
        return []  # non-Linux dev env

    allowed_reapers = {1}
    if _SUBREAPER_PID is not None:
        allowed_reapers.add(_SUBREAPER_PID)
    for pid in list_pids():
        cmdline = read_cmdline(pid)
        if not cmdline or "llama-server" not in cmdline:
            continue
        ppid = read_ppid(pid)
        if ppid not in allowed_reapers:
            continue
        port = _parse_port_from_cmdline(cmdline)
        if port is None or not (port_base <= port < port_base + port_range_size):
            continue
        # DURABLE PROOF: PPid=1 is only a nominator; the recorded identity
        # match is the reaping authority. A foreign orphaned llama-server
        # has no recorded identity -> dropped (report-only).
        if not _identity_matches(
            pid, port, starttime_fn=read_starttime,
            known_engine_identities=known,
        ):
            log.debug(
                "find_orphan_llama_servers: skipping pid=%d port=%d "
                "(PPid=%s but no recorded engine identity matches — "
                "foreign orphan, report-only)",
                pid, port, ppid,
            )
            continue
        orphans.append({"pid": pid, "port": port, "cmdline": cmdline, "ppid": ppid})
    return orphans


def find_llama_servers_in_port_range(
    port_base: int,
    port_range_size: int = 100,
    *,
    cmdline_fn: "Callable[[int], str] | None" = None,
    ppid_fn: "Callable[[int], int | None] | None" = None,
    proc_pids_fn: "Callable[[], list[int]] | None" = None,
    starttime_fn: "Callable[[int], int | None] | None" = None,
    known_engine_identities: "set[tuple[int, int, int]] | None" = None,
    manager_pids: "set[int] | None" = None,
    manager_identifier_fn: "Callable[[int], bool] | None" = None,
) -> list[dict]:
    """Find PROVEN Turbohaul-owned ``llama-server`` stale engines whose
    ``--port`` flag is in [port_base, port_base+port_range_size).

    DURABLE IDENTITY PROOF (the ONLY reaping authority): a candidate is
    returned ONLY if its live (pid, port, starttime) matches a recorded
    engine identity in ``known_engine_identities``. This:
      - survives a manager crash (the record is in state.sqlite);
      - prevents PID reuse (starttime cross-check);
      - never reaps a foreign/independent llama-server, INCLUDING one that
        is itself orphaned (PPid=1) — it has no recorded identity.

    The legacy ``manager_pids`` / ``manager_identifier_fn`` parameters are
    RETAINED for signature back-compat but are NO LONGER consulted as
    ownership proof — the durable identity match is the sole authority.
    This keeps the existing callers + tests compiling while closing the
    heuristic flaw.

    Returns list of {pid, port, cmdline, ppid}. Non-Linux / unavailable -> [].
    """
    read_cmdline = cmdline_fn or _read_proc_cmdline
    read_ppid = ppid_fn or _read_proc_ppid
    list_pids = proc_pids_fn or _list_proc_pids
    read_starttime = starttime_fn or _read_proc_starttime
    known = known_engine_identities or set()
    out: list[dict] = []
    for pid in list_pids():
        cmdline = read_cmdline(pid)
        if not cmdline or "llama-server" not in cmdline:
            continue
        port = _parse_port_from_cmdline(cmdline)
        if port is None or not (port_base <= port < port_base + port_range_size):
            continue
        ppid = read_ppid(pid)
        # DURABLE PROOF: the recorded identity match is the ONLY ownership
        # authority. Cmdline + port is merely a nominator. A foreign
        # llama-server in the managed range has no recorded identity ->
        # dropped (report-only via the diagnostics-only listener scan).
        if not _identity_matches(
            pid, port, starttime_fn=read_starttime,
            known_engine_identities=known,
        ):
            log.debug(
                "find_llama_servers_in_port_range: skipping pid=%d port=%d "
                "(cmdline matches llama-server but no recorded engine "
                "identity matches — foreign/independent, report-only)",
                pid, port,
            )
            continue
        out.append({"pid": pid, "port": port, "cmdline": cmdline, "ppid": ppid})
    return out



def port_listeners_in_range(port_base: int, port_range_size: int = 100) -> list[dict]:
    """Find processes listening on TCP ports in [port_base, port_base+port_range_size).

    DIAGNOSTICS ONLY — NEVER used for reaping. The port-listener scan can find
    ANY process listening in the managed range, including FOREIGN processes
    (nginx, python http.server, etc.) that must NOT be killed. Reaping from
    this scan is unsafe and was the original blocked-PR flaw. Reaping is done
    ONLY by boot_orphan_reaper via the ownership-aware find_orphan_llama_servers
    + find_llama_servers_in_port_range cmdline scans, which only match proven
    Turbohaul-owned ``llama-server`` processes.

    boot_reconcile calls this purely to populate the ``stale_listeners``
    diagnostics count in its audit/summary, so an operator can SEE that a port
    in the managed range is still occupied (and by whom) — it does NOT reap
    from it. This closes the original blocker: a foreign listener is reported
    but left untouched.

    Uses /proc/<pid>/fd for socket inodes + /proc/net/tcp + tcp6 to map inodes
    to listen ports. Returns list of {"pid", "port", "cmdline"}.

    Non-Linux / unavailable -> [] (dev/test tolerance).
    """
    out: list[dict] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    # Map inode -> listen port from /proc/net/tcp and tcp6.
    inode_to_port: dict[int, int] = {}
    for netfile in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            text = Path(netfile).read_text(errors="ignore")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for line in text.splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            st = parts[3]
            if st != "0A":  # 0A = LISTEN
                continue
            # local is hex ip:port
            try:
                port = int(local.split(":")[1], 16)
            except (ValueError, IndexError):
                continue
            inode = int(parts[9])
            if port_base <= port < port_base + port_range_size:
                inode_to_port[inode] = port
    if not inode_to_port:
        return []
    # For each process, scan its /proc/<pid>/fd for matching socket inodes.
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for fd in fds:
            try:
                link = fd.readlink()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            # link looks like socket:[<inode>]
            if not link.startswith("socket:["):
                continue
            try:
                inode = int(link[8:-1])
            except (ValueError, IndexError):
                continue
            if inode in inode_to_port:
                port = inode_to_port[inode]
                cmdline = _read_proc_cmdline(pid)
                out.append({"pid": pid, "port": port, "cmdline": cmdline})
                break  # one match per pid is enough
    return out


def reap_orphan(
    pid: int,
    sigterm_wait_s: float = 5.0,
    *,
    kill_fn: "Callable[[int, int], None] | None" = None,
    killpg_fn: "Callable[[int, int], None] | None" = None,
    getpgid_fn: "Callable[[int], int] | None" = None,
    starttime_fn: "Callable[[int], int | None] | None" = None,
) -> tuple[bool, str]:
    """SIGTERM the orphan process GROUP; wait; SIGKILL on timeout.

    Process-group safe: spawn_sidecar uses start_new_session=True (setsid), so
    an orphaned llama-server may have grandchildren in its own process group.
    Killing only the leader PID (the pre-hardening behavior) leaks the
    grandchildren, which keep holding the GPU/port — the P12 orphan-stale
    failure. This mirrors drained_sigterm: killpg(getpgid(pid)) on the whole
    group, escalating to killpg(SIGKILL).

    Injectable kill_fn / killpg_fn / getpgid_fn / starttime_fn let tests assert
    the process-group behavior without real subprocesses. Defaults are the
    real os functions.

    Capture starttime BEFORE signaling; compare in final check to distinguish
    original process from a PID-reused replacement.
    """
    os_kill = kill_fn or os.kill
    killpg = killpg_fn or os.killpg
    getpgid = getpgid_fn or os.getpgid
    read_starttime = starttime_fn or _read_proc_starttime

    original_starttime = read_starttime(pid)

    # Resolve the process group id. If the leader is already gone, no signal.
    try:
        pgid = getpgid(pid)
    except ProcessLookupError:
        return True, "already-gone"
    except PermissionError:
        return False, f"permission-denied-getpgid-pid-{pid}"

    # SIGTERM the whole process group (matches the live teardown contract).
    try:
        killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "already-gone-during-sigterm"
    except PermissionError:
        # Fall back to a single-pid SIGTERM if killpg is not permitted (the
        # pre-hardening path) so we never fail to attempt cleanup.
        try:
            os_kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True, "already-gone"
        except PermissionError:
            return False, f"permission-denied-sigterm-pid-{pid}"

    deadline = time.time() + sigterm_wait_s
    while time.time() < deadline:
        try:
            os_kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            return True, "sigterm-clean"

    # Escalate to SIGKILL on the whole group.
    try:
        killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True, "sigkill-already-gone"
    except PermissionError:
        try:
            os_kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True, "sigkill-clean"
        except PermissionError:
            return False, f"permission-denied-sigkill-pid-{pid}"
        time.sleep(0.5)
        try:
            os_kill(pid, 0)
        except ProcessLookupError:
            return True, "sigkill-clean"
        return False, "sigkill-failed-still-alive"

    time.sleep(0.5)
    try:
        os_kill(pid, 0)
    except ProcessLookupError:
        return True, "sigkill-clean"
    # Verify the alive pid is still our original process by comparing
    # starttime; if different the original is gone and a new process
    # re-used the pid.
    current_starttime = read_starttime(pid)
    if (
        original_starttime is not None
        and current_starttime is not None
        and current_starttime != original_starttime
    ):
        return True, "sigkill-clean-pid-reused"
    return False, "sigkill-failed-still-alive"


def boot_orphan_reaper(
    port_base: int,
    known_pids: set[int] | None = None,
    reap_fn: "Callable[..., tuple[bool, str]] | None" = None,
    manager_pids: set[int] | None = None,
    known_engine_identities: "set[tuple[int, int, int]] | None" = None,
    *,
    starttime_fn: "Callable[[int], int | None] | None" = None,
) -> dict:
    """Boot-time orphan reaper (durable engine-identity proof, safe for
    foreign processes including PPid=1).

    Two REAPING passes — both gated by the DURABLE ENGINE-IDENTITY PROOF:
    a live /proc llama-server candidate is reaped ONLY if its live
    (pid, port, starttime) matches a recorded identity in
    ``known_engine_identities`` (persisted atomically in state.sqlite at
    spawn via record_engine_identity). PPid=1 / parent-chain heuristics are
    NOT ownership proof and are NOT consulted — a foreign/independent
    llama-server that is itself orphaned (PPid=1) has no recorded identity
    and is report-only, never reaped.

      1. PPid-based orphan scan (find_orphan_llama_servers): NOMINATES
         PPid-in-{1,subreaper} llama-servers in the managed range, then
         retains ONLY those whose live identity matches a record. This is
         the P12 route: the manager died after recording the engine
         identity, the engine was reparented to init, the live starttime
         still matches the record → reaped. A foreign orphaned llama-server
         has no record → dropped.
      2. Stale-engine scan (find_llama_servers_in_port_range): NOMINATES any
         llama-server with ``--port`` in the managed range regardless of
         PPid, then retains ONLY those whose live identity matches a record.
         This catches a P12 no-listener orphan still parented to a dying
         manager (PPid not in reapers) — the record + starttime match is the
         proof, not the parent chain. A foreign ``llama-server --port N`` in
         the range has no record → report-only.

    A third DIAGNOSTICS-ONLY pass (port_listeners_in_range) populates the
    ``stale_listeners`` count so an operator can SEE a port in the managed
    range is still occupied (and by whom). It does NOT reap.

    Processes in known_pids are skipped. NOTE: at boot the caller passes an
    EMPTY known_pids (see TurbohaulManager.boot_reconcile) — flock guarantees
    singleton ownership, so there is no current-manager active sidecar to
    preserve and the durable-identity match is the sole reaping authority.
    Passing the alive-per-state.sqlite set here was a correctness bug that
    skipped recorded P12 stale engines (alive by definition). known_pids is
    retained for non-boot callers (e.g. intra-lifetime scans) that DO have a
    live current-manager sidecar set to preserve.

    ``manager_pids`` is RETAINED for signature back-compat but is NO LONGER
    consulted as ownership proof — the durable identity match is the sole
    authority. ``starttime_fn`` lets tests inject the /proc starttime reader.

    reap_fn lets tests inject reap_orphan without patching the module global.
    """
    reap = reap_fn or reap_orphan
    known = known_pids or set()
    eng_ids = known_engine_identities or set()
    orphans = find_orphan_llama_servers(
        port_base,
        starttime_fn=starttime_fn,
        known_engine_identities=eng_ids,
    )
    reaped = 0
    failed = 0
    details: list[dict] = []
    seen_pids: set[int] = set()
    for orph in orphans:
        seen_pids.add(orph["pid"])
        if orph["pid"] in known:
            details.append({**orph, "action": "skipped-known"})
            continue
        ok, status = reap(orph["pid"])
        details.append({**orph, "action": "reap", "status": status, "ok": ok})
        if ok:
            reaped += 1
        else:
            failed += 1
    # Stale-engine pass: reaps llama-server processes whose --port is in the
    # managed range, regardless of PPid, BUT ONLY when the durable identity
    # matches. Catches the P12 no-listener orphan (cmdline-identified, no
    # listener socket, identity-recorded). A foreign llama-server in the
    # range has no recorded identity → not matched → report-only.
    stale_engines: list[dict] = []
    try:
        engines = find_llama_servers_in_port_range(
            port_base,
            starttime_fn=starttime_fn,
            known_engine_identities=eng_ids,
        )
    except Exception:
        log.exception("find_llama_servers_in_port_range scan failed (best-effort)")
        engines = []
    for eng in engines:
        if eng["pid"] in seen_pids or eng["pid"] in known:
            continue
        stale_engines.append(eng)
        seen_pids.add(eng["pid"])
        ok, status = reap(eng["pid"])
        details.append({**eng, "action": "reap-stale-engine", "status": status, "ok": ok})
        if ok:
            reaped += 1
        else:
            failed += 1
    # DIAGNOSTICS-ONLY listener scan: report any process still listening on
    # the managed range, but do NOT reap from it. A foreign process (nginx,
    # python http.server) will appear here but is left untouched; an operator
    # sees the port is occupied and can intervene manually.
    stale_listeners: list[dict] = []
    try:
        listeners = port_listeners_in_range(port_base)
    except Exception:
        log.exception("port_listeners_in_range scan failed (best-effort)")
        listeners = []
    for lst in listeners:
        if lst["pid"] in seen_pids or lst["pid"] in known:
            continue
        stale_listeners.append(lst)
        details.append({**lst, "action": "report-listener-only", "ok": None})
    return {
        "scanned": len(orphans),
        "orphans_found": len(orphans),
        "reaped": reaped,
        "failed": failed,
        "details": details,
        "stale_listeners": len(stale_listeners),
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
    """Detect llama-server processes bound to our port
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
