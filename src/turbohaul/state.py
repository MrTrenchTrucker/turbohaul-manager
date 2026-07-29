"""state.sqlite - persistent queue snapshot + slot history + audit events.

Per ARCHITECTURE.md. Supports cold-recovery on boot
(orphan reconciliation).
"""
import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2

_SCHEMA: list[str] = [
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS slots (
        slot_id TEXT PRIMARY KEY,
        model_tag TEXT NOT NULL,
        thread_id TEXT,
        state TEXT NOT NULL,
        port INTEGER,
        pid INTEGER,
        engine_starttime INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        ended_at TEXT,
        end_reason TEXT,
        extension_count INTEGER NOT NULL DEFAULT 0,
        client_meta_json TEXT
    )""",
    # v2 migration: add engine_starttime to pre-v2 slots tables. ALTER is
    # idempotent via PRAGMA table_info guard (SQLite re-ALTER raises, so we
    # check the column list first). This runs on every open_state_db call,
    # which is fine: the guard short-circuits once the column exists.
    "ALTER-v2-engine_starttime",  # sentinel; handled below, not executed as SQL
    "CREATE INDEX IF NOT EXISTS idx_slots_state ON slots(state)",
    "CREATE INDEX IF NOT EXISTS idx_slots_pid ON slots(pid)",
    "CREATE INDEX IF NOT EXISTS idx_slots_thread ON slots(thread_id)",
    """CREATE TABLE IF NOT EXISTS audit_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        slot_id TEXT,
        event_type TEXT NOT NULL,
        payload_json TEXT,
        occurred_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_audit_slot ON audit_events(slot_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_events(occurred_at)",
    """CREATE TABLE IF NOT EXISTS pull_history (
        pull_id INTEGER PRIMARY KEY AUTOINCREMENT,
        requester TEXT,
        url TEXT NOT NULL,
        resolved_ip TEXT,
        bytes_done INTEGER NOT NULL DEFAULT 0,
        bytes_expected INTEGER,
        sha256 TEXT,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT
    )""",
]


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp to-the-second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_state_db(
    state_db_path: Path, check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open + initialize state.sqlite. Idempotent.

    PRAGMA busy_timeout = 5000 so transient SQLITE_BUSY
    on concurrent open_state_db calls retry-wait up to 5s instead of
    failing the request with HTTP 500. There are 8 direct callers
    (boot_reconcile + submit + _process_slot + _teardown + _force_cold +
    _audit + _audit_event_only + state_db_session) so contention IS real
    on burst traffic + concurrent audit writes.

    Each thread gets its own connection via `init_audit_pool`
    (thread-local, check_same_thread=True — no cross-thread sharing).
    All other callers keep the sqlite3 default (True).
    """
    state_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(state_db_path),
        isolation_level=None,
        check_same_thread=check_same_thread,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # retry-wait on transient SQLITE_BUSY
    for stmt in _SCHEMA:
        if stmt == "ALTER-v2-engine_starttime":
            # Idempotent v2 migration: add engine_starttime column if absent.
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(slots)").fetchall()
            }
            if "engine_starttime" not in cols:
                conn.execute(
                    "ALTER TABLE slots ADD COLUMN engine_starttime INTEGER"
                )
            continue
        conn.execute(stmt)
    cur = conn.execute(
        "SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,)
    )
    if cur.fetchone() is None:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow_iso()),
        )
    return conn


@contextmanager
def state_db_session(state_db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_state_db(state_db_path)
    try:
        yield conn
    finally:
        conn.close()


# === Audit-write connection pool ============================================
# Thread-local connections — replaces per-event sqlite3.connect/close in
# manager.py's 7 record_audit_event call sites. Each thread gets its own
# connection (created lazily, reused within the thread). No shared mutable
# connection, no check_same_thread=False, no lock needed for connection access.
# NARROW SCOPE: audit_events writes only. state_db_session keeps reads + slot
# writes (mark_slot_ended / upsert_slot / reconcile_orphaned_slots).
_audit_local = threading.local()


def init_audit_pool(state_db_path: Path) -> None:
    """Eager-init the audit connection for the calling thread. Called from
    FastAPI lifespan startup. Idempotent: re-call is a no-op while the
    connection is open for this thread.

    Uses check_same_thread=True (sqlite3 default) — each thread has its own
    connection via threading.local(), so there are no cross-thread access
    issues. No lock needed.
    """
    if getattr(_audit_local, "conn", None) is not None:
        return
    _audit_local.conn = open_state_db(state_db_path, check_same_thread=True)


def close_audit_pool() -> None:
    """Close the audit conn for the calling thread. Called from FastAPI lifespan
    shutdown. Idempotent: re-call is a no-op once closed."""
    conn = getattr(_audit_local, "conn", None)
    if conn is None:
        return
    try:
        conn.close()
    finally:
        _audit_local.conn = None


@contextmanager
def audit_db_session(
    fallback_state_db_path: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Sync-only audit-write session. Yields the thread-local audit conn.

    Each thread gets its own sqlite3 connection (created lazily by
    init_audit_pool). No lock needed — no shared mutable connection.

    SYNC-ONLY (critical): the ctx mgr asserts there is no running event
    loop in the calling thread. From an async context, wrap:
        await asyncio.to_thread(my_sync_audit_fn, args)

    Autocommit is preserved: open_state_db sets isolation_level=None +
    journal_mode=WAL, so each statement auto-commits without an explicit
    transaction. Tested by record_audit_event INSERT becoming visible to a
    second raw conn's SELECT without an intervening commit.

    `fallback_state_db_path`: when the pool has not been initialized (e.g.,
    non-lifespan callers like tests or direct CLI boot), open a per-call conn
    matching the pre-pool behavior. Production callers from FastAPI lifespan
    will always hit the pool branch (init_audit_pool ran at startup); the
    fallback exists so existing unit tests that drive boot_reconcile/_audit
    paths without lifespan still work.
    """
    # Guard: prevent async-context deadlock.
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "audit_db_session is sync-only; threading.Lock would deadlock event loop. "
            "From async context, wrap: await asyncio.to_thread(audit_call, args)"
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e):
            raise
    conn = getattr(_audit_local, "conn", None)
    if conn is not None:
        yield conn
        return
    if fallback_state_db_path is None:
        raise RuntimeError(
            "audit_db_session called before init_audit_pool() and no "
            "fallback_state_db_path provided; production callers should run "
            "inside the FastAPI lifespan (which eager-inits the pool)"
        )
    # Lazy per-call fallback (non-lifespan callers).
    conn = open_state_db(fallback_state_db_path)
    try:
        yield conn
    finally:
        conn.close()


def record_audit_event(
    conn: sqlite3.Connection,
    event_type: str,
    payload: dict | None = None,
    slot_id: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO audit_events (slot_id, event_type, payload_json, occurred_at)
           VALUES (?, ?, ?, ?)""",
        (slot_id, event_type, json.dumps(payload or {}), utcnow_iso()),
    )


def upsert_slot(conn: sqlite3.Connection, slot: dict[str, Any]) -> None:
    """Insert or update a slot row."""
    now = utcnow_iso()
    conn.execute(
        """INSERT INTO slots (
            slot_id, model_tag, thread_id, state, port, pid,
            created_at, updated_at, ended_at, end_reason,
            extension_count, client_meta_json
        ) VALUES (
            :slot_id, :model_tag, :thread_id, :state, :port, :pid,
            :created_at, :updated_at, :ended_at, :end_reason,
            :extension_count, :client_meta_json
        )
        ON CONFLICT(slot_id) DO UPDATE SET
            state=excluded.state,
            thread_id=excluded.thread_id,
            port=excluded.port,
            pid=excluded.pid,
            updated_at=excluded.updated_at,
            ended_at=excluded.ended_at,
            end_reason=excluded.end_reason,
            extension_count=excluded.extension_count,
            client_meta_json=excluded.client_meta_json""",
        {
            "slot_id": slot["slot_id"],
            "model_tag": slot["model_tag"],
            "thread_id": slot.get("thread_id"),
            "state": slot["state"],
            "port": slot.get("port"),
            "pid": slot.get("pid"),
            "created_at": slot.get("created_at") or now,
            "updated_at": now,
            "ended_at": slot.get("ended_at"),
            "end_reason": slot.get("end_reason"),
            "extension_count": slot.get("extension_count", 0),
            "client_meta_json": json.dumps(slot.get("client_meta", {})),
        },
    )


def record_engine_identity(
    conn: sqlite3.Connection,
    slot_id: str,
    pid: int,
    port: int,
    engine_starttime: int | None,
) -> None:
    """DURABLE ENGINE IDENTITY (v2 schema): persist the Turbohaul-spawned
    engine's (pid, port, starttime) atomically with the slot row.

    This is the ONLY ownership proof that survives a manager crash and
    prevents PID reuse. ``pid`` alone is recyclable after a process exits;
    ``engine_starttime`` (jiffies-since-boot from /proc/<pid>/stat field 22)
    is unique per process instance on a given boot and never changes for the
    life of a process, so (pid, starttime) uniquely identifies a process
    instance. ``port`` is a redundant cross-check (the engine's --port).

    The identity triple is written atomically by this single SQL UPDATE in its
    own state-db transaction. It is called by the manager immediately after
    spawn_sidecar returns and the live starttime is read from /proc. It is not
    atomically coupled to process creation: a manager crash between spawn and
    this update leaves an unrecorded engine report-only at recovery time.

    ``engine_starttime`` may be None on non-Linux / when /proc is unavailable
    (dev/test); in that case the column is NULL and the identity proof
    degrades to report-only (a NULL starttime can never match a live one,
    so the candidate is never reaped — the conservative contract holds).
    """
    conn.execute(
        "UPDATE slots SET engine_starttime=?, port=?, pid=?, updated_at=? "
        "WHERE slot_id=?",
        (engine_starttime, port, pid, utcnow_iso(), slot_id),
    )


def known_engine_identities(conn: sqlite3.Connection) -> set[tuple[int, int, int]]:
    """DURABLE ENGINE IDENTITY (v2 schema): the set of (pid, port, starttime)
    triples recorded for slots that should still be running.

    This is the durable ownership proof consumed by boot_orphan_reaper: a
    live /proc llama-server candidate is Turbohaul-owned ONLY if its live
    (pid, port, starttime) matches one of these recorded triples. A NULL
    engine_starttime is excluded (no proof) — the candidate is report-only.

    Mirrors known_active_pids' state filter (NOT POPPED/COLD, ended_at NULL)
    so an ended slot's identity is not treated as live proof.
    """
    cur = conn.execute(
        """SELECT pid, port, engine_starttime FROM slots
           WHERE pid IS NOT NULL
             AND port IS NOT NULL
             AND engine_starttime IS NOT NULL
             AND state NOT IN ('POPPED', 'COLD')
             AND ended_at IS NULL"""
    )
    out: set[tuple[int, int, int]] = set()
    for row in cur.fetchall():
        pid = row["pid"]
        port = row["port"]
        st = row["engine_starttime"]
        if pid is not None and port is not None and st is not None:
            out.add((int(pid), int(port), int(st)))
    return out


def known_active_pids(conn: sqlite3.Connection) -> set[int]:
    """PIDs of slots that should still be running per state.sqlite reconciliation."""
    cur = conn.execute(
        """SELECT pid FROM slots
           WHERE pid IS NOT NULL
             AND state NOT IN ('POPPED', 'COLD')
             AND ended_at IS NULL"""
    )
    return {row["pid"] for row in cur.fetchall() if row["pid"]}


def mark_slot_ended(conn: sqlite3.Connection, slot_id: str, reason: str) -> None:
    now = utcnow_iso()
    conn.execute(
        "UPDATE slots SET state='COLD', ended_at=?, end_reason=?, updated_at=? WHERE slot_id=?",
        (now, reason, now, slot_id),
    )


def reconcile_orphaned_slots(conn: sqlite3.Connection, live_pids: set[int]) -> int:
    """Mark slots as COLD if their pid is no longer alive OR if pre-active orphan.

    Called at boot after orphan reaper runs. Returns count of slots marked.

    Two passes:
    1. Slots with pid set but pid NOT in live_pids -> 'boot-reconcile-orphaned-pid'
    2. Slots with pid IS NULL in a pre-active state (RECEIVED / STAGED /
       LOADING / LOADING_FAIL / GRACE / ACTIVE_MATCH) -> 
       'boot-reconcile-pre-active-orphan'. These cannot be live since they
       were never assigned a pid (caller crashed pre-spawn).

    Second-pass fix: previously pid=NULL slots survived reboots in pre-active
    state forever; the second pass below catches them.
    """
    cur = conn.execute(
        """SELECT slot_id, pid FROM slots
           WHERE pid IS NOT NULL
             AND state NOT IN ('POPPED', 'COLD')
             AND ended_at IS NULL"""
    )
    rows = cur.fetchall()
    n = 0
    for row in rows:
        if row["pid"] not in live_pids:
            mark_slot_ended(conn, row["slot_id"], "boot-reconcile-orphaned-pid")
            n += 1
    # Second pass: pid-NULL pre-active orphans (never spawned, never have a pid)
    cur = conn.execute(
        """SELECT slot_id FROM slots
           WHERE pid IS NULL
             AND state NOT IN ('POPPED', 'COLD', 'IDLE_HOT')
             AND ended_at IS NULL"""
    )
    for row in cur.fetchall():
        mark_slot_ended(
            conn, row["slot_id"], "boot-reconcile-pre-active-orphan"
        )
        n += 1
    return n
