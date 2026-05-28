"""state.sqlite - persistent queue snapshot + slot history + audit events.

Per v0.2 ARCHITECTURE.md §12. Supports cold-recovery on boot
(orphan reconciliation in §3.1 / §10).
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


SCHEMA_VERSION = 1

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
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        ended_at TEXT,
        end_reason TEXT,
        extension_count INTEGER NOT NULL DEFAULT 0,
        client_meta_json TEXT
    )""",
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

    NEMO V2 4.1 fix: PRAGMA busy_timeout = 5000 so transient SQLITE_BUSY
    on concurrent open_state_db calls retry-wait up to 5s instead of
    failing the request with HTTP 500. GitNexus confirms 8 direct callers
    (boot_reconcile + submit + _process_slot + _teardown + _force_cold +
    _audit + _audit_event_only + state_db_session) so contention IS real
    on burst traffic + concurrent audit writes.

    : `check_same_thread=False` is used only by `init_audit_pool` for
    the long-lived audit conn (cross-thread access serialized by
    `_audit_lock`); all other callers keep the sqlite3 default (True).
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
    conn.execute("PRAGMA busy_timeout=5000")  # NEMO V2 4.1
    for stmt in _SCHEMA:
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


# === Audit-write connection pool ( ) ==========================
# Module-level long-lived conn + threading.Lock — replaces per-event
# sqlite3.connect/close in manager.py's 7 record_audit_event call sites.
# NARROW SCOPE: audit_events writes only. state_db_session keeps reads + slot
# writes (mark_slot_ended / upsert_slot / reconcile_orphaned_slots).
_audit_conn: sqlite3.Connection | None = None
_audit_lock = threading.Lock()


def init_audit_pool(state_db_path: Path) -> None:
    """Eager-init the long-lived audit conn. Called once from FastAPI lifespan
    startup. Idempotent: re-call is a no-op while the pool is open.

    Opens with `check_same_thread=False` because record_audit_event sites can
    fire from worker threads (asyncio.to_thread wrap from async manager paths)
    as well as the main thread. Cross-thread access is serialized by the
    module-level `_audit_lock` so SQLite-level race safety is preserved.
    """
    global _audit_conn
    if _audit_conn is not None:
        return
    _audit_conn = open_state_db(state_db_path, check_same_thread=False)


def close_audit_pool() -> None:
    """Close the long-lived audit conn. Called once from FastAPI lifespan
    shutdown. Idempotent: re-call is a no-op once closed."""
    global _audit_conn
    if _audit_conn is None:
        return
    try:
        _audit_conn.close()
    finally:
        _audit_conn = None


@contextmanager
def audit_db_session(
    fallback_state_db_path: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Sync-only audit-write session. Yields the module-level _audit_conn under
    a threading.Lock to serialize concurrent writers across threads.

    SYNC-ONLY (F1 CRIT): the threading.Lock blocks the current thread and
    would deadlock an asyncio event loop. The ctx mgr asserts there is no
    running event loop in the calling thread. From an async context, wrap:
        await asyncio.to_thread(my_sync_audit_fn, args)

    Autocommit is preserved: open_state_db sets isolation_level=None +
    journal_mode=WAL, so each statement auto-commits without an explicit
    transaction. Tested by record_audit_event INSERT becoming visible to a
    second raw conn's SELECT without an intervening commit.

    `fallback_state_db_path`: when the pool has not been initialized (e.g.,
    non-lifespan callers like tests or direct CLI boot), open a per-call conn
    matching the pre- behavior. Production callers from FastAPI lifespan
    will always hit the pool branch (init_audit_pool ran at startup); the
    fallback exists so existing unit tests that drive boot_reconcile/_audit
    paths without lifespan still work.
    """
    # F1 guard: prevent async-context deadlock.
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "audit_db_session is sync-only; threading.Lock would deadlock event loop. "
            "From async context, wrap: await asyncio.to_thread(audit_call, args)"
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e):
            raise
    if _audit_conn is not None:
        with _audit_lock:
            yield _audit_conn
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

    Fix: previously pid=NULL slots survived reboots in pre-active
    state forever; new second pass catches them.
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
    # Note: pid-NULL pre-active orphans (never spawned, never have a pid)
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
