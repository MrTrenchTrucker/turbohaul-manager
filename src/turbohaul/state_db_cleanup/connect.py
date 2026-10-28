"""Opening the database: read-only for the dry run, immutable for copies, exclusive for the real run."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Stop


def _uri(path, query):
    return Path(path).resolve().as_uri() + "?" + query


def _open_readonly(path, env=None):
    conn = sqlite3.connect(_uri(path, "mode=ro"), uri=True, timeout=1.0, isolation_level=None)
    if env is not None and env.on_connect:
        env.on_connect(conn)
    return conn


def _open_immutable(path, env=None):
    conn = sqlite3.connect(_uri(path, "immutable=1"), uri=True, isolation_level=None)
    if env is not None and env.on_connect:
        env.on_connect(conn)
    return conn


def _open_exclusive(path, timeout_s, env=None):
    """Open read-write and take exclusive access for the life of the connection.

    In exclusive locking mode SQLite keeps the lock after the first access, so any other process that
    opens the database while this connection is alive fails with "database is locked". The same holds
    in reverse: if another connection is attached (even an idle one) this open fails.
    """
    conn = None
    try:
        conn = sqlite3.connect(_uri(path, "mode=rw"), uri=True, timeout=timeout_s, isolation_level=None)
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        if conn is not None:
            conn.close()
        raise Stop("lock", f"could not get exclusive access to the database ({exc}). Is the manager stopped "
                           f"and is nothing else using the file?") from exc
    if env is not None and env.on_connect:
        env.on_connect(conn)
    return conn
