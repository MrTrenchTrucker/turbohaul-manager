"""Measuring a database: row counts, hashes of every column but the stored value, schema, sizes, integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import _BODY_CHUNK, _BODY_COLUMN

# Every SQL string here is built from constants or from table and column names read out of the database's own
# catalogue (and quoted); nothing typed by a user is ever put into a statement.
# ruff: noqa: S608


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _enc(v):
    if v is None:
        return b"N"
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        b = str(v).encode()
        return b"I%d:" % len(b) + b
    if isinstance(v, float):
        b = repr(v).encode()
        return b"F%d:" % len(b) + b
    if isinstance(v, bytes):
        return b"B%d:" % len(v) + v
    b = str(v).encode("utf-8", "surrogatepass")
    return b"S%d:" % len(b) + b


class _Fingerprint:
    """A count plus an order-independent hash of a set of rows (rows may be renumbered by a rebuild)."""

    MOD = 1 << 256

    def __init__(self):
        self.count = 0
        self.acc = 0

    def add(self, row):
        h = hashlib.sha256()
        for v in row:
            h.update(_enc(v))
        self.acc = (self.acc + int.from_bytes(h.digest(), "big")) % self.MOD
        self.count += 1

    def as_dict(self):
        return {"count": self.count, "hash": format(self.acc, "064x")}


def _stream(cursor, chunk):
    while True:
        rows = cursor.fetchmany(chunk)
        if not rows:
            return
        yield from rows


def _list_tables(conn):
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
        "ORDER BY name")]
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone():
        names.append("sqlite_sequence")
    return names


def _table_fingerprints(conn):
    """Per table: row count and hash of every column except the stored request value."""
    out = {}
    for t in _list_tables(conn):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote(t)})")]
        if t == "slots":
            cols = [c for c in cols if c != _BODY_COLUMN]
        fp = _Fingerprint()
        cur = conn.execute(f"SELECT {', '.join(_quote(c) for c in cols)} FROM {_quote(t)}")
        for row in _stream(cur, 1000):
            fp.add(row)
        out[t] = fp.as_dict()
    return out


def _body_fingerprint(conn, where="1=1", params=()):
    """Count and hash of (slot_id, stored value) for the rows that match; values are read a few at a time."""
    fp = _Fingerprint()
    cur = conn.execute(f"SELECT slot_id, {_BODY_COLUMN} FROM slots WHERE {where}", params)
    for row in _stream(cur, _BODY_CHUNK):
        fp.add(row)
    return fp.as_dict()


def _schema_snapshot(conn):
    rows = [list(r) for r in conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")]
    return {
        "objects": rows,
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "application_id": conn.execute("PRAGMA application_id").fetchone()[0],
    }


def _state_counts(conn):
    return {(r[0] or ""): r[1] for r in conn.execute("SELECT state, count(*) FROM slots GROUP BY state")}


def _sum_body_length(conn):
    return conn.execute(f"SELECT coalesce(sum(length({_BODY_COLUMN})), 0) FROM slots").fetchone()[0]


def _file_sizes(db):
    db = Path(db)
    sizes = {"db": db.stat().st_size}
    for suffix in ("-wal", "-shm"):
        p = Path(str(db) + suffix)
        sizes[suffix[1:]] = p.stat().st_size if p.exists() else 0
    return sizes


def _page_info(conn):
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    free = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return {"page_size": page_size, "page_count": page_count, "freelist_count": free,
            "live_bytes": (page_count - free) * page_size, "file_bytes": page_count * page_size}


def _snapshot(conn):
    """Everything compared between the database, its backup and the rebuilt file."""
    tables = _list_tables(conn)
    return {
        "schema": _schema_snapshot(conn),
        "row_counts": {t: conn.execute(f"SELECT count(*) FROM {_quote(t)}").fetchone()[0] for t in tables},
        "state_counts": _state_counts(conn),
        "sum_body_length": _sum_body_length(conn),
        "fingerprints": _table_fingerprints(conn),
    }


def _diff_snapshots(a, b, keys=("schema", "row_counts", "state_counts", "fingerprints")):
    return [k for k in keys if a[k] != b[k]]


def _integrity(conn):
    rows = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    return rows == ["ok"], rows[:5]
