"""Replacing the database with the verified new file, by an atomic rename."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from . import measure
from .config import Stop
from .connect import _open_immutable
from .world import _fsync_file

# Every SQL string here is built from constants or from table and column names read out of the database's own
# catalogue (and quoted); nothing typed by a user is ever put into a statement.
# ruff: noqa: S608


def _owner(path):
    st = Path(path).stat()
    return st.st_uid, st.st_gid


def _swap(cfg, env, new_file, before):
    db = cfg.db
    wal = Path(str(db) + "-wal")
    shm = Path(str(db) + "-shm")
    if wal.exists() and wal.stat().st_size > 0:
        raise Stop("swap", f"{wal} is not empty, so the database is not cleanly closed; not swapping")
    for p in (wal, shm):
        if p.exists():
            p.unlink()
    st = db.stat()
    os.chmod(new_file, stat.S_IMODE(st.st_mode))
    try:
        os.chown(new_file, st.st_uid, st.st_gid)
    except PermissionError:
        pass
    if _owner(new_file) != (st.st_uid, st.st_gid):
        raise Stop("swap", "could not give the new file the database's owner; not swapping")
    try:
        os.replace(new_file, db)
    except OSError as exc:
        raise Stop("swap", f"could not replace the database file ({exc}). The verified new file is at {new_file} "
                           f"and the backup is intact; with the manager stopped, put the new file in place by "
                           f"hand") from exc
    _fsync_file(db.parent)
    conn = _open_immutable(db, env)
    try:
        counts = {t: conn.execute(f"SELECT count(*) FROM {measure._quote(t)}").fetchone()[0]
                  for t in before["row_counts"]}
        ok, rows = measure._integrity(conn)
    finally:
        conn.close()
    if counts != before["row_counts"] or not ok:
        raise Stop("swap", f"the swapped database failed its check ({rows}); restore from the backup")
    return {"ok": True, "integrity": rows}
