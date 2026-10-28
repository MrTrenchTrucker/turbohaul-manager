"""The backup: a consistent snapshot that must verify before anything is written to the database."""

from __future__ import annotations

import os
from pathlib import Path

from . import measure
from .checks import _verify_copy
from .config import Stop
from .private import _create_private, _require_private
from .report import _step
from .world import _fsync_file


def _backup_paths(cfg, stamp):
    """The final name of this run's backup and the name it carries while it is unverified."""
    final = cfg.backup_dir / f"{cfg.db.name}.pre-cleanup-{stamp}"
    return final, Path(str(final) + ".partial")


def _take_backup(cfg, env, conn, report, before, stamp):
    ck = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if ck[0] != 0:
        raise Stop("backup", "could not drain the write-ahead log into the database before the backup")
    final, partial = _backup_paths(cfg, stamp)
    if final.exists() or partial.exists():
        raise Stop("backup", f"{final} (or its .partial) already exists; the tool never overwrites a backup")
    pages = measure._page_info(conn)

    def make():
        # The file exists, empty and owner-only, before SQLite writes a byte into it, so the copy of the whole
        # database is never readable by others, not even for a moment; a copy that ends up wider is refused.
        _create_private(partial)
        env.backup_fn(conn, partial)
        _require_private(partial, "backup")

    _step(report, env, "backup", make)
    check = _step(report, env, "verify-backup", lambda: _verify_copy(
        partial, before, env, "backup", check_pages=pages["file_bytes"]))
    if not check["ok"]:
        raise Stop("verify-backup", "the backup did not verify, so nothing was written to the database: " +
                                    "; ".join(check["problems"]) + f" (left at {partial})")
    os.replace(partial, final)
    _fsync_file(cfg.backup_dir)
    check["path"] = str(final)
    check["mode"] = _require_private(final, "backup")
    return check
