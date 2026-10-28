"""The gates that run before anything is written: the paths, the free space, and the comparison of a copy."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from . import measure
from .config import _MIN_SQLITE, Stop
from .connect import _open_immutable
from .marker import _refuse_if_present


def _free_bytes(path, env):
    st = env.statvfs(str(path))
    return st.f_bavail * st.f_frsize


def _space_gate(cfg, env, sizes, pages, plan):
    """free >= backup + estimated new file + margin, summed per volume, measured now."""
    backup_bytes = sizes["db"] + sizes["wal"]
    new_file_bytes = max(pages["live_bytes"] - plan["bytes_removed"], 0)
    backup_dev = env.dev_of(str(cfg.backup_dir))
    work_dev = env.dev_of(str(cfg.work_dir))
    need = {}
    for label, dev, nbytes, path in (("backup", backup_dev, backup_bytes, cfg.backup_dir),
                                     ("new_file", work_dev, new_file_bytes, cfg.work_dir)):
        entry = need.setdefault(dev, {"items": {}, "path": str(path)})
        entry["items"][label] = nbytes
    volumes = []
    failed = []
    for entry in need.values():
        required = sum(entry["items"].values()) + cfg.margin_bytes
        free = _free_bytes(entry["path"], env)
        ok = free >= required
        volumes.append({"path": entry["path"], "items": entry["items"], "margin": cfg.margin_bytes,
                        "required": required, "free": free, "ok": ok})
        if not ok:
            failed.append(entry["path"])
    result = {"backup_bytes": backup_bytes, "new_file_estimate": new_file_bytes, "volumes": volumes,
              "ok": not failed}
    if failed:
        err = Stop("space", "not enough free space on " + ", ".join(failed) +
                            " for the backup plus the new file plus the margin (figures are in the report)")
        err.detail = result
        raise err
    return result

def _check_paths(cfg, env):
    db = cfg.db
    if not db.is_file():
        raise Stop("preflight", f"database not found: {db}")
    if not cfg.backup_dir.is_dir():
        raise Stop("preflight", f"backup directory not found: {cfg.backup_dir}")
    if not cfg.work_dir.is_dir():
        raise Stop("preflight", f"work directory not found: {cfg.work_dir}")
    if cfg.backup_dir.resolve() == db.resolve().parent:
        raise Stop("preflight", "the backup directory is the database's own directory; use a separate one")
    if env.dev_of(str(cfg.work_dir)) != env.dev_of(str(db)):
        raise Stop("preflight", "the work directory must be on the same filesystem as the database so the "
                                "final swap is a rename")
    if tuple(env.sqlite_version) < _MIN_SQLITE:
        raise Stop("preflight", f"SQLite {'.'.join(map(str, env.sqlite_version))} is too old for VACUUM INTO "
                                f"(need {'.'.join(map(str, _MIN_SQLITE))})")
    if cfg.execute and not os.access(cfg.backup_dir, os.W_OK):
        raise Stop("preflight", f"backup directory is not writable: {cfg.backup_dir}")
    if cfg.execute and cfg.backup_dir.stat().st_mode & 0o022:
        raise Stop("preflight", f"the backup directory {cfg.backup_dir} is writable by group or others, so another "
                                f"user could swap or replace the backup; make it writable by its owner only "
                                f"(for example chmod 700)")
    if cfg.execute and env.geteuid() not in (0, db.stat().st_uid):
        raise Stop("preflight", "run as the database file's owner or as root, so the new file can be given the "
                                "same owner")
    if cfg.execute:
        _refuse_if_present(cfg)


def _verify_copy(path, expect, env, label, check_pages=None):
    """Open a copy of the database read-only and compare it with what was measured on the source."""
    problems = []
    size = Path(path).stat().st_size
    if size <= 0:
        problems.append("file is empty")
    if check_pages is not None and size != check_pages:
        problems.append(f"size {size} differs from the source ({check_pages})")
    conn = _open_immutable(path, env)
    try:
        ok, rows = measure._integrity(conn)
        if not ok:
            problems.append(f"integrity_check: {rows}")
        got = measure._snapshot(conn)
        problems += [f"{k} differs" for k in measure._diff_snapshots(expect, got)]
        if expect.get("sum_body_length") is not None:
            if got["sum_body_length"] != expect["sum_body_length"]:
                problems.append("sum(length(client_meta_json)) differs")
            elif expect.get("body_fingerprint") is not None and \
                    measure._body_fingerprint(conn) != expect["body_fingerprint"]:
                # equal lengths can still hide changed text; the hash is read only when the cheap check passed
                problems.append("stored request values (client_meta_json) differ by content hash")
    except sqlite3.Error as exc:
        problems.append(f"could not be read: {exc}")
    finally:
        conn.close()
    return {"label": label, "path": str(path), "size": size, "problems": problems, "ok": not problems}
