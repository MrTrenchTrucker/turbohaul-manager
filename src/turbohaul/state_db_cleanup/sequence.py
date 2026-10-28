"""The gated sequence of the cleanup: each step runs only if the one before it passed."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import timedelta

from turbohaul.slot_meta_projection import IDENTITY_KEYS

from . import measure, strip, swap
from .backup import _backup_paths, _take_backup
from .checks import _check_paths, _space_gate
from .config import _TMP_PREFIX, Stop
from .connect import _open_exclusive, _open_readonly
from .marker import _create_marker, _is_finished, _remove_marker
from .rebuild import _verify_new
from .report import _not_written_line, _report_path, _step, _write_report
from .world import Env


def run(cfg, env=None):
    env = env or Env()
    cfg.work_dir = cfg.work_dir or cfg.db.parent
    if cfg.min_age_hours and cfg.min_age_hours > 0 and not cfg.cutoff_iso:
        cfg.cutoff_iso = (env.now() - timedelta(hours=cfg.min_age_hours)).isoformat(timespec="seconds")
    report = {"mode": "execute" if cfg.execute else "dry-run", "db": str(cfg.db),
              "backup_dir": str(cfg.backup_dir), "work_dir": str(cfg.work_dir),
              "started": env.now().isoformat(timespec="seconds"),
              "sqlite_version": ".".join(map(str, env.sqlite_version)), "allowlist": list(IDENTITY_KEYS), "steps": [],
              "db_modified": False, "status": "running"}
    conn = None
    tmp_dir = None
    marker_path = None
    saved_tmpdir = os.environ.get("SQLITE_TMPDIR")
    stamp = env.now().strftime("%Y%m%dT%H%M%SZ")
    report["stamp"] = stamp
    planned_report = _report_path(cfg, report)
    if planned_report is not None:
        report["report_path"] = str(planned_report)
    try:
        _step(report, env, "preflight", lambda: _check_paths(cfg, env))
        if cfg.execute:
            tmp_dir = cfg.work_dir / (_TMP_PREFIX + stamp)
            tmp_dir.mkdir()
            os.environ["SQLITE_TMPDIR"] = str(tmp_dir)
            conn = _step(report, env, "lock", lambda: _open_exclusive(cfg.db, cfg.lock_timeout_s, env))
            conn.execute("PRAGMA temp_store=FILE")
        else:
            conn = _open_readonly(cfg.db, env)
        report["sizes_before"] = measure._file_sizes(cfg.db)
        report["pages_before"] = measure._page_info(conn)
        before = _step(report, env, "measure", lambda: _measure(conn, cfg))
        report["before"] = {k: before[k] for k in ("row_counts", "state_counts", "sum_body_length", "fingerprints",
                                                    "body_fingerprint") if k in before}
        plan = _step(report, env, "plan", lambda: strip._strip_pass(conn, cfg, env, apply=False))
        report["plan"] = plan
        try:
            report["space"] = _step(report, env, "space", lambda: _space_gate(
                cfg, env, report["sizes_before"], report["pages_before"], plan))
        except Stop as stop:
            report["space"] = getattr(stop, "detail", None)
            raise
        if not cfg.execute:
            report["status"] = "dry-run complete"
            return report

        non_cold = measure._body_fingerprint(conn, "state != 'COLD'")
        report["backup_planned"] = str(_backup_paths(cfg, stamp)[0])
        marker_path = _create_marker(cfg, env, report, stamp)
        report["marker"] = str(marker_path)
        backup = _take_backup(cfg, env, conn, report, before, stamp)
        report["backup"] = backup

        report["db_modified"] = True
        applied = _step(report, env, "strip", lambda: strip._strip_pass(conn, cfg, env, apply=True))
        report["strip"] = applied
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _step(report, env, "check-strip", lambda: strip._check_strip(conn, before, plan, applied, non_cold))

        new_file = cfg.work_dir / f"{cfg.db.name}.rebuilt-{stamp}"
        if new_file.exists():
            raise Stop("rebuild", f"{new_file} already exists")
        _step(report, env, "rebuild", lambda: env.vacuum_fn(conn, new_file))
        after_strip = {"body": measure._body_fingerprint(conn)}
        report["new_file"] = str(new_file)
        report["verify_new"] = _step(report, env, "verify-new", lambda: _verify_new(
            cfg, env, new_file, before, after_strip, applied))
        report["after"] = report["verify_new"]["after"]
        if not cfg.swap:
            report["status"] = "completed without swap (new file kept for a manual swap)"
            return report

        # The rename happens while this connection still holds the exclusive lock; it is closed only after the
        # swapped file has been checked (or the swap has failed), by the ``finally`` below.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if env.before_swap:
            env.before_swap(cfg.db)
        report["swap"] = _step(report, env, "swap", lambda: swap._swap(cfg, env, new_file, before))
        report["sizes_after"] = measure._file_sizes(cfg.db)
        report["status"] = "completed"
        return report
    except Stop as stop:
        report["status"] = "stopped"
        report["stop"] = {"step": stop.step, "message": stop.message}
        if report["db_modified"]:
            report["stop"]["note"] = ("the database was already being reduced; the run is safe to repeat, and "
                                      "the verified backup is the way back to the original")
        return report
    except Exception as exc:  # noqa: BLE001 - whatever went wrong, the report must say what and where
        _record_failure(report, exc)
        return report
    except BaseException as exc:
        # an interrupt or an exit request: leave the same trail, then let it go on
        _record_failure(report, exc)
        report["finished"] = env.now().isoformat(timespec="seconds")
        try:
            _write_report(cfg, report, json.dumps(report, indent=1, default=str))
        except OSError as write_error:
            print(_not_written_line(report, write_error), file=sys.stderr)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if tmp_dir is not None:
            _remove_tmp(tmp_dir, cfg.work_dir)
            if saved_tmpdir is None:
                os.environ.pop("SQLITE_TMPDIR", None)
            else:
                os.environ["SQLITE_TMPDIR"] = saved_tmpdir
        if marker_path is not None and _is_finished(report):
            report["marker_removed"] = _remove_marker(marker_path, env)
        report["finished"] = env.now().isoformat(timespec="seconds")


def _measure(conn, cfg):
    snapshot = measure._snapshot(conn)
    if cfg.execute:
        # the stored values themselves, for the content check of the backup (a dry run has no backup to check)
        snapshot["body_fingerprint"] = measure._body_fingerprint(conn)
    return snapshot


def _record_failure(report, exc):
    report["status"] = "failed"
    report["failure"] = {"step": report.get("current_step"), "type": type(exc).__name__, "message": str(exc)}


def _remove_tmp(tmp_dir, work_dir):
    if tmp_dir.parent == work_dir and tmp_dir.name.startswith(_TMP_PREFIX):
        shutil.rmtree(tmp_dir, ignore_errors=True)
