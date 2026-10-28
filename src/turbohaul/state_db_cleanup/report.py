"""One step of the run: announce it, time it, and record it in the report; and where the report file goes."""

from __future__ import annotations

import time
from pathlib import Path

from .private import _write_private


def _step(report, env, name, fn):
    t0 = time.monotonic()
    env.say(f"[{name}]")
    report["current_step"] = name
    try:
        result = fn()
    except BaseException:
        # a Stop, any other error, or an interrupt: the step did not finish, and the report says so
        report["steps"].append({"step": name, "ok": False, "seconds": round(time.monotonic() - t0, 2)})
        raise
    report["steps"].append({"step": name, "ok": True, "seconds": round(time.monotonic() - t0, 2)})
    return result


def _report_path(cfg, report):
    """Where this run's report file goes: --report if given, else beside the backup for a real run, else nowhere."""
    if cfg.report_path is not None:
        return Path(cfg.report_path)
    if cfg.execute:
        return cfg.backup_dir / f"{cfg.db.name}.cleanup-report-{report['stamp']}.json"
    return None


def _write_report(cfg, report, text):
    """Write the report file (owner-only) to the path the run planned at its start; returns the path or None."""
    planned = report.get("report_path")
    path = Path(planned) if planned else _report_path(cfg, report)
    if path is None:
        return None
    _write_private(path, text)
    return path


def _not_written_line(report, error):
    """The one plain line for stderr when the report file could not be written: where the backup and marker are."""
    backup = (report.get("backup") or {}).get("path") or report.get("backup_planned") or "none"
    marker = None if report.get("marker_removed") else report.get("marker")
    return f"REPORT NOT WRITTEN ({error}); backup: {backup}; unfinished-run marker: {marker or 'none'}"
