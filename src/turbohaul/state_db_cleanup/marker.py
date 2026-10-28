"""The unfinished-run marker: a file beside the database that says a real run started and did not end cleanly."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Stop
from .private import _create_private_text


def _marker_path(db):
    db = Path(db)
    return db.parent / f"{db.name}.cleanup-unfinished"


def _refuse_if_present(cfg):
    """Stop before touching anything when an earlier real run left its marker behind."""
    path = _marker_path(cfg.db)
    if not (path.exists() or path.is_symlink()):
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        detail = (f"its report: {data.get('report_path')}; its planned backup: {data.get('backup_planned')}")
    except (OSError, ValueError) as exc:
        detail = f"the marker could not be read ({exc}), so its report and backup are unknown"
    raise Stop("preflight", f"an earlier real run did not finish: {path} exists ({detail}). Nothing was changed "
                            f"by this run. Read that report, keep that backup, and delete the marker only "
                            f"after you have looked; then run again.")


def _create_marker(cfg, env, report, stamp):
    """Create the marker (owner-only, never over an existing one) and return its path."""
    path = _marker_path(cfg.db)
    content = {"db": str(cfg.db), "report_path": report.get("report_path"),
               "backup_planned": report.get("backup_planned"), "stamp": stamp,
               "created": env.now().isoformat(timespec="seconds"), "pid": os.getpid()}
    try:
        _create_private_text(path, json.dumps(content, indent=1))
    except OSError as exc:
        raise Stop("marker", f"could not create the unfinished-run marker {path} ({exc}); nothing was "
                             f"written to the database or the backup directory") from exc
    return path


def _is_finished(report):
    """True when the run ended in a state that needs no looking at: done, or stopped before any write."""
    status = str(report.get("status", ""))
    return status.startswith("completed") or (status == "stopped" and not report.get("db_modified"))


def _remove_marker(path, env):
    """Delete the marker; True if it is gone, False (with one line on stderr) if it stays."""
    try:
        path.unlink()
    except OSError as exc:
        env.say(f"could not remove the unfinished-run marker {path} ({exc}); it stays, delete it by hand "
                f"after you have looked at the report")
        return False
    return True
