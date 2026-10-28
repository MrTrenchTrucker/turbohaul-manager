"""The two functions of the cold slot metadata retention pass. See the module's AGENTS.md."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

# The manager's own logger name, so the pass logs under the same logger and filters as the manager.
_log = logging.getLogger("turbohaul.manager")


class _SweepHost(Protocol):
    """What ``maybe_sweep`` needs from the manager: the setting, the timer, the counters and the run."""

    runtime: Any
    _COLD_META_IDLE_AFTER_PASS_S: int
    _cold_meta_next_pass_monotonic: float
    _cold_meta_stripped_lifetime: int
    _cold_meta_invalid_lifetime: int

    def _cold_meta_sweep_once(self, age_hours: float) -> dict: ...


def run_once(
    session: Callable[[Any], AbstractContextManager[Any]],
    strip_one: Callable[[Any, str], str],
    db_path: Any,
    watermark: int,
    age_hours: float,
    batch: int,
) -> tuple[dict, int]:
    """One bounded run of the retention pass over ended slots. Synchronous; call it off the event loop.

    Takes up to ``batch`` COLD rows older than ``age_hours``, in rowid order after ``watermark``, and
    reduces each one's stored metadata to its identity keys with ``strip_one``. Rows are picked from cheap
    columns only, so no stored body is read to choose them, and bodies are read one at a time. Returns
    the counts and the watermark for the next run. The caller keeps the watermark, and only after this
    returns, so a failed run leaves it where it was.
    """
    cutoff_iso = (
        datetime.now(UTC) - timedelta(hours=age_hours)
    ).isoformat(timespec="seconds")
    counts = {"stripped": 0, "unchanged": 0, "invalid": 0, "missing": 0, "error": 0}
    last_rowid = watermark
    with session(db_path) as conn:
        rows = conn.execute(
            """SELECT rowid, slot_id FROM slots
                   WHERE rowid > ? AND state = 'COLD' AND updated_at < ?
                   ORDER BY rowid LIMIT ?""",
            (watermark, cutoff_iso, batch),
        ).fetchall()
        for rowid, slot_id in rows:
            counts[strip_one(conn, slot_id)] += 1
            last_rowid = rowid
    pass_done = len(rows) < batch
    new_watermark = 0 if pass_done else last_rowid
    counts["rows"] = len(rows)
    counts["pass_done"] = pass_done
    return counts, new_watermark


async def maybe_sweep(host: _SweepHost) -> dict | None:
    """Run one retention run if the setting is on and the pass is due. Returns its counts, or None.

    ``kv.cold_meta_purge_age_hours`` of 0 (the default) turns the pass off. It is read fresh, so a runtime
    change applies on the next tick. The database work runs in a worker thread.
    """
    age_hours = float(host.runtime.kv.cold_meta_purge_age_hours)
    if age_hours <= 0:
        return None
    if time.monotonic() < host._cold_meta_next_pass_monotonic:
        return None
    counts = await asyncio.to_thread(host._cold_meta_sweep_once, age_hours)
    if counts["pass_done"]:
        host._cold_meta_next_pass_monotonic = time.monotonic() + host._COLD_META_IDLE_AFTER_PASS_S
    host._cold_meta_stripped_lifetime += counts["stripped"]
    host._cold_meta_invalid_lifetime += counts["invalid"]
    if counts["stripped"] or counts["invalid"] or counts["error"]:
        _log.log(
            logging.WARNING if (counts["invalid"] or counts["error"]) else logging.INFO,
            "cold slot metadata sweep: rows=%d stripped=%d unchanged=%d invalid=%d error=%d "
            "(age_hours=%s, pass_done=%s)",
            counts["rows"], counts["stripped"], counts["unchanged"], counts["invalid"],
            counts["error"], age_hours, counts["pass_done"],
        )
    return counts
