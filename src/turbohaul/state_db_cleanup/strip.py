"""Choosing the COLD rows and reducing their stored value to the identity keys, a batch at a time."""

from __future__ import annotations

from turbohaul.slot_meta_projection import strip_stored_meta

from .config import _BODY_COLUMN, Stop
from .measure import _body_fingerprint, _quote, _state_counts, _table_fingerprints

# Every SQL string here is built from constants or from table and column names read out of the database's own
# catalogue (and quoted); nothing typed by a user is ever put into a statement.
# ruff: noqa: S608


def _candidate_clause(cfg):
    """SQL for the rows this run may touch: every COLD row, optionally only those older than N hours."""
    if cfg.cutoff_iso:
        return "state = 'COLD' AND updated_at < ?", (cfg.cutoff_iso,)
    return "state = 'COLD'", ()


def _classify(stored):
    """(outcome, new value). outcome is stripped / unchanged / invalid / missing."""
    if stored is None:
        return "missing", None
    new = strip_stored_meta(stored)
    if new is None:
        return "invalid", None
    if new == stored:
        return "unchanged", new
    return "stripped", new


def _chars(s):
    return len(s) if s is not None else 0


def _bytes(s):
    return len(s.encode("utf-8", "replace")) if s is not None else 0


def _strip_pass(conn, cfg, env, apply):
    """Walk the candidate rows in rowid order, a batch at a time, one stored value at a time.

    With ``apply`` False nothing is written: the counts are what an applying pass would produce.
    """
    where, params = _candidate_clause(cfg)
    counts = {"rows": 0, "stripped": 0, "unchanged": 0, "invalid": 0, "missing": 0,
              "chars_before": 0, "chars_after": 0, "bytes_removed": 0, "batches": 0, "checkpoints": 0}
    last = 0
    while True:
        rowids = [r[0] for r in conn.execute(
            f"SELECT rowid FROM slots WHERE rowid > ? AND {where} ORDER BY rowid LIMIT ?",
            (last, *params, cfg.batch))]
        if not rowids:
            break
        # A batch is one transaction. An exception ends the run, and closing the connection rolls the open
        # batch back, so a batch is committed whole or not at all.
        if apply:
            conn.execute("BEGIN IMMEDIATE")
        for rowid in rowids:
            row = conn.execute(f"SELECT {_BODY_COLUMN} FROM slots WHERE rowid = ?", (rowid,)).fetchone()
            stored = row[0] if row else None
            outcome, new = _classify(stored)
            counts["rows"] += 1
            counts[outcome] += 1
            counts["chars_before"] += _chars(stored)
            if outcome == "stripped":
                counts["chars_after"] += _chars(new)
                counts["bytes_removed"] += _bytes(stored) - _bytes(new)
                if apply:
                    conn.execute(f"UPDATE slots SET {_BODY_COLUMN} = ? WHERE rowid = ?", (new, rowid))
            else:
                counts["chars_after"] += _chars(stored)
        if apply:
            conn.execute("COMMIT")
        last = rowids[-1]
        counts["batches"] += 1
        if apply:
            env.sleep(cfg.sleep_ms / 1000.0)
        if counts["batches"] % cfg.checkpoint_every == 0:
            if apply:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                counts["checkpoints"] += 1
            verb = "reduced" if apply else "to reduce"
            env.say(f"  {'strip' if apply else 'plan'}: {counts['rows']} rows, {counts['stripped']} {verb}")
    return counts

def _check_strip(conn, before, plan, applied, non_cold_before):
    problems = []
    for key in ("rows", "stripped", "unchanged", "invalid", "missing"):
        if plan[key] != applied[key]:
            problems.append(f"{key}: planned {plan[key]}, done {applied[key]}")
    counts = {t: conn.execute(f"SELECT count(*) FROM {_quote(t)}").fetchone()[0] for t in before["row_counts"]}
    if counts != before["row_counts"]:
        problems.append("row counts changed")
    if _state_counts(conn) != before["state_counts"]:
        problems.append("counts by state changed")
    if _body_fingerprint(conn, "state != 'COLD'") != non_cold_before:
        problems.append("a row that is not COLD changed")
    if _table_fingerprints(conn) != before["fingerprints"]:
        problems.append("a column other than client_meta_json changed")
    if problems:
        raise Stop("check-strip", "; ".join(problems))
    return True
