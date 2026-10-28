"""Checking the rebuilt file (the VACUUM INTO output) before it may replace the database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import measure, strip
from .checks import _verify_copy
from .config import _BODY_CHUNK, _BODY_COLUMN, _SAMPLE_SIZE, Stop
from .connect import _open_immutable

# Every SQL string here is built from constants or from table and column names read out of the database's own
# catalogue (and quoted); nothing typed by a user is ever put into a statement.
# ruff: noqa: S608


def _verify_new(cfg, env, new_file, before, after_strip, applied):
    check = _verify_copy(
        new_file, {**before, "sum_body_length": None, "body_fingerprint": None}, env, "new file")
    if check["ok"]:
        problems = []
        try:
            _check_new_values(cfg, env, new_file, after_strip, applied, check, problems)
        except sqlite3.Error as exc:
            problems.append(f"could not be read: {exc}")
        check["problems"] = problems
        check["ok"] = not problems
    if not check["ok"]:
        raise Stop("verify-new", "the rebuilt file did not verify; the database was not replaced: " +
                                 "; ".join(check["problems"]) + f" (left at {new_file})")
    return check

def _check_new_values(cfg, env, new_file, after_strip, applied, check, problems):
    conn = _open_immutable(new_file, env)
    try:
        if measure._body_fingerprint(conn) != after_strip["body"]:
            problems.append("stored values differ from the database just before the rebuild")
        where, params = strip._candidate_clause(cfg)
        left = {"stripped": 0, "unchanged": 0, "invalid": 0, "missing": 0}
        sample = []
        cur = conn.execute(f"SELECT slot_id, {_BODY_COLUMN} FROM slots WHERE {where} ORDER BY rowid", params)
        for slot_id, stored in measure._stream(cur, _BODY_CHUNK):
            outcome, _ = strip._classify(stored)
            left[outcome] += 1
            if outcome == "unchanged":
                sample.append((slot_id, sorted(json.loads(stored))))
        if left["stripped"]:
            problems.append(f"{left['stripped']} COLD value(s) still hold more than the identity keys")
        if left["invalid"] != applied["invalid"] or left["missing"] != applied["missing"]:
            problems.append("the count of untouched (invalid/NULL) values changed")
        step = max(1, len(sample) // _SAMPLE_SIZE)
        check["sample"] = [{"slot_id": s, "keys": k} for s, k in sample[::step][:_SAMPLE_SIZE]]
        check["cold_values"] = left
        check["after"] = {
            "row_counts": {t: conn.execute(f"SELECT count(*) FROM {measure._quote(t)}").fetchone()[0]
                           for t in measure._list_tables(conn)},
            "state_counts": measure._state_counts(conn),
            "sum_body_length": measure._sum_body_length(conn),
            "file_bytes": Path(new_file).stat().st_size,
        }
    finally:
        conn.close()
