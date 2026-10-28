"""The command line: parse the options, run, print the report, set the exit status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import _DEFAULT_BATCH, _DEFAULT_CHECKPOINT_EVERY, _DEFAULT_MARGIN_BYTES, _DEFAULT_SLEEP_MS, _GIB, Config
from .report import _not_written_line, _write_report
from .sequence import run


def parse_args(argv):
    p = argparse.ArgumentParser(description="Reduce old request bodies in state.sqlite, then rebuild the file.")
    p.add_argument("--db", required=True, type=Path, help="path to state.sqlite")
    p.add_argument("--backup-dir", required=True, type=Path,
                   help="existing directory (not the database's own) that receives the verified backup")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="directory for the rebuilt file; must be on the database's filesystem (default: its directory)")
    p.add_argument("--execute", action="store_true", help="do the work; without it nothing is written")
    p.add_argument("--no-swap", action="store_true", help="stop after the rebuilt file verifies; leave the swap to you")
    p.add_argument("--min-age-hours", type=float, default=0.0,
                   help="only rows not updated for N hours (default 0: all COLD)")
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    p.add_argument("--sleep-ms", type=float, default=_DEFAULT_SLEEP_MS)
    p.add_argument("--checkpoint-every", type=int, default=_DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--margin-gib", type=float, default=_DEFAULT_MARGIN_BYTES / _GIB)
    p.add_argument("--report", type=Path, default=None,
                   help="where to write the JSON report (execute default: backup dir)")
    return p.parse_args(argv)


def main(argv=None, env=None):
    args = parse_args(argv)
    cfg = Config(db=args.db, backup_dir=args.backup_dir, work_dir=args.work_dir, execute=args.execute,
                 swap=not args.no_swap, min_age_hours=args.min_age_hours, batch=args.batch,
                 sleep_ms=args.sleep_ms, checkpoint_every=args.checkpoint_every,
                 margin_bytes=int(args.margin_gib * _GIB), report_path=args.report)
    try:
        report = run(cfg, env)
    except Exception as exc:  # noqa: BLE001 - whatever went wrong, still print a report and exit 1
        print(json.dumps({"status": "error", "error": repr(exc)}, indent=1))
        return 1
    text = json.dumps(report, indent=1, default=str)
    print(text)
    try:
        _write_report(cfg, report, text)
    except OSError as exc:
        # the report is already on stdout; say where the backup and the marker are, and do not crash
        print(_not_written_line(report, exc), file=sys.stderr)
    return {"stopped": 2, "failed": 1}.get(report["status"], 0)
