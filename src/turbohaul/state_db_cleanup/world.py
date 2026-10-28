"""Everything the run reads from the outside world, so a test can replace one piece at a time."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _fsync_file(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _default_backup(src_conn, dest):
    dst = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst, pages=1024, sleep=0.0)
    finally:
        dst.close()
    _fsync_file(dest)


def _default_vacuum(conn, dest):
    conn.execute("VACUUM INTO ?", (str(dest),))
    _fsync_file(dest)


@dataclass
class Env:
    """The run's view of the outside world."""

    statvfs: object = os.statvfs
    dev_of: object = lambda p: os.stat(p).st_dev
    sleep: object = time.sleep
    now: object = lambda: datetime.now(UTC)
    sqlite_version: tuple = field(default_factory=lambda: sqlite3.sqlite_version_info)
    backup_fn: object = _default_backup
    vacuum_fn: object = _default_vacuum
    say: object = _say
    on_connect: object = None  # called with each connection the run opens (a test traces statements here)
    before_swap: object = None  # called just before the swap, with the exclusive lock still held
    geteuid: object = os.geteuid
