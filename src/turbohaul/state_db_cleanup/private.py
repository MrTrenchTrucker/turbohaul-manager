"""Files only their owner may read: the backup, the report and the unfinished-run marker.

A file created with the ambient umask can be readable by other users. These helpers create the file with
mode 0600 and then set the mode again on the open descriptor, so the umask cannot widen it.
"""

from __future__ import annotations

import os
import stat

from .config import Stop

_PRIVATE = 0o600


def _create_private(path):
    """Create an empty file that must not exist yet, readable and writable by its owner only."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, _PRIVATE)
    try:
        os.fchmod(fd, _PRIVATE)
    finally:
        os.close(fd)


def _create_private_text(path, text):
    """Create a file that must not exist yet, owner-only, holding ``text``; a half-written file is removed."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, _PRIVATE)
    try:
        os.fchmod(fd, _PRIVATE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(str(path))
        except OSError:
            pass
        raise


def _write_private(path, text):
    """Write ``text`` to ``path`` (replacing what is there), owner-only even if the file already existed."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _PRIVATE)
    try:
        os.fchmod(fd, _PRIVATE)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _require_private(path, step):
    """The octal mode of the file, e.g. "0600"; a mode with any group or other bit is a stop at ``step``."""
    mode = stat.S_IMODE(os.stat(str(path)).st_mode)
    if mode & 0o077:
        raise Stop(step, f"{path} has mode {mode:04o}; it must be readable by its owner only (0600), so the "
                         f"run stops before anything is written to the database")
    return f"{mode:04o}"
