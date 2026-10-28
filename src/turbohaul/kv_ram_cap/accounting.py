"""Byte accounting for the slot KV save directory.

The caller names the directory. This module only reads names, sizes and mtimes, and it
logs nothing: the caller passes ``on_error`` if it wants to record a failed scan.
"""

from __future__ import annotations

import os
from typing import Callable


def dir_accounting(save_dir: str, on_error: Callable[[], None] | None = None) -> tuple[int, list[tuple]]:
    """Return ``(total_bytes, bins)`` for the ``.bin`` files in ``save_dir``.

    ``bins`` is a list of ``(fn, fpath, mtime, size)``. ``size`` is the ``.bin`` plus its
    ``.bin.ckpt`` sidecar. The sidecar counts because the checkpoint ladder is about as large
    as the bin, and a ceiling that ignored it would let the directory grow past the cap
    unseen. A missing sidecar counts as zero bytes. A file that vanishes or cannot be
    stat'ed is skipped, so one bad entry does not hide the rest.

    Best-effort and never raises. On any failure ``on_error`` runs inside the except block
    (so a caller's logger can attach exc_info), and the totals gathered so far are returned.
    """
    total = 0
    bins: list[tuple] = []
    try:
        if not os.path.isdir(save_dir):
            return 0, bins
        for fn in os.listdir(save_dir):
            if not fn.endswith(".bin"):
                continue
            fpath = os.path.join(save_dir, fn)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            size = st.st_size
            try:
                size += os.stat(fpath + ".ckpt").st_size
            except OSError:
                pass
            bins.append((fn, fpath, st.st_mtime, size))
            total += size
    except Exception:
        if on_error is not None:
            on_error()
    return total, bins
