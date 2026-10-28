"""The two value types of kv_stale_pairs: a stale pair, and the result of a drop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StalePair:
    """One saved bin of the thread on a port it left, and the bytes its pair held."""

    name: str
    port: int
    slot_id: int
    pair_bytes: int


@dataclass(frozen=True)
class DropResult:
    """What a drop did.

    ``dropped`` holds the bins that were removed, sorted by name. ``skipped`` is None when the drop
    ran, otherwise one of ``landed_size_mismatch``, ``served_unknown``, ``dir_missing`` or
    ``scan_failed``, and nothing was removed. ``landed_bytes`` and ``n_written`` repeat the arguments
    the caller passed, so the caller can log them.
    """

    dropped: tuple[StalePair, ...]
    skipped: str | None
    landed_bytes: int | None
    n_written: int | None
