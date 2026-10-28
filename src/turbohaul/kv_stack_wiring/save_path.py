"""The manager-facing side of the slot KV save path: accounting with a log hook, the save-time cap gate,
the stale-pair drop after a save, and the GC's SOFT-tier fallback.

Each function takes the manager as ``host`` (typed by a small protocol of the members it uses) and
anything the manager looks up in its own module at call time as an argument. Nothing here keeps state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

from turbohaul.kv_ram_cap import dir_accounting, enforce_save_cap, soft_ceiling_order

# The manager's own logger name, so these lines log under the manager's logger and filters.
_log = logging.getLogger("turbohaul.manager")


class _SaveHost(Protocol):
    """The members of the manager the save path reads or calls. Nothing else is touched."""

    runtime: Any

    def _kvcache_dir_accounting(self) -> Any: ...

    def _gc_kv_cache(self, max_age_hours: float | None = None, max_files: int = 100) -> Any: ...

    def _live_turn_groups(self) -> Any: ...

    def _kvcache_scan_cache_invalidate(self) -> Any: ...


def dir_accounting_logged(save_dir: str) -> tuple[int, list[tuple]]:
    """Return ``(total_bytes, bins)`` for ``save_dir``, logging a failed scan at DEBUG.

    ``bins`` is a list of ``(fn, fpath, mtime, size)`` where ``size`` is the ``.bin`` plus its
    ``.bin.ckpt`` sidecar. Shared by the GC byte ceiling and the save-time cap gate, so both agree
    on how many bytes are on the RAM disk right now. Best-effort; never raises.
    """
    return dir_accounting(
        save_dir,
        on_error=lambda: _log.debug("slot KV dir accounting failed (best-effort)", exc_info=True),
    )


async def save_declined_by_cap(host: _SaveHost) -> bool:
    """True when the slot KV save must be declined because the RAM cap holds.

    Save-time HARD RAM cap. The GC only reclaims on its own background tick, so without this
    check repeated per-turn saves can drive the RAM disk past the configured cap. Before the engine
    save is posted, if the directory is already at or over the cap, the existing GC is driven
    (byte ceiling only: age and count disabled) to evict unprotected and idle-held bins, oldest
    first, to make room. If it still cannot get under the cap (a bin of a live turn holds the
    floor), the snapshot is declined: the live KV stays in the engine and only the on-disk copy
    is skipped. A bin of a live turn is never evicted, even an anchor. It reuses the GC's config field and the shared
    directory accounting. A declined save writes one warning.
    """
    _cap = int(host.runtime.kv.ram_cache_max_bytes)
    gate = await enforce_save_cap(
        _cap,
        lambda: host._kvcache_dir_accounting()[0],
        lambda: host._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000_000),
    )
    if not gate.proceed:
        _log.warning(
            "slot KV save SKIP (hard RAM cap): dir=%d cap=%d — "
            "HARD active/in-flight bins hold the floor; snapshot "
            "declined, live KV retained in-engine",
            gate.total_bytes, gate.cap_bytes,
        )
        return True
    return False


def drop_stale_pairs_after_save(host: _SaveHost, drop_fn: Callable[..., Any], save_dir: str,
                                model_tag: str, thread_hash: str, port: int, final_path: str,
                                n_written: Any, bin_fn: str) -> None:
    """One pair per thread: the new pair has landed, so the pairs this thread left on other ports go.

    ``drop_fn`` is the stale-pair drop the manager hands in. Best-effort and never fails the save:
    any exception is logged at DEBUG and swallowed. A drop that removed pairs clears the manager's
    scan cache once.
    """
    try:
        try:
            _landed = os.path.getsize(final_path)
        except OSError:
            _landed = None
        _dr = drop_fn(
            save_dir, model_tag, thread_hash, port,
            host._live_turn_groups(), _landed, n_written)
        if _dr.skipped == "landed_size_mismatch":
            _log.warning(
                "slot KV stale-port drop skipped for %s: landed_bytes=%s "
                "n_written=%r differ or are unproven -- old pair kept",
                bin_fn, _landed, n_written)
        elif _dr.skipped is not None:
            _log.debug("slot KV stale-port drop skipped (%s) for %s",
                      _dr.skipped, bin_fn)
        for _sp in _dr.dropped:
            _log.info("slot KV stale-port pair dropped: %s (port=%d, now on port=%d, %d bytes)",
                     _sp.name, _sp.port, port, _sp.pair_bytes)
        if _dr.dropped:
            host._kvcache_scan_cache_invalidate()
    except Exception:  # noqa: BLE001 - best-effort drop: a failure is logged and never fails the save
        _log.debug("slot KV stale-port drop failed (best-effort)", exc_info=True)


def evict_soft_floor(surviving, deleted_paths, hard_hashes, soft_hashes, pinned_paths, hash_by_path,
                     total: int, max_bytes: int, unlink_pair: Callable[[str], bool]) -> tuple[int, int]:
    """The GC's fallback when a HARD floor holds the directory over its cap.

    Walks the SOFT bins oldest first: idle-held and resident-idle thread bins and warm clean
    anchors (a live HARD thread's anchor is excluded and never evicted). These sit behind the same
    protection as live threads in the normal prune, so without this fallback the ceiling prune
    would give up at the cap while they held the floor. For each bin: stop once ``total`` is at or under
    ``max_bytes``; otherwise unlink it and, when that removed the bin, count it and take its size
    off ``total``. Returns ``(removed_count, total)``.
    """
    removed_count = 0
    soft_evictable = soft_ceiling_order(
        surviving, deleted_paths, hard_hashes, soft_hashes, pinned_paths, hash_by_path)
    for _sfn, _sfp, _smt, _ssz in soft_evictable:
        if total <= max_bytes:
            break
        if unlink_pair(_sfp):
            removed_count += 1
            total -= _ssz
    return removed_count, total
