"""Reading the live manager for the KV save directory: which threads a live turn or an idle slot owns.

Two scans, both best-effort and both read-only. ``live_kv_protection`` sorts the live threads into the
HARD and SOFT tiers the GC needs. ``live_turn_groups`` lists, per engine port, the thread hashes of the
turns that port serves right now, which is what the stale-pair drop spares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from turbohaul.kv_ram_cap import split_protection

# The manager's own logger name, so these lines log under the manager's logger and filters.
_log = logging.getLogger("turbohaul.manager")


class _LiveStateHost(Protocol):
    """The members of the manager these scans read. Nothing else is touched."""

    _active_slot: Any
    _inflight: Any
    _active_handle: Any
    _idle_thread_id: Any
    _idle_client_meta: Any
    _residents: Any

    def _model_residents(self) -> Any: ...

    def _thread_hash(self, thread_id: str) -> str: ...


def live_turn_groups(host: _LiveStateHost, bin_identity: Callable[..., str]):
    """Which engine ports a live turn serves, as the stale-pair drop wants them.

    Returns ``[(port, [hash_set, ...]), ...]``, one entry per resident (or one for the legacy
    singleton when there are no residents). Each hash set holds the thread hashes of one live
    turn (the active slot or an in-flight rider): the hash of its thread id and, when it
    differs, the hash of its bin identity. Empty sets are left out. Returns None when the
    scan fails, so the caller knows the served ports are unknown.
    """
    def _hashes(sl) -> frozenset:
        tid = getattr(sl, "thread_id", None)
        ident = bin_identity(tid or "", getattr(sl, "client_meta", None),
                             getattr(sl, "admission_hash_chain", None))
        out = set()
        if tid:
            out.add(host._thread_hash(tid))
        if ident and ident != (tid or ""):
            out.add(host._thread_hash(ident))
        return frozenset(out)

    def _groups(live) -> list:
        return [g for g in (_hashes(sl) for sl in live if sl is not None) if g]

    try:
        residents = host._model_residents()
        if residents:
            return [(r.port, _groups([r.active_slot, *list(r.inflight)])) for r in residents]
        return [(getattr(host._active_handle, "port", None),
                 _groups([host._active_slot, *list(host._inflight)]))]
    except Exception:  # noqa: BLE001 - best-effort scan: any failure is logged and swallowed
        _log.debug("live turn scan failed (served ports unknown)", exc_info=True)
        return None


def live_kv_protection(host: _LiveStateHost, bin_identity: Callable[..., str]) -> tuple[set[str], set[str]]:
    """Split live KV-protection into (HARD, SOFT) thread_hash sets.

    HARD = a live turn's bins: the active slot + every in-flight teardown
    thread. The GC NEVER evicts these, on any axis (age, count, or ceiling).

    SOFT = idle-held (`host._idle_thread_id`) + every resident's idle thread.
    SOFT bins are reclaimable under pressure: the GC age-prunes a SOFT bin past
    its cutoff and falls back to SOFT oldest-first before it gives up over cap.

    Resolves cap-awarely. At cap>=2, the manager's own _active_slot / _inflight
    are permanently unwritten, so the live per-resident active_slot / inflight
    are scanned instead. Best-effort: never raises; an empty/unknown
    thread_id is skipped (``nothread`` is never protected, so it cannot
    silently defeat the ceiling).
    """
    hard_threads: list = []
    soft_threads: list = []
    try:
        # Resolve cap-awarely. At cap>=2, the manager's _active_slot is
        # permanently unwritten: scan the per-resident active_slot / inflight
        # that the resident drivers maintain. HARD tier.
        residents = host._model_residents()
        if residents:
            for r in residents:
                active = r.active_slot
                hard_threads.append((
                    getattr(active, "thread_id", None) if active is not None else None,
                    getattr(active, "client_meta", None) if active is not None else None,
                    getattr(active, "admission_hash_chain", None) if active is not None else None,
                ))
                for s in list(r.inflight):
                    hard_threads.append((getattr(s, "thread_id", None), getattr(s, "client_meta", None),
                                         getattr(s, "admission_hash_chain", None)))
        else:
            # cap<=1: legacy singleton
            active = host._active_slot
            hard_threads.append((
                getattr(active, "thread_id", None) if active is not None else None,
                getattr(active, "client_meta", None) if active is not None else None,
                getattr(active, "admission_hash_chain", None) if active is not None else None,
            ))
            for s in list(host._inflight):
                hard_threads.append((getattr(s, "thread_id", None), getattr(s, "client_meta", None),
                                     getattr(s, "admission_hash_chain", None)))
        # SOFT tier: the idle-held thread + every resident's idle thread.
        soft_threads.append((host._idle_thread_id, host._idle_client_meta, None))
        for r in list(host._residents.values()):
            soft_threads.append((getattr(r, "idle_thread_id", None), getattr(r, "idle_client_meta", None), None))
    except Exception:  # noqa: BLE001 - best-effort scan: any failure is logged and swallowed
        _log.debug("live KV protection-class scan failed (best-effort)", exc_info=True)
    # Protect BOTH the (session,role) identity hash and
    # the raw thread_id hash (legacy bins during the one-time migration).
    # The split runs after the gather, even when the gather raised part-way,
    # so the protection gathered so far is kept.
    split = split_protection(
        hard_threads, soft_threads, host._thread_hash, bin_identity,
        on_error=lambda: _log.debug("live KV protection-class scan failed (best-effort)", exc_info=True),
    )
    return split.hard, split.soft
