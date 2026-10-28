"""The manager-facing half of the keep-warm decision: measure the live inputs, ask resident_exit, log a hold.

The caller holds the registry lock. Nothing here awaits or keeps state; the manager is
read through a small Protocol view of the members this function uses.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Protocol

from turbohaul.resident_exit import RetentionInputs, decide_retention

# The logger name is the manager's, so existing log handlers and filters keep matching.
_log = logging.getLogger("turbohaul.manager")


class _HoldHost(Protocol):
    """The members of the manager that the hold measurement reads."""

    runtime: Any

    def _model_residents(self) -> Iterable[Any]: ...

    def _is_designated_unload_target_locked(self, r: Any) -> bool: ...


def hold_warm_window(
    host: _HoldHost, r: Any, idle_window: float, hold_window_s: float, *, active_state: Any,
) -> float | None:
    """Keep a finishing co-resident warm for a bounded window.

    CALLER HOLDS the registry lock. Sync, no await. Measures the live inputs and
    lets resident_exit.decide_retention answer. The identity class of the finished
    request is not an input: it only decides whether the KV is saved, not whether
    the resident lives. Returns the window to park with, or None to evict now.
    ``hold_window_s`` is the hold the caller offers and ``active_state`` is the
    caller's ACTIVE resident state.
    """
    live_active_siblings = sum(
        1 for x in host._model_residents()
        if x is not r and x.model_tag == r.model_tag and x.state is active_state
    )
    designated_now = host._is_designated_unload_target_locked(r)
    claims = getattr(host, "_fastlane_claims", None) or {}
    pending = getattr(host, "_pending_reclaim_mib", None) or {}
    capacity_demand = any(
        getattr(c.get("slot"), "model_tag", None) not in (None, r.model_tag)
        for c in list(claims.values())
    ) or pending.get(r.main_gpu, 0) > 0
    decision = decide_retention(RetentionInputs(
        idle_window_s=idle_window,
        hold_window_s=hold_window_s,
        max_parallel_sidecars=host.runtime.queue.max_parallel_sidecars,
        live_active_siblings=live_active_siblings,
        designated_victim=designated_now,
        # Only called when no parked claim outranks r (the caller's condition); the module still checks the input.
        outranking_claim=False,
        capacity_demand_pending=capacity_demand,
    ))
    if decision.hold:
        _log.info(
            "resident hold-warm: model_tag=%s resident_key=%s window=%.0fs "
            "active_siblings=%d reason=%s (keep_alive<=0 on a co-resident; bounded, "
            "make-room can still evict it)",
            r.model_tag, r.resident_key, decision.window_s,
            live_active_siblings, decision.reason,
        )
        return decision.window_s
    return None
