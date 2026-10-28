"""Fit-based periodic re-check for a model tag that already runs an engine.

A request that arrives while the tag's engine is busy is declined once by the arrival path
and parked behind that engine. Nothing asks again whether a further engine would now fit,
so the card stays empty. This module asks again, on a bounded schedule, and starts the
engine when the placement and the cross-resident VRAM gate both pass. The box-wide engine
budget does not limit that re-check; the card and VRAM fit is the only ceiling.

* ``run_recheck`` is the dispatch-loop entry point: gate, scan, probe, take the request,
  reserve the engine directly.
* ``fit_verdict`` (re-check) and ``arrival_verdict`` (arrival path) answer "may this tag
  start a further engine, and on which card?" and write the CO_RESIDENCE_* decline line.
* ``recheck_window``, ``demand_is_eligible``, ``decide_recheck`` and ``NoRoomMemo`` hold the
  rules of the schedule; ``RecheckState`` holds what it remembers between passes.
* ``gather_candidates``, ``pick_candidate``, ``inbox_slots`` and ``take_from_inbox`` find
  and take the waiting request.
* ``CoResidenceHost`` is the structural type of the one object the module is handed: the
  manager. The module imports only ``turbohaul.queue_liveness`` (the liveness predicate) and
  takes the host for everything else.
"""

from __future__ import annotations

from .constants import (
    DEMAND_MIN_AGE_S,
    MEMO_MAX_ENTRIES,
    NO_ROOM_MEMO_S,
    PROTECTED_DEMAND_MIN_AGE_S,
    RECHECK_MIN_INTERVAL_S,
)
from .decision import RecheckDecision, RecheckInputs, decide_recheck
from .host import CoResidenceHost
from .lines import DeclineReason, format_recheck_action, format_step2_decline
from .memo import NoRoomMemo
from .recheck import run_recheck
from .scan import Candidate, gather_candidates, inbox_slots, pick_candidate, take_from_inbox
from .state import RecheckState
from .verdict import Verdict, arrival_verdict, fit_verdict
from .window import RecheckGate, demand_is_eligible, recheck_window

__all__ = [
    "RECHECK_MIN_INTERVAL_S",
    "NO_ROOM_MEMO_S",
    "DEMAND_MIN_AGE_S",
    "PROTECTED_DEMAND_MIN_AGE_S",
    "MEMO_MAX_ENTRIES",
    "RecheckGate",
    "recheck_window",
    "demand_is_eligible",
    "RecheckInputs",
    "RecheckDecision",
    "decide_recheck",
    "NoRoomMemo",
    "DeclineReason",
    "format_step2_decline",
    "format_recheck_action",
    "RecheckState",
    "Candidate",
    "inbox_slots",
    "take_from_inbox",
    "gather_candidates",
    "pick_candidate",
    "Verdict",
    "fit_verdict",
    "arrival_verdict",
    "run_recheck",
    "CoResidenceHost",
]
