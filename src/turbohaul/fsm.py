"""State machine transitions per v0.2 ARCHITECTURE.md §6.

Pure functions — no I/O, no side effects beyond mutating slot.state.
The TurbohaulManager (manager.py) owns the timing + subprocess + queue interactions
and calls these transition primitives.

Encodes the 10-state machine plus the new GRACE-BUSY, ACTIVE-MATCH, COLD-RECOVERY
transitions added in v0.2 per F2 must-fix.
"""
from turbohaul.slot import Slot, SlotState


# Legal transitions per v0.2 §6.
# Each entry: from_state → set of allowed to_states.
LEGAL_TRANSITIONS: dict[SlotState, set[SlotState]] = {
    SlotState.RECEIVED: {SlotState.ACCEPT_BUFFER, SlotState.STAGED},
    SlotState.ACCEPT_BUFFER: {SlotState.STAGED, SlotState.COLD},
    # STAGED → ACTIVE_MATCH added in v0.2.1-FSM-STAGED-AM:
    # a staged slot whose (thread_id, model_tag) matches the currently-active
    # warm slot during the GRACE window fast-tracks straight to ACTIVE_MATCH
    # (skipping LOADING) because the model is already in VRAM and the
    # _active_handle is reused. Without this entry, the worker_loop's
    # ACTIVE_MATCH promotion raises InvalidTransition and the slot vanishes
    # from queue.remove() while its DB state stays STAGED forever — the
    # an observed alternating-pattern bug under bursty load.
    SlotState.STAGED: {SlotState.LOADING, SlotState.COLD, SlotState.ACTIVE_MATCH},
    SlotState.LOADING: {SlotState.ACTIVE, SlotState.LOADING_FAIL, SlotState.COLD},  # +COLD per review
    # Retry from LOADING_FAIL → re-STAGED; on retry-exhaust → POPPED
    SlotState.LOADING_FAIL: {SlotState.STAGED, SlotState.POPPED},
    SlotState.ACTIVE: {SlotState.GRACE, SlotState.ACTIVE_MATCH},
    SlotState.GRACE: {SlotState.GRACE_BUSY, SlotState.POPPED, SlotState.ACTIVE},
    # GRACE-BUSY: matched follow-up running on warm slot; back to GRACE or pop
    SlotState.GRACE_BUSY: {SlotState.GRACE, SlotState.POPPED},
    # ACTIVE-MATCH: mid-stream matched thread queued at FIFO head; on completion → ACTIVE
    SlotState.ACTIVE_MATCH: {SlotState.ACTIVE},
    SlotState.POPPED: {SlotState.IDLE_HOT, SlotState.STAGED, SlotState.COLD},
    SlotState.IDLE_HOT: {SlotState.ACTIVE, SlotState.POPPED, SlotState.COLD},
    SlotState.COLD: set(),  # terminal
}


class InvalidTransition(ValueError):
    """Raised when a transition is not in the LEGAL_TRANSITIONS table."""


def transition(slot: Slot, to_state: SlotState) -> None:
    """Validate + apply a state transition. Mutates slot.state in place.

    Raises InvalidTransition if the transition is not legal per the FSM table.
    """
    legal = LEGAL_TRANSITIONS.get(slot.state, set())
    if to_state not in legal:
        raise InvalidTransition(
            f"slot {slot.slot_id}: illegal transition {slot.state.value} → {to_state.value}"
            f" (legal from {slot.state.value}: {sorted(s.value for s in legal)})"
        )
    slot.state = to_state


def can_transition(slot: Slot, to_state: SlotState) -> bool:
    """Predicate: is the transition allowed without raising?"""
    return to_state in LEGAL_TRANSITIONS.get(slot.state, set())


def legal_targets(from_state: SlotState) -> set[SlotState]:
    """Return the set of states that from_state can transition to."""
    return set(LEGAL_TRANSITIONS.get(from_state, set()))


def is_terminal(state: SlotState) -> bool:
    """True if the state has no outgoing transitions."""
    return len(LEGAL_TRANSITIONS.get(state, set())) == 0


def is_warm_state(state: SlotState) -> bool:
    """States where llama-server is loaded in VRAM."""
    return state in {
        SlotState.LOADING,
        SlotState.ACTIVE,
        SlotState.GRACE,
        SlotState.GRACE_BUSY,
        SlotState.ACTIVE_MATCH,
        SlotState.IDLE_HOT,
    }


def is_request_in_flight(state: SlotState) -> bool:
    """States where the request itself is actively being processed."""
    return state in {SlotState.ACTIVE, SlotState.GRACE_BUSY, SlotState.ACTIVE_MATCH}
