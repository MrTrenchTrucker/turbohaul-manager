"""co_residence.recheck: taking the slot out (phase B) and reserving it directly (phase C)."""

import asyncio
import logging

import pytest
from _co_residence_fakes import FakeHost, FakeResident, FakeSlot

from turbohaul.co_residence import run_recheck

NOW = 1000.0
LOGGER = "turbohaul.co_residence"


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def staged_host(cls=FakeHost) -> FakeHost:
    host = cls(residents=[FakeResident()], staged=[FakeSlot(created_at=NOW - 30.0)], now=NOW)
    host.cap, host.cap_raises = 1, True
    host.pick_result = (2, "none")
    return host


def inbox_host(cls=FakeHost):
    """One ACTIVE engine with three requests in its inbox; the middle one is the oldest."""
    first, oldest, last = (FakeSlot(slot_id=n, created_at=NOW - age) for n, age in
                           (("first", 20.0), ("oldest", 90.0), ("last", 10.0)))
    r = FakeResident().put(first, oldest, last)
    host = cls(residents=[r], now=NOW)
    host.cap, host.cap_raises = 1, True
    return host, r, (first, oldest, last)


@pytest.fixture(autouse=True)
def _info(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)


# -- phase B -------------------------------------------------------------------------------

async def test_a_staged_slot_is_taken_from_the_queue_and_reserved_with_the_fit_check_card():
    host = staged_host()
    slot = host.staged[0]
    await run_recheck(host)
    assert host.staged == [], why("the slot must leave the staging queue")
    assert host.args.get("_reserve_and_start_locked", []) == [(slot, 2, "none")], why("reserve gets slot, card, split")
    assert "_unpark_fastlane_claim" not in host.calls, why("only an inbox slot has a Fast Lane claim to unpark")


async def test_an_inbox_slot_is_taken_with_the_others_kept_in_order():
    host, r, (first, oldest, last) = inbox_host()
    await run_recheck(host)
    assert list(r.inbox._queue) == [first, last], why("the others stay in the inbox, in order")
    assert host.args.get("_unpark_fastlane_claim", []) == [(oldest,)], why("the taken slot's claim is released")
    assert host.args.get("_reserve_and_start_locked", [[None]])[0][0] is oldest and "queue.remove" not in host.calls


async def test_a_slot_someone_else_took_first_ends_the_attempt():
    host, r, (first, oldest, last) = inbox_host()

    class Raced(type(host)):
        def _vram_admits_locked(self, *a):
            r.inbox._queue.remove(oldest)  # the engine got it while the probe ran
            return super()._vram_admits_locked(*a)

    host.__class__ = Raced
    await run_recheck(host)
    assert "_reserve_and_start_locked" not in host.calls and "_unpark_fastlane_claim" not in host.calls
    assert "_handle_unloaded_slot" not in host.calls and len(host.state.norom) == 0, why("a lost race is not a verdict")


async def test_a_staged_slot_the_dispatcher_took_first_ends_the_attempt():
    class Raced(FakeHost):
        def _vram_admits_locked(self, *a):
            self.staged.clear()
            return super()._vram_admits_locked(*a)

    host = staged_host(Raced)
    await run_recheck(host)
    assert "_reserve_and_start_locked" not in host.calls and len(host.state.norom) == 0


async def test_an_evicted_slot_goes_to_the_unloaded_handler_and_is_not_reserved():
    host = staged_host()
    slot = host.staged[0]
    original = host.queue.remove

    async def remove(slot_id):
        taken = await original(slot_id)
        taken.is_evicted = True
        return taken

    host.queue.remove = remove
    await run_recheck(host)
    assert host.args.get("_handle_unloaded_slot", []) == [(slot,)], why("an evicted slot goes to the handler")
    assert "_reserve_and_start_locked" not in host.calls and len(host.state.norom) == 0


async def test_an_evicted_inbox_slot_still_has_its_claim_released_first():
    host, r, (first, oldest, last) = inbox_host()

    class Evicting(type(host)):
        def _unpark_fastlane_claim(self, slot):
            super()._unpark_fastlane_claim(slot)
            slot.is_evicted = True

    host.__class__ = Evicting
    await run_recheck(host)
    order = [c for c in host.calls if c in {"_unpark_fastlane_claim", "_handle_unloaded_slot"}]
    assert order == ["_unpark_fastlane_claim", "_handle_unloaded_slot"], why(f"order was {order!r}")
    assert "_reserve_and_start_locked" not in host.calls


# -- phase C -------------------------------------------------------------------------------

async def test_phase_c_reserves_directly_and_never_routes():
    host = staged_host()
    await run_recheck(host)
    assert not hasattr(host, "_route_or_reserve")
    assert [c for c in host.calls if "route" in c and c != "_route_cap_and_placement"] == [], why(
        "the re-check must not call any routing method")
    assert host.held_at.get("_reserve_and_start_locked", []) == [True], why("the reserve runs under the registry lock")
    assert host.held_at.get("_remember_affinity", []) == [True]


async def test_affinity_is_remembered_for_a_successful_reserve():
    host = staged_host()
    host.staged[0].thread_id = "thr"
    await run_recheck(host)
    assert host.args.get("_remember_affinity", []) == [("m", "tid:thr", "new-key")], why("tag, session key, engine key")


async def test_success_clears_a_no_room_note_made_meanwhile():
    class Noted(FakeHost):
        async def _reserve_and_start_locked(self, *a, **k):
            self.state.norom.note("m", NOW)  # another path noted the tag during the reserve
            return await super()._reserve_and_start_locked(*a, **k)

    host = staged_host(Noted)
    await run_recheck(host)
    assert not host.state.norom.blocked("m", NOW + 1.0), why("an admitted engine clears the tag's no-room memo")


async def test_a_reserve_that_places_nothing_notes_the_memo_and_remembers_no_affinity():
    host = staged_host()
    host.reserve_result = None  # the VRAM gate in the reserve requeued the slot
    await run_recheck(host)
    assert "_remember_affinity" not in host.calls, why("nothing was placed, so no affinity")
    assert host.state.norom.blocked("m", NOW + 59.0), why("a lost race remembers the tag for the memo window")


async def test_a_failing_reserve_fails_the_request_future_and_never_escapes(caplog):
    host = staged_host()
    host.reserve_exc = RuntimeError("spawn exploded")
    slot = host.staged[0]
    await run_recheck(host)
    calls = host.args.get("_fail_completion_future", [])
    assert len(calls) == 1, why("exactly one request must be failed")
    call = calls[0]
    assert call[0] is slot and isinstance(call[1], RuntimeError) and str(call[1]) == "dispatch routing failed"
    assert host.held is False, why("the registry lock must be released after the failure")
    assert host.state.norom.blocked("m", NOW + 1.0), why("a failed attempt remembers the tag")
    assert "_remember_affinity" not in host.calls
    assert any(r.levelno == logging.ERROR for r in caplog.records), why("the failure must be logged with a traceback")


async def test_a_cancellation_during_the_reserve_propagates_without_failing_the_request():
    host = staged_host()
    host.reserve_exc = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run_recheck(host)
    assert "_fail_completion_future" not in host.calls, why("a cancellation is not a routing failure")
    assert host.held is False


# -- the in-flight reservation marker ------------------------------------------------------------

def _marker_span(host):
    assert "reserving_enter" in host.calls and "reserving_exit" in host.calls, why(
        f"the reserve must run under the reservation marker: {host.calls!r}")
    return host.calls.index("reserving_enter"), host.calls.index("reserving_exit")


async def test_the_reserve_runs_strictly_inside_the_reservation_marker():
    host = staged_host()
    await run_recheck(host)
    enter, leave = _marker_span(host)
    assert host.calls[enter + 1:leave] == [
        "lock_enter", "_reserve_and_start_locked", "_session_key_for", "_remember_affinity", "lock_exit",
    ], why(f"the marker must wrap exactly the locked reserve, not phase A or B: {host.calls!r}")
    assert host.args.get("reserving_enter") == [("m",)], why("the marker names the tag")
    assert host.held_at.get("reserving_enter") == [False], why("the marker is set before the registry lock is taken")
    assert host.calls.count("reserving_enter") == 1 and host.calls.count("reserving_exit") == 1


@pytest.mark.parametrize("exc", [RuntimeError("spawn exploded"), asyncio.CancelledError()])
async def test_the_marker_is_cleared_when_the_reserve_raises(exc):
    host = staged_host()
    host.reserve_exc = exc
    try:
        await run_recheck(host)
    except asyncio.CancelledError:
        pass
    assert host.calls.count("reserving_enter") == 1 and host.calls.count("reserving_exit") == 1, why(
        f"the marker must be cleared on every exit: {host.calls!r}")
    enter, leave = _marker_span(host)
    assert enter < host.calls.index("_reserve_and_start_locked") < leave


async def test_the_marker_is_not_entered_when_the_verdict_declines():
    host = staged_host()
    host.vram_ok = False
    await run_recheck(host)
    assert "reserving_enter" not in host.calls, why("a no-room probe must not flicker the status display")


async def test_the_marker_is_not_entered_when_the_slot_was_not_taken():
    class Raced(FakeHost):
        def _vram_admits_locked(self, *a):
            self.staged.clear()
            return super()._vram_admits_locked(*a)

    host = staged_host(Raced)
    await run_recheck(host)
    assert "reserving_enter" not in host.calls and "_reserve_and_start_locked" not in host.calls


async def test_the_marker_is_not_entered_for_an_evicted_slot():
    host = staged_host()
    original = host.queue.remove

    async def remove(slot_id):
        taken = await original(slot_id)
        taken.is_evicted = True
        return taken

    host.queue.remove = remove
    await run_recheck(host)
    assert "_handle_unloaded_slot" in host.calls and "reserving_enter" not in host.calls


# -- scenario: fit decides, not a count -------------------------------------

class Loading(FakeHost):
    """A box where a further engine fits only while fewer than ``room`` engines are loaded."""

    room = 3

    def _vram_admits_locked(self, need, parallel, main_gpu, split_mode):
        super()._vram_admits_locked(need, parallel, main_gpu, split_mode)
        return len(self.residents) < self.room

    async def _reserve_and_start_locked(self, slot, main_gpu_override=None, split_mode_override=None):
        await super()._reserve_and_start_locked(slot, main_gpu_override, split_mode_override)
        new = FakeResident(model_tag=slot.model_tag, resident_key=f"m#{len(self.residents)}")
        self.residents.append(new)
        return new


async def test_three_engines_that_fit_all_load_and_the_fourth_stays_out_with_vram_refused(caplog):
    host = Loading(residents=[FakeResident()], now=NOW)
    host.cap, host.cap_raises = 1, True  # a budget of one engine, and it must not matter
    counts = []
    for i in range(5):
        host.now = NOW + 10.0 * i
        host.staged.append(FakeSlot(slot_id=f"w{i}", created_at=host.now - 30.0))
        await run_recheck(host)
        counts.append(len(host.residents))
    assert counts[:2] == [2, 3], why(f"engines after each pass: {counts!r}")
    assert max(counts) == 3, why("the fourth engine does not fit and must stay out")
    declines = [r.getMessage() for r in caplog.records if "CO_RESIDENCE_DECLINED" in r.getMessage()]
    assert declines and all("reason=vram_refused" in d and "via=recheck" in d for d in declines), why(
        f"declines were {declines!r}")
    assert not any(("reason=box_full" in d or "reason=cap" in d) for d in declines)
