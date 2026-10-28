"""co_residence.recheck: the gate, the scan, the probe (phase A) and the memo, on a recording host."""

import asyncio
import logging

import pytest
from _co_residence_fakes import FakeHost, FakeResident, FakeSlot

from turbohaul.co_residence import NO_ROOM_MEMO_S, RECHECK_MIN_INTERVAL_S, run_recheck

NOW = 1000.0
LOGGER = "turbohaul.co_residence"


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def staged_host(**kw) -> FakeHost:
    """One ACTIVE engine of tag m, and one request that has waited 30 s in the staging queue."""
    host = FakeHost(residents=[FakeResident()], staged=[FakeSlot(created_at=NOW - 30.0)], now=NOW, **kw)
    host.cap, host.cap_raises = 1, True
    return host


def probes(host: FakeHost) -> int:
    return host.calls.count("_vram_admits_locked")


@pytest.fixture(autouse=True)
def _info(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)


# -- the gate ---------------------------------------------------------------------

async def test_a_closed_gate_touches_nothing_but_the_state_and_the_clock():
    host = staged_host()
    host.state.last_recheck_at = NOW - 5.0
    await run_recheck(host)
    assert host.calls == ["_co_residence_state", "_co_residence_now"], why(f"closed gate used {host.calls!r}")
    assert host.state.last_recheck_at == NOW - 5.0, why("a closed gate must not move the attempt time")


async def test_the_gate_opens_after_exactly_the_interval():
    host = staged_host()
    host.state.last_recheck_at = NOW - RECHECK_MIN_INTERVAL_S
    await run_recheck(host)
    assert "_grace_active_exclusions" in host.calls, why("10.0 s since the last attempt opens the gate")


async def test_every_attempt_is_counted_even_when_the_scan_finds_nothing():
    host = FakeHost(residents=[FakeResident()], now=NOW)
    await run_recheck(host)
    assert host.state.last_recheck_at == NOW, why("an empty scan is an attempt too")
    snaps = host.calls.count("queue.staging_snapshot")
    host.now = NOW + 5.0
    await run_recheck(host)
    assert host.calls.count("queue.staging_snapshot") == snaps, why("5 s later the scan must be debounced")
    host.now = NOW + 10.0
    await run_recheck(host)
    assert host.calls.count("queue.staging_snapshot") == snaps + 1 and host.state.last_recheck_at == NOW + 10.0


async def test_no_candidate_means_no_lock_and_no_probe():
    host = FakeHost(residents=[FakeResident()], staged=[FakeSlot(created_at=NOW - 2.0)], now=NOW)
    await run_recheck(host)
    assert "lock_enter" not in host.calls and probes(host) == 0, why("2 s old demand is not starved demand")


async def test_the_box_budget_never_gates_the_recheck():
    host = staged_host()  # budget 1 and the budget attribute raises if it is read
    await run_recheck(host)
    assert "_reserve_and_start_locked" in host.calls, why("a full budget must not stop a fitting engine")


# -- phase A: the probe under the registry lock, no slot taken ----------------------------

async def test_the_probe_runs_under_the_lock_and_the_grace_query_outside_it():
    host = staged_host()
    await run_recheck(host)
    for name in ("_route_cap_and_placement", "_resolve_placement_locked", "_pick_card_for_new_instance",
                 "_vram_admits_locked"):
        assert host.held_at.get(name, []) == [True], why(f"{name} must run with the registry lock held")
    assert host.held_at.get("_grace_active_exclusions", []) == [False], why("the grace query takes the lock itself")
    assert host.held_at.get("queue.remove", []) == [False], why("a slot is taken out with no registry lock held")


async def test_a_declined_verdict_notes_the_memo_and_takes_no_slot(caplog):
    host = staged_host()
    host.vram_ok = False
    await run_recheck(host)
    assert host.calls.count("queue.remove") == 0 and "_reserve_and_start_locked" not in host.calls
    assert len(host.staged) == 1, why("the declined request stays staged")
    assert host.state.norom.blocked("m", NOW + 59.9), why("a decline remembers the tag for the memo window")
    assert not host.state.norom.blocked("m", NOW + NO_ROOM_MEMO_S)
    assert any("reason=vram_refused" in r.getMessage() for r in caplog.records), why("the decline line is missing")


async def test_a_plain_vram_shortage_reads_vram_refused_and_the_cards_are_read_under_the_lock(caplog):
    host = staged_host()
    host.pick_result = (None, "none")  # the picker finds no card; cards are visible
    await run_recheck(host)
    assert host.held_at.get("_card_avail_mib") == [True], why("the card reading is part of the probe, under the lock")
    assert "queue.remove" not in host.calls and host.state.norom.blocked("m", NOW + 1.0)
    assert any("reason=vram_refused" in r.getMessage() for r in caplog.records), why("a shortage is vram_refused")


async def test_the_memo_stops_a_second_probe_until_it_expires():
    host = staged_host()
    host.vram_ok = False
    await run_recheck(host)
    host.now = NOW + 10.0
    await run_recheck(host)
    assert probes(host) == 1, why("inside the 60 s memo the tag is not probed again")
    host.now = NOW + 60.0
    await run_recheck(host)
    assert probes(host) == 2, why("after 60 s the tag is probed again")


async def test_no_live_engine_inside_the_lock_ends_the_attempt_quietly():
    class Gone(FakeHost):
        def _instances_for(self, model_tag):
            found = super()._instances_for(model_tag)
            return [] if self.held else found

    host = Gone(residents=[FakeResident()], staged=[FakeSlot(created_at=NOW - 30.0)], now=NOW)
    await run_recheck(host)
    assert probes(host) == 0 and "queue.remove" not in host.calls
    assert len(host.state.norom) == 0, why("no engine to ask about is not a no-room verdict")


# -- the whole flow, in call order ----------------------------------------------------------

async def test_a_staged_request_is_admitted_through_exactly_these_calls():
    host = staged_host()
    await run_recheck(host)
    assert host.calls == [
        "_co_residence_state", "_co_residence_now", "_grace_active_exclusions",
        "queue.staging_snapshot", "_instances_for", "_model_residents", "_instances_for",
        "lock_enter", "_instances_for", "_route_cap_and_placement", "_resolve_placement_locked",
        "_pick_card_for_new_instance", "_vram_admits_locked", "lock_exit",
        "_model_residents", "queue.remove",
        "reserving_enter", "lock_enter", "_reserve_and_start_locked", "_session_key_for", "_remember_affinity",
        "lock_exit", "reserving_exit",
    ], why(f"call order was {host.calls!r}")
    assert not hasattr(host, "_route_or_reserve"), why("the host has no routing method; none may be called")


async def test_the_action_line_is_written_before_the_slot_is_taken(caplog):
    seen = []

    host = FakeHost(residents=[FakeResident()], staged=[FakeSlot(created_at=NOW - 30.0)], now=NOW)
    original = host.queue.remove

    async def remove(slot_id):
        seen.extend(r.getMessage() for r in caplog.records)
        return await original(slot_id)

    host.queue.remove = remove
    await run_recheck(host)
    assert seen == ["CO_RESIDENCE_RECHECK tag=m source=staged slot=s1 waited_s=30.0 instances=1 residents=1"], why(
        f"line before the take was {seen!r}")


async def test_the_recheck_never_raises_and_logs_when_something_inside_fails(caplog):
    class Broken(FakeHost):
        async def _grace_active_exclusions(self):
            raise RuntimeError("grace failed")

    host = Broken(residents=[FakeResident()], now=NOW)
    await run_recheck(host)
    assert any("re-check failed" in r.getMessage() for r in caplog.records), why("the failure must be logged")


async def test_a_cancellation_always_propagates():
    class Cancelled(FakeHost):
        async def _grace_active_exclusions(self):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_recheck(Cancelled(residents=[FakeResident()], now=NOW))
