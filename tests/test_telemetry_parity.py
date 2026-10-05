"""Port the three stranded telemetry hooks (on_slot_assign,
on_prefill_start, on_completion) from Loop B (_process_slot) into Loop A
(_serve_on_resident), so that a real, wired consumer (`GET
/v1/telemetry/events`, src/turbohaul/api/telemetry.py) sees these event
types at cap>=2 too, not only at cap<=1.

MUST FAIL ON UNMODIFIED CODE: pre-fix, _serve_on_resident never calls these
three, so at cap>=2 `mgr._telemetry.get_events(event_type="slot_assign")`
etc. return EMPTY lists regardless of how many turns are driven.

★ A known trap: asserting the telemetry *method was
called* passes whether or not the payload is right. Every test here reads
back the REAL FlapTelemetry ring buffer (the same object the wired read
endpoint queries) and asserts the actual event payload fields, not a mock's
call count.

GOTCHA WORTH KNOWING (not a defect of this port, navigated around here):
`turbohaul.telemetry.init_telemetry` is a MODULE-LEVEL SINGLETON
(`global _telemetry`, idempotent) -- every TurbohaulManager constructed in
the same pytest process shares ONE FlapTelemetry unless the singleton is
reset first. Every test below resets it before constructing its manager
(fresh ring buffer scoped to that test's own tmp_path) and restores it to
None afterward, so no test here leaks into another test in the suite, in
either direction.
"""
import asyncio

import pytest

import turbohaul.telemetry as telemetry_module
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db

from _fastlane_fixture import (
    boot_ranked_runtime,
    seed_manifest,
    make_fakes,
    high_vram,
    wait_until,
    resident_for,
)


def _fresh_telemetry_boot_and_manager(tmp_path, *, max_parallel_sidecars=2,
                                       grace_seconds=2, idle_hot_load_seconds=60):
    telemetry_module._telemetry = None
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=max_parallel_sidecars,
        grace_seconds=grace_seconds, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, "kvtest", main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({})
    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=spawn_fn, health_fn=health_fn, sigterm_fn=sigterm_fn,
        vram_fn=vram_fn, complete_fn=complete_fn,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    return boot, mgr


def _events_for_slot(mgr, event_type, slot_id):
    """Filter by slot_id too, not just event_type -- defensive against the
    singleton having anything else in it despite the reset above."""
    result = mgr._telemetry.get_events(event_type=event_type, limit=1000)
    return [e for e in result["events"] if e.get("slot_id") == slot_id]


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


@pytest.mark.asyncio
class TestTelemetryParity:
    async def test_all_three_events_emitted_at_two_resident_cap_with_correct_payload(
        self, tmp_path
    ):
        """THE killing test. Drives one anchor turn at cap>=2 through
        ACTIVE->GRACE and reads back the REAL ring buffer (the same object
        `GET /v1/telemetry/events` queries) for all three event types,
        asserting actual payload fields -- not just that a call happened."""
        try:
            boot, mgr = _fresh_telemetry_boot_and_manager(tmp_path)
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={"marker": "anchor"},
                )
                await wait_until(
                    lambda: _events_for_slot(mgr, "completion", anchor.slot_id),
                    timeout=5.0,
                )

                assign_events = _events_for_slot(mgr, "slot_assign", anchor.slot_id)
                prefill_events = _events_for_slot(mgr, "prefill_start", anchor.slot_id)
                completion_events = _events_for_slot(mgr, "completion", anchor.slot_id)

                assert len(assign_events) == 1, (
                    f"expected exactly 1 slot_assign event for the anchor at "
                    f"cap>=2, got {len(assign_events)} -- on unmodified code "
                    f"this is 0 (the hook is never called on this path)"
                )
                a = assign_events[0]
                assert a["thread_id"] == "t1"
                assert a["model_tag"] == "kvtest"
                assert a["wait_in_queue_s"] is not None, (
                    "wait_in_queue_s is None -- proves on_slot_assign fired "
                    "before on_queue_state ever stamped _slot_queue_enter, "
                    "i.e. the wrong moment (this is exactly why "
                    "_reserve_and_start_locked was rejected as the site)"
                )

                assert len(prefill_events) == 1, (
                    f"expected exactly 1 prefill_start event, got "
                    f"{len(prefill_events)}"
                )
                assert prefill_events[0]["thread_id"] == "t1"
                assert prefill_events[0]["model_tag"] == "kvtest"

                assert len(completion_events) == 1, (
                    f"expected exactly 1 completion event, got "
                    f"{len(completion_events)}"
                )
                c = completion_events[0]
                assert c["thread_id"] == "t1"
                assert c["model_tag"] == "kvtest"
                assert c["reason"] == "grace_enter", (
                    f"completion event's reason={c['reason']!r}, expected "
                    "'grace_enter' -- matching _process_slot's own call "
                    "exactly, not a different reason string"
                )
                assert c["total_lifecycle_s"] is not None

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None

    async def test_active_match_followup_does_not_get_its_own_events(
        self, tmp_path
    ):
        """Asymmetry control: _process_slot has exactly ONE call site each
        for these three hooks, so a grace-window ACTIVE_MATCH-promoted
        follow-up gets none of them -- only the anchor does, once. A port
        that fired these inside the grace loop's matched-promotion branch
        too would emit MORE telemetry than the cap<=1 path ever did for the
        equivalent scenario. Must stay exactly 1 even after a promotion."""
        try:
            boot, mgr = _fresh_telemetry_boot_and_manager(tmp_path)
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={"marker": "anchor"},
                )
                await wait_until(
                    lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                    timeout=5.0,
                )
                # Precondition: the anchor really did get its one event
                # before the follow-up is even submitted.
                assert len(_events_for_slot(mgr, "slot_assign", anchor.slot_id)) == 1

                followup = await mgr.submit(
                    model_tag="kvtest", prompt="again", thread_id="t1",
                    client_meta={"marker": "followup"},
                )
                await wait_until(
                    lambda: followup.state is SlotState.POPPED, timeout=5.0,
                )

                # Non-vacuity: the follow-up really was promoted via
                # ACTIVE_MATCH (not e.g. queued as a brand-new anchor turn),
                # which is the only scenario this asymmetry claim is about.
                r = resident_for(mgr, "kvtest")
                assert r is not None

                all_slot_assign = mgr._telemetry.get_events(
                    event_type="slot_assign", limit=1000
                )["events"]
                all_completion = mgr._telemetry.get_events(
                    event_type="completion", limit=1000
                )["events"]
                assert len(all_slot_assign) == 1, (
                    f"expected exactly 1 slot_assign total (anchor only) even "
                    f"after an ACTIVE_MATCH promotion, got "
                    f"{len(all_slot_assign)} -- the promoted follow-up wrongly "
                    f"got its own slot_assign event"
                )
                assert len(all_completion) == 1, (
                    f"expected exactly 1 completion total (anchor only), got "
                    f"{len(all_completion)} -- the promoted follow-up wrongly "
                    f"got its own completion event"
                )
                assert all_slot_assign[0]["slot_id"] == anchor.slot_id
                assert all_completion[0]["slot_id"] == anchor.slot_id

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None

    async def test_assign_guard_shields_the_serve_path_and_suppresses_its_sibling(
        self, tmp_path
    ):
        """NO NEW BUGS control, directly targeting the class of regression
        where a faithful port of the CODE that
        drops the SAFETY of the original's guard can break a passing test
        elsewhere. on_slot_assign is made to raise; asserts (a) the anchor's
        own turn still completes -- the exception does not escape into the
        serve path -- and (b) on_prefill_start, which shares the SAME guard
        as on_slot_assign on the cap<=1 side, is correctly suppressed too
        (byte-for-byte guard shape, not just "some guard" wrapping each call
        separately) while (c) on_completion, which has its OWN separate
        guard, still fires normally -- proving the two guards are scoped
        independently, matching _process_slot's own two guard
        sites exactly."""
        try:
            boot, mgr = _fresh_telemetry_boot_and_manager(tmp_path)

            real_on_slot_assign = mgr._telemetry.on_slot_assign
            calls = {"prefill_start": 0}

            def _raising_on_slot_assign(slot):
                real_on_slot_assign(slot)  # still record the real event...
                raise RuntimeError("boom -- simulated telemetry failure")

            real_on_prefill_start = mgr._telemetry.on_prefill_start

            def _counting_on_prefill_start(slot):
                calls["prefill_start"] += 1
                real_on_prefill_start(slot)

            mgr._telemetry.on_slot_assign = _raising_on_slot_assign
            mgr._telemetry.on_prefill_start = _counting_on_prefill_start

            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={"marker": "anchor"},
                )
                await wait_until(
                    lambda: _events_for_slot(mgr, "completion", anchor.slot_id),
                    timeout=5.0,
                )

                assert _events_for_slot(mgr, "completion", anchor.slot_id), (
                    "anchor never reached completion -- the simulated "
                    "on_slot_assign exception escaped into the serve path "
                    "instead of being swallowed by its try/except"
                )
                assert calls["prefill_start"] == 0, (
                    "on_prefill_start WAS called despite on_slot_assign "
                    "raising first -- means the two calls are NOT under the "
                    "same try/except any more (a guard-shape regression: "
                    "_process_slot's cap<=1 site shares ONE guard for both)"
                )

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None

    async def test_one_resident_cap_loop_b_control_still_emits_all_three_unaffected(
        self, tmp_path
    ):
        """★ DISJOINT CONTROL -- must PASS both before and after this port.
        Loop B (_process_slot, cap<=1) is untouched by this port; drives one
        turn through it and confirms all three events still emit exactly as
        they did before, proving the port did not disturb the original
        cap<=1 call sites."""
        try:
            boot, mgr = _fresh_telemetry_boot_and_manager(
                tmp_path, max_parallel_sidecars=1,
            )
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={"marker": "anchor"},
                )
                await wait_until(
                    lambda: _events_for_slot(mgr, "completion", anchor.slot_id),
                    timeout=5.0,
                )

                assert len(_events_for_slot(mgr, "slot_assign", anchor.slot_id)) == 1
                assert len(_events_for_slot(mgr, "prefill_start", anchor.slot_id)) == 1
                completion_events = _events_for_slot(mgr, "completion", anchor.slot_id)
                assert len(completion_events) == 1
                assert completion_events[0]["reason"] == "grace_enter"

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None
