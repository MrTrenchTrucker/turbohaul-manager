"""ONE new per-turn telemetry event,
`turn_dispatch`, that fires at the TRUE engine handoff on BOTH the
cold/anchor turn and the warm ACTIVE_MATCH grace-window follow-up -- the
second (or third, fourth...) turn to the SAME warm resident, served on the
same handle/KV without ever re-entering slot_assign/prefill_start. That is
the case that stays silent for the large majority
of arrivals (slot_assign/prefill_start counts far below request_arrival),
and it is the case existing telemetry (slot_assign, prefill_start) cannot
see: those two fire exactly once, for the anchor slot pulled fresh off
r.inbox, never for a matched follow-up promoted inside the grace loop
(the telemetry-parity test's own
test_active_match_followup_does_not_get_its_own_events pins that asymmetry
as correct for the THREE existing hooks -- it must stay correct for them).

ANCHOR: the event is placed in manager.py,
in _serve_on_resident, on
both paths --
  cold (anchor):  slot.engine_op="prefill"    -> _probe_and_save_clean_kv(handle, slot)
                  -> slot.engine_op="decode"  -> turn_dispatch -> _complete_fn(slot, handle)
                  (streaming arm: turn_dispatch immediately before stream_ready_event.set())
  warm (matched): matched.engine_op="prefill" -> _probe_and_save_clean_kv(handle, matched)
                  -> matched.engine_op="decode" -> turn_dispatch -> _complete_fn(matched, handle)
                  (streaming arm: turn_dispatch immediately before stream_ready_event.set())
Placed AFTER _probe_and_save_clean_kv on both arms, never before: that probe is
"no-op unless single-series + large ctx + no equal/larger clean bin already
saved" (its own call-site comment) but does real KV-probe/disk-save I/O when
it isn't -- firing turn_dispatch earlier (mirroring prefill_start's OWN
existing, un-retimed placement) would fold that latency into the PREFILL
bucket (turn_dispatch -> first_token) on the turns where it fires, biasing
exactly the split this event exists to make trustworthy.
The event covers BOTH paths, not warm-only: if only ACTIVE_MATCH emitted,
cold turns would carry a
differently-biased WAIT/PREFILL split than warm turns, and nobody reading the
numbers would know. Hence FOUR call sites, ONE event type, additive.

MUST FAIL ON THE UNMODIFIED TREE: `test_turn_dispatch_fires_on_second_turn_
to_same_warm_resident` is the killing test -- pre-fix, no code anywhere calls
`self._telemetry.on_turn_dispatch`, so `mgr._telemetry.get_events(event_
type="turn_dispatch")` returns an empty list regardless of how many turns
are driven. Reverting src/turbohaul/manager.py +
src/turbohaul/telemetry.py only (this test file is unaffected by
the revert) makes this test fail, since the emitter is then absent.

State is DRIVEN by real traffic through `mgr.submit()` (the production
admission path), never hand-set on a Slot -- an anti-pattern, since
hand-set state can diverge from real traffic. `max_parallel_sidecars` is set EXPLICITLY to 2
in every boot below (the env lies: a deployment sets TURBOHAUL_MAX_PARALLEL=2, config
Field(default=1)) and asserted right after boot, not merely passed, so a
silent default cannot slip through unnoticed.

GOTCHA (not this test's defect, navigated around here, per
the telemetry-parity test's own docstring): `turbohaul.telemetry.
init_telemetry` is a MODULE-LEVEL SINGLETON -- every TurbohaulManager built in
the same pytest process shares ONE FlapTelemetry unless the singleton is
reset first. Every test below resets it before constructing its manager and
restores it to None afterward.

VERIFICATION: new observable behaviour -- a new telemetry
event now fires where nothing fired before. The killing test fails on
the code without the emitter (see the paragraph above) and passes once
the emitter is in place (src/turbohaul/manager.py, src/turbohaul/telemetry.py).
The change is additive: one event type, four call sites.
"""
import asyncio

import pytest

import turbohaul.telemetry as telemetry_module
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db

from _fastlane_fixture import (
    boot_ranked_runtime,
    seed_manifest,
    make_fakes,
    high_vram,
    wait_until,
)


def _fresh_telemetry_boot_and_manager(tmp_path, *, max_parallel_sidecars=2,
                                       grace_seconds=5, idle_hot_load_seconds=60):
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
    return boot, runtime, mgr


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
class TestTurnDispatchWarmReuse:
    async def test_turn_dispatch_fires_on_second_turn_to_same_warm_resident(
        self, tmp_path
    ):
        """THE killing test. Drives an anchor turn to GRACE, confirms it got
        its one turn_dispatch (ground truth the cold-path sites work),
        then drives a SECOND real turn on the SAME thread while the resident
        is still inside its grace window -- the ACTIVE_MATCH promotion --
        and asserts the follow-up ALSO gets an turn_dispatch, with the
        right payload. On unmodified code the follow-up gets zero: no hook
        exists anywhere in that branch."""
        try:
            boot, runtime, mgr = _fresh_telemetry_boot_and_manager(
                tmp_path, max_parallel_sidecars=2, grace_seconds=5,
            )
            assert runtime.queue.max_parallel_sidecars == 2, (
                "explicit cap -- the env lies (TURBOHAUL_MAX_PARALLEL=2 on "
                "a typical deployment); a test that inherits a default risks "
                "silently measuring the wrong dispatch path"
            )
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={"marker": "anchor"}, admission_ctx_len=500,
                )
                await wait_until(
                    lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                    timeout=5.0,
                )

                # Ground truth: the cold/anchor call site fired exactly once,
                # with the real admission_ctx_len this call passed in --
                # proves the fixture and the anchor-side wiring both work
                # before the interesting (warm) assertion below is judged.
                anchor_events = _events_for_slot(mgr, "turn_dispatch", anchor.slot_id)
                assert len(anchor_events) == 1, (
                    f"expected exactly 1 turn_dispatch for the anchor turn, "
                    f"got {len(anchor_events)}"
                )
                assert anchor_events[0]["thread_id"] == "t1"
                assert anchor_events[0]["model_tag"] == "kvtest"
                assert anchor_events[0]["prompt_tokens"] == 500, (
                    f"anchor turn_dispatch prompt_tokens = "
                    f"{anchor_events[0]['prompt_tokens']!r}, expected 500 "
                    "(the real admission_ctx_len this submit() call passed)"
                )

                followup = await mgr.submit(
                    model_tag="kvtest", prompt="again", thread_id="t1",
                    client_meta={"marker": "followup"}, admission_ctx_len=750,
                )
                await wait_until(
                    lambda: followup.state is SlotState.POPPED, timeout=5.0,
                )

                # Non-vacuity: the follow-up really was promoted via
                # ACTIVE_MATCH (a genuine grace-window continuation), not
                # served as a fresh anchor cycle -- if it were, this whole
                # test would be measuring the wrong branch. Reuse the telemetry-parity test's
                # own asymmetry fact as the control: slot_assign/prefill_start
                # fire exactly once (anchor only) and NEVER for a promoted
                # follow-up, on both unmodified and fixed code alike.
                assert _events_for_slot(mgr, "slot_assign", followup.slot_id) == [], (
                    "sanity: the follow-up got its own slot_assign event -- "
                    "it was served as a fresh anchor turn, not an ACTIVE_MATCH "
                    "promotion, so this test is not exercising the silent "
                    "case it claims to"
                )
                assert _events_for_slot(mgr, "prefill_start", followup.slot_id) == [], (
                    "sanity: the follow-up got its own prefill_start event -- "
                    "same problem as above"
                )

                # THE assertion.
                followup_events = _events_for_slot(mgr, "turn_dispatch", followup.slot_id)
                assert len(followup_events) == 1, (
                    "expected exactly 1 turn_dispatch event for "
                    "the grace-window ACTIVE_MATCH follow-up. On unmodified "
                    "code this is 0 -- no hook exists anywhere in that "
                    "branch -- which IS the silent-for-most-arrivals case "
                    "the inter-turn race analysis measured."
                )
                assert followup_events[0]["thread_id"] == "t1"
                assert followup_events[0]["model_tag"] == "kvtest"
                assert followup_events[0]["prompt_tokens"] == 750, (
                    f"follow-up turn_dispatch prompt_tokens = "
                    f"{followup_events[0]['prompt_tokens']!r}, expected 750"
                )

                all_dispatch = mgr._telemetry.get_events(
                    event_type="turn_dispatch", limit=1000
                )["events"]
                assert len(all_dispatch) == 2, (
                    f"expected exactly 2 turn_dispatch events total (anchor "
                    f"+ follow-up), got {len(all_dispatch)}"
                )

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None

    async def test_turn_dispatch_never_carries_reused_tokens(self, tmp_path):
        """Payload-shape control -- direct evidence for the payload design (a):
        reused_tokens/engine_reused_tokens is only known post-hoc (see
        _log_kv_reuse_outcome's own docstring: never printed for any
        wait_state other than "completed", and even then only after matching
        a live engine heartbeat) -- so it must never appear on this event.
        Shipping it here would be null or wrong on the one path this event
        exists for."""
        try:
            boot, runtime, mgr = _fresh_telemetry_boot_and_manager(tmp_path)
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="kvtest", prompt="hi", thread_id="t1",
                    client_meta={}, admission_ctx_len=123,
                )
                await wait_until(
                    lambda: _events_for_slot(mgr, "turn_dispatch", anchor.slot_id),
                    timeout=5.0,
                )
                ev = _events_for_slot(mgr, "turn_dispatch", anchor.slot_id)[0]
                assert ev["prompt_tokens"] == 123
                assert "reused_tokens" not in ev, (
                    "turn_dispatch payload carries reused_tokens -- this "
                    "figure is never known at handoff, see (a) in the payload "
                    "design notes"
                )
                assert "engine_reused_tokens" not in ev

            await mgr.shutdown()
        finally:
            telemetry_module._telemetry = None
