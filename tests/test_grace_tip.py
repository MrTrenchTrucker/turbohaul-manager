"""Port the `_grace_tip` anchor-vs-tip KV identity
discipline from Loop B (`_process_slot`) into Loop A
(`_serve_on_resident` / `_drive_resident`).

`_unload_teardown` and `_drive_resident`'s own finally-block save both source
the pre-SIGTERM KV save's identity from `r.idle_thread_id` /
`r.idle_client_meta` / `r.idle_admission_ctx_len` -- written by
`_drive_resident`'s two park-stamping sites. Before this change those sites read
the raw anchor `slot` directly, so a grace-window ACTIVE_MATCH follow-up that
actually lands newer content in the engine's KV had its teardown-time save
wrongly attribute to the STALE anchor identity -- exactly the defect class
`_grace_tip` was built to prevent, previously only prevented on the
Loop B side.

All three tests drive the REAL dispatcher (mgr.worker_loop -> _dispatch_loop
-> _drive_resident -> _serve_on_resident), not a hand-assembled Resident, so
the turn-start reset (_drive_resident) and the grace-loop promotion
(_serve_on_resident) are exercised exactly as the real code wires them
together -- not reimplemented in the test.

MUST FAIL ON UNMODIFIED CODE: pre-fix, the park-stamping sites read the raw
anchor `slot` (the `grace_tip` Resident field does not even exist), so
r.idle_client_meta / r.idle_admission_ctx_len show the ANCHOR's values even
after a follow-up was promoted.
"""
import asyncio

import pytest

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


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


def _boot_and_manager(tmp_path, *, grace_seconds=2, idle_hot_load_seconds=60):
    boot, runtime = boot_ranked_runtime(
        tmp_path, grace_seconds=grace_seconds, idle_hot_load_seconds=idle_hot_load_seconds,
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


@pytest.mark.asyncio
class TestGraceTipAttribution:
    async def test_teardown_save_attributes_to_promoted_followup_not_stale_anchor(
        self, tmp_path
    ):
        """THE defect this change fixes. An ACTIVE_MATCH follow-up promoted during
        the anchor's grace window lands DIFFERENT content in the engine's KV
        (different client_meta / admission_ctx_len, same thread_id -- that is
        what "matched" means: pop_matched_thread keys on thread_id+model_tag).
        The park-time save must attribute to the FOLLOW-UP's identity, not the
        anchor's.

        MUST FAIL ON UNMODIFIED CODE: r.idle_client_meta / r.idle_admission_
        ctx_len would show the ANCHOR's values here (111 / "anchor"), not the
        follow-up's (777 / "followup") -- pre-fix, the park-stamping sites
        read the raw anchor `slot` directly."""
        boot, mgr = _boot_and_manager(tmp_path)
        with high_vram():
            anchor = await mgr.submit(
                model_tag="kvtest", prompt="hi", thread_id="t1",
                client_meta={"marker": "anchor"}, admission_ctx_len=111,
            )
            await wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                timeout=5.0,
            )

            followup = await mgr.submit(
                model_tag="kvtest", prompt="again", thread_id="t1",
                client_meta={"marker": "followup"}, admission_ctx_len=777,
            )
            # NOTE: "active_match_completed" is written to the slots table's
            # end_reason column via mark_slot_ended, NOT to audit_events --
            # poll the Slot object's own state instead (same in-memory
            # reference pop_matched_thread hands back to _serve_on_resident).
            await wait_until(
                lambda: followup.state is SlotState.POPPED,
                timeout=5.0,
            )

            r = resident_for(mgr, "kvtest")
            await wait_until(lambda: r.state is ResidentState.IDLE_EVICTABLE, timeout=5.0)

            assert r.idle_thread_id == "t1"  # non-vacuity: the field was written at all
            assert r.idle_client_meta is not None, (
                "idle_client_meta is None -- park-stamping never ran, or ran "
                "with parallel != 1; a fixture/non-vacuity failure, not the "
                "assertion under test"
            )
            assert r.idle_client_meta.get("marker") == "followup", (
                f"teardown save attributed to marker="
                f"{r.idle_client_meta.get('marker')!r}, expected the PROMOTED "
                "follow-up's identity ('followup'), not the stale anchor's "
                "('anchor')"
            )
            assert r.idle_admission_ctx_len == 777, (
                f"teardown save attributed admission_ctx_len="
                f"{r.idle_admission_ctx_len}, expected the follow-up's 777, "
                "not the anchor's 111"
            )

        await mgr.shutdown()

    async def test_second_turn_uses_its_own_anchor_not_stale_tip_from_turn_one(
        self, tmp_path
    ):
        """Regression control for the STALE-TIP case: turn 1 promotes a
        follow-up (grace_tip now points at it); turn 2 starts FRESH on a
        brand-new anchor, on the SAME Resident, with no promotion. The
        park-time save after turn 2 must attribute to turn 2's own anchor, not
        turn 1's leftover follow-up.

        Constructed to FAIL if the turn-start reset (_drive_resident's
        `r.grace_tip = slot`) is missing or removed: without it, r.grace_tip
        would still hold turn 1's follow-up, and turn 2's park-stamping would
        wrongly show thread_id="t1" / marker="followup1" instead of
        thread_id="t2" / marker="anchor2"."""
        boot, mgr = _boot_and_manager(tmp_path)
        with high_vram():
            # --- turn 1: anchor1 + a promoted follow-up ---
            anchor1 = await mgr.submit(
                model_tag="kvtest", prompt="hi", thread_id="t1",
                client_meta={"marker": "anchor1"}, admission_ctx_len=111,
            )
            await wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor1.slot_id),
                timeout=5.0,
            )
            followup1 = await mgr.submit(
                model_tag="kvtest", prompt="again", thread_id="t1",
                client_meta={"marker": "followup1"}, admission_ctx_len=777,
            )
            await wait_until(
                lambda: followup1.state is SlotState.POPPED,
                timeout=5.0,
            )

            r = resident_for(mgr, "kvtest")
            await wait_until(lambda: r.state is ResidentState.IDLE_EVICTABLE, timeout=5.0)
            # Precondition: turn 1 really did attribute to followup1 -- if
            # this fails, the fixture itself is broken, not a turn-2 defect.
            assert r.idle_client_meta is not None
            assert r.idle_client_meta.get("marker") == "followup1", (
                "precondition failed: turn 1 did not attribute to its own "
                "promoted follow-up -- fixture broken, not a turn-2 defect"
            )

            # --- turn 2: a brand-new anchor, DIFFERENT thread, no promotion,
            # admitted onto the SAME (now-idle) resident via warm reuse ---
            anchor2 = await mgr.submit(
                model_tag="kvtest", prompt="new", thread_id="t2",
                client_meta={"marker": "anchor2"}, admission_ctx_len=222,
            )
            await wait_until(
                lambda: r.state is ResidentState.ACTIVE
                and r.active_slot is not None
                and r.active_slot.thread_id == "t2",
                timeout=5.0,
            )
            assert anchor2.slot_id  # non-vacuity: turn 2 really was admitted
            await wait_until(
                lambda: r.state is ResidentState.IDLE_EVICTABLE
                and r.idle_thread_id == "t2",
                timeout=5.0,
            )

            assert r.idle_thread_id == "t2", (
                f"idle_thread_id={r.idle_thread_id!r} after turn 2 -- expected "
                "'t2'; a value of 't1' means a STALE tip survived from turn 1"
            )
            assert r.idle_client_meta is not None
            assert r.idle_client_meta.get("marker") == "anchor2", (
                f"teardown save attributed to marker="
                f"{r.idle_client_meta.get('marker')!r} after turn 2 -- "
                "expected turn 2's own anchor ('anchor2'), not a stale tip "
                "surviving from turn 1 ('followup1')"
            )
            assert r.idle_admission_ctx_len == 222

        await mgr.shutdown()

    async def test_no_promotion_control_still_attributes_to_the_anchor(
        self, tmp_path
    ):
        """★ NEGATIVE CONTROL -- must PASS both before and after this change. With
        no grace-window follow-up at all, the anchor's own identity must
        still be what gets attributed at park: proves the port is a pure
        addition on the promoted-follow-up case, not a regression on the
        dominant no-promotion case."""
        boot, mgr = _boot_and_manager(tmp_path)
        with high_vram():
            anchor = await mgr.submit(
                model_tag="kvtest", prompt="hi", thread_id="t1",
                client_meta={"marker": "anchor"}, admission_ctx_len=111,
            )
            r = resident_for(mgr, "kvtest")
            await wait_until(lambda: r.state is ResidentState.IDLE_EVICTABLE, timeout=5.0)

            assert r.idle_thread_id == "t1"
            assert r.idle_client_meta is not None
            assert r.idle_client_meta.get("marker") == "anchor", (
                f"teardown save attributed to marker="
                f"{r.idle_client_meta.get('marker')!r} with NO follow-up ever "
                "promoted -- expected the anchor's own identity unchanged"
            )
            assert r.idle_admission_ctx_len == 111
            assert "grace_fastlane_breakout" not in _audit_events(boot, anchor.slot_id)

        await mgr.shutdown()

    async def test_immediate_evict_path_also_attributes_to_promoted_followup(
        self, tmp_path
    ):
        """Companion to the FIRST killing test above, but for the OTHER
        park-stamping site: _drive_resident's IMMEDIATE-EVICT branch
        (`idle_window <= 0`), not the regular IDLE_EVICTABLE park.
        `idle_hot_load_seconds=0` forces that branch once grace ends, while
        the grace loop itself still runs completely normally -- skip_grace
        is gated on designated-victim status alone, never on idle_window --
        so a follow-up can still be promoted before this resident evicts.

        This test covers ONLY the immediate-evict branch's grace_tip, which
        must not read back the raw anchor `slot`; the other
        tests above -- none of them ever drives this branch at all (all
        three use idle_hot_load_seconds=60, which only ever reaches the
        regular-park branch). This test closes that gap by driving the
        immediate-evict branch directly and asserting the promoted
        follow-up's identity."""
        boot, mgr = _boot_and_manager(tmp_path, idle_hot_load_seconds=0)
        with high_vram():
            anchor = await mgr.submit(
                model_tag="kvtest", prompt="hi", thread_id="t1",
                client_meta={"marker": "anchor"}, admission_ctx_len=111,
            )
            await wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                timeout=5.0,
            )

            followup = await mgr.submit(
                model_tag="kvtest", prompt="again", thread_id="t1",
                client_meta={"marker": "followup"}, admission_ctx_len=777,
            )
            await wait_until(
                lambda: followup.state is SlotState.POPPED,
                timeout=5.0,
            )

            r = resident_for(mgr, "kvtest")
            # Non-vacuity: this resident must actually take the IMMEDIATE-
            # EVICT branch, not the regular park -- if idle_client_meta never
            # gets written at all, that is a fixture failure, not a defect
            # in grace_tip.
            await wait_until(lambda: r.idle_client_meta is not None, timeout=5.0)

            assert r.idle_thread_id == "t1"
            assert r.idle_client_meta.get("marker") == "followup", (
                f"immediate-evict teardown save attributed to marker="
                f"{r.idle_client_meta.get('marker')!r}, expected the "
                "PROMOTED follow-up's identity ('followup'), not the stale "
                "anchor's ('anchor')"
            )
            assert r.idle_admission_ctx_len == 777, (
                f"immediate-evict teardown save attributed admission_ctx_len="
                f"{r.idle_admission_ctx_len}, expected the follow-up's 777, "
                "not the anchor's 111"
            )

        await mgr.shutdown()
