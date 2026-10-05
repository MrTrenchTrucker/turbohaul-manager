"""Scoping rules for the starvation notification: the
notification `_defer_unroutable` -> `_notify_grace_starvation_candidate`, added
to cover the grace-loop starvation blind spot, has FOUR scoping rules, each
pinned by one test in this file:

  registered-claim -- a Fast Lane-REGISTERED candidate must never break a non-victim's
        grace (the Fast Lane exclusion, applied at _starvation_
        breakout_gate).
  fresh-candidate -- a FRESH (not-yet-aged) candidate must never be notified.
  same-model-defer -- a SAME-model candidate must never be notified (the starving
        population is, by definition, every OTHER model).
  stale-candidate -- a candidate notified in one grace window must not leak into a later
        window on the same resident (the grace-entry clear).

Each test below drives the REAL `TurbohaulManager._defer_unroutable` path
directly (never `queue.enqueue`) --
that is the one and only production entrance to `_notify_grace_starvation_
candidate`. These tests pin
the four scoping rules, one per rule, which no other test in
the suite covers.
"""
import asyncio
import time

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, grace_seconds=30, max_other_model_wait_s=20.0,
                   fastlane_rules=None):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
            max_grace_extensions=50,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=fastlane_rules or []),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.pid = 77_777
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


async def _wait_until(predicate, *, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def _resident(mgr, model_tag: str, port: int, grace_seconds: float) -> Resident:
    """Builds a Resident AND registers it into mgr._residents -- required
    because _notify_grace_starvation_candidate iterates
    mgr._residents.values(), unlike the buffer-scan path
    (_starvation_breakout_candidate), which never touches the registry at
    all. A Resident built without this step is invisible to the
    notification path regardless of correctness."""
    handle = _make_fake_handle(model_tag, port)
    r = Resident(
        model_tag=model_tag,
        handle=handle,
        port=handle.port,
        grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
    )
    mgr._residents[model_tag] = r
    return r, handle


@pytest.mark.asyncio
class TestDeferUnroutableStarvationNotificationScoping:
    """Drives the real _defer_unroutable path for every case; never
    queue.enqueue. Each test's positive assertion is the decisive one
    (the scoping rule under test); each also confirms non-vacuity (the
    predicate branch it depends on genuinely fired) where practical."""

    async def test_fastlane_registered_candidate_does_not_break_grace(
        self, tmp_path
    ):
        """Registered-claim rule: an AGED, Fast Lane-REGISTERED other-model slot fails
        routing (via the real _defer_unroutable) while a resident is in
        grace for a DIFFERENT model. Must NOT break that grace -- the Fast Lane
        exclusion rule, applied at _starvation_breakout_gate, which
        _notify_grace_starvation_candidate's caller (the grace loop) must
        still honour for the notified path exactly as it does for
        the buffer-scan path.

        Guards against a notified candidate skipping the Fast Lane gate: in that case
        the grace loop breaks on the registered candidate and this
        test's final assertion (no grace_starvation_breakout) fails."""
        grace_seconds = 5
        max_other_model_wait_s = 0.2
        # A real rule, so the anchor resident below can be given a
        # RESOLVABLE identity that TIES slot_b's claim rank. Without this,
        # r.rank_client_meta has nothing to resolve against
        # (_resident_priority_key's table), the anchor stays UNRESOLVABLE,
        # sorts worst by construction, and becomes the DESIGNATED
        # VICTIM the instant ANY claim registers -- a real, correct,
        # UNRELATED mechanism (grace_designated_unload_target_break) that
        # would end this grace window first and make the test vacuous for
        # what it actually claims to cover (the Fast Lane gate on the
        # notified path specifically).
        rules = [FastLaneRule(address="10.0.0.9", tag_ranks=FastLaneTagRanks(main=1))]
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
            fastlane_rules=rules,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )
        try:
            r, handle = _resident(mgr, "m1", 59900, grace_seconds)
            # Tie r's own identity to the SAME rule/rank slot_b's claim will
            # carry (see the comment above `rules`).
            r.rank_client_meta = {"ip": "10.0.0.9", "is_main": True}
            anchor = Slot.new("m1", prompt="hi", thread_id="t1")
            anchor.state = SlotState.STAGED

            task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
            assert await _wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                timeout=5.0,
            ), "fixture failure: grace never entered"

            # AGED, REGISTERED, different model -- fails routing via the
            # REAL _defer_unroutable, never queue.enqueue.
            slot_b = Slot.new("m2", prompt="hi", thread_id="t2")
            slot_b.created_at = time.monotonic() - 100.0
            slot_b.fastlane = FastLaneMatch(
                rule_index=0, raw_address="10.0.0.9", label="registered",
                effective_tag="main", rank=1,
            )
            async with mgr._registry_lock:
                mgr._defer_unroutable(slot_b, evict_pending=False)

            # Non-vacuity: the tie above must not have made this resident
            # the designated victim (which would end grace for the WRONG
            # reason and make the assertion below vacuous either way).
            events_pre = _audit_events(boot, anchor.slot_id)
            assert "grace_designated_unload_target_break" not in events_pre, (
                "fixture failure: the anchor resident still became the "
                "designated victim -- the identity tie above did "
                "not work, so this test cannot isolate the Fast Lane gate"
            )

            # Non-vacuity: the notification DID fire (the field is set) --
            # proves the test scenario reaches the code under test at all.
            assert r.starvation_breakout_candidate is slot_b, (
                "fixture failure: _notify_grace_starvation_candidate never "
                "set the field -- this test cannot exercise the gate"
            )

            # THE ASSERTION: gated, must not break out. Bounded well under
            # grace_seconds so a wrongly-firing breakout is caught fast.
            await asyncio.sleep(max(grace_seconds - 1.0, 1.5))
            assert not task.done(), (
                "grace ended (task completed) on a Fast-Lane-registered "
                "candidate -- Fast Lane exclusion not honoured on "
                "the notified path"
            )
            events = _audit_events(boot, anchor.slot_id)
            assert "grace_starvation_breakout" not in events, (
                "registered-claim regression undetected: a registered claim broke a non-victim's "
                f"grace via the notified path. events={events}"
            )
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await mgr.shutdown()

    async def test_fresh_unregistered_candidate_is_not_notified(
        self, tmp_path
    ):
        """Fresh-candidate rule: a FRESH (just-created, NOT aged past max_other_model_wait_s)
        UNREGISTERED other-model slot fails routing (via the real
        _defer_unroutable) while a resident is in grace. Must NOT set the
        candidate -- the age check exists precisely so a not-yet-starved
        defer can't shorten anyone's grace.

        Guards against a missing age check (every fresh defer notifies): in that case
        r.starvation_breakout_candidate is set for this fresh slot
        and this test's assertion fails."""
        grace_seconds = 5
        max_other_model_wait_s = 3600.0  # deliberately large: "fresh" must
        # mean fresh under ANY realistic threshold, not just a coincidence
        # of this test's own timing.
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )
        try:
            r, handle = _resident(mgr, "m1", 59901, grace_seconds)
            anchor = Slot.new("m1", prompt="hi", thread_id="t1")
            anchor.state = SlotState.STAGED

            task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
            assert await _wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                timeout=5.0,
            ), "fixture failure: grace never entered"

            # FRESH (created_at left at "now"), UNREGISTERED, different
            # model -- fails routing via the REAL _defer_unroutable.
            slot_b = Slot.new("m2", prompt="hi", thread_id="t2")
            assert slot_b.fastlane is None  # non-vacuity precondition
            age_s = time.monotonic() - slot_b.created_at
            assert age_s < max_other_model_wait_s, (
                "fixture failure: slot_b is not actually fresh relative to "
                "the configured threshold"
            )
            async with mgr._registry_lock:
                mgr._defer_unroutable(slot_b, evict_pending=False)

            # THE ASSERTION: age check must have suppressed the notify.
            assert r.starvation_breakout_candidate is None, (
                "fresh-candidate regression undetected: a FRESH, not-yet-aged other-model slot "
                "was notified as a starvation candidate -- the age check is "
                "gone or inverted"
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await mgr.shutdown()

    async def test_same_model_candidate_is_not_notified(self, tmp_path):
        """Same-model-defer rule: an AGED, UNREGISTERED slot for the SAME model as the
        resident's own grace window fails routing (via the real
        _defer_unroutable). Must NOT set the candidate -- the starving
        population this mechanism exists for is, by construction, every
        OTHER model; a same-model defer is a same-model capacity question,
        not this resident's starvation signal.

        Guards against same-model residents being notified too: in that case
        r.starvation_breakout_candidate is set for this same-model slot and
        this test's assertion fails."""
        grace_seconds = 5
        max_other_model_wait_s = 0.2
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )
        try:
            r, handle = _resident(mgr, "m1", 59902, grace_seconds)
            anchor = Slot.new("m1", prompt="hi", thread_id="t1")
            anchor.state = SlotState.STAGED

            task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))
            assert await _wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor.slot_id),
                timeout=5.0,
            ), "fixture failure: grace never entered"

            # AGED, UNREGISTERED, SAME model ("m1") -- fails routing via the
            # REAL _defer_unroutable.
            slot_b = Slot.new("m1", prompt="hi", thread_id="t3")
            slot_b.created_at = time.monotonic() - 100.0
            assert slot_b.model_tag == r.model_tag  # non-vacuity precondition
            async with mgr._registry_lock:
                mgr._defer_unroutable(slot_b, evict_pending=False)

            # THE ASSERTION: same-model scope check must have suppressed it.
            assert r.starvation_breakout_candidate is None, (
                "same-model-defer regression undetected: a same-model deferred slot was "
                "notified as this resident's starvation candidate -- the "
                "r.model_tag != slot.model_tag scope check is gone"
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await mgr.shutdown()

    async def test_stale_candidate_does_not_leak_into_a_fresh_window(
        self, tmp_path
    ):
        """Stale-candidate rule: a candidate notified (via the real _defer_unroutable) and
        consumed in grace window 1 must not still be sitting on the
        resident when window 2 begins.

        Two parts, deliberately separated:
          (a) drives a REAL window 1 to completion via the real
              _defer_unroutable + the real breakout, proving the
              notify-and-consume mechanism genuinely works end to end
              (non-vacuity for the whole feature, not just this test).
          (b) re-creates the exact "stale leftover" shape a REMOVED
              grace-entry clear would produce -- reusing the SAME slot
              object window 1 already legitimately produced via
              _defer_unroutable (never fabricated, never queue.enqueue) --
              and asserts window 2's own entry clears it. This isolates the
              entry-clear specifically: window 1's own exit already clears
              the field on every real run (by design, redundant defense),
              so relying on two sequential real windows alone would not
              reliably detect a missing entry-clear. Re-arming the exact
              stale value right before window 2 is what makes ONLY the
              entry-clear responsible for the outcome under test.

        Guards against a missing clear on grace entry: in that case window 2's
        r.starvation_breakout_candidate still holds the stale slot and this
        test's assertion fails."""
        grace_seconds = 5
        max_other_model_wait_s = 0.2
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
        )

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )
        try:
            r, handle = _resident(mgr, "m1", 59903, grace_seconds)

            # --- window 1: real end-to-end notify + consume ---
            anchor1 = Slot.new("m1", prompt="hi", thread_id="t1")
            anchor1.state = SlotState.STAGED
            task1 = asyncio.create_task(mgr._serve_on_resident(r, anchor1, handle))
            assert await _wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor1.slot_id),
                timeout=5.0,
            ), "fixture failure: window 1 grace never entered"

            slot_other = Slot.new("m2", prompt="hi", thread_id="t2")
            slot_other.created_at = time.monotonic() - 100.0
            async with mgr._registry_lock:
                mgr._defer_unroutable(slot_other, evict_pending=False)

            await asyncio.wait_for(task1, timeout=grace_seconds)
            events1 = _audit_events(boot, anchor1.slot_id)
            assert "grace_starvation_breakout" in events1, (
                "fixture failure: window 1 never broke out on the real "
                "notified candidate -- cannot test the leak from a window "
                "that never legitimately had one"
            )

            # _requeue_after_backoff's background task (spawned inside
            # _defer_unroutable) lands slot_other back in the queue via
            # enqueue_head -- and with no real dispatcher running in this
            # unit-style test, nothing ever drains it back out. Left alone,
            # window 2's own BUFFER-SCAN fallback (_starvation_breakout_
            # candidate, kept deliberately for a waiter the dispatcher
            # hasn't attempted yet) would find it independently and break
            # window 2 for a reason that has nothing to do with the grace-entry clear -- that
            # would falsely look like a kill. Drain it so window 2's outcome
            # is attributable ONLY to the notified-candidate path re-armed
            # below.
            await _wait_until(
                lambda: slot_other in mgr.queue._staging
                or slot_other in mgr.queue._accept_buf,
                timeout=2.0,
            )
            async with mgr.queue._lock:
                if slot_other in mgr.queue._staging:
                    mgr.queue._staging.remove(slot_other)
                if slot_other in mgr.queue._accept_buf:
                    mgr.queue._accept_buf.remove(slot_other)

            # --- re-arm the SAME (real, _defer_unroutable-produced) slot,
            # simulating exactly what a removed grace-entry clear would
            # leave behind ---
            r.starvation_breakout_candidate = slot_other

            # --- window 2: fresh anchor, fresh thread, no new defer ---
            anchor2 = Slot.new("m1", prompt="hi", thread_id="t9-fresh")
            anchor2.state = SlotState.STAGED
            task2 = asyncio.create_task(mgr._serve_on_resident(r, anchor2, handle))
            assert await _wait_until(
                lambda: "grace_enter" in _audit_events(boot, anchor2.slot_id),
                timeout=5.0,
            ), "fixture failure: window 2 grace never entered"

            # THE ASSERTION: entry must have cleared it.
            assert r.starvation_breakout_candidate is None, (
                "stale-candidate regression undetected: a stale candidate from a prior grace "
                "window is still attached at the start of a fresh window -- "
                "the grace-entry clear is gone"
            )

            # Behavioural confirmation: no spurious immediate breakout
            # while we hold the observation window open.
            await asyncio.sleep(0.3)
            events2 = _audit_events(boot, anchor2.slot_id)
            assert "grace_starvation_breakout" not in events2, (
                f"window 2 broke out with no new defer -- stale leak. "
                f"events={events2}"
            )
        finally:
            for t in ("task1", "task2"):
                obj = locals().get(t)
                if obj is not None and not obj.done():
                    obj.cancel()
                    try:
                        await obj
                    except asyncio.CancelledError:
                        pass
            await mgr.shutdown()
