"""The victim-skip change (designated victim skips grace) / the lapse-wake change (grace-lapse
wakes any parked make-room waiter). The existing
`grace_fastlane_breakout` check is left completely unchanged
(the full `test_grace_fastlane_breakout.py` suite,
passes unmodified alongside this file); the victim-skip change is added additively,
right before the grace `while` loop even starts, reusing
`_is_designated_unload_target_locked` as-is (no new selection logic).

The victim-skip change skips grace ENTIRELY for the designated victim, not just the wait.

The designated victim gets NO grace timer and NO idle-unload countdown:
both are surrendered the instant it is designated. Concretely, the victim
transitions into no GRACE state, stamps no `grace_started_at`, arms no
`GraceTimer` and emits no `grace_enter`. The only grace-family event on
its slot is `grace_designated_victim_skip`, and the slot leaves
ACTIVE -> POPPED directly, in near-zero wall-clock time instead of
holding to `grace_seconds`.

This works because the FSM table has a legal ACTIVE->POPPED edge
(in `fsm.py`), so GRACE is not a structurally required waypoint
out of `_serve_on_resident`: the grace ENTRY is gated, not only
the wait.

What the tests assert for the victim-skip change:
  - the elapsed time is near zero (the grace wait is skipped);
  - the distinct `grace_designated_victim_skip` audit event exists;
  - no GRACE state is ever entered for the victim, so no
    `grace_enter` marker exists (see also the dedicated
    victim-enters-no-grace test).

A non-victim resident is untouched: it still holds its
normal grace window.

The lapse-wake change is independent of designation: whenever any resident's grace loop
exits (natural lapse), `notify_all()` fires on the make-room signal, so
a parked make-room waiter wakes early instead of sleeping out its full
backoff.

Direct `_serve_on_resident` calls, same harness shape as
`test_grace_fastlane_breakout.py`'s own `TestServeOnResidentGraceFastlaneBreakout`
-- exercises exactly the loop this change touches without needing the full
dispatcher/GPU-reservation machinery.
"""
import asyncio
import time
from unittest.mock import MagicMock

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


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


def _boot_runtime(tmp_path, *, grace_seconds=5, fastlane_rules=None):
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
        queue=QueueConfig(safety_enabled=False, grace_seconds=grace_seconds,
                           max_grace_extensions=50, idle_hot_load_seconds=0),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=fastlane_rules or []),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
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


_RULES = [FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1))]


def _resident(mgr, model_tag, port, *, rank_client_meta, grace_seconds):
    handle = _make_fake_handle(model_tag, port)
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, handle=handle, port=port,
        grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=rank_client_meta,
        last_active_monotonic=time.monotonic(),
    )
    mgr._residents[model_tag] = r
    return r, handle


@pytest.mark.asyncio
class TestDesignatedVictimSkipsGrace:
    async def test_discriminator_designated_victim_skips_the_wait(self, tmp_path):
        """MUST FAIL ON UNMODIFIED CODE (before the victim-skip change): grace holds to the full
        deadline for every resident regardless of designation; no
        `grace_designated_victim_skip` audit event exists at all."""
        grace_seconds = 5
        # Designation requires a LIVE claim that STRICTLY outranks the
        # resident (the designation rule's precondition). The module-level _RULES has a
        # single entry, so both residents resolve to rule_index 0 and NOTHING could
        # outrank them -- there would be no legal way to designate a victim at all.
        # Scoped to this test so the siblings rule table is untouched: 9.9.9.9 takes
        # index 0 (the claimant), 1.2.3.4 moves to index 1 (both residents).
        _rules = [
            FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
        ]
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds, fastlane_rules=_rules)

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
        # Two loaded residents under the one rule, both ACTIVE. `victim` is
        # LEAST recently active -- _is_designated_unload_target_locked's own
        # tie-break direction (worst = LEAST recently active), so `victim`
        # is unambiguously the designated one.
        from turbohaul.manager import ResidentState
        victim, v_handle = _resident(
            mgr, "victim-model", 59901,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        victim.state = ResidentState.ACTIVE
        victim.last_active_monotonic = 1.0
        survivor, _ = _resident(
            mgr, "survivor-model", 59902,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        survivor.state = ResidentState.ACTIVE
        survivor.last_active_monotonic = 1000.0

        # The designation rule's precondition: a higher-priority request must actually be
        # waiting. Registered through the production path so this fixture cannot
        # drift from the claim shape _claim_is_live reads.
        claimant = Slot.new("claimant-model", prompt="hi", thread_id="claim")
        claimant.fastlane = FastLaneMatch(
            rule_index=0, raw_address="9.9.9.9", label="",
            effective_tag="main", rank=1,
        )
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")

        assert mgr._is_designated_unload_target_locked(victim) is True
        assert mgr._is_designated_unload_target_locked(survivor) is False
        # And the precondition is load-bearing, not decoration: drop the claim and
        # the victim stops being one (grace is unconditional in this state).
        mgr._release_fastlane_claim_locked(claimant, "test_control")
        assert mgr._is_designated_unload_target_locked(victim) is False
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")

        anchor = Slot.new("victim-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(victim, anchor, v_handle),
            timeout=grace_seconds - 0.5,
        )
        elapsed = time.monotonic() - started

        assert elapsed < grace_seconds - 1.0, (
            f"_serve_on_resident took {elapsed:.2f}s for the designated "
            f"victim -- the victim-skip change did not skip the grace wait"
        )
        events = _audit_events(boot, anchor.slot_id)
        # The victim skips the grace ENTRY, not just the wait, so no
        # grace_enter marker may appear in its events (asserted below).
        #
        # This relies on the FSM table having a legal ACTIVE->POPPED
        # edge: GRACE is not a structurally required waypoint out of
        # _serve_on_resident, so the victim has a way out of it
        # without passing through GRACE.
        #
        # If that edge were absent, GRACE would be a required waypoint
        # and the victim could skip only the wait, never the entry; an
        # assertion that "grace_enter" must still fire would then be the
        # right one. With the edge present, the absence of grace_enter
        # is the contract.
        #
        # The elapsed time and the distinct skip audit event are
        # asserted above, and the checks that follow
        # cover the other grace-family events.
        assert "grace_enter" not in events, (
            "Rule: the designated victim gets NO grace timer and NO "
            "idle-unload countdown; both are surrendered the instant it is "
            "designated, so no grace state is ever "
            "entered for the victim. It skips the grace ENTRY, not "
            f"just the wait, so no grace_enter marker may exist, events={events!r}"
        )
        assert "grace_designated_victim_skip" in events
        assert "grace_fastlane_breakout" not in events
        assert "grace_starvation_breakout" not in events

        await mgr.shutdown()

    async def test_negative_control_non_victim_holds_normal_grace(self, tmp_path):
        """★ NEGATIVE CONTROL -- must PASS both before and after the victim-skip change. A solo,
        UNRESOLVABLE (no fastlane match) resident is never the designated
        victim (unregistered ranks below all registered ones) -- it
        must still hold its normal grace window."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds, fastlane_rules=_RULES)

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
        from turbohaul.manager import ResidentState
        solo, handle = _resident(
            mgr, "solo-model", 59903,
            rank_client_meta=None,  # unresolvable -- never a candidate
            grace_seconds=grace_seconds,
        )
        solo.state = ResidentState.ACTIVE

        assert mgr._is_designated_unload_target_locked(solo) is False

        anchor = Slot.new("solo-model", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(solo, anchor, handle),
            timeout=grace_seconds + 2.0,
        )
        elapsed = time.monotonic() - started

        events = _audit_events(boot, anchor.slot_id)
        assert "grace_designated_victim_skip" not in events
        assert elapsed >= grace_seconds - 1.0, (
            f"grace exited early ({elapsed:.2f}s) for a non-victim resident"
        )

        await mgr.shutdown()


@pytest.mark.asyncio
class TestGraceLapseWakesAParkedWaiter:
    async def test_notify_reaches_a_waiter_before_its_own_backoff(self, tmp_path):
        """A coroutine parked in `_wait_for_park_or_timeout` with a LONG
        backoff must wake EARLY when a resident's grace loop exits (natural
        lapse) -- proving the lapse-wake change's `notify_all()` actually reaches a real
        waiter on the real `_make_room_signal` Condition, not just that the
        code runs without raising. MUST FAIL ON UNMODIFIED CODE (before the lapse-wake change):
        the waiter would sleep out its full long backoff instead."""
        grace_seconds = 1  # short: natural lapse well inside the test timeout
        long_backoff_s = 30.0  # must NOT be waited out -- proves the wake, not a timeout race
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds, fastlane_rules=_RULES)

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
        handle = _make_fake_handle("m1", 59904)
        r = Resident(
            model_tag="m1", resident_key="m1", handle=handle, port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED
        # No fastlane match -- irrelevant to the lapse-wake change (fires on any loop exit,
        # not gated on designation), keeps this test isolated from the victim-skip change.

        waiter_task = asyncio.create_task(mgr._wait_for_park_or_timeout(long_backoff_s))
        serve_task = asyncio.create_task(mgr._serve_on_resident(r, anchor, handle))

        started = time.monotonic()
        await asyncio.wait_for(waiter_task, timeout=grace_seconds + 3.0)
        elapsed = time.monotonic() - started

        assert elapsed < long_backoff_s - 5.0, (
            f"waiter took {elapsed:.2f}s to wake -- notify_all() from the lapse-wake change "
            f"never reached it (fell back to the {long_backoff_s}s timeout)"
        )

        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        await mgr.shutdown()
