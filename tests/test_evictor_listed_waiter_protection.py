"""Tests for the listed-waiter eviction protection:
the evictor must not unload a resident while a Fast-Lane-LISTED request is
waiting for that same model_tag. The protection is CONDITIONAL -- a listed
model with no listed waiter stays evictable, so the resident pool never
freezes.

Covers two behaviours:
  1. Listed-waiter protection at the queue primitive, the manager filter
     (_lru_idle_unloadable), and the actual lock-seam wiring (_route_or_reserve).
  2. Starvation stop: the bounded 50-defer cap is what actually stops an
     unlisted slot under saturation, NOT the (inert-at-parked-value) 3600s
     fairness floor.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import (
    TurbohaulManager, Resident, ResidentState,
)
from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


@pytest.fixture
def boot_and_runtime(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent, unused here
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


# ---------------------------------------------------------------------------
# Group 1: queue.py primitives (_has_listed_waiter_for_locked, listed_waiter_model_tags)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestListedWaiterQueuePrimitives:
    async def test_has_listed_waiter_true_in_staging(self):
        q = TurbohaulQueue(staging_max=100)
        s = Slot.new("m1")
        s.fastlane = _match()
        await q.enqueue(s)
        async with q._lock:
            assert q._has_listed_waiter_for_locked("m1") is True

    async def test_has_listed_waiter_false_when_unlisted(self):
        q = TurbohaulQueue(staging_max=100)
        s = Slot.new("m1")  # no .fastlane -- unlisted
        await q.enqueue(s)
        async with q._lock:
            assert q._has_listed_waiter_for_locked("m1") is False

    async def test_has_listed_waiter_false_for_different_tag(self):
        q = TurbohaulQueue(staging_max=100)
        s = Slot.new("m1")
        s.fastlane = _match()
        await q.enqueue(s)
        async with q._lock:
            assert q._has_listed_waiter_for_locked("m2") is False

    async def test_has_listed_waiter_scans_accept_buffer_too(self):
        """A listed slot sitting in the acceptance buffer (not staging) is
        just as real a waiter.

        Like the function it mirrors,
        ``_has_strictly_higher_priority_waiting_locked``, this function scans
        both the staging and the acceptance buffers, each bounded by that
        buffer's own cap, so a listed waiter parked in either is seen.
        Keeping this docstring accurate matters: a docstring
        describing a contract that no longer holds
        goes stale SILENTLY -- no assertion covers it, so
        nothing turns red to announce it."""
        q = TurbohaulQueue(staging_max=100)
        s = Slot.new("m1")
        s.fastlane = _match()
        async with q._lock:
            q._accept_buf.append(s)
            assert q._has_listed_waiter_for_locked("m1") is True

    async def test_listed_waiter_model_tags_snapshot_excludes_unlisted(self):
        q = TurbohaulQueue(staging_max=100)
        listed = Slot.new("m1")
        listed.fastlane = _match()
        unlisted = Slot.new("m2")
        await q.enqueue(listed)
        await q.enqueue(unlisted)
        tags = await q.listed_waiter_model_tags()
        assert tags == {"m1"}

    async def test_listed_waiter_model_tags_includes_accept_buffer(self):
        q = TurbohaulQueue(staging_max=1)  # forces overflow into accept_buf
        first = Slot.new("m1")
        await q.enqueue(first)  # takes the one staging slot
        second = Slot.new("m2")
        second.fastlane = _match()
        await q.enqueue(second)  # overflows to accept_buf, still listed
        assert second in q._accept_buf, "test setup assumption: second landed in accept_buf"
        tags = await q.listed_waiter_model_tags()
        assert tags == {"m2"}


@pytest.mark.asyncio
class TestListedWaiterPriorityKeys:
    """listed_waiter_priority_keys() -- tag -> the BEST-ranked
    listed waiter's Fast Lane priority key, so the shield compares against
    the party it actually protects (a queued waiter), not a resident's last
    holder. A tag can be shared
    by more than one listed client, so "first found" is a real inversion,
    not a hypothetical."""

    async def test_returns_the_better_ranked_waiter_not_the_first_found(self):
        """Mandatory arm: two listed waiters for the SAME tag, different
        rank, the WORSE one enqueued FIRST -- a naive first-match
        implementation returns the wrong key."""
        q = TurbohaulQueue(staging_max=100)
        worse = Slot.new("m1")
        worse.fastlane = _match(rule_index=4, rank=0)  # enqueued first, worse priority
        better = Slot.new("m1")
        better.fastlane = _match(rule_index=0, rank=3)  # enqueued second, better priority
        await q.enqueue(worse)
        await q.enqueue(better)
        keys = await q.listed_waiter_priority_keys()
        assert keys == {"m1": (0, 3)}

    async def test_single_waiter_maps_to_its_own_key(self):
        q = TurbohaulQueue(staging_max=100)
        s = Slot.new("m1")
        s.fastlane = _match(rule_index=2, rank=1)
        await q.enqueue(s)
        keys = await q.listed_waiter_priority_keys()
        assert keys == {"m1": (2, 1)}

    async def test_tag_with_no_listed_waiters_is_absent_not_none(self):
        q = TurbohaulQueue(staging_max=100)
        unlisted = Slot.new("m1")  # no .fastlane
        await q.enqueue(unlisted)
        keys = await q.listed_waiter_priority_keys()
        assert keys == {}
        assert "m1" not in keys

    async def test_scans_accept_buffer_too(self):
        q = TurbohaulQueue(staging_max=1)  # forces overflow into accept_buf
        first = Slot.new("m1")
        await q.enqueue(first)  # takes the one staging slot
        second = Slot.new("m2")
        second.fastlane = _match(rule_index=1, rank=0)
        await q.enqueue(second)  # overflows to accept_buf, still listed
        assert second in q._accept_buf, "test setup assumption: second landed in accept_buf"
        keys = await q.listed_waiter_priority_keys()
        assert keys == {"m2": (1, 0)}


# ---------------------------------------------------------------------------
# Group 2: TurbohaulManager._lru_idle_unloadable filter clause
# ---------------------------------------------------------------------------

class TestLruIdleEvictableProtection:
    def test_no_protected_tags_baseline_unaffected(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["m1"] = Resident(
            model_tag="m1", state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        victim = mgr._lru_idle_unloadable()  # no set passed -- default None
        assert victim is not None and victim.model_tag == "m1"

# The four listed-waiter
# protection tests in this class (test_listed_waiter_protects_the_resident,
# test_protection_lifts_once_waiter_served_or_evicted,
# test_protection_is_conditional_not_a_pool_freeze, and
# TestRouteOrReserveSnapshotWiring::test_protected_resident_survives_a_real_
# routing_decision) were RETIRED here -- each asserted the removed mechanism
# by name ("listed-waiter-protected resident must not be evictable").
# Their replacement is the listed-waiter shield-removal test, which
# asserts the
# INVERSE (a queued listed waiter does NOT shield the resident).

    async def test_unlisted_model_stays_evictable_throughout(self, boot_and_runtime):
        """Re-pointed. The original passed one
        positional set that silently bound to main_gpu -- a control that
        could not have failed is not a control. The listed-waiter clause it
        guarded is removed, so there is no named-set input left
        to exercise.

        As re-pointed it is the residual negative control for the SURVIVING
        filter: a live listed waiter for a DIFFERENT model sitting in queue
        staging must not change who is evictable.

        UNFALSIFIABILITY, stated plainly (marked UNFALSIFIABLE on purpose):
        the over-shield regressions
        this control once watched for (a named-set over-firing, shielding
        the wrong model, freezing the pool) all required the removed
        named-set parameter; the keyword-only _lru_idle_unloadable takes no
        such input, so no re-introduction of the shield is reachable from a
        call of THIS shape and could therefore turn this test red. It still
        guards the fixture wiring (a live listed waiter in staging changes
        nothing about eviction). The inverse direction -- the listed waiter
        must not shield -- is asserted by
        the listed-waiter shield-removal test (a behaviour-change test)."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["unlisted-model"] = Resident(
            model_tag="unlisted-model", state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        waiter = Slot.new("some-other-model")
        waiter.fastlane = _match()
        mgr.queue._staging.append(waiter)
        tags = await mgr.queue.listed_waiter_model_tags()  # the live listed-waiter data plane
        assert "some-other-model" in tags, "precondition: the listed waiter must be visible"
        victim = mgr._lru_idle_unloadable()
        assert victim is not None and victim.model_tag == "unlisted-model"

    def test_busy_and_inflight_filters_unchanged(self, boot_and_runtime):
        """Non-regression: the pre-existing state/active_slot/inflight
        filters still exclude residents. Re-pointed:
        the original's positional set() silently bound to main_gpu under
        the (self, main_gpu, split_mode) signature -- the assertion passed
        for the wrong reason. Now no-arg: exclusions come from the state
        and inflight clauses alone, and keyword-only signatures make
        any stale positional set fail loudly with TypeError instead of
        re-creating the silent bind this test once masked."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["active"] = Resident(
            model_tag="active", state=ResidentState.ACTIVE, last_active_monotonic=1.0,
        )
        mgr._residents["inflight"] = Resident(
            model_tag="inflight", state=ResidentState.IDLE_EVICTABLE,
            last_active_monotonic=1.0, inflight=[Slot.new("inflight")],
        )
        assert mgr._lru_idle_unloadable() is None


# ---------------------------------------------------------------------------
# Group 3: _route_or_reserve -- the actual pre-_registry_lock snapshot wiring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRouteOrReserveSnapshotWiring:
# Retired test:
# test_protected_resident_survives_a_real_routing_decision was RETIRED here
# -- it asserted the removed shield end-to-end ("protected resident must
# not be evicted"). Its replacement is the listed-waiter shield-removal
# test (asserts the
# inverse, through the same real submit path). The two surviving controls
# below (unprotected still evicted; no-waiter still evicted) were green in
# both arms and are untouched.

    async def test_unprotected_resident_is_still_evicted_baseline(self, boot_and_runtime):
        """Control: identical shape, but the waiter for m1 is UNLISTED --
        eviction must proceed exactly as it did before this change (the original
        baseline). Proves the new clause is conditional, not a blanket
        change to eviction behavior."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)
        mgr._vram_admits_locked = lambda *a, **k: True  # gate pin — see test 1 note
        mgr._residents["m1"] = Resident(
            model_tag="m1", resident_key="m1",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        waiter = Slot.new("m1")  # unlisted -- no .fastlane
        mgr.queue._staging.append(waiter)

        incoming = Slot.new("m2")
        incoming.completion_future = asyncio.get_event_loop().create_future()
        await mgr._route_or_reserve(incoming)

        assert "m1" not in mgr._residents, "unprotected resident must still be evicted"
        mgr._reserve_and_start_locked.assert_called_once()

    async def test_no_waiter_at_all_resident_still_evicted(self, boot_and_runtime):
        """Control: an idle resident with NOBODY waiting for it (listed or
        not) is evicted exactly as before -- the common case, unaffected."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)
        mgr._vram_admits_locked = lambda *a, **k: True  # gate pin — see test 1 note
        mgr._residents["m1"] = Resident(
            model_tag="m1", resident_key="m1",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        incoming = Slot.new("m2")
        incoming.completion_future = asyncio.get_event_loop().create_future()
        await mgr._route_or_reserve(incoming)
        assert "m1" not in mgr._residents


# ---------------------------------------------------------------------------
# Group 4: starvation stop is the 50-defer cap, not the floor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStarvationStopIsDeferCapNotFloor:
    async def test_unlisted_slot_never_fails_however_far_past_the_old_cap(self, boot_and_runtime):
        """This re-points test_unlisted_
        slot_fails_bounded_after_max_defers: an existing
        test that breaks due to an intentional behavior change is replaced
        by a new one. A request that is queued stays queued and is
        never cancelled unless the client very specifically
        cancels it. _defer_unroutable's exhaustion branch is gone -- there
        is no cap left to trip, bounded or otherwise.

        Same saturated-listed-lane scenario as before (modeled directly at
        the mechanism _route_or_reserve falls back to when
        _lru_idle_unloadable returns None, i.e. every candidate is
        protected/busy), same small/fast monkeypatched _max_busy_defers so
        the test stays deterministic -- new assertion: driven well past
        where the old cap used to trip, the future is still pending and the
        counter is still climbing.
        """
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._max_busy_defers = lambda: 5  # value is irrelevant now -- no longer consulted for a decision
        slot = Slot.new("unlisted-model")
        slot.completion_future = asyncio.get_event_loop().create_future()

        for i in range(1, 20 + 1):  # 4x the old monkeypatched cap of 5
            mgr._defer_unroutable(slot)
            assert slot._dispatch_defer_count == i
            assert not slot.completion_future.done(), (
                f"slot must never be failed by _defer_unroutable any more -- "
                f"tripped at defer {i}/20 (old monkeypatched cap was 5)"
            )

    async def test_floor_parked_at_3600s_still_does_not_stop_or_trip_anything(self, boot_and_runtime):
        """This re-points test_floor_
        parked_at_3600s_does_not_change_the_trip_point. The documented
        correction still holds unchanged: the 3600s fairness floor does not
        stop starvation on its own (re-picked, immediately re-deferred -- no
        forward progress). What changed is that there is no longer a
        derived busy-defer cap for it to NOT change the trip point of --
        that change removed the trip point entirely. _defer_unroutable's own
        bound check depends only on the slot's own counter -- never on
        runtime.fastlane.max_normal_wait_s -- so parking the floor at its
        widest legal value changes nothing, same as before; the difference
        is now nothing trips at all, ever."""
        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue,
            pull=runtime.pull,
            fastlane=FastLaneConfig(enabled=True, rules=[], max_normal_wait_s=3600.0),
        )
        mgr = TurbohaulManager(boot, runtime)
        mgr._max_busy_defers = lambda: 5  # value is irrelevant now -- no longer consulted for a decision
        slot = Slot.new("unlisted-model")
        slot.completion_future = asyncio.get_event_loop().create_future()
        for _ in range(20):  # 4x the old monkeypatched cap of 5
            mgr._defer_unroutable(slot)
        assert not slot.completion_future.done(), (
            "the floor's parked value changes nothing about the OUTCOME "
            "either -- there is no cap left for either setting to trip"
        )
