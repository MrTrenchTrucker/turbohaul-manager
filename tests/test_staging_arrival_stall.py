"""The staging-arrival stall fix (designated-victim fallback).

REPRO: when BOTH loaded residents are state_not_idle (actively serving
follow-ups) and neither is IDLE_EVICTABLE, the dispatcher's make-room path
calls `_lru_idle_unloadable` and gets None -- MAKE_ROOM_STARVED fires again
every 1s for the entire stall window. The idle-eviction guard correctly
prevents non-victims from being
evicted, but the DESIGNATED victim (the worst-ranked loaded resident under a
live strictly-outranking claim) has already had its grace broken by
`grace_designated_unload_target_break` -- it IS the
intended eviction target, yet `_lru_idle_unloadable` still declines it
because it filters on `IDLE_EVICTABLE` exclusively.

FIX (designated-victim fallback): `_lru_idle_unloadable` gains a fallback --
when no IDLE_EVICTABLE candidate exists, it returns the resident
`_is_designated_unload_target_locked` has already named the victim, even
if that resident is state_not_idle. Non-designated residents are structurally
untouched: `_is_designated_unload_target_locked` is False for them.

RED-FIRST: this test asserts on `_lru_idle_unloadable`'s return value
directly (a real behavioral gate, not a missing symbol). On the unfixed
tree it returns None (RED); after the fix it returns the
designated victim (GREEN). The `_starvation_reason` mirror is kept in
lockstep by the same patch; the drift test
(test_lru_idle_unloadable_and_starvation_reason_never_disagree) verifies
that independently.

The Fast Lane claim registered at staging-arrival is NOT reverted
-- the fix is purely additive to `_lru_idle_unloadable`. The staging-arrival
registration stays; the stall is resolved by making the eviction selector
aware of the already-designated victim, not by removing the claim.
"""
from __future__ import annotations

import asyncio
import ipaddress
import time

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.config import FastLaneTagRanks
from turbohaul.fastlane import CompiledRule, match_fastlane
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer
from turbohaul.slot import Slot


def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index,
        raw_address=raw_address,
        address=addr,
        container_name=None,
        match_addresses=frozenset({addr}),
        label=f"rule{index}",
        tag_ranks={"unclassified": 1},
    )


# claimant = rule_index 0 (highest priority, strictly outranks both victims)
# victim_a   = rule_index 1 (better-ranked victim)
# victim_b   = rule_index 2 (WORSE-ranked victim -- the designated victim)
_TABLE = [_rule(0, "10.0.0.1"), _rule(1, "10.0.0.2"), _rule(2, "10.0.0.3")]


def _boot_runtime(tmp_path):
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
            default_port_base=59710,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            max_parallel_sidecars=2,
            grace_seconds=5,
            max_grace_extensions=50,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=[
            FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(unclassified=1)),
            FastLaneRule(address="10.0.0.2", tag_ranks=FastLaneTagRanks(unclassified=1)),
            FastLaneRule(address="10.0.0.3", tag_ranks=FastLaneTagRanks(unclassified=1)),
        ]),
    )
    return boot, runtime


def _active_resident(tag, ip, *, grace_started=False):
    """A resident that is ACTIVE (state_not_idle) -- the stall condition.

    When ``grace_started=True`` the resident's GraceTimer is started (simulating
    grace was entered and then broken via grace_designated_unload_target_break
    and the loop exited, victim re-served by a follow-up -- the staging-arrival
    stall signature). ``in_grace_loop=False`` because the loop has exited.
    When ``grace_started=False`` the GraceTimer exists but was NEVER started
    (first-turn resident, mid-serve) -- must NOT be evicted by the fallback.
    """
    now = 1000.0
    r = Resident(
        model_tag=tag,
        resident_key=tag,
        state=ResidentState.ACTIVE,
        main_gpu=0,
        split_mode="none",
        reserved_need_mib=100,
        last_active_monotonic=now,
        rank_client_meta={"ip": ip},
    )
    if grace_started:
        r.grace = GraceTimer(grace_seconds=5, max_extensions=50)
        r.grace.start(f"t-{tag}", tag)
        # in_grace_loop defaults to False -- the grace loop exited, victim
        # was re-served by a follow-up.
    return r


def _claimant_slot(tag, ip):
    """A claimant slot matching a Fast Lane rule at the given index."""
    s = Slot.new(tag, prompt="hi", thread_id=f"t-{tag}", client_meta={"ip": ip})
    s.fastlane = match_fastlane(_TABLE, ip, {"ip": ip})
    return s


class TestLRUIdleUnloadableDesignatedVictimFallback:
    """RED on the unfixed tree; GREEN after the fix."""

    async def test_RED_unfixed_returns_none_when_both_state_not_idle(self, tmp_path):
        """Reproduces the stall signature: both residents are ACTIVE
        (state_not_idle), neither is IDLE_EVICTABLE. Pre-fix,
        `_lru_idle_unloadable` returns None -- MAKE_ROOM_STARVED -- because
        it only considers IDLE_EVICTABLE residents. After the fix it returns
        the designated victim (worst-ranked under a live claim)."""
        boot, runtime = _boot_runtime(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_table = lambda: _TABLE
        try:
            now = 1000.0
            # Two ACTIVE residents -- the stall condition. victim_b (10.0.0.3)
            # is worse-ranked (rule_index 2) vs victim_a (rule_index 1).
            # victim_b has grace_started=True (stall signature): grace was
            # entered and broken, loop exited, re-served by a follow-up --
            # this is the ONLY state the fallback should widen for.
            resident_a = _active_resident("modelA", "10.0.0.2")
            resident_b = _active_resident("modelB", "10.0.0.3", grace_started=True)
            mgr._residents["modelA"] = resident_a
            mgr._residents["modelB"] = resident_b

            # Register a live claim from 10.0.0.1 (rule_index 0, strictly
            # outranks both -- this is what the staging-arrival
            # registration produces).
            claimant = _claimant_slot("claimant", "10.0.0.1")
            async with mgr._registry_lock:
                mgr._register_fastlane_claim_locked(claimant, "claim_test")

            # Sanity: victim_b IS the designated victim under the claim.
            async with mgr._registry_lock:
                assert mgr._is_designated_unload_target_locked(resident_b) is True
                assert mgr._is_designated_unload_target_locked(resident_a) is False

            # THE ASSERTION: _lru_idle_unloadable must return the designated
            # victim (resident_b) even though it is state_not_idle.
            async with mgr._registry_lock:
                victim = mgr._lru_idle_unloadable()
            assert victim is resident_b, (
                f"Expected designated victim resident_b (modelB), got "
                f"{victim.model_tag if victim else 'None'}. On unfixed tree "
                f"this returns None (RED -- stall reproduces); on the fixed "
                f"tree it must return the designated victim (GREEN)."
            )

            # Negative case: a NON-designated resident must NOT be returned.
            # Remove the claim so neither is a designated victim.
            async with mgr._registry_lock:
                mgr._release_fastlane_claim_locked(claimant, "released")
                victim_no_claim = mgr._lru_idle_unloadable()
            assert victim_no_claim is None, (
                f"With no live claim, _lru_idle_unloadable must return None "
                f"(no IDLE_EVICTABLE resident, no designated victim) -- got "
                f"{victim_no_claim.model_tag if victim_no_claim else 'None'}. "
                f"The fix must NOT widen eligibility to ARBITRARY state_not_idle "
                f"residents, only to the designated victim."
            )
        finally:
            await mgr.shutdown()

    async def test_GREEN_designated_victim_returned_with_card_filter(self, tmp_path):
        """The designated-victim fallback must respect the same card/split
        filter as the IDLE_EVICTABLE path. If the designated victim is on the
        wrong card for a 'none'-split claimant targeting card 1, and there
        are no eligible idle candidates, the fallback must also decline."""
        boot, runtime = _boot_runtime(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_table = lambda: _TABLE
        try:
            resident_b = _active_resident("modelB", "10.0.0.3", grace_started=True)
            resident_b.main_gpu = 0  # on card 0
            mgr._residents["modelB"] = resident_b

            claimant = _claimant_slot("claimant", "10.0.0.1")
            async with mgr._registry_lock:
                mgr._register_fastlane_claim_locked(claimant, "claim_test")

            # Ask for main_gpu=1 (card 1) with split_mode='none'. The
            # designated victim is on card 0 -- the card filter should
            # prevent it from being returned.
            async with mgr._registry_lock:
                victim = mgr._lru_idle_unloadable(main_gpu=1, split_mode="none")
            assert victim is None, (
                f"Designated victim on card 0 must NOT be returned when "
                f"claimant targets card 1 with split_mode='none'. Got "
                f"{victim.model_tag if victim else 'None'}. The card/split "
                f"filter must apply to the fallback path too."
            )
        finally:
            await mgr.shutdown()

    async def test_RED_first_turn_resident_not_evicted(self, tmp_path):
        """The fallback must NOT fire for a designated victim that has NEVER
        entered grace (first-turn resident, mid-serve). Victims on
        their first turn are ACTIVE with grace_started=False -- the grace outer
        loop handles their normal completion, NOT the dispatcher
        fallback. With this test: victim_b is the designated victim under the
        claim, but grace_started=False (r.grace._started_at is None) -- the
        fallback must return None so the resident can complete normally."""
        boot, runtime = _boot_runtime(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._fastlane_table = lambda: _TABLE
        try:
            resident_a = _active_resident("modelA", "10.0.0.2")
            resident_b = _active_resident("modelB", "10.0.0.3")  # grace NOT started
            mgr._residents["modelA"] = resident_a
            mgr._residents["modelB"] = resident_b

            claimant = _claimant_slot("claimant", "10.0.0.1")
            async with mgr._registry_lock:
                mgr._register_fastlane_claim_locked(claimant, "claim_test")

            # victim_b IS designated (worst-ranked under claimant) -- sanity:
            async with mgr._registry_lock:
                assert mgr._is_designated_unload_target_locked(resident_b) is True

            # But grace_started=False -- the fallback must NOT fire:
            async with mgr._registry_lock:
                victim = mgr._lru_idle_unloadable()
            assert victim is None, (
                f"Resident with grace not started (first turn) must NOT be "
                f"returned by fallback -- got {victim.model_tag if victim else 'None'}. "
                f"The fallback must only fire for victims whose grace was ALREADY broken."
            )
        finally:
            await mgr.shutdown()
