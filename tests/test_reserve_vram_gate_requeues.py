"""Requeue on a cross-resident VRAM gate miss -- the SECOND site
of its kind: `_reserve_and_start_locked`'s cross-resident VRAM
gate re-check (the "LAST-RESORT refuse" for the narrow residual race:
a count-cap in-band reserve whose freed card has not yet released
VRAM, or a probe that flipped the card between the pre-check and here) now
REQUEUES instead of refusing. Distinct code site from
`_defer_unroutable`'s own exhaustion-branch removal (see
the busy-defer budget tests) -- this site is requeued rather than
refused: a transient VRAM miss should wait for memory to free,
not fail the request; the slot just waits a moment for VRAM to
become free.

Mutant check: reverting manager.py's
``if not self._vram_admits_locked(...)`` branch back to its OLD body
(``_fail_completion_future`` + ``_release_fastlane_claim_locked(slot,
"unroutable_exhausted")`` -- preserved verbatim, uncalled, as
``_legacy_reserve_vram_gate_refusal``) turns this file's core assertion
RED (the future resolves with a
VramOverCommitError instead of staying pending). That mutant is the
one the core assertion is built to catch.
"""
from __future__ import annotations

import asyncio

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState

_RULES = [FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1))]


@pytest.fixture
def mgr(tmp_path):
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return TurbohaulManager(boot, runtime)


def _fastlane_slot(model_tag="claimant-model"):
    # Fastlane-eligible on purpose (note: a bare Slot.new(...) has
    # fastlane=None, so _fastlane_claim_key silently no-ops claim
    # registration -- constructing this directly avoids that trap and lets
    # this test also verify the claim-registry side of the requeue.
    return Slot(
        slot_id="reap-reserve-gate-slot",
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        thread_id="t1",
        client_meta={"ip": "10.0.0.1"},
        fastlane=FastLaneMatch(
            rule_index=0, raw_address="10.0.0.1", label="",
            effective_tag="main", rank=1,
        ),
    )


async def _drain_bg(mgr):
    tasks = [t for t in list(mgr._bg_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
class TestReserveVramGateRequeuesInsteadOfRefusing:
    async def test_gate_miss_requeues_via_the_defer_unroutable_funnel(self, mgr):
        """Force the cross-resident VRAM gate re-check to miss (as it would
        on the narrow settle-window race described in the module docstring) and drive
        the REAL _reserve_and_start_locked, not a stub."""
        mgr._vram_admits_locked = lambda *a, **k: False
        mgr._resolve_placement_locked = lambda *a, **k: (1024, 1, 0, "none", 0, False, False)
        slot = _fastlane_slot()
        slot.completion_future = asyncio.get_event_loop().create_future()

        try:
            async with mgr._registry_lock:
                r = await mgr._reserve_and_start_locked(slot)
            await _drain_bg(mgr)

            assert r is None, "unchanged contract: still returns None on a gate miss"
            assert not slot.completion_future.done(), (
                "the cross-resident VRAM gate must REQUEUE, not "
                "refuse -- reverting to the old refuse-and-fail body "
                "(_legacy_reserve_vram_gate_refusal) is the mutant this "
                "assertion catches"
            )
            assert slot._vram_defer_count == 1, (
                "must route through _defer_unroutable(evict_pending=True) "
                "-- the same funnel _defer_unroutable's other callers use, "
                "not a bespoke requeue"
            )
            rows = mgr.fastlane_claims_snapshot()
            assert len(rows) == 1, (
                "a fastlane claim must be registered, same as any other "
                "defer -- proves this reuses the existing single "
                "funnel rather than a parallel, undiscoverable path"
            )
            assert rows[0]["reason"] == "make_room_starved_vram", (
                "must reuse evict_pending's own claim reason -- this is "
                "the evict-pending regime (an eviction was already begun "
                "for this reserve attempt), not the busy regime"
            )
        finally:
            for t in list(mgr._bg_tasks):
                t.cancel()
            await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)

    async def test_gate_admits_still_returns_a_resident_unchanged(self, mgr):
        """Non-regression control: the SUCCESS path (gate admits) is
        untouched by this fix -- only the miss branch changed. Without
        this, a broken mutant that always requeues (even on admit) would
        slip through undetected."""
        mgr._vram_admits_locked = lambda *a, **k: True
        mgr._resolve_placement_locked = lambda *a, **k: (1024, 1, 0, "none", 0, False, False)
        slot = _fastlane_slot(model_tag="admits-fine")

        try:
            async with mgr._registry_lock:
                r = await mgr._reserve_and_start_locked(slot)
            assert r is not None, "gate admits -> a real Resident must still be created"
            assert getattr(slot, "_vram_defer_count", 0) == 0, (
                "the success path must never touch the defer counter"
            )
        finally:
            for t in list(mgr._bg_tasks):
                t.cancel()
            await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)
