"""Grace AND idle timer never render on the FE — both status_snapshot()
grace/idle_hot fields are always None because they read manager-level scalars
(self.grace, self._idle_handle, self._idle_expires_at) that are PERMANENTLY
unwritten: their only write sites lived inside _process_slot (since retired).

Frontend impact: the resident-synthesis module builds its model from active/loading/
grace/idle_hot, while api.ts holds the live residents[] array. When the BE
returns None for all four, the synthesised model is EMPTY while residents[]
is FULL — the two halves of the UI disagree. Observed symptom: the frontend
flickers during model swaps or resident swaps.

ROOT CAUSE for BOTH surfaces:
- grace_info reads self.grace, whose only start() was in _process_slot.
  The live per-resident grace timer is maintained by _serve_on_resident
  (r.grace.start(...), fired on every turn completion).
- idle_info reads self._idle_handle/self.idle, whose only write sites are
  _set_idle_holder (called from _process_slot) and self.idle.start().
  The live per-resident idle state is maintained by _serve_on_resident
  (r.idle_expires_at = time.monotonic() + idle_window).

FIX: _resolve_top_level_grace_resident + _resolve_top_level_idle_resident
(template: _resolve_top_level_active_slot) resolve the representative
resident's state instead of the retired manager-level scalars.

These tests document the fix so future maintainers know exactly why they
exist and what breaks in the product if they go red: a frontend blip / UI
half-truth that looks like "lag" or "disconnect" to the user.
"""
import asyncio

import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime,
    drive_to_active,
    high_vram,
    make_fakes,
    resident_for,
    seed_manifest,
)
from turbohaul.manager import TurbohaulManager


async def _drive_real_turn_to_grace(tmp_path, *, grace_seconds=5.0, idle_hot_load_seconds=10.0):
    """Drive ONE resident through admission -> ACTIVE -> turn-complete via the
    shared fixture, so its GraceTimer is populated by production code
    (_serve_on_resident's r.grace.start(...)), never hand-constructed."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=grace_seconds,
        idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
        gate.set()
        await asyncio.wait_for(task, timeout=5.0)
    return mgr, resident_for(mgr, "m1")


@pytest.mark.asyncio
async def test_grace_info_renders_after_real_turn_completion(tmp_path):
    """After a real turn completes with no follow-up, status_snapshot()["grace"]
    must be non-None with a sane remaining_s. The FE draws nothing if this is
    None — the user sees the grace countdown silently absent.

    WHY THIS EXISTS: grace_info was always None because it read self.grace,
    whose only start() call site lived inside _process_slot (since retired).
    The live grace timer is on the resident, not the manager singleton.
    If this test goes red: the FE grace countdown is broken (blip on swap)."""
    mgr, r = await _drive_real_turn_to_grace(tmp_path, grace_seconds=5.0)
    try:
        assert r.grace is not None and not r.grace.expired(), (
            "sanity: a real turn completion must leave a live, unexpired "
            "GraceTimer on the resident -- otherwise this test measures nothing"
        )
        snap = mgr.status_snapshot()
        grace = snap.get("grace")
        assert grace is not None, (
            "status_snapshot()['grace'] must be non-None after a real "
            "turn completion -- the FE has nothing to draw otherwise"
        )
        assert 0 < grace["remaining_s"] <= 5, (
            f"grace remaining_s must be in (0, 5], got {grace['remaining_s']}"
        )
        assert grace["thread_id_prefix"] == "t1"[:8]
        assert grace["model_tag"] == "m1"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_idle_hot_info_renders_after_grace_expires(tmp_path):
    """After a real turn completes and the grace window passes, the resident
    transitions to IDLE_EVICTABLE and status_snapshot()["idle_hot"] must be
    non-None with a sane remaining_s. The FE draws nothing if this is None —
    the user silently loses the idle-hot countdown.

    WHY THIS EXISTS: idle_info was always None because it read
    self._idle_handle / self._idle_expires_at, whose only write site
    (_set_idle_holder) was called exclusively from _process_slot (since
    retired). The live idle state is on the resident, not the manager singleton.
    If this test goes red: the FE idle-hot countdown is broken (blip on swap).
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path, grace_seconds=5.0)
    try:
        # Wait for grace to expire so the resident transitions to IDLE_EVICTABLE
        # (grace_seconds=5, so 6s is enough). The resident must then have a
        # non-None idle_expires_at set by _serve_on_resident on the IDLE flip.
        await asyncio.sleep(6.0)
        snap = mgr.status_snapshot()
        idle_hot = snap.get("idle_hot")
        assert idle_hot is not None, (
            "status_snapshot()['idle_hot'] must be non-None after grace "
            "expires and the resident becomes IDLE_EVICTABLE -- the FE has "
            "nothing to draw otherwise"
        )
        assert 0 < idle_hot["remaining_s"] <= 10, (
            f"idle_hot remaining_s must be in (0, 10], got {idle_hot['remaining_s']}"
        )
        assert idle_hot["model_tag"] == "m1"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_residents_non_empty_implies_at_least_one_field_populated(tmp_path):
    """Invariant: when residents[] is non-empty, at least one of
    active/loading/grace/idle_hot must be non-None. If ALL are None while
    residents[] is populated, the FE synthesised model is EMPTY while the
    resident list is FULL — the two halves of the UI disagree. This is the
    frontend blip observed in use.

    WHY THIS EXISTS: Before the fix, all four fields were ALWAYS None at cap>=2
    because their only write sites lived inside _process_slot (since retired).
    The live state is per-resident, not on the manager singleton.
    If this test goes red: the synthesised model and residents[] disagree,
    producing a UI blip that looks like lag or disconnect to the user.
    """
    mgr, r = await _drive_real_turn_to_grace(tmp_path, grace_seconds=5.0)
    try:
        snap = mgr.status_snapshot()
        active = snap.get("active")
        loading = snap.get("loading")
        grace = snap.get("grace")
        idle_hot = snap.get("idle_hot")

        residents = [r for r in mgr._model_residents() if r.state.value != "DEAD"]
        assert len(residents) > 0, (
            "sanity: at least one resident must be live for this invariant "
            "to be meaningful"
        )

        at_least_one = any(
            field is not None
            for field in (active, loading, grace, idle_hot)
        )
        assert at_least_one, (
            "residents[] is non-empty but ALL of active/loading/"
            "grace/idle_hot are None — the synthesised model is EMPTY while "
            "the resident list is FULL. This is the frontend blip."
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_grace_info_suppressed_while_active(tmp_path):
    """Control: while a serve holds the engine (active_info is non-None),
    grace_info must be None — the FE must never render a mid-prefill 'unload
    in Ns' countdown. This is the existing behaviour, preserved
    by the resident-based resolution.

    WHY THIS EXISTS: preserving a pre-existing invariant. If this test goes
    red: the FE will render a misleading mid-prefill countdown."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=5.0
    )
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            snap = mgr.status_snapshot()
            assert snap["grace"] is None, (
                "grace_info must be None while a serve holds the engine "
                "(active_info is non-None) -- mid-prefill countdown is a lie"
            )
        finally:
            await mgr.shutdown()
