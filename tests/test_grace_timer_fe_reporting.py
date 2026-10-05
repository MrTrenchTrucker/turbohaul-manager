"""Grace timer must render on the FE: status_snapshot()["grace"] was
always None because it read self.grace, whose only start() call site lived
inside _process_slot (manager.py:12655), which has been retired. The live per-resident
grace timer is maintained by _serve_on_resident (r.grace.start(...) at
manager.py:10965, fired on every turn completion regardless of cap).

This test boots at max_parallel_sidecars=2 EXPLICITLY (the env lies: the
deployment may set TURBOHAUL_MAX_PARALLEL=2, so a test that does not set it
measures the wrong config while looking correct), drives a real turn to
completion via the shared fixture, and asserts status_snapshot()["grace"] is
non-None with a sane remaining_s while the resident is in GRACE. The state
ARISES from real traffic — NOT hand-set.

RED->GREEN watched: base (before fix) grace_info is None forever; after fix
grace_info is populated from the resident's grace timer.
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


async def _drive_real_turn_to_grace(tmp_path, *, grace_seconds=5.0):
    """Drive ONE resident through a real admission -> ACTIVE -> turn-complete
    lifecycle via the shared fixture, so its GraceTimer is populated by
    production code (_serve_on_resident's r.grace.start(...)), never
    hand-constructed. Returns (mgr, resident) with the manager's dispatcher
    already running in the background and the request's own task already
    awaited to completion.
    """
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=grace_seconds
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
    """The acceptance criterion: after a real turn completes, the resident
    enters GRACE and status_snapshot()["grace"] must be non-None with a sane
    remaining_s. The state must ARISE from real traffic, NOT be hand-set."""
    mgr, r = await _drive_real_turn_to_grace(tmp_path, grace_seconds=5.0)
    try:
        # Sanity: the resident's own grace timer is live (proof the turn
        # completed through production code, not a hand-set scalar).
        assert r.grace is not None and not r.grace.expired(), (
            "sanity: a real turn completion must leave a live, unexpired "
            "GraceTimer on the resident -- otherwise this test measures nothing"
        )

        # The actual assertion: status_snapshot()["grace"] must be non-None.
        snap = mgr.status_snapshot()
        grace = snap.get("grace")
        assert grace is not None, (
            "status_snapshot()['grace'] must be non-None after a real "
            "turn completion -- the FE has nothing to draw otherwise"
        )

        # remaining_s must be sane (positive, <= grace_seconds).
        assert 0 < grace["remaining_s"] <= 5, (
            f"grace remaining_s must be in (0, 5], got {grace['remaining_s']}"
        )

        # thread_id and model_tag must match the completed turn.
        assert grace["thread_id_prefix"] == "t1"[:8]
        assert grace["model_tag"] == "m1"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_grace_info_suppressed_while_active(tmp_path):
    """Control: while a serve holds the engine (active_info is non-None),
    grace_info must be None — the FE must never render a mid-prefill 'unload
    in Ns' countdown. This is the existing SPEC-V2 behaviour, preserved."""
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
            # Drive to ACTIVE but do NOT complete the turn (gate stays unset).
            await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
            snap = mgr.status_snapshot()
            assert snap["grace"] is None, (
                "grace_info must be None while a serve holds the engine "
                "(active_info is non-None) -- mid-prefill countdown is a lie"
            )
        finally:
            await mgr.shutdown()
