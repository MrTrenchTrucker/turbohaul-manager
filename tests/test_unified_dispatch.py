"""cap=1 ACCEPTANCE TESTS for the unified dispatcher.

The acceptance criterion, which this file exists to measure exactly:

    The old single-slot path stays in the code, marked as dead code. The default is
    max_parallel_sidecars=2 or more, and when that setting is 1 dispatch is still the
    parallel path, but only one sidecar is spawned at a time, after the dispatcher
    unification.

So "done" is not "cap=1 boots". It is: cap=1 runs THE PARALLEL PATH, spawns exactly ONE
sidecar at a time, and QUEUES the rest rather than refusing or dropping them.

WHY THESE COULD NOT BE WRITTEN BEFORE THE UNIFICATION: at cap=1 `worker_loop` ran its legacy body,
which never creates a model-keyed `Resident` at all — pinned at three checkpoints by
the cap-unification golden-master test, and the root cause of a preemption gap
(`_model_residents()` excluded the legacy singleton, so fast-lane preemption was
structurally inert at cap=1). That combination — cap=1 AND the parallel dispatcher — was
not reachable before the unification, so it could not be tested.

EVERY test sets max_parallel_sidecars EXPLICITLY to 1. The environment lies about this
value (TURBOHAUL_MAX_PARALLEL is 2 in a typical deployment), so a test that does not set it
is measuring the wrong configuration while looking correct.
"""

import asyncio

import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime, seed_manifest, make_fakes, high_vram, wait_until,
)
from turbohaul.manager import TurbohaulManager


async def _drain(mgr, gates, tasks):
    for g in gates:
        g.set()
    for t in tasks:
        if t is not None:
            try:
                await asyncio.wait_for(t, timeout=10.0)
            except BaseException:
                pass
    mgr._worker_task.cancel()
    try:
        await mgr._worker_task
    except BaseException:
        pass


@pytest.mark.asyncio
async def test_one_resident_cap_runs_the_parallel_path_and_creates_a_real_model_keyed_resident(tmp_path):
    """THE UNIFICATION ITSELF. At cap=1, a served request must produce a REAL model-keyed
    Resident — proof the request went through `_dispatch_loop`, not the legacy body.

    This is the exact inversion of the pre-unification golden master, which pinned
    `_model_residents() == []` at cap=1. It is therefore RED on any tree without the unification, and
    it cannot pass by accident: the legacy path has no code that writes a model-keyed
    Resident at all.
    """
    boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=1, grace_seconds=5)
    assert runtime.queue.max_parallel_sidecars == 1, "explicit cap — the env lies about this"
    seed_manifest(boot, "m1", main_gpu=0)
    gate1 = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate1})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task1 = None
        try:
            task1 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p", thread_id="t1", client_meta={}))
            await wait_until(lambda: len(mgr._model_residents()) == 1, timeout=6.0)
            residents = mgr._model_residents()
            assert [r.model_tag for r in residents] == ["m1"], (
                f"cap=1 must create a real model-keyed resident for m1 via _dispatch_loop; "
                f"got {[r.model_tag for r in residents]}. Empty here means the request went "
                "down the RETAINED legacy body and the unification is not in effect."
            )
        finally:
            await _drain(mgr, [gate1], [task1])


@pytest.mark.asyncio
async def test_one_resident_cap_never_spawns_more_than_one_sidecar_under_contention(tmp_path):
    """THE ACCEPTANCE RULE: only one sidecar is spawned at a time.

    m1 is held mid-turn; m2 (a DIFFERENT model, same GPU so they cannot co-reside) arrives.
    The resident count must NEVER exceed 1 at any observation. Sampled repeatedly rather
    than once, because a single check could miss a transient second spawn — which is
    precisely the failure this test exists to catch.
    """
    boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=1, grace_seconds=5)
    assert runtime.queue.max_parallel_sidecars == 1, "explicit cap — the env lies about this"
    seed_manifest(boot, "m1", main_gpu=0)
    seed_manifest(boot, "m2", main_gpu=0)
    gate1, gate2 = asyncio.Event(), asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate1, "m2": gate2})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task1 = task2 = None
        try:
            task1 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p", thread_id="t1", client_meta={}))
            await wait_until(lambda: len(mgr._model_residents()) == 1, timeout=6.0)

            task2 = asyncio.create_task(
                mgr.submit_and_wait("m2", "p", thread_id="t2", client_meta={}))

            worst = 0
            for _ in range(120):
                await asyncio.sleep(0.02)
                worst = max(worst, len(mgr._model_residents()))
                if worst > 1:
                    break
            assert worst <= 1, (
                f"cap=1 spawned {worst} residents concurrently. The acceptance criterion is "
                "that only one sidecar is spawned at a time — the parallel path must serialise "
                "at cap=1, not co-reside."
            )
            assert not task2.done(), (
                "m2 completed while m1 was still gated mid-turn — it cannot have been "
                "serialised behind m1, so the one-at-a-time claim is not being measured"
            )
        finally:
            await _drain(mgr, [gate1, gate2], [task1, task2])


@pytest.mark.asyncio
async def test_one_resident_cap_queues_the_second_request_and_eventually_serves_it(tmp_path):
    """Only one sidecar is spawned at a time — the OTHER half: the rest are
    QUEUED, not refused and not dropped.

    A test that only proved "never more than one resident" would pass just as happily if
    the second request were rejected outright, which would be a far worse product. So this
    asserts the second request SURVIVES the wait and is genuinely served once the first
    releases.
    """
    boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=1, grace_seconds=0)
    assert runtime.queue.max_parallel_sidecars == 1, "explicit cap — the env lies about this"
    seed_manifest(boot, "m1", main_gpu=0)
    seed_manifest(boot, "m2", main_gpu=0)
    gate1 = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate1})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task1 = task2 = None
        try:
            task1 = asyncio.create_task(
                mgr.submit_and_wait("m1", "p", thread_id="t1", client_meta={}))
            await wait_until(lambda: len(mgr._model_residents()) == 1, timeout=6.0)

            task2 = asyncio.create_task(
                mgr.submit_and_wait("m2", "p", thread_id="t2", client_meta={}))
            await asyncio.sleep(0.5)
            assert not task2.done(), "precondition: m2 must still be waiting behind m1"

            gate1.set()
            await asyncio.wait_for(task1, timeout=10.0)

            # m2 must now be SERVED — not refused, not dropped, not left parked forever.
            res2 = await asyncio.wait_for(task2, timeout=15.0)
            assert res2 is not None, (
                "m2 returned nothing after m1 released the single slot — a queued request "
                "must be served, not silently discarded"
            )
        finally:
            await _drain(mgr, [gate1], [task1, task2])
