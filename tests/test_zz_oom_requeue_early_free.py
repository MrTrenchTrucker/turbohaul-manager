"""Regression probe: memory frees BEFORE the companion's
first backstock reading.

REWRITTEN from the original probe shape: the
original copy of this file scripted a single flat, always-fitting value
patched right after park -- the exact same input shape as
the empty-box no-rise test (which patches a
different flat, always-fitting value at the same point and must NEVER
attempt). A flat constant cannot show growth against itself under any
comparison-based rule, so no implementation could make both tests pass;
that was a mis-shape in the original probe, not a timing bug.

WHY THE PRIMARY ASSERTION IS STRUCTURAL, NOT A retried BOOLEAN: building
this as first specified (script insufficient-then-sufficient, assert
exactly one retry) surfaced a second, deeper problem. Given ANY rising
(insufficient-then-sufficient) probe sequence and a generous timeout, the
UNFIXED companion ALSO eventually retries -- its own tick-1
establishes a baseline from whatever the first post-failure reading is,
and its tick-2 shows growth past THAT baseline and attempts, exactly like
the pre-existing "rise" test already proves works. A `retried == True`
boolean, given enough ticks, cannot tell "the seed made this possible"
apart from "the OLD tick-based baseline machinery got there one tick
later" -- both eventually retry; only the TIMING differs (this needs one
extra ~10ms-scale real backstock tick on the unfixed code), and asserting on that
timing margin would be exactly the kind of sleep-shaped, flaky test
that should be avoided.

The actual, provable claim -- the seed-timing rule -- is structural:
the seed READ completes strictly before the slot is observably parked.
That is directly assertable, with no timing margin and no growth-sequence
cleverness needed: patch a probe that flags when it has taken its first
post-failure reading (gated on the spawn-attempt counter, ``calls[0]``, so
it is correct regardless of how many ADMISSION-phase reads happen first --
empirically an environment-dependent number, not a fixed 1), then check
that flag the instant ``_wait_until_parked`` returns True.
  - On the unfixed code (no seed step -- the flag flips synchronously, before the
    companion is even spawned): the flag is FALSE at that instant. Verified
    by running this exact test against a scratch checkout of the unfixed
    code -- RED for the claimed reason, not a coincidental one.
  - On the fix (the seed read is awaited to completion before
    ``oom_requeue_pending`` flips): the flag is TRUE at that instant.

A second, functional assertion follows: given the seed reads insufficient
and the very next (companion) reading is sufficient, the request DOES
retry -- proving the mechanism is wired end to end, not just proving the
seed's timing in isolation.
"""
import asyncio
from turbohaul import safety
from tests.test_oom_requeue_resident_driver import (
    _OOM_SCAN, _make_manager, _seed_manifest, _wait_until_parked,
    _pump_backstock_only_until_retried, _bounded_shutdown, _run_worker,
)


def _make_seed_timed_probe(calls):
    """Returns HUGE/fitting for every admission-phase read (however many
    there are, before the first spawn/health attempt); INSUFFICIENT for
    the first read taken after that attempt (the seed, or -- on unfixed
    code -- the companion's own tick-1); SUFFICIENT for every read after
    that."""
    state = {"seeded": False, "n": 0}

    def fake():
        state["n"] += 1
        if calls[0] == 0:
            return [999_999_999]
        if not state["seeded"]:
            state["seeded"] = True
            return [8000]
        return [25000]

    return fake, state


async def test_zz_seed_read_completes_before_park_is_observable(
    tmp_path, monkeypatch
):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN, poll_s=0.01)
    _seed_manifest(boot, "earlyfree", expected_vram_bytes=15000 * 1024 * 1024, split_mode="none")
    probe, state = _make_seed_timed_probe(calls)
    monkeypatch.setattr(safety, "_read_free_vram_all_mib", probe)
    ev = asyncio.Event()
    try:
        slot = await mgr.submit(model_tag="earlyfree", prompt="x", wait_for_completion=True, disconnect_event=ev)
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "sanity: must OOM-park first"
        assert state["seeded"], (
            "the empty-box seed reading must complete BEFORE the slot is "
            "observably parked (oom_requeue_pending visible) -- if this is "
            "False, the seed is being taken too late (e.g. inside the "
            "companion, after the flag already flipped) and the original "
            "'memory freed before the first tick' bug is back"
        )
        retried = await _pump_backstock_only_until_retried(mgr, calls, calls[0], timeout=1.0)
        print("EARLYFREE probe_calls=", state["n"], "retried=", retried, "parked=", slot.oom_requeue_pending)
        assert retried, (
            "insufficient at the seed, sufficient on the very next reading: "
            "the request must retry, not stay parked"
        )
    finally:
        ev.set()
        await _bounded_shutdown(mgr)
