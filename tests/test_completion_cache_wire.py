"""Cap>=2 completion-cache wiring: wire the existing
``_completion_cache_store`` helper into ``_serve_on_resident`` (the live
cap>=2 per-resident driver) -- it currently fires only from the retired
cap<=1 ``_process_slot`` body.

WITHOUT THE WIRING (this test MUST FAIL): ``_serve_on_resident``'s non-streaming
completion site resolves ``completion_future`` but never
calls ``_completion_cache_store``, so at cap>=2 a byte-identical retry never
HITs -- ``mgr._completion_cache`` stays empty after request 1, and the fake
decode fires a SECOND time on the retry.

Cap>=2 mirror of ``tests/test_completion_cache.py::
test_end_to_end_write_then_identical_retry_hits`` -- same shape, same two
assertions (cache populated, decode-call-count pinned at 1), but driven
through ``worker_loop`` -> ``_dispatch_loop`` -> ``_serve_on_resident``
instead of the cap<=1 ``_process_slot`` body. ⚠ THE VACUITY WARNING:
a cache test that passes because nothing retries is exactly the
vacuity to guard against -- the ``calls`` list is a REAL decode
counter, not merely "the cache dict is non-empty".
"""
import asyncio

import pytest

from turbohaul.manager import TurbohaulManager

from _fastlane_fixture import boot_ranked_runtime, seed_manifest, make_fakes, high_vram

MSGS = [{"role": "user", "content": "hi"}]


def _mgr_two_resident_cap(tmp_path, complete_fn):
    # grace + idle kept warm (like the cap<=1 counterpart's grace_seconds=30,
    # idle_hot_load_seconds=30) so the WRITE survives to the retry.
    boot, runtime = boot_ranked_runtime(
        tmp_path, max_parallel_sidecars=2, grace_seconds=30, idle_hot_load_seconds=30,
    )
    seed_manifest(boot, "m1", main_gpu=0)
    fake_spawn, fake_health, fake_sigterm, fake_vram, _ = make_fakes({})
    return TurbohaulManager(
        boot, runtime,
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=complete_fn,
    )


@pytest.mark.asyncio
async def test_two_resident_cap_end_to_end_write_then_identical_retry_hits(tmp_path):
    """End-to-end at cap>=2: worker_loop -> _dispatch_loop -> _serve_on_resident
    WRITEs the completion; a byte-identical retry HITs the cache and does NOT
    trigger a second decode."""
    calls = []

    async def complete_fn(slot, handle):
        calls.append(slot.slot_id)
        return {"answer": "decoded", "n": len(calls)}

    mgr = _mgr_two_resident_cap(tmp_path, complete_fn)
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            _slot1, result1 = await asyncio.wait_for(
                mgr.submit_and_wait(
                    model_tag="m1", prompt="hi", thread_id="t1", context=MSGS,
                    client_meta={},
                ),
                timeout=5.0,
            )
            assert result1["answer"] == "decoded"
            assert len(mgr._completion_cache) == 1, (
                "WRITE did not land at the cap>=2 completion site -- "
                "_serve_on_resident never called _completion_cache_store "
                "at that site"
            )
            # Byte-identical retry -> instant cache HIT, NO second decode.
            _slot2, result2 = await asyncio.wait_for(
                mgr.submit_and_wait(
                    model_tag="m1", prompt="hi", thread_id="t1", context=MSGS,
                    client_meta={},
                ),
                timeout=5.0,
            )
            assert result2 == result1
            assert len(calls) == 1, (
                f"expected exactly ONE decode across the original + retry, "
                f"got {len(calls)} -- the retry paid for a full redundant "
                f"decode instead of hitting the cache"
            )
    finally:
        await mgr.shutdown()
