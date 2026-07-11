"""Idempotent completion-cache + single-flight.

Fast + targeted — mocks spawn/health/sigterm/vram/complete via DI (no real
llama-server) and exercises the cache helpers + submit_and_wait admission
directly. Covers the key hazards: wrong-thread/collision, single-flight
future leak, cache-across-swap, and a mutated-messages retry (must MISS).
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import (
    _COMPLETION_CACHE_MAX,
    _FLIGHT_FAILED,
    TurbohaulManager,
)
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

MSGS = [{"role": "user", "content": "hi"}]


def _boot_runtime(tmp_path, grace_seconds=0, idle_hot_load_seconds=0):
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _mgr(tmp_path, complete_fn=None, grace_seconds=0, idle_hot_load_seconds=0):
    boot, runtime = _boot_runtime(tmp_path, grace_seconds, idle_hot_load_seconds)

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = 77_001
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def _noop_complete(slot, handle):
        return None

    return TurbohaulManager(
        boot,
        runtime,
        spawn_fn=fake_spawn,
        health_fn=fake_health,
        sigterm_fn=fake_sigterm,
        vram_fn=fake_vram,
        complete_fn=complete_fn or _noop_complete,
    )


def test_key_deterministic_and_field_sensitive(tmp_path):
    """Hazard (a) collision / wrong-thread + (d) mutated messages."""
    mgr = _mgr(tmp_path)
    k1 = mgr._completion_cache_key("m1", "t1", MSGS, {})
    assert k1 is not None
    # canonical (sort_keys) => key-order within a message is normalized => same req
    k_reordered = mgr._completion_cache_key(
        "m1", "t1", [{"content": "hi", "role": "user"}], {}
    )
    assert k1 == k_reordered
    # model_tag sensitivity — a result is model-specific, never cross-served
    assert mgr._completion_cache_key("m2", "t1", MSGS, {}) != k1
    # thread_id sensitivity — never serve one thread's answer to another
    assert mgr._completion_cache_key("m1", "t2", MSGS, {}) != k1
    # (d) mutated messages => different key => a mutated retry MISSES
    assert (
        mgr._completion_cache_key("m1", "t1", [{"role": "user", "content": "hi!"}], {})
        != k1
    )
    # output-knob sensitivity — identical messages+thread+model but a different
    # generation knob => DIFFERENT key, so a regenerate/distinct request is never
    # served a params-mismatched cached answer.
    assert mgr._completion_cache_key(
        "m1", "t1", MSGS, {"temperature": 0.0}
    ) != mgr._completion_cache_key("m1", "t1", MSGS, {"temperature": 1.5})
    assert mgr._completion_cache_key(
        "m1", "t1", MSGS, {"max_tokens": 50}
    ) != mgr._completion_cache_key("m1", "t1", MSGS, {"max_tokens": 2000})
    # identical knobs (the byte-identical retry-storm case) => SAME key => still HITS
    assert mgr._completion_cache_key(
        "m1", "t1", MSGS, {"temperature": 0.7, "top_p": 0.9}
    ) == mgr._completion_cache_key("m1", "t1", MSGS, {"temperature": 0.7, "top_p": 0.9})
    # streaming => cache disabled
    assert mgr._completion_cache_key("m1", "t1", MSGS, {"stream": True}) is None
    # absent context => cache disabled
    assert mgr._completion_cache_key("m1", "t1", None, {}) is None


async def test_cache_hit_returns_without_enqueue(tmp_path):
    calls = []

    async def complete_fn(slot, handle):
        calls.append(slot.slot_id)
        return {"answer": 42}

    mgr = _mgr(tmp_path, complete_fn)
    key = mgr._completion_cache_key("m1", "t1", MSGS, {})
    cached = {"answer": "from-cache"}
    mgr._completion_cache[key] = (cached, time.monotonic() + 100)
    # No worker_loop started: a MISS would enqueue + hang; a HIT returns instantly.
    slot, result = await asyncio.wait_for(
        mgr.submit_and_wait(model_tag="m1", thread_id="t1", context=MSGS, client_meta={}),
        timeout=2.0,
    )
    assert result is cached
    assert slot is None  # cache-hit return shape is (None, result)
    assert calls == []  # no decode happened
    d = mgr.queue.depth()  # nothing was enqueued (hit short-circuits before submit)
    assert d["staging_queue_depth"] == 0 and d["acceptance_buffer_depth"] == 0


async def test_single_flight_rider_rides_leader(tmp_path):
    """A concurrent byte-identical retry rides the leader's flight (dedupe)."""
    mgr = _mgr(tmp_path)
    key = mgr._completion_cache_key("m1", "t1", MSGS, {})
    leader_fut = asyncio.get_running_loop().create_future()
    mgr._completion_inflight[key] = leader_fut
    rider = asyncio.create_task(
        mgr.submit_and_wait(model_tag="m1", thread_id="t1", context=MSGS, client_meta={})
    )
    await asyncio.sleep(0.05)
    assert not rider.done()  # rider is awaiting the leader's flight, not enqueuing
    leader_fut.set_result({"answer": "leader"})
    slot, result = await asyncio.wait_for(rider, timeout=2.0)
    assert slot is None
    assert result == {"answer": "leader"}


async def test_rider_falls_through_when_leader_failed(tmp_path):
    """Hazard (b): a leader whose flight failed releases riders with the sentinel;
    riders must NOT hang and must NOT serve garbage — they fall through to a real
    submit (which then awaits its own completion, here left pending / cancelled)."""
    mgr = _mgr(tmp_path)
    key = mgr._completion_cache_key("m1", "t1", MSGS, {})
    leader_fut = asyncio.get_running_loop().create_future()
    mgr._completion_inflight[key] = leader_fut
    rider = asyncio.create_task(
        mgr.submit_and_wait(model_tag="m1", thread_id="t1", context=MSGS, client_meta={})
    )
    await asyncio.sleep(0.05)
    leader_fut.set_result(_FLIGHT_FAILED)
    await asyncio.sleep(0.05)
    assert not rider.done()  # fell through to a real submit (awaiting completion)
    rider.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rider


async def test_leader_finally_releases_inflight_on_failure(tmp_path):
    """Hazard (b): the leader's try/finally pops + resolves its inflight future
    even when the flight raises, so a later identical request never wedges."""

    async def boom_complete(slot, handle):
        raise RuntimeError("decode blew up")

    mgr = _mgr(tmp_path, boom_complete)
    key = mgr._completion_cache_key("m1", "t1", MSGS, {})
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    with pytest.raises(Exception):  # noqa: B017 — surfaces the worker failure
        await asyncio.wait_for(
            mgr.submit_and_wait(
                model_tag="m1", prompt="hi", thread_id="t1", context=MSGS, client_meta={}
            ),
            timeout=5.0,
        )
    # The inflight future must NOT leak (else a retry rides a dead flight forever).
    assert key not in mgr._completion_inflight
    await mgr.shutdown()


def test_swap_clear_empties_and_releases_riders(tmp_path):
    """Hazard (c): a swap-clear drops every entry + releases every rider."""

    async def _run():
        mgr = _mgr(tmp_path)
        key = mgr._completion_cache_key("m1", "t1", MSGS, {})
        mgr._completion_cache[key] = ({"a": 1}, time.monotonic() + 100)
        fut = asyncio.get_running_loop().create_future()
        mgr._completion_inflight["other"] = fut
        mgr._completion_cache_clear("test")
        assert len(mgr._completion_cache) == 0
        assert len(mgr._completion_inflight) == 0
        assert fut.done() and fut.result() is _FLIGHT_FAILED

    asyncio.run(_run())


def test_expired_entry_misses_and_is_dropped(tmp_path):
    mgr = _mgr(tmp_path)
    key = mgr._completion_cache_key("m1", "t1", MSGS, {})
    mgr._completion_cache[key] = ({"a": 1}, time.monotonic() - 1)  # already expired
    assert mgr._completion_cache_lookup(key) is None
    assert key not in mgr._completion_cache  # dropped on access


def test_lru_evicts_oldest_over_max(tmp_path):
    mgr = _mgr(tmp_path)
    for i in range(_COMPLETION_CACHE_MAX + 5):
        s = Slot.new(model_tag="m1", thread_id=f"t{i}", context=MSGS)
        s.completion_cache_key = f"k{i}"
        mgr._completion_cache_store(s, {"n": i})
    assert len(mgr._completion_cache) == _COMPLETION_CACHE_MAX
    assert "k0" not in mgr._completion_cache  # oldest evicted
    assert f"k{_COMPLETION_CACHE_MAX + 4}" in mgr._completion_cache  # newest kept


def test_store_skips_non_leader_and_none_result(tmp_path):
    mgr = _mgr(tmp_path)
    non_leader = Slot.new(model_tag="m1", thread_id="t1", context=MSGS)  # no key
    mgr._completion_cache_store(non_leader, {"a": 1})
    assert len(mgr._completion_cache) == 0  # not a leader -> no write
    leader = Slot.new(model_tag="m1", thread_id="t1", context=MSGS)
    leader.completion_cache_key = "k1"
    mgr._completion_cache_store(leader, None)  # None result -> not cached
    assert "k1" not in mgr._completion_cache


async def test_end_to_end_write_then_identical_retry_hits(tmp_path):
    """End-to-end: worker_loop WRITEs the completion; a byte-identical retry HITs
    the cache and does NOT trigger a second decode."""
    calls = []

    async def complete_fn(slot, handle):
        calls.append(slot.slot_id)
        return {"answer": "decoded", "n": len(calls)}

    # Keep the sidecar warm (grace + idle) so the WRITE survives to the retry.
    mgr = _mgr(tmp_path, complete_fn, grace_seconds=30, idle_hot_load_seconds=30)
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    _slot1, result1 = await asyncio.wait_for(
        mgr.submit_and_wait(
            model_tag="m1", prompt="hi", thread_id="t1", context=MSGS, client_meta={}
        ),
        timeout=5.0,
    )
    assert result1["answer"] == "decoded"
    assert len(mgr._completion_cache) == 1  # WRITE landed at the completion site
    # Byte-identical retry -> instant cache HIT, NO second decode.
    _slot2, result2 = await asyncio.wait_for(
        mgr.submit_and_wait(
            model_tag="m1", prompt="hi", thread_id="t1", context=MSGS, client_meta={}
        ),
        timeout=5.0,
    )
    assert result2 == result1
    assert len(calls) == 1  # exactly ONE decode across the original + retry
    await mgr.shutdown()
