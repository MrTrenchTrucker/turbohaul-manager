"""per-resident keying: two residents of the SAME model must keep their own data.

Per-resident generation blocks are keyed by Resident.resident_key, not by model_tag
(manager.live_generations, live_monitor _pollers / _poll_one / _update_primary_alias,
api/live_stream ?resident_key=). Residents 'm' and 'm#1' both have model_tag 'm', so they must not
share one entry. These tests pin the observable contract: per-resident generation
blocks across interleaved supervisor ticks, the per-resident SSE stream, and the unchanged
cap<=1 path.

Helpers come from test_multislot_concurrency (tests/ is on pytest's pythonpath).
"""
from __future__ import annotations

import contextlib
import json
from unittest.mock import patch
from urllib.parse import urlencode

from fastapi import FastAPI
from starlette.requests import Request
from test_multislot_concurrency import _boot_runtime_multislot, _fake_handle, _mk, _mocks

from turbohaul.api.live_stream import live_output_stream
from turbohaul.live_monitor import LiveResidentsSupervisor, ResidentSlotsPoller, compute_generation_id
from turbohaul.manager import Resident, ResidentState
from turbohaul.slot import Slot, SlotState


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Client:
    """Scripted /slots client: returns its current payload; on_get fires first (mid-poll hook)."""

    def __init__(self, payload=None, on_get=None):
        self.payload = payload if payload is not None else []
        self.on_get = on_get
        self.closed = False

    async def get(self, url):
        if self.on_get is not None:
            self.on_get(url)
        return _Resp(self.payload)

    async def aclose(self):
        self.closed = True


def _slots(n_ctx: int, n_decoded: int) -> list[dict]:
    """A llama.cpp /slots body with ONE processing slot carrying this resident's n_ctx and n_decoded."""
    return [{
        "id_task": 0,
        "is_processing": True,
        "next_token": [{"n_decoded": n_decoded, "n_remain": 100 - n_decoded, "has_next_token": True}],
        "n_prompt_tokens": 5,
        "n_prompt_tokens_processed": 5,
        "n_ctx": n_ctx,
        "params": {"max_tokens": 100, "stream": True},
    }]


def _resident(key: str, idx: int, *, pid: int, port: int, spawn_seq: int) -> Resident:
    """An ACTIVE resident of model 'm' with its own handle, spawn_seq and serving slot."""
    slot = Slot.new("m", thread_id=f"thread-{key}")
    slot.state = SlotState.ACTIVE
    return Resident(
        model_tag="m",
        handle=_fake_handle("m", port, pid),
        port=port,
        spawn_seq=spawn_seq,
        active_slot=slot,
        state=ResidentState.ACTIVE,
        resident_key=key,
        instance_idx=idx,
    )


def _two_residents():
    a = _resident("m", 0, pid=41001, port=59500, spawn_seq=1)
    b = _resident("m#1", 1, pid=41002, port=59501, spawn_seq=2)
    return a, b


def _register(mgr, *residents):
    for r in residents:
        mgr._residents[r.resident_key] = r


def _gen_id(pid: int, spawn_seq: int, resident: Resident) -> str:
    return compute_generation_id(pid, spawn_seq, resident.active_slot.slot_id)


def _manager(tmp_path, *, max_parallel_sidecars: int = 2):
    boot, runtime = _boot_runtime_multislot(
        tmp_path, max_parallel_sidecars=max_parallel_sidecars, idle_hot_load_seconds=120)
    return _mk(boot, runtime, **_mocks([]))


def _poller(mgr, client):
    """A ResidentSlotsPoller whose httpx client is the given fake (no real client is built)."""
    with patch("turbohaul.live_monitor.httpx.AsyncClient", return_value=client):
        return ResidentSlotsPoller(mgr, interval_s=1.0)


@contextlib.contextmanager
def _supervisor_env(*clients):
    """Pollers the supervisor creates take these clients in creation order; no nvidia-smi probe."""
    with (
        patch("turbohaul.live_monitor.httpx.AsyncClient", side_effect=list(clients)),
        patch("turbohaul.live_monitor._read_free_vram_all_mib", return_value=[111, 222]),
        patch("turbohaul.live_monitor._read_gpu_util_all_percent", return_value=None),
    ):
        yield


async def _teardown(mgr, sup):
    await sup._close_all()
    await mgr.shutdown()


class TestSameModelResidentsKeepOwnGeneration:
    async def test_interleaved_ticks_keep_separate_generation_blocks(self, tmp_path):
        """per-resident T1: over three interleaved ticks, 'm' and 'm#1' each read ITS OWN generation block
        (generation_id, n_ctx, n_decoded). If _tick/_poll_one keyed by model_tag and
        _residents_snapshot read live_generations.get(model_tag), both would show the last writer's block."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        client_a, client_b = _Client(), _Client()
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(client_a, client_b):
                for t in (1, 2, 3):
                    client_a.payload = _slots(1000 + t, 10 * t)
                    client_b.payload = _slots(2000 + t, 20 * t)
                    await sup._tick()
                    snap = {e["pid"]: e for e in mgr._residents_snapshot()}
                    ga, gb = snap[41001]["generation"], snap[41002]["generation"]
                    assert ga is not None and gb is not None, f"per-resident tick {t}: a resident has no generation block"
                    assert ga["generation_id"] == _gen_id(41001, 1, a), f"per-resident tick {t}: 'm' shows a foreign gid"
                    assert gb["generation_id"] == _gen_id(41002, 2, b), f"per-resident tick {t}: 'm#1' shows a foreign gid"
                    assert ga["n_ctx"] == 1000 + t and ga["n_decoded"] == 10 * t, (
                        f"per-resident tick {t}: 'm' carries another resident's n_ctx/n_decoded: {ga!r}")
                    assert gb["n_ctx"] == 2000 + t and gb["n_decoded"] == 20 * t, (
                        f"per-resident tick {t}: 'm#1' carries another resident's n_ctx/n_decoded: {gb!r}")
                    assert ga["generation_id"] != gb["generation_id"], (
                        f"per-resident tick {t}: both residents share one generation_id")
        finally:
            await _teardown(mgr, sup)

    async def test_live_generations_is_keyed_by_resident_key(self, tmp_path):
        """per-resident T2: after one tick the shared dict holds one entry per RESIDENT KEY, {'m', 'm#1'}.
        If the writer stored under model_tag, only {'m'} would exist."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1000, 10)), _Client(_slots(2000, 20))):
                await sup._tick()
            assert set(mgr.live_generations) == {"m", "m#1"}, (
                f"live_generations keys are {sorted(mgr.live_generations)}, "
                f"expected one entry per resident key")
        finally:
            await _teardown(mgr, sup)

    async def test_each_resident_gets_its_own_poller(self, tmp_path):
        """per-resident T3: with no pre-seeded pollers, one tick creates one poller per RESIDENT KEY, and
        the two are distinct objects with distinct rate state. If _tick keyed _pollers by
        model_tag, 'm' and 'm#1' would share ONE poller."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1000, 10)), _Client(_slots(2000, 20))):
                await sup._tick()
            assert set(sup._pollers) == {"m", "m#1"}, f"poller keys are {sorted(sup._pollers)}"
            pa, pb = sup._pollers["m"], sup._pollers["m#1"]
            assert pa is not pb, "both residents share one poller object"
            assert pa._samples is not pb._samples, "both residents share one rate-sample table"
        finally:
            await _teardown(mgr, sup)

    async def test_sibling_dead_midpoll_drops_only_its_own_block(self, tmp_path):
        """per-resident T4: 'm#1' goes DEAD during its own /slots await; _poll_one must drop ONLY 'm#1's
        block and leave 'm' untouched. Guard: the
        liveness re-check resolves the key 'm#1' to the right resident. It pins the sibling-independence contract that the keying must keep."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        pa = _poller(mgr, _Client(_slots(1000, 10)))
        pb = _poller(mgr, _Client(_slots(2000, 20)))

        def _kill_b(url):
            b.state = ResidentState.DEAD

        try:
            await sup._poll_one("m", pa, a)
            await sup._poll_one("m#1", pb, b)
            block_a = mgr.live_generations["m"]
            assert "m#1" in mgr.live_generations, "per-resident setup: 'm#1' has no block before it dies"
            pb._client = _Client(_slots(2000, 25), on_get=_kill_b)
            await sup._poll_one("m#1", pb, b)
            assert "m#1" not in mgr.live_generations, "the DEAD sibling still has a live block"
            assert mgr.live_generations.get("m") is block_a, "the living sibling's block was dropped"
        finally:
            await pa._client.aclose()
            await pb._client.aclose()
            await mgr.shutdown()

    async def test_sibling_dead_midpoll_via_tick_drops_only_its_own_block(self, tmp_path):
        """per-resident T9: the REAL caller. _tick (not _poll_one) runs both residents, and 'm#1' goes DEAD
        during its own /slots await in the SECOND tick. The dead sibling's block must be under no
        key, and 'm' keeps its own block. If _tick passed the
        model_tag to _poll_one, the liveness check would look up instance 0, find it alive, and
        write the dead sibling's block under 'm'."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        client_a, client_b = _Client(), _Client()
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        old_b_gid = _gen_id(41002, 2, b)

        def _kill_b_on_its_port(url):
            # Fires only during the poll of 'm#1' (port 59501), whichever client carries that poll.
            if url.endswith(":59501/slots"):
                b.state = ResidentState.DEAD

        try:
            with _supervisor_env(client_a, client_b):
                client_a.payload = _slots(1001, 10)
                client_b.payload = _slots(2001, 20)
                await sup._tick()  # first tick: both residents get their block
                client_a.on_get = _kill_b_on_its_port  # armed for the second tick only
                client_b.on_get = _kill_b_on_its_port
                await sup._tick()  # 'm#1' dies during its own /slots await
            assert "m#1" not in mgr.live_generations, "the DEAD sibling 'm#1' still has a block under its key"
            assert not any(v.get("generation_id") == old_b_gid for v in mgr.live_generations.values()), (
                f"the DEAD sibling's generation {old_b_gid!r} survives under another key: "
                f"{mgr.live_generations!r}")
            blk_a = mgr.live_generations.get("m")
            assert blk_a is not None and blk_a.get("generation_id") == _gen_id(41001, 1, a), (
                f"the living sibling 'm' lost its own block after the sibling died: {blk_a!r}")
        finally:
            await _teardown(mgr, sup)

    async def test_primary_alias_follows_most_recently_active_resident(self, tmp_path):
        """per-resident T5: the back-compat alias mirrors the MOST RECENTLY ACTIVE resident, 'm#1' here.
        Block writes are arranged so that reading
        live_generations.get(r.model_tag) for both residents would pick 'm''s block."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        block_a = {"state": "generating", "generation_id": "gid-a", "n_ctx": 1000, "n_decoded": 10}
        block_b = {"state": "generating", "generation_id": "gid-b", "n_ctx": 2000, "n_decoded": 20}
        mgr.live_generations["m#1"] = block_b
        mgr.live_generations["m"] = block_a  # last writer is 'm'
        a.last_active_monotonic = 5.0
        b.last_active_monotonic = 9.0  # 'm#1' is the most recently active
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            sup._update_primary_alias([a, b])
            assert mgr.live_generation == block_b, (
                f"primary alias is {mgr.live_generation!r}, expected the most recently active "
                f"resident's block {block_b!r}")
        finally:
            await mgr.shutdown()

    async def test_gc_drops_only_the_vanished_resident_key(self, tmp_path):
        """per-resident T6: when 'm#1' vanishes, _gc_vanished closes ITS poller and drops ITS block; 'm' keeps
        both. Guard: the GC is key-based and the
        same shape as test_supervisor_gcs_vanished_resident."""
        mgr = _manager(tmp_path)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        pa = _poller(mgr, _Client())
        pb = _poller(mgr, _Client())
        sup._pollers["m"] = pa
        sup._pollers["m#1"] = pb
        block_a = {"state": "generating", "generation_id": "gid-a"}
        mgr.live_generations["m"] = block_a
        mgr.live_generations["m#1"] = {"state": "generating", "generation_id": "gid-b"}
        try:
            await sup._gc_vanished(live_tags={"m"})
            assert "m#1" not in sup._pollers, "vanished sibling's poller was not GC'd"
            assert "m#1" not in mgr.live_generations, "vanished sibling's block was not GC'd"
            assert pb._client.closed, "vanished sibling's httpx client was not closed"
            assert sup._pollers.get("m") is pa and not pa._client.closed, "living sibling was GC'd"
            assert mgr.live_generations.get("m") is block_a, "living sibling's block was dropped"
        finally:
            await mgr.shutdown()

    async def test_reused_resident_key_does_not_inherit_the_dead_block(self, tmp_path):
        """per-resident T7: a NEW resident registered under the key 'm#1' after the old one was GC'd must
        publish its OWN block, never the dead resident's. If blocks were keyed by model_tag, the new resident would read the shared 'm' entry (written
        by the living sibling), and the assertion on its generation_id would fail."""
        mgr = _manager(tmp_path)
        a, dead_b = _two_residents()
        _register(mgr, a, dead_b)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        dead_gid = _gen_id(41002, 2, dead_b)
        new_b = _resident("m#1", 1, pid=42002, port=59501, spawn_seq=3)
        with _supervisor_env(_Client(_slots(1000, 10)), _Client(_slots(2000, 20)), _Client(_slots(3000, 30))):
            try:
                await sup._tick()  # both live: a block under 'm' and under 'm#1'
                del mgr._residents["m#1"]  # the 'm#1' resident is gone
                await sup._tick()  # GC: 'm#1' loses its poller and its block
                assert "m#1" not in mgr.live_generations, "dead resident's block survived the GC"
                _register(mgr, new_b)  # a new resident takes the same key
                await sup._tick()
                snap = {e["pid"]: e for e in mgr._residents_snapshot()}
                gen = snap[42002]["generation"]
                assert gen is not None, "the new resident under 'm#1' has no generation block"
                assert gen["generation_id"] != dead_gid, "the new resident inherited the dead block"
                assert gen["generation_id"] == _gen_id(42002, 3, new_b), (
                    "the new resident's generation_id is not its own")
                assert gen["n_ctx"] == 3000, f"new resident's n_ctx is {gen['n_ctx']!r}, expected 3000"
            finally:
                await _teardown(mgr, sup)

    async def test_close_all_clears_every_resident_block(self, tmp_path):
        """per-resident T8: teardown closes every per-resident poller and clears every per-resident block,
        including the alias."""
        mgr = _manager(tmp_path)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        pa = _poller(mgr, _Client())
        pb = _poller(mgr, _Client())
        sup._pollers["m"] = pa
        sup._pollers["m#1"] = pb
        mgr.live_generations["m"] = {"state": "generating", "generation_id": "gid-a"}
        mgr.live_generations["m#1"] = {"state": "generating", "generation_id": "gid-b"}
        mgr.live_generation = mgr.live_generations["m"]
        try:
            await sup._close_all()
            assert sup._pollers == {}, "pollers survived _close_all"
            assert mgr.live_generations == {}, "blocks survived _close_all"
            assert mgr.live_generation is None, "primary alias survived _close_all"
            assert pa._client.closed and pb._client.closed, "a poller's httpx client was not closed"
        finally:
            await mgr.shutdown()

    async def test_heartbeat_of_a_sibling_request_reads_its_own_block(self, tmp_path, caplog):
        """per-resident H1: the KV_REUSE_OUTCOME heartbeat of a request served on the SIBLING "m#1" reads the
        block of "m#1" (its distinctive n_prompt_cache), not the instance-0 block of "m". The block is
        proven by its generation_id, which is compute_generation_id(pid, spawn_seq, slot_id) of "m#1".
        If the reader took live_generations.get(slot.model_tag) and the spawn_seq
        of instance 0, the sibling's block would never match and the line would say no_matching_sample."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        slot = b.active_slot
        mgr.live_generations["m"] = {"generation_id": _gen_id(41001, 1, a), "n_prompt_cache": 11111}
        mgr.live_generations["m#1"] = {"generation_id": _gen_id(41002, 2, b), "n_prompt_cache": 77777}
        caplog.set_level("INFO")
        try:
            mgr._log_kv_reuse_outcome(slot, b.handle, wait_state="completed")
        finally:
            await mgr.shutdown()
        lines = [r.message for r in caplog.records if r.message.startswith("KV_REUSE_OUTCOME")]
        assert len(lines) == 1, f"per-resident H1: expected one KV_REUSE_OUTCOME line, got {lines!r}"
        assert "reason=no_matching_sample" not in lines[0], (
            f"per-resident H1: the sibling 'm#1' request found no matching sample: {lines[0]}")
        assert "engine_reused_tokens=77777" in lines[0], (
            f"per-resident H1: the sibling must report its OWN block's 77777, not instance 0's 11111: {lines[0]}")


async def _first_sse_event(mgr, query: dict) -> dict:
    """Drive the REAL route live_output_stream and return its first SSE data frame.

    While the route has no anchor it emits ': keep-alive' comments (after its own 0.5s idle wait),
    so this skips those and reads at most 3 chunks: bounded, and the route's wait is the only clock."""
    app = FastAPI()
    app.state.manager = mgr
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/ui/live/output/stream",
        "headers": [],
        "query_string": urlencode(query).encode("ascii"),
        "app": app,
    }
    response = await live_output_stream(Request(scope))
    frames = response.body_iterator
    found = None
    try:
        for _ in range(3):
            chunk = await frames.__anext__()
            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if text.startswith("data: "):
                found = json.loads(text[len("data: "):].split("\n\n", 1)[0])
                break
    finally:
        await frames.aclose()
    if found is None:
        raise AssertionError(f"the route sent no data frame in 3 chunks (query {query!r}): only keep-alives")
    return found


def _seed_two_blocks(mgr):
    block_a = {"state": "generating", "generation_id": "gid-a", "n_ctx": 1000, "n_decoded": 10}
    block_b = {"state": "generating", "generation_id": "gid-b", "n_ctx": 2000, "n_decoded": 20}
    mgr.live_generations["m"] = block_a
    mgr.live_generations["m#1"] = block_b
    return block_a, block_b


class TestLiveStreamResidentKey:
    async def test_resident_key_param_follows_that_residents_block(self, tmp_path):
        """per-resident L1: ?resident_key=m#1 streams the m#1 block and ?resident_key=m streams the m block.
        The alias is None here, so only the new param can pick a sibling. If the param were
        ignored (only model_tag and the alias being read), the sibling would not be picked."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        block_a, block_b = _seed_two_blocks(mgr)
        mgr.live_generation = None
        try:
            ev_b = await _first_sse_event(mgr, {"resident_key": "m#1"})
            ev_a = await _first_sse_event(mgr, {"resident_key": "m"})
            assert ev_b["generation_id"] == block_b["generation_id"], (
                f"?resident_key=m#1 streamed {ev_b.get('generation_id')!r}, expected the m#1 block")
            assert ev_a["generation_id"] == block_a["generation_id"], (
                f"?resident_key=m streamed {ev_a.get('generation_id')!r}, expected the m block")
        finally:
            await mgr.shutdown()

    async def test_legacy_model_tag_param_follows_instance_zero(self, tmp_path):
        """per-resident L2: the legacy ?model_tag=m still streams instance 0's block (key 'm'). Guard for
        cap<=1 back-compat."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        block_a, _block_b = _seed_two_blocks(mgr)
        try:
            ev = await _first_sse_event(mgr, {"model_tag": "m"})
            assert ev["generation_id"] == block_a["generation_id"], (
                f"legacy ?model_tag=m streamed {ev.get('generation_id')!r}, expected instance 0's block")
        finally:
            await mgr.shutdown()

    async def test_sse_payload_model_tag_is_the_real_model_tag_for_a_sibling(self, tmp_path):
        """per-resident L3: for ?resident_key=m#1 the frame's model_tag is the REAL model 'm' (never the key
        'm#1'), and the frame carries resident_key 'm#1'. The alias points at 'm#1''s block so the
        _tag_for_gid must not report the key as the model."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        _block_a, block_b = _seed_two_blocks(mgr)
        mgr.live_generation = block_b
        try:
            ev = await _first_sse_event(mgr, {"resident_key": "m#1"})
            assert ev.get("model_tag") == "m", (
                f"sibling frame reports model_tag {ev.get('model_tag')!r}, expected the real model 'm'")
            assert ev.get("resident_key") == "m#1", (
                f"sibling frame carries resident_key {ev.get('resident_key')!r}, expected 'm#1'")
        finally:
            await mgr.shutdown()

    async def test_no_param_follows_the_primary_alias(self, tmp_path):
        """per-resident L4: with no param the stream follows the primary alias (unchanged). Guard."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        _block_a, block_b = _seed_two_blocks(mgr)
        mgr.live_generation = block_b
        try:
            ev = await _first_sse_event(mgr, {})
            assert ev["generation_id"] == block_b["generation_id"], (
                f"no-param stream followed {ev.get('generation_id')!r}, expected the primary alias")
            assert "model_tag" in ev, "no-param frame lost its model_tag field"
        finally:
            await mgr.shutdown()


class TestCapOneByteIdentical:
    async def test_single_resident_block_is_keyed_by_model_tag(self, tmp_path):
        """per-resident C1: one instance-0 resident (resident_key 'm' == model_tag) keys its block by 'm' and
        the snapshot carries it. Guard."""
        mgr = _manager(tmp_path)
        a = _resident("m", 0, pid=41001, port=59500, spawn_seq=1)
        _register(mgr, a)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1000, 10))):
                await sup._tick()
            assert set(mgr.live_generations) == {"m"}, (
                f"single resident keys are {sorted(mgr.live_generations)}, expected ['m']")
            snap = mgr._residents_snapshot()
            assert [e["pid"] for e in snap] == [41001], f"per-resident snapshot residents are {snap!r}"
            assert snap[0]["generation"] is not None, "single resident lost its generation block"
        finally:
            await _teardown(mgr, sup)

    async def test_cap1_snapshot_residents_empty_and_alias_unchanged(self, tmp_path):
        """per-resident C2: at max_parallel_sidecars=1 the status surface keeps residents == [] and the
        top-level generation key. Guard."""
        mgr = _manager(tmp_path, max_parallel_sidecars=1)
        try:
            snap = mgr.status_snapshot()
            assert snap["residents"] == [], f"per-resident cap<=1 residents is {snap['residents']!r}, expected []"
            assert "generation" in snap, "per-resident cap<=1 snapshot lost the top-level generation key"
        finally:
            await mgr.shutdown()
