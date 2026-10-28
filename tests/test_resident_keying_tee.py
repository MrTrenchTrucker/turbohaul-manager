"""per-resident tee plane: the streaming text tee must feed the SERVING resident's own live pane.

The tee's generation_id must be built from the SERVING resident's own spawn_seq (via the handle
accessors), not from instance 0 only. A sibling "m#1" that streams would otherwise feed a buffer
that no pane subscribes to, because its pane anchor uses its own spawn_seq. These tests drive the REAL ``_openai_chat_completions_stream`` route generator. Only
admission (``submit_for_streaming``, which would queue the slot) and the sidecar HTTP
(``httpx.AsyncClient`` in the route's module) are replaced. Every behaviour test first proves the
tee statement was reached: the recorded feed list and the upstream URL list must be non-empty.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import json
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request
from test_multislot_concurrency import _boot_runtime_multislot, _fake_handle, _mk, _mocks

from turbohaul import manager as manager_module
from turbohaul.api import chat_completion
from turbohaul.live_monitor import LiveResidentsSupervisor, compute_generation_id
from turbohaul.manager import Resident, ResidentState
from turbohaul.slot import Slot, SlotState

_SINGLETON_KEY = manager_module._SINGLETON_RESIDENT_KEY
_PORT_A = 59500
_PORT_B = 59501
_TEXT_BY_PORT = {_PORT_A: "AAA", _PORT_B: "BBB"}
_DONE = b"data: [DONE]\n\n"


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Client:
    """Scripted /slots client for the supervisor's pollers (same shape as the keying harness)."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else []
        self.closed = False

    async def get(self, url):
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
    a = _resident("m", 0, pid=41001, port=_PORT_A, spawn_seq=1)
    b = _resident("m#1", 1, pid=41002, port=_PORT_B, spawn_seq=2)
    return a, b


def _register(mgr, *residents):
    for r in residents:
        mgr._residents[r.resident_key] = r


def _anchor(r: Resident) -> str:
    """The poller's anchor generation_id for resident r: compute_generation_id(pid, spawn_seq, slot_id)."""
    return compute_generation_id(r.handle.pid, r.spawn_seq, r.active_slot.slot_id)


def _manager(tmp_path, *, max_parallel_sidecars: int = 2):
    boot, runtime = _boot_runtime_multislot(
        tmp_path, max_parallel_sidecars=max_parallel_sidecars, idle_hot_load_seconds=120)
    return _mk(boot, runtime, **_mocks([]))


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


class _FakeStreamResp:
    """What the open upstream stream hands back: a 200 status and a fixed list of SSE chunks."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aread(self):
        return b""

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCM:
    def __init__(self, chunks):
        self._resp = _FakeStreamResp(chunks)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *_exc):
        return False


class _FakeUpstream:
    """Stands in for httpx.AsyncClient inside api/chat_completion.py. ``stream`` records the URL
    and answers with one SSE content frame carrying the text registered for that sidecar port."""

    def __init__(self, text_by_port, urls):
        self._text_by_port = text_by_port
        self._urls = urls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, url, **_kwargs):
        self._urls.append(url)
        port = int(url.split(":")[2].split("/")[0])
        body = {"choices": [{"delta": {"content": self._text_by_port[port]}}]}
        frame = "data: " + json.dumps(body) + "\n\n"
        return _FakeStreamCM([frame.encode("utf-8"), _DONE])


async def _drain(body):
    async for _chunk in body:
        pass


async def _route_once(mgr, slot, handle, text_by_port) -> list:
    """Drive the REAL ``_openai_chat_completions_stream`` for one request whose slot is served by
    ``handle``. Returns the generation_ids the tee fed to ``mgr.live_output``. Only admission and
    the sidecar HTTP are replaced; the route's own tee statement runs unchanged."""
    slot.stream_ready_event = asyncio.Event()
    slot.stream_ready_event.set()
    slot.stream_done_event = asyncio.Event()
    slot.stream_handle = handle
    fed: list = []
    real_feed = mgr.live_output.feed

    def _feed(generation_id, sse_bytes):
        fed.append(generation_id)
        real_feed(generation_id, sse_bytes)

    async def _submit(**_kwargs):
        return slot

    urls: list = []
    messages = [{"role": "user", "content": "hi"}]
    payload = {"model": "m", "messages": messages, "stream": True}
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    with (
        patch.object(mgr, "submit_for_streaming", _submit),
        patch.object(mgr.live_output, "feed", _feed),
        patch("turbohaul.api.chat_completion.httpx.AsyncClient",
              lambda *_a, **_k: _FakeUpstream(text_by_port, urls)),
    ):
        response = await chat_completion._openai_chat_completions_stream(
            request, mgr, "m", messages, "hi", slot.thread_id, payload, "req-1")
        await asyncio.wait_for(_drain(response.body_iterator), 5)
    assert urls, "the route never opened the upstream stream, so the tee statement was not reached"
    return fed


def _pane_text(q) -> str:
    """Everything a pane queue holds up to its done marker (None)."""
    out = []
    while not q.empty():
        piece = q.get_nowait()
        if piece is None:
            break
        out.append(piece)
    return "".join(out)


def _capture(mgr) -> dict:
    """A deep, comparable copy of every manager state the accessors could touch."""
    return {
        "residents": {
            k: (r.model_tag, str(r.state), r.spawn_seq, r.port, id(r.handle), id(r.active_slot),
                r.resident_key, r.instance_idx)
            for k, r in mgr._residents.items()
        },
        "spawn_seq": mgr._spawn_seq,
        "live_generations": copy.deepcopy(dict(mgr.live_generations)),
        "live_generation": copy.deepcopy(mgr.live_generation),
        "identity": copy.deepcopy(dict(mgr._last_request_identity_by_model)),
    }


def _called_names(path: Path, func: str, *, cls: str | None = None) -> set:
    """Attribute and bare names called anywhere inside one function (nested defs included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = tree
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in ast.walk(scope)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func)
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Attribute):
                names.add(call.func.attr)
            elif isinstance(call.func, ast.Name):
                names.add(call.func.id)
    return names


class TestTeeFeedsTheServingResidentsPane:
    async def test_tee_gen_id_equals_each_residents_own_pane_anchor(self, tmp_path):
        """TE1: the tee's generation_id for a request served by resident r equals r's own pane anchor,
        for BOTH same-model residents, and the two ids differ. The published anchor block (the
        supervisor tick) must agree too. If the tee for 'm#1' were
        built from instance 0's spawn_seq (via _spawn_seq_for_model), it would not be 'm#1''s anchor."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        sup = LiveResidentsSupervisor(mgr, interval_s=1.0)
        try:
            with _supervisor_env(_Client(_slots(1001, 10)), _Client(_slots(2001, 20))):
                await sup._tick()  # the metrics plane publishes each resident's anchor block
            tee_ids = {}
            for r in (a, b):
                fed = await _route_once(mgr, r.active_slot, r.handle, _TEXT_BY_PORT)
                assert fed, f"per-resident TE1: the tee was never reached for resident {r.resident_key!r}"
                assert set(fed) == {_anchor(r)}, (
                    f"per-resident TE1: resident {r.resident_key!r} (pid {r.handle.pid}, spawn_seq {r.spawn_seq}) "
                    f"was fed {sorted(set(fed))}, expected its own anchor {_anchor(r)}")
                tee_ids[r.resident_key] = fed[0]
            assert tee_ids["m"] != tee_ids["m#1"], "per-resident TE1: both residents' tees feed one generation_id"
            snap = {e["pid"]: e for e in mgr._residents_snapshot()}
            for r in (a, b):
                block = snap[r.handle.pid]["generation"]
                assert block is not None, f"per-resident TE1: resident {r.resident_key!r} has no published block"
                assert block["generation_id"] == tee_ids[r.resident_key], (
                    f"per-resident TE1: published anchor {block['generation_id']!r} != tee id for {r.resident_key!r}")
        finally:
            await _teardown(mgr, sup)

    async def test_each_residents_pane_receives_its_own_text(self, tmp_path):
        """TE2: a pane subscribed to each resident's anchor before streaming receives exactly that
        resident's text ('AAA' through 'm', 'BBB' through 'm#1'), with no cross-feed and nothing
        missing. If the 'm#1' tee fed instance 0's id, its pane
        would stay empty."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        panes = {}
        for r in (a, b):
            q, _tail, _done = mgr.live_output.subscribe(_anchor(r), allow_create=True)
            assert q is not None, f"per-resident TE2: could not subscribe the pane of {r.resident_key!r}"
            panes[r.resident_key] = q
        try:
            fed_a = await _route_once(mgr, a.active_slot, a.handle, _TEXT_BY_PORT)
            fed_b = await _route_once(mgr, b.active_slot, b.handle, _TEXT_BY_PORT)
            assert fed_a and fed_b, "per-resident TE2: the tee was never reached for one of the residents"
            text_a = _pane_text(panes["m"])
            text_b = _pane_text(panes["m#1"])
            assert text_a == "AAA", f"per-resident TE2: pane of 'm' got {text_a!r}, expected 'AAA'"
            assert text_b == "BBB", f"per-resident TE2: pane of 'm#1' got {text_b!r}, expected 'BBB'"
        finally:
            await mgr.shutdown()


class TestCapOneTeeByteIdentical:
    async def test_cap_le_1_gen_id_is_byte_identical_to_the_old_expression(self, tmp_path):
        """TE3 (control, holds for both the singleton and the foreign-resident arms): with no model resident owning the handle
        (the singleton path) the tee's generation_id is the legacy expression's value; and with a
        resident 'm' that serves ANOTHER handle the value is still the legacy expression's (the
        resident's spawn_seq via _spawn_seq_for_model)."""
        mgr = _manager(tmp_path, max_parallel_sidecars=1)
        handle = _fake_handle("m", 59600, 77001)
        slot = Slot.new("m", thread_id="thread-cap1")
        slot.state = SlotState.ACTIVE
        want = compute_generation_id(handle.pid, mgr._spawn_seq_for_model("m"), slot.slot_id)
        fed = await _route_once(mgr, slot, handle, {59600: "CCC"})
        assert fed, "per-resident TE3: the tee was never reached (singleton arm)"
        assert set(fed) == {want}, f"per-resident TE3: singleton arm fed {sorted(set(fed))}, expected {want}"

        other = _fake_handle("m", 59601, 77002)
        mgr._residents["m"] = Resident(
            model_tag="m", handle=other, port=59601, spawn_seq=5,
            state=ResidentState.ACTIVE, resident_key="m", instance_idx=0)
        slot2 = Slot.new("m", thread_id="thread-cap1-b")
        slot2.state = SlotState.ACTIVE
        assert mgr._spawn_seq_for_model("m") == 5, "per-resident TE3: precondition, the resident's spawn_seq"
        want2 = compute_generation_id(handle.pid, 5, slot2.slot_id)
        fed2 = await _route_once(mgr, slot2, handle, {59600: "DDD"})
        assert fed2, "per-resident TE3: the tee was never reached (foreign-resident arm)"
        assert set(fed2) == {want2}, (
            f"per-resident TE3: foreign-resident arm fed {sorted(set(fed2))}, expected the legacy expression {want2}")
        await mgr.shutdown()


class TestServingResidentAccessors:
    async def test_serving_resident_for_handle_resolves_each_and_is_read_only(self, tmp_path):
        """TE4: _serving_resident_for_handle resolves each resident by its own handle; None, an
        unknown handle, the singleton's handle and a DEAD resident's handle all give None; and no
        call writes manager state."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        stray = _fake_handle("m", 59700, 88100)
        mgr._residents[_SINGLETON_KEY] = Resident(
            model_tag="m", handle=stray, port=59700,
            state=ResidentState.ACTIVE, resident_key=_SINGLETON_KEY)
        try:
            before = _capture(mgr)
            assert mgr._serving_resident_for_handle(a.handle) is a, "per-resident TE4: 'm' not resolved by its handle"
            assert mgr._serving_resident_for_handle(b.handle) is b, "per-resident TE4: 'm#1' not resolved by its handle"
            assert mgr._serving_resident_for_handle(None) is None, "per-resident TE4: None handle resolved"
            assert mgr._serving_resident_for_handle(_fake_handle("m", 59999, 88999)) is None, (
                "per-resident TE4: an unknown handle resolved")
            assert mgr._serving_resident_for_handle(stray) is None, "per-resident TE4: the singleton's handle resolved"
            assert _capture(mgr) == before, "per-resident TE4: a read-only accessor changed manager state"

            b.state = ResidentState.DEAD
            dead_before = _capture(mgr)
            assert mgr._serving_resident_for_handle(b.handle) is None, "per-resident TE4: a DEAD resident resolved"
            assert _capture(mgr) == dead_before, "per-resident TE4: resolving a DEAD resident changed manager state"
        finally:
            await mgr.shutdown()

    async def test_spawn_seq_for_handle_uses_serving_resident_and_falls_back(self, tmp_path):
        """TE5: _spawn_seq_for_handle returns the serving resident's own spawn_seq for each handle, and
        for an unknown or None handle returns _spawn_seq_for_model(model_tag)."""
        mgr = _manager(tmp_path)
        a, b = _two_residents()
        _register(mgr, a, b)
        try:
            assert mgr._spawn_seq_for_handle(a.handle, "m") == 1, "per-resident TE5: 'm' did not get its own spawn_seq"
            assert mgr._spawn_seq_for_handle(b.handle, "m") == 2, "per-resident TE5: 'm#1' did not get its own spawn_seq"
            stray = _fake_handle("m", 59777, 88001)
            assert mgr._spawn_seq_for_handle(stray, "m") == mgr._spawn_seq_for_model("m"), (
                "per-resident TE5: an unknown handle did not fall back to _spawn_seq_for_model")
            assert mgr._spawn_seq_for_handle(None, "m") == mgr._spawn_seq_for_model("m"), (
                "per-resident TE5: a None handle did not fall back to _spawn_seq_for_model")
            assert mgr._spawn_seq_for_handle(stray, "zzz") == mgr._spawn_seq_for_model("zzz"), (
                "per-resident TE5: an unknown handle and model did not fall back to _spawn_seq_for_model")
        finally:
            await mgr.shutdown()

    def test_heartbeat_and_tee_share_one_resolver(self):
        """TE6 (static pin, not the only pin): the route's tee statement and _log_kv_reuse_outcome both
        call the two accessors, and neither calls _spawn_seq_for_model directly."""
        route = _called_names(Path(chat_completion.__file__), "_openai_chat_completions_stream")
        assert "_spawn_seq_for_handle" in route, "per-resident TE6: the route's tee does not call _spawn_seq_for_handle"
        assert "_spawn_seq_for_model" not in route, "per-resident TE6: the route's tee still calls _spawn_seq_for_model"
        heartbeat = _called_names(
            Path(manager_module.__file__), "_log_kv_reuse_outcome", cls="TurbohaulManager")
        assert "_serving_resident_for_handle" in heartbeat, "per-resident TE6: the heartbeat does not resolve the resident"
        assert "_spawn_seq_for_handle" in heartbeat, "per-resident TE6: the heartbeat does not call _spawn_seq_for_handle"
        assert "_spawn_seq_for_model" not in heartbeat, "per-resident TE6: the heartbeat still calls _spawn_seq_for_model"
