"""An unload never cuts a turn that is running, for any client.

Two Fast Lane clients are configured on a card that holds exactly one model. Client 1 is listed
first, so it is the better client; client 2 owns the resident model A. A better client asks for a
different model B while A is in the middle of a turn. The turn must run to its end: A is not
unloaded, not marked dead, its engine is not stopped, and its response reaches the caller intact.
Only after that turn has ended does A unload (saving its KV first) and B load.

The cases differ only in which turn is held and where:

  * second turn, held mid-generation (the engine is deciding tokens);
  * second turn, held mid-prefill (no byte produced yet; prefill is part of the turn);
  * second turn, streamed slowly for several seconds (long is not the same as stalled);
  * first turn, held mid-generation (control: the resident has never been through a grace window).

"Second turn" means the resident already served one turn, passed through a full grace window and
was parked before the turn under test arrived. That state is what distinguishes the cases: a
resident that has never entered a grace window is protected today, one that has is not.

Every case asserts WHICH events happened and in WHAT ORDER from one append-only event list (list
position is the order; the loop is single-threaded so position is exact). The response of the
running turn is observed at the moment it is set on the slot's completion future, which is the
earliest point a caller can see it; the unload is observed at the start of the detached teardown.

How the KV save is observed. The engine is a fake HTTP client installed in place of the manager's
httpx module: `GET /slots` reports one populated slot and `POST /slots/<id>?action=save` writes the
requested file into a temporary directory that stands in for the engine's save directory. The
manager's real save routine then renames the file and writes its metadata sidecar. The tests assert
the ARTIFACT on disk (a non-empty bin with its metadata, readable at the moment the engine is
stopped), plus the order of the save call against the engine stop. A log line is never the evidence.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field

import httpx as _real_httpx
import pytest
import yaml

from _fastlane_fixture import boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule
from turbohaul.manager import ResidentState, TurbohaulManager

pytestmark = pytest.mark.asyncio

IP_1 = "192.0.2.10"
IP_2 = "198.51.100.20"
FOOTPRINT_MIB = 20000
CARD_MIB = 24000
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)
TAG_A, TAG_B = "model-a", "model-b"
CLIENT_1_FIRST = [FastLaneRule(address=IP_1), FastLaneRule(address=IP_2)]
CLIENT_2_META = {"ip": IP_2, "is_main": True}      # owns the resident (rule index 1, the worse client)
CLIENT_1_META = {"ip": IP_1, "is_main": True}      # the claimant (rule index 0, the better client)
OBSERVE_S = 3.0           # several polls of the 0.05 s grace and dispatch loops, with margin
STREAM_TOKENS = 13        # one token every STREAM_INTERVAL_S: a stream of about 3.9 s
STREAM_INTERVAL_S = 0.3
KV_BYTES = b"kv-cache-bytes-" * 64

DECODE, PREFILL, STREAM = "decode", "prefill", "stream"


class _EngineClient:
    """Stands in for httpx.AsyncClient against the engine: /slots plus the slot save endpoint."""

    def __init__(self, box_holder):
        self._holder = box_holder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    async def get(self, url, **kw):
        if "/slots" in url and "action=" not in url:
            return self._Resp([{"id": 0, "n_prompt_tokens": 4000, "thread_id": "th-a"}])
        return self._Resp({})

    async def post(self, url, json=None, **kw):
        if "action=save" in url and json and "filename" in json:
            import os
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            os.makedirs(SLOT_SAVE_DIR, exist_ok=True)
            with open(os.path.join(SLOT_SAVE_DIR, json["filename"]), "wb") as f:
                f.write(KV_BYTES)
            return self._Resp({"n_saved": json.get("save_token_limit"), "n_written": len(KV_BYTES)})
        return self._Resp({"status": "ok"})


class _FakeHttpx:
    """The manager's httpx module with only the engine client replaced."""

    def __init__(self, holder):
        self.AsyncClient = lambda *a, **k: _EngineClient(holder)
        self.Timeout = lambda *a, **k: None

    def __getattr__(self, name):
        return getattr(_real_httpx, name)


@dataclass
class Box:
    mgr: TurbohaulManager
    kv_dir: object
    t0: float
    holds: dict = field(default_factory=dict)          # label -> Event the fake engine waits on
    prefill_holds: dict = field(default_factory=dict)  # label -> Event the prefill step waits on
    events: list = field(default_factory=list)
    tasks: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    kv_at_stop: dict = field(default_factory=dict)     # model tag -> snapshot of the save dir at engine stop
    search_passes: int = 0
    search_returned: list = field(default_factory=list)

    # -- event list ---------------------------------------------------------------------------
    def stamp(self, kind, name):
        self.events.append((time.monotonic(), kind, name))

    def idx(self, kind, name):
        for i, (_t, k, n) in enumerate(self.events):
            if (k, n) == (kind, name):
                return i
        return None

    def seen(self, kind, name):
        return self.idx(kind, name) is not None

    def when(self, kind, name):
        i = self.idx(kind, name)
        return None if i is None else self.events[i][0]

    def rel(self):
        return [(round(t - self.t0, 3), k, n) for t, k, n in self.events]

    async def until(self, predicate, what, timeout=8.0):
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"never happened within {timeout}s: {what}; events={self.rel()}")
            await asyncio.sleep(0.005)

    def one_model_fits(self, tag):
        need, parallel, main_gpu, split_mode, *_ = self.mgr._resolve_placement_locked(tag)
        return self.mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)

    # -- clients ------------------------------------------------------------------------------
    def submit(self, label, tag, meta, thread, hold=False, prefill_hold=False):
        if hold:
            self.holds[label] = asyncio.Event()
        if prefill_hold:
            self.prefill_holds[label] = asyncio.Event()

        async def run():
            res = await self.mgr.submit_and_wait(
                tag, label, thread_id=thread, client_meta=dict(meta))
            self.stamp("caller_got_response", label)
            return res

        task = asyncio.create_task(asyncio.wait_for(run(), timeout=60))
        self.tasks[label] = task
        return task

    def submit_streamed(self, label, tag, meta, thread):
        """A streaming client: it takes the engine handle, emits a token every STREAM_INTERVAL_S
        (the slow generation under test, never held), then reports the stream finished."""
        self.tokens[label] = []

        async def run():
            slot = await self.mgr.submit_for_streaming(
                tag, label, thread_id=thread, client_meta=dict(meta, stream=True))
            await asyncio.wait_for(slot.stream_ready_event.wait(), 30)
            self.stamp("stream_open", label)
            for i in range(STREAM_TOKENS):
                await asyncio.sleep(STREAM_INTERVAL_S)       # pacing of the fake engine's output
                self.tokens[label].append(f"tok{i}")
                self.stamp("token", label)
            self.stamp("turn_end", label)
            slot.stream_done_event.set()
            result = await asyncio.wait_for(slot.completion_future, 10)
            self.stamp("caller_got_response", label)
            return slot, result

        task = asyncio.create_task(asyncio.wait_for(run(), timeout=60))
        self.tasks[label] = task
        return task


def _manifest(boot, tag):
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": FOOTPRINT_MIB * 1024 * 1024, "context_size": 2048,
        "expected_vram_bytes": FOOTPRINT_MIB * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


def _kv_snapshot(kv_dir):
    """Finished save artifacts present right now: {bin name: size} and parsed metadata sidecars."""
    bins, metas = {}, {}
    for p in sorted(kv_dir.iterdir()):
        if p.name.endswith(".tmp") or ".tmp." in p.name:
            continue
        if p.name.endswith(".bin"):
            bins[p.name] = p.stat().st_size
        elif p.name.endswith(".json"):
            with contextlib.suppress(Exception):
                metas[p.name] = json.loads(p.read_text())
    return {"bins": bins, "metas": metas}


@contextlib.asynccontextmanager
async def box(tmp_path, monkeypatch, *, grace_s):
    import turbohaul.manager as manager_mod
    import turbohaul.subprocess_mgr as subprocess_mgr

    kv_dir = tmp_path / "kvcache"
    kv_dir.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(kv_dir))
    holder: dict = {}
    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx(holder))

    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=CLIENT_1_FIRST, max_parallel_sidecars=2, grace_seconds=grace_s,
        max_grace_extensions=5, idle_hot_load_seconds=120)
    for t in (TAG_A, TAG_B):
        _manifest(boot, t)
    spawn, health, _sigterm, vram, _complete = make_fakes({})

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        holder["box"].stamp("load", model_tag)
        return spawn(binary, gguf, port, model_tag, argv, **kw)

    async def sigterm_rec(handle, *a, **k):
        b = holder["box"]
        tag = getattr(handle, "model_tag", "?")
        b.kv_at_stop[tag] = _kv_snapshot(b.kv_dir)
        b.stamp("engine_stop", tag)
        return True, "sigterm-clean"

    async def complete_rec(slot, handle):
        b = holder["box"]
        label = slot.prompt
        b.stamp("turn_start", label)
        gate = b.holds.get(label)
        try:
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            b.stamp("turn_cancelled", label)
            raise
        b.stamp("turn_end", label)
        return {"ok": True, "label": label, "model": handle.model_tag, "text": f"reply-to-{label}"}

    def free_vram():
        loaded = [r for r in holder["mgr"]._model_residents() if r.state in LOADED]
        return [CARD_MIB - FOOTPRINT_MIB * len(loaded)]

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = Box(mgr=mgr, kv_dir=kv_dir, t0=time.monotonic())
    holder["mgr"], holder["box"] = mgr, b

    # -- spies (observe only; each calls the real method) -------------------------------------
    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):
        b.stamp("unload_decided", r.model_tag)
        return orig_unload(r)

    mgr._begin_unload_locked = unload_spy

    orig_teardown = mgr._unload_teardown

    async def teardown_spy(r):
        b.stamp("unload_teardown_begin", r.model_tag)
        return await orig_teardown(r)

    mgr._unload_teardown = teardown_spy

    orig_save = mgr._save_slot_kv

    async def save_spy(port, model_tag, *a, **k):
        b.stamp("kv_save_begin", model_tag)
        out = await orig_save(port, model_tag, *a, **k)
        b.stamp("kv_save_end", f"{model_tag}|{bool(out)}")
        return out

    mgr._save_slot_kv = save_spy

    orig_mark_active = mgr._mark_slot_active

    def mark_active_spy(slot):
        fut = getattr(slot, "completion_future", None)
        if fut is not None:
            label = slot.prompt
            fut.add_done_callback(lambda _f, _l=label: b.stamp("response_set", _l))
        return orig_mark_active(slot)

    mgr._mark_slot_active = mark_active_spy

    orig_probe = mgr._probe_and_save_clean_kv

    async def probe_spy(handle, slot):
        label = slot.prompt
        b.stamp("prefill_start", label)
        gate = b.prefill_holds.get(label)
        if gate is not None:
            await gate.wait()
        return await orig_probe(handle, slot)

    mgr._probe_and_save_clean_kv = probe_spy

    orig_search = mgr._lru_idle_unloadable

    def search_spy(*a, **k):
        out = orig_search(*a, **k)
        b.search_passes += 1
        b.search_returned.append(getattr(out, "model_tag", None))
        return out

    mgr._lru_idle_unloadable = search_spy

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # Release every held turn, stop the loop, then cancel the worker; every await is bounded.
        for g in list(b.holds.values()) + list(b.prefill_holds.values()):
            g.set()
        mgr._stop_event.set()
        mgr._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(mgr._worker_task, 10)
        for t in b.tasks.values():
            if not t.done():
                t.cancel()
        for t in b.tasks.values():
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(t, 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(mgr.shutdown(), 10)


async def _park_after_first_turn(b):
    """A (client 2) serves one turn to completion and its grace window ends: A is parked and
    evictable, with a grace window that has started at least once (the state of a second turn)."""
    mgr = b.mgr
    b.submit("A1", TAG_A, CLIENT_2_META, "th-a")
    await b.until(lambda: b.seen("caller_got_response", "A1"), "A's first turn to be delivered")
    a = resident_for(mgr, TAG_A)
    await b.until(lambda: a.grace is not None and a.grace._started_at is not None,
                  "A's grace window to start")
    assert a.grace._started_at is not None
    await b.until(lambda: a.in_grace_loop, "A to be inside its grace loop")
    await b.until(lambda: (not a.in_grace_loop) and a.state is ResidentState.IDLE_EVICTABLE,
                  "A's grace window to end and A to be parked", 12.0)
    assert a.grace._started_at is not None, "the grace window start was cleared"
    assert not a.in_grace_loop and a.state is ResidentState.IDLE_EVICTABLE, (
        f"A is not parked before its next turn: {a.state}")
    assert not b.seen("unload_decided", TAG_A), "A was unloaded before the turn under test"
    return a


async def _hold_turn_then_claim(b, mode, *, second_turn):
    """Put A mid-turn (held or slowly streaming), then register a better client's claim for B."""
    mgr = b.mgr
    if second_turn:
        a = await _park_after_first_turn(b)
        label, thread = "A2", "th-a"
    else:
        a, label, thread = None, "A1", "th-a"
    if mode == STREAM:
        b.submit_streamed(label, TAG_A, CLIENT_2_META, thread)
        await b.until(lambda: b.seen("stream_open", label), "A's streamed turn to open")
    elif mode == PREFILL:
        b.submit(label, TAG_A, CLIENT_2_META, thread, prefill_hold=True)
        await b.until(lambda: b.seen("prefill_start", label), "A's turn to reach its prefill step")
    else:
        b.submit(label, TAG_A, CLIENT_2_META, thread, hold=True)
        await b.until(lambda: b.seen("turn_start", label), "A's held turn to start")
    a = resident_for(mgr, TAG_A)
    assert a.state is ResidentState.ACTIVE, f"A is not mid-turn: {a.state}"
    assert not b.tasks[label].done(), "A's turn already finished"
    assert a.active_slot is not None, "A has no active slot while mid-turn"
    if mode == PREFILL:
        # before any byte of the turn: the engine has not been asked to generate yet
        assert a.active_slot.engine_op == "prefill", (
            f"the held turn is not in its prefill step: {a.active_slot.engine_op}")
        assert not b.seen("turn_start", label), "the held turn already started generating"
    if mode == STREAM:
        assert not b.seen("turn_end", label), "the stream already ended"
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
    assert not _kv_snapshot(b.kv_dir)["bins"], "a KV blob exists before any unload"
    b.stamp("claimant_submit", "M1")
    b.submit("M1", TAG_B, CLIENT_1_META, "th-main")
    await b.until(lambda: len(mgr._fastlane_claims) > 0, "the claimant's claim to register")
    return label, a


def _boundary(b, label):
    """Position in the event list where the running turn is over: the hold release (held turns) or
    the end of the stream. Events stamped before it happened while the turn was still running;
    events stamped after it are the legitimate hand-over. No boundary yet = the whole list."""
    marks = [i for i in (b.idx("release", label), b.idx("turn_end", label)) if i is not None]
    return min(marks) if marks else len(b.events)


def _cut_reasons(b, label, a, mode):
    """Decide from the STAMPED event order, never from what a poll happened to see together: a cut
    is an unload, teardown, engine stop, cancel or early end stamped BEFORE the turn's boundary."""
    limit = _boundary(b, label)
    reasons = []
    for kind, name, text in (("unload_decided", TAG_A, "unload_decided(A)"),
                             ("unload_teardown_begin", TAG_A, "unload_teardown_begin(A)"),
                             ("engine_stop", TAG_A, "engine_stop(A)"),
                             ("turn_cancelled", label, "turn_cancelled")):
        i = b.idx(kind, name)
        if i is not None and i < limit:
            reasons.append(text)
    # Resident state is not stamped, so it counts only while no boundary has been stamped: the
    # state and the event list are read in the same step, so DEAD with the turn still running is
    # a cut and DEAD after the turn's end is the hand-over.
    if a.state is ResidentState.DEAD and limit == len(b.events):
        reasons.append("A.state=DEAD")
    if mode != STREAM:
        i = b.idx("turn_end", label)
        if i is not None and i < limit:
            reasons.append("turn_ended_early")
    return reasons


async def _observe_no_cut(b, label, a, mode):
    """Watch the running turn. Held turns: for OBSERVE_S seconds. Streamed turn: until the stream
    ends on its own. The live poll only fails fast; the verdict is re-derived from the stamped
    event order once the window is over, so poll lag can never produce a false cut."""
    t0 = time.monotonic()
    while True:
        reasons = _cut_reasons(b, label, a, mode)
        if reasons:
            return reasons
        if mode == STREAM:
            if b.seen("turn_end", label):
                break
        elif time.monotonic() - t0 >= OBSERVE_S:
            break
        await asyncio.sleep(0.01)
    return _cut_reasons(b, label, a, mode)


def _release(b, label, mode):
    """End the held turn. A streamed turn ends by itself, so there is nothing to release."""
    if mode == STREAM:
        return
    b.stamp("release", label)
    (b.prefill_holds if mode == PREFILL else b.holds)[label].set()


async def _assert_not_cut_then_ordered_handover(b, label, a, mode, case):
    cut = await _observe_no_cut(b, label, a, mode)
    assert not cut, (
        f"{case}: the running turn was CUT while it was still running: {cut}; "
        f"victim_search_passes={b.search_passes} returned={b.search_returned}; events={b.rel()}")
    if mode == STREAM:
        opened, ended = b.when("stream_open", label), b.when("turn_end", label)
        assert ended - opened >= 3.0, f"{case}: the stream was shorter than 3 s: {ended - opened}"
        assert len(b.tokens[label]) >= 10, f"{case}: too few tokens streamed: {b.tokens[label]}"
    _release(b, label, mode)

    # the response of the running turn is delivered intact
    res = await asyncio.wait_for(b.tasks[label], 10)
    if mode == STREAM:
        _slot, result = res
        assert result == {"_streamed": True}, f"{case}: the streamed turn did not complete: {result!r}"
        assert b.tokens[label] == [f"tok{i}" for i in range(STREAM_TOKENS)], (
            f"{case}: the streamed content is not intact: {b.tokens[label]}")
    else:
        assert res[1] == {"ok": True, "label": label, "model": TAG_A, "text": f"reply-to-{label}"}, (
            f"{case}: the held turn did not complete intact: {res[1]!r}")

    await _assert_teardown_saves_kv_then_hands_over(b, case, label)


async def _assert_teardown_saves_kv_then_hands_over(b, case, label):
    """A's unload, KV save, engine stop, B's load and the claimant's turn happen in this order,
    and A's KV artifact is on disk when A's engine stops. `label` is the turn A was running when
    the claim arrived (None for an idle resident)."""
    await b.until(lambda: b.seen("turn_end", "M1") and b.seen("caller_got_response", "M1"),
                  "the claimant's turn to end and be delivered", 15.0)
    ev = b.rel()

    def need(kind, name):
        i = b.idx(kind, name)
        assert i is not None, f"{case}: event never happened: {(kind, name)}; events={ev}"
        return i

    i_decided = need("unload_decided", TAG_A)
    i_save_begin = need("kv_save_begin", TAG_A)
    i_save_end = need("kv_save_end", f"{TAG_A}|True")
    i_stop = need("engine_stop", TAG_A)
    # The teardown of A is carried out either by the detached teardown task or by the resident's
    # own driver on its way out; whichever runs first exactly once, so the start of the teardown
    # is the earliest of the three observable steps (task start, KV save call, engine stop).
    i_begin = min(i for i in (b.idx("unload_teardown_begin", TAG_A), i_save_begin, i_stop)
                  if i is not None)
    i_load = need("load", TAG_B)
    i_m1 = need("turn_end", "M1")

    if label is not None:
        i_end = need("turn_end", label)
        i_set = need("response_set", label)
        i_got = need("caller_got_response", label)
        assert i_end < i_decided, f"{case}: A's unload was decided before its turn ended; events={ev}"
        assert i_set < i_begin, (
            f"{case}: A's teardown began before its turn's response was delivered; events={ev}")
        assert i_end < i_got, f"{case}: the response was delivered before the turn ended; events={ev}"
    if label is None:
        assert i_decided < i_save_begin, (
            f"{case}: A's KV save was called before its unload was decided; events={ev}")
    assert i_begin <= i_save_begin < i_save_end < i_stop, (
        f"{case}: A's KV was not saved (and confirmed) before the engine stop; events={ev}")
    assert i_stop < i_load, f"{case}: B loaded before A's engine stopped; events={ev}"
    assert i_load < i_m1, f"{case}: the claimant's turn ran before B loaded; events={ev}"

    # the KV artifact itself, as it stood when A's engine was stopped
    snap = b.kv_at_stop.get(TAG_A)
    assert snap is not None, f"{case}: no snapshot of the save directory at A's engine stop"
    a_bins = {n: s for n, s in snap["bins"].items() if n.startswith(f"{TAG_A}.")}
    assert a_bins and all(s == len(KV_BYTES) for s in a_bins.values()), (
        f"{case}: A's KV blob was not in the store when its engine stopped: {snap['bins']}")
    a_metas = [m for n, m in snap["metas"].items() if n.startswith(f"{TAG_A}.")]
    assert a_metas and all(m.get("model_tag") == TAG_A for m in a_metas), (
        f"{case}: A's KV metadata was not in the store when its engine stopped: {snap['metas']}")


class TestUnloadNeverCutsRunningTurn:
    async def test_second_turn_is_not_cut_by_a_better_client_claim(self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            label, a = await _hold_turn_then_claim(b, DECODE, second_turn=True)
            await _assert_not_cut_then_ordered_handover(b, label, a, DECODE, "second-turn-decode")

    async def test_second_turn_held_before_first_byte_is_not_cut(self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            label, a = await _hold_turn_then_claim(b, PREFILL, second_turn=True)
            await _assert_not_cut_then_ordered_handover(b, label, a, PREFILL, "second-turn-prefill")

    async def test_slow_streaming_turn_stays_protected(self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            label, a = await _hold_turn_then_claim(b, STREAM, second_turn=True)
            await _assert_not_cut_then_ordered_handover(b, label, a, STREAM, "second-turn-stream")

    async def test_first_turn_is_not_cut(self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=30) as b:
            label, a = await _hold_turn_then_claim(b, DECODE, second_turn=False)
            await _assert_not_cut_then_ordered_handover(b, label, a, DECODE, "first-turn-decode")

    async def test_idle_resident_unloaded_for_a_claimant_has_its_kv_saved(self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            a = await _park_after_first_turn(b)
            assert a.state is ResidentState.IDLE_EVICTABLE, f"A is not idle: {a.state}"
            assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
            assert not _kv_snapshot(b.kv_dir)["bins"], "a KV blob exists before any unload"
            b.stamp("claimant_submit", "M1")
            b.submit("M1", TAG_B, CLIENT_1_META, "th-main")
            await b.until(lambda: b.seen("turn_end", "M1") and b.seen("caller_got_response", "M1"),
                          "the claimant's turn to end and be delivered", 15.0)
            await _assert_teardown_saves_kv_then_hands_over(b, "idle-unload", None)
