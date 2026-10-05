"""A follow-up that has not started is never lost when its resident is torn down.

Two Fast Lane clients are configured on a card that holds exactly one model. Client 1 is listed
first, so it is the better client; client 2 owns the resident model A. A better client asks for a
different model B. A finished a turn earlier, its grace window has ended and it is the designated
victim of the better client's claim.

A follow-up for A that is only waiting in A's inbox has not started its turn: it is not protected,
so A may be torn down underneath it. What must hold is that the follow-up is NOT LOST: it is put
back on the queue, A's model is loaded again after the better client has run, and the follow-up is
served with its own content. A turn that HAS started (the engine is working on it) is different: it
is protected, and only after it has ended may the resident unload and hand the waiting follow-up
back to the queue.

The cases:

  * the follow-up sits in the inbox, the resident's driver has not taken it yet, the claimant tears
    the resident down: the follow-up is re-queued and served after the claimant;
  * control: a turn that is running is protected while another follow-up waits behind it; after the
    running turn ends the resident unloads and the waiting follow-up is re-queued and served;
  * the narrow window: the driver has already taken the follow-up out of the inbox but has not yet
    marked it active (it is about to take the registry lock) when the claimant's make-room tears
    the resident down. The follow-up is not started, so it is re-queued exactly like one still in
    the inbox: it is neither failed nor lost, and is served after the claimant. Run in both orders
    in which the driver's own exit and the detached teardown can claim the exit;
  * the same window with the queue closed: re-queueing is impossible, so the follow-up is failed
    with a visible error and is not left waiting.

Every case asserts WHICH events happened and in WHAT ORDER from one append-only event list (list
position is the order; the loop is single-threaded so position is exact). The engine is a fake HTTP
client installed in place of the manager's httpx module; the KV save is asserted by the artifact it
leaves in a temporary save directory, readable at the moment the engine is stopped.
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
RESOLVE_S = 30.0          # bound on how long a follow-up may stay unresolved before it counts as lost
KV_BYTES = b"kv-cache-bytes-" * 64


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
    events: list = field(default_factory=list)
    tasks: dict = field(default_factory=dict)
    kv_at_stop: dict = field(default_factory=dict)     # model tag -> snapshot of the save dir at engine stop
    decisions: list = field(default_factory=list)      # what the resident looked like when it was unloaded
    search_returned: list = field(default_factory=list)
    releasers: list = field(default_factory=list)      # callables run first at teardown

    # -- event list ---------------------------------------------------------------------------
    def stamp(self, kind, name):
        self.events.append((time.monotonic(), kind, name))

    def idxs(self, kind, name):
        return [i for i, (_t, k, n) in enumerate(self.events) if (k, n) == (kind, name)]

    def idx(self, kind, name):
        found = self.idxs(kind, name)
        return found[0] if found else None

    def seen(self, kind, name):
        return self.idx(kind, name) is not None

    def count(self, kind, name):
        return len(self.idxs(kind, name))

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
    def submit(self, label, tag, meta, thread, hold=False):
        if hold:
            self.holds[label] = asyncio.Event()

        async def run():
            res = await self.mgr.submit_and_wait(
                tag, label, thread_id=thread, client_meta=dict(meta))
            self.stamp("caller_got_response", label)
            return res

        task = asyncio.create_task(asyncio.wait_for(run(), timeout=90))
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
        b.kv_at_stop.setdefault(tag, _kv_snapshot(b.kv_dir))
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
        # what the resident looked like at the instant it was chosen for unload
        b.decisions.append({
            "tag": r.model_tag, "state": r.state, "active_slot_none": r.active_slot is None,
            "inflight": len(r.inflight), "inbox": r.inbox.qsize() if r.inbox is not None else None,
        })
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

    orig_search = mgr._lru_idle_unloadable

    def search_spy(*a, **k):
        out = orig_search(*a, **k)
        b.search_returned.append(getattr(out, "model_tag", None))
        return out

    mgr._lru_idle_unloadable = search_spy

    # -- the re-queue path: the drain task, both queue entry points, and any failed future -----
    orig_requeue = mgr._requeue_slots_or_fail

    async def requeue_spy(slots):
        for s in slots:
            b.stamp("requeue_drained_head", s.prompt)
        return await orig_requeue(slots)

    mgr._requeue_slots_or_fail = requeue_spy

    orig_requeue_tail = mgr._requeue_slots_tail_or_fail

    async def requeue_tail_spy(slots):
        for s in slots:
            b.stamp("requeue_drained_tail", s.prompt)
        return await orig_requeue_tail(slots)

    mgr._requeue_slots_tail_or_fail = requeue_tail_spy

    orig_enq_head = mgr.queue.enqueue_head

    async def enq_head_spy(slot):
        out = await orig_enq_head(slot)
        b.stamp("enqueue_head", slot.prompt)
        return out

    mgr.queue.enqueue_head = enq_head_spy

    orig_enq_tail = mgr.queue.enqueue_tail

    async def enq_tail_spy(slot):
        out = await orig_enq_tail(slot)
        b.stamp("enqueue_tail", slot.prompt)
        return out

    mgr.queue.enqueue_tail = enq_tail_spy

    orig_fail = mgr._fail_completion_future

    def fail_spy(slot, exc):
        b.stamp("future_failed", f"{slot.prompt}|{type(exc).__name__}: {exc}")
        return orig_fail(slot, exc)

    mgr._fail_completion_future = fail_spy

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # Release every held turn and gate, stop the loop, then cancel the worker; every await is bounded.
        for rel in b.releasers:
            with contextlib.suppress(Exception):
                rel()
        for g in list(b.holds.values()):
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


async def _first_turn(b):
    """A (client 2) serves one turn to completion; return its resident while the grace window is
    still running (the grace window has started)."""
    b.submit("A1", TAG_A, CLIENT_2_META, "th-a")
    await b.until(lambda: b.seen("caller_got_response", "A1"), "A's first turn to be delivered")
    a = resident_for(b.mgr, TAG_A)
    await b.until(lambda: a.grace is not None and a.grace._started_at is not None,
                  "A's grace window to start")
    return a


async def _wait_parked(b, a):
    """A's grace window ends and A is parked and evictable, with a grace window that has started
    at least once."""
    await b.until(lambda: (not a.in_grace_loop) and a.state is ResidentState.IDLE_EVICTABLE,
                  "A's grace window to end and A to be parked", 12.0)
    assert a.grace._started_at is not None, "the grace window start was cleared"
    assert not a.in_grace_loop and a.state is ResidentState.IDLE_EVICTABLE, (
        f"A is not parked before its next turn: {a.state}")
    assert not b.seen("unload_decided", TAG_A), "A was unloaded before the case under test"


async def _claim(b):
    """Register the better client's claim for B (the card cannot hold both models)."""
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
    assert not _kv_snapshot(b.kv_dir)["bins"], "a KV blob exists before any unload"
    b.stamp("claimant_submit", "M1")
    b.submit("M1", TAG_B, CLIENT_1_META, "th-main")
    await b.until(lambda: len(b.mgr._fastlane_claims) > 0, "the claimant's claim to register")


def _fail_stamps(b, label):
    return [n for _t, k, n in b.events if k == "future_failed" and n.startswith(f"{label}|")]


async def _resolved(b, label, case, bound=RESOLVE_S):
    """Wait (bounded) for the caller of ``label`` to get an outcome. Returns ('ok', result) or
    ('error', exception). A caller still waiting after the bound means the request was lost."""
    task = b.tasks[label]
    done, _pending = await asyncio.wait({task}, timeout=bound)
    assert done, (
        f"{case}: DROP - the request {label} was neither served nor failed within {bound}s "
        f"(its caller is still waiting); fail_stamps={_fail_stamps(b, label)}; events={b.rel()}")
    exc = task.exception()
    if exc is not None:
        return "error", exc
    return "ok", task.result()


def _need(b, case, kind, name):
    i = b.idx(kind, name)
    assert i is not None, f"{case}: event never happened: {(kind, name)}; events={b.rel()}"
    return i


def _assert_kv_saved_before_stop(b, case, i_decided):
    """A's KV was saved (and confirmed) before A's engine stopped, and the artifact itself was in
    the store at the moment the engine was stopped."""
    ev = b.rel()
    i_save_begin = _need(b, case, "kv_save_begin", TAG_A)
    i_save_end = _need(b, case, "kv_save_end", f"{TAG_A}|True")
    i_stop = _need(b, case, "engine_stop", TAG_A)
    i_begin = min(i for i in (b.idx("unload_teardown_begin", TAG_A), i_save_begin, i_stop)
                  if i is not None)
    assert i_decided < i_begin <= i_save_begin < i_save_end < i_stop, (
        f"{case}: A's KV was not saved (and confirmed) before the engine stop; events={ev}")
    snap = b.kv_at_stop.get(TAG_A)
    assert snap is not None, f"{case}: no snapshot of the save directory at A's engine stop"
    a_bins = {n: s for n, s in snap["bins"].items() if n.startswith(f"{TAG_A}.")}
    assert a_bins and all(s == len(KV_BYTES) for s in a_bins.values()), (
        f"{case}: A's KV blob was not in the store when its engine stopped: {snap['bins']}")
    a_metas = [m for n, m in snap["metas"].items() if n.startswith(f"{TAG_A}.")]
    assert a_metas and all(m.get("model_tag") == TAG_A for m in a_metas), (
        f"{case}: A's KV metadata was not in the store when its engine stopped: {snap['metas']}")
    return i_stop


def _assert_served_after_claimant(b, case, label, i_stop):
    """The follow-up ran on a fresh load of A, after the claimant's turn, with its own content."""
    ev = b.rel()
    i_load_b = _need(b, case, "load", TAG_B)
    i_m1 = _need(b, case, "turn_end", "M1")
    loads_a = b.idxs("load", TAG_A)
    assert len(loads_a) == 2, f"{case}: A's model was not loaded again after the claimant: {ev}"
    i_start = _need(b, case, "turn_start", label)
    i_set = _need(b, case, "response_set", label)
    i_got = _need(b, case, "caller_got_response", label)
    assert i_stop < i_load_b < i_m1, f"{case}: claimant ran out of order; events={ev}"
    assert i_m1 < loads_a[1] < i_start < i_set <= i_got, (
        f"{case}: the follow-up was not served on a fresh load of A after the claimant's turn; "
        f"events={ev}")


# =========================================================================================
class TestInboxFollowUpSurvivesVictimTeardown:
    async def test_inbox_follow_up_is_requeued_not_dropped_when_the_victim_is_torn_down(
            self, tmp_path, monkeypatch):
        case = "inbox-follow-up"
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            a = await _first_turn(b)

            # The driver takes nothing out of A's inbox until released: it waits inside the
            # inbox read it would otherwise be parked in. The wrapper is in place before the
            # grace window ends, so the driver's park goes through it.
            assert a.in_grace_loop, "A's grace window already ended; the inbox read is not yet armed"
            gate, entered = asyncio.Event(), asyncio.Event()
            real_get = a.inbox.get

            async def gated_get():
                b.stamp("driver_waits_on_inbox", a.model_tag)
                entered.set()
                await gate.wait()
                item = await real_get()
                b.stamp("driver_took_from_inbox", item.prompt)
                return item

            a.inbox.get = gated_get
            b.releasers.append(gate.set)
            await _wait_parked(b, a)
            await b.until(entered.is_set, "A's driver to wait on its inbox")
            assert a.grace._started_at is not None and not a.in_grace_loop

            # A same-client follow-up enters by the normal route and lands in A's inbox.
            b.submit("A2", TAG_A, CLIENT_2_META, "th-a")
            await b.until(lambda: a.inbox.qsize() == 1, "the follow-up to land in A's inbox")
            assert a.state is ResidentState.ACTIVE, f"A is not active: {a.state}"
            assert a.active_slot is None and not a.inflight, "A has a running turn; setup is wrong"
            assert not b.seen("turn_start", "A2"), "the follow-up already started"
            assert not b.seen("driver_took_from_inbox", "A2"), "the driver already took the follow-up"
            assert not b.tasks["A2"].done()

            await _claim(b)
            await b.until(lambda: b.seen("unload_decided", TAG_A),
                          "A to be chosen for unload by the claimant's make-room", 15.0)
            # the decision was taken on the resident the way the setup built it
            d = b.decisions[0]
            assert d["tag"] == TAG_A and d["state"] is ResidentState.ACTIVE, (
                f"{case}: the unloaded resident was not the active, grace-ended one: {d}")
            assert d["active_slot_none"] and d["inflight"] == 0 and d["inbox"] == 1, (
                f"{case}: the unload did not find the waiting follow-up in the inbox: {d}")
            assert TAG_A in b.search_returned, f"{case}: the victim search never returned A"

            kind, res = await _resolved(b, "A2", case)
            ev = b.rel()
            assert kind == "ok", (
                f"{case}: the follow-up's caller got an error instead of the reply: {res!r}; "
                f"fail_stamps={_fail_stamps(b, 'A2')}; events={ev}")
            assert res[1] == {"ok": True, "label": "A2", "model": TAG_A, "text": "reply-to-A2"}, (
                f"{case}: the follow-up was not served intact: {res[1]!r}; events={ev}")
            assert not _fail_stamps(b, "A2"), f"{case}: the follow-up's future was failed: {ev}"

            # re-queued exactly once by the drain, and entered the queue
            assert b.count("requeue_drained_head", "A2") + b.count("requeue_drained_tail", "A2") == 1, (
                f"{case}: the follow-up was not handed to the re-queue path exactly once: {ev}")
            i_decided = _need(b, case, "unload_decided", TAG_A)
            i_enq = min(i for i in (b.idx("enqueue_head", "A2"), b.idx("enqueue_tail", "A2"))
                        if i is not None) if (b.seen("enqueue_head", "A2")
                                              or b.seen("enqueue_tail", "A2")) else None
            assert i_enq is not None and i_enq > i_decided, (
                f"{case}: the follow-up never re-entered the queue after the unload decision: {ev}")
            assert not b.seen("driver_took_from_inbox", "A2"), (
                f"{case}: the driver took the follow-up after all (setup lost its meaning): {ev}")

            await b.until(lambda: b.seen("caller_got_response", "M1"), "the claimant's reply", 15.0)
            i_stop = _assert_kv_saved_before_stop(b, case, i_decided)
            assert not b.seen("turn_start", "A2") or b.idx("turn_start", "A2") > i_stop, (
                f"{case}: the follow-up ran on the engine that was being torn down: {ev}")
            _assert_served_after_claimant(b, case, "A2", i_stop)

    async def test_started_turn_is_protected_while_the_inbox_has_a_waiting_follow_up(
            self, tmp_path, monkeypatch):
        case = "started-turn-protected"
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            a = await _first_turn(b)
            await _wait_parked(b, a)

            # A's second turn starts and is held in the engine; another follow-up waits behind it.
            b.submit("A2", TAG_A, CLIENT_2_META, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "A2"), "A's second turn to start")
            assert a.state is ResidentState.ACTIVE and a.active_slot is not None, (
                f"A is not mid-turn: {a.state}")
            b.submit("A3", TAG_A, CLIENT_2_META, "th-a")
            await b.until(lambda: a.inbox.qsize() == 1, "the second follow-up to wait in A's inbox")
            assert not b.seen("turn_start", "A3"), "the waiting follow-up already started"
            assert not b.tasks["A2"].done()

            await _claim(b)

            # While the turn is held nothing of A is cut, however long the claimant waits.
            t0 = time.monotonic()
            while time.monotonic() - t0 < OBSERVE_S:
                cut = [n for n in ("unload_decided", "unload_teardown_begin", "engine_stop")
                       if b.seen(n, TAG_A)]
                if b.seen("turn_cancelled", "A2"):
                    cut.append("turn_cancelled(A2)")
                if a.state is ResidentState.DEAD:
                    cut.append("A.state=DEAD")
                if b.tasks["A2"].done():
                    cut.append("held_task_done")
                assert not cut, (
                    f"{case}: the running turn was CUT while it was still running: {cut}; "
                    f"events={b.rel()}")
                await asyncio.sleep(0.01)
            assert a.inbox.qsize() == 1, f"{case}: the waiting follow-up left the inbox early"
            assert not _fail_stamps(b, "A3"), f"{case}: the waiting follow-up's future was failed early"

            b.stamp("release", "A2")
            b.holds["A2"].set()
            res = await asyncio.wait_for(b.tasks["A2"], 15)
            assert res[1] == {"ok": True, "label": "A2", "model": TAG_A, "text": "reply-to-A2"}, (
                f"{case}: the held turn did not complete intact: {res[1]!r}")

            kind, res3 = await _resolved(b, "A3", case)
            ev = b.rel()
            assert kind == "ok", (
                f"{case}: the waiting follow-up's caller got an error: {res3!r}; "
                f"fail_stamps={_fail_stamps(b, 'A3')}; events={ev}")
            assert res3[1] == {"ok": True, "label": "A3", "model": TAG_A, "text": "reply-to-A3"}, (
                f"{case}: the waiting follow-up was not served intact: {res3[1]!r}")
            assert not _fail_stamps(b, "A3"), f"{case}: the waiting follow-up's future was failed: {ev}"
            await b.until(lambda: b.seen("caller_got_response", "M1"), "the claimant's reply", 15.0)

            i_end = _need(b, case, "turn_end", "A2")
            i_set = _need(b, case, "response_set", "A2")
            i_got = _need(b, case, "caller_got_response", "A2")
            i_decided = _need(b, case, "unload_decided", TAG_A)
            assert i_end < i_set <= i_got, f"{case}: delivery order wrong; events={ev}"
            assert i_set < i_decided, (
                f"{case}: A's unload was decided before the running turn's response was set; events={ev}")
            assert b.count("requeue_drained_head", "A3") + b.count("requeue_drained_tail", "A3") == 1, (
                f"{case}: the waiting follow-up was not handed to the re-queue path exactly once: {ev}")
            i_stop = _assert_kv_saved_before_stop(b, case, i_decided)
            _assert_served_after_claimant(b, case, "A3", i_stop)

    @pytest.mark.parametrize("order", [
        "driver_first", "teardown_first",
        "queue_closed_driver_first", "queue_closed_teardown_first",
    ])
    async def test_popped_but_not_started_follow_up_is_not_lost(self, tmp_path, monkeypatch, order):
        closed = order.startswith("queue_closed")
        first = order.removeprefix("queue_closed_")
        case = f"popped-follow-up[{order}]"
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            a = await _first_turn(b)
            await _wait_parked(b, a)
            driver = a.driver_task
            assert driver is not None and not driver.done(), "A's driver is not running"

            # The registry lock, with one observed gate: the next time A's DRIVER asks for it (which,
            # for a parked driver, is right after it takes a follow-up out of the inbox) it waits
            # here, before it queues on the real lock. Every other taker passes straight through.
            mgr = b.mgr
            inner = mgr._registry_lock
            at_lock, gate = asyncio.Event(), asyncio.Event()
            armed = {"task": driver}

            class GatedLock:
                async def __aenter__(self):
                    if armed["task"] is not None and asyncio.current_task() is armed["task"]:
                        armed["task"] = None
                        b.stamp("driver_waits_for_registry_lock", a.model_tag)
                        at_lock.set()
                        await gate.wait()
                        b.stamp("driver_takes_registry_lock", a.model_tag)
                    await inner.acquire()

                async def __aexit__(self, *exc):
                    inner.release()
                    return False

                async def acquire(self):
                    await inner.acquire()
                    return True

                def release(self):
                    inner.release()

                def locked(self):
                    return inner.locked()

            mgr._registry_lock = GatedLock()
            b.releasers.append(gate.set)

            # Right after the unload decision for A (still under the registry lock, before any await):
            # optionally close the queue (re-queueing becomes impossible), and in the driver-first
            # order let the driver go ahead of the detached teardown.
            inner_unload = mgr._begin_unload_locked

            def unload_then(r):
                out = inner_unload(r)
                if r is a:
                    if closed:
                        mgr.queue._closed = True
                        b.stamp("queue_closed", TAG_A)
                    if first == "driver_first":
                        gate.set()
                return out

            mgr._begin_unload_locked = unload_then

            b.submit("A2", TAG_A, CLIENT_2_META, "th-a")
            await b.until(at_lock.is_set, "A's driver to reach the registry lock holding the follow-up")

            # setup state: the follow-up is out of the inbox, held by the driver, and not started
            frame = driver.get_coro().cr_frame
            held = frame.f_locals.get("slot") if frame is not None else None
            assert a.inbox.qsize() == 0, "the follow-up is still in the inbox"
            assert a.state is ResidentState.ACTIVE and a.active_slot is None and not a.inflight, (
                f"A is not in the popped-not-started state: {a.state} {a.active_slot}")
            assert held is not None and held.prompt == "A2", (
                f"the driver is not holding the follow-up: {held!r}")
            assert not b.seen("turn_start", "A2") and not b.tasks["A2"].done()
            assert not b.seen("unload_decided", TAG_A)

            await _claim(b)
            await b.until(lambda: b.seen("unload_decided", TAG_A),
                          "A to be chosen for unload by the claimant's make-room", 15.0)
            d = b.decisions[0]
            assert d["tag"] == TAG_A and d["state"] is ResidentState.ACTIVE and d["active_slot_none"] \
                and d["inbox"] == 0, f"{case}: the unload did not find the popped-not-started state: {d}"
            if first == "teardown_first":
                # the detached teardown claims the exit and stops A's engine BEFORE the driver runs again
                await b.until(lambda: b.seen("engine_stop", TAG_A),
                              "the detached teardown to claim and stop A's engine", 15.0)
                assert a.torn_down, f"{case}: the detached teardown did not claim A first"
                gate.set()
            await b.until(lambda: b.seen("driver_takes_registry_lock", TAG_A),
                          "A's driver to take the registry lock", 10.0)

            kind, res = await _resolved(b, "A2", case)
            ev = b.rel()
            i_decided = _need(b, case, "unload_decided", TAG_A)
            if closed:
                # re-queueing is impossible: the follow-up is failed visibly, never served, never pending
                assert kind == "error" and str(res), (
                    f"{case}: with the queue closed the follow-up must fail with a visible error: "
                    f"{kind} {res!r}; events={ev}")
                assert _fail_stamps(b, "A2"), f"{case}: the follow-up's future was not failed: {ev}"
                assert not b.seen("turn_start", "A2"), f"{case}: the follow-up was served: {ev}"
                assert not b.seen("enqueue_head", "A2") and not b.seen("enqueue_tail", "A2"), (
                    f"{case}: the follow-up re-entered a closed queue: {ev}")
                return

            # the queue is open: the follow-up is re-queued, not failed, and served after the claimant
            assert kind == "ok", (
                f"{case}: the popped follow-up was not served; its caller got {res!r} "
                f"(failed instead of re-queued); fail_stamps={_fail_stamps(b, 'A2')}; events={ev}")
            assert res[1] == {"ok": True, "label": "A2", "model": TAG_A, "text": "reply-to-A2"}, (
                f"{case}: the follow-up was not served intact: {res[1]!r}; events={ev}")
            assert not _fail_stamps(b, "A2"), f"{case}: the follow-up's future was failed: {ev}"
            heads, tails = b.count("requeue_drained_head", "A2"), b.count("requeue_drained_tail", "A2")
            assert sorted((heads, tails)) == [0, 1], (
                f"{case}: the follow-up was not handed to exactly one re-queue path exactly once "
                f"(head={heads}, tail={tails}): {ev}")
            i_enq = [i for i in (b.idx("enqueue_head", "A2"), b.idx("enqueue_tail", "A2")) if i is not None]
            assert i_enq and min(i_enq) > i_decided, (
                f"{case}: the follow-up never re-entered the queue after the unload decision: {ev}")
            await b.until(lambda: b.seen("caller_got_response", "M1"), "the claimant's reply", 20.0)
            i_stop = _assert_kv_saved_before_stop(b, case, i_decided)
            assert not b.seen("turn_start", "A2") or b.idx("turn_start", "A2") > i_stop, (
                f"{case}: the follow-up ran on the engine that was being torn down: {ev}")
            _assert_served_after_claimant(b, case, "A2", i_stop)
