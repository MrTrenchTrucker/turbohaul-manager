"""Reclaim first at the cold-start decision.

A model that fits no single card as things stand used to be given a layer split across the cards even
when one card would take it once the idle models on that card are unloaded. These tests pin the new
order: if any single card holds the model once the idle residents of that card count as free, the
model goes on that one card and those residents are unloaded (their KV saved first, no running turn
ever cut); only when no single card fits even so does the split path run, exactly as before. Rank
decides WHO is unloaded; reclaim decides WHERE the model goes.
"""
import asyncio
import contextlib
import json
import re
import time
from dataclasses import dataclass, field

import httpx as _real_httpx
import pytest
import yaml

from _fastlane_fixture import boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import (
    _STALENESS_GRANT_THRESHOLD_S,
    Resident,
    ResidentState,
    TurbohaulManager,
    _fastlane_census_key,
)
from turbohaul.slot import Slot

pytestmark = pytest.mark.asyncio

IP_1 = "192.0.2.10"          # listed first: the better client; owns the busy and the idle model
IP_2 = "198.51.100.20"       # listed second: the claimant
IP_UNLISTED = "203.0.113.77"  # in no rule
RULES = [FastLaneRule(address=IP_1), FastLaneRule(address=IP_2)]
META_OWNER = {"ip": IP_1, "is_main": True}
META_CLAIMANT = {"ip": IP_2, "is_main": True}
META_UNLISTED = {"ip": IP_UNLISTED, "is_main": True}
IP_3 = "198.51.100.30"       # listed third: ranked below both of the others
RULES3 = [FastLaneRule(address=IP_1), FastLaneRule(address=IP_2), FastLaneRule(address=IP_3)]
META_SECOND = META_CLAIMANT  # the second-listed client
META_THIRD = {"ip": IP_3, "is_main": True}

CARD_MIB = 24000
FLOOR_MIB = 1000
CUDA_RESIDUE_MIB = 1300      # what a one-card model leaves on every card it is not placed on
TAG_BUSY, TAG_IDLE, TAG_IDLE2, TAG_CLAIM = "model-busy", "model-idle", "model-idle-b", "model-claim"
BUSY_MIB, IDLE_MIB, CLAIM_MIB, TOO_BIG_MIB = 11000, 12500, 22800, 26000
SPAN_MIB = 20000             # fits no card as things stand, the two cards together take it (a layer span)
SMALL_IDLE_MIB = 6000        # two of these on one card: unloading ONE of them is enough for the order claimant
ORDER_CLAIM_MIB = 15000      # needs exactly one of the two small idle models gone
KV_BYTES = b"kv-cache-bytes-" * 64
SERVED_BOUND_S = 8.0         # a relocation needs: one deferral (1 s), the teardown, the retry, the turn
STARVED_PASSES = 2           # a control waits for this many make-room passes that ended in "starved"


# ------------------------------------------------------------------------------------------------
# fake engine and world
# ------------------------------------------------------------------------------------------------
class _EngineClient:
    """Stands in for httpx.AsyncClient against the engine: /slots plus the slot save endpoint."""

    def __init__(self, holder):
        self._holder = holder

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
class World:
    mgr: TurbohaulManager
    kv_dir: object
    t0: float
    n_cards: int
    events: list = field(default_factory=list)
    holds: dict = field(default_factory=dict)          # label -> Event the fake turn waits on
    tasks: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)          # label -> the Slot object the routing step saw
    alive: dict = field(default_factory=dict)          # model tag -> (card, MiB) of a running fake engine
    foreign: dict = field(default_factory=dict)        # card -> MiB used by something that is not ours
    steer: dict = field(default_factory=dict)          # card -> MiB hidden while a placement is steered
    kv_at_stop: dict = field(default_factory=dict)     # model tag -> save directory snapshot at engine stop
    spawn_argv: dict = field(default_factory=dict)     # model tag -> argv the engine was started with
    logs: list = field(default_factory=list)
    peak_residents: int = 0                            # most residents registered at any stamped moment
    peak_alive: int = 0                                # most fake engines running at once
    pending_after_unload: dict = field(default_factory=dict)  # tag -> pending-reclaim credit just after its unload was decided
    stop_holds: dict = field(default_factory=dict)     # model tag -> Event the fake engine stop waits on
    route_hooks: dict = field(default_factory=dict)    # label -> function run once, on the first sight of that slot

    def sample_residents(self):
        with contextlib.suppress(Exception):
            self.peak_residents = max(self.peak_residents, len(list(self.mgr._model_residents())))
        self.peak_alive = max(self.peak_alive, len(self.alive))

    # -- event list ---------------------------------------------------------------------------
    def stamp(self, kind, name):
        self.events.append((time.monotonic(), kind, name))
        self.sample_residents()

    def idx(self, kind, name):
        for i, (_t, k, n) in enumerate(self.events):
            if (k, n) == (kind, name):
                return i
        return None

    def seen(self, kind, name):
        return self.idx(kind, name) is not None

    def count(self, kind, name):
        return sum(1 for (_t, k, n) in self.events if (k, n) == (kind, name))

    def rel(self):
        return [(round(t - self.t0, 3), k, n) for t, k, n in self.events]

    def observed(self):
        unloads = [n for (_t, k, n) in self.events if k == "unload_decided"]
        loads = [n for (_t, k, n) in self.events if k == "load"]
        residents = {r.model_tag: (r.state.name, r.main_gpu, r.split_mode) for r in self.mgr._model_residents()}
        gates = []
        for (_t, k, n) in self.events:          # the gate's answers, repeats folded into "xN"
            if k != "gate":
                continue
            if gates and gates[-1][0] == n:
                gates[-1][1] += 1
            else:
                gates.append([n, 1])
        gates = [n if c == 1 else f"{n} x{c}" for n, c in gates]
        return (f"unloads={unloads} loads={loads} residents(state, card, split)={residents} gates={gates} "
                f"starved_passes={self.count('log', 'starved')} "
                f"reclaim_first_chosen={self.count('log', 'reclaim')} reclaim_first_refusals={self.count('log', 'refused')}")

    async def until(self, predicate, what, timeout=10.0):
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"never happened within {timeout}s: {what}; observed: {self.observed()}; "
                    f"events={self.rel()}")
            await asyncio.sleep(0.005)

    # -- the fake box ---------------------------------------------------------------------------
    def need(self, tag):
        return self.mgr._read_model_footprint(tag)[0]

    def alive_mib(self, card):
        return sum(m for (c, m) in self.alive.values() if c == card)

    def free_list(self):
        return [CARD_MIB - self.alive_mib(c) - self.foreign.get(c, 0) - self.steer.get(c, 0)
                for c in range(self.n_cards)]

    def set_free(self, card, free_mib):
        """Make card read exactly free_mib free by adding use that is not ours."""
        self.foreign[card] = CARD_MIB - self.alive_mib(card) - free_mib
        assert self.foreign[card] >= 0, f"card {card} cannot read {free_mib} free"
        assert self.free_list()[card] == free_mib

    def steer_to(self, card):
        self.steer = {c: CARD_MIB for c in range(self.n_cards) if c != card}

    def steer_clear(self):
        self.steer = {}

    # -- clients ------------------------------------------------------------------------------
    def submit(self, label, tag, meta, thread, hold=False):
        if hold:
            self.holds[label] = asyncio.Event()

        async def run():
            res = await self.mgr.submit_and_wait(tag, label, thread_id=thread, client_meta=dict(meta))
            self.stamp("caller_got_response", label)
            return res

        task = asyncio.create_task(asyncio.wait_for(run(), timeout=60))
        self.tasks[label] = task
        return task

    def release(self, label):
        self.stamp("release", label)
        self.holds[label].set()


def _manifest(boot, tag, *, size_mib, main_gpu, auto_place=False, split_mode="none", expected_mib=None):
    """`expected_mib` sets the footprint the manager reserves exactly (it is used whenever it is above
    the file size plus context); without it the footprint follows `size_mib`."""
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": size_mib * 1024 * 1024, "context_size": 2048,
        "expected_vram_bytes": (expected_mib or size_mib) * 1024 * 1024, "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _models(*, claim_mib=CLAIM_MIB, claim_auto=False, claim_split="none", claim_pin=0, with_second_idle=False):
    m = {
        TAG_BUSY: dict(size_mib=BUSY_MIB, main_gpu=0),
        TAG_IDLE: dict(size_mib=IDLE_MIB, main_gpu=1),
        TAG_CLAIM: dict(size_mib=claim_mib, main_gpu=claim_pin, auto_place=claim_auto, split_mode=claim_split),
    }
    if with_second_idle:
        m[TAG_IDLE2] = dict(size_mib=IDLE_MIB, main_gpu=0)
    return m


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


class _LogTap:
    """Captures the manager's log lines into the world's event list as they are emitted."""

    def __init__(self, holder):
        import logging

        class _H(logging.Handler):
            def emit(self_h, record):  # noqa: N805
                b = holder.get("box")
                if b is None:
                    return
                msg = record.getMessage()
                b.logs.append(msg)
                if msg.startswith("reclaim_first_refused "):
                    b.stamp("log", "refused")
                elif msg.startswith("reclaim_first "):
                    b.stamp("log", "reclaim")
                elif msg.startswith("MAKE_ROOM_STARVED "):
                    b.stamp("log", "starved")
                elif msg.startswith("MAKE_ROOM_EVICTION "):
                    b.stamp("log", "eviction")

        self._logging = logging
        self.logger = logging.getLogger("turbohaul.manager")
        self.handler = _H(level=logging.INFO)
        self._old_level = self.logger.level

    def __enter__(self):
        self.logger.addHandler(self.handler)
        self.logger.setLevel(self._logging.INFO)
        return self

    def __exit__(self, *a):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._old_level)


@contextlib.asynccontextmanager
async def world(tmp_path, monkeypatch, *, models, rules=RULES, grace_s=1, n_cards=2, max_parallel=3):
    import turbohaul.manager as manager_mod
    import turbohaul.subprocess_mgr as subprocess_mgr

    kv_dir = tmp_path / "kvcache"
    kv_dir.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(kv_dir))
    holder: dict = {}
    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx(holder))

    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=max_parallel, grace_seconds=grace_s,
        max_grace_extensions=5, idle_hot_load_seconds=120)
    runtime.queue.safety_min_free_vram_mib = FLOOR_MIB
    for tag, spec in models.items():
        _manifest(boot, tag, **spec)
    spawn, health, _sigterm, vram, _complete = make_fakes({})

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        b = holder["box"]
        b.stamp("load", model_tag)
        b.spawn_argv[model_tag] = list(argv)
        card = next((r.main_gpu for r in b.mgr._model_residents() if r.model_tag == model_tag), 0)
        b.alive[model_tag] = (card, b.need(model_tag))
        b.sample_residents()
        return spawn(binary, gguf, port, model_tag, argv, **kw)

    async def sigterm_rec(handle, *a, **k):
        b = holder["box"]
        tag = getattr(handle, "model_tag", "?")
        gate = b.stop_holds.get(tag)
        if gate is not None:
            b.stamp("engine_stop_waiting", tag)
            await gate.wait()
        b.kv_at_stop[tag] = _kv_snapshot(b.kv_dir)
        b.stamp("engine_stop", tag)
        b.alive.pop(tag, None)          # the freed MiB appear in the next reading from here on
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
        return holder["box"].free_list()

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = World(mgr=mgr, kv_dir=kv_dir, t0=time.monotonic(), n_cards=n_cards)
    holder["mgr"], holder["box"] = mgr, b

    # -- spies (observe only; each calls the real method) -------------------------------------
    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):
        b.stamp("unload_decided", r.model_tag)
        out = orig_unload(r)
        # the credit as it stands the moment the decision is made, before any teardown step ran
        b.pending_after_unload[r.model_tag] = dict(mgr._pending_reclaim_mib)
        b.sample_residents()
        return out

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

    orig_gate = mgr._vram_admits_locked

    def gate_spy(*a, **k):
        out = orig_gate(*a, **k)
        split = a[3] if len(a) > 3 else k.get("split_mode")
        card = a[2] if len(a) > 2 else k.get("main_gpu")
        b.stamp("gate", f"{split}|card{card}|{out}")      # observe only: the real gate answered
        return out

    mgr._vram_admits_locked = gate_spy

    orig_route = mgr._route_or_reserve

    async def route_spy(slot):
        hook = b.route_hooks.get(slot.prompt)
        if hook is not None and slot.prompt not in b.slots:
            hook(slot)
        b.slots.setdefault(slot.prompt, slot)
        b.stamp("route", slot.prompt)
        return await orig_route(slot)

    mgr._route_or_reserve = route_spy

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    with _LogTap(holder):
        try:
            yield b
        finally:
            for g in list(b.holds.values()) + list(b.stop_holds.values()):
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


# ------------------------------------------------------------------------------------------------
# scenario steps
# ------------------------------------------------------------------------------------------------
async def start_busy(b, label, tag, card, meta=META_OWNER, thread="th-b"):
    """A model on `card` in the middle of a turn that is held until the test releases it."""
    b.steer_to(card)
    try:
        b.submit(label, tag, meta, thread, hold=True)
        await b.until(lambda: b.seen("turn_start", label), f"{tag}'s held turn to start")
    finally:
        b.steer_clear()
    r = resident_for(b.mgr, tag)
    assert r.main_gpu == card, f"{tag} landed on card {r.main_gpu}, wanted {card}"
    assert r.state is ResidentState.ACTIVE and r.active_slot is not None, (
        f"{tag} is not mid-turn: {r.state}")
    return r


async def serve_and_park(b, label, tag, card, meta=META_OWNER, thread="th-a", *, lapse=True):
    """The model serves one turn on `card`; with lapse=True its grace window ends and it is parked
    idle and evictable (a warm grace start behind it), with lapse=False it is left inside grace."""
    b.steer_to(card)
    try:
        b.submit(label, tag, meta, thread)
        await b.until(lambda: b.seen("caller_got_response", label), f"{tag}'s turn to be delivered")
    finally:
        b.steer_clear()
    r = resident_for(b.mgr, tag)
    assert r.main_gpu == card, f"{tag} landed on card {r.main_gpu}, wanted {card}"
    await b.until(lambda: r.grace is not None and r.grace._started_at is not None,
                  f"{tag}'s grace window to start")
    await b.until(lambda: r.in_grace_loop, f"{tag} to be inside its grace loop")
    if lapse:
        await b.until(lambda: (not r.in_grace_loop) and r.state is ResidentState.IDLE_EVICTABLE,
                      f"{tag}'s grace window to end and the model to be parked", 12.0)
        assert r.state is ResidentState.IDLE_EVICTABLE and not r.in_grace_loop
        assert not b.seen("unload_decided", tag), f"{tag} was unloaded before the claimant came"
    return r


async def busy_card0_idle_card1(b, *, idle_lapse=True, idle_busy=False):
    """The shared scenario: card 0 holds a busy model, card 1 an idle one (or, for the controls, a
    model that only looks idle)."""
    rb = await start_busy(b, "BUSY1", TAG_BUSY, 0)
    if idle_busy:
        ri = await start_busy(b, "I1", TAG_IDLE, 1, thread="th-a")
    else:
        ri = await serve_and_park(b, "I1", TAG_IDLE, 1, lapse=idle_lapse)
    return rb, ri


def assert_shape(b, ri, *, claim_tag=TAG_CLAIM, fits_after=True, card0_freer=True):
    """The scenario is what it says it is: the claimant fits nowhere now, the two cards together
    cannot take it (no layer-split fallback), and card 1 holds it once the idle model is gone."""
    free = b.free_list()
    need_c = b.need(claim_tag)
    assert ri.reserved_need_mib == b.need(ri.model_tag), (
        f"the idle model's reserved figure {ri.reserved_need_mib} differs from its footprint")
    assert all(f < need_c for f in free), f"the claimant ({need_c}) fits a card as it stands: {free}"
    assert all(f - FLOOR_MIB < need_c for f in free), f"the auto-placer would pick a card: {free}"
    assert sum(free) - FLOOR_MIB * len(free) < need_c, (
        f"the auto-placer would fall back to a layer split: free={free} need={need_c}")
    if fits_after:
        assert free[1] + ri.reserved_need_mib >= need_c, (
            f"card 1 would not hold the claimant even without the idle model: {free} need={need_c}")
    else:
        assert free[1] + ri.reserved_need_mib < need_c, (
            f"card 1 would hold the claimant; the scenario wanted no fit: {free} need={need_c}")
    if card0_freer:
        assert free[0] > free[1], f"card 0 is not the freer card: {free}"
    return need_c


def submit_claimant(b, meta=META_CLAIMANT, label="C1"):
    b.stamp("claimant_submit", label)
    return b.submit(label, TAG_CLAIM, meta, "th-claim")


def untouched(b, tag, *, upto=None):
    """Events that mean `tag` was acted on (unload decided, teardown begun, KV save, engine stop)."""
    end = len(b.events) if upto is None else upto
    hits = []
    for kind in ("unload_decided", "unload_teardown_begin", "kv_save_begin", "engine_stop"):
        i = b.idx(kind, tag)
        if i is not None and i < end:
            hits.append(kind)
    return hits


async def await_starved(b, passes=STARVED_PASSES):
    await b.until(lambda: b.count("log", "starved") >= passes,
                  f"the claimant to go through make-room {passes} times and be deferred each time", 12.0)


def assert_no_relocation_attempt(b, claim_label="C1"):
    slot = b.slots.get(claim_label)
    assert slot is not None, "the claimant was never routed"
    assert slot.fastlane_relocated_main_gpu is None, (
        f"the claimant was relocated to card {slot.fastlane_relocated_main_gpu}; observed: {b.observed()}")
    assert slot.fastlane_relocated_split_mode is None
    assert getattr(slot, "fastlane_reclaim_first", False) is False, f"the claimant is marked reclaim-first; observed: {b.observed()}"
    assert not b.seen("log", "reclaim"), f"a reclaim-first decision was logged; observed: {b.observed()}"
    assert not b.seen("load", TAG_CLAIM), f"the claimant was started; observed: {b.observed()}"




def flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def reclaim_lines(b):
    """The lines that say a card was chosen for a claimant (not the refusal lines)."""
    return [m for m in b.logs if m.startswith("reclaim_first ")]


def refusal_lines(b):
    return [m for m in b.logs if m.startswith("reclaim_first_refused ")]


async def wait_served(b, what, label="C1", bound=SERVED_BOUND_S):
    try:
        await b.until(lambda: b.seen("turn_end", label) and b.seen("caller_got_response", label), what, bound)
    except AssertionError as e:
        raise AssertionError(f"the claimant was never served within {bound}s ({what}): {e}") from None


def assert_served_on_card(b, card, *, tag=TAG_CLAIM, split="none"):
    """The claimant runs on `card`, in the manager's ledger and in the process it started."""
    rc = resident_for(b.mgr, tag)
    assert rc.main_gpu == card and rc.split_mode == split, (
        f"claimant resident: card {rc.main_gpu}, split {rc.split_mode}; wanted card {card}, split {split}; "
        f"observed: {b.observed()}")
    argv = b.spawn_argv[tag]
    assert flag_value(argv, "--split-mode") == split, f"claimant argv: {argv}"
    if split == "none":
        assert flag_value(argv, "--main-gpu") == str(card), f"claimant argv: {argv}"


def assert_saved_then_unloaded(b, tag):
    """`tag` was unloaded exactly once, and its KV was saved (and confirmed) before its engine stopped."""
    ev = b.rel()

    def need(kind, name):
        i = b.idx(kind, name)
        assert i is not None, f"event never happened: {(kind, name)}; events={ev}"
        return i

    i_decided = need("unload_decided", tag)
    i_save_begin = need("kv_save_begin", tag)
    i_save_end = need("kv_save_end", f"{tag}|True")
    i_stop = need("engine_stop", tag)
    assert i_decided <= i_save_begin < i_save_end < i_stop, (
        f"{tag}'s KV was not saved (and confirmed) before its engine stop; events={ev}")
    assert b.count("unload_decided", tag) == 1, f"{tag} was unloaded more than once; events={ev}"
    snap = b.kv_at_stop.get(tag)
    assert snap is not None, f"no snapshot of the save directory at {tag}'s engine stop"
    bins = {n: s for n, s in snap["bins"].items() if n.startswith(f"{tag}.")}
    assert bins and all(s == len(KV_BYTES) for s in bins.values()), (
        f"{tag}'s KV blob was not in the store when its engine stopped: {snap['bins']}")
    metas = [m for n, m in snap["metas"].items() if n.startswith(f"{tag}.")]
    assert metas and all(m.get("model_tag") == tag for m in metas), (
        f"{tag}'s KV metadata was not in the store when its engine stopped: {snap['metas']}")


async def assert_busy_intact(b, rb):
    ev = b.rel()
    assert untouched(b, TAG_BUSY) == [], f"the busy model was acted on: {untouched(b, TAG_BUSY)}; events={ev}"
    assert not b.seen("turn_cancelled", "BUSY1") and not b.seen("turn_end", "BUSY1")
    assert rb.state is ResidentState.ACTIVE and rb.active_slot is not None and rb.handle is not None
    b.release("BUSY1")
    res = await asyncio.wait_for(b.tasks["BUSY1"], 10)
    assert res[1] == {"ok": True, "label": "BUSY1", "model": TAG_BUSY, "text": "reply-to-BUSY1"}, (
        f"the busy model's turn did not complete intact: {res[1]!r}")


def assert_span_shape(b, ri, *, claim_tag=TAG_CLAIM, fits_after=True):
    """The span case: the claimant fits no card as things stand, the auto-placer would give it a layer
    split (the two cards together take it), and card 1 holds it on its own once the idle model is gone."""
    free = b.free_list()
    need_c = b.need(claim_tag)
    assert ri.reserved_need_mib == b.need(ri.model_tag), (
        f"the idle model's reserved figure {ri.reserved_need_mib} differs from its footprint")
    assert all(f < need_c for f in free), f"the claimant ({need_c}) fits a card as it stands: {free}"
    pick = b.mgr._auto_pick_gpu(need_c)
    assert pick == (0, "layer"), f"the auto-placer would not give the claimant a layer split: {pick}; free={free} need={need_c}"
    got = free[1] + ri.reserved_need_mib - FLOOR_MIB >= need_c
    assert got is fits_after, (
        f"card 1 without the idle model: fits={got}, wanted {fits_after}; free={free} need={need_c}")
    assert free[0] - CUDA_RESIDUE_MIB >= FLOOR_MIB, f"card 0 cannot take the CUDA residue: {free}"
    return need_c


async def _span3(b, *, owner1=META_OWNER, owner2=META_OWNER, park_order=(1, 2)):
    """Card 0: a busy model. Cards 1 and 2: one idle model each, parked in `park_order` (the first
    parked is the less recently active)."""
    rb = await start_busy(b, "BUSY1", TAG_BUSY, 0)
    plan = {1: (TAG_IDLE, owner1, "th-a"), 2: (TAG_IDLE2, owner2, "th-a2")}
    idle = {}
    for card in park_order:
        tag, meta, thread = plan[card]
        idle[card] = await serve_and_park(b, f"I{card}", tag, card, meta=meta, thread=thread)
    return rb, idle


def assert_span3(b, idle, *, fits, claim_tag=TAG_CLAIM):
    """Three cards: the claimant fits none as things stand, the auto-placer would give it a layer
    split, `fits[card]` says whether that card holds it once its idle model is gone, and every card
    could take the CUDA residue."""
    free = b.free_list()
    need_c = b.need(claim_tag)
    assert all(f < need_c for f in free), f"the claimant ({need_c}) fits a card as it stands: {free}"
    pick = b.mgr._auto_pick_gpu(need_c)
    assert pick == (0, "layer"), f"the auto-placer would not give the claimant a layer split: {pick}; free={free}"
    for card, r in idle.items():
        got = free[card] + r.reserved_need_mib - FLOOR_MIB >= need_c
        assert got is fits[card], (
            f"card {card}: without {r.model_tag} fits={got}, wanted {fits[card]}; free={free} need={need_c}")
    for o in range(len(free)):
        assert free[o] - CUDA_RESIDUE_MIB >= FLOOR_MIB, f"card {o} cannot take the residue: {free}"
    return need_c


def _models3(*, claim_auto=False, claim_mib=SPAN_MIB, idle2_mib=IDLE_MIB):
    m = _models(claim_mib=claim_mib, claim_auto=claim_auto)
    m[TAG_IDLE2] = dict(size_mib=idle2_mib, main_gpu=2)
    return m


def _free_for_residue(delta):
    return FLOOR_MIB + CUDA_RESIDUE_MIB + delta


def _bare_manager(tmp_path, monkeypatch, free, rules=RULES, models=None):
    """A manager with no worker loop and a scripted probe: for the direct, in-process checks."""
    boot, runtime = boot_ranked_runtime(tmp_path, rules=rules, max_parallel_sidecars=4, grace_seconds=1)
    runtime.queue.safety_min_free_vram_mib = FLOOR_MIB
    holder = {"free": list(free)}
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", lambda: list(holder["free"]))
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", lambda: list(holder["free"]))
    for tag, spec in (models or _models()).items():
        _manifest(boot, tag, **spec)
    return TurbohaulManager(boot, runtime), holder


def _resident(tag, card, mib, state=ResidentState.IDLE_EVICTABLE, split="none", meta=META_OWNER, last_active=1.0):
    return Resident(model_tag=tag, resident_key=tag, state=state, last_active_monotonic=last_active,
                    main_gpu=card, split_mode=split, idle_client_meta=dict(meta), reserved_need_mib=mib)


def _claim_slot(tag=TAG_CLAIM, rule_index=1, meta=META_CLAIMANT):
    s = Slot.new(tag)
    s.client_meta = dict(meta)
    if meta is META_CLAIMANT:
        s.fastlane = FastLaneMatch(rule_index=rule_index, raw_address=IP_2, label="", effective_tag="main", rank=1)
    return s


async def _residue_scenario(b, free_by_card):
    rb, ri = await busy_card0_idle_card1(b)
    for card, free in free_by_card.items():
        b.set_free(card, free)
    assert_shape(b, ri, card0_freer=False)
    return rb, ri


AUTO_IDS = ["auto_place_site", "pinned_site"]


# ================================================================================================
# The span case: the claimant fits no card now, the cards together would take it, one card takes
# it once the idle model there is gone. One card, no split.
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_one_card_after_the_idle_model_is_unloaded_beats_a_layer_split(tmp_path, monkeypatch, auto_place):
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB, claim_auto=auto_place)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        need_c = assert_span_shape(b, ri)
        reclaim = ri.reserved_need_mib
        assert not _kv_snapshot(b.kv_dir)["bins"], "a KV blob exists before any unload"

        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1 after the idle model there was unloaded "
                             f"({'auto_place' if auto_place else 'pinned'} manifest)")

        assert_served_on_card(b, 1)
        assert_saved_then_unloaded(b, TAG_IDLE)
        i_stop, i_load = b.idx("engine_stop", TAG_IDLE), b.idx("load", TAG_CLAIM)
        assert i_stop < i_load, f"the claimant loaded before the idle engine stopped; events={b.rel()}"
        await assert_busy_intact(b, rb)

        lines = reclaim_lines(b)
        assert len(lines) == 1, f"expected one reclaim_first line, got {lines}; observed: {b.observed()}; events={b.rel()}"
        for want in ("card=1", f"reclaimable_mib={reclaim}", "pending_mib=0", f"need_mib={need_c}"):
            assert re.search(rf"(^|\s){re.escape(want)}(\s|$)", lines[0]), f"{want!r} missing from: {lines[0]}"
        assert "claimant=" in lines[0], lines[0]


# the claimant is not a fast-lane claimant: the lane is on but its client is in no rule, or the lane
# is off and the manifest asks for auto placement. Every request the auto-placer decides gets it.
NON_LANE_CASES = {
    "unlisted_client_lane_on_pinned_manifest": dict(rules=RULES, meta=META_UNLISTED, auto=False),
    "unlisted_client_lane_on_auto_place_manifest": dict(rules=RULES, meta=META_UNLISTED, auto=True),
    "lane_off_auto_place_manifest": dict(rules=[], meta=META_CLAIMANT, auto=True),
}


@pytest.mark.parametrize("shape", ["span", "no_aggregate"])
@pytest.mark.parametrize("case", list(NON_LANE_CASES))
async def test_a_claimant_that_is_not_a_fast_lane_claimant_also_gets_one_card_after_reclaim(
        tmp_path, monkeypatch, case, shape):
    spec = NON_LANE_CASES[case]
    claim_mib = SPAN_MIB if shape == "span" else CLAIM_MIB
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=claim_mib, claim_auto=spec["auto"]),
                     rules=spec["rules"]) as b:
        rb, ri = await busy_card0_idle_card1(b)
        if shape == "span":
            assert_span_shape(b, ri)
        else:
            assert_shape(b, ri)
        submit_claimant(b, meta=spec["meta"])
        await wait_served(b, f"the {case} claimant to be placed on card 1 after the idle model there was unloaded")
        slot = b.slots["C1"]
        assert b.mgr.queue._fastlane_priority_key(slot) is None, "the claimant holds a live claim; not the case under test"
        assert_served_on_card(b, 1)
        assert_saved_then_unloaded(b, TAG_IDLE)
        await assert_busy_intact(b, rb)
        lines = reclaim_lines(b)
        assert len(lines) == 1 and re.search(r"(^|\s)card=1(\s|$)", lines[0]), (
            f"[{case}/{shape}] expected one reclaim_first line for card 1, got {lines}; "
            f"observed: {b.observed()}; events={b.rel()}")


async def test_the_span_case_leaves_a_pinned_claimant_with_the_lane_off_where_it_is(tmp_path, monkeypatch):
    """A manifest that is not auto-placed, with the lane off, is a pin: nothing relocates it, nothing
    is unloaded for it, and no layer split is made for it."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB, claim_auto=False), rules=[]) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_span_shape(b, ri)
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], f"the idle model was acted on: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)
        assert not refusal_lines(b), refusal_lines(b)
        assert [r.split_mode for r in b.mgr._model_residents()] == ["none", "none"], "a split placement appeared"


# ================================================================================================
# One exact scenario. Two 24000 MiB cards, floor 1000. Card 0: a model mid-turn that uses 9000
# (free 15000, the freer card, it cannot hold the claimant and has no idle model to give up). Card 1:
# an idle model that uses 13000 (free 11000). The claimant needs 22732 including its context: no card
# holds it as things stand, and card 1 holds it once the idle model is gone (24000 - floor >= 22732).
# With card 0 reading 15000 the cards together take it (24000 after both floors): the placer would
# give it a layer split. In the no-aggregate variant another process holds 3000 of card 0 (free 12000,
# 21000 after both floors), so the cards together do not take it either.
# ================================================================================================
LEAD_NEED_MIB = 22732
LEAD_BUSY_MIB, LEAD_IDLE_MIB = 9000, 13000
LEAD_VARIANTS = {
    "auto_place_site": dict(auto=True, shrink=False),
    "pinned_site": dict(auto=False, shrink=False),
    "no_aggregate_auto_place_site": dict(auto=True, shrink=True),
    "no_aggregate_pinned_site": dict(auto=False, shrink=True),
}


def _lead_models(*, claim_auto):
    return {
        TAG_BUSY: dict(size_mib=1000, expected_mib=LEAD_BUSY_MIB, main_gpu=0),
        TAG_IDLE: dict(size_mib=1000, expected_mib=LEAD_IDLE_MIB, main_gpu=1),
        TAG_CLAIM: dict(size_mib=1000, expected_mib=LEAD_NEED_MIB, main_gpu=0, auto_place=claim_auto),
    }


def assert_lead_shape(b, ri, *, shrink):
    need_c = b.need(TAG_CLAIM)
    assert need_c == LEAD_NEED_MIB, f"claimant need {need_c}"
    assert b.need(TAG_BUSY) == LEAD_BUSY_MIB and b.need(TAG_IDLE) == LEAD_IDLE_MIB
    free = b.free_list()
    pick = b.mgr._auto_pick_gpu(need_c)
    print(f"LEAD_SHAPE free={free} auto_pick={pick} need={need_c} idle_reserved={ri.reserved_need_mib} "
          f"card1_after_reclaim={free[1] + ri.reserved_need_mib - FLOOR_MIB} card0_after_residue={free[0] - CUDA_RESIDUE_MIB}")
    assert free == [12000 if shrink else 15000, CARD_MIB - LEAD_IDLE_MIB], free
    assert all(f < need_c for f in free), f"the claimant fits a card as it stands: {free}"
    assert pick == ((None, "none") if shrink else (0, "layer")), f"auto-placer pick {pick}; free={free}"
    assert ri.reserved_need_mib == LEAD_IDLE_MIB
    assert free[1] + ri.reserved_need_mib - FLOOR_MIB >= need_c, "card 1 would not hold it after reclaim"
    assert free[0] - CUDA_RESIDUE_MIB >= FLOOR_MIB, "card 0 cannot take the CUDA residue"


async def _one_card_after_reclaim_scenario(tmp_path, monkeypatch, variant):
    spec = LEAD_VARIANTS[variant]
    async with world(tmp_path, monkeypatch, models=_lead_models(claim_auto=spec["auto"])) as b:
        rb, ri = await busy_card0_idle_card1(b)
        if spec["shrink"]:
            b.set_free(0, 12000)
        assert_lead_shape(b, ri, shrink=spec["shrink"])
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1 after the idle model there was unloaded")

        # behaviour first
        assert_served_on_card(b, 1)
        assert_saved_then_unloaded(b, TAG_IDLE)
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] == [TAG_IDLE], b.observed()
        assert b.idx("engine_stop", TAG_IDLE) < b.idx("load", TAG_CLAIM), f"events={b.rel()}"
        await assert_busy_intact(b, rb)

        # then the decision line
        lines = reclaim_lines(b)
        assert len(lines) == 1, (
            f"expected one reclaim_first line, got {lines}; claimant argv={b.spawn_argv.get(TAG_CLAIM)}; "
            f"observed: {b.observed()}; events={b.rel()}")
        for want in ("card=1", f"reclaimable_mib={LEAD_IDLE_MIB}", "pending_mib=0", f"need_mib={LEAD_NEED_MIB}"):
            assert re.search(rf"(^|\s){re.escape(want)}(\s|$)", lines[0]), f"{want!r} missing from: {lines[0]}"


@pytest.mark.parametrize("variant", ["no_aggregate_auto_place_site", "no_aggregate_pinned_site"])
async def test_no_single_card_fits_and_the_cards_together_cannot_either_one_card_after_reclaim(
        tmp_path, monkeypatch, variant):
    """The claimant fits no card as things stand and the cards together cannot take it either, but
    card 1 holds it once the idle model there is gone. Before this change the claimant waits: the
    placer answers no card, the manifest pin is kept, the gate refuses, and the idle model on the
    other card is never unloaded. With it, the claimant is served on card 1 after the idle model
    there is unloaded. This is the behavioural test for the order."""
    await _one_card_after_reclaim_scenario(tmp_path, monkeypatch, variant)


@pytest.mark.parametrize("variant", ["auto_place_site", "pinned_site"])
async def test_lead_numbers_the_base_already_reaches_one_card_decision_line_pin_and_over_eviction_control(
        tmp_path, monkeypatch, variant):
    """Here the cards together do take the claimant (the placer would answer a layer split). Before
    this change the claimant already ends on card 1, through the global make-room step, so only the
    decision-line assertion differs. The test also asserts that exactly the idle model is unloaded
    and the busy one is untouched (an over-eviction control)."""
    await _one_card_after_reclaim_scenario(tmp_path, monkeypatch, variant)


@pytest.mark.parametrize("variant", list(LEAD_VARIANTS))
@pytest.mark.parametrize("case", ["i_mid_turn", "i_in_grace"])
async def test_lead_scenario_a_resident_that_cannot_be_reclaimed_leaves_the_claimant_waiting(
        tmp_path, monkeypatch, case, variant):
    spec = LEAD_VARIANTS[variant]
    in_grace = case == "i_in_grace"
    async with world(tmp_path, monkeypatch, models=_lead_models(claim_auto=spec["auto"]),
                     grace_s=30 if in_grace else 1) as b:
        rb, ri = await busy_card0_idle_card1(b, idle_lapse=False, idle_busy=not in_grace)
        if spec["shrink"]:
            b.set_free(0, 12000)
        if in_grace:
            assert ri.in_grace_loop and ri.state is not ResidentState.IDLE_EVICTABLE, f"not in grace: {ri.state}"
        else:
            assert ri.state is ResidentState.ACTIVE and ri.active_slot is not None
        assert_lead_shape(b, ri, shrink=spec["shrink"])
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], f"[{case}] the resident was acted on: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)
        assert not refusal_lines(b), refusal_lines(b)


# ================================================================================================
# The population: one predicate says which requests get the decision
# ================================================================================================
POPULATION = [
    # (case, rules, claimant is listed, manifest auto_place, manifest split, expected answer)
    ("listed_client_pinned_manifest", RULES, True, False, "none", True),
    ("unlisted_client_lane_on", RULES, False, False, "none", True),
    ("auto_place_manifest_lane_off", [], False, True, "none", True),
    ("pinned_manifest_lane_off", [], False, False, "none", False),
    ("layer_manifest_lane_on", RULES, True, False, "layer", False),
    ("layer_manifest_lane_off_auto_place", [], False, True, "layer", False),
]


@pytest.mark.parametrize("case,rules,listed,auto_place,split,expected", POPULATION, ids=[p[0] for p in POPULATION])
async def test_the_population_predicate_answers_for_each_kind_of_request(
        tmp_path, monkeypatch, case, rules, listed, auto_place, split, expected):
    """The predicate is fed what the placement resolver really returned for the request, in the span
    case (the cards together fit it, no single card does). It takes (slot, auto_place, auto_picked,
    split_mode) by keyword."""
    models = _models(claim_mib=SPAN_MIB, claim_auto=auto_place, claim_split=split)
    mgr, _h = _bare_manager(tmp_path, monkeypatch, [CARD_MIB - BUSY_MIB, CARD_MIB - IDLE_MIB], rules=rules, models=models)
    mgr._residents[TAG_BUSY] = _resident(TAG_BUSY, 0, BUSY_MIB, state=ResidentState.ACTIVE)
    mgr._residents[TAG_IDLE] = _resident(TAG_IDLE, 1, IDLE_MIB)
    slot = _claim_slot(meta=META_CLAIMANT if listed else META_UNLISTED)
    async with mgr._registry_lock:
        need, parallel, main_gpu, split_mode, _sleep, got_auto_place, auto_picked = mgr._resolve_placement_locked(TAG_CLAIM)
        answer = mgr._reclaim_first_applies(
            slot=slot, auto_place=got_auto_place, auto_picked=auto_picked, split_mode=split_mode)
    assert got_auto_place is auto_place
    assert answer is expected, (
        f"[{case}] the predicate answered {answer}, wanted {expected}; resolver gave "
        f"auto_picked={auto_picked} split={split_mode} card={main_gpu}")


# ================================================================================================
# No single card fits even with reclaim: the split path runs as it does today
# ================================================================================================
LAYER_ONLY_MIB = 24000    # larger than one card can ever hold, smaller than the cards together


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_when_no_card_fits_even_after_reclaim_the_claimant_takes_the_split_path(
        tmp_path, monkeypatch, auto_place):
    models = _models(claim_mib=LAYER_ONLY_MIB, claim_auto=auto_place)
    models[TAG_BUSY] = dict(size_mib=4000, main_gpu=0)
    async with world(tmp_path, monkeypatch, models=models) as b:
        rb, ri = await busy_card0_idle_card1(b)
        need_c = b.need(TAG_CLAIM)
        free = b.free_list()
        assert b.mgr._auto_pick_gpu(need_c) == (0, "layer"), f"the placer would not split it: {free} need={need_c}"
        assert free[1] + ri.reserved_need_mib - FLOOR_MIB < need_c, "card 1 would hold it after reclaim"
        assert free[0] - FLOOR_MIB < need_c

        submit_claimant(b)
        await await_starved(b)
        # while the busy model holds card 0 the split cannot be placed: the claimant is not put on a card
        assert not b.seen("load", TAG_CLAIM), f"the claimant was started on one card: {b.observed()}"
        assert not reclaim_lines(b), f"a card was chosen for a model that fits none: {reclaim_lines(b)}"
        slot = b.slots["C1"]
        assert slot.fastlane_relocated_main_gpu is None and slot.fastlane_relocated_split_mode is None
        assert getattr(slot, "fastlane_reclaim_first", False) is False
        assert untouched(b, TAG_BUSY) == []

        # the busy turn ends, the cards empty out, and the claimant is placed as a layer split
        b.release("BUSY1")
        await wait_served(b, "the claimant to be placed as a layer split once the cards are free", bound=20.0)
        assert_served_on_card(b, 0, split="layer")
        assert not b.seen("log", "reclaim"), f"a card was chosen: {reclaim_lines(b)}"


# ================================================================================================
# Controls: something that looks idle is not counted, so the claimant is not relocated onto its card
# ================================================================================================
PROTECTIONS = ["in_grace_window", "mid_turn_real", "active_slot_while_parked",
               "riders_while_parked", "staleness_grant", "state_active_without_a_turn",
               "state_still_loading"]


@pytest.mark.parametrize("protection", PROTECTIONS)
async def test_a_resident_that_only_looks_idle_is_not_counted_and_the_claimant_defers(
        tmp_path, monkeypatch, protection):
    grace_s = 30 if protection == "in_grace_window" else 1
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB), grace_s=grace_s) as b:
        rb, ri = await busy_card0_idle_card1(
            b, idle_lapse=(protection not in ("in_grace_window", "mid_turn_real")),
            idle_busy=(protection == "mid_turn_real"))
        mgr = b.mgr
        fake_slot = None
        try:
            if protection == "in_grace_window":
                assert ri.in_grace_loop and ri.state is not ResidentState.IDLE_EVICTABLE, (
                    f"the model is not inside its grace window: {ri.state}")
            elif protection == "mid_turn_real":
                assert ri.state is ResidentState.ACTIVE and ri.active_slot is not None
            elif protection == "active_slot_while_parked":
                assert ri.state is ResidentState.IDLE_EVICTABLE
                fake_slot = Slot.new(TAG_IDLE)
                ri.active_slot = fake_slot
                assert mgr._resident_has_running_turn(ri)
            elif protection in ("state_active_without_a_turn", "state_still_loading"):
                # only the STATE differs from a parked model: no running turn, no grant. The state
                # of a model that is serving, or still booting, must keep it out on its own.
                assert ri.state is ResidentState.IDLE_EVICTABLE and not mgr._resident_has_running_turn(ri)
                assert not mgr._fastlane_staleness_protects(ri)
                ri.state = (ResidentState.ACTIVE if protection == "state_active_without_a_turn"
                            else ResidentState.RESERVED_LOADING)
            elif protection == "riders_while_parked":
                assert ri.state is ResidentState.IDLE_EVICTABLE
                fake_slot = Slot.new(TAG_IDLE)
                ri.inflight = [fake_slot]
                assert mgr._resident_has_running_turn(ri)
            else:
                assert ri.state is ResidentState.IDLE_EVICTABLE and not mgr._resident_has_running_turn(ri)
                assert not mgr._fastlane_staleness_protects(ri), "protected before the grant was written"
                mgr._fastlane_staleness_grants[_fastlane_census_key(IP_1)] = (
                    time.monotonic() - (_STALENESS_GRANT_THRESHOLD_S + 1.0))
                assert mgr._fastlane_staleness_protects(ri) is True, "the staleness grant is not in force"
            assert_span_shape(b, ri)

            submit_claimant(b)
            await await_starved(b)
            assert untouched(b, TAG_IDLE) == [], (
                f"[{protection}] the resident was acted on: {untouched(b, TAG_IDLE)}; observed: {b.observed()}")
            assert untouched(b, TAG_BUSY) == []
            assert_no_relocation_attempt(b)
            assert not refusal_lines(b), (
                f"[{protection}] a resident that is not reclaimable was counted as a candidate: {b.observed()}")
            assert mgr._residents.get(TAG_IDLE) is ri and ri.handle is not None
        finally:
            if fake_slot is not None:
                ri.active_slot = None
                ri.inflight = []
            if protection in ("state_active_without_a_turn", "state_still_loading"):
                ri.state = ResidentState.IDLE_EVICTABLE


async def test_an_idle_layer_split_resident_is_not_counted_per_card(tmp_path, monkeypatch):
    """A resident that spans the cards has no recorded share of any one card, so it is not counted as
    reclaimable on card 1 and the claimant is not relocated onto that card. (The ordinary make-room
    step may still unload it, as it does today; that is not what is under test.)"""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB), max_parallel=4) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_span_shape(b, ri)
        ri.split_mode = "layer"
        try:
            submit_claimant(b)
            await b.until(lambda: b.seen("unload_decided", TAG_IDLE) or b.count("log", "starved") >= 2,
                          "the claimant to be dealt with by the ordinary make-room step", 12.0)
            slot = b.slots["C1"]
            assert slot.fastlane_relocated_main_gpu is None, (
                f"the claimant was relocated to card {slot.fastlane_relocated_main_gpu} on the strength of a "
                f"layer-split resident: {b.observed()}")
            assert slot.fastlane_relocated_split_mode is None
            assert getattr(slot, "fastlane_reclaim_first", False) is False
            assert not b.seen("log", "reclaim"), f"a card was chosen: {reclaim_lines(b)}"
            assert untouched(b, TAG_BUSY) == []
        finally:
            ri.split_mode = "none"


# ================================================================================================
# The old shape: the cards together cannot take the claimant either, and the idle model on the other
# card makes room
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_listed_claimant_is_placed_on_the_other_card_when_the_idle_resident_there_is_unloaded(
        tmp_path, monkeypatch, auto_place):
    async with world(tmp_path, monkeypatch, models=_models(claim_auto=auto_place)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        need_c = assert_shape(b, ri)
        reclaim = ri.reserved_need_mib
        assert not _kv_snapshot(b.kv_dir)["bins"], "a KV blob exists before any unload"

        submit_claimant(b)
        await wait_served(b, f"the claimant to be placed on card 1 and served ({'auto_place' if auto_place else 'pinned'} "
                             f"manifest, idle model on card 1 unloadable, busy model on card 0)")
        i_log = b.idx("log", "reclaim")
        assert i_log is not None, f"no reclaim_first line; events={b.rel()}"
        assert i_log < b.idx("unload_decided", TAG_IDLE), f"the decision line came after the unload; events={b.rel()}"
        assert_saved_then_unloaded(b, TAG_IDLE)
        assert b.idx("engine_stop", TAG_IDLE) < b.idx("load", TAG_CLAIM) < b.idx("turn_end", "C1"), (
            f"the claimant loaded before the idle engine stopped, or ran before loading; events={b.rel()}")
        assert_served_on_card(b, 1)
        await assert_busy_intact(b, rb)

        lines = reclaim_lines(b)
        assert len(lines) == 1, f"expected one reclaim_first line, got {lines}"
        for want in ("card=1", f"reclaimable_mib={reclaim}", "pending_mib=0", f"need_mib={need_c}"):
            assert re.search(rf"(^|\s){re.escape(want)}(\s|$)", lines[0]), f"{want!r} missing from: {lines[0]}"


# ------------------------------------------------------------------------------------------------
# no fit even after the unload
# ------------------------------------------------------------------------------------------------
async def test_no_fit_even_after_reclaim_defers_and_nothing_is_unloaded(tmp_path, monkeypatch):
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=TOO_BIG_MIB)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri, fits_after=False)
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], f"the idle model was unloaded for a claimant that cannot fit: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)
        assert [r.split_mode for r in b.mgr._model_residents()] == ["none", "none"], "a split placement appeared"


async def test_no_fit_refusal_is_logged_with_its_card_and_reason(tmp_path, monkeypatch):
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=TOO_BIG_MIB)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri, fits_after=False)
        submit_claimant(b)
        await b.until(lambda: b.seen("log", "refused"), "a reclaim_first_refused line for the claimant that cannot fit", 8.0)
        lines = [m for m in refusal_lines(b) if re.search(r"(^|\s)card=1(\s|$)", m)]
        assert lines, f"no refusal names card 1: {refusal_lines(b)}"
        assert any(re.search(r"(^|\s)reason=no_fit(\s|$)", m) for m in lines), lines
        assert "claimant=" in lines[0], lines[0]
        assert untouched(b, TAG_IDLE) == []


# ------------------------------------------------------------------------------------------------
# the claimant's own card: when it fits after reclaim there too, the lower card wins the tie
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_own_card_fit_after_reclaim_is_used_and_the_other_cards_idle_model_is_left_alone(
        tmp_path, monkeypatch, auto_place):
    async with world(tmp_path, monkeypatch, models=_models(claim_auto=auto_place, with_second_idle=True),
                     max_parallel=4) as b:
        # the idle model on card 1 is the OLDER one, so a victim search that ignored cards would pick it
        old = await serve_and_park(b, "I1", TAG_IDLE, 1)
        own = await serve_and_park(b, "I0", TAG_IDLE2, 0, thread="th-a2")
        assert old.last_active_monotonic < own.last_active_monotonic, "the card 1 model is not the older one"
        free = b.free_list()
        need_c = b.need(TAG_CLAIM)
        assert all(f < need_c for f in free) and sum(free) - FLOOR_MIB * 2 < need_c, f"shape: {free} need={need_c}"
        assert free[0] + own.reserved_need_mib >= need_c and free[1] + old.reserved_need_mib >= need_c

        submit_claimant(b)
        await wait_served(b, "the claimant to be served after the idle model on its own card was unloaded")
        assert b.seen("unload_decided", TAG_IDLE2), f"the own-card idle model was not the victim: {b.observed()}"
        assert untouched(b, TAG_IDLE) == [], f"the other card's idle model was acted on: {untouched(b, TAG_IDLE)}"
        assert_served_on_card(b, 0)
        assert not any(re.search(r"(^|\s)card=1(\s|$)", m) for m in reclaim_lines(b)), reclaim_lines(b)
        assert resident_for(b.mgr, TAG_IDLE) is old and old.state is ResidentState.IDLE_EVICTABLE


# ------------------------------------------------------------------------------------------------
# a pin with the lane off stays put; a layer-split manifest is never relocated
# ------------------------------------------------------------------------------------------------
async def test_lane_off_pinned_model_is_never_relocated(tmp_path, monkeypatch):
    async with world(tmp_path, monkeypatch, models=_models(), rules=[]) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri)
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], f"the other card's idle model was acted on: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)
        assert not refusal_lines(b)


async def test_unlisted_auto_place_claimant_is_served_after_the_idle_model_is_unloaded(tmp_path, monkeypatch):
    """An auto_place claimant with no live claim: the idle model on card 1 is unloaded for it and it
    is served on card 1; the busy model is not touched."""
    async with world(tmp_path, monkeypatch, models=_models(claim_auto=True)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri)
        submit_claimant(b, meta=META_UNLISTED)
        await wait_served(b, "the unlisted auto_place claimant to be served after the eviction")
        assert b.mgr.queue._fastlane_priority_key(b.slots["C1"]) is None
        assert b.seen("unload_decided", TAG_IDLE) and untouched(b, TAG_BUSY) == []
        assert resident_for(b.mgr, TAG_CLAIM).main_gpu == 1
        assert not refusal_lines(b)


async def test_layer_split_claimant_is_never_relocated(tmp_path, monkeypatch):
    """A claimant whose own split spans cards is looked up GLOBALLY by the existing own-card search
    (for a spanning model, freeing any card helps), so the idle model on card 1 is unloaded by the
    existing rule, as today. What must not appear is a relocation or a card choice: the claimant
    is never pinned to a card, and the busy model is never touched."""
    async with world(tmp_path, monkeypatch, models=_models(claim_split="layer")) as b:
        rb, ri = await busy_card0_idle_card1(b)
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)
        assert not refusal_lines(b)
        assert b.count("unload_decided", TAG_IDLE) <= 1, f"the idle model was unloaded twice: {b.observed()}"
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] in ([TAG_IDLE], []), b.observed()


# ------------------------------------------------------------------------------------------------
# the CUDA residue on every card the process does not use
# ------------------------------------------------------------------------------------------------
async def test_card_is_not_chosen_when_another_card_cannot_take_the_cuda_residue(tmp_path, monkeypatch):
    tight = _free_for_residue(-3)
    async with world(tmp_path, monkeypatch, models=_models()) as b:
        rb, ri = await _residue_scenario(b, {0: tight})
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], f"the idle model was unloaded although card 0 cannot take the residue: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_no_relocation_attempt(b)


@pytest.mark.parametrize("delta", [3, 0], ids=["3_mib_above_the_boundary", "exactly_on_the_boundary"])
async def test_card_is_chosen_when_the_other_card_can_take_the_residue(tmp_path, monkeypatch, delta):
    async with world(tmp_path, monkeypatch, models=_models()) as b:
        rb, ri = await _residue_scenario(b, {0: _free_for_residue(delta)})
        submit_claimant(b)
        await wait_served(b, f"the claimant to be placed on card 1 (card 0 reads {b.free_list()[0]} free: "
                             f"floor {FLOOR_MIB} + residue {CUDA_RESIDUE_MIB} {delta:+d})")
        assert_served_on_card(b, 1)
        assert untouched(b, TAG_BUSY) == []


@pytest.mark.parametrize("tight_card", [0, 2], ids=["card_0_tight", "card_2_tight"])
async def test_card_is_not_chosen_when_a_third_card_cannot_take_the_residue(tmp_path, monkeypatch, tight_card):
    ok = _free_for_residue(50)
    free = {0: ok, 2: ok}
    free[tight_card] = _free_for_residue(-3)
    async with world(tmp_path, monkeypatch, models=_models(), n_cards=3) as b:
        rb, ri = await _residue_scenario(b, free)
        submit_claimant(b)
        await await_starved(b)
        assert untouched(b, TAG_IDLE) == [], (
            f"the idle model was unloaded although card {tight_card} cannot take the residue: {b.observed()}")
        assert_no_relocation_attempt(b)


async def test_card_is_chosen_when_every_other_card_can_take_the_residue_with_three_cards(tmp_path, monkeypatch):
    ok = _free_for_residue(50)
    async with world(tmp_path, monkeypatch, models=_models(), n_cards=3) as b:
        rb, ri = await _residue_scenario(b, {0: ok, 2: ok})
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1 with three cards")
        assert_served_on_card(b, 1)
        assert untouched(b, TAG_BUSY) == []


@pytest.mark.parametrize("n_cards,tight_card", [(2, 0), (3, 0), (3, 2)],
                         ids=["two_cards_card_0", "three_cards_card_0", "three_cards_card_2"])
async def test_residue_refusal_log_names_the_card_and_the_numbers(tmp_path, monkeypatch, n_cards, tight_card):
    tight = _free_for_residue(-3)
    free = {0: _free_for_residue(50)}
    if n_cards == 3:
        free[2] = _free_for_residue(50)
    free[tight_card] = tight
    async with world(tmp_path, monkeypatch, models=_models(), n_cards=n_cards) as b:
        rb, ri = await _residue_scenario(b, free)
        submit_claimant(b)
        await b.until(lambda: b.seen("log", "refused"), "a reclaim_first_refused line for the residue", 8.0)
        line = next(m for m in refusal_lines(b) if re.search(r"(^|\s)reason=residue(\s|$)", m))
        for want in ("card=1", "reason=residue"):
            assert re.search(rf"(^|\s){re.escape(want)}(\s|$)", line), f"{want!r} missing from: {line}"
        assert "claimant=" in line, line
        detail = line.split("detail=", 1)[1] if "detail=" in line else ""
        assert re.search(rf"card\W*{tight_card}\b", detail), f"the card {tight_card} is not named in: {detail!r}"
        for num in (tight, CUDA_RESIDUE_MIB, FLOOR_MIB):
            assert re.search(rf"(?<!\d){num}(?!\d)", detail), f"{num} missing from the detail: {detail!r}"


# ------------------------------------------------------------------------------------------------
# the starvation attribution covers the cards the decision covered
# ------------------------------------------------------------------------------------------------
def _starved_attribution(b):
    line = next((m for m in b.logs if m.startswith("MAKE_ROOM_STARVED reason=make_room_starved_vram")), None)
    assert line is not None, f"no make-room starved line; observed: {b.observed()}"
    per = re.search(r"per_resident=(\S+)", line)
    elig = re.search(r"\beligible=(\d+)", line)
    assert per and elig, f"attribution fields missing from: {line}"
    entries = dict(e.split(":", 1) for e in per.group(1).split(","))
    return entries, int(elig.group(1)), line


@pytest.mark.parametrize("case", ["listed_pinned", "listed_auto_place", "unlisted_pinned_lane_on",
                                  "lane_off_pinned"])
async def test_starvation_attribution_uses_the_decisions_card_scope(tmp_path, monkeypatch, case):
    auto = case == "listed_auto_place"
    meta = META_UNLISTED if case == "unlisted_pinned_lane_on" else META_CLAIMANT
    rules = [] if case == "lane_off_pinned" else RULES
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=TOO_BIG_MIB, claim_auto=auto), rules=rules) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri, fits_after=False)
        submit_claimant(b, meta=meta)
        await b.until(lambda: any(m.startswith("MAKE_ROOM_STARVED reason=make_room_starved_vram") for m in b.logs),
                      "the claimant to starve", 8.0)
        entries, eligible, line = _starved_attribution(b)
        assert set(entries) == {TAG_BUSY, TAG_IDLE}, f"both residents must be listed: {line}"
        if case == "lane_off_pinned":
            # the decision does not apply to a pin with the lane off: the attribution stays on its own card
            assert entries[TAG_IDLE] == "card_split" and eligible == 0, (
                f"the pinned claimant's attribution must stay on its own card: {line}")
        else:
            # the decision covers every card, so the idle model on the other card is reported as
            # eligible, not as excluded by the card filter
            assert entries[TAG_IDLE] == "eligible" and eligible == 1, (
                f"[{case}] the attribution covers only the claimant's own card although the decision "
                f"covers every card: {line}")
        assert "state_not_idle" in entries[TAG_BUSY] and "active_slot" in entries[TAG_BUSY], line


# ------------------------------------------------------------------------------------------------
# the retired starvation value never appears
# ------------------------------------------------------------------------------------------------
def _starved_whys(b):
    out = []
    for m in b.logs:
        if m.startswith("MAKE_ROOM_STARVED reason=make_room_starved_vram"):
            g = re.search(r"\bwhy=(.*?) model_tag=", m)
            assert g, f"no why= field in: {m}"
            out.append(g.group(1))
    return out


async def _assert_why_is_reason_unknown(b, ri):
    await await_starved(b)
    whys = _starved_whys(b)
    assert whys, "no make-room starved line for the vram path"
    assert all(w == "reason_unknown" for w in whys), (
        f"the starved line carried a why other than reason_unknown: {sorted(set(whys))}")
    assert not any("all candidates protected as listed waiters" in m for m in b.logs if m.startswith("MAKE_ROOM_STARVED"))
    assert ri.state is ResidentState.IDLE_EVICTABLE and untouched(b, ri.model_tag) == [], (
        f"the idle model on the other card was acted on: {b.observed()}")


@pytest.mark.parametrize("refusal", ["no_fit", "residue"])
@pytest.mark.parametrize("auto_place", [False, True], ids=["pinned_site", "auto_place_site"])
async def test_a_refused_card_choice_leaves_why_as_reason_unknown(tmp_path, monkeypatch, refusal, auto_place):
    """The listed claimant starves because the card check refused, while an idle model sits on the other
    card. The retired value 'all candidates protected as listed waiters' must never be what the line says."""
    models = _models(claim_mib=TOO_BIG_MIB if refusal == "no_fit" else CLAIM_MIB, claim_auto=auto_place)
    async with world(tmp_path, monkeypatch, models=models) as b:
        rb, ri = await busy_card0_idle_card1(b)
        if refusal == "residue":
            b.set_free(0, _free_for_residue(-3))
            assert_shape(b, ri, card0_freer=False)
        else:
            assert_shape(b, ri, fits_after=False)
        submit_claimant(b)
        await _assert_why_is_reason_unknown(b, ri)


# ------------------------------------------------------------------------------------------------
# the victim order inside the chosen card, and the fit before the order
# ------------------------------------------------------------------------------------------------
def _models_two_small_idle_on_card_1(*, claim_auto=False, claim_mib=ORDER_CLAIM_MIB):
    m = _models(claim_mib=claim_mib, claim_auto=claim_auto)
    m[TAG_IDLE] = dict(size_mib=SMALL_IDLE_MIB, main_gpu=1)
    m[TAG_IDLE2] = dict(size_mib=SMALL_IDLE_MIB, main_gpu=1)
    return m


async def _two_on_card_1(b, *, owner_a, owner_b, park_order=("a", "b")):
    """Card 0: a busy model. Card 1: two small idle models (a = TAG_IDLE, b = TAG_IDLE2)."""
    rb = await start_busy(b, "BUSY1", TAG_BUSY, 0)
    plan = {"a": (TAG_IDLE, owner_a, "th-a"), "b": (TAG_IDLE2, owner_b, "th-b2")}
    idle = {}
    for key in park_order:
        tag, meta, thread = plan[key]
        idle[key] = await serve_and_park(b, f"I{key}", tag, 1, meta=meta, thread=thread)
    return rb, idle


def assert_order_shape(b, idle, *, claim_tag=TAG_CLAIM):
    """Neither idle model alone is in the way of the claimant's fit: unloading ONE of them is enough."""
    free = b.free_list()
    need_c = b.need(claim_tag)
    assert all(f < need_c for f in free), f"the claimant ({need_c}) fits a card as it stands: {free}"
    assert b.mgr._auto_pick_gpu(need_c) == (0, "layer"), f"free={free} need={need_c}"
    for key, r in idle.items():
        assert free[1] + r.reserved_need_mib - FLOOR_MIB >= need_c, f"unloading {r.model_tag} alone would not be enough: {free}"
    assert free[0] - CUDA_RESIDUE_MIB >= FLOOR_MIB
    return need_c


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("worse", ["a", "b"], ids=["worse_ranked_is_the_first_parked", "worse_ranked_is_the_second_parked"])
async def test_inside_the_card_the_worse_ranked_owners_idle_model_is_unloaded_first(
        tmp_path, monkeypatch, worse, auto_place):
    better_meta, worse_meta = META_OWNER, META_SECOND
    owners = {"a": better_meta, "b": better_meta}
    owners[worse] = worse_meta
    async with world(tmp_path, monkeypatch, models=_models_two_small_idle_on_card_1(claim_auto=auto_place),
                     rules=RULES3, max_parallel=4) as b:
        rb, idle = await _two_on_card_1(b, owner_a=owners["a"], owner_b=owners["b"])
        assert_order_shape(b, idle)
        submit_claimant(b, meta=META_THIRD)
        await wait_served(b, "the claimant to be placed on card 1")
        victim, other = idle[worse], idle["b" if worse == "a" else "a"]
        assert b.seen("unload_decided", victim.model_tag), f"the worse-ranked owner's model was not unloaded: {b.observed()}"
        assert untouched(b, other.model_tag) == [], f"the better-ranked owner's model was acted on: {b.observed()}"
        assert untouched(b, TAG_BUSY) == []
        assert_served_on_card(b, 1)


@pytest.mark.parametrize("older", ["a", "b"], ids=["older_is_a", "older_is_b"])
async def test_inside_the_card_between_equal_ranks_the_less_recently_active_model_is_unloaded_first(
        tmp_path, monkeypatch, older):
    async with world(tmp_path, monkeypatch, models=_models_two_small_idle_on_card_1(), rules=RULES3,
                     max_parallel=4) as b:
        order = (older, "b" if older == "a" else "a")
        rb, idle = await _two_on_card_1(b, owner_a=META_OWNER, owner_b=META_OWNER, park_order=order)
        old, new = idle[order[0]], idle[order[1]]
        assert old.last_active_monotonic < new.last_active_monotonic
        assert_order_shape(b, idle)
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1")
        assert b.seen("unload_decided", old.model_tag), f"the less recently active model was not unloaded: {b.observed()}"
        assert untouched(b, new.model_tag) == [], f"the more recently active model was acted on: {b.observed()}"
        assert_served_on_card(b, 1)


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("small_card", [1, 2], ids=["card_1_cannot_fit", "card_2_cannot_fit"])
async def test_the_fit_comes_before_the_victim_order(tmp_path, monkeypatch, small_card, auto_place):
    """The worse-ranked owner's model is first in the victim order, but on its card the claimant would
    not fit even with that model gone (another process holds part of the card). The better-ranked
    owner's model is on a card where it does fit. The claimant goes to the card where it fits and the
    first model in the order is left alone."""
    worse_card, good_card = small_card, 3 - small_card
    owners = {worse_card: META_SECOND, good_card: META_OWNER}
    async with world(tmp_path, monkeypatch, models=_models3(claim_auto=auto_place), rules=RULES3, n_cards=3,
                     max_parallel=4) as b:
        rb, idle = await _span3(b, owner1=owners[1], owner2=owners[2])
        b.foreign[worse_card] = 3500
        assert_span3(b, idle, fits={worse_card: False, good_card: True})
        submit_claimant(b, meta=META_THIRD)
        await wait_served(b, f"the claimant to be placed on card {good_card}")
        assert b.seen("unload_decided", idle[good_card].model_tag), b.observed()
        assert untouched(b, idle[worse_card].model_tag) == [], (
            f"{idle[worse_card].model_tag} (first in the order, card {worse_card}) was acted on: {b.observed()}")
        assert untouched(b, TAG_BUSY) == []
        assert_served_on_card(b, good_card)


# ------------------------------------------------------------------------------------------------
# the choice between cards that all fit: fewest MiB still to unload, then the most headroom, then the
# lowest card
# ------------------------------------------------------------------------------------------------
async def test_between_cards_that_fit_equally_the_lowest_card_is_chosen(tmp_path, monkeypatch):
    async with world(tmp_path, monkeypatch, models=_models3(), rules=RULES3, n_cards=3, max_parallel=4) as b:
        # card 2's model is parked first, so it is the less recently active one: a choice made by the
        # victim order alone would unload it and the claimant would end up on card 2
        rb, idle = await _span3(b, park_order=(2, 1))
        assert idle[2].last_active_monotonic < idle[1].last_active_monotonic
        assert_span3(b, idle, fits={1: True, 2: True})
        assert b.free_list()[1] == b.free_list()[2]
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1")
        assert b.seen("unload_decided", idle[1].model_tag), b.observed()
        assert untouched(b, idle[2].model_tag) == [], f"card 2's model was acted on: {b.observed()}"
        assert_served_on_card(b, 1)


async def test_between_cards_with_the_same_free_memory_the_one_with_more_headroom_is_chosen(tmp_path, monkeypatch):
    """Both cards read the same free memory (the same MiB still to unload). Card 2's idle model is
    larger, so after the unload card 2 keeps more headroom: it wins although card 1 is the lower card."""
    big = 14000
    async with world(tmp_path, monkeypatch, models=_models3(idle2_mib=big), rules=RULES3, n_cards=3,
                     max_parallel=4) as b:
        rb, idle = await _span3(b, park_order=(1, 2))
        b.set_free(1, b.free_list()[2])
        assert b.free_list()[1] == b.free_list()[2]
        assert idle[2].reserved_need_mib > idle[1].reserved_need_mib
        assert_span3(b, idle, fits={1: True, 2: True})
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 2")
        assert b.seen("unload_decided", idle[2].model_tag), b.observed()
        assert untouched(b, idle[1].model_tag) == [], f"card 1's model was acted on: {b.observed()}"
        assert_served_on_card(b, 2)


async def test_a_card_that_fits_through_an_unload_already_in_flight_is_chosen_first(tmp_path, monkeypatch):
    """Card 2 has memory on its way back from an unload that is already running, enough for the
    claimant without unloading anything more. Card 1 would need a new unload. Card 2 is chosen."""
    async with world(tmp_path, monkeypatch, models=_models3(), rules=RULES3, n_cards=3, max_parallel=4) as b:
        mgr = b.mgr
        rb, idle = await _span3(b)
        assert_span3(b, idle, fits={1: True, 2: True})
        owner = asyncio.create_task(asyncio.Event().wait())
        credit = idle[2].reserved_need_mib
        mgr._pending_reclaim_mib[2] = credit
        mgr._register_card_release_task(2, owner)
        try:
            submit_claimant(b)
            await b.until(lambda: b.seen("log", "reclaim"), "a card to be chosen for the claimant", 8.0)
            slot = b.slots["C1"]
            assert slot.fastlane_relocated_main_gpu == 2 and getattr(slot, "fastlane_reclaim_first", False) is True, (
                f"override card={slot.fastlane_relocated_main_gpu}; observed: {b.observed()}")
            line = reclaim_lines(b)[0]
            assert re.search(r"(^|\s)card=2(\s|$)", line) and re.search(rf"(^|\s)pending_mib={credit}(\s|$)", line), line
            before = b.count("route", "C1")
            await b.until(lambda: b.count("route", "C1") >= before + 2,
                          "the claimant to be routed twice more with the credit in force", 12.0)
            assert slot.fastlane_relocated_main_gpu == 2, f"the decision moved off card 2: {b.observed()}"
            assert untouched(b, idle[1].model_tag) == [], f"card 1's model was acted on: {b.observed()}"
            assert untouched(b, idle[2].model_tag) == [], f"card 2's model was unloaded twice over: {b.observed()}"
        finally:
            owner.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(owner, 2)
            mgr._pending_reclaim_mib[2] = 0


# ------------------------------------------------------------------------------------------------
# the credit is taken once and the engine budget is not changed by the placement
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("auto_place,cap", [(True, 3), (False, 3), (False, 2)],
                         ids=["auto_place_site_cap_3", "pinned_site_cap_3", "pinned_site_cap_2"])
async def test_the_freed_card_is_credited_once_and_the_engine_budget_is_kept(
        tmp_path, monkeypatch, auto_place, cap):
    async with world(tmp_path, monkeypatch, models=_models(claim_auto=auto_place), max_parallel=cap) as b:
        mgr = b.mgr
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri)
        reserved = ri.reserved_need_mib
        assert not any(mgr._pending_reclaim_mib.values()), f"credit before the unload: {mgr._pending_reclaim_mib}"
        # a cold claimant may start its one engine whenever the box has a free slot, and none when it is full
        cap_before = mgr._effective_cap(TAG_CLAIM)
        assert cap_before == (1 if cap == 3 else 0), f"engine cap before: {cap_before} (box cap {cap}, 2 residents)"

        submit_claimant(b)
        await wait_served(b, "the claimant to be served on card 1")
        credit = {c: v for c, v in b.pending_after_unload[TAG_IDLE].items() if v}
        assert credit == {1: reserved}, (
            f"the credit right after the unload decision is {credit}, wanted exactly {{1: {reserved}}} "
            f"(the idle model's reserved figure, once)")
        await b.until(lambda: not any(mgr._pending_reclaim_mib.values()),
                      "the credit to be released after the teardown", 10.0)
        assert mgr._effective_cap(TAG_CLAIM) == 1, "engine cap after the placement"
        assert b.peak_residents <= cap and b.peak_alive <= cap, (
            f"the box cap {cap} was exceeded: residents peaked at {b.peak_residents}, engines at {b.peak_alive}")
        assert len([r for r in mgr._model_residents()]) == 2


# ------------------------------------------------------------------------------------------------
# a model still booting on the card that stays holds part of that card's free memory
# ------------------------------------------------------------------------------------------------
TAG_BOOTING = "model-booting"


@contextlib.asynccontextmanager
async def _booting_sibling(b, card, reserved_mib):
    """A model in RESERVED_LOADING on `card`: its weights are not in the live reading yet. Put there
    by hand and taken out again before the world shuts down."""
    sib = _resident(TAG_BOOTING, card, reserved_mib, state=ResidentState.RESERVED_LOADING)
    b.mgr._residents[TAG_BOOTING] = sib
    try:
        yield sib
    finally:
        b.mgr._residents.pop(TAG_BOOTING, None)


BOOT_FREE_MIB = _free_for_residue(500)      # card 0 reads 2800 free: 1500 above the floor after the residue


async def test_card_is_not_chosen_when_a_booting_model_takes_the_free_memory_of_the_other_card(tmp_path, monkeypatch):
    """Card 0 reads 2800 free, so on its own it clears floor + residue (2300) with 500 to spare. A
    sibling still booting there holds 503 MiB the reading does not show: 2800 - 503 - 1300 = 997,
    one below the floor. The idle model on card 1 must be left alone."""
    reserved = 503
    assert BOOT_FREE_MIB - reserved - CUDA_RESIDUE_MIB < FLOOR_MIB <= BOOT_FREE_MIB - CUDA_RESIDUE_MIB
    async with world(tmp_path, monkeypatch, models=_models(), max_parallel=4) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, BOOT_FREE_MIB)
        assert_shape(b, ri, card0_freer=False)
        async with _booting_sibling(b, 0, reserved):
            submit_claimant(b)
            await await_starved(b)
            assert untouched(b, TAG_IDLE) == [], f"the idle model was unloaded: {b.observed()}"
            assert untouched(b, TAG_BUSY) == []
            assert_no_relocation_attempt(b)


async def test_card_is_chosen_when_a_booting_model_leaves_the_other_card_enough(tmp_path, monkeypatch):
    reserved = 497      # 2800 - 497 - 1300 = 1003: three MiB above the floor
    assert BOOT_FREE_MIB - reserved - CUDA_RESIDUE_MIB == FLOOR_MIB + 3
    async with world(tmp_path, monkeypatch, models=_models(), max_parallel=4) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, BOOT_FREE_MIB)
        assert_shape(b, ri, card0_freer=False)
        async with _booting_sibling(b, 0, reserved):
            submit_claimant(b)
            await wait_served(b, f"the claimant to be placed on card 1 (card 0 keeps {FLOOR_MIB + 3} MiB above the "
                                 f"floor after the booting model's {reserved} MiB and the residue)")
            assert_served_on_card(b, 1)


async def test_booting_model_refusal_log_names_the_reservation(tmp_path, monkeypatch):
    reserved = 503
    async with world(tmp_path, monkeypatch, models=_models(), max_parallel=4) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, BOOT_FREE_MIB)
        async with _booting_sibling(b, 0, reserved):
            submit_claimant(b)
            await b.until(lambda: b.seen("log", "refused"), "a reclaim_first_refused line for the residue", 8.0)
            line = next(m for m in refusal_lines(b) if re.search(r"(^|\s)reason=residue(\s|$)", m))
            assert re.search(r"(^|\s)card=1(\s|$)", line), line
            detail = line.split("detail=", 1)[1]
            assert re.search(r"card\W*0\b", detail), detail
            assert re.search(rf"(booting|reserved)\w*\W*{reserved}(?!\d)", detail), f"the reservation is not in: {detail!r}"
            assert re.search(rf"(?<!\d){BOOT_FREE_MIB}(?!\d)", detail), f"the live free figure is not in: {detail!r}"


# ================================================================================================
# The over-eviction check: an idle model on a card where the claimant could not fit is left alone
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_an_idle_model_on_a_card_where_the_claimant_cannot_fit_is_left_alone(tmp_path, monkeypatch, auto_place):
    """Card 0: a busy model and a small idle model (the older one). Card 1: an idle model. The claimant
    fits card 1 once its idle model is gone, and fits card 0 even with the small idle model gone: so
    only the idle model on card 1 is in the way, and the small one on card 0 must not be unloaded."""
    models = _models(claim_mib=17000, claim_auto=auto_place)
    models[TAG_IDLE2] = dict(size_mib=4000, main_gpu=0)
    async with world(tmp_path, monkeypatch, models=models, max_parallel=4) as b:
        rb = await start_busy(b, "BUSY1", TAG_BUSY, 0)
        small = await serve_and_park(b, "I0", TAG_IDLE2, 0, thread="th-a2")     # parked first: the older one
        ri = await serve_and_park(b, "I1", TAG_IDLE, 1)
        assert small.last_active_monotonic < ri.last_active_monotonic
        free = b.free_list()
        need_c = b.need(TAG_CLAIM)
        assert all(f < need_c for f in free) and b.mgr._auto_pick_gpu(need_c) == (0, "layer"), f"{free} need={need_c}"
        assert free[0] + small.reserved_need_mib - FLOOR_MIB < need_c, "card 0 would hold the claimant after all"
        assert free[1] + ri.reserved_need_mib - FLOOR_MIB >= need_c

        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1")
        assert untouched(b, TAG_IDLE2) == [], (
            f"the small idle model on card 0 was acted on although the claimant could not fit there: {b.observed()}")
        assert b.seen("unload_decided", TAG_IDLE), b.observed()
        assert_served_on_card(b, 1)
        assert untouched(b, TAG_BUSY) == []


# ================================================================================================
# New behaviour: the unload is confined to the decided card
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("claimant", ["outranks_both_idle_owners", "outranks_neither"])
async def test_the_unload_stays_on_the_decided_card_whatever_the_rank_gate_would_pick(
        tmp_path, monkeypatch, claimant, auto_place):
    """Card 1 holds an idle model of the second-listed client, card 2 one of the third-listed client
    (the worst-ranked in the box). The claimant fits card 1 once its model is gone but not card 2
    (another process holds part of it). With the claimant listed first, the rank gate on its own would
    pick the card 2 model (it outranks it): the decided card is card 1, so that model must be left
    alone and the card 1 model unloaded."""
    meta = META_OWNER if claimant == "outranks_both_idle_owners" else META_THIRD
    async with world(tmp_path, monkeypatch, models=_models3(claim_auto=auto_place), rules=RULES3, n_cards=3,
                     max_parallel=4) as b:
        rb, idle = await _span3(b, owner1=META_SECOND, owner2=META_THIRD)
        b.foreign[2] = 3500
        assert_span3(b, idle, fits={1: True, 2: False})
        submit_claimant(b, meta=meta)
        await wait_served(b, "the claimant to be placed on card 1")
        assert b.seen("unload_decided", idle[1].model_tag), b.observed()
        assert untouched(b, idle[2].model_tag) == [], (
            f"the worst-ranked model on card 2 was acted on although the decided card is card 1: {b.observed()}")
        assert untouched(b, TAG_BUSY) == []
        assert_served_on_card(b, 1)


# ================================================================================================
# New behaviour: no second unload while an unload in flight already covers the need
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_no_second_unload_while_an_unload_in_flight_already_covers_the_need(tmp_path, monkeypatch, auto_place):
    async with world(tmp_path, monkeypatch, models=_models_two_small_idle_on_card_1(claim_auto=auto_place),
                     max_parallel=4) as b:
        mgr = b.mgr
        rb, idle = await _two_on_card_1(b, owner_a=META_OWNER, owner_b=META_OWNER)
        assert_order_shape(b, idle)
        first, second = idle["a"], idle["b"]          # equal ranks: the less recently active (a) goes first
        b.stop_holds[first.model_tag] = asyncio.Event()
        submit_claimant(b)
        await b.until(lambda: b.seen("engine_stop_waiting", first.model_tag),
                      "the first idle model's engine stop to be under way", 10.0)
        assert mgr._pending_reclaim_mib.get(1) == first.reserved_need_mib, (
            f"the credit while the unload is in flight: {mgr._pending_reclaim_mib}")
        before = b.count("route", "C1")
        await b.until(lambda: b.count("route", "C1") >= before + 2,
                      "the claimant to be routed twice more while the unload is still in flight", 12.0)
        assert untouched(b, second.model_tag) == [], (
            f"a second model was unloaded although the unload in flight already covers the need: {b.observed()}")
        assert b.count("unload_decided", first.model_tag) == 1
        assert not b.seen("load", TAG_CLAIM)
        b.stop_holds[first.model_tag].set()
        await wait_served(b, "the claimant to be placed on card 1 once the first engine has stopped")
        assert untouched(b, second.model_tag) == [], b.observed()
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] == [first.model_tag], b.observed()
        assert_served_on_card(b, 1)


# ================================================================================================
# New behaviour: the decision is taken again on every pass
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_the_decision_is_taken_again_and_dropped_when_the_card_no_longer_fits(tmp_path, monkeypatch, auto_place):
    """The claimant needs BOTH small idle models on card 1 gone. The first is unloaded (its stop is
    held, so the unload stays in flight). Meanwhile the second one turns busy. On the next pass the
    claimant no longer fits card 1: the override and the reclaim-first mark are cleared, and the
    second model is not touched."""
    both = 18000
    async with world(tmp_path, monkeypatch, models=_models_two_small_idle_on_card_1(claim_auto=auto_place, claim_mib=both),
                     max_parallel=4) as b:
        mgr = b.mgr
        rb, idle = await _two_on_card_1(b, owner_a=META_OWNER, owner_b=META_OWNER)
        first, second = idle["a"], idle["b"]
        free = b.free_list()
        need_c = b.need(TAG_CLAIM)
        assert all(f < need_c for f in free) and mgr._auto_pick_gpu(need_c) == (0, "layer"), f"{free} need={need_c}"
        assert free[1] + first.reserved_need_mib - FLOOR_MIB < need_c, "one idle model gone would be enough"
        assert free[1] + first.reserved_need_mib + second.reserved_need_mib - FLOOR_MIB >= need_c
        b.stop_holds[first.model_tag] = asyncio.Event()
        fake_turn = None
        try:
            submit_claimant(b)
            await b.until(lambda: b.seen("engine_stop_waiting", first.model_tag),
                          "the first idle model's engine stop to be under way", 10.0)
            slot = b.slots["C1"]
            assert slot.fastlane_relocated_main_gpu == 1 and getattr(slot, "fastlane_reclaim_first", False) is True, (
                f"while the unload is in flight the claimant should be decided on card 1: "
                f"override={slot.fastlane_relocated_main_gpu} mark={getattr(slot, 'fastlane_reclaim_first', None)}; "
                f"{b.observed()}")
            # the second model turns busy before it could be unloaded
            fake_turn = Slot.new(second.model_tag)
            second.active_slot = fake_turn
            assert mgr._resident_has_running_turn(second)
            before = b.count("route", "C1")
            await b.until(lambda: b.count("route", "C1") >= before + 2,
                          "the claimant to be routed twice more", 12.0)
            assert slot.fastlane_relocated_main_gpu is None and slot.fastlane_relocated_split_mode is None, (
                f"the override stayed after the card stopped fitting: {slot.fastlane_relocated_main_gpu}; {b.observed()}")
            assert getattr(slot, "fastlane_reclaim_first", False) is False, "the reclaim-first mark stayed"
            assert untouched(b, second.model_tag) == [], f"the busy model was acted on: {b.observed()}"
            assert not b.seen("load", TAG_CLAIM)
        finally:
            if fake_turn is not None:
                second.active_slot = None


# ================================================================================================
# New behaviour: an override that is not ours is left alone
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_an_override_set_by_the_rank_path_is_left_alone_by_the_decision(tmp_path, monkeypatch, auto_place):
    """The claimant already carries an override to card 0 that something else set (the mark is off).
    The decision must not replace it with card 1, and must not mark the claimant."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB, claim_auto=auto_place)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_span_shape(b, ri)

        def set_override(slot):
            slot.fastlane_relocated_main_gpu = 0
            slot.fastlane_relocated_split_mode = "none"

        b.route_hooks["C1"] = set_override
        submit_claimant(b)
        await await_starved(b)
        slot = b.slots["C1"]
        assert slot.fastlane_relocated_main_gpu == 0 and slot.fastlane_relocated_split_mode == "none", (
            f"the override that was not the decision's own changed to card {slot.fastlane_relocated_main_gpu}; {b.observed()}")
        assert getattr(slot, "fastlane_reclaim_first", False) is False, "the claimant was marked reclaim-first"
        assert not b.seen("log", "reclaim"), f"a card was chosen over the existing override: {reclaim_lines(b)}"
        assert untouched(b, TAG_IDLE) == [] and untouched(b, TAG_BUSY) == [], b.observed()


# ================================================================================================
# A card the placer picks that fits now never triggers the decision
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_a_card_the_placer_picks_that_fits_now_triggers_nothing(tmp_path, monkeypatch, auto_place):
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=8000, claim_auto=auto_place)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        need_c = b.need(TAG_CLAIM)
        free = b.free_list()
        assert b.mgr._auto_pick_gpu(need_c) == (0, "none"), f"the placer should pick card 0: {free} need={need_c}"
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on the card the placer picked")
        assert_served_on_card(b, 0)
        assert not b.seen("log", "reclaim") and not refusal_lines(b), b.observed()
        slot = b.slots["C1"]
        assert slot.fastlane_relocated_main_gpu is None and getattr(slot, "fastlane_reclaim_first", False) is False
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] == [], f"something was unloaded: {b.observed()}"
        await assert_busy_intact(b, rb)


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_a_stale_decision_is_dropped_when_the_placer_now_picks_a_card_that_fits(tmp_path, monkeypatch, auto_place):
    """A claimant that carries a reclaim-first decision for card 1 from an earlier pass, but now fits
    card 0 as the placer picks it, runs on card 0: the override and the mark are cleared."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=11500, claim_auto=auto_place)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        need_c = b.need(TAG_CLAIM)
        free = b.free_list()
        assert b.mgr._auto_pick_gpu(need_c) == (0, "none"), f"the placer should pick card 0: {free} need={need_c}"
        assert free[1] < need_c, f"the stale card must not hold the claimant as it stands: {free} need={need_c}"

        def stale_decision(slot):
            slot.fastlane_relocated_main_gpu = 1
            slot.fastlane_relocated_split_mode = "none"
            slot.fastlane_reclaim_first = True

        b.route_hooks["C1"] = stale_decision
        submit_claimant(b)
        await wait_served(b, "the claimant to be served")
        slot = b.slots["C1"]
        assert slot.fastlane_relocated_main_gpu is None and slot.fastlane_relocated_split_mode is None, (
            f"the stale override stayed: card {slot.fastlane_relocated_main_gpu}; {b.observed()}")
        assert getattr(slot, "fastlane_reclaim_first", False) is False
        assert_served_on_card(b, 0)
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] == []


# ================================================================================================
# Direct checks of the card choice and of the gate
# ================================================================================================
def _scripted_world(tmp_path, monkeypatch, *, claim_mib=SPAN_MIB, extra_idle=None):
    mgr, probe = _bare_manager(tmp_path, monkeypatch, [CARD_MIB - BUSY_MIB, CARD_MIB - IDLE_MIB],
                               models=_models(claim_mib=claim_mib))
    mgr._residents[TAG_BUSY] = _resident(TAG_BUSY, 0, BUSY_MIB, state=ResidentState.ACTIVE)
    ri = _resident(TAG_IDLE, 1, IDLE_MIB)
    mgr._residents[TAG_IDLE] = ri
    return mgr, probe, ri


async def test_card_choice_answers_with_the_card_and_the_figures(tmp_path, monkeypatch):
    mgr, _p, ri = _scripted_world(tmp_path, monkeypatch)
    need_c = mgr._read_model_footprint(TAG_CLAIM)[0]
    async with mgr._registry_lock:
        choice = mgr._reclaim_first_card_locked(_claim_slot(), need_c, 1)
    assert choice is not None, "the claimant fits card 1 once the idle model is gone but got no card"
    assert (choice.card, choice.reclaimable_mib, choice.pending_mib) == (1, ri.reserved_need_mib, 0), choice
    assert choice.needed_mib > 0, f"nothing is on its way back yet, so something must still be unloaded: {choice}"
    assert ri.state is ResidentState.IDLE_EVICTABLE and mgr._residents.get(TAG_IDLE) is ri, "the helper acted on a resident"


async def test_card_choice_counts_a_credit_pending_and_does_not_stop(tmp_path, monkeypatch):
    mgr, _p, ri = _scripted_world(tmp_path, monkeypatch)
    need_c = mgr._read_model_footprint(TAG_CLAIM)[0]
    slot = _claim_slot()
    async with mgr._registry_lock:
        none_pending = mgr._reclaim_first_card_locked(slot, need_c, 1)
        mgr._pending_reclaim_mib[1] = 400
        with_pending = mgr._reclaim_first_card_locked(slot, need_c, 1)
        mgr._pending_reclaim_mib[1] = ri.reserved_need_mib
        covered = mgr._reclaim_first_card_locked(slot, need_c, 1)
        mgr._pending_reclaim_mib[1] = 0
        after = mgr._reclaim_first_card_locked(slot, need_c, 1)
    assert none_pending is not None and none_pending.pending_mib == 0
    assert with_pending is not None and with_pending.card == 1 and with_pending.pending_mib == 400, (
        f"a credit pending on the card stopped the choice or was not counted: {with_pending!r}")
    assert with_pending.needed_mib < none_pending.needed_mib, (
        f"the credit did not lower what is still to unload: {with_pending} vs {none_pending}")
    assert covered is not None and covered.needed_mib == 0, f"a credit that covers the need: {covered!r}"
    assert after == none_pending, f"credit back to zero: {after!r}"


async def test_card_choice_does_not_count_a_layer_split_resident(tmp_path, monkeypatch):
    mgr, _p, ri = _scripted_world(tmp_path, monkeypatch)
    need_c = mgr._read_model_footprint(TAG_CLAIM)[0]
    slot = _claim_slot()
    async with mgr._registry_lock:
        counted = mgr._reclaim_first_card_locked(slot, need_c, 1)
        ri.split_mode = "layer"
        not_counted = mgr._reclaim_first_card_locked(slot, need_c, 1)
    assert counted is not None, "positive control: the same resident, split none, was not counted"
    assert not_counted is None, f"a layer-split resident was counted as reclaimable on one card: {not_counted!r}"


async def test_card_choice_refuses_when_the_probe_reads_nothing(tmp_path, monkeypatch, caplog):
    import logging
    mgr, probe, _ri = _scripted_world(tmp_path, monkeypatch)
    need_c = mgr._read_model_footprint(TAG_CLAIM)[0]
    slot = _claim_slot()
    probe["free"] = []
    with caplog.at_level(logging.INFO, logger="turbohaul.manager"):
        async with mgr._registry_lock:
            choice = mgr._reclaim_first_card_locked(slot, need_c, 1)
    assert choice is None
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reclaim_first_refused ")]
    assert lines and all(re.search(r"(^|\s)reason=probe_unreadable(\s|$)", m) for m in lines), lines


# needs on either side of every figure the probe can produce (10000, 9000 and their sum 19000), so
# a gate that moved by a single MiB for an empty reclaim would flip at least one cell
NEEDS = (8000, 8999, 9000, 9001, 9999, 10000, 10001, 12000, 18999, 19000, 19001, 19500)


def _gate(mgr, need, parallel, card, split, credit, reloc, variant, reclaim=None):
    kw = dict(credit_pending=credit, relocatable=reloc)
    if variant == "none":
        kw["reclaim_mib_by_card"] = None
    elif variant == "empty":
        kw["reclaim_mib_by_card"] = {}
    elif variant == "dict":
        kw["reclaim_mib_by_card"] = reclaim
    return mgr._vram_admits_locked(need, parallel, card, split, **kw)


async def test_gate_with_empty_reclaim_is_identical_to_the_plain_gate(tmp_path, monkeypatch):
    """On a tree whose gate has no `reclaim_mib_by_card` argument this fails with TypeError at the
    first call that passes it (the argument does not exist yet): the expected structural failure."""
    mgr, probe = _bare_manager(tmp_path, monkeypatch, [10000, 9000])
    siblings = {
        "none": [],
        "idle_one_card_on_0": [_resident("sib", 0, 1500)],
        "booting_one_card_on_0": [_resident("sib", 0, 1500, state=ResidentState.RESERVED_LOADING)],
        "layer_split": [_resident("sib", 0, 1500, split="layer")],
    }
    results = []
    cells = 0
    for sib_name, sibs in siblings.items():
        mgr._residents.clear()
        for r in sibs:
            mgr._residents[r.resident_key] = r
        for pending in ({}, {0: 3000, 1: 500}):
            mgr._pending_reclaim_mib.clear()
            mgr._pending_reclaim_mib.update(pending)
            for free in ([10000, 9000], []):
                probe["free"] = free
                for split in ("none", "layer"):
                    for credit in (False, True):
                        for reloc in (False, True):
                            for card in (0, 1):
                                for need in NEEDS:
                                    plain = _gate(mgr, need, 1, card, split, credit, reloc, "omitted")
                                    got_none = _gate(mgr, need, 1, card, split, credit, reloc, "none")
                                    got_empty = _gate(mgr, need, 1, card, split, credit, reloc, "empty")
                                    cells += 1
                                    results.append(plain)
                                    assert plain == got_none == got_empty, (
                                        f"gate differs: siblings={sib_name} pending={pending} free={free} split={split} "
                                        f"credit={credit} reloc={reloc} card={card} need={need}: "
                                        f"omitted={plain} None={got_none} {{}}={got_empty}")
    assert cells == 4 * 2 * 2 * 2 * 2 * 2 * 2 * len(NEEDS)
    assert True in results and False in results, "the grid never produced both answers: the instrument is blind"
    assert results.count(True) >= 100 and results.count(False) >= 100, (
        f"the grid is lopsided ({results.count(True)} admits, {results.count(False)} refusals)")


async def test_gate_with_reclaim_flips_exactly_at_the_fit_boundary(tmp_path, monkeypatch):
    mgr, probe = _bare_manager(tmp_path, monkeypatch, [10000, 9000])

    def gate(need, card, split="none", reclaim=None, credit=False, reloc=False):
        return _gate(mgr, need, 1, card, split, credit, reloc, "dict", reclaim)

    # one card: the reclaimed MiB are added to that card's own free figure and to no other
    assert gate(12000, 0) is False
    assert gate(12000, 0, reclaim={0: 1999}) is False
    assert gate(12000, 0, reclaim={0: 2000}) is True
    assert gate(12000, 0, reclaim={1: 99999}) is False, "another card's reclaim changed a one-card answer"
    assert gate(12000, 0, reclaim={0: 2000, 1: 99999}) is True
    assert gate(12000, 1, reclaim={1: 2999}) is False
    assert gate(12000, 1, reclaim={1: 3000}) is True
    assert gate(12000, 1, reclaim={0: 99999}) is False
    # a spanning claimant budgets against the sum of every card, so any card's reclaim counts
    assert gate(21000, 0, split="layer") is False
    assert gate(21000, 0, split="layer", reclaim={0: 1999}) is False
    assert gate(21000, 0, split="layer", reclaim={0: 1000, 1: 1000}) is True
    assert gate(21000, 0, split="layer", reclaim={1: 2000}) is True
    assert gate(21000, 0, split="layer", reclaim={0: 1000, 1: 999}) is False
    # a model that is still booting on the card keeps its reserved share out of that card
    mgr._residents["boot"] = _resident("boot", 0, 1500, state=ResidentState.RESERVED_LOADING)
    assert gate(9000, 0) is False
    assert gate(9000, 0, reclaim={0: 499}) is False
    assert gate(9000, 0, reclaim={0: 500}) is True
    # a sibling that spans cards still refuses a one-card claimant, however much is reclaimed
    mgr._residents["boot"] = _resident("boot", 0, 1500, split="layer")
    assert gate(100, 0, reclaim={0: 99999, 1: 99999}) is False


# ================================================================================================
# The card choice, number by number. The gate on its own has no floor term and nets nothing for the
# floor, so the decided card's safety margin, the booting models, the credit of unloads in flight and
# the gate's structural refusals each need their own pin.
# ================================================================================================
def _decided_card_world(tmp_path, monkeypatch, *, free0=13000, free1=11500, idle1=12500, busy_split="none",
                        pending1=0, extra=()):
    """Two cards. Card 0: a busy model (and anything in `extra`). Card 1: an idle one-card model that
    holds `idle1` MiB. The probe reads `free0` / `free1`; `pending1` MiB are on their way back on
    card 1 from an unload in flight."""
    mgr, probe = _bare_manager(tmp_path, monkeypatch, [free0, free1])
    mgr._residents[TAG_BUSY] = _resident(TAG_BUSY, 0, CARD_MIB - free0, state=ResidentState.ACTIVE, split=busy_split)
    ri = _resident(TAG_IDLE, 1, idle1)
    mgr._residents[TAG_IDLE] = ri
    for r in extra:
        mgr._residents[r.resident_key] = r
    if pending1:
        mgr._pending_reclaim_mib[1] = pending1
    return mgr, probe, ri


async def _choose(mgr, need, slot=None):
    async with mgr._registry_lock:
        return mgr._reclaim_first_card_locked(slot or _claim_slot(), need, 1)


# the card must keep the safety floor free: free 11500 + reclaim 12500 = 24000, so the floor holds
# exactly up to a need of 23000
@pytest.mark.parametrize("need_mib,admitted", [
    (22000, True), (23000, True), (23001, False), (23500, False), (24000, False), (24001, False)])
async def test_the_decided_card_must_keep_the_floor_free_after_the_unload(tmp_path, monkeypatch, need_mib, admitted):
    mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch)
    choice = await _choose(mgr, need_mib)
    if admitted:
        assert choice is not None and choice.card == 1, (
            f"need {need_mib}: card 1 holds it with the floor ({FLOOR_MIB}) to spare but got {choice!r}")
    else:
        assert choice is None, (
            f"need {need_mib}: free 11500 + reclaim 12500 - floor {FLOOR_MIB} = 23000 is below the need, "
            f"yet card 1 was chosen: {choice!r}")


@pytest.mark.parametrize("site", ["no_aggregate_auto_place_site", "no_aggregate_pinned_site"])
@pytest.mark.parametrize("need_mib,fits", [(23000, True), (23001, False), (23500, False)])
async def test_the_decided_card_keeps_the_floor_free_on_the_real_path(tmp_path, monkeypatch, site, need_mib, fits):
    """Card 1 reads 11000 free and its idle model holds 13000, so with the floor the claimant fits
    there up to a need of exactly 23000. One MiB more and it must wait, with the idle model untouched."""
    spec = LEAD_VARIANTS[site]
    models = _lead_models(claim_auto=spec["auto"])
    models[TAG_CLAIM]["expected_mib"] = need_mib
    async with world(tmp_path, monkeypatch, models=models) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, 12000)
        assert b.need(TAG_CLAIM) == need_mib
        free = b.free_list()
        assert free == [12000, 11000], free
        assert b.mgr._auto_pick_gpu(need_mib) == (None, "none"), "the cards together would take it"
        assert free[1] + ri.reserved_need_mib - FLOOR_MIB == 23000
        submit_claimant(b)
        if fits:
            await wait_served(b, "the claimant to be placed on card 1 (it fits with the floor to spare)")
            assert_served_on_card(b, 1)
            assert_saved_then_unloaded(b, TAG_IDLE)
            await assert_busy_intact(b, rb)
        else:
            await await_starved(b)
            assert untouched(b, TAG_IDLE) == [], (
                f"the idle model was unloaded for a claimant that would leave card 1 below the floor: {b.observed()}")
            assert untouched(b, TAG_BUSY) == []
            assert_no_relocation_attempt(b)
            assert any(re.search(r"(^|\s)card=1(\s|$)", m) and re.search(r"(^|\s)reason=no_fit(\s|$)", m)
                       for m in refusal_lines(b)), refusal_lines(b)


# a model still booting on the decided card holds memory the reading does not show
@pytest.mark.parametrize("need_mib,admitted", [(22000, True), (22001, False), (22500, False)])
async def test_a_booting_model_on_the_decided_card_counts_against_the_floor_margin(
        tmp_path, monkeypatch, need_mib, admitted):
    """Card 1 reads 11500 free, its idle model holds 12500, and a sibling still booting there holds
    1000 the reading does not show. Net of the booting model the card has 10500 + 12500 = 23000, so with
    the floor it holds a claimant up to exactly 22000. Without the booting share it would look as if
    it held one up to 23000."""
    boot = _resident(TAG_BOOTING, 1, 1000, state=ResidentState.RESERVED_LOADING)
    mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch, extra=(boot,))
    choice = await _choose(mgr, need_mib)
    if admitted:
        assert choice is not None and choice.card == 1, f"need {need_mib} fits net of the booting model: {choice!r}"
    else:
        assert choice is None, (
            f"need {need_mib}: card 1 net of the booting model has 23000 - floor {FLOOR_MIB} = 22000, "
            f"yet it was chosen: {choice!r}")


@pytest.mark.parametrize("site", ["no_aggregate_auto_place_site", "no_aggregate_pinned_site"])
@pytest.mark.parametrize("booting_mib,fits", [(0, True), (500, False)])
async def test_a_booting_model_on_the_decided_card_counts_on_the_real_path(
        tmp_path, monkeypatch, site, booting_mib, fits):
    """The claimant needs 22600. Card 1 reads 11000 free and its idle model holds 13000: 24000 less the
    floor holds it. With a model still booting on card 1 that holds 500 more, net 23500 less the floor
    is 22500, one short of 22600 on the card the reading shows, so it must wait."""
    spec = LEAD_VARIANTS[site]
    models = _lead_models(claim_auto=spec["auto"])
    models[TAG_CLAIM]["expected_mib"] = 22600
    async with world(tmp_path, monkeypatch, models=models, max_parallel=4) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, 12000)
        assert b.need(TAG_CLAIM) == 22600 and b.free_list() == [12000, 11000]
        async with (_booting_sibling(b, 1, booting_mib) if booting_mib else contextlib.nullcontext()):
            assert b.mgr._auto_pick_gpu(22600) == (None, "none"), "the cards together would take it"
            submit_claimant(b)
            if fits:
                await wait_served(b, "the claimant to be placed on card 1 (no booting model there)")
                assert_served_on_card(b, 1)
            else:
                await await_starved(b)
                assert untouched(b, TAG_IDLE) == [], (
                    f"the idle model was unloaded although the booting model leaves card 1 short: {b.observed()}")
                assert_no_relocation_attempt(b)


# unloads already in flight count once, in the fit and in the figures of the choice
@pytest.mark.parametrize("need_mib,admitted", [(21500, True), (21501, False), (22000, False)])
async def test_a_credit_pending_on_the_decided_card_is_counted_once_in_the_fit(
        tmp_path, monkeypatch, need_mib, admitted):
    """Card 1 reads 7000 free, its idle model holds 12500 and 3000 are on their way back from an
    unload in flight: 7000 + 3000 + 12500 = 22500, so with the floor the card holds a claimant up to
    exactly 21500. Counting the 3000 twice would admit one up to 24500."""
    mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch, free1=7000, pending1=3000)
    choice = await _choose(mgr, need_mib)
    if admitted:
        assert choice is not None, f"need {need_mib} fits once, with the floor to spare, but got no card"
        assert tuple(choice) == (1, 12500, 3000, 12500), (
            f"the figures of the choice (card, reclaimable, pending, still to unload): {tuple(choice)}")
    else:
        assert choice is None, f"need {need_mib}: 22500 - floor = 21500 is below it, yet card 1 was chosen: {choice!r}"


async def test_a_card_that_fits_only_through_the_credit_pending_is_chosen(tmp_path, monkeypatch):
    """Free 7000 + the idle model's 12500 = 19500 is short of the need of 22000 on its own; the 5000
    already on their way back make it fit with the floor to spare. The gate must credit them."""
    mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch, free1=7000, pending1=5000)
    choice = await _choose(mgr, 22000)
    assert choice is not None and choice.card == 1, (
        f"the card fits through the credit pending (7000 + 5000 + 12500 - floor >= 22000) but got {choice!r}")
    assert (choice.reclaimable_mib, choice.pending_mib) == (12500, 5000), tuple(choice)


# a resident that is not tensor-isolated makes every one-card placement unsafe
@pytest.mark.parametrize("sibling", ["none_split_control", "busy_layer_split_on_card_0", "idle_layer_split_on_card_0"])
async def test_a_sibling_that_is_not_one_card_refuses_every_card(tmp_path, monkeypatch, sibling):
    """The numbers let the claimant fit card 1 once its idle model is gone (11500 + 12500 - 1000 >=
    22000). Only the split of the other resident differs: with every resident on one card the choice is
    card 1, with a layer-split resident anywhere it must be none."""
    if sibling == "none_split_control":
        mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch)
    elif sibling == "busy_layer_split_on_card_0":
        mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch, busy_split="layer")
    else:
        lay = _resident("model-layer", 0, 3000, split="layer")
        mgr, _p, _ri = _decided_card_world(tmp_path, monkeypatch, extra=(lay,))
    choice = await _choose(mgr, 22000)
    if sibling == "none_split_control":
        assert choice is not None and choice.card == 1, f"positive control: {choice!r}"
    else:
        assert choice is None, f"[{sibling}] a card was chosen although a resident spans the cards: {choice!r}"


async def test_a_busy_layer_split_resident_leaves_the_claimant_to_todays_path(tmp_path, monkeypatch):
    """The busy model on card 0 spans the cards (hand-set). The idle model on card 1 would make the
    claimant fit card 1 by the numbers, but no one-card placement is safe next to a layer-split
    resident: nothing is relocated and nothing is unloaded."""
    models = _lead_models(claim_auto=False)
    async with world(tmp_path, monkeypatch, models=models) as b:
        rb, ri = await busy_card0_idle_card1(b)
        b.set_free(0, 12000)
        rb.split_mode = "layer"
        try:
            assert free_ok(b)
            submit_claimant(b)
            await await_starved(b)
            assert untouched(b, TAG_IDLE) == [], f"the idle model was unloaded: {b.observed()}"
            assert untouched(b, TAG_BUSY) == []
            assert_no_relocation_attempt(b)
        finally:
            rb.split_mode = "none"


def free_ok(b):
    assert b.free_list() == [12000, 11000] and b.need(TAG_CLAIM) == LEAD_NEED_MIB
    return True


# a layer-split idle resident is not reclaimable on any card, so it must not even cause a refusal line
@pytest.mark.parametrize("case", ["layer_split_idle", "one_card_idle_control"])
@pytest.mark.parametrize("probe", ["readable", "unreadable"])
async def test_nothing_reclaimable_means_nothing_is_logged(tmp_path, monkeypatch, caplog, case, probe):
    import logging
    split = "layer" if case == "layer_split_idle" else "none"
    mgr, p = _bare_manager(tmp_path, monkeypatch, [13000, 11500])
    mgr._residents["idle0"] = _resident("idle0", 0, 11000, split=split)
    if probe == "unreadable":
        p["free"] = []
    with caplog.at_level(logging.INFO, logger="turbohaul.manager"):
        async with mgr._registry_lock:
            choice = mgr._reclaim_first_card_locked(_claim_slot(), 30000, 1)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reclaim_first_refused ")]
    assert choice is None
    if case == "layer_split_idle":
        assert lines == [], f"a layer-split resident is not reclaimable, yet a refusal was logged: {lines}"
    else:
        assert lines, "positive control: a reclaimable one-card resident that cannot fit is logged"


# ================================================================================================
# The make-room arm that looks only on the decided card runs only for a claimant that carries the
# decision. A request without it keeps the arm it always had.
# ================================================================================================
async def test_a_request_the_decision_found_no_card_for_keeps_the_global_eviction_arm(tmp_path, monkeypatch):
    """An auto_place claimant with no live claim whose model fits no card even after reclaim carries no
    decision, so it takes the plain make-room arm: the least recently active idle model of ANY card is
    unloaded, as before. A victim lookup confined to the claimant's own card (card 0, which holds no
    idle model) would unload nothing."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=TOO_BIG_MIB, claim_auto=True)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_shape(b, ri, fits_after=False)
        submit_claimant(b, meta=META_UNLISTED)
        await b.until(lambda: b.seen("unload_decided", TAG_IDLE) or b.count("log", "starved") >= 2,
                      "the claimant to be dealt with by the make-room step", 12.0)
        slot = b.slots["C1"]
        assert b.mgr.queue._fastlane_priority_key(slot) is None
        assert getattr(slot, "fastlane_reclaim_first", False) is False and slot.fastlane_relocated_main_gpu is None
        assert b.seen("unload_decided", TAG_IDLE), (
            f"the plain make-room arm unloads the idle model of any card; nothing was unloaded: {b.observed()}")
        assert untouched(b, TAG_BUSY) == []


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_the_rank_arm_still_unloads_the_worse_ranked_idle_model_when_the_decision_found_no_card(
        tmp_path, monkeypatch, auto_place):
    """The claimant is listed first and its model fits no card even after reclaim, so it carries no
    decision. The idle model of the second-listed client on card 1 is worse ranked, so the rank arm
    unloads it (and on a pinned manifest relocates the claimant to its card, with the reclaim-first mark
    still off). An arm confined to the claimant's own card would find no victim."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=TOO_BIG_MIB, claim_auto=auto_place),
                     rules=RULES) as b:
        rb = await start_busy(b, "BUSY1", TAG_BUSY, 0)
        ri = await serve_and_park(b, "I1", TAG_IDLE, 1, meta=META_SECOND)
        assert_shape(b, ri, fits_after=False)
        submit_claimant(b, meta=META_OWNER)
        await b.until(lambda: b.seen("unload_decided", TAG_IDLE) or b.count("log", "starved") >= 2,
                      "the claimant to be dealt with by the make-room step", 12.0)
        slot = b.slots["C1"]
        assert getattr(slot, "fastlane_reclaim_first", False) is False, "the claimant was marked reclaim-first"
        assert b.seen("unload_decided", TAG_IDLE), (
            f"the rank arm should unload the worse-ranked model on card 1; nothing was unloaded: {b.observed()}")
        assert slot.fastlane_relocated_main_gpu == (None if auto_place else 1), (
            f"override {slot.fastlane_relocated_main_gpu}; observed: {b.observed()}")
        assert untouched(b, TAG_BUSY) == []


# ================================================================================================
# The choice between usable cards: the fewest MiB still to unload comes before the headroom
# ================================================================================================
@pytest.mark.parametrize("a_card", [0, 1], ids=["fewer_to_unload_on_card_0", "fewer_to_unload_on_card_1"])
async def test_the_card_that_needs_the_fewest_mib_unloaded_wins_over_the_card_with_more_headroom(
        tmp_path, monkeypatch, a_card):
    """Need 15000, floor 1000. Card A reads 14000 free and holds a small idle model (3000): still to
    unload 15000 + 1000 - 14000 = 2000, headroom 14000 + 3000 - 1000 - 15000 = 1000. Card B reads 6000
    free and holds a large idle model (17000): still to unload 10000, headroom 6000 + 17000 - 1000 -
    15000 = 7000. Both are usable. Card A needs far less unloading and must win, although B keeps
    more headroom; the card index plays no part (A is card 0 in one case and card 1 in the other)."""
    b_card = 1 - a_card
    free = [0, 0]
    free[a_card], free[b_card] = 14000, 6000
    mgr, _p = _bare_manager(tmp_path, monkeypatch, free)
    mgr._residents["idle-a"] = _resident("idle-a", a_card, 3000)
    mgr._residents["idle-b"] = _resident("idle-b", b_card, 17000)
    choice = await _choose(mgr, 15000)
    assert choice is not None, "both cards are usable but no card was chosen"
    assert tuple(choice) == (a_card, 3000, 0, 2000), (
        f"(card, reclaimable, pending, still to unload) = {tuple(choice)}; wanted card {a_card} "
        f"(2000 MiB still to unload) over card {b_card} (10000 still to unload, larger headroom)")


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_the_card_that_needs_the_fewest_mib_unloaded_wins_on_the_real_path(tmp_path, monkeypatch, auto_place):
    """Card 0 reads 14000 free (another process holds 7000) with an idle model of 3000; card 1 reads
    6000 free (another process holds 1000) with an idle model of 17000. The claimant needs 15000: no card
    holds it as things stand, the cards together do, and both cards hold it after their idle model is
    gone. The card with the fewer MiB still to unload (card 0) wins: only its small idle model is
    unloaded and the claimant is served there."""
    models = {
        TAG_IDLE: dict(size_mib=1000, expected_mib=17000, main_gpu=1),
        TAG_IDLE2: dict(size_mib=1000, expected_mib=3000, main_gpu=0),
        TAG_CLAIM: dict(size_mib=1000, expected_mib=15000, main_gpu=0, auto_place=auto_place),
    }
    async with world(tmp_path, monkeypatch, models=models) as b:
        big = await serve_and_park(b, "I1", TAG_IDLE, 1)                    # parked first: the older one
        small = await serve_and_park(b, "I0", TAG_IDLE2, 0, thread="th-a2")
        b.set_free(0, 14000)
        b.set_free(1, 6000)
        need_c = b.need(TAG_CLAIM)
        assert need_c == 15000 and big.reserved_need_mib == 17000 and small.reserved_need_mib == 3000
        assert b.free_list() == [14000, 6000]
        assert b.mgr._auto_pick_gpu(need_c) == (0, "layer"), "the placer should give the claimant a layer split"
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 0 after the small idle model there was unloaded")
        assert_served_on_card(b, 0)
        assert [m for (_t, k, m) in b.events if k == "unload_decided"] == [TAG_IDLE2], (
            f"only the small idle model on card 0 should be unloaded: {b.observed()}")
        assert untouched(b, TAG_IDLE) == [], f"the large idle model on card 1 was acted on: {b.observed()}"
        lines = reclaim_lines(b)
        assert len(lines) == 1 and re.search(r"(^|\s)card=0(\s|$)", lines[0]) and "reclaimable_mib=3000" in lines[0], lines


# ================================================================================================
# The decision line is logged when the decision is made or changes, not on every pass
# ================================================================================================
async def test_the_decision_line_is_logged_once_while_the_decision_stays_the_same(tmp_path, monkeypatch):
    """The claimant is decided for card 1 and its idle model's unload is held in flight, so the claimant
    is routed again and again with the same decision. The line that names the card is written once."""
    async with world(tmp_path, monkeypatch, models=_models(claim_mib=SPAN_MIB)) as b:
        rb, ri = await busy_card0_idle_card1(b)
        assert_span_shape(b, ri)
        b.stop_holds[TAG_IDLE] = asyncio.Event()
        submit_claimant(b)
        await b.until(lambda: b.seen("engine_stop_waiting", TAG_IDLE), "the idle model's engine stop to be under way", 10.0)
        before = b.count("route", "C1")
        await b.until(lambda: b.count("route", "C1") >= before + 3,
                      "the claimant to be routed three more times with the same decision", 15.0)
        lines = reclaim_lines(b)
        assert len(lines) == 1 and re.search(r"(^|\s)card=1(\s|$)", lines[0]), (
            f"the decision line should be written once for an unchanged decision, got {len(lines)}: {lines}")
        b.stop_holds[TAG_IDLE].set()
        await wait_served(b, "the claimant to be placed on card 1 once the engine has stopped")
        assert len(reclaim_lines(b)) == 1, reclaim_lines(b)


async def test_the_decision_line_is_logged_again_when_the_decided_card_changes(tmp_path, monkeypatch):
    """Cards 1 and 2 each hold an idle model and both would take the claimant; card 1 is decided first.
    With its unload held in flight, another process takes most of card 1, so it no longer fits and
    card 2 is decided. A second line, naming card 2, is written; and only that one more."""
    async with world(tmp_path, monkeypatch, models=_models3(), n_cards=3, max_parallel=4) as b:
        rb, idle = await _span3(b)
        assert_span3(b, idle, fits={1: True, 2: True})
        b.stop_holds[TAG_IDLE] = asyncio.Event()
        submit_claimant(b)
        await b.until(lambda: b.seen("engine_stop_waiting", TAG_IDLE), "card 1's engine stop to be under way", 10.0)
        before = b.count("route", "C1")
        await b.until(lambda: b.count("route", "C1") >= before + 2,
                      "the claimant to be routed twice more with the same decision", 12.0)
        first = reclaim_lines(b)
        assert len(first) == 1 and re.search(r"(^|\s)card=1(\s|$)", first[0]), first
        b.foreign[1] = 9000          # card 1 now reads 2460 free: its credit pending no longer makes it fit
        assert b.free_list()[1] == 2460, b.free_list()
        await b.until(lambda: b.seen("unload_decided", TAG_IDLE2),
                      "the claimant to be decided for card 2 and its idle model unloaded", 15.0)
        await wait_served(b, "the claimant to be placed on card 2 once its idle model was unloaded")
        lines = reclaim_lines(b)
        assert len(lines) == 2 and re.search(r"(^|\s)card=2(\s|$)", lines[1]), (
            f"expected the first line (card 1) and one more for card 2, got {len(lines)}: {lines}")


# ================================================================================================
# The reclaimable set is exactly the idle-evictable state
# ================================================================================================
async def test_the_reclaimable_set_is_pinned_to_idle_evictable_against_a_future_grace_state(tmp_path, monkeypatch):
    """Nothing assigns the grace state to a resident today (a model in its grace window is in the
    active state). This pins the reclaimable set to the idle-evictable state, so that if the grace state
    is ever assigned, a model inside its grace window is not counted as reclaimable by mistake. The
    state is set by hand and restored."""
    mgr, _p, ri = _decided_card_world(tmp_path, monkeypatch)
    positive = await _choose(mgr, 22000)
    assert positive is not None and positive.card == 1, f"positive control: the idle model is counted: {positive!r}"
    ri.state = ResidentState.GRACE
    try:
        grace = await _choose(mgr, 22000)
    finally:
        ri.state = ResidentState.IDLE_EVICTABLE
    assert grace is None, f"a resident in the grace state was counted as reclaimable: {grace!r}"


# ================================================================================================
# Routing a stand-in slot that has no reclaim-first attribute (a plain namespace with only the
# fields that existed before). Every read of the mark on the routing path must tolerate its absence.
# ================================================================================================
def _stand_in_slot(tag=TAG_CLAIM, **extra):
    import types
    fields = dict(model_tag=tag, fastlane=None, fastlane_relocated_main_gpu=None, fastlane_relocated_split_mode=None)
    fields.update(extra)
    return types.SimpleNamespace(**fields)


async def _route_stand_in(mgr, slot):
    try:
        await mgr._route_or_reserve(slot)
    except AttributeError as e:
        raise AssertionError(f"routing a slot without the reclaim-first attribute raised: {e}") from None


def _route_world(tmp_path, monkeypatch, *, free, models, residents, rules, on_evict=None, ):
    mgr, probe = _bare_manager(tmp_path, monkeypatch, free, rules=rules, models=models)
    for r in residents:
        mgr._residents[r.resident_key] = r
    calls = {"reserved": [], "deferred": [], "evicted": [], "gate": []}

    async def spy_reserve(slot, main_gpu_override=None, split_mode_override=None):
        calls["reserved"].append((slot.model_tag, main_gpu_override, split_mode_override))

    def spy_defer(slot, *, evict_pending=False):
        calls["deferred"].append((slot.model_tag, evict_pending))

    def spy_evict(r):
        calls["evicted"].append(r.model_tag)
        if on_evict is not None:
            on_evict(mgr, probe, r)

    real_gate = mgr._vram_admits_locked

    def spy_gate(*a, **k):
        out = real_gate(*a, **k)
        calls["gate"].append((a[2] if len(a) > 2 else k.get("main_gpu"), dict(k), out))
        return out

    mgr._reserve_and_start_locked = spy_reserve
    mgr._defer_unroutable = spy_defer
    mgr._begin_unload_locked = spy_evict
    mgr._vram_admits_locked = spy_gate
    return mgr, probe, calls


def _stand_in_models(*, claim_mib, claim_auto):
    return {
        TAG_BUSY: dict(size_mib=1000, expected_mib=11000, main_gpu=0),
        TAG_IDLE: dict(size_mib=1000, expected_mib=13000, main_gpu=1),
        TAG_CLAIM: dict(size_mib=1000, expected_mib=claim_mib, main_gpu=0, auto_place=claim_auto),
    }


def _stand_in_residents():
    return [_resident(TAG_BUSY, 0, 11000, state=ResidentState.ACTIVE), _resident(TAG_IDLE, 1, 13000)]


async def test_a_stand_in_slot_the_decision_finds_no_card_for_is_routed_to_the_plain_arm(tmp_path, monkeypatch):
    """The model fits no card even with the idle model gone, so the decision chooses nothing and the
    plain make-room arm unloads the idle model of the other card, as before."""
    mgr, _p, calls = _route_world(
        tmp_path, monkeypatch, free=[13000, 11000], rules=[],
        models=_stand_in_models(claim_mib=TOO_BIG_MIB, claim_auto=True), residents=_stand_in_residents())
    slot = _stand_in_slot()
    await _route_stand_in(mgr, slot)
    assert calls["evicted"] == [TAG_IDLE], calls
    assert calls["deferred"] == [(TAG_CLAIM, True)] and calls["reserved"] == [], calls
    assert not hasattr(slot, "fastlane_reclaim_first") and slot.fastlane_relocated_main_gpu is None


async def test_a_stand_in_slot_the_decision_chooses_a_card_for_is_unloaded_for_and_served_there(tmp_path, monkeypatch):
    """The model fits card 1 once its idle model is gone. First pass: the idle model is unloaded and the
    claimant waits, marked for card 1. After the release shows in the reading, the next pass places it."""
    def released(mgr, probe, r):
        mgr._residents.pop(r.resident_key, None)
        probe["free"] = [13000, 24000]

    mgr, probe, calls = _route_world(
        tmp_path, monkeypatch, free=[13000, 11000], rules=[], on_evict=released,
        models=_stand_in_models(claim_mib=22000, claim_auto=True), residents=_stand_in_residents())
    slot = _stand_in_slot()
    assert mgr._auto_pick_gpu(22000) == (0, "layer")
    await _route_stand_in(mgr, slot)
    assert calls["evicted"] == [TAG_IDLE], calls
    assert calls["deferred"] == [(TAG_CLAIM, True)] and calls["reserved"] == [], calls
    assert slot.fastlane_relocated_main_gpu == 1 and slot.fastlane_reclaim_first is True, vars(slot)
    assert mgr._auto_pick_gpu(22000) == (1, "none")
    await _route_stand_in(mgr, slot)
    assert [c[0] for c in calls["reserved"]] == [TAG_CLAIM], calls
    assert calls["evicted"] == [TAG_IDLE], "a second unload on the second pass"


async def test_a_stand_in_slot_with_a_pinned_manifest_reaches_the_make_room_arm_with_the_mark_unset(
        tmp_path, monkeypatch):
    """A pin with the lane off is outside the decision. The make-room arm is reached with the mark
    never set: the on-card victim is unloaded, not the one on the other card."""
    models = _stand_in_models(claim_mib=22000, claim_auto=False)
    models[TAG_CLAIM]["main_gpu"] = 1
    mgr, _p, calls = _route_world(
        tmp_path, monkeypatch, free=[13000, 11000], rules=[], models=models, residents=_stand_in_residents())
    slot = _stand_in_slot()
    await _route_stand_in(mgr, slot)
    assert calls["evicted"] == [TAG_IDLE], calls
    assert calls["deferred"] == [(TAG_CLAIM, True)], calls
    assert not hasattr(slot, "fastlane_reclaim_first") and slot.fastlane_relocated_main_gpu is None


async def test_a_stand_in_slot_that_carries_an_override_from_elsewhere_is_left_alone(tmp_path, monkeypatch):
    """The override was set by something other than the decision, so there is no mark. The decision
    must neither read a missing mark as set nor replace the override."""
    mgr, _p, calls = _route_world(
        tmp_path, monkeypatch, free=[13000, 11000], rules=[],
        models=_stand_in_models(claim_mib=22000, claim_auto=True), residents=_stand_in_residents())
    slot = _stand_in_slot(fastlane_relocated_main_gpu=0, fastlane_relocated_split_mode="none")
    await _route_stand_in(mgr, slot)
    assert slot.fastlane_relocated_main_gpu == 0 and slot.fastlane_relocated_split_mode == "none", vars(slot)
    assert not hasattr(slot, "fastlane_reclaim_first"), "the claimant was marked"
    assert calls["reserved"] == [] and calls["deferred"] == [(TAG_CLAIM, True)], calls


# ================================================================================================
# The probe of free VRAM is not read when there is nothing to reclaim or to wait for
# ================================================================================================
def _probe_counter(monkeypatch, free):
    count = {"manager": 0, "safety": 0}

    def manager_read():
        count["manager"] += 1
        return list(free)

    def safety_read():
        count["safety"] += 1
        return list(free)

    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", manager_read)
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", safety_read)
    return count


@pytest.mark.parametrize("case,probe_read", [
    ("no_resident_at_all", False),
    ("only_a_layer_split_idle_resident", False),
    ("only_a_busy_resident", False),
    ("only_a_zero_credit_pending", False),
    ("a_one_card_idle_resident", True),
    ("only_a_credit_pending_no_candidate", True),
])
async def test_the_free_vram_probe_is_read_only_when_there_is_something_to_reclaim_or_wait_for(
        tmp_path, monkeypatch, case, probe_read):
    mgr, _p = _bare_manager(tmp_path, monkeypatch, [13000, 11000])
    count = _probe_counter(monkeypatch, [13000, 11000])
    if case == "only_a_layer_split_idle_resident":
        mgr._residents["lay"] = _resident("lay", 0, 11000, split="layer")
    elif case == "only_a_busy_resident":
        mgr._residents[TAG_BUSY] = _resident(TAG_BUSY, 0, 11000, state=ResidentState.ACTIVE)
    elif case == "only_a_zero_credit_pending":
        mgr._pending_reclaim_mib[1] = 0
    elif case == "a_one_card_idle_resident":
        mgr._residents[TAG_IDLE] = _resident(TAG_IDLE, 1, 13000)
    elif case == "only_a_credit_pending_no_candidate":
        mgr._pending_reclaim_mib[1] = 500
    async with mgr._registry_lock:
        mgr._reclaim_first_card_locked(_claim_slot(), 22000, 1)
    reads = count["manager"] + count["safety"]
    if probe_read:
        assert reads >= 1, f"[{case}] the probe was never read although the decision had something to look at"
    else:
        assert reads == 0, f"[{case}] the probe was read {reads} time(s) with nothing to reclaim or wait for"


# ================================================================================================
# A claimant on the decided card is not relocatable: another card's credit must not hide the unload
# ================================================================================================
@pytest.mark.parametrize("auto_place", [True, False], ids=["auto_place_manifest", "pinned_manifest_control"])
async def test_another_cards_credit_pending_does_not_hide_the_unload_on_the_decided_card(
        tmp_path, monkeypatch, auto_place):
    """Floor 700, need 14000. Card 0 reads 4000 free and holds an idle model of 12000; card 1 reads 14500
    free with 100 on their way back from an unload in flight. The decision picks card 0 (card 1 keeps
    only 13900 with its floor). The idle model on card 0 must be unloaded on the FIRST pass: the credit
    on card 1 must not make the recheck think the claimant already fits."""
    models = {
        TAG_IDLE: dict(size_mib=1000, expected_mib=12000, main_gpu=0),
        TAG_CLAIM: dict(size_mib=1000, expected_mib=14000, main_gpu=0, auto_place=auto_place),
    }
    idle0 = _resident(TAG_IDLE, 0, 12000)

    def released(mgr, probe, r):
        mgr._residents.pop(r.resident_key, None)
        probe["free"] = [16000, 14500]

    mgr, probe, calls = _route_world(
        tmp_path, monkeypatch, free=[4000, 14500], rules=RULES, models=models, residents=[idle0], on_evict=released)
    mgr.runtime.queue.safety_min_free_vram_mib = 700
    owner = asyncio.create_task(asyncio.Event().wait())
    mgr._pending_reclaim_mib[1] = 100
    mgr._register_card_release_task(1, owner)
    try:
        need = mgr._read_model_footprint(TAG_CLAIM)[0]
        assert need == 14000
        async with mgr._registry_lock:
            choice = mgr._reclaim_first_card_locked(_claim_slot(), need, 1)
        assert choice is not None and tuple(choice) == (0, 12000, 0, 10700), choice
        slot = _claim_slot()
        await mgr._route_or_reserve(slot)
        assert calls["evicted"] == [TAG_IDLE], (
            f"the idle model on the decided card was not unloaded on the first pass: {calls}")
        assert calls["deferred"] == [(TAG_CLAIM, True)] and calls["reserved"] == [], calls
        assert slot.fastlane_relocated_main_gpu == 0 and slot.fastlane_reclaim_first is True
        recheck = [c for c in calls["gate"] if c[1].get("credit_pending") is True]
        assert recheck and recheck[-1][1].get("relocatable") is False, (
            f"the recheck on the decided card must not be relocatable: {recheck}")
        assert mgr._auto_pick_gpu(need)[0] == 0
        await mgr._route_or_reserve(slot)
        assert [c[0] for c in calls["reserved"]] == [TAG_CLAIM], calls
    finally:
        owner.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(owner, 2)
        mgr._pending_reclaim_mib[1] = 0


# ================================================================================================
# The mark on a slot: a real slot always carries it, and every tolerant read uses its default
# ================================================================================================
def _slot_field_default():
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(Slot)}
    assert "fastlane_reclaim_first" in fields, "the slot class declares no fastlane_reclaim_first field"
    return fields["fastlane_reclaim_first"].default


async def test_a_real_slot_carries_the_reclaim_first_field_with_its_default():
    import dataclasses
    slot = Slot.new(TAG_CLAIM)
    assert hasattr(slot, "fastlane_reclaim_first"), "a real slot has no fastlane_reclaim_first attribute"
    assert slot.fastlane_reclaim_first is False, f"a new slot carries {slot.fastlane_reclaim_first!r}, not False"
    declared = {f.name: f for f in dataclasses.fields(Slot)}
    assert "fastlane_reclaim_first" in declared, "fastlane_reclaim_first is not a declared field of Slot"
    assert declared["fastlane_reclaim_first"].default is False, (
        f"the declared default is {declared['fastlane_reclaim_first'].default!r}, not False")


async def test_every_getattr_default_for_the_reclaim_first_field_equals_the_slot_field_default():
    """Reads the source of the manager module that is actually imported (so it follows the tree under
    test). There are exactly eight tolerant reads of the mark and each must default to the field's own
    default."""
    import ast
    import gc
    import turbohaul.manager as manager_mod
    default = _slot_field_default()
    with open(manager_mod.__file__) as f:
        source = f.read()
    # Tasks left over from earlier tests can be finalized by the garbage collector in the middle of
    # ast.parse, which some Python 3.11 builds report as an AST recursion-depth SystemError. Collect
    # first and keep the collector off while parsing.
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        tree = ast.parse(source)
    finally:
        if was_enabled:
            gc.enable()
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
             and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "fastlane_reclaim_first"]
    assert len(reads) == 8, f"expected exactly 8 getattr reads of fastlane_reclaim_first, found {len(reads)} (lines {[n.lineno for n in reads]})"
    for n in reads:
        assert len(n.args) == 3 and isinstance(n.args[2], ast.Constant), f"line {n.lineno}: the default is not a constant"
        assert n.args[2].value is default or n.args[2].value == default and type(n.args[2].value) is type(default), (
            f"line {n.lineno}: getattr default {n.args[2].value!r} differs from the Slot field default {default!r}")


async def _twin_run(tmp_path, monkeypatch, sub, *, kind, scenario):
    d = tmp_path / sub
    d.mkdir()
    if scenario == "decided":
        def released(mgr, probe, r):
            mgr._residents.pop(r.resident_key, None)
            probe["free"] = [13000, 24000]
        mgr, _p, calls = _route_world(
            d, monkeypatch, free=[13000, 11000], rules=[], on_evict=released,
            models=_stand_in_models(claim_mib=22000, claim_auto=True), residents=_stand_in_residents())
        passes = 2
    else:
        mgr, _p, calls = _route_world(
            d, monkeypatch, free=[13000, 11000], rules=[],
            models=_stand_in_models(claim_mib=TOO_BIG_MIB, claim_auto=True), residents=_stand_in_residents())
        passes = 1
    slot = Slot.new(TAG_CLAIM) if kind == "real" else _stand_in_slot()
    for _ in range(passes):
        await _route_stand_in(mgr, slot)
    return (list(calls["evicted"]), list(calls["deferred"]), list(calls["reserved"]),
            slot.fastlane_relocated_main_gpu, slot.fastlane_relocated_split_mode,
            getattr(slot, "fastlane_reclaim_first", False))


@pytest.mark.parametrize("scenario", ["decided", "no_decision"])
async def test_a_stand_in_slot_behaves_exactly_like_a_real_slot_with_the_default(tmp_path, monkeypatch, scenario):
    real = await _twin_run(tmp_path, monkeypatch, "real", kind="real", scenario=scenario)
    stand_in = await _twin_run(tmp_path, monkeypatch, "stand_in", kind="stand_in", scenario=scenario)
    assert real == stand_in, (
        f"[{scenario}] (evicted, deferred, reserved, override card, override split, mark): "
        f"real slot {real} vs stand-in {stand_in}")
    assert real[0] == [TAG_IDLE], real


# ================================================================================================
# The choice between usable cards: the least protected client to unload comes before everything else
# ================================================================================================
def _ranked_manager(tmp_path, monkeypatch, free, residents, pending=None):
    """A bare manager that carries the three-rule Fast Lane table (first, second and third listed
    client), with the given idle residents registered and `pending` ({card: MiB}) on their way back."""
    mgr, _p = _bare_manager(tmp_path, monkeypatch, free, rules=RULES3)
    for r in residents:
        mgr._residents[r.resident_key] = r
    for card, mib in (pending or {}).items():
        mgr._pending_reclaim_mib[card] = mib
    return mgr


def _assert_card_chosen(choice, want, why):
    assert choice is not None, f"both cards are usable but no card was chosen; wanted card {want}: {why}"
    assert choice.card == want, f"card {choice.card} was chosen, wanted card {want}: {why}; choice={tuple(choice)}"


@pytest.mark.parametrize("low_card", [0, 1], ids=["lower_ranked_client_on_card_0", "lower_ranked_client_on_card_1"])
async def test_the_card_whose_idle_model_belongs_to_the_lower_ranked_client_wins_over_the_card_that_needs_fewer_mib(
        tmp_path, monkeypatch, low_card):
    """Need 15000, floor 1000. The high card reads 14000 free and holds a 3000 idle model of the
    first-listed client: 2000 MiB still to unload, headroom 1000. The low card reads 6000 free and
    holds a 17000 idle model of the third-listed client: 10000 still to unload, headroom 7000. Both are
    usable. The model of the third-listed client is unloaded before the first-listed client's, so the
    low card wins although it needs far more unloading; the card index plays no part (the low card is
    card 0 in one case and card 1 in the other)."""
    high_card = 1 - low_card
    free = [0, 0]
    free[high_card], free[low_card] = 14000, 6000
    mgr = _ranked_manager(tmp_path, monkeypatch, free, [
        _resident("idle-high", high_card, 3000, meta=META_OWNER),
        _resident("idle-low", low_card, 17000, meta=META_THIRD),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, low_card, (
        f"card {high_card} (first-listed client's 3000 MiB model, 2000 still to unload) against card {low_card} "
        f"(third-listed client's 17000 MiB model, 10000 still to unload): the less protected client goes first"))
    assert tuple(choice) == (low_card, 17000, 0, 10000), tuple(choice)


def _rank_real_models(a_card, auto_place):
    b_card = 1 - a_card
    return {
        TAG_IDLE: dict(size_mib=1000, expected_mib=3000, main_gpu=a_card),
        TAG_IDLE2: dict(size_mib=1000, expected_mib=17000, main_gpu=b_card),
        TAG_CLAIM: dict(size_mib=1000, expected_mib=15000, main_gpu=0, auto_place=auto_place),
    }


async def _rank_real_scene(b, a_card, a_meta, b_meta):
    """Card A holds a small idle model (3000, client `a_meta`) and reads 14000 free; card B holds a
    large one (17000, client `b_meta`) and reads 6000 free. The claimant needs 15000: it fits no card as
    things stand, the cards together take it, and each card takes it once its idle model is gone."""
    b_card = 1 - a_card
    small = await serve_and_park(b, "IA", TAG_IDLE, a_card, meta=a_meta)
    big = await serve_and_park(b, "IB", TAG_IDLE2, b_card, meta=b_meta, thread="th-a2")
    b.set_free(a_card, 14000)
    b.set_free(b_card, 6000)
    need_c = b.need(TAG_CLAIM)
    assert need_c == 15000 and small.reserved_need_mib == 3000 and big.reserved_need_mib == 17000
    assert b.free_list()[a_card] == 14000 and b.free_list()[b_card] == 6000
    assert b.mgr._auto_pick_gpu(need_c)[1] == "layer", "the placer should give the claimant a layer split"
    return small, big


async def _assert_served_after_only_this_unload(b, served_card, unloaded_tag, left_alone_tag):
    await wait_served(b, f"the claimant to be placed on card {served_card}")
    assert_served_on_card(b, served_card)
    assert [m for (_t, k, m) in b.events if k == "unload_decided"] == [unloaded_tag], (
        f"only {unloaded_tag} should be unloaded: {b.observed()}")
    assert untouched(b, left_alone_tag) == [], f"{left_alone_tag} was acted on: {b.observed()}"
    lines = reclaim_lines(b)
    assert len(lines) == 1 and re.search(rf"(^|\s)card={served_card}(\s|$)", lines[0]), lines


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("a_card", [0, 1], ids=["small_model_on_card_0", "small_model_on_card_1"])
async def test_the_lower_ranked_clients_idle_model_is_the_one_unloaded_even_when_the_other_card_needs_less_unloading(
        tmp_path, monkeypatch, a_card, auto_place):
    """Need 15000. Card A reads 14000 free (another process holds 7000) with a 3000 idle model of the
    first-listed client; card B reads 6000 free (another process holds 1000) with a 17000 idle model of
    the third-listed client. Card A needs 2000 MiB unloaded, card B 10000, but the third-listed client's
    model goes first: only it is unloaded and the claimant is served on card B. Card A is card 0 in one
    case and card 1 in the other."""
    b_card = 1 - a_card
    async with world(tmp_path, monkeypatch, models=_rank_real_models(a_card, auto_place), rules=RULES3) as b:
        await _rank_real_scene(b, a_card, META_OWNER, META_THIRD)
        submit_claimant(b)
        await _assert_served_after_only_this_unload(b, b_card, TAG_IDLE2, TAG_IDLE)


# ------------------------------------------------------------------------------------------------
# equal rank: today's order is kept (these pin behaviour that must not change)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a_card", [0, 1], ids=["fewer_to_unload_on_card_0", "fewer_to_unload_on_card_1"])
async def test_between_equally_ranked_cards_the_fewer_mib_to_unload_wins_over_the_larger_headroom(
        tmp_path, monkeypatch, a_card):
    """Control: passes on the old order too, which must not change. Need 15000, floor 1000. Both cards
    hold an idle model of the same (second-listed) client. Card A reads 14000 free and holds 3000: 2000
    still to unload, headroom 1000. Card B reads 6000 free and holds 17000: 10000 still to unload,
    headroom 7000. With the rank equal, the fewer MiB to unload decides: card A, whatever its index."""
    b_card = 1 - a_card
    free = [0, 0]
    free[a_card], free[b_card] = 14000, 6000
    mgr = _ranked_manager(tmp_path, monkeypatch, free, [
        _resident("idle-a", a_card, 3000, meta=META_SECOND),
        _resident("idle-b", b_card, 17000, meta=META_SECOND),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, a_card, "equal rank: 2000 still to unload beats 10000 although B keeps more headroom")
    assert tuple(choice) == (a_card, 3000, 0, 2000), tuple(choice)


async def test_between_equally_ranked_cards_with_the_same_mib_to_unload_the_larger_headroom_wins(
        tmp_path, monkeypatch):
    """Control: passes on the old order too, which must not change. Need 15000, floor 1000. Both cards
    read 12000 free (4000 still to unload on each) and hold an idle model of the same (first-listed)
    client. Card 0's model is 5000: headroom 12000 + 5000 - 1000 - 15000 = 1000. Card 1's is 9000:
    headroom 5000. The rank and the MiB to unload are equal, so card 1 wins on headroom although card 0
    is the lower card."""
    mgr = _ranked_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident("idle-a", 0, 5000, meta=META_OWNER),
        _resident("idle-b", 1, 9000, meta=META_OWNER),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, 1, "equal rank, equal 4000 still to unload: the larger headroom (5000 against 1000) wins")
    assert tuple(choice) == (1, 9000, 0, 4000), tuple(choice)


async def test_between_equally_ranked_cards_that_are_equal_in_everything_the_lowest_card_wins(tmp_path, monkeypatch):
    """Control: passes on the old order too, which must not change. Need 15000, floor 1000. Both cards
    read 12000 free and hold a 6000 idle model of the same (third-listed) client: 4000 still to unload and
    headroom 2000 on each. The lowest card index decides: card 0."""
    mgr = _ranked_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident("idle-a", 0, 6000, meta=META_THIRD),
        _resident("idle-b", 1, 6000, meta=META_THIRD),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, 0, "everything equal: the lowest card index decides")
    assert tuple(choice) == (0, 6000, 0, 4000), tuple(choice)


# in each case the card that must win has the MORE recently active resident, so a choice that
# followed the victim order's recency term would pick the other card
RECENCY_CASES = {
    "index_decides": dict(free=[12000, 12000], sizes=[6000, 6000], recent=0, want=(0, 6000, 0, 4000)),
    "fewer_mib_decides": dict(free=[6000, 14000], sizes=[17000, 3000], recent=1, want=(1, 3000, 0, 2000)),
    "headroom_decides": dict(free=[12000, 12000], sizes=[5000, 9000], recent=1, want=(1, 9000, 0, 4000)),
}


@pytest.mark.parametrize("case", list(RECENCY_CASES), ids=list(RECENCY_CASES))
async def test_between_equally_ranked_cards_how_recently_the_model_was_used_does_not_decide_the_card(
        tmp_path, monkeypatch, case):
    """Control: passes on the old order too, which must not change. Need 15000, floor 1000, an idle model
    of the same (second-listed) client on each card, and `recent` names the card whose model was used
    more recently (last_active 100.0 against 1.0). index_decides: both read 12000 free, both hold 6000,
    card 0 is the recent one and wins on the card index. fewer_mib_decides: card 0 reads 6000 free
    and holds 17000 (10000 to unload), card 1 reads 14000 and holds 3000 (2000 to unload); card 1 is
    the recent one and wins on the fewer MiB. headroom_decides: both read 12000 free (4000 to unload),
    card 0 holds 5000 and card 1 holds 9000; card 1 is the recent one and wins on the headroom. The
    less recently active model is the one the victim order would unload first, yet it must not draw
    the choice."""
    spec = RECENCY_CASES[case]
    residents = [
        _resident(f"idle-{card}", card, spec["sizes"][card], meta=META_SECOND,
                  last_active=100.0 if card == spec["recent"] else 1.0)
        for card in (0, 1)
    ]
    mgr = _ranked_manager(tmp_path, monkeypatch, spec["free"], residents)
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, spec["want"][0], f"[{case}] equal rank: recency (card {spec['recent']} is the recent one) must not decide")
    assert tuple(choice) == spec["want"], tuple(choice)


# ------------------------------------------------------------------------------------------------
# unlisted before listed
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("unlisted_card", [0, 1], ids=["unlisted_client_on_card_0", "unlisted_client_on_card_1"])
async def test_the_card_whose_idle_model_belongs_to_an_unlisted_client_wins_over_a_listed_clients_card(
        tmp_path, monkeypatch, unlisted_card):
    """Need 15000, floor 1000. The listed card reads 14000 free and holds a 3000 idle model of the
    lowest-ranked listed client (third): 2000 still to unload. The unlisted card reads 6000 free and
    holds a 17000 idle model of a client in no rule: 10000 still to unload. A client in no rule is
    unloaded before any listed client, even the lowest-ranked one: the unlisted card wins."""
    listed_card = 1 - unlisted_card
    free = [0, 0]
    free[listed_card], free[unlisted_card] = 14000, 6000
    mgr = _ranked_manager(tmp_path, monkeypatch, free, [
        _resident("idle-listed", listed_card, 3000, meta=META_THIRD),
        _resident("idle-unlisted", unlisted_card, 17000, meta=META_UNLISTED),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, unlisted_card, (
        f"card {listed_card} (listed client, 2000 still to unload) against card {unlisted_card} "
        f"(unlisted client, 10000 still to unload): the unlisted client goes first"))
    assert tuple(choice) == (unlisted_card, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("a_card", [0, 1], ids=["listed_client_on_card_0", "listed_client_on_card_1"])
async def test_the_unlisted_clients_idle_model_is_the_one_unloaded_on_the_real_path(
        tmp_path, monkeypatch, a_card, auto_place):
    """Need 15000. Card A reads 14000 free with a 3000 idle model of the third-listed client; card B
    reads 6000 free with a 17000 idle model of a client in no rule. Card A needs 2000 MiB unloaded,
    card B 10000, but the unlisted client's model goes first: only it is unloaded and the claimant is
    served on card B."""
    b_card = 1 - a_card
    async with world(tmp_path, monkeypatch, models=_rank_real_models(a_card, auto_place), rules=RULES3) as b:
        await _rank_real_scene(b, a_card, META_THIRD, META_UNLISTED)
        submit_claimant(b)
        await _assert_served_after_only_this_unload(b, b_card, TAG_IDLE2, TAG_IDLE)


# ------------------------------------------------------------------------------------------------
# a card that unloads nobody is the best
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("x_card", [0, 1], ids=["zero_unload_card_0", "zero_unload_card_1"])
async def test_a_card_that_fits_through_an_unload_in_flight_wins_over_a_card_that_must_unload_an_unlisted_model(
        tmp_path, monkeypatch, x_card):
    """Control: passes on the old order too (it already puts the card needing 0 MiB first), so it cannot
    fail on the old code. Need 15000, floor 1000. Card X reads 6000 free with 12000 on their way back
    from an unload in flight and an unlisted client's 3000 idle model that need not be touched: 0 MiB
    to unload. Card Y reads 6000 free and holds an unlisted client's 17000 idle model: 10000 still to
    unload. Card X wins."""
    y_card = 1 - x_card
    free = [6000, 6000]
    mgr = _ranked_manager(tmp_path, monkeypatch, free, [
        _resident("idle-x", x_card, 3000, meta=META_UNLISTED),
        _resident("idle-y", y_card, 17000, meta=META_UNLISTED),
    ], pending={x_card: 12000})
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, x_card, "the card that unloads nobody beats a card that has to unload a model")
    assert tuple(choice) == (x_card, 3000, 12000, 0), tuple(choice)


@pytest.mark.parametrize("how", ["fits_already", "unload_in_flight"])
@pytest.mark.parametrize("x_card", [0, 1], ids=["zero_unload_card_0", "zero_unload_card_1"])
async def test_a_card_that_unloads_nobody_wins_whatever_the_rank_of_the_idle_model_it_holds(
        tmp_path, monkeypatch, x_card, how):
    """Control on the old order (it prefers the card needing 0 MiB, so it passes there too); it goes
    RED only on a choice that ranks the idle models of a card that unloads nobody. Need 15000, floor
    1000. Card X holds a 3000 idle model of the FIRST-listed client (the most protected) that is not
    touched, and needs 0 MiB unloaded: fits_already means it reads 16000 free, unload_in_flight means
    it reads 6000 free with 12000 on their way back from an unload already running. Card Y reads 6000
    free and holds a 17000 idle model of a client in no rule: 10000 still to unload. Card X wins: it
    unloads nobody, so the high rank of its idle model is irrelevant."""
    y_card = 1 - x_card
    free = [6000, 6000]
    free[x_card] = 16000 if how == "fits_already" else 6000
    pending = {x_card: 12000} if how == "unload_in_flight" else None
    mgr = _ranked_manager(tmp_path, monkeypatch, free, [
        _resident("idle-x", x_card, 3000, meta=META_OWNER),
        _resident("idle-y", y_card, 17000, meta=META_UNLISTED),
    ], pending=pending)
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, x_card, f"[{how}] the card that unloads nobody beats a card that has to unload a model")
    assert tuple(choice) == (x_card, 3000, 12000 if pending else 0, 0), tuple(choice)


# ------------------------------------------------------------------------------------------------
# the rank of a card is the rank of the models it must unload, not of every idle model on it
# ------------------------------------------------------------------------------------------------
# (prefix card, size of its unlisted model, the card that must win): the first-listed client's 4000 model
# is only reached when the unlisted model alone does not cover the 4000 still to unload
PREFIX_CASES = {
    "unlisted_covers_exactly_card_0": (0, 4000, 0),
    "unlisted_covers_exactly_card_1": (1, 4000, 1),
    "unlisted_one_short_card_0": (0, 3999, 1),
    "unlisted_one_short_card_1": (1, 3999, 0),
}


@pytest.mark.parametrize("case", list(PREFIX_CASES), ids=list(PREFIX_CASES))
async def test_the_rank_of_a_card_is_taken_over_the_models_it_must_unload_and_not_over_every_idle_model_on_it(
        tmp_path, monkeypatch, case):
    """Need 15000, floor 1000; both cards read 12000 free: 4000 still to unload on each. Card Q holds one
    12000 idle model of the second-listed client (headroom 8000). Card P holds an unlisted client's idle
    model (4000 or 3999 MiB) and the first-listed client's 4000 MiB idle model. Models are unloaded
    least protected first: the unlisted one goes first. At 4000 it alone covers the 4000 still to unload,
    so P unloads only an unlisted model and wins although the most protected model on P is above Q's
    (and Q keeps more headroom). At 3999 it falls 1 MiB short, the first-listed client's model must go
    too, and Q wins. Only the 3999 cases are green on the old order (Q has the larger headroom)."""
    p_card, unlisted_mib, want = PREFIX_CASES[case]
    q_card = 1 - p_card
    mgr = _ranked_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident("idle-p-unlisted", p_card, unlisted_mib, meta=META_UNLISTED),
        _resident("idle-p-first", p_card, 4000, meta=META_OWNER),
        _resident("idle-q", q_card, 12000, meta=META_SECOND),
    ])
    choice = await _choose(mgr, 15000)
    _assert_card_chosen(choice, want, (
        f"[{case}] P=card {p_card} (unlisted {unlisted_mib} then first-listed 4000), "
        f"Q=card {q_card} (second-listed 12000), 4000 still to unload on each"))


# ------------------------------------------------------------------------------------------------
# the Fast Lane table is read on the event loop thread, not by the card choice
# ------------------------------------------------------------------------------------------------
async def _spied_card_choice_run(tmp_path, monkeypatch, auto_place):
    """Runs the real routing path of the real-path scenario above with spies that call the real
    functions: one on the card choice (the thread it runs on and the arguments it gets), one on the
    Fast Lane table (the thread each read happens on, and whether the card choice was running on
    that thread at the time)."""
    import threading
    loop_thread = threading.get_ident()
    choice_threads: set = set()
    choice_runs: list = []
    table_reads: list = []
    async with world(tmp_path, monkeypatch, models=_rank_real_models(0, auto_place), rules=RULES3) as b:
        mgr = b.mgr
        real_choice = mgr._reclaim_first_card_locked
        real_table = mgr._fastlane_table

        def choice_spy(*args, **kwargs):
            tid = threading.get_ident()
            choice_threads.add(tid)
            choice_runs.append((tid, args, kwargs))
            try:
                return real_choice(*args, **kwargs)
            finally:
                choice_threads.discard(tid)

        def table_spy(*args, **kwargs):
            tid = threading.get_ident()
            table_reads.append((tid, tid in choice_threads))
            return real_table(*args, **kwargs)

        monkeypatch.setattr(mgr, "_reclaim_first_card_locked", choice_spy)
        monkeypatch.setattr(mgr, "_fastlane_table", table_spy)
        await _rank_real_scene(b, 0, META_OWNER, META_THIRD)
        submit_claimant(b)
        await wait_served(b, "the claimant to be placed on card 1")
    return loop_thread, choice_runs, table_reads


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_the_fast_lane_table_is_not_read_by_the_card_choice_on_a_worker_thread(
        tmp_path, monkeypatch, auto_place):
    """Same scenario as the real-path test of the lower-ranked client's model (need 15000, card 0 reads
    14000 free with a 3000 model of the first-listed client, card 1 reads 6000 free with a 17000 model
    of the third-listed client). The card choice runs off the event loop; the table it ranks by must
    not be read from inside it, on the worker thread. The spies call the real functions. The card choice
    must have run (at least once, on a thread other than the loop's), so the test cannot pass without
    it. Passes on the old order, which never reads the table in the card choice."""
    loop_thread, runs, reads = await _spied_card_choice_run(tmp_path, monkeypatch, auto_place)
    assert len(runs) >= 1, "the card choice never ran: the test observed nothing"
    assert all(tid != loop_thread for tid, _a, _k in runs), (
        f"the card choice ran on the event loop thread: {[tid for tid, _a, _k in runs]} (loop {loop_thread})")
    assert reads, "the Fast Lane table was never read: the spy observed nothing"
    inside = [tid for tid, in_choice in reads if in_choice]
    assert inside == [], f"the Fast Lane table was read {len(inside)} time(s) from inside the card choice, on worker thread(s) {inside}"


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
async def test_the_card_choice_is_handed_the_table_that_the_event_loop_took(tmp_path, monkeypatch, auto_place):
    """The same run: every call of the card choice gets, as its fourth argument, the Fast Lane table
    (a list, the one read by the event loop thread just before) so it has nothing to read itself."""
    loop_thread, runs, reads = await _spied_card_choice_run(tmp_path, monkeypatch, auto_place)
    assert len(runs) >= 1, "the card choice never ran: the test observed nothing"
    for tid, args, kwargs in runs:
        got = args[3] if len(args) >= 4 else kwargs.get("table")
        assert isinstance(got, list) and len(got) == 3, (
            f"the card choice was called without the three-rule table: args={args!r} kwargs={kwargs!r}")
    assert any(tid == loop_thread for tid, _in in reads), "the table was never read on the event loop thread"


# ================================================================================================
# The count-cap arm for a claimant the decision pinned to a card
#
# Two idle residents sit on different cards and the model count is at its cap. The claimant is
# pinned by the decision to the card it fits on once that card's idle model is unloaded, but the
# card reads only a little more free than the claimant needs (free >= need, free < need + floor),
# so the claimant fits the memory gate (which has no floor) while the card would be left below the
# floor. The idle model to unload must then be the one ON the decided card, the claimant must be
# queued until that card has really been freed, and the claimant must start with the decided card
# at or above the floor.
# ================================================================================================
_cc_card_g = 1              # the card the decision picks for the claimant
_cc_card_o = 0              # the other card
_cc_idle_g_mib = 5000       # the idle model on the decided card: its unload frees far more than the floor shortfall
_cc_idle_o_mib = 6000       # the idle model on the other card
_cc_claim_mib = 15000
_cc_free_o = 3000           # the other card: keeps the floor after the residue, the claimant never fits there
_cc_served_s = 14.0
_cc_needed_to_free_mib = 500   # the memory still to free that the decision is made to report


def _cc_models(*, claim_pin=_cc_card_o):
    return {
        TAG_IDLE: dict(size_mib=_cc_idle_g_mib, main_gpu=_cc_card_g),
        TAG_IDLE2: dict(size_mib=_cc_idle_o_mib, main_gpu=_cc_card_o),
        TAG_CLAIM: dict(size_mib=_cc_claim_mib, main_gpu=claim_pin),
    }


class _CcRecorder:
    """What the claimant's passes through the routing step did, plus what its engine start saw."""

    def __init__(self):
        self.n = 0               # passes of the claimant through the routing step so far
        self.passes = []         # one dict per finished pass
        self.reserves = []       # (pass number, model tag) of every in-band reserve call
        self.loads = []          # the claimant's engine starts: free MiB at that moment, per card

    def pass_reserved(self, n):
        return any(p == n and t == TAG_CLAIM for (p, t) in self.reserves)


def _cc_instrument(b, decided_card):
    """Observe only: record every claimant pass, every in-band reserve, and the free MiB the
    claimant's engine start sees (the stamp is made before the fake engine's own MiB are counted)."""
    rec = _CcRecorder()
    mgr = b.mgr

    orig_stamp = b.stamp

    def stamp2(kind, name):
        if kind == "load" and name == TAG_CLAIM:
            free = b.free_list()
            rec.loads.append(dict(free=free, decided_after=free[decided_card] - b.need(TAG_CLAIM)))
        return orig_stamp(kind, name)

    b.stamp = stamp2

    orig_reserve = mgr._reserve_and_start_locked

    async def reserve_spy(slot, *a, **kw):
        rec.reserves.append((rec.n, slot.model_tag))
        return await orig_reserve(slot, *a, **kw)

    mgr._reserve_and_start_locked = reserve_spy

    orig_route = mgr._route_or_reserve

    async def pass_spy(slot):
        if slot.prompt != "C1":
            return await orig_route(slot)
        rec.n += 1
        n = rec.n
        ev0, lg0 = len(b.events), len(b.logs)
        out = await orig_route(slot)
        rec.passes.append(dict(
            n=n,
            pin=getattr(slot, "fastlane_reclaim_first", False),
            reloc=slot.fastlane_relocated_main_gpu,
            unloaded=[nm for (_t, k, nm) in b.events[ev0:] if k == "unload_decided"],
            logs=[m for m in b.logs[lg0:] if m.startswith(("MAKE_ROOM", "reclaim_first"))],
            reserved=rec.pass_reserved(n),
        ))
        return out

    mgr._route_or_reserve = pass_spy
    return rec


def _cc_unloaded(b):
    return [n for (_t, k, n) in b.events if k == "unload_decided"]


def _cc_ctx(b, rec):
    return (f"claimant passes={rec.n}; per pass (n, pin, unloaded, reserved in-band)="
            f"{[(p['n'], p['pin'], p['unloaded'], p['reserved']) for p in rec.passes]}; "
            f"observed: {b.observed()}")


def _cc_eviction_lines(b, reason):
    return [m for m in b.logs if m.startswith("MAKE_ROOM_EVICTION ") and f"reason={reason} " in m]


def _cc_starved_lines(b, reason):
    return [m for m in b.logs if m.startswith("MAKE_ROOM_STARVED ") and f"reason={reason} " in m]


async def _cc_build(b, free_g_offset):
    """Two idle residents (the unlisted one on the other card parked FIRST, so the older, and the
    worse-ranked one: a search that ignores cards picks it) and the decided card reading
    need + free_g_offset free. Returns (idle model on the decided card, idle model on the other card, need)."""
    b.mgr._vram_total_mib = [CARD_MIB] * 2
    ib = await serve_and_park(b, "IB", TAG_IDLE2, _cc_card_o, meta=META_UNLISTED, thread="th-b2")
    ia = await serve_and_park(b, "IA", TAG_IDLE, _cc_card_g, meta=META_OWNER, thread="th-a")
    need = b.need(TAG_CLAIM)
    b.set_free(_cc_card_g, need + free_g_offset)
    b.set_free(_cc_card_o, _cc_free_o)
    assert ib.last_active_monotonic < ia.last_active_monotonic, "the other card's model is not the older one"
    glob = b.mgr._lru_idle_unloadable()
    oncard = b.mgr._lru_idle_unloadable(main_gpu=_cc_card_g, split_mode="none")
    assert glob is ib and oncard is ia, (
        f"scenario: the global victim must be {TAG_IDLE2} and the decided card's {TAG_IDLE}; "
        f"got global={getattr(glob, 'model_tag', None)} on_card={getattr(oncard, 'model_tag', None)}")
    assert len(list(b.mgr._model_residents())) >= b.mgr.runtime.queue.max_parallel_sidecars, "not at the count cap"
    assert _cc_free_o - CUDA_RESIDUE_MIB >= FLOOR_MIB, "the other card cannot pass the residue check"
    return ia, ib, need


async def _cc_wait_served(b, bound=_cc_served_s):
    """True when the claimant was served, False when it was not within the bound (the assertions that
    follow then say what happened instead of this wait)."""
    try:
        await b.until(lambda: b.seen("turn_end", "C1") and b.seen("caller_got_response", "C1"),
                      "the claimant to be served", bound)
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize("k", [1, 10, 500, 999, 0], ids=lambda k: f"free_is_need_plus_{k}")
async def test_in_the_band_the_decided_cards_idle_model_is_unloaded_and_the_other_cards_is_left_alone(
        tmp_path, monkeypatch, k):
    """At the model-count cap, the decided card reads need + k free (below need + floor, so the
    memory gate admits the claimant but the card would end below the floor). The claimant carries the
    reclaim-first pin, so the unloaded model is the idle one on the decided card, it is logged as the
    count-cap eviction, the claimant is queued on that pass (no in-band reserve), and when its engine
    starts the decided card keeps the floor free after the claimant's own footprint."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=2) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _cc_build(b, k)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB, (
            f"scenario: decided card free {b.free_list()[_cc_card_g]}, need {need}, floor {FLOOR_MIB}")

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        assert unloaded == [TAG_IDLE], (
            f"models unloaded: {unloaded}; wanted only the decided card's idle model {TAG_IDLE} "
            f"(the other card's {TAG_IDLE2} must stay); {_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE2) == [], (
            f"the other card's {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert resident_for(b.mgr, TAG_IDLE2) is ib and ib.state is ResidentState.IDLE_EVICTABLE
        assert rec.passes, f"the claimant was never routed; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        first = rec.passes[0]
        assert b.count("log", "reclaim") >= 1 and first["pin"] is True and first["reloc"] == _cc_card_g, (
            f"the claimant was not pinned to card {_cc_card_g} on its first pass: pin={first['pin']} "
            f"card={first['reloc']}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        lines = _cc_eviction_lines(b, "make_room_count_cap")
        assert len(lines) == 1 and f"model_tag={TAG_IDLE} " in lines[0], (
            f"count-cap eviction lines {lines}; wanted exactly one, naming {TAG_IDLE}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert first["unloaded"] == [TAG_IDLE] and first["reserved"] is False, (
            f"on its first pass the claimant must unload {TAG_IDLE} and be queued without an in-band "
            f"reserve; unloaded on that pass {first['unloaded']}, reserved in-band={first['reserved']}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        assert_served_on_card(b, _cc_card_g)
        assert len(rec.loads) == 1, f"claimant engine starts: {rec.loads}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        seen_free = rec.loads[0]["decided_after"]
        assert seen_free >= FLOOR_MIB, (
            f"the claimant's engine start left the decided card with {seen_free} MiB free after its own "
            f"footprint {need} (floor {FLOOR_MIB}; card free at start {rec.loads[0]['free']}); "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")


async def test_without_the_pin_the_count_cap_arm_still_takes_the_global_victim_and_reserves_in_band(
        tmp_path, monkeypatch):
    """CONTROL: behaviour that does not change. The same two idle models, the same count cap and a
    band free figure on the card the claimant's manifest pins, but the claimant carries no reclaim-first
    pin (the lane is off and the manifest pin is kept, so the decision does not apply). The count-cap
    arm takes the global victim (the older model on the other card) and reserves in-band on the same pass."""
    async with world(tmp_path, monkeypatch, models=_cc_models(claim_pin=_cc_card_g), rules=[], n_cards=2,
                     max_parallel=2) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _cc_build(b, 500)

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        slot = b.slots["C1"]
        assert getattr(slot, "fastlane_reclaim_first", False) is False and not b.seen("log", "reclaim"), (
            f"the claimant carries the pin although the lane is off; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE2], (
            f"models unloaded: {unloaded}; wanted only the global victim {TAG_IDLE2} (the older model on the "
            f"other card); {_cc_ctx(b, rec)}")
        lines = _cc_eviction_lines(b, "make_room_count_cap")
        assert len(lines) == 1 and f"model_tag={TAG_IDLE2} " in lines[0], (
            f"count-cap eviction lines {lines}; wanted exactly one, naming {TAG_IDLE2}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE) == [], (
            f"the decided card's {TAG_IDLE} was acted on: {untouched(b, TAG_IDLE)}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert rec.passes and rec.passes[0]["reserved"] is True and rec.passes[0]["unloaded"] == [TAG_IDLE2], (
            f"the claimant must be reserved in-band on its first pass; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"


async def test_with_the_pin_and_no_idle_model_on_the_decided_card_nothing_is_unloaded_on_the_pass(
        tmp_path, monkeypatch):
    """At the model-count cap the claimant is pinned to a card that holds no model at all; that card
    is usable only through VRAM already on its way back from an unload in flight (the credit is held
    open by a task that does not finish, so no real teardown timing is involved). The other card
    holds a busy model and an idle one. The decision itself never reports memory still to be freed for
    a card that holds no idle model (such a card is only usable when its credit already covers the
    need), so the test makes this one pass report a positive figure: the real decision is called and
    only its figure of memory still to free is replaced (every pass is checked to have been changed).
    With memory still to free and no idle model on the decided card, nothing is unloaded on the pass:
    the starved line for the count-cap arm is logged, the idle model on the other card stays and the
    claimant is not started. (With a figure of zero, the credit covering the need, the arm frees a
    count slot instead; another test covers that.)"""
    models = {
        TAG_BUSY: dict(size_mib=BUSY_MIB, main_gpu=_cc_card_o),
        TAG_IDLE2: dict(size_mib=_cc_idle_o_mib, main_gpu=_cc_card_o),
        TAG_CLAIM: dict(size_mib=_cc_claim_mib, main_gpu=_cc_card_o),
    }
    credit_mib = 2000
    async with world(tmp_path, monkeypatch, models=models, rules=RULES, n_cards=2, max_parallel=2) as b:
        mgr = b.mgr
        mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        await start_busy(b, "BUSY", TAG_BUSY, _cc_card_o)
        q = await serve_and_park(b, "Q1", TAG_IDLE2, _cc_card_o, meta=META_UNLISTED, thread="th-q")
        need = b.need(TAG_CLAIM)
        b.set_free(_cc_card_g, need + 500)
        b.set_free(_cc_card_o, _cc_free_o)
        assert len(list(mgr._model_residents())) >= mgr.runtime.queue.max_parallel_sidecars, "not at the count cap"
        assert mgr._lru_idle_unloadable() is q, "scenario: the global victim must be the other card's idle model"
        assert mgr._lru_idle_unloadable(main_gpu=_cc_card_g, split_mode="none") is None, (
            "scenario: the decided card must hold no idle model")
        real_decision = mgr._reclaim_first_card_locked
        decisions = []

        def decision_with_memory_to_free(*a, **kw):
            choice = real_decision(*a, **kw)
            if choice is not None:
                choice = choice._replace(needed_mib=_cc_needed_to_free_mib)
            decisions.append(choice)
            return choice

        monkeypatch.setattr(mgr, "_reclaim_first_card_locked", decision_with_memory_to_free)
        owner = asyncio.create_task(asyncio.Event().wait())
        mgr._pending_reclaim_mib[_cc_card_g] = credit_mib
        mgr._register_card_release_task(_cc_card_g, owner)
        try:
            submit_claimant(b)
            await b.until(
                lambda: b.count("log", "starved") >= 2 or b.seen("unload_decided", TAG_IDLE2)
                or b.seen("load", TAG_CLAIM),
                "the claimant to be starved twice, or something to be unloaded, or the claimant to start", 12.0)
            await asyncio.sleep(0.3)

            unloaded = _cc_unloaded(b)
            assert decisions and all(
                c is not None and c.card == _cc_card_g and c.needed_mib == _cc_needed_to_free_mib
                for c in decisions), (
                f"scenario: the decision must pick card {_cc_card_g} with memory still to free on every pass; "
                f"decisions: {decisions}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert unloaded == [], (
                f"models unloaded: {unloaded}; wanted none (the decided card holds no idle model); "
                f"{_cc_ctx(b, rec)}")
            assert not _cc_eviction_lines(b, "make_room_count_cap"), (
                f"a count-cap eviction line was logged: {_cc_eviction_lines(b, 'make_room_count_cap')}; "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            starved = _cc_starved_lines(b, "make_room_starved_count_cap")
            assert starved, (
                f"no count-cap starved line was logged; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert rec.passes and all(p["pin"] is True and p["reloc"] == _cc_card_g for p in rec.passes), (
                f"the claimant was not pinned to card {_cc_card_g} on every pass; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            assert untouched(b, TAG_IDLE2) == [], (
                f"the idle model {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            assert resident_for(mgr, TAG_IDLE2) is q and q.state is ResidentState.IDLE_EVICTABLE
            claimant_reserves = [r for r in rec.reserves if r[1] == TAG_CLAIM]
            assert not b.seen("load", TAG_CLAIM) and not claimant_reserves, (
                f"the claimant was started or reserved; models unloaded: {unloaded}; reserves={claimant_reserves}; "
                f"{_cc_ctx(b, rec)}")
        finally:
            owner.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(owner, 2)
            mgr._pending_reclaim_mib[_cc_card_g] = 0


async def test_one_point_below_need_goes_to_the_vram_arm_not_the_count_cap_arm(tmp_path, monkeypatch):
    """CONTROL: behaviour that does not change. The decided card reads one MiB less free than the
    claimant needs, so the memory gate refuses and the make-room arm for memory runs, not the count-cap
    arm. The decided card's idle model is unloaded through it (logged as the memory arm), no count-cap
    line is logged at all, and the other card's idle model is left alone."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=2) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _cc_build(b, -1)
        assert b.free_list()[_cc_card_g] == need - 1

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        assert unloaded == [TAG_IDLE], (
            f"models unloaded: {unloaded}; wanted only the decided card's idle model {TAG_IDLE}; {_cc_ctx(b, rec)}")
        vram_lines = _cc_eviction_lines(b, "make_room_vram")
        assert len(vram_lines) == 1 and f"model_tag={TAG_IDLE} " in vram_lines[0], (
            f"memory-arm eviction lines {vram_lines}; wanted exactly one, naming {TAG_IDLE}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert not any("make_room_count_cap" in m for m in b.logs), (
            f"the count-cap arm was entered: {[m for m in b.logs if 'make_room_count_cap' in m]}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE2) == [], (
            f"the other card's {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert rec.passes and rec.passes[0]["pin"] is True and rec.passes[0]["reserved"] is False, (
            f"the claimant must be pinned and queued on its first pass; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"


# ================================================================================================
# A pinned claimant is not started while the decided card's unload is still in flight
# ================================================================================================
_HD_DECIDED = 1                  # the card the reclaim-first decision picks
_HD_OTHER = 0                    # the other card
_HD_ON_CARD_IDLE_MIB = 5000      # model-idle on the decided card: unloading it frees far more than the floor shortfall
_HD_OFF_CARD_IDLE_MIB = 6000     # model-idle-b on the other card (older, less protected owner)
_HD_CLAIM_MIB = 15000            # claimant footprint 15048 with the context allowance
_HD_OTHER_FREE_MIB = 3000        # the other card passes the residue check but never holds the claimant
_HD_HOLD_S = 3.0                 # wall clock the victim's engine stop stays held after the claimant is submitted


def _hd_models():
    return {
        TAG_IDLE: dict(size_mib=_HD_ON_CARD_IDLE_MIB, main_gpu=_HD_DECIDED),
        TAG_IDLE2: dict(size_mib=_HD_OFF_CARD_IDLE_MIB, main_gpu=_HD_OTHER),
        TAG_CLAIM: dict(size_mib=_HD_CLAIM_MIB, main_gpu=0),
    }


def _hd_unloaded(b):
    return [n for (_t, kind, n) in b.events if kind == "unload_decided"]


def _hd_describe(b, loads, t_submit, t_hold_end=None):
    """What the run did, in words: which models were unloaded, when the claimant's engine started."""
    if loads:
        stamps = ", ".join(
            f"t={ld['t'] - t_submit:.3f}s after submit (decided card read {ld['free']} free before its own "
            f"footprint, {ld['free'] - ld['need']} after)" for ld in loads)
        started = f"the claimant's engine was started: {stamps}"
    else:
        started = "the claimant's engine was not started"
    held = "" if t_hold_end is None else f"; the victim's engine stop was held until t={t_hold_end - t_submit:.3f}s"
    return (f"unloaded={_hd_unloaded(b)}; {started}{held}; engine alive now={sorted(b.alive)}; "
            f"observed: {b.observed()}")


@pytest.mark.parametrize("k", [1, 10, 500, 999], ids=lambda k: f"k{k}")
async def test_a_pinned_claimant_does_not_start_while_the_decided_cards_unload_is_still_in_flight(
        tmp_path, monkeypatch, k):
    """Two cards, room for two models. The claimant is decided for card 1, which reads free = need + k
    with k below the floor, and the unload of card 1's idle model is held in flight (its engine stop
    does not return). The registry already lost that model, so the claimant is routed again and again
    while the engine of the victim still runs. It must NOT be started in that time: its start would
    leave the decided card k MiB free, below the floor, and the memory the victim still holds is not
    yet free. Once the stop returns the claimant is started, with at least the floor left on the card,
    and the idle model on the other card was never unloaded."""
    async with world(tmp_path, monkeypatch, models=_hd_models(), rules=RULES, n_cards=2, max_parallel=2) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        loads = []                                  # one record per start of the claimant's engine
        orig_stamp = b.stamp

        def stamp_and_record(kind, name):
            if kind == "load" and name == TAG_CLAIM:
                # called before the fake engine takes its MiB: this is what the start saw
                loads.append(dict(t=time.monotonic(), free=b.free_list()[_HD_DECIDED], need=b.need(TAG_CLAIM)))
            return orig_stamp(kind, name)

        b.stamp = stamp_and_record

        # the other card's model is parked first, so it is the older one; it belongs to an unlisted client
        ib = await serve_and_park(b, "IB", TAG_IDLE2, _HD_OTHER, meta=META_UNLISTED, thread="th-b2")
        ia = await serve_and_park(b, "IA", TAG_IDLE, _HD_DECIDED, meta=META_OWNER, thread="th-a")
        need = b.need(TAG_CLAIM)
        b.set_free(_HD_DECIDED, need + k)
        b.set_free(_HD_OTHER, _HD_OTHER_FREE_MIB)

        # the scenario is what it says it is
        assert ib.last_active_monotonic < ia.last_active_monotonic, "the off-card idle model is not the older one"
        assert b.mgr._lru_idle_unloadable() is ib, "the global victim should be the off-card idle model"
        assert b.mgr._lru_idle_unloadable(main_gpu=_HD_DECIDED, split_mode="none") is ia, (
            "the on-card victim should be the decided card's idle model")
        free0 = b.free_list()
        assert free0[_HD_DECIDED] == need + k and need <= free0[_HD_DECIDED] < need + FLOOR_MIB, free0
        assert free0[_HD_OTHER] - CUDA_RESIDUE_MIB >= FLOOR_MIB and free0[_HD_OTHER] < need, free0

        # the decided card's victim: its engine stop is held, set before the claimant is submitted
        b.stop_holds[TAG_IDLE] = asyncio.Event()
        t_submit = time.monotonic()
        submit_claimant(b)
        await asyncio.sleep(_HD_HOLD_S)             # past the one second backoff: several passes
        t_hold_end = time.monotonic()

        # ---- during the hold ----
        started_in_hold = bool(loads) or b.seen("load", TAG_CLAIM) or TAG_CLAIM in b.alive
        assert not started_in_hold, (
            f"the claimant's engine was started while the decided card's unload was still in flight "
            f"(card {_HD_DECIDED}: need {need}, free {need + k} = need + {k}, floor {FLOOR_MIB}); "
            + _hd_describe(b, loads, t_submit))
        assert _hd_unloaded(b) == [TAG_IDLE], (
            f"during the hold the only model unloaded should be the decided card's {TAG_IDLE}; "
            + _hd_describe(b, loads, t_submit))
        assert b.seen("engine_stop_waiting", TAG_IDLE) and not b.seen("engine_stop", TAG_IDLE), (
            "the victim's engine stop should be under way and still held; " + _hd_describe(b, loads, t_submit))

        # ---- release and let the claimant be served ----
        b.stop_holds[TAG_IDLE].set()
        await wait_served(b, "the claimant to be served once the victim's engine had stopped")

        # ---- after the hold ----
        assert len(loads) == 1, (
            "the claimant's engine should be started exactly once; " + _hd_describe(b, loads, t_submit, t_hold_end))
        left = loads[0]["free"] - loads[0]["need"]
        assert left >= FLOOR_MIB, (
            f"the claimant's start left {left} MiB free on the decided card (it read {loads[0]['free']} free, "
            f"its footprint is {loads[0]['need']}), below the floor of {FLOOR_MIB}; "
            + _hd_describe(b, loads, t_submit, t_hold_end))
        assert untouched(b, TAG_IDLE2) == [], (
            f"the idle model on the other card was acted on: {untouched(b, TAG_IDLE2)}; "
            + _hd_describe(b, loads, t_submit, t_hold_end))
        assert _hd_unloaded(b) == [TAG_IDLE], (
            "only the decided card's idle model should have been unloaded; "
            + _hd_describe(b, loads, t_submit, t_hold_end))


# The point where the card reads exactly the claimant's need plus the floor is outside the band above:
# the automatic placer lands the claimant there by itself, so the decision is not what places it.
async def test_at_need_plus_floor_exactly_the_placer_lands_by_itself_and_the_unpinned_count_cap_arm_runs(
        tmp_path, monkeypatch):
    """CONTROL: behaviour that does not change. The decided-looking card reads exactly need + floor
    free. This point cannot be a pinned test: the automatic placer lands the claimant on that card by
    itself (it keeps the floor), so the reclaim-first decision does not run and no pin is set. The
    count-cap arm therefore runs on its unpinned lines: the global victim (the older, less protected
    model on the other card) is unloaded and the claimant is reserved in-band on the same pass, while
    the idle model on the decided card is left alone."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=2) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _cc_build(b, FLOOR_MIB)

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        slot = b.slots["C1"]
        assert getattr(slot, "fastlane_reclaim_first", False) is False and not b.seen("log", "reclaim"), (
            f"the claimant carries the pin at need + floor; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE2], (
            f"models unloaded: {unloaded}; wanted only the global victim {TAG_IDLE2} (the older model on the "
            f"other card); {_cc_ctx(b, rec)}")
        lines = _cc_eviction_lines(b, "make_room_count_cap")
        assert len(lines) == 1 and f"model_tag={TAG_IDLE2} " in lines[0], (
            f"count-cap eviction lines {lines}; wanted exactly one, naming {TAG_IDLE2}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert rec.passes and rec.passes[0]["reserved"] is True, (
            f"the claimant must be reserved in-band on its first pass; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE) == [], (
            f"the decided card's {TAG_IDLE} was acted on: {untouched(b, TAG_IDLE)}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"


# ================================================================================================
# A claimant pinned to a card, under the model-count cap, reads free = need + k with k below the floor
#
# Two idle residents sit on different cards and the model-count cap is 3, so the claimant is under
# the cap (the plain arm runs, not the count-cap arm). The decided card reads only a little more
# free than the claimant needs, so the memory gate (which has no floor) admits it, but starting it
# there would leave the card below the floor.
# ================================================================================================
_gd_k = 500                   # in the band: need <= free < need + floor
_gd_room_extra_mib = 500      # "room": free = need + floor + this
_gd_start_passes = 3          # passes the claimant may take once the card has room
_gd_start_wall_s = 4.0        # wall clock the claimant may take once the card has room
_gd_credit_mib = 2000        # a credited unload in flight on the decided card: more than the shortfall below the floor
_gd_wait_s = 2.5              # how long the claimant is watched while the card has no room
_gd_unreadable_s = 8.0        # how long the claimant may take to be served while the room reading is unavailable


async def _gd_build(b, k):
    """Two idle residents (the unlisted one on the other card parked FIRST, so the older one) and the
    decided card reading need + k free; the model count is below its cap. Returns (idle model on the
    decided card, idle model on the other card, need)."""
    b.mgr._vram_total_mib = [CARD_MIB] * 2
    ib = await serve_and_park(b, "IB", TAG_IDLE2, _cc_card_o, meta=META_UNLISTED, thread="th-b2")
    ia = await serve_and_park(b, "IA", TAG_IDLE, _cc_card_g, meta=META_OWNER, thread="th-a")
    need = b.need(TAG_CLAIM)
    b.set_free(_cc_card_g, need + k)
    b.set_free(_cc_card_o, _cc_free_o)
    assert ib.last_active_monotonic < ia.last_active_monotonic, "the other card's model is not the older one"
    glob = b.mgr._lru_idle_unloadable()
    oncard = b.mgr._lru_idle_unloadable(main_gpu=_cc_card_g, split_mode="none")
    assert glob is ib and oncard is ia, (
        f"scenario: the global victim must be {TAG_IDLE2} and the decided card's {TAG_IDLE}; "
        f"got global={getattr(glob, 'model_tag', None)} on_card={getattr(oncard, 'model_tag', None)}")
    cap = b.mgr.runtime.queue.max_parallel_sidecars
    assert len(list(b.mgr._model_residents())) < cap, "the claimant must be under the model-count cap"
    assert _cc_free_o - CUDA_RESIDUE_MIB >= FLOOR_MIB, "the other card cannot pass the residue check"
    return ia, ib, need


async def test_a_pinned_claimant_waits_while_the_decided_cards_unload_covers_the_need_and_starts_once_the_card_has_room(
        tmp_path, monkeypatch):
    """The claimant is pinned to card 1, under the model-count cap, and the card reads need + 500
    free (need <= free < need + floor), while an unload already in flight on that card is credited
    with more than the shortfall (the credit is held open by a task that does not finish, so no real
    teardown timing is involved). Starting the claimant there would leave the card 500 MiB free after
    its own footprint, below the floor of 1000, and the credited unload already covers what is
    missing, so the claimant must only wait: in the first 2.5 s its engine is not started, nothing is
    unloaded (not the idle model on the decided card either, and not the one on the other card), no
    in-band reserve is made and the pin stays set on its slot. Then the credit ends and the card is
    given room (free = need + floor + 500) and the claimant's engine must start within a bounded
    number of further passes through the routing step (at most 3 passes and at most 4.0 s of wall
    clock; measured: one pass, about 0.54 s, three runs) with at least the floor left free on the
    decided card after its own footprint."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
        mgr = b.mgr
        mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _gd_build(b, _gd_k)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB, (
            f"scenario: decided card free {b.free_list()[_cc_card_g]}, need {need}, floor {FLOOR_MIB}")

        owner = asyncio.create_task(asyncio.Event().wait())
        mgr._pending_reclaim_mib[_cc_card_g] = _gd_credit_mib
        mgr._register_card_release_task(_cc_card_g, owner)
        try:
            # ---- phase 1: the credited unload covers the need, no room yet ----
            submit_claimant(b)
            await asyncio.sleep(_gd_wait_s)

            unloaded = _cc_unloaded(b)
            started = b.seen("load", TAG_CLAIM)
            assert not started, (
                f"the claimant's engine was started while the decided card read {need + _gd_k} free (need {need}, "
                f"floor {FLOOR_MIB}), which leaves {_gd_k} MiB free after its own footprint; "
                f"engine starts: {rec.loads}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert unloaded == [], (
                f"models unloaded while the credited unload covered the need: {unloaded}; wanted none; "
                f"{_cc_ctx(b, rec)}")
            assert rec.passes, f"the claimant was never routed; {_cc_ctx(b, rec)}"
            assert not any(t == TAG_CLAIM for (_p, t) in rec.reserves), (
                f"the claimant was reserved in-band: {rec.reserves}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            slot = b.slots.get("C1")
            assert slot is not None and getattr(slot, "fastlane_reclaim_first", False) is True, (
                f"the claimant is not pinned: slot={slot is not None}, "
                f"pin={getattr(slot, 'fastlane_reclaim_first', None)}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        finally:
            owner.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(owner, 2)
            mgr._pending_reclaim_mib[_cc_card_g] = 0

        # ---- phase 2: the credit is gone and the card has room ----
        passes_before = rec.n
        t_room = time.monotonic()
        b.set_free(_cc_card_g, need + FLOOR_MIB + _gd_room_extra_mib)
        try:
            await b.until(lambda: b.seen("load", TAG_CLAIM),
                          "the claimant's engine to start once the card has room", _gd_start_wall_s)
        except AssertionError:
            pass
        elapsed = time.monotonic() - t_room
        passes_taken = rec.n - passes_before
        unloaded = _cc_unloaded(b)
        assert b.seen("load", TAG_CLAIM), (
            f"the claimant's engine did not start within {_gd_start_wall_s}s after the card read "
            f"{need + FLOOR_MIB + _gd_room_extra_mib} free (need {need}, floor {FLOOR_MIB}); passes taken "
            f"{passes_taken}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert passes_taken <= _gd_start_passes and elapsed <= _gd_start_wall_s, (
            f"the claimant took {passes_taken} passes and {elapsed:.2f}s to start once the card had room; "
            f"allowed {_gd_start_passes} passes and {_gd_start_wall_s}s; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert len(rec.loads) == 1, f"claimant engine starts: {rec.loads}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        left = rec.loads[0]["decided_after"]
        assert left >= FLOOR_MIB, (
            f"the claimant's engine start left the decided card with {left} MiB free after its own footprint "
            f"{need} (floor {FLOOR_MIB}; card free at start {rec.loads[0]['free']}); "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        served = await _cc_wait_served(b)
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"


async def test_a_pinned_claimant_waits_when_the_room_reading_is_unavailable(tmp_path, monkeypatch):
    """The claimant is pinned to card 1, under the model-count cap, and the card reads need + 500
    free (the band: a pinned claimant started at once would leave 500 MiB on the card, below the
    floor of 1000). From before the claimant is submitted, the helper that reports the room per
    card answers with an empty list, as when the memory probe is unavailable. An unreadable room
    reading never lets the claimant start below the floor: on its first pass the claimant is pinned,
    is NOT reserved in-band, and the idle
    model on the decided card is the one unloaded (exactly one unload; the model on the other card is
    left alone). The claimant is then served within a bounded wait
    and when its engine starts the decided card keeps at least the floor free after its own footprint;
    the worker task is still running and no error is logged by this test (the error tap is attached
    after a collection and the garbage collector is paused while it is attached, so messages about
    tasks leaked by earlier tests are not recorded)."""
    import gc
    import logging

    errors = []

    class _ErrTap(logging.Handler):
        def emit(self, record):
            errors.append(f"{record.name}: {record.getMessage()}")

    tap = _ErrTap(level=logging.ERROR)
    root = logging.getLogger()
    # Garbage left by earlier tests is collected before the tap is attached, and the collector stays
    # paused while the tap is attached, so the tap sees only what this test logs. The collector is
    # re-enabled after the tap is removed.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    try:
        gc.disable()
        root.addHandler(tap)
        async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
            b.mgr._vram_total_mib = [CARD_MIB] * 2
            rec = _cc_instrument(b, _cc_card_g)
            ia, ib, need = await _gd_build(b, _gd_k)

            # the room reading is unavailable from the start
            asked = []

            def unreadable():
                asked.append(time.monotonic())
                return []

            monkeypatch.setattr(b.mgr, "_card_avail_mib", unreadable)
            submit_claimant(b)
            served = await _cc_wait_served(b, _gd_unreadable_s)
            await asyncio.sleep(0.3)

            unloaded = _cc_unloaded(b)
            assert rec.passes, f"the claimant was never routed; room reads: {len(asked)}; {_cc_ctx(b, rec)}"
            first = rec.passes[0]
            assert first["pin"] is True and first["reloc"] == _cc_card_g, (
                f"the claimant was not pinned to card {_cc_card_g} on its first pass: pin={first['pin']} "
                f"card={first['reloc']}; room reads: {len(asked)}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert first["reserved"] is False, (
                f"the claimant was reserved in-band on its first pass while the room reading was unavailable; "
                f"engine starts: {rec.loads}; room reads: {len(asked)}; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            assert first["unloaded"] == [TAG_IDLE] and unloaded == [TAG_IDLE], (
                f"models unloaded on the first pass: {first['unloaded']}, in all: {unloaded}; wanted only the "
                f"decided card's idle model {TAG_IDLE}; room reads: {len(asked)}; {_cc_ctx(b, rec)}")
            assert untouched(b, TAG_IDLE2) == [] and resident_for(b.mgr, TAG_IDLE2) is ib, (
                f"the other card's {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            assert served, (
                f"the claimant was not served within {_gd_unreadable_s}s; room reads: {len(asked)}; "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert len(rec.loads) == 1, (
                f"claimant engine starts: {rec.loads}; room reads: {len(asked)}; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            left = rec.loads[0]["decided_after"]
            assert left >= FLOOR_MIB, (
                f"the claimant's engine start left the decided card with {left} MiB free after its own footprint "
                f"{need} (floor {FLOOR_MIB}; card free at start {rec.loads[0]['free']}); room reads: {len(asked)}; "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert not b.mgr._worker_task.done(), "the worker task stopped"
            assert errors == [], f"errors were logged while the room reading was unavailable: {errors}"
    finally:
        root.removeHandler(tap)
        if gc_was_enabled:
            gc.enable()


async def test_an_unpinned_slot_in_the_same_band_under_the_cap_starts_at_once_as_before(tmp_path, monkeypatch):
    """CONTROL: behaviour that does not change. The same two idle models, the model count below its cap
    and the same band figure (the card the claimant's manifest pins reads need + 500 free, so after its
    own footprint 500 MiB are left, below the floor of 1000), but the claimant carries no reclaim-first
    pin (the lane is off and the manifest pin is kept, so the decision does not apply). This is today's
    behaviour and it must not change for a slot without the pin: nothing is unloaded and the claimant's
    engine is started on its first pass."""
    async with world(tmp_path, monkeypatch, models=_cc_models(claim_pin=_cc_card_g), rules=[], n_cards=2,
                     max_parallel=3) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _gd_build(b, _gd_k)

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        slot = b.slots["C1"]
        assert getattr(slot, "fastlane_reclaim_first", False) is False and not b.seen("log", "reclaim"), (
            f"the claimant carries the pin although the lane is off; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert unloaded == [], f"models unloaded: {unloaded}; wanted none; {_cc_ctx(b, rec)}"
        assert rec.passes and rec.passes[0]["reserved"] is True, (
            f"the claimant must be reserved in-band on its first pass; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert len(rec.loads) == 1 and rec.loads[0]["decided_after"] == _gd_k, (
            f"claimant engine starts: {rec.loads}; wanted one start with {_gd_k} MiB left on the card; "
            f"{_cc_ctx(b, rec)}")
        assert served, f"the claimant was not served; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"


# ================================================================================================
# A pinned claimant that cannot start on this pass: one unload on the decided card, then it waits
#
# The decided card (card 1) reads need + k free with k below the floor, so the claimant is queued
# until the card has really been freed. These tests cover the plain arm (the model count is below its
# cap), the unload that is already in flight (no second unload for the same shortfall) and the
# model-count cap with the memory already covered (a count slot is freed, nothing is reserved in-band).
# ================================================================================================
_cu_served_passes = 4         # passes the claimant may take to be served
_cu_served_wall_s = 8.0       # wall clock the claimant may take to be served
_cu_hold_s = 3.0              # wall clock the first victim's engine stop stays held (or the credit stays held)
_cu_unlisted_idle_mib = 3000        # the unlisted (older) idle model on the decided card
_cu_unlisted_idle2_mib = 4000       # the second idle model on the decided card


async def _cu_wait_served(b, rec, bound=_cu_served_wall_s):
    """Wait (bounded) for the claimant to be served. Returns (served, passes the claimant took, seconds)."""
    t0 = time.monotonic()
    served = await _cc_wait_served(b, bound)
    return served, rec.n, time.monotonic() - t0


@pytest.mark.parametrize("k", [1, 500, 999], ids=lambda k: f"k{k}")
async def test_a_pinned_claimant_under_the_count_cap_in_the_band_gets_the_decided_cards_idle_model_unloaded_and_starts_with_the_floor(
        tmp_path, monkeypatch, k):
    """The model count is below its cap (3 slots, 2 residents), so the plain arm runs. The claimant is
    pinned to card 1, which reads need + k free (k below the floor: the memory gate admits it but an
    in-band start would leave k MiB free after its own footprint). The idle model ON card 1 is unloaded
    (exactly one unload, logged as the memory-arm eviction, the other card's idle model is left alone),
    nothing is reserved in-band on the first pass, and the claimant is started once the card has room,
    with at least the floor left free after its own footprint. Bounds: at most 4 passes and at most
    8.0 s of wall clock from the submit to the served reply (measured on the changed code: 2 passes, about
    0.05 s, for each of k = 1, 500, 999)."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _gd_build(b, k)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB, (
            f"scenario: decided card free {b.free_list()[_cc_card_g]}, need {need}, floor {FLOOR_MIB}")

        submit_claimant(b)
        served, passes, elapsed = await _cu_wait_served(b, rec)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        assert rec.passes, f"the claimant was never routed; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        first = rec.passes[0]
        assert b.count("log", "reclaim") >= 1 and first["pin"] is True and first["reloc"] == _cc_card_g, (
            f"the claimant was not pinned to card {_cc_card_g} on its first pass: pin={first['pin']} "
            f"card={first['reloc']}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert first["reserved"] is False, (
            f"the claimant was reserved in-band on its first pass while the card read {need + k} free "
            f"(need {need}, floor {FLOOR_MIB}); engine starts: {rec.loads}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE], (
            f"models unloaded: {unloaded}; wanted exactly the decided card's idle model {TAG_IDLE} "
            f"(the other card's {TAG_IDLE2} must stay); {_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE2) == [] and resident_for(b.mgr, TAG_IDLE2) is ib, (
            f"the other card's {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        lines = _cc_eviction_lines(b, "make_room_vram")
        assert len(lines) == 1 and f"model_tag={TAG_IDLE} " in lines[0], (
            f"memory-arm eviction lines {lines}; wanted exactly one, naming {TAG_IDLE}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert not _cc_eviction_lines(b, "make_room_count_cap"), (
            f"a count-cap eviction was logged although the count is below its cap: "
            f"{_cc_eviction_lines(b, 'make_room_count_cap')}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert len(rec.loads) <= 1 and all(ld["decided_after"] >= FLOOR_MIB for ld in rec.loads), (
            f"the claimant's engine start left the decided card below the floor ({FLOOR_MIB}): "
            f"engine starts (free per card, MiB left after its own footprint {need}): {rec.loads}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert served, (
            f"the claimant was not served within {_cu_served_wall_s}s ({passes} passes); "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert len(rec.loads) == 1, f"claimant engine starts: {rec.loads}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}"
        assert passes <= _cu_served_passes and elapsed <= _cu_served_wall_s, (
            f"the claimant took {passes} passes and {elapsed:.2f}s to be served; allowed {_cu_served_passes} "
            f"passes and {_cu_served_wall_s}s; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert_served_on_card(b, _cc_card_g)


def _cu_unlisted_models():
    return {
        TAG_IDLE: dict(size_mib=_cu_unlisted_idle_mib, main_gpu=_cc_card_g),
        TAG_IDLE2: dict(size_mib=_cu_unlisted_idle2_mib, main_gpu=_cc_card_g),
        TAG_CLAIM: dict(size_mib=_cc_claim_mib, main_gpu=_cc_card_o),
    }


async def test_no_second_unload_on_the_decided_card_while_the_first_one_already_covers_the_need(
        tmp_path, monkeypatch):
    """Two idle models sit on the decided card 1, which reads need + 500 free (the shortfall to the
    floor is 500 MiB). The victim order takes the older, unlisted-owner model first and its 4000 MiB
    cover the whole shortfall, but its engine stop is held for 3.0 s (past the one second backoff, so
    the claimant is routed three or more times). During the hold exactly ONE unload happens (the
    other idle model on the card is not touched, the claimant is not started). When the hold is
    released the claimant is served, there is still exactly one unload in total, and its start left at
    least the floor free after its own footprint."""
    async with world(tmp_path, monkeypatch, models=_cu_unlisted_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        # the unlisted owner's model is parked FIRST, so it is the older one and the first victim
        first_victim = await serve_and_park(b, "IB", TAG_IDLE2, _cc_card_g, meta=META_UNLISTED, thread="th-b2")
        second = await serve_and_park(b, "IA", TAG_IDLE, _cc_card_g, meta=META_OWNER, thread="th-a")
        need = b.need(TAG_CLAIM)
        b.set_free(_cc_card_g, need + 500)
        b.set_free(_cc_card_o, _cc_free_o)
        assert first_victim.last_active_monotonic < second.last_active_monotonic, "the first victim is not the older one"
        assert b.mgr._lru_idle_unloadable(main_gpu=_cc_card_g, split_mode="none") is first_victim, (
            f"scenario: the card's first victim must be {TAG_IDLE2}")
        assert first_victim.reserved_need_mib >= FLOOR_MIB - 500, "scenario: the first unload cannot cover the shortfall"
        assert _cc_free_o - CUDA_RESIDUE_MIB >= FLOOR_MIB, "the other card cannot pass the residue check"
        assert len(list(b.mgr._model_residents())) < b.mgr.runtime.queue.max_parallel_sidecars, "not under the cap"
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB

        b.stop_holds[TAG_IDLE2] = asyncio.Event()      # set before the claimant is submitted
        submit_claimant(b)
        await asyncio.sleep(_cu_hold_s)

        # ---- during the hold ----
        unloaded = _cc_unloaded(b)
        assert not b.seen("load", TAG_CLAIM) and not rec.loads, (
            f"the claimant's engine was started during the hold while the card read {need + 500} free "
            f"(need {need}, floor {FLOOR_MIB}); engine starts: {rec.loads}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE2], (
            f"during the hold the models unloaded are {unloaded}; wanted exactly one, {TAG_IDLE2}; "
            f"{_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE) == [], (
            f"the second idle model {TAG_IDLE} was acted on while the first unload was in flight: "
            f"{untouched(b, TAG_IDLE)}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert resident_for(b.mgr, TAG_IDLE) is second and second.state is ResidentState.IDLE_EVICTABLE
        assert b.seen("engine_stop_waiting", TAG_IDLE2) and not b.seen("engine_stop", TAG_IDLE2), (
            f"the first victim's engine stop is not held; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert rec.n >= 3, (
            f"the claimant was routed only {rec.n} times in {_cu_hold_s}s, so the hold did not span the "
            f"passes this test is about; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert rec.passes[0]["pin"] is True and b.count("log", "reclaim") >= 1, (
            f"the claimant was not pinned on its first pass; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert b.mgr._pending_reclaim_mib.get(_cc_card_g, 0) >= FLOOR_MIB - 500, (
            f"the credit of the unload in flight does not cover the shortfall: "
            f"{b.mgr._pending_reclaim_mib}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")

        # ---- release and let the claimant be served ----
        b.stop_holds[TAG_IDLE2].set()
        served, passes, elapsed = await _cu_wait_served(b, rec)
        await asyncio.sleep(0.3)
        unloaded = _cc_unloaded(b)
        assert served, (
            f"the claimant was not served within {_cu_served_wall_s}s after the hold was released "
            f"({passes} passes); models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE2], (
            f"in total the models unloaded are {unloaded}; wanted exactly one, {TAG_IDLE2}; {_cc_ctx(b, rec)}")
        assert untouched(b, TAG_IDLE) == [], (
            f"the second idle model {TAG_IDLE} was acted on: {untouched(b, TAG_IDLE)}; models unloaded: "
            f"{unloaded}; {_cc_ctx(b, rec)}")
        assert len(rec.loads) == 1 and rec.loads[0]["decided_after"] >= FLOOR_MIB, (
            f"the claimant's engine start left the decided card below the floor ({FLOOR_MIB}): "
            f"engine starts (free per card, MiB left after its own footprint {need}): {rec.loads}; "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert_served_on_card(b, _cc_card_g)


async def test_at_the_count_cap_with_the_memory_already_covered_one_count_slot_is_freed_and_the_claimant_starts_with_the_floor(
        tmp_path, monkeypatch):
    """The model count is AT its cap (2 slots, 2 idle residents: TAG_IDLE on the decided card 1, the
    older unlisted-owner TAG_IDLE2 on card 0). Card 1 reads need + 500 free, and an unload already in
    flight on card 1 (a credit of 2000 MiB held open by a task that does not finish) covers the
    memory, so the decision reports nothing to free: the shortfall is the count slot. While the credit
    is held, exactly ONE unload happens (the arm's usual victim, the older model on the other card,
    logged as the count-cap eviction), the decided card's idle model is NOT unloaded, and nothing is
    reserved in-band for 3.0 s. Then the credit is lifted and card 1 is given room (free = need +
    floor + 500): the claimant is served (at most 4 passes in total and at most 8.0 s of wall clock,
    measured on the changed code: 1 pass, about 0.06 s, after the lift; 5 passes in all) with at least the floor left free after its own footprint, and still
    exactly one count-cap unload in total."""
    credit_mib = 2000
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=2) as b:
        mgr = b.mgr
        mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _cc_build(b, 500)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB

        owner = asyncio.create_task(asyncio.Event().wait())
        mgr._pending_reclaim_mib[_cc_card_g] = credit_mib
        mgr._register_card_release_task(_cc_card_g, owner)
        credit_lifted = False
        try:
            submit_claimant(b)
            await asyncio.sleep(_cu_hold_s)

            # ---- while the credit is held ----
            unloaded = _cc_unloaded(b)
            assert not b.seen("load", TAG_CLAIM) and not rec.loads and not any(
                t == TAG_CLAIM for (_p, t) in rec.reserves), (
                f"the claimant was reserved or started in-band while the card read {need + 500} free "
                f"(need {need}, floor {FLOOR_MIB}) and its memory was only covered by an unload in flight; "
                f"engine starts: {rec.loads}; reserves: {rec.reserves}; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")
            assert unloaded == [TAG_IDLE2], (
                f"models unloaded while the credit is held: {unloaded}; wanted exactly one, the older model "
                f"on the other card {TAG_IDLE2}; {_cc_ctx(b, rec)}")
            assert untouched(b, TAG_IDLE) == [] and resident_for(mgr, TAG_IDLE) is ia, (
                f"the decided card's {TAG_IDLE} was acted on: {untouched(b, TAG_IDLE)}; models unloaded: "
                f"{unloaded}; {_cc_ctx(b, rec)}")
            lines = _cc_eviction_lines(b, "make_room_count_cap")
            assert len(lines) == 1 and f"model_tag={TAG_IDLE2} " in lines[0], (
                f"count-cap eviction lines {lines}; wanted exactly one, naming {TAG_IDLE2}; "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert rec.passes and rec.passes[0]["pin"] is True and rec.passes[0]["reloc"] == _cc_card_g and b.count(
                "log", "reclaim") >= 1, (
                f"the claimant was not pinned to card {_cc_card_g} on its first pass; models unloaded: "
                f"{unloaded}; {_cc_ctx(b, rec)}")
            assert rec.n >= 2, (
                f"the claimant was routed only {rec.n} times in {_cu_hold_s}s; models unloaded: {unloaded}; "
                f"{_cc_ctx(b, rec)}")

            # ---- lift the shortfall: the credit goes, the card has room ----
            owner.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(owner, 2)
            mgr._pending_reclaim_mib[_cc_card_g] = 0
            credit_lifted = True
            passes_before = rec.n
            b.set_free(_cc_card_g, need + FLOOR_MIB + 500)
            served, _passes, elapsed = await _cu_wait_served(b, rec)
            await asyncio.sleep(0.3)
            passes_taken = rec.n - passes_before

            unloaded = _cc_unloaded(b)
            assert served, (
                f"the claimant was not served within {_cu_served_wall_s}s after the credit was lifted and "
                f"the card read {need + FLOOR_MIB + 500} free ({passes_taken} passes since); "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert passes_taken <= _cu_served_passes and elapsed <= _cu_served_wall_s, (
                f"the claimant took {passes_taken} passes and {elapsed:.2f}s to be served once the card had "
                f"room; allowed {_cu_served_passes} passes and {_cu_served_wall_s}s; models unloaded: "
                f"{unloaded}; {_cc_ctx(b, rec)}")
            assert len(rec.loads) == 1 and rec.loads[0]["decided_after"] >= FLOOR_MIB, (
                f"the claimant's engine start left the decided card below the floor ({FLOOR_MIB}): "
                f"engine starts (free per card, MiB left after its own footprint {need}): {rec.loads}; "
                f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
            assert unloaded == [TAG_IDLE2] and len(_cc_eviction_lines(b, "make_room_count_cap")) == 1, (
                f"in total the models unloaded are {unloaded} and the count-cap eviction lines "
                f"{_cc_eviction_lines(b, 'make_room_count_cap')}; wanted exactly one, {TAG_IDLE2}; "
                f"{_cc_ctx(b, rec)}")
            assert untouched(b, TAG_IDLE) == [], (
                f"the decided card's {TAG_IDLE} was acted on: {untouched(b, TAG_IDLE)}; models unloaded: "
                f"{unloaded}; {_cc_ctx(b, rec)}")
            assert_served_on_card(b, _cc_card_g)
        finally:
            if not credit_lifted:
                owner.cancel()
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(owner, 2)
                mgr._pending_reclaim_mib[_cc_card_g] = 0


# ================================================================================================
# Which kind of wait a pinned claimant gets
#
# A claimant the decision pinned to a card, for which one idle model on that card was unloaded and
# which cannot start on this pass, is queued for MEMORY (the one second backoff, counted as a memory
# deferral on its slot), never for a busy model (the 50 ms backoff, counted as a dispatch deferral).
# ================================================================================================
_df_free_offset = 500       # the decided card reads need + this much free (below the floor shortfall)


@pytest.mark.parametrize("arm", ["count_cap", "plain"], ids=lambda a: a)
async def test_a_pinned_claimant_that_must_wait_is_deferred_as_a_memory_wait_not_a_busy_wait(
        tmp_path, monkeypatch, arm):
    """Two idle residents (the decided card's and the other card's). count_cap: the model count is at
    its cap (2 slots). plain: the count is below its cap (3 slots). In both the decided card reads
    need + 500 free, so the claimant is pinned, the decided card's idle model is unloaded (logged as the
    count-cap eviction in the first arm, as the memory eviction in the second) and the claimant is
    queued. Once it is served, its slot must show at least one memory deferral and no dispatch
    deferral: it was only ever queued for memory, never for a busy model."""
    max_parallel = 2 if arm == "count_cap" else 3
    reason = "make_room_count_cap" if arm == "count_cap" else "make_room_vram"
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2,
                     max_parallel=max_parallel) as b:
        b.mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        if arm == "count_cap":
            ia, ib, need = await _cc_build(b, _df_free_offset)
        else:
            ia, ib, need = await _gd_build(b, _df_free_offset)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB, (
            f"scenario: decided card free {b.free_list()[_cc_card_g]}, need {need}, floor {FLOOR_MIB}")

        submit_claimant(b)
        served = await _cc_wait_served(b)
        await asyncio.sleep(0.3)

        unloaded = _cc_unloaded(b)
        slot = b.slots["C1"]
        vram_defers = getattr(slot, "_vram_defer_count", 0)
        busy_defers = getattr(slot, "_dispatch_defer_count", 0)
        counters = f"memory deferrals={vram_defers}, dispatch deferrals={busy_defers}"
        lines = _cc_eviction_lines(b, reason)
        assert len(lines) == 1 and f"model_tag={TAG_IDLE} " in lines[0], (
            f"arm {arm}: {reason} eviction lines {lines}; wanted exactly one, naming {TAG_IDLE}; "
            f"{counters}; models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert unloaded == [TAG_IDLE], (
            f"arm {arm}: models unloaded: {unloaded}; wanted only the decided card's idle model {TAG_IDLE}; "
            f"{counters}; {_cc_ctx(b, rec)}")
        assert served, (
            f"arm {arm}: the claimant was not served; {counters}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert vram_defers >= 1, (
            f"arm {arm}: the claimant was never queued for memory; {counters}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert busy_defers == 0, (
            f"arm {arm}: the claimant was queued as a busy wait; {counters}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")



# ================================================================================================
# The credit of an unload in flight is released while the victim still holds its memory
#
# A claimant pinned to card 1, under the model-count cap, reads need + 500 free there (500 is below
# the floor). The first unload on that card has begun and its engine stop is held, so the victim's
# memory is not free yet. The per-card credit raised when that unload began is then released early
# (production does this when the probe fails, or when a drop of 90 percent or more is confirmed)
# although the memory is still held. The claimant must still not be started on that card while it
# would be left below the floor: what decides is the live room figure, not whether a credit is open.
# ================================================================================================
_rl_first_mib = 4000        # the older, unlisted-owner idle model on the decided card: the first victim
_rl_second_mib = 3000       # the second idle model on the decided card: keeps the card reclaimable
_rl_free_offset = 500       # the decided card reads need + this much free (below the floor)
_rl_window_s = 3.0          # how long the claimant is watched after the credit is released
_rl_served_s = 8.0          # how long the claimant may take to be served once the stops return


def _rl_models():
    return {
        TAG_IDLE2: dict(size_mib=_rl_first_mib, main_gpu=_cc_card_g),
        TAG_IDLE: dict(size_mib=_rl_second_mib, main_gpu=_cc_card_g),
        TAG_CLAIM: dict(size_mib=_cc_claim_mib, main_gpu=_cc_card_o),
    }


async def test_a_credit_released_while_the_victim_still_holds_memory_does_not_let_a_pinned_claimant_start_below_the_floor(
        tmp_path, monkeypatch):
    """Three model slots, two idle models on the decided card 1 (the older unlisted-owner one is the
    first victim, the other keeps the card reclaimable) and the claimant pinned to card 1, which reads
    need + 500 free (below the floor once the claimant's own footprint is counted). An unload of the
    first victim is already in flight when the claimant arrives (started through the real unload
    step), and its engine stop is held, so its memory stays in use and the credit it raised for card 1
    is open. The engine stop of the second idle model is held too, so that nothing the test does not
    control can free the card. The claimant's first pass only waits (nothing reserved). Then the
    credit is released early (set to 0 straight in the per-card table, which is the value a real
    early release leaves; the card still reads need + 500 free). For 3.0 s the claimant is routed
    again and again: it must not be started and must not be reserved in-band (an in-band start would
    leave 500 MiB free after its own footprint, below the floor of 1000). It stays pinned to card 1;
    the second idle model may be unloaded in that time (the shortfall is real again, its credit is
    raised, and it is held); the models unloaded are named in every failure message. When both stops
    are let go, the claimant is served (at most 8.0 s) and its engine start left at least the floor
    free on card 1 after its own footprint."""
    first, second = TAG_IDLE2, TAG_IDLE
    async with world(tmp_path, monkeypatch, models=_rl_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
        mgr = b.mgr
        mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        # the unlisted owner's model is parked FIRST: it is the older one and the first victim
        v1 = await serve_and_park(b, "IB", first, _cc_card_g, meta=META_UNLISTED, thread="th-b2")
        v2 = await serve_and_park(b, "IA", second, _cc_card_g, meta=META_OWNER, thread="th-a")
        need = b.need(TAG_CLAIM)
        b.set_free(_cc_card_g, need + _rl_free_offset)
        b.set_free(_cc_card_o, _cc_free_o)

        # the scenario is what it says it is
        assert v1.last_active_monotonic < v2.last_active_monotonic, "the first victim is not the older one"
        assert mgr._lru_idle_unloadable(main_gpu=_cc_card_g, split_mode="none") is v1, (
            f"scenario: the card's first victim must be {first}")
        assert v1.reserved_need_mib >= FLOOR_MIB - _rl_free_offset and v2.reserved_need_mib >= FLOOR_MIB - _rl_free_offset, (
            "scenario: unloading either idle model alone must free more than the shortfall")
        assert _cc_free_o - CUDA_RESIDUE_MIB >= FLOOR_MIB, "the other card cannot pass the residue check"
        assert len(list(mgr._model_residents())) < mgr.runtime.queue.max_parallel_sidecars, "not under the cap"
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB

        b.stop_holds[first] = asyncio.Event()        # both set before anything is unloaded
        b.stop_holds[second] = asyncio.Event()

        # an unload of the first victim is already in flight when the claimant arrives (an earlier
        # eviction on that card), started through the real unload step under the registry lock; its
        # engine stop is held, so its memory stays in use and its credit is open
        async with mgr._registry_lock:
            mgr._begin_unload_locked(v1)
        await b.until(lambda: b.seen("engine_stop_waiting", first), "the first victim's engine stop to wait")
        credit_before = dict(mgr._pending_reclaim_mib)
        assert credit_before.get(_cc_card_g, 0) >= FLOOR_MIB - _rl_free_offset, (
            f"scenario: the unload in flight raised no credit for card {_cc_card_g}: {credit_before}")
        assert b.free_list()[_cc_card_g] == need + _rl_free_offset, b.free_list()

        submit_claimant(b)
        # the claimant's first pass runs while the credit is open: it only waits
        await b.until(lambda: rec.passes or b.seen("load", TAG_CLAIM),
                      "the claimant's first pass to finish, or the claimant to start")
        first_pass = dict(rec.passes[0]) if rec.passes else None

        # the early release: the credit goes while the victim's memory is still held
        released = not b.seen("load", TAG_CLAIM)
        passes_at_release = rec.n
        if released:
            mgr._pending_reclaim_mib[_cc_card_g] = 0
            assert mgr._pending_reclaim_mib.get(_cc_card_g, 0) == 0
            assert b.free_list()[_cc_card_g] == need + _rl_free_offset, (
                f"scenario: the decided card no longer reads {need + _rl_free_offset} free: {b.free_list()}")
        await asyncio.sleep(_rl_window_s)

        # ---- while the first victim's stop is held, after the credit was released ----
        unloaded = _cc_unloaded(b)
        window = (f"credit before the release {credit_before}, released={released}; first pass {first_pass}; "
                  f"claimant passes since the release: {rec.n - passes_at_release}; "
                  f"credit now {dict(mgr._pending_reclaim_mib)}; "
                  f"models unloaded: {unloaded}; reserves: {rec.reserves}; engine starts: {rec.loads}; "
                  f"{_cc_ctx(b, rec)}")
        assert first_pass is not None and first_pass["reserved"] is False and first_pass["unloaded"] == [], (
            f"on its first pass, with the credit open, the claimant should only wait; {window}")
        assert not b.seen("load", TAG_CLAIM) and not rec.loads and TAG_CLAIM not in b.alive, (
            f"the claimant was started while the first victim's engine stop was still held and its memory in use "
            f"(card {_cc_card_g} read {need + _rl_free_offset} free, need {need}, floor {FLOOR_MIB}); {window}")
        assert not any(t == TAG_CLAIM for (_p, t) in rec.reserves), (
            f"the claimant was reserved in-band while the card read {need + _rl_free_offset} free "
            f"(need {need}, floor {FLOOR_MIB}); {window}")
        assert unloaded[:1] == [first], f"the first unload should be {first}; {window}"
        assert b.seen("engine_stop_waiting", first) and not b.seen("engine_stop", first), (
            f"the first victim's engine stop is not held; {window}")
        assert rec.n - passes_at_release >= 2, (
            f"the claimant was routed only {rec.n - passes_at_release} times after the credit was released, "
            f"so the window did not span the passes this test is about; {window}")
        assert rec.passes[0]["pin"] is True and all(p["pin"] is True for p in rec.passes[passes_at_release:]), (
            f"the claimant did not stay pinned to card {_cc_card_g}; {window}")
        assert b.free_list()[_cc_card_g] == need + _rl_free_offset, (
            f"the decided card's memory changed during the hold: {b.free_list()}; {window}")

        # ---- let both stops go and let the claimant be served ----
        b.stop_holds[first].set()
        b.stop_holds[second].set()
        served, passes, elapsed = await _cu_wait_served(b, rec, _rl_served_s)
        await asyncio.sleep(0.3)
        unloaded = _cc_unloaded(b)
        assert served, (
            f"the claimant was not served within {_rl_served_s}s after the stops returned ({passes} passes); "
            f"models unloaded: {unloaded}; {_cc_ctx(b, rec)}")
        assert len(rec.loads) == 1 and rec.loads[0]["decided_after"] >= FLOOR_MIB, (
            f"the claimant's engine start left the decided card below the floor ({FLOOR_MIB}): engine starts "
            f"(free per card, MiB left after its own footprint {need}): {rec.loads}; models unloaded: {unloaded}; "
            f"{_cc_ctx(b, rec)}")
        assert_served_on_card(b, _cc_card_g)


# ================================================================================================
# The plain memory arm never starts a pinned claimant, whatever a room reading says
#
# A claimant pinned to card 1, under the model-count cap, reads need + 500 free there (the memory
# gate, which has no floor, admits it; the placer's own reading finds no card with room for it).
# The helper that reports the room per card is patched to report ample room on every card. The
# plain arm must not look at it: it unloads the idle model on the decided card and queues the
# claimant, which is started only after the placer lands it and the pin is cleared.
# ================================================================================================
_pw_window_s = 3.0            # how long the claimant is watched after its first pass


async def test_a_pinned_claimant_in_the_plain_arm_is_deferred_and_never_started_when_the_room_reading_disagrees_with_the_placer(
        tmp_path, monkeypatch):
    """Deterministic: nothing here depends on timing beyond one bounded window of 3.0 s. Three model
    slots (the claimant is under the model-count cap), two idle models (one on the decided card 1,
    one on card 0), and the claimant pinned to card 1, which reads need + 500 free: the memory gate
    admits the claimant there and the placer's own reading finds no card with room for it. The
    helper that reports the room per card (the one the plain arm used to read) is patched to report
    ample room on every card, so only the arm itself can keep the claimant from starting. The
    engine stop of the idle model on the decided card is held, so its memory stays in use and the
    card keeps reading need + 500 free for the whole window. The test first checks the scenario (the
    claimant is pinned, the memory gate admits it on the decided card, the registry is under its cap)
    and that the arm reached is the plain one (the shared unload step is asked with the memory
    reasons, never the count-cap ones). Then, for 3.0 s and at least two passes, the claimant must
    not be started, must not be reserved in-band, must stay pinned, and must be queued as a memory
    wait and never as a busy wait; the model unloaded is the one on the decided card, never the one
    on the other card."""
    async with world(tmp_path, monkeypatch, models=_cc_models(), rules=RULES, n_cards=2, max_parallel=3) as b:
        mgr = b.mgr
        mgr._vram_total_mib = [CARD_MIB] * 2
        rec = _cc_instrument(b, _cc_card_g)
        ia, ib, need = await _gd_build(b, _gd_k)
        assert need <= b.free_list()[_cc_card_g] < need + FLOOR_MIB, (
            f"scenario: decided card free {b.free_list()[_cc_card_g]}, need {need}, floor {FLOOR_MIB}")
        b.stop_holds[TAG_IDLE] = asyncio.Event()      # set before anything is unloaded
        n_res_before = len(list(mgr._model_residents()))
        assert n_res_before < mgr.runtime.queue.max_parallel_sidecars, (
            f"scenario: the claimant is not under the model-count cap: {n_res_before} residents, "
            f"cap {mgr.runtime.queue.max_parallel_sidecars}; {_cc_ctx(b, rec)}")
        assert b.free_list()[_cc_card_g] == need + _gd_k, (
            f"scenario: the decided card does not read {need + _gd_k} free (need {need}): "
            f"{b.free_list()}; {_cc_ctx(b, rec)}")

        # the room helper reports ample room on every card; the placer's own reading is unchanged
        room_reads = []

        def ample_room():
            room_reads.append(time.monotonic())
            return [CARD_MIB] * 2

        monkeypatch.setattr(mgr, "_card_avail_mib", ample_room)

        # observe only: the memory gate as the routing step asks it, the shared unload step, the deferrals
        gate_calls = []
        real_gate = mgr._vram_admits_locked

        def gate_spy(*a, **k):
            out = real_gate(*a, **k)
            if not k and len(a) >= 4 and a[0] == need:
                gate_calls.append((a[2], a[3], out))
            return out

        mgr._vram_admits_locked = gate_spy

        unload_calls = []
        real_unload = mgr._unload_on_decided_card_locked

        def unload_spy(slot, *a, **k):
            unload_calls.append((slot.model_tag, a[3] if len(a) > 3 else None, a[4] if len(a) > 4 else None))
            return real_unload(slot, *a, **k)

        mgr._unload_on_decided_card_locked = unload_spy

        defer_calls = []
        real_defer = mgr._defer_unroutable

        def defer_spy(slot, *, evict_pending=False):
            defer_calls.append((slot.model_tag, evict_pending))
            return real_defer(slot, evict_pending=evict_pending)

        mgr._defer_unroutable = defer_spy

        try:
            submit_claimant(b)
            await b.until(lambda: rec.passes or b.seen("load", TAG_CLAIM),
                          "the claimant's first pass to finish, or the claimant to start")
            await asyncio.sleep(_pw_window_s)

            unloaded = _cc_unloaded(b)
            slot = b.slots.get("C1")
            vram_defers = getattr(slot, "_vram_defer_count", 0)
            busy_defers = getattr(slot, "_dispatch_defer_count", 0)
            seen = (f"memory deferrals={vram_defers}, dispatch deferrals={busy_defers}; deferrals {defer_calls}; "
                    f"shared unload step asked {unload_calls}; memory gate (card, split, admitted) {gate_calls}; "
                    f"room helper reads {len(room_reads)}; models unloaded: {unloaded}; "
                    f"reserve calls: {rec.reserves}; engine starts: {rec.loads}; {_cc_ctx(b, rec)}")

            # ---- the scenario is what it says it is ----
            assert rec.passes and rec.passes[0]["pin"] is True and rec.passes[0]["reloc"] == _cc_card_g, (
                f"the claimant was not pinned to card {_cc_card_g} on its first pass; {seen}")
            assert slot is not None and getattr(slot, "fastlane_reclaim_first", False) is True, (
                f"the claimant is not pinned after the window: slot={slot is not None}, "
                f"pin={getattr(slot, 'fastlane_reclaim_first', None)}; {seen}")
            assert gate_calls and all(card == _cc_card_g for (card, _s, _o) in gate_calls), (
                f"the memory gate was not asked for the claimant on the decided card {_cc_card_g}; {seen}")
            assert gate_calls[-1][2] is True, (
                f"the memory gate did not admit the claimant on the decided card; {seen}")
            assert unload_calls and all(
                    (t, r, sr) == (TAG_CLAIM, "make_room_vram", "make_room_starved_vram")
                    for (t, r, sr) in unload_calls), (
                f"the shared unload step was not asked only with the memory reasons for the claimant "
                f"(wanted make_room_vram, make_room_starved_vram); {seen}")
            assert not any("count_cap" in m for m in b.logs if m.startswith("MAKE_ROOM")), (
                f"the count-cap arm was reached; {seen}")
            assert b.seen("engine_stop_waiting", TAG_IDLE) and not b.seen("engine_stop", TAG_IDLE), (
                f"the decided card's idle model was not held in its engine stop; {seen}")

            # ---- the measured behaviour ----
            assert not b.seen("load", TAG_CLAIM) and not rec.loads and TAG_CLAIM not in b.alive, (
                f"the claimant's engine was started while the card read {need + _gd_k} free (need {need}, "
                f"floor {FLOOR_MIB}) and the room helper was patched to report ample room; {seen}")
            assert not any(t == TAG_CLAIM for (_p, t) in rec.reserves), (
                f"the claimant was reserved in-band; {seen}")
            assert not any(p["reserved"] for p in rec.passes), (
                f"a claimant pass reserved in-band; {seen}")
            assert rec.n >= 2, (
                f"the claimant was routed only {rec.n} times in {_pw_window_s}s, so the window did not span "
                f"more than one pass through the arm; {seen}")
            assert all(p["pin"] is True for p in rec.passes), (
                f"the claimant did not stay pinned to card {_cc_card_g} on every pass; {seen}")
            assert any(t == TAG_CLAIM and ev is True for (t, ev) in defer_calls) and vram_defers >= 1, (
                f"the claimant was never queued as a memory wait; {seen}")
            assert not any(t == TAG_CLAIM and ev is False for (t, ev) in defer_calls) and busy_defers == 0, (
                f"the claimant was queued as a busy wait; {seen}")
            assert unloaded == [TAG_IDLE], (
                f"models unloaded: {unloaded}; wanted only the decided card's idle model {TAG_IDLE} and never "
                f"{TAG_IDLE2} on the other card; {seen}")
            assert untouched(b, TAG_IDLE2) == [] and resident_for(mgr, TAG_IDLE2) is ib, (
                f"the other card's {TAG_IDLE2} was acted on: {untouched(b, TAG_IDLE2)}; {seen}")
        finally:
            b.stop_holds[TAG_IDLE].set()
