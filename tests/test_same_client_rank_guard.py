"""Pick-stage tests for tag rank inside ONE Fast Lane client.

Rule under test: inside ONE client the better (lower) tag rank goes first, also
at the grace-window pick: a worse-ranked same-thread follow-up is not served
while a better-ranked request of the same client is waiting, and an equal rank
does not outrank.

Everything runs through the real manager (`submit_and_wait`, the worker loop,
`_dispatch_loop`, `_route_or_reserve`, the designated-target predicate, the
per-resident driver and its grace loop). Only the sidecar process, the health
probe, the teardown call and the completion call are faked; the GPU free-memory
probe is a model of ONE card that holds exactly one 20000 MiB model. Each test
asserts WHICH request ran and WHICH resident was unloaded, and the ORDER of
events from stamps taken by the fakes at the moment the real code reached them.
Free VRAM: `box()` patches BOTH `turbohaul.safety._read_free_vram_all_mib` and
`turbohaul.manager._read_free_vram_all_mib` with the same card model.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import pytest
import yaml

from _fastlane_fixture import assert_resolves, boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager

IP_1 = "192.0.2.10"        # the listed client
FOOTPRINT_MIB = 20000
CARD_MIB = 24000           # one model fits, two never do
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)

TAG_A, TAG_B, TAG_C = "model-a", "model-b", "model-c"

# one client (one address): main = 1, curator = 3, unclassified = 5
RANKS = FastLaneTagRanks(main=1, curator=3, unclassified=5)
ONE_CLIENT = [FastLaneRule(address=IP_1, tag_ranks=RANKS)]

CURATOR_1 = {"ip": IP_1, "is_curator": True, "is_sub_agent": True}
MAIN_1 = {"ip": IP_1, "is_main": True}


@dataclass
class Box:
    mgr: TurbohaulManager
    holds: dict                       # prompt label -> asyncio.Event its turn blocks on
    t0: float
    events: list = field(default_factory=list)   # (monotonic_stamp, kind, name)
    tasks: dict = field(default_factory=dict)    # label -> submit task
    gates: list = field(default_factory=list)    # extra events to release at teardown
    dispatch_gate: object = None

    def stamp(self, kind, name):
        self.events.append((time.monotonic(), kind, name))

    def first(self, kind, name):
        for t, k, n in self.events:
            if (k, n) == (kind, name):
                return t
        return None

    def seen(self, kind, name):
        return self.first(kind, name) is not None

    def order(self):
        return [(k, n) for _t, k, n in self.events]

    def rel(self):
        return [(round(t - self.t0, 3), k, n) for t, k, n in self.events]

    def submit(self, label, tag, meta, thread, hold=False):
        if hold:
            self.holds[label] = asyncio.Event()
        task = asyncio.create_task(asyncio.wait_for(
            self.mgr.submit_and_wait(tag, label, thread_id=thread, client_meta=dict(meta)),
            timeout=60))
        self.tasks[label] = task
        return task

    def loaded_tags(self):
        return {r.model_tag for r in self.mgr._model_residents() if r.state in LOADED}

    async def until(self, predicate, what, timeout=8.0):
        """Poll an observable predicate; AssertionError names what never happened."""
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"never happened within {timeout}s: {what}; events={self.rel()} "
                    f"loaded={sorted(self.loaded_tags())}")
            await asyncio.sleep(0.005)
        return time.monotonic() - t0

    async def stays_false(self, predicate, what, seconds):
        """The predicate must stay false for `seconds` (observed by polling)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            assert not predicate(), f"happened while it must not: {what}; events={self.rel()}"
            await asyncio.sleep(0.01)

    def one_model_fits(self, tag):
        need, parallel, main_gpu, split_mode, *_ = self.mgr._resolve_placement_locked(tag)
        return self.mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)


def _manifest(boot, tag, main_gpu=0):
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": FOOTPRINT_MIB * 1024 * 1024, "context_size": 2048,
        "expected_vram_bytes": FOOTPRINT_MIB * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


@contextlib.asynccontextmanager
async def box(tmp_path, monkeypatch, *, rules, grace_s=0, card_mib=CARD_MIB, cap=2,
              gate_dispatch=False, main_gpus=None):
    """A real manager over a card of `card_mib` (default: one model fits);
    `Box.holds[label]` blocks that turn. `main_gpus` maps a tag to its card (default 0);
    with it `card_mib` may be a list, one entry per card. With `gate_dispatch` the dispatcher's own
    staging pop waits on `Box.dispatch_gate` (grace-window and rider pops are not gated)."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=cap, grace_seconds=grace_s,
        max_grace_extensions=5 if grace_s else 0, idle_hot_load_seconds=120)
    for t in (TAG_A, TAG_B, TAG_C):
        _manifest(boot, t, (main_gpus or {}).get(t, 0))
    spawn, health, sigterm, vram, _complete = make_fakes({})
    holder: dict = {}
    holds: dict = {}

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        holder["box"].stamp("load", model_tag)
        return spawn(binary, gguf, port, model_tag, argv, **kw)

    async def complete_rec(slot, handle):
        b = holder["box"]
        label = slot.prompt
        b.stamp("turn_start", label)
        gate = b.holds.get(label)
        if gate is not None:
            await gate.wait()
        b.stamp("turn_end", label)
        return {"ok": True, "label": label, "model": handle.model_tag}

    cards = list(card_mib) if isinstance(card_mib, (list, tuple)) else [card_mib]

    def free_vram():
        loaded = [r for r in holder["mgr"]._model_residents() if r.state in LOADED]
        return [cap_mib - FOOTPRINT_MIB * sum(1 for r in loaded if r.main_gpu == i)
                for i, cap_mib in enumerate(cards)]

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    async def sigterm_rec(*a, **k):  # pass-through: records that an engine was stopped
        holder["box"].stamp("engine_stop", "sigterm")
        return await sigterm(*a, **k)

    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = Box(mgr=mgr, holds=holds, t0=time.monotonic())
    holder["mgr"], holder["box"] = mgr, b
    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):  # pass-through: records WHO the real code unloaded
        b.stamp("unload", r.model_tag)
        return orig_unload(r)

    mgr._begin_unload_locked = unload_spy
    if gate_dispatch:
        b.dispatch_gate = asyncio.Event()
        b.dispatch_gate.set()          # open until a test closes it
        b.gates.append(b.dispatch_gate)
        orig_pop = mgr.queue.pop_next

        async def gated_pop(*a, **k):
            if "room_available" in k:      # the dispatcher's own call shape
                await b.dispatch_gate.wait()
            return await orig_pop(*a, **k)

        mgr.queue.pop_next = gated_pop
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # teardown order: release turns, stop flag, THEN cancel the worker, bounded awaits
        for g in list(b.holds.values()) + list(b.gates):
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


async def _result(b, label, timeout=10.0):
    await b.until(lambda: b.tasks[label].done(), f"request {label!r} to finish", timeout)
    return b.tasks[label].result()   # raises (not an AssertionError) if the request failed


def _assert_completed(res, label, tag):
    slot, out = res
    assert out == {"ok": True, "label": label, "model": tag}, (
        f"request {label!r} did not complete intact: {out!r}")



@pytest.mark.asyncio
class TestDeclineAtTheGraceWindowPick:
    """The grace-window pick (not the designation) keeps rank order inside one client.

    A's last turn was a main turn (rank 1). A request WAIT of rank 1 for another model
    does not strictly outrank A, so A is not designated and keeps its grace window; the
    dispatcher is gated so WAIT stays in staging while a worse-ranked (rank 3) same-thread
    follow-up FUP of A arrives in that window. The card holds two models, so WAIT can load
    beside A once the dispatcher is released."""

    async def _setup(self, b, waiting_meta):
        b.submit("A1", TAG_A, MAIN_1, "th-a", hold=True)
        await b.until(lambda: b.seen("turn_start", "A1"), "A's turn to start")
        b.holds["A1"].set()
        _assert_completed(await _result(b, "A1"), "A1", TAG_A)
        a = resident_for(b.mgr, TAG_A)
        await b.until(lambda: a.in_grace_loop, "A to be inside its grace window")
        assert assert_resolves(b.mgr, TAG_A) == (0, 1), "A's last turn must resolve to rank 1"
        assert b.one_model_fits(TAG_B) is True, "the card does not hold two models"
        b.dispatch_gate.clear()        # from now on the dispatcher stops popping staging
        b.submit("WAIT", TAG_B, waiting_meta, "th-w")
        await b.until(lambda: len(b.mgr.queue._staging) == 1, "WAIT to be staged")
        assert b.mgr._is_designated_unload_target_locked(a) is False, (
            "A is designated: this arm would test the designation path, not the pick")
        b.submit("FUP", TAG_A, CURATOR_1, "th-a")

    async def test_follow_up_waits_for_better_ranked_staged_request(
            self, tmp_path, monkeypatch):
        """WAIT is main (rank 1) and staged; FUP is a curator (rank 3) same-thread follow-up
        of A arriving in A's grace window. Expected: FUP is not served while WAIT is still staged
        (the grace-window pick skips it), and once the dispatcher takes WAIT (its model starts
        loading) FUP is served after that; FUP is not lost."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=8, card_mib=44000,
                       gate_dispatch=True) as b:
            # Latch instead of a timed wait: a pass-through spy counts the grace loop's
            # polls that began while FUP sat in staging; the check below runs once three
            # such polls have gone by (or FUP was served, which is the failure).
            polls = []
            real_pop = b.mgr.queue.pop_matched_thread

            async def spy_pop(*a, **k):
                fup_staged = any(s.prompt == "FUP" for s in b.mgr.queue._staging)
                res = await real_pop(*a, **k)
                if fup_staged:
                    polls.append(res)
                return res

            b.mgr.queue.pop_matched_thread = spy_pop
            await self._setup(b, MAIN_1)
            await b.until(lambda: b.seen("turn_start", "FUP") or len(polls) >= 3,
                          "three grace-loop polls with FUP staged", 10.0)
            assert not b.seen("turn_start", "FUP"), (
                f"FUP served warm while the better-ranked WAIT is staged; events={b.rel()}")
            assert all(res is None for res in polls), f"a poll served a request: {polls}"
            assert len(b.mgr.queue._staging) == 2, "WAIT and FUP should both still be staged"
            b.dispatch_gate.set()
            await b.until(lambda: b.seen("turn_end", "FUP") and b.seen("turn_end", "WAIT"),
                          "WAIT and FUP to complete", 14.0)
            print(f"follow-up-waits events={b.rel()}")
            # WAIT is taken out of staging (its model starts loading) before FUP is served;
            # FUP may then run warm on A while WAIT's model finishes loading.
            assert b.first("load", TAG_B) < b.first("turn_start", "FUP"), f"events={b.rel()}"
            _assert_completed(await _result(b, "FUP"), "FUP", TAG_A)
            _assert_completed(await _result(b, "WAIT"), "WAIT", TAG_B)

    async def test_control_equal_rank_follow_up_is_served_warm_first(
            self, tmp_path, monkeypatch):
        """Control: WAIT is curator (rank 3), the same rank as the follow-up. Expected:
        FUP is served warm in the grace window while WAIT is still staged, so the
        gated dispatcher does not hide a served-first request and the arm can go red."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=8, card_mib=44000,
                       gate_dispatch=True) as b:
            await self._setup(b, CURATOR_1)
            await b.until(lambda: b.seen("turn_end", "FUP"), "FUP to be served warm", 6.0)
            assert not b.seen("turn_start", "WAIT"), f"WAIT ran with the dispatcher gated: {b.rel()}"
            print(f"follow-up-served-warm control events={b.rel()}")
            _assert_completed(await _result(b, "FUP"), "FUP", TAG_A)
