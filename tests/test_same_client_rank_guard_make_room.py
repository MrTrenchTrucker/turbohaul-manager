"""Guard and control tests for tag rank inside ONE Fast Lane client.

Rule under test: between clients the client order (rule_index) decides and a tag
rank never lifts one client above another; inside ONE client the better (lower)
tag rank goes first, a running turn is never cut, and a worse-ranked resident is
unloaded only after its turn has ended.

Everything runs through the real manager (`submit_and_wait`, the worker loop,
`_dispatch_loop`, `_route_or_reserve`, the designated-target predicate, the
per-resident driver and its grace loop). Only the sidecar process, the health
probe, the teardown call and the completion call are faked; the GPU free-memory
probe is a model of ONE card that holds exactly one 20000 MiB model. Each test
asserts WHICH request ran and WHICH resident was unloaded, and the ORDER of
events from stamps taken by the fakes at the moment the real code reached them.
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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import Slot

IP_1 = "192.0.2.10"        # listed client, rule position set per test
IP_2 = "198.51.100.20"     # listed client, rule position set per test
FOOTPRINT_MIB = 20000
CARD_MIB = 24000           # one model fits, two never do
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)

TAG_A, TAG_B, TAG_C = "model-a", "model-b", "model-c"

# one client (one address): main = 1, curator = 3, unclassified = 5
RANKS = FastLaneTagRanks(main=1, curator=3, unclassified=5)
ONE_CLIENT = [FastLaneRule(address=IP_1, tag_ranks=RANKS)]
CLIENT_1_FIRST = [FastLaneRule(address=IP_1, tag_ranks=RANKS),
                  FastLaneRule(address=IP_2, tag_ranks=RANKS)]
CLIENT_2_FIRST = [FastLaneRule(address=IP_2, tag_ranks=RANKS),
                  FastLaneRule(address=IP_1, tag_ranks=RANKS)]

CURATOR_1 = {"ip": IP_1, "is_curator": True, "is_sub_agent": True}
MAIN_1 = {"ip": IP_1, "is_main": True}
UNCLASSIFIED_1 = {"ip": IP_1}   # no class label: rank 5
MAIN_2 = {"ip": IP_2, "is_main": True}
CURATOR_2 = {"ip": IP_2, "is_curator": True, "is_sub_agent": True}


@dataclass
class Box:
    mgr: TurbohaulManager
    holds: dict                       # prompt label -> asyncio.Event its turn blocks on
    t0: float
    events: list = field(default_factory=list)   # (monotonic_stamp, kind, name)
    tasks: dict = field(default_factory=dict)    # label -> submit task
    gates: list = field(default_factory=list)    # extra events to release at teardown
    dispatch_gate: object = None
    passes: list = field(default_factory=list)   # result of every victim search the real code made

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

    async def after_victim_searches(self, n, what, timeout=20.0):
        """Wait until the real make-room code has searched for a victim `n` more times.
        A search is a point the code under test must pass, so what is asserted after it
        is decided by what that code did, not by how long the test waited."""
        target = len(self.passes) + n
        await self.until(lambda: len(self.passes) >= target, what, timeout)

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

    # Free VRAM: the capacity decision reads the manager's own binding, so BOTH the
    # manager's and safety's `_read_free_vram_all_mib` are pinned to the card model.
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
    orig_search = mgr._lru_idle_unloadable

    def search_spy(*a, **k):  # pass-through: records every victim search and its answer
        out = orig_search(*a, **k)
        b.passes.append(getattr(out, "model_tag", None))
        return out

    mgr._lru_idle_unloadable = search_spy
    orig_audit = mgr._audit_async

    async def audit_spy(slot, event_type):  # pass-through: records grace-family events by label
        b.stamp("audit", f"{slot.prompt}:{event_type}")
        return await orig_audit(slot, event_type)

    mgr._audit_async = audit_spy
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


async def _held_a_main_waiting(b, a_meta, main_meta, main_thread="th-main"):
    """A's turn HELD mid-generation on the one-model box, then a better-ranked
    request for the OTHER model arrives. Asserts the setup before returning."""
    b.submit("A1", TAG_A, a_meta, "th-a", hold=True)
    await b.until(lambda: b.seen("turn_start", "A1"), "A's turn to start")
    a = resident_for(b.mgr, TAG_A)
    assert a.state is ResidentState.ACTIVE, f"A is not mid-turn: {a.state}"
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model: box is not full"
    b.submit("M1", TAG_B, main_meta, main_thread)
    await b.until(lambda: len(b.mgr._fastlane_claims) > 0, "main's claim to register")
    return a



@pytest.mark.asyncio
class TestRunningTurnIsNeverCut:
    async def test_held_turn_finishes_then_unload_then_better_rank_loads(
            self, tmp_path, monkeypatch):
        """One client. Resident A is curator (rank 3) with a turn HELD mid-generation; a
        main (rank 1) request for a different model arrives on a one-model box. Expected:
        the held turn keeps running and A is not unloaded while it is held (A is the
        designated target, checked), the released turn completes with its full response,
        and only after that turn ended is A unloaded, only after A's unload does main's
        model load, and main's request then runs and completes."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=30) as b:
            a = await _held_a_main_waiting(b, CURATOR_1, MAIN_1)
            assert b.mgr._is_designated_unload_target_locked(a) is True, (
                "A is not the designated target of main's claim; the hold below would be vacuous")
            await b.after_victim_searches(3, "three victim searches while A's turn is held")
            assert not (b.seen("unload", TAG_A) or b.seen("turn_end", "A1") or b.seen("load", TAG_B)), (
                f"A unloaded, ended, or B loaded while A's turn is held; events={b.rel()}")
            assert a.state is ResidentState.ACTIVE and not b.tasks["A1"].done(), (
                f"held turn did not stay running: state={a.state}")
            b.holds["A1"].set()
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            await b.until(lambda: b.seen("turn_end", "M1"), "main's turn to complete", 12.0)
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)
            t = {k: b.first(*k) for k in (("turn_end", "A1"), ("unload", TAG_A),
                                          ("load", TAG_B), ("turn_end", "M1"))}
            print("held-turn events", b.rel())
            assert None not in t.values(), f"missing events {t}; events={b.rel()}"
            assert (t[("turn_end", "A1")] < t[("unload", TAG_A)] < t[("load", TAG_B)]
                    < t[("turn_end", "M1")]), f"wrong order; events={b.rel()}"
            unloaded = [n for k, n in b.order() if k == "unload"]
            assert unloaded[0] == TAG_A, f"first unloaded model was {unloaded[0]}"

    async def test_control_better_rank_resident_is_not_unloaded_for_worse_rank(
            self, tmp_path, monkeypatch):
        """Control for the held-turn test above (the instrument can see a rank): A is main (rank 1) held
        mid-turn and a curator (rank 3) request for the other model arrives. A is NOT
        designated, is not unloaded for the worse-ranked claim, and the curator request
        only loads after A is gone (A's turn ends and its grace window expires or is
        cut by the claim rules)."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=30) as b:
            a = await _held_a_main_waiting(b, MAIN_1, CURATOR_1, "th-cur")
            assert b.mgr._is_designated_unload_target_locked(a) is False, (
                "a rank-1 resident was designated for a rank-3 claim")
            await b.after_victim_searches(3, "three victim searches while A's turn is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B)), (
                f"better-ranked A unloaded or the worse-ranked model loaded; events={b.rel()}")
            b.holds["A1"].set()
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            await b.until(lambda: b.seen("audit", "A1:grace_enter"), "A to enter its grace window")
            await b.after_victim_searches(3, "three victim searches while A sits in its grace window")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B)), (
                f"A unloaded inside its grace window for a worse-ranked claim; events={b.rel()}")
            assert not b.seen("audit", "A1:grace_designated_victim_skip"), f"events={b.rel()}"


@pytest.mark.asyncio
class TestHeldTurnFinishesBeforeItsModelIsUnloaded:
    async def test_response_is_delivered_before_the_unload_begins_then_the_better_tag_loads(
            self, tmp_path, monkeypatch):
        """One client. A (curator, rank 3) has a turn held by a latch; a main (rank 1) request
        for another model waits on a one-model box and A is its designated target. The real
        make-room code is made to search for a victim three times while the latch is closed:
        nothing is unloaded, stopped, ended or loaded. After the latch is released, the
        order of events is: the held turn ends, its full response reaches the caller, only
        then does the unload begin, then the engine stop the fake teardown records, then
        main's model loads, then main's turn ends."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=30) as b:
            a = await _held_a_main_waiting(b, CURATOR_1, MAIN_1)
            b.tasks["A1"].add_done_callback(lambda _t: b.stamp("delivered", "A1"))
            assert b.mgr._is_designated_unload_target_locked(a) is True, (
                "A is not the designated target of main's claim; the hold below would be vacuous")
            await b.after_victim_searches(3, "three victim searches while A's turn is held")
            assert not any(b.seen(k, n) for k, n in (
                ("unload", TAG_A), ("engine_stop", "sigterm"), ("turn_end", "A1"),
                ("delivered", "A1"), ("load", TAG_B))), (
                f"something past the held turn happened while the latch was closed; events={b.rel()}")
            b.holds["A1"].set()
            await b.until(lambda: b.seen("turn_end", "M1"), "main's turn to complete", 12.0)
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)
            names = [(k, n) for k, n in b.order() if k != "audit"]
            print("held-turn events", b.rel())
            wanted = [("turn_end", "A1"), ("delivered", "A1"), ("unload", TAG_A),
                      ("engine_stop", "sigterm"), ("load", TAG_B), ("turn_end", "M1")]
            positions = []
            for item in wanted:
                assert item in names, f"missing event {item}; events={b.rel()}"
                positions.append(names.index(item))
            assert positions == sorted(positions) and len(set(positions)) == len(positions), (
                f"events happened in the wrong order; events={b.rel()}")
            assert [n for k, n in names if k == "unload"][0] == TAG_A


@pytest.mark.asyncio
class TestRankIsNeverComparedAcrossClients:
    async def _cross(self, b):
        """Client 1 (address IP_1) holds resident A (curator, rank 3) mid-turn; client 2
        (address IP_2, main, rank 1) asks for the other model. Returns A's resident."""
        return await _held_a_main_waiting(b, CURATOR_1, MAIN_2, "th-y")

    async def test_better_tag_rank_of_a_lower_client_does_not_unload_the_first_client(
            self, tmp_path, monkeypatch):
        """Unchanged cross-client behaviour. Client 1 (rule position 0) holds A with a
        worse tag (curator, rank 3) mid-turn; client 2 (rule position 1) asks for the
        other model with a better tag (main, rank 1). Client order decides: A is not
        designated, is not unloaded, finishes its turn, and client 2's request only runs
        after A has left on its own (A's grace window ends) -- never because of rank."""
        async with box(tmp_path, monkeypatch, rules=CLIENT_1_FIRST, grace_s=2) as b:
            a = await self._cross(b)
            assert b.mgr._is_designated_unload_target_locked(a) is False, (
                "client 1's resident was designated for client 2's better-ranked tag")
            await b.after_victim_searches(3, "three victim searches while A is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B)), (
                f"A unloaded or client 2's model loaded while A is held; events={b.rel()}")
            b.holds["A1"].set()
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            await b.until(lambda: b.seen("turn_end", "M1"), "client 2's request to run", 14.0)
            print(f"client1-first events={b.rel()}")
            ev = b.order()
            assert ev.index(("turn_end", "A1")) < ev.index(("audit", "A1:grace_enter")) < ev.index(
                ("unload", TAG_A)) < ev.index(("load", TAG_B)), (
                f"A did not serve its grace window before leaving; events={b.rel()}")
            assert ("audit", "A1:grace_designated_victim_skip") not in ev, (
                f"A's grace was cut for a lower client's better tag; events={b.rel()}")
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)

    async def test_control_swapped_client_order_unloads_promptly(
            self, tmp_path, monkeypatch):
        """Control for the control: same two requests but client 2 is listed FIRST. Now the
        waiting request belongs to the better client, so A is designated, loses its grace,
        and the waiting request runs promptly after A's turn. A rank-only comparison could
        not tell this arm from the previous one; client order does."""
        async with box(tmp_path, monkeypatch, rules=CLIENT_2_FIRST, grace_s=2) as b:
            a = await self._cross(b)     # A: client 1 (listed second), claim: client 2
            assert b.mgr._is_designated_unload_target_locked(a) is True, (
                "the better client's claim did not designate the lower client's resident")
            await b.after_victim_searches(3, "three victim searches while A is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B)), (
                f"A unloaded or client 2's model loaded while A is held; events={b.rel()}")
            b.holds["A1"].set()
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            await b.until(lambda: b.seen("turn_end", "M1"), "client 2's request to run", 14.0)
            print(f"client2-first events={b.rel()}")
            ev = b.order()
            assert ev.index(("turn_end", "A1")) < ev.index(("unload", TAG_A)) < ev.index(("load", TAG_B))
            assert ("audit", "A1:grace_designated_victim_skip") in ev and (
                "audit", "A1:grace_enter") not in ev, (
                f"A was not stripped of its grace window by the better client; events={b.rel()}")
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)


    async def test_worse_tag_of_a_better_client_still_designates_the_lower_clients_best_tag(
            self, tmp_path, monkeypatch):
        """Converse of the first test: client 1 is listed first. Client 2 holds A with its BEST
        tag (main, rank 1) mid-turn; client 1 asks for the other model with its WORST tag
        (curator, rank 3). Client order decides: A is designated, loses its grace window,
        and is unloaded only after its held turn ended; client 1's request then runs."""
        async with box(tmp_path, monkeypatch, rules=CLIENT_1_FIRST, grace_s=2) as b:
            a = await _held_a_main_waiting(b, MAIN_2, CURATOR_1, "th-y")
            assert b.mgr._is_designated_unload_target_locked(a) is True, (
                "a better client's worse tag did not designate the lower client's best-tag resident")
            await b.after_victim_searches(3, "three victim searches while A is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B)), (
                f"A unloaded or the claimant's model loaded while A is held; events={b.rel()}")
            b.holds["A1"].set()
            _assert_completed(await _result(b, "A1"), "A1", TAG_A)
            await b.until(lambda: b.seen("turn_end", "M1"), "client 1's request to run", 14.0)
            ev = b.order()
            assert ev.index(("turn_end", "A1")) < ev.index(("unload", TAG_A)) < ev.index(("load", TAG_B)), (
                f"events={b.rel()}")
            assert ("audit", "A1:grace_designated_victim_skip") in ev and (
                "audit", "A1:grace_enter") not in ev, (
                f"A was not stripped of its grace window by the better client; events={b.rel()}")
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)


@pytest.mark.asyncio
class TestUnloadKeyPicksTheWorseRankedIdleResident:
    @pytest.mark.parametrize("curator_is_newer", [True, False])
    async def test_worse_rank_idle_resident_is_unloaded_first(
            self, tmp_path, monkeypatch, curator_is_newer):
        """One client, two idle residents (A curator rank 3, C main rank 1) on a card that
        holds two models; a main request for a third model needs one unloaded. Expected:
        the curator resident goes first whichever of the two was active more recently
        (rank, not recency, orders the pick)."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, card_mib=48000, cap=3) as b:
            for label, tag, meta in (("A1", TAG_A, CURATOR_1), ("C1", TAG_C, MAIN_1)):
                _assert_completed(await asyncio.wait_for(b.submit(label, tag, meta, "th-" + label), 10),
                                  label, tag)
                await b.until(lambda t=tag: resident_for(b.mgr, t).state is ResidentState.IDLE_EVICTABLE,
                              f"{tag} to park idle")
            assert b.loaded_tags() == {TAG_A, TAG_C}
            assert b.one_model_fits(TAG_B) is False, "box is not full with two residents"
            old, new = (TAG_C, TAG_A) if curator_is_newer else (TAG_A, TAG_C)
            resident_for(b.mgr, old).last_active_monotonic = 1.0
            resident_for(b.mgr, new).last_active_monotonic = time.monotonic()
            b.submit("M1", TAG_B, MAIN_1, "th-main")
            await b.until(lambda: b.seen("turn_end", "M1"), "main to complete", 10.0)
            unloaded = [n for k, n in b.order() if k == "unload"]
            print(f"curator_is_newer={curator_is_newer} events={b.rel()}")
            assert unloaded and unloaded[0] == TAG_A, (
                f"first unloaded model was {unloaded[:1]}; events={b.rel()}")
            assert TAG_C in b.loaded_tags(), "the main resident was unloaded too"


async def _second_turn_held_then_claim(b, a_meta, claim_meta, *, hold=True):
    """A serves a first turn, passes through a whole grace window (the window ends and A
    parks), and then serves a SECOND turn that is HELD mid-generation (`hold`) or runs to
    completion. A request for the other model then arrives on the one-model box. Asserts the
    setup: A's grace was started and has ended, so the unload fallback for a resident that
    left its grace window is the code under test, not the first-turn path."""
    b.submit("A1", TAG_A, a_meta, "th-a")
    await b.until(lambda: b.seen("turn_end", "A1"), "A's first turn to end")
    a = resident_for(b.mgr, TAG_A)
    await b.until(lambda: a.grace is not None and a.grace._started_at is not None,
                  "A's grace window to start")
    await b.until(lambda: a.in_grace_loop, "A to be inside its grace loop")
    await b.until(lambda: (not a.in_grace_loop) and a.state is ResidentState.IDLE_EVICTABLE,
                  "A's grace window to end", 12.0)
    assert a.grace._started_at is not None, "A's grace start was cleared"
    if hold:
        b.submit("A2", TAG_A, a_meta, "th-a", hold=True)
        await b.until(lambda: b.seen("turn_start", "A2"), "A's second turn to start")
        assert a.state is ResidentState.ACTIVE, f"A is not mid-turn: {a.state}"
        assert a.active_slot is not None and not b.tasks["A2"].done()
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model: box is not full"
    b.stamp("claim_submit", "M1")
    b.submit("M1", TAG_B, claim_meta, "th-main")
    await b.until(lambda: len(b.mgr._fastlane_claims) > 0, "main's claim to register")
    return a


@pytest.mark.asyncio
class TestSecondTurnOfAReusedResidentIsNotCut:
    """A resident that already left one grace window and is now serving a SECOND turn is
    reached by the unload fallback that accepts a non-idle resident. Inside one client a
    better tag rank must still wait for that running turn to end."""

    async def test_second_turn_held_is_not_cut_by_better_rank_of_the_same_client(
            self, tmp_path, monkeypatch):
        """One client, a 1 s grace window. A (curator, rank 3) serves a turn, its grace window
        ends, and a second turn is HELD mid-generation. A main (rank 1) request for the other
        model arrives. Expected: A is the designated target (checked) yet for longer than
        three dispatcher passes it is not unloaded, its engine is not stopped and the held
        turn stays running; after release the response is complete, and only then is A
        unloaded, then main's model loads and main completes."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=1) as b:
            a = await _second_turn_held_then_claim(b, CURATOR_1, MAIN_1)
            if not b.seen("unload", TAG_A):    # a cut A is no longer a target: the hold below reports it
                assert b.mgr._is_designated_unload_target_locked(a) is True, (
                    "A is not the designated target of main's claim; the hold below would be vacuous")
            n0 = len(b.passes)
            await b.after_victim_searches(3, "three victim searches while A's second turn is held")
            assert not (b.seen("unload", TAG_A) or b.seen("engine_stop", "sigterm")
                        or a.state is ResidentState.DEAD or b.seen("turn_end", "A2")
                        or b.seen("load", TAG_B) or b.tasks["A2"].done()), (
                f"A unloaded, stopped, ended or B loaded while A's second turn is held; "
                f"events={b.rel()}")
            assert b.passes[n0:] == [None] * (len(b.passes) - n0), (
                f"the victim search returned {b.passes[n0:]} while A was running")
            b.holds["A2"].set()
            _assert_completed(await _result(b, "A2"), "A2", TAG_A)
            await b.until(lambda: b.seen("turn_end", "M1"), "main's turn to complete", 12.0)
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)
            t = {k: b.first(*k) for k in (("turn_end", "A2"), ("unload", TAG_A),
                                          ("load", TAG_B), ("turn_end", "M1"))}
            print("second-turn-held events", b.rel())
            assert None not in t.values(), f"missing events {t}; events={b.rel()}"
            assert (t[("turn_end", "A2")] < t[("unload", TAG_A)] < t[("load", TAG_B)]
                    < t[("turn_end", "M1")]), f"wrong order; events={b.rel()}"
            assert [n for k, n in b.order() if k == "unload"][0] == TAG_A


    async def test_control_idle_worse_rank_resident_is_unloaded_promptly(
            self, tmp_path, monkeypatch):
        """Control: the same client, grace and ranks as the second-turn-held case, but A's second turn is NOT
        running: A finished its first turn and its grace window ended, so it is idle.
        Expected: the main (rank 1) request for the other model is not made to wait: A is
        unloaded (its engine stopped) promptly, then main's model loads and main completes.
        The guard does not make the unload inert for a resident that is not mid-turn."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=1) as b:
            a = await _second_turn_held_then_claim(b, CURATOR_1, MAIN_1, hold=False)
            await b.until(lambda: b.seen("turn_end", "M1"), "main's turn to complete", 10.0)
            t_claim = b.first("claim_submit", "M1")
            print("idle-control events", b.rel())
            _assert_completed(await _result(b, "M1"), "M1", TAG_B)
            assert b.first("unload", TAG_A) is not None, f"A was never unloaded; events={b.rel()}"
            assert b.first("unload", TAG_A) - t_claim < 3.0, (
                f"A was unloaded {b.first('unload', TAG_A) - t_claim:.2f}s after the claim; events={b.rel()}")
            assert (b.first("unload", TAG_A) <= b.first("engine_stop", "sigterm")
                    < b.first("load", TAG_B) < b.first("turn_end", "M1")), f"events={b.rel()}"
            assert [n for k, n in b.order() if k == "unload"][0] == TAG_A


@pytest.mark.asyncio
class TestWorseRequestFirstThenBetterRequestOfOneModel:
    async def test_the_better_request_governs_and_designates_the_resident_at_its_boundary(
            self, tmp_path, monkeypatch):
        """One client. A (curator, rank 3) has a turn held. Two requests for the other model
        arrive, the worse one first (unclassified, rank 5), then the better one (main, rank 1).
        The one claim of that (client, model) is upgraded to the better request, so A is the
        designated target: nothing is unloaded while its turn is held, then A's turn ends, its
        grace window is skipped, A is unloaded, the model loads and the better request runs
        before the worse one."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=30) as b:
            b.submit("A1", TAG_A, CURATOR_1, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "A1"), "A's turn to start")
            a = resident_for(b.mgr, TAG_A)
            b.submit("W1", TAG_B, UNCLASSIFIED_1, "th-w")
            await b.until(lambda: len(b.mgr._fastlane_claims) == 1, "the worse request's claim")
            b.submit("M1", TAG_B, MAIN_1, "th-m")
            key = (IP_1, TAG_B)
            await b.until(lambda: b.mgr._fastlane_claims[key]["slot"].prompt == "M1",
                          "the better request to take the claim over")
            assert len(b.mgr._fastlane_claims) == 1
            assert b.mgr._governing_claim_priority_key_locked() == (0, 1)
            assert b.mgr._is_designated_unload_target_locked(a) is True, (
                "the better request's claim did not designate A")
            await b.after_victim_searches(3, "three victim searches while A's turn is held")
            assert not (b.seen("unload", TAG_A) or b.seen("turn_end", "A1") or b.seen("load", TAG_B)), (
                f"something past the held turn happened while the latch was closed; events={b.rel()}")
            b.holds["A1"].set()
            await b.until(lambda: b.seen("turn_start", "M1"), "the better request to run", 15.0)
            # The better request has started its turn: the worse one that still waits keeps
            # a claim (the shared one, or its own once it is parked in the loaded model's
            # inbox), and no claim points at the request that already started.
            def holders():
                return [c["slot"].prompt for c in b.mgr._fastlane_claims.values()]
            await b.until(lambda: holders() == ["W1"],
                          "the worse request that still waits to be the only claim holder")
            assert not b.seen("turn_start", "W1"), f"the worse request ran first; events={b.rel()}"
            ev = b.order()
            print("worse-first events", b.rel())
            assert ev.index(("turn_end", "A1")) < ev.index(("audit", "A1:grace_designated_victim_skip")) \
                < ev.index(("unload", TAG_A)) < ev.index(("load", TAG_B)) \
                < ev.index(("turn_start", "M1")), f"wrong order; events={b.rel()}"
            assert ("audit", "A1:grace_enter") not in ev, f"A sat in grace; events={b.rel()}"


async def _grace_after_main_turn(b):
    """A runs a main turn (th-a) to its end and sits in its grace window."""
    b.submit("A1", TAG_A, MAIN_1, "th-a")
    await b.until(lambda: b.seen("turn_end", "A1"), "A's first turn to end")
    a = resident_for(b.mgr, TAG_A)
    await b.until(lambda: a.in_grace_loop, "A to be inside its grace window")
    return a


@pytest.mark.asyncio
class TestResidentRankFollowsItsLatestTurn:
    async def test_main_turn_then_unclassified_follow_up_in_grace_makes_it_a_rank_five_resident(
            self, tmp_path, monkeypatch):
        """One client. A's first turn is main (rank 1); a same-thread follow-up of NO class
        (rank 5) is served inside the grace window (the warm path) and held. While it runs the
        resident still counts as it did, a curator (rank 3) request for the other model
        waits and nothing is cut. After the follow-up ends the resident's rank is the
        follow-up's, so the rank-3 request designates it: A's grace window is ended, it is
        unloaded after the follow-up's turn, and the other model loads."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=30) as b:
            a = await _grace_after_main_turn(b)
            b.submit("FUP", TAG_A, UNCLASSIFIED_1, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "FUP"), "the follow-up's turn to start")
            assert a.state is ResidentState.ACTIVE and a.active_slot is not None
            b.submit("C1", TAG_B, CURATOR_1, "th-c")
            await b.until(lambda: len(b.mgr._fastlane_claims) == 1, "the curator request's claim")
            await b.after_victim_searches(3, "three victim searches while the follow-up is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B) or b.seen("turn_end", "FUP")), (
                f"something past the held follow-up happened; events={b.rel()}")
            b.holds["FUP"].set()
            await b.until(lambda: b.seen("turn_end", "C1"), "the curator request to run", 15.0)
            ev = b.order()
            print("rank-five events", b.rel())
            assert ("audit", "FUP:active_match_completed") in ev, "the follow-up did not ride the warm path"
            assert ("audit", "A1:grace_designated_unload_target_break") in ev, (
                f"the resident was not designated after its follow-up; events={b.rel()}")
            assert ev.index(("turn_end", "FUP")) < ev.index(("unload", TAG_A)) < ev.index(("load", TAG_B)) \
                < ev.index(("turn_end", "C1")), f"wrong order; events={b.rel()}"

    async def test_outrank_gate_and_victim_search_name_the_same_resident_after_a_follow_up(
            self, tmp_path, monkeypatch):
        """Same setup without a claimant in flight; after the grace window ends and the
        resident parks idle, a rank-3 claimant of the same client is outranked by it as
        a rank-5 resident: the outrank gate and the victim search both name that resident,
        and its designation key and eviction key carry the same rank."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=2) as b:
            a = await _grace_after_main_turn(b)
            b.submit("FUP", TAG_A, UNCLASSIFIED_1, "th-a")
            await b.until(lambda: b.seen("audit", "FUP:active_match_completed"), "the warm follow-up")
            await b.until(lambda: a.state is ResidentState.IDLE_EVICTABLE and not a.in_grace_loop,
                          "A to park idle", 15.0)
            table = b.mgr._fastlane_table()
            assert b.mgr._resident_priority_key(a, table) == (0, 5)
            assert b.mgr._resident_unload_priority_key(a, table)[:3] == (1, 0, -5)
            claimant = Slot.new("claimant")
            claimant.fastlane = FastLaneMatch(
                rule_index=0, raw_address=IP_1, label="", effective_tag="curator", rank=3)
            gate = b.mgr._fastlane_outrank_unload_target_locked(claimant, "none")
            search = b.mgr._lru_idle_unloadable(split_mode="none")
            assert gate is a and search is a, (
                f"gate named {getattr(gate, 'model_tag', None)!r}, search named "
                f"{getattr(search, 'model_tag', None)!r}; both must name A")

    async def test_unclassified_turn_then_main_follow_up_protects_the_resident_as_main(
            self, tmp_path, monkeypatch):
        """One client. A's first turn has no class (rank 5); a same-thread MAIN follow-up
        (rank 1) arrives inside the grace window and is held. A rank-3 request for the other
        model then waits: the resident now counts as main, so it is not designated, nothing
        is cut, and after the follow-up ends A serves its whole grace window before it is
        unloaded."""
        async with box(tmp_path, monkeypatch, rules=ONE_CLIENT, grace_s=2) as b:
            b.submit("A1", TAG_A, UNCLASSIFIED_1, "th-a")
            await b.until(lambda: b.seen("turn_end", "A1"), "A's first turn to end")
            a = resident_for(b.mgr, TAG_A)
            await b.until(lambda: a.in_grace_loop, "A to be inside its grace window")
            b.submit("FUP", TAG_A, MAIN_1, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "FUP"), "the follow-up's turn to start")
            assert assert_resolves(b.mgr, TAG_A) == (0, 1), "the resident must count as main"
            b.submit("C1", TAG_B, CURATOR_1, "th-c")
            await b.until(lambda: len(b.mgr._fastlane_claims) >= 1, "the curator request's claim")
            assert b.mgr._is_designated_unload_target_locked(a) is False, (
                "a rank-3 claim designated a resident whose latest turn is main")
            await b.after_victim_searches(3, "three victim searches while the follow-up is held")
            assert not (b.seen("unload", TAG_A) or b.seen("load", TAG_B) or b.seen("turn_end", "FUP")), (
                f"something past the held follow-up happened; events={b.rel()}")
            b.holds["FUP"].set()
            await b.until(lambda: b.seen("turn_end", "C1"), "the curator request to run", 15.0)
            ev = b.order()
            print("main-protected events", b.rel())
            assert ev.index(("turn_end", "FUP")) < ev.index(("audit", "FUP:grace_enter")) \
                < ev.index(("unload", TAG_A)) < ev.index(("load", TAG_B)), f"events={b.rel()}"
            assert not any(n.startswith("FUP:") and n.endswith(
                ("grace_designated_victim_skip", "grace_designated_unload_target_break"))
                for k, n in ev if k == "audit"), (
                f"the resident was stripped of the grace window of its latest turn; events={b.rel()}")
