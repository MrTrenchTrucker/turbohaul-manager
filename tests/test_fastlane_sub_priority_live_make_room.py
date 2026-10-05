"""Live make-room ordering for ONE Fast Lane client with two class labels.

One rule (one address) ranks its classes: main = 1, curator = 3 (lower is
served first). The client sends two kinds of request from that address, each
needing a DIFFERENT model, and the box holds only ONE model (two models on one
card, a card that fits one 20000 MiB footprint). Everything runs through the
real manager: `submit_and_wait`, the worker loop, `_dispatch_loop`,
`_route_or_reserve`, make-room, `_is_designated_unload_target_locked` and the
per-resident driver. Only the sidecar process, the health probe and the
completion call are faked, and the nvidia-smi free-VRAM probe is replaced by a
card model computed from the manager's own resident table.

Expected direction asserted throughout: within ONE client the rank-1 (main)
request's model is not held out by the rank-3 (curator) request's model; a
worse-ranked resident of the client is unloaded before its better-ranked one; a
running turn is never cut; across clients rank is never compared (rule_index
decides).

Deciders (src/turbohaul/manager.py): `_is_designated_unload_target_locked`
(compares (rule_index, rank) of the claim and the resident, so within one
client the better rank outranks the worse), the make-room branch of
`_route_or_reserve`, `_lru_idle_unloadable` (picks by
`_resident_unload_priority_key`, which orders by rank inside one client),
`_unload_priority_key_for_meta`, `_vram_admits_locked`.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import pytest
import yaml

from _fastlane_fixture import (
    assert_resolves, boot_ranked_runtime, drive_to_active, make_fakes,
    resident_for, wait_until,
)
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager

IP_A = "192.0.2.10"        # rule 0: the client under test
IP_B = "198.51.100.20"     # rule 1: a lower-priority listed client
FOOTPRINT_MIB = 20000
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)

TAG_MAIN, TAG_CURATOR, TAG_OTHER, TAG_MAIN_2 = "model-main", "model-curator", "model-other", "model-main-2"

MAIN_A = {"ip": IP_A, "is_main": True}
CURATOR_A = {"ip": IP_A, "is_curator": True, "is_sub_agent": True}
MAIN_B = {"ip": IP_B, "is_main": True}

RULES = [
    FastLaneRule(address=IP_A, tag_ranks=FastLaneTagRanks(main=1, curator=3)),
    FastLaneRule(address=IP_B, tag_ranks=FastLaneTagRanks(main=1, curator=3)),
]


@dataclass
class Box:
    mgr: TurbohaulManager
    holds: dict
    events: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    searches: list = field(default_factory=list)   # answer of every victim search the real code made

    def submit(self, tag, meta, thread):
        t = asyncio.create_task(
            self.mgr.submit_and_wait(tag, "p", thread_id=thread, client_meta=meta))
        self.tasks.append(t)
        return t

    def started(self, kind, tag):
        return (kind, tag) in self.events

    async def seen(self, kind, tag, timeout):
        """True if the event shows up within `timeout` seconds (never raises)."""
        with contextlib.suppress(AssertionError):
            await wait_until(lambda: self.started(kind, tag), timeout=timeout)
        return self.started(kind, tag)

    async def claims_registered(self, n, timeout=8.0):
        """Wait until the dispatcher has deferred `n` requests and registered their claims."""
        await wait_until(lambda: len(self.mgr._fastlane_claims) >= n, timeout=timeout)

    async def after_victim_searches(self, n, timeout=20.0):
        """Wait until the real make-room code has searched for a victim `n` more times; what
        is asserted afterwards is decided by what that code did, not by how long was waited."""
        target = len(self.searches) + n
        await wait_until(lambda: len(self.searches) >= target, timeout=timeout)

    def one_model_fits(self, tag):
        """The real capacity check's answer for `tag` against the current residents."""
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
async def box(tmp_path, monkeypatch, *, tags, card_mib=24000, cap=2, grace_s=0,
              hold=()):
    """A real manager over a card of `card_mib`; each loaded model takes 20000 MiB.

    `hold` names tags whose turn blocks until its hold is set. Events record
    spawn / serve-start / unload in the order the real code produced them.
    """
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=cap, grace_seconds=grace_s,
        max_grace_extensions=5 if grace_s else 0,
        idle_hot_load_seconds=120)
    for t in tags:
        _manifest(boot, t)
    holds = {t: asyncio.Event() for t in hold}
    spawn, health, sigterm, vram, complete = make_fakes(holds)
    holder = {}
    events: list = []

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        events.append(("spawn", model_tag))
        return spawn(binary, gguf, port, model_tag, argv, **kw)

    async def complete_rec(slot, handle):
        events.append(("serve", handle.model_tag))
        return await complete(slot, handle)

    def free_vram():
        m = holder["mgr"]
        used = FOOTPRINT_MIB * sum(1 for r in m._model_residents() if r.state in LOADED)
        return [card_mib - used]

    # Free VRAM: the capacity decision reads the manager's own binding, so BOTH the
    # manager's and safety's `_read_free_vram_all_mib` are pinned to the card model.
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm, vram_fn=vram, complete_fn=complete_rec)
    holder["mgr"] = mgr
    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):  # pass-through: records WHO the real make-room named
        events.append(("unload", r.model_tag))
        return orig_unload(r)

    mgr._begin_unload_locked = unload_spy
    orig_search = mgr._lru_idle_unloadable

    def search_spy(*a, **k):  # pass-through: records every victim search and its answer
        out = orig_search(*a, **k)
        b.searches.append(getattr(out, "model_tag", None))
        return out

    mgr._lru_idle_unloadable = search_spy
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    b = Box(mgr=mgr, holds=holds, events=events)
    try:
        yield b
    finally:
        for g in holds.values():
            g.set()
        for t in b.tasks:
            t.cancel()
        # Same order as shutdown(): set the stop flag BEFORE cancelling the
        # dispatcher. A cancel that lands in the same loop iteration as an
        # event wake can be swallowed by the bounded wait inside the dispatch
        # loop; with the flag already set the loop then ends within one poll
        # instead of running forever. The wait is bounded so that any future
        # hang fails loudly here instead of hanging the run.
        mgr._stop_event.set()
        mgr._worker_task.cancel()
        done, _pending = await asyncio.wait({mgr._worker_task}, timeout=10)
        assert done, "teardown: the manager worker task did not finish within 10s of stop_event + cancel"
        with contextlib.suppress(asyncio.CancelledError, Exception):
            mgr._worker_task.result()
        await mgr.shutdown()


async def _park_idle(b, tag, meta, thread):
    """Run one request to completion so the resident parks IDLE_EVICTABLE."""
    await asyncio.wait_for(b.submit(tag, meta, thread), timeout=5.0)
    await wait_until(lambda: resident_for(b.mgr, tag).state is ResidentState.IDLE_EVICTABLE)


async def _curator_active_main_waiting(b):
    """Curator's model ACTIVE mid-turn, then a main request for the other model."""
    cur = await drive_to_active(b.mgr, TAG_CURATOR, thread_id="cur", client_meta=CURATOR_A)
    assert assert_resolves(b.mgr, TAG_CURATOR) == (0, 3), "curator resident must resolve to rule 0, rank 3"
    assert b.one_model_fits(TAG_MAIN) is False, "capacity check admits a second model: box is not full"
    main = b.submit(TAG_MAIN, MAIN_A, "main")
    await b.claims_registered(1)  # the dispatcher routed and deferred it
    await b.after_victim_searches(3)
    assert not b.started("spawn", TAG_MAIN), (
        f"main model co-resided with the busy curator model: {b.events}")
    return cur, main


@pytest.mark.asyncio
class TestCuratorMidTurnMainArrives:
    async def test_main_claim_designates_the_busy_curator_resident(self, tmp_path, monkeypatch):
        """Expects the rank-1 claim to make the rank-3 resident of the same rule the designated victim (the designation compares (rule_index, rank), so the worse rank of one client loses to the better one). The resident is mid-turn, so it is only designated, not cut."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR], grace_s=120,
                       hold=[TAG_CURATOR]) as b:
            await _curator_active_main_waiting(b)
            designated = b.mgr._is_designated_unload_target_locked(resident_for(b.mgr, TAG_CURATOR))
            assert designated is True, (
                f"main (rank 1) is waiting on a full box and the curator (rank 3) resident of the same "
                f"rule is not designated; events={b.events}")

    async def test_main_runs_promptly_once_the_curator_turn_ends(self, tmp_path, monkeypatch):
        """Expects the main model to be spawned and served within 4 s of the curator turn ending (the designation, then the make-room branch), not after the curator resident's 120 s grace window."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR], grace_s=120,
                       hold=[TAG_CURATOR]) as b:
            await _curator_active_main_waiting(b)
            b.holds[TAG_CURATOR].set()
            ran = await b.seen("serve", TAG_MAIN, timeout=4.0)
            assert ran, f"main never ran within 4 s of the curator turn ending; events={b.events}"

    async def test_control_rule_index_designates_the_busy_resident(self, tmp_path, monkeypatch):
        """Control: a rule-0 main claim against a busy rule-1 resident of EQUAL rank designates it, and main then runs within 4 s (the designation sees rule_index)."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR], grace_s=120,
                       hold=[TAG_CURATOR]) as b:
            await drive_to_active(b.mgr, TAG_CURATOR, thread_id="res", client_meta=MAIN_B)
            assert assert_resolves(b.mgr, TAG_CURATOR) == (1, 1)
            assert b.one_model_fits(TAG_MAIN) is False
            b.submit(TAG_MAIN, MAIN_A, "main")
            await b.claims_registered(1)
            await b.after_victim_searches(3)
            assert b.mgr._is_designated_unload_target_locked(resident_for(b.mgr, TAG_CURATOR)) is True, (
                f"rule 0 claim did not designate the rule 1 resident; events={b.events}")
            b.holds[TAG_CURATOR].set()
            assert await b.seen("serve", TAG_MAIN, timeout=4.0), f"claimant never ran; events={b.events}"

    async def test_control_main_resident_makes_curator_claim_wait(self, tmp_path, monkeypatch):
        """Control: with the main (rank 1) model busy, a curator (rank 3) claim of the same rule waits and is not served within 3 s of the turn ending (equal (rule_index, rank) keys do not designate). Proves the harness can show a wait."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR], grace_s=120,
                       hold=[TAG_MAIN]) as b:
            await drive_to_active(b.mgr, TAG_MAIN, thread_id="main", client_meta=MAIN_A)
            assert assert_resolves(b.mgr, TAG_MAIN) == (0, 1)
            assert b.one_model_fits(TAG_CURATOR) is False
            b.submit(TAG_CURATOR, CURATOR_A, "cur")
            await b.claims_registered(1)
            await b.after_victim_searches(3)
            assert not b.started("spawn", TAG_CURATOR), f"curator co-resided: {b.events}"
            b.holds[TAG_MAIN].set()
            main_resident = resident_for(b.mgr, TAG_MAIN)
            await wait_until(lambda: main_resident.in_grace_loop, timeout=5.0)
            await b.after_victim_searches(3)
            assert not b.started("serve", TAG_CURATOR), (
                f"curator was served right after the main turn; events={b.events}")
            assert not b.started("unload", TAG_MAIN), (
                f"the main resident was unloaded for a worse-ranked claim; events={b.events}")


@pytest.mark.asyncio
class TestCuratorIdleMainArrives:
    @pytest.mark.parametrize("age", ["older", "newer"])
    async def test_main_evicts_the_idle_curator_resident_whatever_its_recency(
            self, tmp_path, monkeypatch, age):
        """Expects the idle rank-3 resident to be unloaded and main served, with last_active older or newer (the only candidate); with one resident recency has nothing to decide."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR]) as b:
            await _park_idle(b, TAG_CURATOR, CURATOR_A, "cur")
            assert assert_resolves(b.mgr, TAG_CURATOR) == (0, 3)
            resident_for(b.mgr, TAG_CURATOR).last_active_monotonic = (
                1.0 if age == "older" else time.monotonic() + 1000.0)
            assert b.one_model_fits(TAG_MAIN) is False
            b.submit(TAG_MAIN, MAIN_A, "main")
            assert await b.seen("serve", TAG_MAIN, timeout=6.0), f"main never ran; events={b.events}"
            assert b.events.index(("unload", TAG_CURATOR)) < b.events.index(("spawn", TAG_MAIN))

    @pytest.mark.parametrize("older", ["curator", "main"])
    async def test_two_idle_same_rule_residents_evict_the_curator_first(
            self, tmp_path, monkeypatch, older):
        """Variant, the card holds two models: main (rank 1) and curator (rank 3) residents of one rule are idle and a third model is claimed by main; expects the curator resident unloaded whichever is older (the unload key orders by tier, rule_index, rank, then recency, so rank decides before recency)."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_MAIN_2, TAG_CURATOR], card_mib=48000,
                       cap=3) as b:
            await _park_idle(b, TAG_CURATOR, CURATOR_A, "cur")
            await _park_idle(b, TAG_MAIN_2, MAIN_A, "main2")
            assert assert_resolves(b.mgr, TAG_CURATOR) == (0, 3)
            assert assert_resolves(b.mgr, TAG_MAIN_2) == (0, 1)
            old, new = (TAG_CURATOR, TAG_MAIN_2) if older == "curator" else (TAG_MAIN_2, TAG_CURATOR)
            resident_for(b.mgr, old).last_active_monotonic = 1.0
            resident_for(b.mgr, new).last_active_monotonic = time.monotonic()
            assert b.one_model_fits(TAG_MAIN) is False, "box is not full with two residents"
            b.submit(TAG_MAIN, MAIN_A, "main")
            assert await b.seen("serve", TAG_MAIN, timeout=6.0), f"main never ran; events={b.events}"
            named = [t for k, t in b.events if k == "unload"]
            assert named and named[0] == TAG_CURATOR, (
                f"first unloaded model was {named[:1]} ({older} resident was older); the rank-3 "
                f"curator resident should go first; events={b.events}")


@pytest.mark.asyncio
class TestBothWaitingWhileAnUnrelatedClientHoldsTheBox:
    async def test_main_model_loads_before_curator_model_after_holder_releases(
            self, tmp_path, monkeypatch):
        """Expects the main model to load first even though the curator request arrived first (queue pick by (rule_index, rank, arrival) plus make-room): the better rank goes first within one client even when it arrived later."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR, TAG_OTHER],
                       grace_s=120, hold=[TAG_OTHER, TAG_MAIN, TAG_CURATOR]) as b:
            await drive_to_active(b.mgr, TAG_OTHER, thread_id="holder", client_meta=MAIN_B)
            assert assert_resolves(b.mgr, TAG_OTHER) == (1, 1)
            assert b.one_model_fits(TAG_MAIN) is False and b.one_model_fits(TAG_CURATOR) is False
            b.submit(TAG_CURATOR, CURATOR_A, "cur")   # arrives first
            await b.claims_registered(1)
            b.submit(TAG_MAIN, MAIN_A, "main")        # arrives second, better rank
            await b.claims_registered(2)
            await b.after_victim_searches(3)
            assert not (b.started("spawn", TAG_MAIN) or b.started("spawn", TAG_CURATOR)), (
                f"a waiting model co-resided with the holder: {b.events}")
            b.holds[TAG_OTHER].set()
            await wait_until(lambda: b.started("spawn", TAG_MAIN) or b.started("spawn", TAG_CURATOR),
                             timeout=8.0)
            first = TAG_MAIN if b.started("spawn", TAG_MAIN) else TAG_CURATOR
            assert first == TAG_MAIN, f"first model loaded was {first}; events={b.events}"

    async def test_control_arrival_order_is_visible_for_equal_requests(self, tmp_path, monkeypatch):
        """Control: two EQUAL (rule, rank) main requests needing different models load in arrival order (queue pick breaks ties by arrival), so a green above is not blind to ordering."""
        async with box(tmp_path, monkeypatch, tags=[TAG_MAIN, TAG_CURATOR, TAG_OTHER],
                       grace_s=120, hold=[TAG_OTHER, TAG_MAIN, TAG_CURATOR]) as b:
            await drive_to_active(b.mgr, TAG_OTHER, thread_id="holder", client_meta=MAIN_B)
            b.submit(TAG_CURATOR, MAIN_A, "first")
            await b.claims_registered(1)
            b.submit(TAG_MAIN, MAIN_A, "second")
            await b.claims_registered(2)
            await b.after_victim_searches(3)
            b.holds[TAG_OTHER].set()
            await wait_until(lambda: b.started("spawn", TAG_MAIN) or b.started("spawn", TAG_CURATOR),
                             timeout=8.0)
            first = TAG_MAIN if b.started("spawn", TAG_MAIN) else TAG_CURATOR
            assert first == TAG_CURATOR, f"first model loaded was {first}; events={b.events}"
