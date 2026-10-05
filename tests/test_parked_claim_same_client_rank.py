"""A request parked in a busy resident's inbox ends that resident's wait for
follow-ups only when it outranks the request the resident just served.

Rule under test: client order (rule_index) decides first; tag rank is compared
only between a parked request and a holder of the SAME client. So a better tag of
the holder's own client that is parked in the inbox is admitted at the turn
boundary (the holder's grace window is skipped or ended, and the holder's KV is
saved before the swap), an equal tag is not, and a better tag of ANOTHER client
never outranks a better client's holder, while a better client outranks whatever
tag its holder served.

The real manager runs end to end (submit, dispatcher, resident driver, grace
loop); only the sidecar process, health probe, teardown and the completion call
are faked. The KV save at the swap is replaced by a recorder (it is the one
place that save is decided, and the fake engine has nothing to save).

Free VRAM: one model and one resident per test, so no path here reads free VRAM
to decide anything; both `turbohaul.safety._read_free_vram_all_mib` and
`turbohaul.manager._read_free_vram_all_mib` are pinned anyway.
"""
import asyncio
import contextlib
from unittest.mock import patch

import pytest

from _fastlane_fixture import (
    boot_ranked_runtime, make_fakes, resident_for, seed_manifest, wait_until,
)
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState

MODEL = "m1"
IP_1 = "192.0.2.10"
IP_2 = "198.51.100.20"
RANKS = FastLaneTagRanks(main=1, sub_agent=3)
ONE_CLIENT = [FastLaneRule(address=IP_1, tag_ranks=RANKS)]
CLIENT_1_FIRST = [FastLaneRule(address=IP_1, tag_ranks=RANKS),
                  FastLaneRule(address=IP_2, tag_ranks=RANKS)]

MAIN_1 = {"ip": IP_1, "is_main": True}
SUB_1 = {"ip": IP_1, "is_sub_agent": True}
MAIN_2 = {"ip": IP_2, "is_main": True}
SUB_2 = {"ip": IP_2, "is_sub_agent": True}
T_HOLDER = "thread-holder"
T_PARKED = "thread-parked"
PROMPT_S = 5.0
GRACE_S = 30          # far longer than any wait below: a grace that is not cut never ends in a test
MIN_EVALUATIONS = 3   # grace-loop evaluations of the parked claim that prove "not outranked"


class World:
    def __init__(self, mgr):
        self.mgr = mgr
        self.slots = {}
        self.labels = {}
        self.starts = []        # slot ids, in the order their completion call began
        self.loads = []         # model tags, one per sidecar spawn
        self.swap_saves = []    # thread ids the swap-time KV save was asked to save
        self.evaluations = []   # results of the parked-claim check made while a claim was parked
        self._audit_ids = []    # (slot id, audit event) as the real code recorded them
        self._park_ids = []     # slot ids of the requests the real code parked in an inbox
        self.events = []        # ("kv_save", thread) and ("unload", model), in the order they happened
        self._gate = {}

    @property
    def audits(self):
        """(request label, audit event). The label is resolved when this is READ: a warm-path
        turn is audited before `send` has returned and registered its label."""
        return [(self.labels.get(sid), ev) for sid, ev in self._audit_ids]

    @property
    def parks(self):
        """Labels of the requests parked in an inbox, resolved when read (see `audits`)."""
        return [self.labels.get(sid) for sid in self._park_ids]

    def gate(self, label):
        return self._gate.setdefault(label, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        label = self.labels.get(slot.slot_id)
        if label in self._gate:
            await self._gate[label].wait()
        return {"ok": True}

    async def send(self, label, meta, thread, hold=False):
        if hold:
            self.gate(label)
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=dict(meta),
            wait_for_completion=True,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        return [self.labels.get(s, s) for s in self.starts]


@contextlib.asynccontextmanager
async def build_world(tmp_path, rules, grace_s=GRACE_S):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=1, grace_seconds=grace_s,
        max_grace_extensions=50, idle_hot_load_seconds=600,
    )
    seed_manifest(boot, MODEL, main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})
    holder = {}

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        holder["w"].loads.append(model_tag)
        return spawn_fn(binary, gguf, port, model_tag, argv, **kw)

    async def complete(slot, handle):
        return await holder["w"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=spawn_rec, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    w = World(mgr)
    holder["w"] = w

    async def save_recorder(r, handle, tip):
        w.swap_saves.append(tip.thread_id)
        w.events.append(("kv_save", tip.thread_id))

    mgr._save_holder_kv_at_swap = save_recorder

    # Pass-through spy: records which grace-family event the real code gave each request.
    real_audit = mgr._audit_async

    async def audit_spy(slot, event_type):
        w._audit_ids.append((slot.slot_id, event_type))
        return await real_audit(slot, event_type)

    mgr._audit_async = audit_spy

    real_park = mgr._park_fastlane_claim_locked

    def park_spy(slot, r):
        w._park_ids.append(slot.slot_id)
        return real_park(slot, r)

    mgr._park_fastlane_claim_locked = park_spy
    real_unload = mgr._begin_unload_locked

    def unload_spy(r):
        w.events.append(("unload", r.model_tag))
        return real_unload(r)

    mgr._begin_unload_locked = unload_spy

    # Pass-through spy: records the real answer whenever a claim is parked on a resident.
    real_check = mgr._parked_claim_outranks_holder_locked

    def check_spy(r, only_slot=None):
        out = real_check(r, only_slot=only_slot)
        if any(c.get("parked_on") == r.resident_key for c in mgr._fastlane_claims.values()):
            w.evaluations.append(out)
        return out

    mgr._parked_claim_outranks_holder_locked = check_spy
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000]), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[80000]):
            yield w
    finally:
        for ev in w._gate.values():
            ev.set()
        await mgr.shutdown()


async def _holder_busy_and_request_parked(w, holder_meta, parked_meta):
    """The holder's turn is HELD; a request of another thread is submitted and the
    real routing parks it in the holder's inbox. Asserts that setup."""
    holder_slot = await w.send("H", holder_meta, T_HOLDER, hold=True)
    await wait_until(lambda: "H" in w.order(), timeout=PROMPT_S)
    parked = await w.send("P", parked_meta, T_PARKED)
    r = resident_for(w.mgr, MODEL)
    await wait_until(
        lambda: r.inbox.qsize() >= 1 and parked not in w.mgr.queue._staging, timeout=PROMPT_S)
    assert r.state is ResidentState.ACTIVE and r.active_slot is holder_slot, (
        "setup: the holder must be mid-turn")
    assert parked.state is SlotState.STAGED, "setup: the parked request is still queued"
    assert w.order() == ["H"], f"setup: {w.order()}"
    assert any(c.get("parked_on") == r.resident_key for c in w.mgr._fastlane_claims.values()), (
        "setup: the parked request holds no claim, so the check under test has nothing to compare")
    return holder_slot, parked, r


async def _release_holder_and_settle(w, r):
    """Release the held turn and wait until the resident either starts the parked
    request or sits in its grace window: the two outcomes the check decides between."""
    w.gate("H").set()
    await wait_until(lambda: "P" in w.order() or r.in_grace_loop, timeout=PROMPT_S)


@pytest.mark.asyncio
class TestParkedBetterTagEndsTheHoldersWait:
    async def test_better_tag_parked_request_is_admitted_at_the_boundary_and_the_holder_saved(
            self, tmp_path):
        """Holder: a sub_agent turn (rank 3), held. A main request (rank 1) of the same client
        is parked in the inbox. After the turn ends the parked request is started at once
        although the grace window is 30 s: the holder never enters its grace window, and its KV
        was saved for the swap, for the holder's own thread."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            _holder, _parked, r = await _holder_busy_and_request_parked(w, SUB_1, MAIN_1)
            await _release_holder_and_settle(w, r)
            assert "P" in w.order(), (
                f"the better-tag parked request was not admitted at the boundary: "
                f"order={w.order()} in_grace_loop={r.in_grace_loop}")
            assert ("H", "grace_designated_victim_skip") in w.audits, (
                f"the holder did not skip its grace window: audits={w.audits}")
            assert ("H", "grace_enter") not in w.audits, (
                f"the holder entered a grace window with a better tag parked: audits={w.audits}")
            assert w.swap_saves == [T_HOLDER], (
                f"the holder's KV was not saved for the swap: saves={w.swap_saves}")
            assert w.loads == [MODEL], f"the engine was reloaded: loads={w.loads}"

    async def test_better_tag_parked_during_grace_ends_the_grace_window(self, tmp_path):
        """Holder: a sub_agent turn that has ended; it sits in its 30 s grace window. A main
        request (rank 1) of the same client is submitted and parked in the inbox. The grace
        window is ended by the next check and the parked request is started; the holder is
        not unloaded."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            await w.send("H", SUB_1, T_HOLDER)
            await wait_until(lambda: "H" in w.order(), timeout=PROMPT_S)
            r = resident_for(w.mgr, MODEL)
            await wait_until(lambda: r.in_grace_loop, timeout=PROMPT_S)
            await w.send("P", MAIN_1, T_PARKED)
            await wait_until(
                lambda: "P" in w.order() or (
                    len(w.evaluations) >= MIN_EVALUATIONS and not any(w.evaluations)),
                timeout=PROMPT_S)
            assert "P" in w.order(), (
                f"the better-tag parked request was not admitted while the holder sat in grace: "
                f"order={w.order()} evaluations={w.evaluations} inbox={r.inbox.qsize()}")
            assert ("H", "grace_designated_unload_target_break") in w.audits, (
                f"the grace window was not ended by the parked request: audits={w.audits}")
            assert w.loads == [MODEL], f"the engine was reloaded: loads={w.loads}"

    async def test_better_tag_parked_request_keeps_the_holder_loaded_when_keep_alive_is_zero(
            self, tmp_path):
        """Holder: a sub_agent turn with keep_alive 0 and no grace window. A main request
        (rank 1) of the same client is parked in the inbox. At the turn boundary the holder
        is not torn down: the parked request runs on the same engine (one spawn, no unload);
        without the better-tag check the holder is unloaded and the parked request reloads it."""
        async with build_world(tmp_path, ONE_CLIENT, grace_s=0) as w:
            unloads = []
            real_unload = w.mgr._begin_unload_locked

            def unload_spy(r):
                unloads.append(r.model_tag)
                return real_unload(r)

            w.mgr._begin_unload_locked = unload_spy
            _holder, _parked, r = await _holder_busy_and_request_parked(
                w, {**SUB_1, "keep_alive_s": 0}, MAIN_1)
            w.gate("H").set()
            await wait_until(lambda: "P" in w.order(), timeout=PROMPT_S)
            assert unloads == [], f"the holder was unloaded for a request it holds: {unloads}"
            assert w.loads == [MODEL], f"the engine was reloaded: loads={w.loads}"


@pytest.mark.asyncio
class TestParkedClaimControls:
    async def test_control_equal_tag_parked_request_does_not_end_the_holders_wait(self, tmp_path):
        """Control: the parked request has the SAME tag as the holder (sub_agent, rank 3).
        An equal key does not outrank, so the holder enters its grace window, the parked
        request is not started at the boundary and the swap-time save is not made."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            _holder, _parked, r = await _holder_busy_and_request_parked(w, SUB_1, SUB_1)
            await _release_holder_and_settle(w, r)
            assert r.in_grace_loop, f"the holder skipped its grace window: order={w.order()}"
            assert ("H", "grace_enter") in w.audits and ("H", "grace_designated_victim_skip") not in w.audits, (
                f"audits={w.audits}")
            assert "P" not in w.order(), f"an equal-tag parked request was admitted: {w.order()}"
            assert w.swap_saves == []

    async def test_control_equal_tag_parked_during_grace_does_not_end_the_grace_window(
            self, tmp_path):
        """Control: the holder sits in grace after a sub_agent turn and a sub_agent request of
        the same client is parked. The grace-loop check ran at least three times on the parked
        claim and answered no every time; the request was not started."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            await w.send("H", SUB_1, T_HOLDER)
            await wait_until(lambda: "H" in w.order(), timeout=PROMPT_S)
            r = resident_for(w.mgr, MODEL)
            await wait_until(lambda: r.in_grace_loop, timeout=PROMPT_S)
            await w.send("P", SUB_1, T_PARKED)
            await wait_until(
                lambda: "P" in w.order() or len(w.evaluations) >= MIN_EVALUATIONS,
                timeout=PROMPT_S)
            assert "P" not in w.order(), f"an equal-tag parked request was admitted: {w.order()}"
            assert w.evaluations and not any(w.evaluations), (
                f"the check was never made or said yes: {w.evaluations}")
            assert ("H", "grace_designated_unload_target_break") not in w.audits, f"audits={w.audits}"

    async def test_control_better_tag_of_another_client_does_not_end_a_better_clients_wait(
            self, tmp_path):
        """Control, across clients: the holder serves client 1 (listed first) with its worst
        tag; a MAIN request (rank 1) of client 2 is parked. A better tag never lifts a worse
        client, so the holder enters its grace window and the parked request is not started."""
        async with build_world(tmp_path, CLIENT_1_FIRST) as w:
            _holder, _parked, r = await _holder_busy_and_request_parked(w, SUB_1, MAIN_2)
            await _release_holder_and_settle(w, r)
            assert r.in_grace_loop, f"the holder skipped its grace window: order={w.order()}"
            assert ("H", "grace_enter") in w.audits and ("H", "grace_designated_victim_skip") not in w.audits, (
                f"audits={w.audits}")
            assert "P" not in w.order(), (
                f"client 2's better tag was admitted ahead of client 1's holder: {w.order()}")
            assert w.swap_saves == []

    async def test_control_better_client_parked_request_ends_the_wait_whatever_its_tag(
            self, tmp_path):
        """Control, across clients: the holder serves client 2 (listed second) with its BEST
        tag (main, rank 1); a SUB_AGENT request (rank 3) of client 1 is parked. Client order
        decides, so the parked request outranks the holder: it is started at the boundary and
        the holder's KV is saved. Pins that the full key keeps client order first."""
        async with build_world(tmp_path, CLIENT_1_FIRST) as w:
            _holder, _parked, r = await _holder_busy_and_request_parked(w, MAIN_2, SUB_1)
            await _release_holder_and_settle(w, r)
            assert "P" in w.order(), (
                f"the better client's parked request was not admitted: order={w.order()}")
            assert ("H", "grace_designated_victim_skip") in w.audits and ("H", "grace_enter") not in w.audits, (
                f"audits={w.audits}")
            assert w.swap_saves == [T_HOLDER]


UNCLASSIFIED_1 = {"ip": IP_1}   # no class label: rank 5


async def _holder_in_grace(w, meta):
    """The holder's turn (thread T_HOLDER) ends and it sits in its grace window."""
    await w.send("H", meta, T_HOLDER)
    await wait_until(lambda: "H" in w.order(), timeout=PROMPT_S)
    r = resident_for(w.mgr, MODEL)
    await wait_until(lambda: r.in_grace_loop, timeout=PROMPT_S)
    w._park_ids.clear()   # the holder's own request was parked when its resident was reserved
    return r


@pytest.mark.asyncio
class TestSameThreadFollowUpInGrace:
    """Pins the warm path: a same-thread follow-up of a better tag than the holder's is
    parked (it is itself the better claim) and ends the grace window; one of the same or a
    worse tag still rides the warm path."""

    async def test_better_tag_follow_up_parks_ends_the_grace_and_the_holder_is_saved_first(
            self, tmp_path):
        """Holder: a sub_agent turn (rank 3) in its 30 s grace window. A same-thread MAIN
        follow-up (rank 1) arrives: it is parked in the inbox, the grace window is ended, the
        follow-up runs as an ordinary turn on the same engine, and the holder's KV save is
        recorded for its thread before anything is unloaded (here nothing is)."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            await _holder_in_grace(w, SUB_1)
            await w.send("F", MAIN_1, T_HOLDER)
            await wait_until(lambda: "F" in w.order() or w.audits.count(("H", "grace_enter")) > 1,
                             timeout=PROMPT_S)
            assert "F" in w.order(), f"the follow-up was not started: order={w.order()}"
            assert w.parks == ["F"], f"the follow-up did not park: parks={w.parks}"
            assert ("H", "grace_designated_unload_target_break") in w.audits, f"audits={w.audits}"
            assert ("F", "active_match_completed") not in w.audits, "it rode the warm path"
            kv = [i for i, e in enumerate(w.events) if e[0] == "kv_save"]
            unloads = [i for i, e in enumerate(w.events) if e[0] == "unload"]
            assert kv and w.events[kv[0]] == ("kv_save", T_HOLDER), (
                f"the holder's KV was not saved for the swap: events={w.events}")
            assert all(kv[0] < u for u in unloads), f"an unload came before the save: {w.events}"
            assert w.loads == [MODEL], f"the engine was reloaded: loads={w.loads}"

    async def test_control_same_tag_follow_up_rides_the_warm_path(self, tmp_path):
        """Control: the follow-up has the SAME tag as the holder's turn. It is served by the
        warm path (audit active_match_completed), is never parked, the grace window is not
        ended and no swap-time save is made."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            r = await _holder_in_grace(w, SUB_1)
            await w.send("F", SUB_1, T_HOLDER)
            await wait_until(lambda: ("F", "active_match_completed") in w.audits, timeout=PROMPT_S)
            assert w.parks == [], f"the follow-up was parked: {w.parks}"
            assert r.in_grace_loop, "the grace window ended"
            assert not any(n == "grace_designated_unload_target_break" for _l, n in w.audits)
            assert w.swap_saves == []
            assert w.loads == [MODEL]

    async def test_control_worse_tag_follow_up_rides_the_warm_path(self, tmp_path):
        """Control: a same-thread follow-up of a WORSE tag (rank 5 after a rank 3 turn) is
        also served warm and never parked."""
        async with build_world(tmp_path, ONE_CLIENT) as w:
            r = await _holder_in_grace(w, SUB_1)
            await w.send("F", UNCLASSIFIED_1, T_HOLDER)
            await wait_until(lambda: ("F", "active_match_completed") in w.audits, timeout=PROMPT_S)
            assert w.parks == [], f"the follow-up was parked: {w.parks}"
            assert r.in_grace_loop, "the grace window ended"
            assert w.swap_saves == []
