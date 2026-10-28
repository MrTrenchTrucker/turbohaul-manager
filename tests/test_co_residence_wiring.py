"""A tag with ONE live engine and waiting same-model demand gets its second engine (the periodic re-check).

The scenario, built by every test below and asserted before the loop runs:
one live ACTIVE engine of a tag on card 0, the box budget has a free sidecar slot (cap 2, one
live engine), a second card is free and fits, a one-card-per-engine manifest, and same-model
demand is WAITING. Two populations wait:

  (b) a request STAGED in the queue that ``pop_next`` holds aside because its (thread, model)
      is the engine's live grace pair, and the grace loop never claims it;
  (a) a request PARKED in the busy engine's inbox (the arrival path declined it once and parked it in the busy engine's inbox).

The tests drive the real ``TurbohaulManager._dispatch_loop`` one iteration at a time (a gate in
front of ``queue.pop_next`` releases one tick per ``step()``, so every wait is bounded and every
count is read at a known point). Only the outside world is faked: the card picker, the VRAM
gate, the placement resolver, the cap/placement answer and the engine spawn. The re-check clock
is the manager's ``_co_residence_now`` seam; ages are made by moving that clock, never by
sleeping. The re-check is reached only through the real dispatch loop, and its log lines are
read from the ``turbohaul.co_residence`` loggers.

Ages used (seconds): an unprotected waiting request counts after 5, a protected one (grace pair
or the engine's idle thread) after 120. The re-check runs at most once per 10 s and a tag that
found no room is not probed again for 60 s. Phase offsets below sit 1 s or more away from those
edges (a phase that must find the debounce open moves the clock 11 s, never exactly 10); the edges
themselves are pinned by the unit tests of the module.
"""
from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock

import pytest
import yaml

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

TAG = "tag-a"
COLD_TAG = "tag-cold"
WARM = "t-warm"            # the engine's grace thread
IDLE = "t-idle"            # the engine's idle thread
FREE_CARD = 1
NEED_MIB = 4000
STEP_TIMEOUT_S = 3.0
DECLINED = "CO_RESIDENCE_DECLINED"
RECHECK = "CO_RESIDENCE_RECHECK"


# --------------------------------------------------------------------------------------
# fixture: a real manager, hand-built engines, faked probes, a stepped dispatch loop
# --------------------------------------------------------------------------------------

def _boot_runtime(tmp_path, cap):
    root = tmp_path / "state"
    root.mkdir()
    for d in ("blobs", "manifests", "import-staging"):
        (root / d).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=cap, grace_seconds=3600,
            max_grace_extensions=50, idle_hot_load_seconds=3600,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _mocks():
    def spawn(binary, gguf, port, model_tag, argv, **kw):
        proc = MagicMock()
        proc.pid = 93_000
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def health(*a, **k):
        return True

    async def sigterm(handle, **k):
        return True, "sigterm-clean"

    async def vram(**k):
        return True, 100

    async def complete(slot, handle):
        return {"ok": True}

    return dict(spawn_fn=spawn, health_fn=health, sigterm_fn=sigterm, vram_fn=vram, complete_fn=complete)


def _resident(mgr, tag, *, state=ResidentState.ACTIVE, grace_pair=None, idle_thread=None):
    r = Resident(
        model_tag=tag, resident_key=tag, port=59701, state=state, main_gpu=0, split_mode="none",
        reserved_need_mib=NEED_MIB,
        grace=GraceTimer(grace_seconds=3600, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=3600),
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    if grace_pair is not None:
        r.grace.start(*grace_pair)
    r.idle_thread_id = idle_thread
    mgr._residents[tag] = r
    return r


class Box:
    """What the faked outside world was asked, the re-check clock, and the loop stepper."""

    def __init__(self, mgr):
        self.mgr = mgr
        self.clock = [time.monotonic()]
        self.step_s = 0.0
        self.reserves = []        # (slot, main_gpu_override, split_mode_override)
        self.caps = []            # _route_cap_and_placement(tag)
        self.picks = []           # _pick_card_for_new_instance(tag, need)
        self.admits = []          # _vram_admits_locked(card, split)
        self.pick_result = (FREE_CARD, "none")
        self.admit_result = True
        # What the per-card free-VRAM reading shows (the re-check reads it to tell "no card is
        # visible" from "cards are visible but none has room"). Faked so no test shells out.
        self.card_avail = [NEED_MIB, NEED_MIB]
        # None: the fake world always has a card. N: it fits exactly N more engines, then the
        # card picker finds no card with room.
        self.fit_limit = None
        self.sem = asyncio.Semaphore(0)
        self.parked = 0
        self.task = None

    def probes(self):
        return (len(self.caps), len(self.picks), len(self.admits))

    def now(self):
        return self.clock[0]

    # -- the dispatcher, one iteration at a time ---------------------------------------
    async def start_loop(self, monkeypatch):
        """Run the loop up to the point where the FIRST iteration's re-check has finished."""
        mgr = self.mgr
        monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", 0.005)
        real_pop = mgr.queue.pop_next

        async def gated_pop(**kw):
            self.parked += 1
            await self.sem.acquire()
            self.clock[0] += self.step_s
            return await real_pop(**kw)

        monkeypatch.setattr(mgr.queue, "pop_next", gated_pop)
        self.task = asyncio.create_task(mgr._dispatch_loop())
        await self._until_parked(1)

    async def _until_parked(self, target):
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while self.parked < target:
            if self.task.done():
                exc = None if self.task.cancelled() else self.task.exception()
                raise AssertionError(f"the dispatch loop ended before the step finished: {exc!r}")
            assert time.monotonic() < deadline, "the dispatch loop did not reach its next tick in time"
            await asyncio.sleep(0.002)

    async def step(self, n=1):
        """Let n more iterations run: each pops (maybe routes) and then runs the next re-check."""
        for _ in range(n):
            target = self.parked + 1
            self.sem.release()
            await self._until_parked(target)

    async def stop_loop(self):
        if self.task is None:
            return
        self.mgr._stop_event.set()
        self.sem.release()
        done, _ = await asyncio.wait({self.task}, timeout=STEP_TIMEOUT_S)
        if not done:
            self.task.cancel()
        elif not self.task.cancelled():
            self.task.exception()     # a loop that died is reported by the step that saw it


async def _world(tmp_path, monkeypatch, *, cap=2):
    boot, runtime = _boot_runtime(tmp_path, cap)
    (boot.storage.manifests_path / f"{TAG}.yaml").write_text(yaml.safe_dump({
        "model_tag": TAG, "gguf_blob_sha256": "a" * 64, "gguf_size_bytes": NEED_MIB * 1024 * 1024,
        "context_size": 2048, "expected_vram_bytes": NEED_MIB * 1024 * 1024, "auto_place": True,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    box = Box(mgr)

    def cap_placement(tag):
        box.caps.append(tag)
        return 2, True

    def resolve(tag):
        return (NEED_MIB, 1, 0, "none", 0, True, False)

    def pick(tag, need):
        box.picks.append((tag, need))
        if box.fit_limit is not None and len(box.reserves) >= box.fit_limit:
            return (None, "none")
        return box.pick_result

    def card_avail_mib():
        return list(box.card_avail)

    def admits(need, parallel, g, sp, *a, **k):
        box.admits.append((g, sp))
        return box.admit_result

    async def reserve(slot, main_gpu_override=None, split_mode_override=None):
        box.reserves.append((slot, main_gpu_override, split_mode_override))
        n = len(mgr._residents)
        r = Resident(
            model_tag=slot.model_tag, resident_key=f"{slot.model_tag}#{n}", port=59800 + n,
            state=ResidentState.RESERVED_LOADING,
            main_gpu=main_gpu_override if main_gpu_override is not None else 0,
            split_mode=split_mode_override or "none", reserved_need_mib=NEED_MIB, instance_idx=n,
            last_active_monotonic=time.monotonic(),
        )
        r.inbox = asyncio.Queue()
        r.inbox.put_nowait(slot)
        mgr._residents[r.resident_key] = r
        return r

    mgr._route_cap_and_placement = cap_placement
    mgr._resolve_placement_locked = resolve
    mgr._pick_card_for_new_instance = pick
    mgr._card_avail_mib = card_avail_mib
    mgr._vram_admits_locked = admits
    mgr._reserve_and_start_locked = reserve
    monkeypatch.setattr(mgr, "_co_residence_now", box.now, raising=False)
    return mgr, box


def _slot(box, tag, thread, age):
    """A waiting request that has been waiting ``age`` seconds on the re-check clock."""
    slot = Slot.new(tag, prompt="p", thread_id=thread)
    slot.created_at = box.clock[0] - age
    return slot


async def _stage(mgr, box, thread, age, tag=TAG):
    slot = _slot(box, tag, thread, age)
    await mgr.queue.enqueue(slot)
    return slot


def _park(box, r, thread, age):
    slot = _slot(box, r.model_tag, thread, age)
    slot.state = SlotState.STAGED
    r.inbox.put_nowait(slot)
    return slot


def _inbox(r):
    return list(r.inbox._queue)


def _ids(slots):
    return [s.slot_id for s in slots]


def _staged(mgr):
    return list(mgr.queue._staging)


def _second_engines(mgr, first):
    return [x for x in mgr._instances_for(TAG) if x is not first]


def _assert_starting_scene(mgr, box, r, *, refusing=None, demand=True):
    """The starting scene, asserted before any tick: the later checks cannot pass trivially because the box never
    had one live ACTIVE engine, a free sidecar slot, a free fitting card and waiting demand."""
    assert mgr.runtime.queue.max_parallel_sidecars == 2
    assert len(mgr._model_residents()) == 1
    assert len(mgr._instances_for(TAG)) == 1 and mgr._instances_for(TAG)[0] is r
    assert r.state is ResidentState.ACTIVE and r.main_gpu == 0
    assert box.pick_result[0] in (FREE_CARD, None) and box.pick_result[0] != r.main_gpu
    if refusing is None:
        assert box.pick_result == (FREE_CARD, "none") and box.admit_result is True
    waiting = [s for s in _staged(mgr) if s.model_tag == TAG] + _inbox(r)
    assert waiting or not demand, "no same-model demand is waiting"


def _lines(caplog, prefix):
    return [rec.getMessage() for rec in caplog.records
            if rec.name.startswith("turbohaul.co_residence") and rec.getMessage().startswith(prefix)]


def _declined_line(via, reason, card):
    return (f"{DECLINED} via={via} tag={TAG} reason={reason} eff_cap=2 instances=1 residents=1 "
            f"max_sidecars=2 need_mib={NEED_MIB} card={card} split=none")


def _recheck_declined_line(reason, card, engines=1):
    """The re-check's decline line: its verdict has no engine ceiling and never reads the box
    budget, so those two fields read ``-``. ``engines`` is the tag's engine count (all engines
    on the box are of the one tag in these fixtures)."""
    return (f"{DECLINED} via=recheck tag={TAG} reason={reason} eff_cap=- instances={engines} "
            f"residents={engines} max_sidecars=- need_mib={NEED_MIB} card={card} split=none")


# --------------------------------------------------------------------------------------
# W1: population (b), a staged request held aside because it is the engine's grace pair
# --------------------------------------------------------------------------------------

async def _staged_world(tmp_path, monkeypatch, age):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG, grace_pair=(WARM, TAG))
    slot = await _stage(mgr, box, WARM, age)
    return mgr, box, r, slot


async def test_w1_staged_grace_follow_up_is_taken_only_once_it_is_old_enough(tmp_path, monkeypatch, caplog):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=119.0)
    try:
        _assert_starting_scene(mgr, box, r)
        assert r.grace.matches(WARM, TAG) and slot in _staged(mgr), "the slot is the engine's grace pair"
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            await box.step(3)
            # 119 s and protected: the warm follow-up is not starved demand yet.
            assert box.reserves == [] and slot in _staged(mgr)
            assert box.probes() == (0, 0, 0), "no eligible demand: nothing may be probed"
            box.clock[0] += 11.0                      # 130 s: past the protected age
            await box.step(2)
            assert len(box.reserves) == 1, "the starved demand never got its second engine"
            got, card, split = box.reserves[0]
            assert got is slot and card == FREE_CARD and split == "none"
            assert slot not in _staged(mgr), "the slot left the queue"
            new = _second_engines(mgr, r)
            assert len(new) == 1 and _ids(_inbox(new[0])) == [slot.slot_id]
            assert _lines(caplog, DECLINED) == [], "a successful spawn logs no decline line"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w1b_the_spawn_is_announced_on_one_recheck_line(tmp_path, monkeypatch, caplog):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=129.0)
    try:
        _assert_starting_scene(mgr, box, r)
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert len(box.reserves) == 1, "the starved demand never got its second engine"
            assert _lines(caplog, RECHECK) == [
                f"{RECHECK} tag={TAG} source=staged slot={slot.slot_id} waited_s=129.0 instances=1 residents=1"
            ]
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W2 / W3: population (a), requests parked in the busy engine's inbox
# --------------------------------------------------------------------------------------

async def test_w2_parked_request_is_taken_out_of_the_inbox_and_the_others_keep_their_order(
    tmp_path, monkeypatch,
):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG, idle_thread=IDLE)
    u1 = _park(box, r, "t-u1", 2.0)
    target = _park(box, r, "t-target", 30.0)
    u2 = _park(box, r, "t-u2", 1.0)
    p = _park(box, r, IDLE, 119.5)          # the engine's idle thread: protected until 120 s
    try:
        _assert_starting_scene(mgr, box, r)
        assert _ids(_inbox(r)) == _ids([u1, target, u2, p])
        await box.start_loop(monkeypatch)
        assert len(box.reserves) == 1, "the parked demand never got its second engine"
        got, card, split = box.reserves[0]
        assert got is target and card == FREE_CARD and split == "none"
        assert _ids(_inbox(r)) == _ids([u1, u2, p]), "only the chosen slot left, the rest in original order"
        new = _second_engines(mgr, r)
        assert len(new) == 1 and _ids(_inbox(new[0])) == [target.slot_id]
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w2b_a_request_on_the_idle_thread_waits_for_the_protected_age(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG, idle_thread=IDLE)
    p = _park(box, r, IDLE, 119.0)
    u = _park(box, r, "t-u", 1.0)
    try:
        _assert_starting_scene(mgr, box, r)
        await box.start_loop(monkeypatch)
        await box.step(3)
        assert box.reserves == [] and _ids(_inbox(r)) == _ids([p, u])
        assert box.probes() == (0, 0, 0), "no eligible demand: nothing may be probed"
        box.clock[0] += 11.0                      # 130 s and 12 s: both eligible, the older goes first
        await box.step(2)
        assert len(box.reserves) == 1, "the parked demand never got its second engine"
        assert box.reserves[0][0] is p and box.reserves[0][1] == FREE_CARD
        assert _ids(_inbox(r)) == [u.slot_id]
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w2c_an_inbox_take_is_announced_with_its_source(tmp_path, monkeypatch, caplog):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    target = _park(box, r, "t-target", 30.0)
    try:
        _assert_starting_scene(mgr, box, r)
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert len(box.reserves) == 1, "the parked demand never got its second engine"
            assert _lines(caplog, RECHECK) == [
                f"{RECHECK} tag={TAG} source=inbox slot={target.slot_id} waited_s=30.0 instances=1 residents=1"
            ]
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w3_a_young_unprotected_request_is_not_demand_yet(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    young = _park(box, r, "t-young", 4.9)
    try:
        _assert_starting_scene(mgr, box, r)
        await box.start_loop(monkeypatch)
        await box.step(3)
        assert box.reserves == [] and _ids(_inbox(r)) == [young.slot_id]
        assert box.probes() == (0, 0, 0), "a request under 5 s old must not cause any probe"
        box.clock[0] += 11.0                      # 15.9 s: now it is real demand
        await box.step(2)
        assert len(box.reserves) == 1, "the parked demand never got its second engine"
        assert box.reserves[0][0] is young and _inbox(r) == []
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W4: a declined re-check is visible, with the reason
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["vram_refused", "no_card"])
async def test_w4_a_declined_recheck_logs_one_line_with_the_reason(tmp_path, monkeypatch, caplog, reason):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=200.0)
    if reason == "no_card":
        box.pick_result = (None, "none")
        box.card_avail = []                # no card is visible at all, so the case means "no card"
    else:
        box.admit_result = False
    try:
        _assert_starting_scene(mgr, box, r, refusing=reason)
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert len(box.picks) == 1, "the re-check never asked the card picker"
            assert box.reserves == [] and slot in _staged(mgr), "a declined re-check takes nothing"
            card = "-" if reason == "no_card" else str(FREE_CARD)
            # The re-check decline line reports no engine ceiling and no box budget
            # (``eff_cap=-``, ``max_sidecars=-``), because the re-check does not read either.
            # One line, the reason, ``via=recheck``, the tag, need and card are still asserted.
            assert _lines(caplog, DECLINED) == [_recheck_declined_line(reason, card)]
            assert len(box.admits) == (0 if reason == "no_card" else 1)
            assert _lines(caplog, RECHECK) == [], "nothing was taken, so nothing is announced"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W5 / W6: debounce and the no-room memo bound the probes
# --------------------------------------------------------------------------------------

async def test_w5_debounce_and_memo_bound_the_probes(tmp_path, monkeypatch, caplog):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=200.0)
    box.admit_result = False
    t0 = box.clock[0]
    scans = []                                  # one entry per look at the queue (one per attempt)
    real_snapshot = mgr.queue.staging_snapshot

    def counting_snapshot():
        scans.append(box.clock[0])
        return real_snapshot()

    mgr.queue.staging_snapshot = counting_snapshot
    try:
        _assert_starting_scene(mgr, box, r, refusing="vram_refused")
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert (len(box.picks), len(box.admits)) == (1, 1), "the first attempt never probed"
            first = box.probes()
            assert len(scans) == 1
            await box.step(3)                                   # same instant: debounced, not even a scan
            assert box.probes() == first and len(scans) == 1
            box.clock[0] = t0 + 9.0
            await box.step()                                    # inside 10 s: debounced
            assert box.probes() == first and len(scans) == 1
            box.clock[0] = t0 + 11.0                            # past the debounce, inside the 60 s memo:
            await box.step()                                    # an attempt (a scan) but no probe
            assert box.probes() == first and len(scans) == 2
            box.clock[0] = t0 + 12.5
            await box.step()                                    # the attempt above is what the debounce counts
            assert box.probes() == first and len(scans) == 2
            for offset in (30.0, 59.0):
                box.clock[0] = t0 + offset
                await box.step()
                assert box.probes() == first, f"a memo-blocked tag was probed again at +{offset}s"
            assert len(scans) == 4
            assert len(_lines(caplog, DECLINED)) == 1, "a blocked window must not repeat the decline line"
            box.clock[0] = t0 + 71.0                            # memo expired, debounce satisfied
            await box.step()
            assert (len(box.picks), len(box.admits)) == (2, 2), "after 60 s the tag gets exactly one more attempt"
            assert len(scans) == 5 and len(_lines(caplog, DECLINED)) == 2
            assert box.reserves == [] and slot in _staged(mgr)
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w6_an_unreadable_probe_never_storms(tmp_path, monkeypatch, caplog):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=200.0)
    box.pick_result = (None, "none")                            # the free-VRAM probe is unreadable
    box.card_avail = []                                         # ... so no card is visible
    try:
        _assert_starting_scene(mgr, box, r, refusing="no_card")
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert len(box.picks) == 1, "the first attempt never probed"
            t0 = box.clock[0]
            box.step_s = 2.0
            await box.step(25)                                  # 50 s of 2 s ticks
            assert abs(box.clock[0] - (t0 + 50.0)) < 1e-6, "the clock must have run the full 50 s"
            assert len(box.picks) == 1, "25 ticks in 50 s may probe only once"
            assert box.reserves == [] and box.admits == []
            assert len(_lines(caplog, DECLINED)) == 1
            assert slot in _staged(mgr)
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W7: the fit, not the box budget, bounds the engines the re-check starts
# --------------------------------------------------------------------------------------

FITTING_ENGINES = 2          # the fake world has room for exactly this many more engines


# The box budget does not limit the periodic re-check; the card and VRAM fit is the only
# ceiling. So the fixture bounds the FIT instead of the budget: the fake
# world fits exactly FITTING_ENGINES more engines. The re-check must start exactly that many
# (past the budget of 2, up to 3 engines on a box with a budget of 2), then stop with ONE
# decline line naming the VRAM refusal, and not probe that tag again inside its no-room window.
async def test_w7_the_spawn_fills_the_budget_and_no_second_spawn_follows(
    tmp_path, monkeypatch, caplog,
):
    mgr, box, r, slot = await _staged_world(tmp_path, monkeypatch, age=200.0)
    box.fit_limit = FITTING_ENGINES
    try:
        _assert_starting_scene(mgr, box, r)
        assert mgr.runtime.queue.max_parallel_sidecars == 2 < 1 + FITTING_ENGINES
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert len(box.reserves) == 1, "the starved demand never got its second engine"
            assert len(mgr._model_residents()) == 2
            for _ in range(FITTING_ENGINES - 1):                 # the box budget is spent: fit still holds
                box.clock[0] += 11.0
                await _stage(mgr, box, WARM, 200.0)
                await box.step()
            assert len(box.reserves) == FITTING_ENGINES, "the budget stopped a re-check that still fits"
            assert len(mgr._model_residents()) == 1 + FITTING_ENGINES
            assert _lines(caplog, DECLINED) == [], "every attempt so far fitted, so none is declined"
            box.clock[0] += 11.0                                 # the world has no room left
            await _stage(mgr, box, WARM, 200.0)
            await box.step()
            assert len(box.reserves) == FITTING_ENGINES, "an engine was started although nothing fits"
            # The picker found no card with room, so the line names no card; cards are visible
            # (``card_avail`` is not empty), which makes the reason a VRAM refusal.
            assert _lines(caplog, DECLINED) == [_recheck_declined_line("vram_refused", "-", 1 + FITTING_ENGINES)], (
                "the refusal must be reported once, as a VRAM refusal"
            )
            after_decline = box.probes()
            for _ in range(4):                                   # inside the 60 s no-room window
                box.clock[0] += 11.0
                await _stage(mgr, box, WARM, 200.0)
                await box.step()
            assert len(box.reserves) == FITTING_ENGINES, "a tag with no room got another engine"
            assert box.probes() == after_decline, "a tag with no room was probed inside its window"
            assert len(_lines(caplog, DECLINED)) == 1, "a blocked window must not repeat the decline line"
            assert len(mgr._model_residents()) == 1 + FITTING_ENGINES
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W8: a budget of 1 does not stop the re-check, and a budget of 2 behaves the same
# --------------------------------------------------------------------------------------

async def _both_populations(tmp_path, monkeypatch, cap):
    mgr, box = await _world(tmp_path, monkeypatch, cap=cap)
    r = _resident(mgr, TAG, grace_pair=(WARM, TAG))
    staged = await _stage(mgr, box, WARM, 200.0)
    parked = _park(box, r, "t-parked", 30.0)
    await box.start_loop(monkeypatch)
    for _ in range(4):
        box.clock[0] += 100.0
        await box.step()
    return mgr, box, r, staged, parked


# The box budget does not limit the periodic re-check, so a budget-1 box probes and starts
# further engines for as long as the fit holds. The two halves are a comparison: the same
# engine, the same two waiting requests and the same ticks give the same probes and the same
# engines at a budget of 1 and at a budget of 2.
async def test_w8_cap_one_is_inert_and_the_same_box_at_cap_two_is_not(tmp_path, monkeypatch):
    one = two = None
    try:
        (tmp_path / "c1").mkdir()
        (tmp_path / "c2").mkdir()
        one = await _both_populations(tmp_path / "c1", monkeypatch, cap=1)
        mgr1, box1, r1, staged1, parked1 = one
        assert mgr1.runtime.queue.max_parallel_sidecars == 1
        assert box1.probes() == (2, 2, 2), "a budget of 1 must not stop the re-check probing"
        assert _ids([x[0] for x in box1.reserves]) == _ids([staged1, parked1]), (
            "both waiting requests got an engine, the older first"
        )
        assert staged1 not in _staged(mgr1) and _inbox(r1) == []
        assert len(mgr1._instances_for(TAG)) == 3
        await box1.stop_loop()
        # the comparison: same engine, same two waiting requests, same ticks, budget 2 -> the same
        two = await _both_populations(tmp_path / "c2", monkeypatch, cap=2)
        mgr2, box2, r2, staged2, parked2 = two
        assert mgr2.runtime.queue.max_parallel_sidecars == 2
        assert box2.probes() == box1.probes(), "a budget of 2 must probe exactly as a budget of 1 does"
        assert _ids([x[0] for x in box2.reserves]) == _ids([staged2, parked2]), (
            "the older waiting request goes first"
        )
        assert len(mgr2._instances_for(TAG)) == 3
    finally:
        for w in (one, two):
            if w is not None:
                await w[1].stop_loop()
                await w[0].shutdown()


# --------------------------------------------------------------------------------------
# W9: the arrival path says why it did not spawn (and only when there is a live engine)
# --------------------------------------------------------------------------------------

async def _arrive(mgr, box, thread, tag=TAG):
    slot = _slot(box, tag, thread, 0.0)
    await mgr.queue.enqueue(slot)
    await box.step()
    return slot


async def test_w9_an_arrival_decline_is_logged_once_per_window_and_a_spawn_logs_nothing(
    tmp_path, monkeypatch, caplog,
):
    mgr, box = await _world(tmp_path, monkeypatch)
    box.admit_result = False
    r = _resident(mgr, TAG)
    t0 = box.clock[0]
    try:
        _assert_starting_scene(mgr, box, r, refusing="vram_refused", demand=False)
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            a1 = await _arrive(mgr, box, "t-a1")
            assert _ids(_inbox(r)) == [a1.slot_id], "step 3 still parks the arrival on the live engine"
            line = _declined_line("arrival", "vram_refused", FREE_CARD)
            assert _lines(caplog, DECLINED) == [line], "the arrival decline was not logged"
            box.clock[0] = t0 + 4.0
            a2 = await _arrive(mgr, box, "t-a2")
            assert _ids(_inbox(r)) == [a1.slot_id, a2.slot_id]
            assert _lines(caplog, DECLINED) == [line], "a repeat inside the window must not log again"
            while r.inbox.qsize():                               # no waiting demand left to cause a re-check
                r.inbox.get_nowait()
            box.clock[0] = t0 + 65.0
            a3 = await _arrive(mgr, box, "t-a3")
            assert _ids(_inbox(r)) == [a3.slot_id]
            assert _lines(caplog, DECLINED) == [line, line], "a new window logs the decline again"
            while r.inbox.qsize():
                r.inbox.get_nowait()
            box.pick_result, box.admit_result = (FREE_CARD, "none"), True
            box.clock[0] = t0 + 66.0
            a4 = await _arrive(mgr, box, "t-a4")
            assert len(box.reserves) == 1 and box.reserves[0][0] is a4, "the arrival should have spawned"
            assert _lines(caplog, DECLINED) == [line, line], "an arrival that spawns logs no decline"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w9c_a_cold_tag_that_cannot_spawn_logs_no_decline(tmp_path, monkeypatch, caplog):
    mgr, box = await _world(tmp_path, monkeypatch)
    box.admit_result = False
    r = _resident(mgr, TAG)
    deferred = []
    mgr._defer_unroutable = lambda slot, **kw: deferred.append(slot)
    mgr._reclaim_first_applies = lambda *a, **k: False
    try:
        _assert_starting_scene(mgr, box, r, refusing="vram_refused", demand=False)
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            a1 = await _arrive(mgr, box, "t-a1")                 # positive arm: a live tag declines loudly
            assert _ids(_inbox(r)) == [a1.slot_id]
            assert len(_lines(caplog, DECLINED)) == 1, "the live-tag decline was not logged"
            cold = await _arrive(mgr, box, "t-cold", tag=COLD_TAG)
            assert (COLD_TAG, NEED_MIB) in box.picks, "the cold tag never reached the step-2 picker"
            assert cold in deferred, "the cold tag should have fallen through to the legacy defer path"
            assert len(_lines(caplog, DECLINED)) == 1, "a cold tag is not a co-residence question"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# W10: a failure inside the re-check does not stop the dispatcher
# --------------------------------------------------------------------------------------

async def _loop_survives(tmp_path, monkeypatch, caplog, install):
    mgr, box = await _world(tmp_path, monkeypatch)
    box.admit_result = False
    r = _resident(mgr, TAG)
    old = _park(box, r, "t-old", 30.0)           # demand for the failing re-check to look at
    state = install(mgr)
    try:
        _assert_starting_scene(mgr, box, r, refusing="vram_refused")
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await box.start_loop(monkeypatch)
            assert state["raised"] == 1, "the injected failure was never reached"
            arrival = await _arrive(mgr, box, "t-after")
            assert not box.task.done(), "the dispatch loop died with the re-check"
            assert _ids(_inbox(r)) == [old.slot_id, arrival.slot_id], "the loop stopped dispatching"
            errors = [rec for rec in caplog.records
                      if rec.levelno >= logging.ERROR and "co-residence re-check failed" in rec.getMessage()]
            assert len(errors) == 1, "the failure must be logged once, with its traceback"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_w10_a_failing_grace_read_in_the_recheck_does_not_kill_the_loop(tmp_path, monkeypatch, caplog):
    def install(mgr):
        real = mgr._grace_active_exclusions
        state = {"calls": 0, "raised": 0}

        async def flaky():
            state["calls"] += 1
            if state["calls"] == 1:
                state["raised"] += 1
                raise RuntimeError("injected grace read failure")
            return await real()

        mgr._grace_active_exclusions = flaky
        return state

    await _loop_survives(tmp_path, monkeypatch, caplog, install)


async def test_w10b_a_failing_queue_scan_in_the_recheck_does_not_kill_the_loop(tmp_path, monkeypatch, caplog):
    def install(mgr):
        real = mgr.queue.staging_snapshot
        state = {"calls": 0, "raised": 0}

        def flaky():
            state["calls"] += 1
            if state["calls"] == 1:
                state["raised"] += 1
                raise RuntimeError("injected queue scan failure")
            return real()

        mgr.queue.staging_snapshot = flaky
        return state

    await _loop_survives(tmp_path, monkeypatch, caplog, install)
