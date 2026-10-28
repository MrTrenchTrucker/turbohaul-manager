"""Fit-based admission of further engines by the periodic co-residence re-check.

The rule under test: when same-model requests are waiting behind a busy engine, the periodic
re-check starts a further engine for that model whenever the fit check passes (a card is
picked for it and the cross-resident VRAM gate admits it). The box-wide engine budget
(``queue.max_parallel_sidecars``) is not what limits it: the ceiling is the card/VRAM fit.
The arrival path (a request routed when it arrives) keeps the budget, and says so with one
``CO_RESIDENCE_DECLINED`` line.

Every test drives the real ``TurbohaulManager``: the real arrival routing, the real engine
reservation and start, real placement and the real cross-resident VRAM gate. Only the outside
world is faked:

* the engine process (a mocked spawn; the engine's request handler waits on a gate, so an
  engine stays busy until the test releases it),
* the free-VRAM reading. ``VramWorld`` answers the way nvidia-smi would: every card has a
  total, every loaded engine takes the model footprint off its card, and a process outside
  the manager can take memory too. The manager's own accounting of engines that are still
  booting stays the manager's. The numbers are chosen so that a precise number of engines
  fit, and the test says why (see ``CARD_TOTAL_MIB``, ``NEED_MIB``, ``FLOOR_MIB``),
* the re-check clock (``_co_residence_now``): ages and intervals are made by moving it,
  never by sleeping.

One re-check pass is triggered by ``recheck_once`` through the module's ``run_recheck``.

How much VRAM a card must have: the card picker admits a card when
``free - footprint_still_booting - FLOOR_MIB >= NEED_MIB``. With a 5000 MiB card, a 4000 MiB
model and a 512 MiB floor, an empty card fits one engine (4488 >= 4000) and a card that
already holds one does not (1000 - 512 = 488 < 4000). So N cards hold exactly N engines.

The decline reasons of the re-check, and the cases that produce each:

* ``vram_refused``: the box has cards but none has the room. Two shapes, both tested: the
  cards are simply full (the card picker finds no card although cards are visible), and a
  race (the picker finds a card and memory is taken before the cross-resident VRAM gate runs,
  so the gate refuses it; modelled by a process outside the manager taking memory on the card
  the instant it is picked).
* ``no_card``: there is no card to talk about: the probe shows no cards at all, or the tag is
  not one card per engine (a card-spanning split), so it cannot be placed as a second engine.

The fixture control test documents why both VRAM shapes exist: with one consistent reading the
picker is the stricter check (it also subtracts the safety floor), so a plain shortage stops
at the picker and the gate only refuses on a race.

The decline reason is parsed from the log line (``reason=...``), never matched as a substring
of the whole line.
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
from turbohaul.fsm import transition as fsm_transition
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

MIB = 1024 * 1024
TAG = "tag-a"
NEED_MIB = 4000          # the model footprint (the manifest's expected VRAM)
FLOOR_MIB = 512          # queue.safety_min_free_vram_mib: kept free on every card
CARD_TOTAL_MIB = 5000    # one engine per card: 5000 - 512 >= 4000, but (5000 - 4000) - 512 < 4000
DECLINED = "CO_RESIDENCE_DECLINED"
SETTLE_TIMEOUT_S = 3.0
BASE_PORT = 59700


# --------------------------------------------------------------------------------------
# fixture: a real manager, a fake free-VRAM world, a fake engine process, a fake clock
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
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=BASE_PORT),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=cap, grace_seconds=3600,
            max_grace_extensions=50, idle_hot_load_seconds=3600, safety_min_free_vram_mib=FLOOR_MIB,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _write_manifest(boot, tag, split="none"):
    """One card per engine by default: the manager picks the card (auto_place) and the engine uses
    one (split none). Any other split_mode is a card-spanning engine."""
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 1000 * MIB,
        "context_size": 512,
        "expected_vram_bytes": NEED_MIB * MIB,
        "auto_place": True,
        "llama_server_flags": {"split_mode": split, "main_gpu": 0},
    }))


class Box:
    """The faked outside world around one real manager, and what it was asked."""

    def __init__(self, mgr, cards):
        self.mgr = mgr
        self.totals = [CARD_TOTAL_MIB] * cards
        self.hog = [0] * cards            # memory taken by a process outside the manager
        self.grab_on_next_pick = 0        # MiB an outside process takes on the card the picker returns next
        self.clock = [time.monotonic()]
        self.gate = asyncio.Event()       # an engine's request handler waits on it: set = engines finish
        self.entered = []                 # pid of the engine, in the order each began serving a request
        self.picks = []                   # (tag, (card, split)) answers of the card picker
        self.pid = 93_000
        self.blind = False                # True: the probe shows no cards at all (nvidia-smi unreadable)

    # -- the fake nvidia-smi ------------------------------------------------------------
    def free_mib(self):
        """Free MiB per card: total, minus every LOADED engine's footprint, minus the outside process."""
        if self.blind:
            return []
        loaded = {}
        for r in list(self.mgr._model_residents()):
            if r.handle is not None:
                loaded[r.main_gpu] = loaded.get(r.main_gpu, 0) + NEED_MIB
        return [t - loaded.get(g, 0) - self.hog[g] for g, t in enumerate(self.totals)]

    def now(self):
        return self.clock[0]

    def cards_of(self, tag=TAG):
        return sorted(r.main_gpu for r in self.mgr._instances_for(tag))


def _engine_process(box):
    def spawn(binary, gguf, port, model_tag, argv, **kw):
        box.pid += 1
        proc = MagicMock()
        proc.pid = box.pid
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def health(*a, **k):
        return True

    async def sigterm(handle, **k):
        return True, "sigterm-clean"

    async def vram(**k):
        return True, 100

    async def complete(slot, handle):
        box.entered.append(handle.pid)
        await box.gate.wait()
        return {"ok": True}

    return dict(spawn_fn=spawn, health_fn=health, sigterm_fn=sigterm, vram_fn=vram, complete_fn=complete)


@pytest.fixture(autouse=True)
def _no_real_orphan_reaper(monkeypatch):
    """A teardown must never scan the host's processes: the reapers are replaced by no-ops."""
    monkeypatch.setattr("turbohaul.manager.boot_orphan_reaper", lambda **k: {"reaped": 0})
    monkeypatch.setattr("turbohaul.manager.intra_lifetime_orphan_scan", lambda **k: {"reaped": 0})


async def _box(tmp_path, monkeypatch, caplog, *, cap, cards, split="none"):
    boot, runtime = _boot_runtime(tmp_path, cap)
    _write_manifest(boot, TAG, split)
    box = Box(None, cards)
    mgr = TurbohaulManager(boot, runtime, **_engine_process(box))
    box.mgr = mgr
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", box.free_mib)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", box.free_mib)
    monkeypatch.setattr(mgr, "_co_residence_now", box.now, raising=False)
    real_pick = mgr._pick_card_for_new_instance

    def pick(tag, need):
        answer = real_pick(tag, need)
        box.picks.append((tag, answer))
        if box.grab_on_next_pick and answer[0] is not None:
            box.hog[answer[0]] += box.grab_on_next_pick     # an outside process allocates right after the pick
            box.grab_on_next_pick = 0
        return answer

    mgr._pick_card_for_new_instance = pick
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="turbohaul")
    return mgr, box


def _request(box, tag, age):
    """A staged request that has been waiting ``age`` seconds on the re-check clock."""
    slot = Slot.new(tag, prompt="p", thread_id="")
    slot.client_meta = {}
    slot.completion_future = asyncio.get_running_loop().create_future()
    fsm_transition(slot, SlotState.STAGED)
    slot.created_at = box.clock[0] - age
    return slot


def _inbox(r):
    return list(r.inbox._queue)


async def _settle(box, timeout_s=SETTLE_TIMEOUT_S):
    """Wait until every engine the manager holds is loaded and has begun serving its first request."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        residents = list(box.mgr._model_residents())
        if all(r.handle is not None for r in residents) and len(box.entered) >= len(residents):
            return True
        await asyncio.sleep(0.005)
    return False


async def _start_busy_engine(box):
    """The first engine of the tag, started the normal way (arrival routing) and busy on a request."""
    first = _request(box, TAG, 0.0)
    await box.mgr._route_or_reserve(first)
    assert await _settle(box), "the first engine never began serving"
    (r,) = box.mgr._instances_for(TAG)
    assert r.state is ResidentState.ACTIVE and r.inbox is not None
    return r


async def recheck_once(mgr):
    """One re-check pass: through the module's ``run_recheck``, else the manager's own re-check method."""
    try:
        from turbohaul import co_residence
    except ImportError:
        co_residence = None
    run = getattr(co_residence, "run_recheck", None)
    if run is not None:
        await run(mgr)
        return
    legacy = getattr(mgr, "_co_residence_recheck", None)
    if legacy is not None:
        await legacy()


async def _pass(box, advance_s):
    """Move the clock, run one re-check pass, and let any engine it started finish loading."""
    box.clock[0] += advance_s
    await recheck_once(box.mgr)
    await _settle(box)


def _declines(caplog):
    """Every CO_RESIDENCE_DECLINED line so far, as {key: value} (the leading token dropped)."""
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith(DECLINED):
            fields = dict(part.split("=", 1) for part in msg.split()[1:] if "=" in part)
            fields["_line"] = msg
            out.append(fields)
    return out


async def _finish(box):
    box.gate.set()
    await box.mgr.shutdown()


# --------------------------------------------------------------------------------------
# the fixture can express both outcomes (a control for everything below)
# --------------------------------------------------------------------------------------

async def test_fixture_four_cards_hold_four_engines_and_a_taken_card_is_refused_by_the_gate_alone(
    tmp_path, monkeypatch, caplog,
):
    """The fake VRAM world itself does not cause the later failures: without the re-check in play, it lets
    the picker and the gate admit a fourth engine on the free card, then refuse it once memory is taken."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=4)
    try:
        assert mgr._read_model_footprint(TAG)[0] == NEED_MIB, "the manifest must give the footprint the numbers assume"
        r1 = await _start_busy_engine(box)
        for card in (1, 2):                       # two more engines, started directly on their own cards
            async with mgr._registry_lock:
                r = await mgr._reserve_and_start_locked(_request(box, TAG, 0.0), main_gpu_override=card,
                                                        split_mode_override="none")
            assert r is not None
        assert await _settle(box)
        assert box.cards_of() == [0, 1, 2] and r1 in mgr._instances_for(TAG)
        assert box.free_mib() == [1000, 1000, 1000, CARD_TOTAL_MIB]
        picked = mgr._pick_card_for_new_instance(TAG, NEED_MIB)
        assert picked == (3, "none"), "the free card must be the one the picker chooses"
        assert mgr._vram_admits_locked(NEED_MIB, 1, 3, "none") is True
        box.hog[3] = 3000                          # an outside process takes memory on that card
        assert mgr._vram_admits_locked(NEED_MIB, 1, 3, "none") is False
        box.hog[3] = CARD_TOTAL_MIB                # every card is now taken: the picker has nothing to pick
        assert mgr._pick_card_for_new_instance(TAG, NEED_MIB) == (None, "none")
    finally:
        await _finish(box)


# --------------------------------------------------------------------------------------
# expected result: three engines that fit all load, a fourth that does not fit stays out
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    pytest.param("race", id="race_outside_process_takes_the_picked_card"),
    pytest.param("cards_full", id="cards_full_plain_shortage"),
])
async def test_three_fitting_engines_load_and_a_fourth_that_does_not_fit_is_declined_for_vram(
    tmp_path, monkeypatch, caplog, case,
):
    """Box budget 3, re-check on, a 4000 MiB model (one card per engine), cards of 5000 MiB. The first
    engine is busy; three more requests wait in its inbox. Passes of the re-check load the second and
    third engine because each fits. The fourth does not fit, by VRAM and not by count: the box stays
    at three engines, the waiting request stays where it is, and exactly one decline line says
    ``vram_refused``, not ``box_full`` or ``cap`` (the budget of 3 is spent, and it is not what stopped
    the engine). Two ways for the fourth not to fit:

    * ``race``: four cards, the fourth is free when the picker chooses it, then an outside process takes
      memory on it, so the cross-resident VRAM gate refuses the picked card;
    * ``cards_full``: three cards, each already holds an engine, so the picker finds no card although
      cards are visible (a plain shortage, no race)."""
    race = case == "race"
    cards = 4 if race else 3
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=cards)
    try:
        r1 = await _start_busy_engine(box)
        assert mgr.runtime.queue.max_parallel_sidecars == 3 and box.cards_of() == [0]
        p1, p2, p3 = (_request(box, TAG, age) for age in (40.0, 30.0, 20.0))
        for p in (p1, p2, p3):
            r1.inbox.put_nowait(p)
        assert box.free_mib() == [1000] + [CARD_TOTAL_MIB] * (cards - 1)
        assert mgr._pick_card_for_new_instance(TAG, NEED_MIB)[0] in (1, 2), "a free card must exist"
        box.picks.clear()

        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 2, "the second engine never loaded on the free card"
        assert p1 not in _inbox(r1) and _inbox(r1) == [p2, p3], "the oldest waiting request must move"
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 3, "the third engine never loaded on the free card"
        assert box.cards_of() == [0, 1, 2], "one engine per card"
        assert _inbox(r1) == [p3]
        assert _declines(caplog) == [], "an admitted engine logs no decline"
        assert len(mgr._model_residents()) == mgr.runtime.queue.max_parallel_sidecars, "the budget is spent"

        if race:
            box.grab_on_next_pick = 3000           # the outside process: card 3 drops to 2000 MiB, below 4000
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 3, "a fourth engine started although it does not fit"
        assert _inbox(r1) == [p3], "a declined request must stay where it was"
        declines = _declines(caplog)
        assert len(declines) == 1, f"one line per decline, got {[d['_line'] for d in declines]}"
        assert declines[0]["reason"] == "vram_refused", declines[0]["_line"]
        assert declines[0]["tag"] == TAG and declines[0]["via"] == "recheck", declines[0]["_line"]
        picked = box.picks[-1][1]                  # what the picker answered on that probe: it says why
        if race:
            assert picked == (3, "none"), "the picker must have chosen the free card, and the gate refused it"
        else:
            assert picked == (None, "none"), "the picker must have found no card"
            avail = mgr._card_avail_mib()
            assert len(avail) == cards and all(a < NEED_MIB for a in avail), f"cards visible, none with room: {avail}"
    finally:
        await _finish(box)


async def test_the_same_fourth_engine_loads_when_its_card_stays_free_past_the_budget(
    tmp_path, monkeypatch, caplog,
):
    """The control of the test above: nothing takes the memory, so the fourth engine fits and loads
    although the box budget is 3. It is the fit that decides, in both directions."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=4)
    try:
        r1 = await _start_busy_engine(box)
        for age in (40.0, 30.0, 20.0):
            r1.inbox.put_nowait(_request(box, TAG, age))
        for expected in (2, 3):
            await _pass(box, 11.0)
            assert len(mgr._instances_for(TAG)) == expected, f"engine {expected} never loaded"
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 4, "a fitting engine was not started because the budget is 3"
        assert box.cards_of() == [0, 1, 2, 3]
        assert _inbox(r1) == []
        assert _declines(caplog) == []
    finally:
        await _finish(box)


# --------------------------------------------------------------------------------------
# a box budget of 1 does not stop the re-check either
# --------------------------------------------------------------------------------------

async def test_a_budget_of_one_box_grows_one_engine_at_a_time_while_each_fits_then_declines_for_vram(
    tmp_path, monkeypatch, caplog,
):
    """Box budget 1, three cards. One busy engine, three waiting requests. Each pass adds an engine
    (1 -> 2 -> 3) because each fits. The next pass finds every card taken by an engine: the picker
    returns no card although the cards are visible, so the one decline line says ``vram_refused``
    (a plain VRAM shortage), and the box stays at three engines."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=1, cards=3)
    try:
        r1 = await _start_busy_engine(box)
        assert mgr.runtime.queue.max_parallel_sidecars == 1
        waiting = [_request(box, TAG, age) for age in (40.0, 30.0, 20.0)]
        for p in waiting:
            r1.inbox.put_nowait(p)
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 2, "a budget of 1 stopped the second engine"
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 3, "a budget of 1 stopped the third engine"
        assert box.cards_of() == [0, 1, 2]
        assert _declines(caplog) == []
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 3, "an engine started although every card is taken"
        assert _inbox(r1) == [waiting[2]]
        declines = _declines(caplog)
        assert len(declines) == 1, f"one line per decline, got {[d['_line'] for d in declines]}"
        assert declines[0]["reason"] == "vram_refused", declines[0]["_line"]
        assert declines[0]["via"] == "recheck"
        assert box.picks[-1][1] == (None, "none"), "the picker must have found no card on that probe"
        avail = mgr._card_avail_mib()
        assert len(avail) == 3 and all(a < NEED_MIB for a in avail), f"cards visible, none with room: {avail}"
    finally:
        await _finish(box)


# --------------------------------------------------------------------------------------
# no_card: there is no card to place a second engine on
# --------------------------------------------------------------------------------------

async def test_a_probe_that_shows_no_cards_declines_with_no_card(tmp_path, monkeypatch, caplog):
    """The first engine is up and busy, a request waits behind it, and the free-VRAM reading now shows
    no cards at all (the probe is unreadable). Nothing can be placed: no second engine, the request
    stays, one decline line with ``no_card`` (not ``vram_refused``: there are no cards to be short of)."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=3)
    try:
        r1 = await _start_busy_engine(box)
        waiting = _request(box, TAG, 30.0)
        r1.inbox.put_nowait(waiting)
        box.blind = True
        assert mgr._card_avail_mib() == [], "the fixture must show no cards"
        box.picks.clear()
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 1, "an engine started although no card is visible"
        assert _inbox(r1) == [waiting]
        declines = _declines(caplog)
        assert len(declines) == 1, f"one line per decline, got {[d['_line'] for d in declines]}"
        assert declines[0]["reason"] == "no_card", declines[0]["_line"]
        assert declines[0]["tag"] == TAG and declines[0]["via"] == "recheck", declines[0]["_line"]
        assert box.picks and box.picks[-1][1] == (None, "none"), "the picker must have found no card"
    finally:
        await _finish(box)


async def test_a_tag_that_is_not_one_card_per_engine_gets_no_second_engine_and_a_no_card_line(
    tmp_path, monkeypatch, caplog,
):
    """The manifest asks for a card-spanning split, so a second copy has no card of its own. Cards are
    free, a request waits behind the busy engine: still no second engine, one ``no_card`` line."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=3, split="layer")
    try:
        r1 = await _start_busy_engine(box)
        assert mgr._route_cap_and_placement(TAG)[1] is False, "the manifest must not be one card per engine"
        assert box.free_mib()[1:] == [CARD_TOTAL_MIB] * 2, "free cards exist, so VRAM is not what stops it"
        waiting = _request(box, TAG, 30.0)
        r1.inbox.put_nowait(waiting)
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 1, "a card-spanning tag got a second engine"
        assert _inbox(r1) == [waiting]
        declines = _declines(caplog)
        assert len(declines) == 1, f"one line per decline, got {[d['_line'] for d in declines]}"
        assert declines[0]["reason"] == "no_card", declines[0]["_line"]
        assert declines[0]["tag"] == TAG and declines[0]["via"] == "recheck", declines[0]["_line"]
    finally:
        await _finish(box)


# --------------------------------------------------------------------------------------
# the arrival path stays capped, and says why
# --------------------------------------------------------------------------------------

async def test_an_arrival_at_a_budget_of_one_loads_no_second_engine_and_logs_the_reason_once(
    tmp_path, monkeypatch, caplog,
):
    """Box budget 1, one busy engine, free cards. A same-model request that arrives is routed by the real
    ``_route_or_reserve``: no second engine, it waits in the busy engine's inbox, and one decline line
    names the reason. The reason is ``cap``: the tag already runs as many engines as the budget derived
    for it (``_effective_cap`` answers with the engine count once the box budget is spent), which is
    tested before the box-wide count. A second arrival inside the throttle window logs nothing more, and
    the same decline logged again once the window is over."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=1, cards=3)
    try:
        r1 = await _start_busy_engine(box)
        started = []
        real_reserve = mgr._reserve_and_start_locked

        async def reserve(*a, **k):
            started.append(a)
            return await real_reserve(*a, **k)

        mgr._reserve_and_start_locked = reserve
        arrival = _request(box, TAG, 0.0)
        await mgr._route_or_reserve(arrival)
        assert started == [] and len(mgr._instances_for(TAG)) == 1, "the arrival path started a second engine"
        assert _inbox(r1) == [arrival], "the arrival waits in the busy engine's inbox"
        declines = _declines(caplog)
        assert len(declines) == 1, f"one line for the decline, got {[d['_line'] for d in declines]}"
        assert declines[0]["reason"] == "cap" and declines[0]["via"] == "arrival", declines[0]["_line"]

        second = _request(box, TAG, 0.0)
        await mgr._route_or_reserve(second)
        assert _inbox(r1) == [arrival, second] and len(_declines(caplog)) == 1, "the same decline is throttled"
        box.clock[0] += 61.0
        third = _request(box, TAG, 0.0)
        await mgr._route_or_reserve(third)
        assert len(_declines(caplog)) == 2, "the decline is logged again after the throttle window"
        assert started == [] and len(mgr._instances_for(TAG)) == 1
    finally:
        await _finish(box)


# --------------------------------------------------------------------------------------
# the bounded shape: one attempt per interval, one probe per tag per no-room memo window
# --------------------------------------------------------------------------------------

async def test_inside_one_debounce_interval_a_second_pass_does_nothing_and_the_next_interval_works(
    tmp_path, monkeypatch, caplog,
):
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=4)
    try:
        r1 = await _start_busy_engine(box)
        for age in (40.0, 30.0, 20.0):
            r1.inbox.put_nowait(_request(box, TAG, age))
        box.picks.clear()
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 2, "the first pass admitted nothing"
        probed = len(box.picks)
        assert probed >= 1, "the first pass never asked the card picker"
        await _pass(box, 5.0)                       # 5 s later: inside the 10 s interval
        assert len(box.picks) == probed, "a second pass inside the interval probed again"
        assert len(mgr._instances_for(TAG)) == 2, "a second pass inside the interval started an engine"
        await _pass(box, 6.0)                       # 11 s after the first pass: the interval is over
        assert len(mgr._instances_for(TAG)) == 3, "the pass after the interval admitted nothing"
        assert len(box.picks) > probed
    finally:
        await _finish(box)


async def test_a_tag_with_no_room_is_not_probed_again_inside_the_memo_window_but_is_after_it(
    tmp_path, monkeypatch, caplog,
):
    """Three cards, three busy engines, one request still waiting. The pass that finds no room notes
    it. Passes 11 s and 22 s later (the interval is over, the 60 s memo is not) probe nothing and log
    nothing. A pass 71 s after the verdict probes the tag again and logs again."""
    mgr, box = await _box(tmp_path, monkeypatch, caplog, cap=3, cards=3)
    try:
        r1 = await _start_busy_engine(box)
        for age in (40.0, 30.0, 20.0):
            r1.inbox.put_nowait(_request(box, TAG, age))
        await _pass(box, 11.0)
        await _pass(box, 11.0)
        assert len(mgr._instances_for(TAG)) == 3, "the box never reached three engines"
        box.picks.clear()
        await _pass(box, 11.0)                      # the verdict: every card is taken
        assert len(box.picks) == 1 and len(_declines(caplog)) == 1, "the no-room verdict was not reached"
        await _pass(box, 11.0)
        await _pass(box, 11.0)
        assert len(box.picks) == 1, "the tag was probed again inside the no-room memo window"
        assert len(_declines(caplog)) == 1, "a memoed tag logged a decline"
        await _pass(box, 49.0)                      # 71 s after the verdict
        assert len(box.picks) == 2, "the tag was not probed again after the memo window"
        assert len(_declines(caplog)) == 2
        assert len(mgr._instances_for(TAG)) == 3
    finally:
        await _finish(box)
