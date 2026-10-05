"""A request parked in an engine's inbox keeps its Fast Lane claim, on the two routes that hand
a request to an engine other than the single already-loaded one.

The single-engine hit is covered elsewhere. Two more places in the manager put a request in an
engine's inbox, and each used to release the request's claim at that moment, so the waiting
request became invisible to the Fast Lane machinery while it sat there:

* the multi-engine route (the ``_route_to`` closure inside ``_route_or_reserve``): a tag with
  several running engines hands the request to the engine picked by session stickiness or by
  load;
* the cold reservation (``_reserve_and_start_locked``): a tag with no running engine gets a new
  one reserved for the request, and the request waits while that engine spawns.

At both places the claim must stay registered, marked as parked on that engine's resident key,
until the request's own turn starts on it, and the engine's rank identity must keep naming the
client of the turn that is running (it is not stamped at the hand-off).

Everything runs real manager code (real submit, dispatcher, routing, drivers and claim registry)
with a fake engine process and a fake completion call that blocks per request, so a test decides
exactly when each turn ends. A spy on the claim-parking call records which function called it,
which proves the route under test is the one that ran. One process, asyncio only, two engines at
most, bounded waits.
"""

import asyncio
import contextlib
import sys

from _fastlane_fixture import wait_until
from _multiinstance_support import _manifest, _new_manager, _probe

from turbohaul.config import FastLaneConfig, FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState

MODEL = "duo"

# Priority order: index 0 outranks index 1, which outranks index 2.
CLAIMANT_IP = "10.0.0.4"
HOLDER_A_IP = "10.0.0.5"
HOLDER_B_IP = "10.0.0.6"
IPS = (CLAIMANT_IP, HOLDER_A_IP, HOLDER_B_IP)

CLAIMANT_THREAD = "thread-claimant"

# Bounds for things that must happen promptly; they are limits, not delays.
PROMPT_S = 1.5
SETUP_S = 5.0


def meta(ip):
    return {"ip": ip, "is_main": True}


class Scene:
    """One manager over a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit``. The fake completion call records
    the order in which turns start, the engine handle each turn ran on and whether the
    request's claim was still registered at the moment its turn began, then blocks until the
    test releases that request. A spy around the claim-parking call records every park made
    for a labelled request: the engine's resident key and the name of the function that made
    the call.
    """

    def __init__(self, tmp_path, *, budget, auto_place):
        self.mgr, self.boot, self.spawn_calls = _new_manager(tmp_path, budget=budget)
        self.mgr.runtime.fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address=ip, tag_ranks=FastLaneTagRanks(main=1)) for ip in IPS],
        )
        self.mgr.runtime.queue.grace_seconds = 2
        self.mgr.runtime.queue.max_grace_extensions = 50
        self.mgr.runtime.queue.idle_hot_load_seconds = 600
        if auto_place:
            # One card per engine: this is what sends the tag down the multi-engine route.
            _manifest(self.boot, MODEL, auto_place=True,
                      llama_server_flags={"split_mode": "none"})
        else:
            _manifest(self.boot, MODEL)
        self.slots = {}           # label -> Slot
        self.labels = {}          # slot_id -> label
        self.starts = []          # slot_ids in the order their turns started
        self.handles = {}         # slot_id -> engine handle the turn ran on
        self.claim_at_start = {}  # slot_id -> claim still registered when the turn began
        self.parks = []           # (slot_id, resident_key, calling function) per park call
        self._gates = {}
        self._open = False
        self.mgr._complete_fn = self._complete
        real_park = self.mgr._park_fastlane_claim_locked

        def spy_park(slot, r):
            caller = sys._getframe(1).f_code.co_name
            self.parks.append((slot.slot_id, r.resident_key, caller))
            return real_park(slot, r)

        self.mgr._park_fastlane_claim_locked = spy_park

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def _complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        self.handles[slot.slot_id] = handle
        self.claim_at_start[slot.slot_id] = any(
            c["slot"] is slot for c in self.mgr._fastlane_claims.values()
        )
        if not self._open:
            await self.gate(slot.slot_id).wait()
        return {"ok": True, "model": handle.model_tag}

    def release(self, label):
        self.gate(self.slots[label].slot_id).set()

    def open_everything(self):
        self._open = True
        for ev in self._gates.values():
            ev.set()

    # -- requests --------------------------------------------------------
    async def send(self, label, ip, thread):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=meta(ip),
            wait_for_completion=True,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        """Labels of the turns that have started, in start order."""
        return [self.labels.get(s, s) for s in self.starts]

    async def started(self, label, timeout=PROMPT_S):
        """True when ``label``'s turn starts within ``timeout`` seconds."""
        try:
            await wait_until(lambda: label in self.order(), timeout=timeout)
        except AssertionError:
            return False
        return True

    # -- claims ----------------------------------------------------------
    def claim_of(self, label):
        """The registered claim record of the labelled request, or None."""
        slot = self.slots[label]
        for c in self.mgr._fastlane_claims.values():
            if c["slot"] is slot:
                return c
        return None

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (the surface the queue tab reads)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]

    def parks_of(self, label):
        slot_id = self.slots[label].slot_id
        return [(label, key, caller) for sid, key, caller in self.parks if sid == slot_id]

    async def settle(self, timeout=SETUP_S):
        """Wait until every claim is gone (every request has had its turn)."""
        await wait_until(lambda: not self.mgr._fastlane_claims, timeout=timeout)


@contextlib.asynccontextmanager
async def build_scene(tmp_path, *, budget, auto_place, cards):
    """A manager with the dispatcher running and ``cards`` cards reporting plenty of free VRAM.

    The card probe is replaced at both places the manager reads it, so no real host reading is
    involved.
    """
    scene = Scene(tmp_path, budget=budget, auto_place=auto_place)
    p1, p2 = _probe([20000] * cards)
    scene.mgr._worker_task = asyncio.create_task(scene.mgr.worker_loop())
    try:
        with p1, p2:
            yield scene
    finally:
        scene.open_everything()
        await scene.mgr.shutdown()


# --------------------------------------------------------------------------------------
# (1) the multi-engine route
# --------------------------------------------------------------------------------------

async def test_multi_engine_route_parks_the_claim_on_the_chosen_engine_and_releases_it_at_the_turn_start(
    tmp_path,
):
    """Two engines of one model are running, each mid-turn for a lower-ranked client. A
    higher-ranked request whose session is pinned to the SECOND engine is routed there by the
    multi-engine route (the ``_route_to`` closure) and waits in that engine's inbox.

    While it waits: its claim is still registered, marked as parked on the second engine's
    resident key (not the first engine's), listed in the claims snapshot, and the engine's
    rank identity still names the client of the turn that is running. When that engine's
    running turn ends, the request's turn starts on the same engine, and at that moment the
    claim is already gone; nothing is left at the end. The spy proves the closure made the
    park call (the single-engine path and the cold reservation are other functions)."""
    async with build_scene(tmp_path, budget=2, auto_place=True, cards=2) as s:
        mgr = s.mgr
        await s.send("HA", HOLDER_A_IP, "thread-a")
        assert await s.started("HA", timeout=SETUP_S), "setup: the first engine's turn never started"
        await s.send("HB", HOLDER_B_IP, "thread-b")
        assert await s.started("HB", timeout=SETUP_S), "setup: the second engine's turn never started"
        engines = {r.resident_key: r for r in mgr._instances_for(MODEL)}
        assert sorted(engines) == [MODEL, f"{MODEL}#1"], (
            f"setup: expected two engines of one tag, got {sorted(engines)}"
        )
        engine_a, engine_b = engines[MODEL], engines[f"{MODEL}#1"]
        assert s.handles[s.slots["HA"].slot_id] is not s.handles[s.slots["HB"].slot_id]
        assert engine_a.active_slot is s.slots["HA"] and engine_b.active_slot is s.slots["HB"]
        assert len(mgr._model_residents()) == mgr.runtime.queue.max_parallel_sidecars, (
            "setup: the box budget must be spent so that the route reuses an existing engine"
        )

        # Pin the claimant's session to the second engine, the way a spawn records it.
        async with mgr._registry_lock:
            mgr._remember_affinity(MODEL, f"tid:{CLAIMANT_THREAD}", engine_b.resident_key)
        claimant = await s.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        await wait_until(lambda: engine_b.inbox.qsize() == 1, timeout=PROMPT_S)
        assert engine_a.inbox.qsize() == 0, "setup: the request must not be on the first engine"
        assert claimant.slot_id not in {x.slot_id for x in mgr.queue._staging}

        claim = s.claim_of("C")
        assert claim is not None, (
            "a request parked in an engine's inbox by the multi-engine route must keep its claim; "
            "it was released when the request was handed to the engine"
        )
        assert claim.get("parked_on") == engine_b.resident_key, (
            f"the claim is parked on {claim.get('parked_on')!r}, not on the engine whose inbox "
            f"holds the request ({engine_b.resident_key!r})"
        )
        assert claimant.slot_id in s.snapshot_ids(), (
            "the parked request is missing from the claims snapshot while it waits in the inbox"
        )
        assert engine_b.rank_client_meta["ip"] == HOLDER_B_IP, (
            f"while its holder is mid-turn the engine is stamped with {engine_b.rank_client_meta['ip']}, "
            "the parked request's client"
        )
        assert engine_a.rank_client_meta["ip"] == HOLDER_A_IP
        assert s.parks_of("C") == [("C", engine_b.resident_key, "_route_to")], (
            f"the park call did not come from the multi-engine route's closure: {s.parks_of('C')}"
        )
        assert s.order() == ["HA", "HB"], f"a turn started while both engines were mid-turn: {s.order()}"

        # The running turn on the second engine ends: the request is served next on that engine.
        s.release("HB")
        assert await s.started("C", timeout=PROMPT_S), (
            f"the parked request never started after its engine's turn ended (order {s.order()})"
        )
        assert s.handles[claimant.slot_id] is s.handles[s.slots["HB"].slot_id], (
            "the parked request ran on a different engine than the one it was parked on"
        )
        assert s.claim_at_start[claimant.slot_id] is False, (
            "the claim was still registered when the request's own turn began"
        )
        assert engine_b.rank_client_meta["ip"] == CLAIMANT_IP, (
            "once the parked request's own turn runs the engine must name its client"
        )
        s.open_everything()
        done, _ = await asyncio.wait({claimant.completion_future}, timeout=PROMPT_S)
        assert done and claimant.completion_future.result()["ok"]
        await s.settle()
        assert not mgr._fastlane_claims, f"claims left at the end: {list(mgr._fastlane_claims)}"
        assert s.snapshot_ids() == []


# --------------------------------------------------------------------------------------
# (2) the cold reservation
# --------------------------------------------------------------------------------------

async def test_cold_reservation_keeps_the_claim_while_the_engine_spawns(tmp_path):
    """The request arrives for a model with no running engine. The manager reserves a new
    engine for it (``_reserve_and_start_locked``) and the request waits for that engine to
    spawn. The fake spawn is held on a latch (the engine's health probe does not answer), so
    the request is still waiting: its claim is still registered (it was registered when the
    request was submitted), marked as parked on the new engine's resident key, and listed in
    the claims snapshot. When the latch is released the engine comes up, the request's turn
    starts and at that moment the claim is already gone; nothing is left at the end. The spy
    proves the reservation made the park call."""
    async with build_scene(tmp_path, budget=1, auto_place=False, cards=1) as s:
        mgr = s.mgr
        entered, hold = asyncio.Event(), asyncio.Event()

        async def held_health(*a, **k):
            entered.set()
            await hold.wait()
            return True

        mgr._wait_healthy = held_health
        try:
            claimant = await s.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
            assert s.claim_of("C") is not None, "setup: submit registers the request's claim"
            await asyncio.wait_for(entered.wait(), timeout=SETUP_S)
            engines = mgr._instances_for(MODEL)
            assert len(engines) == 1, f"setup: expected one new engine, got {len(engines)}"
            engine = engines[0]
            assert engine.state is ResidentState.RESERVED_LOADING, (
                f"setup: the engine must still be loading, not {engine.state}"
            )
            assert s.starts == [], "setup: the request's turn must not have started while the spawn is held"

            claim = s.claim_of("C")
            assert claim is not None, (
                "a request waiting for its reserved engine to spawn must keep its claim; "
                "it was released when the engine was reserved"
            )
            assert claim.get("parked_on") == engine.resident_key, (
                f"the claim is parked on {claim.get('parked_on')!r}, not on the engine reserved "
                f"for the request ({engine.resident_key!r})"
            )
            assert claimant.slot_id in s.snapshot_ids(), (
                "the waiting request is missing from the claims snapshot while its engine spawns"
            )
            assert s.parks_of("C") == [("C", engine.resident_key, "_reserve_and_start_locked")], (
                f"the park call did not come from the cold reservation: {s.parks_of('C')}"
            )
            assert len(s.spawn_calls) == 1, f"setup: expected exactly one spawn, got {len(s.spawn_calls)}"

            # Let the engine come up: the request's turn starts and its claim is gone by then.
            hold.set()
            assert await s.started("C", timeout=PROMPT_S), (
                f"the request's turn never started after the engine came up (order {s.order()})"
            )
            assert s.claim_at_start[claimant.slot_id] is False, (
                "the claim was still registered when the request's own turn began"
            )
            s.open_everything()
            done, _ = await asyncio.wait({claimant.completion_future}, timeout=PROMPT_S)
            assert done and claimant.completion_future.result()["ok"]
            await s.settle()
            assert not mgr._fastlane_claims, f"claims left at the end: {list(mgr._fastlane_claims)}"
            assert s.snapshot_ids() == []
        finally:
            hold.set()
