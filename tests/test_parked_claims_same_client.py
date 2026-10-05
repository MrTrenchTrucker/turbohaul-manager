"""Several requests of ONE client parked in a resident's inbox each keep a Fast Lane claim.

The claim registry deduplicates on (the client's rule address, the model tag): a second
request of the same client for the same model does not get a claim of its own, it collapses
onto the claim of the first one. If the claim of a parked request is released when that
request's own turn starts, every OTHER request of the same client that is still parked in the
inbox is left with no claim at all, and the Fast Lane machinery cannot see it any more (the
same invisibility that parking a request used to cause, one request later).

The requirement pinned here: every parked request keeps a claim until its OWN turn starts.
These tests assert outcomes only (which slots hold a claim at which moment, in which order the
turns start); they do not depend on how the claim is kept (re-armed for the next parked
request, or shared and kept until the last one starts).

Everything runs real manager code (real dispatcher, drivers, grace loop, claim registry) with
a fake engine process and a fake completion call that blocks per request, so a test decides
exactly when each turn ends. The fake completion call also records, at the instant a turn
starts, which slots hold a claim, so "at the moment its turn starts" is observed from inside
the turn and not guessed from outside. One process, asyncio only, at most three parked
requests, bounded waits, no sleep above one second. The engine's slot save is replaced by a
recorder so no request ever leaves the process.
"""

import asyncio
import contextlib

import pytest
from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState

MODEL = "m1"

# Priority order: index 0 outranks index 1, which outranks index 2.
CLIENT_A_IP = "10.0.0.4"
CLIENT_B_IP = "10.0.0.3"
HOLDER_IP = "10.0.0.5"
RULES = ranked_rules((CLIENT_A_IP, 1), (CLIENT_B_IP, 1), (HOLDER_IP, 1))

HOLDER_THREAD = "thread-holder"

# How long a test waits for something that must happen promptly, and for something that
# legitimately waits behind a short grace window. Both are bounds, not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0


def meta(ip):
    return {"ip": ip, "is_main": True}


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake completion
    call records the order in which turns start (``starts``) and, at the moment each turn
    starts, the slot ids that hold a claim in the registry (``registry_at_start``) and the
    slot ids the claims snapshot lists (``snapshot_at_start``). It then blocks until the test
    releases that request.
    """

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns
        self.slots = {}               # label -> Slot
        self.labels = {}              # slot_id -> label
        self.starts = []              # slot_ids in the order their turns started
        self.registry_at_start = {}   # slot_id -> slot ids holding a claim when the turn began
        self.snapshot_at_start = {}   # slot_id -> slot ids the claims snapshot listed then
        self.unloads = []             # resident keys passed to _begin_unload_locked
        self.tail_requeues = []       # slot_ids handed to the tail re-queue helper
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        self.registry_at_start[slot.slot_id] = [
            c["slot"].slot_id for c in self.mgr._fastlane_claims.values()
        ]
        self.snapshot_at_start[slot.slot_id] = self.snapshot_ids()
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
    async def send(self, label, ip, thread, **kw):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=meta(ip),
            wait_for_completion=True, **kw,
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

    async def parked(self, labels, timeout=PROMPT_S):
        """Wait until every request in ``labels`` sits in the resident's inbox.

        The real HIT path of the dispatcher put them there: each one has left the staging
        queue and the inbox holds as many requests as were asked for.
        """
        r = resident_for(self.mgr, MODEL)
        slots = [self.slots[lbl] for lbl in labels]
        await wait_until(
            lambda: r.inbox.qsize() >= len(slots)
            and all(s not in self.mgr.queue._staging for s in slots),
            timeout=timeout,
        )
        return r

    async def turn_done(self, label, timeout=PROMPT_S):
        """Wait until ``label``'s turn has completed (it left ACTIVE)."""
        slot = self.slots[label]
        await wait_until(
            lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout,
        )

    # -- single-stepping the dispatcher ----------------------------------
    async def pause_dispatcher(self):
        """Stop the dispatcher task (the resident's driver keeps running).

        Afterwards a request submitted with ``send`` stays in staging, so a test can look at
        the state of requests that are queued but not routed.
        """
        task = self.mgr._worker_task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # -- claims ----------------------------------------------------------
    def registry_slot_ids(self):
        """Slot ids holding a registered claim right now (a plain read, never prunes)."""
        return [c["slot"].slot_id for c in self.mgr._fastlane_claims.values()]

    def has_claim(self, label):
        return self.slots[label].slot_id in self.registry_slot_ids()

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (the surface the queue tab reads)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]

    def claims_of_client(self, ip):
        """Claims registered for the client at ``ip`` (by the claimed slot's own client)."""
        return [
            c["slot"].slot_id for c in self.mgr._fastlane_claims.values()
            if (c["slot"].client_meta or {}).get("ip") == ip
        ]


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=1, max_grace_extensions=50,
                      idle_hot_load_seconds=600):
    """A manager with ONE box slot and the dispatcher running, over three ranked clients."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=1, grace_seconds=grace_seconds,
        max_grace_extensions=max_grace_extensions, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, MODEL, main_gpu=0)
    spawns = []
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})

    def counting_spawn(binary, gguf, port, model_tag, argv, **kw):
        spawns.append(model_tag)
        return spawn_fn(binary, gguf, port, model_tag, argv, **kw)

    holder = {}

    async def complete(slot, handle):
        return await holder["world"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=counting_spawn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    world = World(mgr, spawns)
    holder["world"] = world

    real_unload = mgr._begin_unload_locked
    real_tail = mgr._requeue_slots_tail_or_fail

    def spy_unload(r):
        world.unloads.append(r.resident_key)
        return real_unload(r)

    async def spy_tail(slots):
        world.tail_requeues.extend(s.slot_id for s in slots)
        return await real_tail(slots)

    async def fake_save(port, model_tag, slot=None, **kwargs):
        return True

    mgr._begin_unload_locked = spy_unload
    mgr._requeue_slots_tail_or_fail = spy_tail
    mgr._save_slot_kv = fake_save

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


async def start_holder(world):
    """The holder's first turn is running (its fake completion is blocked)."""
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    return h1


async def park(world, h1, specs):
    """Park each ``(label, ip, thread, kwargs)`` request in the holder's inbox, one at a time.

    Every request is submitted through the real ``submit`` and then waited for until the
    dispatcher's real HIT path has put it in the resident's inbox, before the next one is
    submitted, so the inbox order equals the submit order. Returns the resident.
    """
    parked_labels = []
    r = None
    for label, ip, thread, kw in specs:
        await world.send(label, ip, thread, **kw)
        parked_labels.append(label)
        r = await world.parked(parked_labels)
    assert r.state is ResidentState.ACTIVE and r.active_slot is h1, (
        "setup: the holder must still be mid-turn"
    )
    for label in parked_labels:
        assert world.slots[label].state is SlotState.STAGED, (
            f"setup: parked request {label} is still queued"
        )
    assert r.inbox.qsize() == len(parked_labels), (
        f"setup: expected {len(parked_labels)} parked requests, the inbox holds {r.inbox.qsize()}"
    )
    return r


def same_client(*labels, ip=CLIENT_A_IP):
    """Specs for requests of one client, each on its own thread."""
    return [(lbl, ip, f"thread-{lbl}", {}) for lbl in labels]


# --------------------------------------------------------------------------------------
# two parked requests of one higher-ranked client, against the holder's follow-up
# --------------------------------------------------------------------------------------

async def test_two_parked_requests_from_one_higher_ranked_client_are_both_served_ahead_of_the_holders_follow_up(
    tmp_path,
):
    """A lower-ranked holder is mid-turn. One higher-ranked client sends two requests (two
    threads), both parked in the holder's inbox through the real HIT path, and the holder's
    same-thread follow-up is ready behind them. When the holder's turn ends, the turns must
    start in the order holder, C1, C2, and only then the holder's follow-up: both parked
    requests ahead of it, and the follow-up still served afterwards, its future resolving
    (re-queued by rank, never dropped).

    The order that fired is part of the assertion message. If the second request were left
    with no claim once the first one started, the holder's follow-up could be served ahead of
    it."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        await park(world, h1, same_client("C1", "C2"))

        # The holder's turn ends and its same-thread follow-up arrives right behind it.
        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)

        assert await world.started("C1", timeout=PROMPT_S), f"order {world.order()}"
        world.release("C1")
        assert await world.started("C2", timeout=AFTER_GRACE_S), (
            f"the second parked request never started; turn start order {world.order()}"
        )
        assert world.order() == ["H1", "C1", "C2"], (
            f"turn start order was {world.order()}: the holder's follow-up (H2) was served "
            "ahead of a higher-ranked request parked in the inbox"
        )
        world.release("C2")

        assert await world.started("H2", timeout=AFTER_GRACE_S), (
            f"the holder's follow-up was never served: {world.order()}"
        )
        assert world.order() == ["H1", "C1", "C2", "H2"], world.order()
        world.release("H2")
        done, _ = await asyncio.wait({h2.completion_future}, timeout=PROMPT_S)
        assert done and h2.completion_future.result()["ok"], "the follow-up's future did not resolve"
        assert world.unloads == [] and world.tail_requeues == [], (
            f"the swap unloaded the resident or drained its inbox: {world.unloads} {world.tail_requeues}"
        )
        assert world.spawns == [MODEL], f"an engine was (re)spawned: {world.spawns}"


# --------------------------------------------------------------------------------------
# claims of every parked request, observed at the moment each turn starts
# --------------------------------------------------------------------------------------

async def test_the_second_parked_request_still_has_a_claim_when_the_first_starts(tmp_path):
    """Two requests of one client are parked. At the moment the first one's turn starts
    (observed from inside its fake completion call) the claims snapshot and the claim registry
    still list a claim for the SECOND request's slot (compared by slot identity), and the first
    request's own claim is gone. When the second one's turn starts its claim is gone too, and
    at the end no claim of this client is registered (nothing leaks).

    Fails when the second request is left with no claim once the first one's turn starts."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        await park(world, h1, same_client("C1", "C2"))
        c1, c2 = world.slots["C1"], world.slots["C2"]
        assert world.has_claim("C1") or world.has_claim("C2"), (
            "setup: a parked request of the client must hold a claim while the holder runs"
        )

        world.release("H1")
        assert await world.started("C1", timeout=PROMPT_S), f"order {world.order()}"
        assert c2.slot_id in world.snapshot_at_start[c1.slot_id], (
            "when the first parked request's turn started, the claims snapshot no longer listed "
            f"the second parked request (listed: {[world.labels.get(s, s) for s in world.snapshot_at_start[c1.slot_id]]})"
        )
        assert c2.slot_id in world.registry_at_start[c1.slot_id], (
            "when the first parked request's turn started, the claim registry no longer held a "
            "claim for the second parked request"
        )
        assert c1.slot_id not in world.snapshot_at_start[c1.slot_id], (
            "the first request's claim was still listed while its own turn was running"
        )

        world.release("C1")
        assert await world.started("C2", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert c2.slot_id not in world.snapshot_at_start[c2.slot_id], (
            "the second request's claim was still listed after its own turn started"
        )
        assert c2.slot_id not in world.registry_at_start[c2.slot_id]
        assert world.claims_of_client(CLIENT_A_IP) == [], "a claim of the client leaked"

        world.release("C2")
        await world.turn_done("C2")
        assert world.claims_of_client(CLIENT_A_IP) == [], "a claim of the client leaked at the end"
        assert world.snapshot_ids() == [], "the claims snapshot lists a claim at the end"


async def test_three_parked_requests_each_keep_a_claim_until_their_own_turn(tmp_path):
    """Three requests of one client are parked (three is enough; the count stays bounded).
    After the first one's turn starts, claims for the second and the third are listed; after
    the second one's turn starts, the third's is listed and the second's is not; once the third
    one's turn has started none is listed, and none is registered at the end.

    Each moment is observed from inside the fake completion call of the request whose turn is
    starting, by slot identity."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        await park(world, h1, same_client("C1", "C2", "C3"))
        c1, c2, c3 = (world.slots[lbl] for lbl in ("C1", "C2", "C3"))

        world.release("H1")
        assert await world.started("C1", timeout=PROMPT_S), f"order {world.order()}"
        at_c1 = world.snapshot_at_start[c1.slot_id]
        assert c2.slot_id in at_c1 and c3.slot_id in at_c1, (
            f"after the first turn started the claims for the second and third parked requests "
            f"must be listed, the snapshot listed {[world.labels.get(s, s) for s in at_c1]}"
        )
        assert c1.slot_id not in at_c1

        world.release("C1")
        assert await world.started("C2", timeout=AFTER_GRACE_S), f"order {world.order()}"
        at_c2 = world.snapshot_at_start[c2.slot_id]
        assert c3.slot_id in at_c2, (
            f"after the second turn started the claim for the third parked request must be "
            f"listed, the snapshot listed {[world.labels.get(s, s) for s in at_c2]}"
        )
        assert c2.slot_id not in at_c2 and c1.slot_id not in at_c2

        world.release("C2")
        assert await world.started("C3", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert world.snapshot_at_start[c3.slot_id] == [], (
            "a claim was still listed when the last parked request's turn started: "
            f"{[world.labels.get(s, s) for s in world.snapshot_at_start[c3.slot_id]]}"
        )
        assert world.order() == ["H1", "C1", "C2", "C3"], world.order()
        world.release("C3")
        await world.turn_done("C3")
        assert world.claims_of_client(CLIENT_A_IP) == [], "a claim of the client leaked at the end"


@pytest.mark.parametrize("gone_first", [True, False], ids=["first_parked_disconnects", "second_parked_disconnects"])
async def test_a_sibling_that_disconnects_while_parked_does_not_strip_the_others(tmp_path, gone_first):
    """Two requests of one client are parked and the client of ONE of them disconnects while
    it is parked (the first one parked, or the second). The other one is live: while it waits
    it must still hold a claim (listed in the claims snapshot, which is also where the dead
    claim is cleaned), and at the holder's turn boundary it must be served ahead of the
    holder's follow-up. The disconnected one must hold no claim.

    The case that can fail is the first-parked one disconnecting: when requests of a client
    share one claim, that claim belongs to the first request, and its going away must not
    leave the live sibling with none. The second-parked variant is the control (the live first
    request owns the claim).

    Only what a disconnected parked request implies is asserted: it is dropped (it holds no
    claim). Whether it is dropped at the disconnect or when the resident would admit it is not
    asserted, and neither is whether the resident spends a turn on it; its fake turn is
    released in advance so it can never block the live request."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        gone = asyncio.Event()
        dead_kw = {"disconnect_event": gone}
        dead_spec = ("DEAD", CLIENT_A_IP, "thread-dead", dead_kw)
        live_spec = ("LIVE", CLIENT_A_IP, "thread-live", {})
        specs = [dead_spec, live_spec] if gone_first else [live_spec, dead_spec]
        await park(world, h1, specs)
        live, dead = world.slots["LIVE"], world.slots["DEAD"]
        assert world.claims_of_client(CLIENT_A_IP), "setup: the client's parked requests hold a claim"
        world.release("DEAD")

        gone.set()
        listed = world.snapshot_ids()
        assert dead.slot_id not in listed, "the disconnected parked request still holds a claim"
        assert live.slot_id in listed, (
            "the live parked request holds no claim once its disconnected sibling is gone "
            f"(the snapshot listed {[world.labels.get(s, s) for s in listed]})"
        )

        world.release("H1")
        await world.turn_done("H1")
        await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("LIVE", timeout=AFTER_GRACE_S), (
            f"the live parked request was never served: {world.order()}"
        )
        assert "H2" not in world.order(), (
            f"turn start order was {world.order()}: the holder's follow-up was served ahead of "
            "the live parked request"
        )
        assert world.unloads == [] and world.tail_requeues == []


async def test_when_the_first_parked_request_disconnects_the_live_sibling_is_served_before_the_holders_follow_up(
    tmp_path,
):
    """The starvation itself, for requests of one client. Two requests of the client are parked
    and the FIRST one's client disconnects. The holder's turn ends and its same-thread
    follow-up arrives right behind it. The live sibling outranks the holder, so the next turn
    on the resident must be the live sibling's, not the holder's follow-up: turn start order
    holder, live sibling.

    The fake turn of the disconnected request is released in advance, so a resident that did
    spend a turn on it could not block the check; the check reads only the first two turn
    starts after the holder's, within a bounded wait (the holder's grace window is one second
    and the follow-up arrives well inside it)."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        gone = asyncio.Event()
        await park(world, h1, [
            ("DEAD", CLIENT_A_IP, "thread-dead", {"disconnect_event": gone}),
            ("LIVE", CLIENT_A_IP, "thread-live", {}),
        ])
        world.release("DEAD")
        gone.set()

        world.release("H1")
        await world.turn_done("H1")
        await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        with contextlib.suppress(AssertionError):
            await wait_until(lambda: len(world.starts) >= 2, timeout=PROMPT_S)
        assert world.order()[:2] == ["H1", "LIVE"], (
            f"turn start order was {world.order()}: the holder's warm follow-up was served ahead "
            "of the live higher-ranked request parked in its inbox"
        )


# --------------------------------------------------------------------------------------
# controls: behaviour that already holds and must stay as it is
# --------------------------------------------------------------------------------------

async def test_control_one_parked_request_from_each_of_two_clients(tmp_path):
    """Control. One request each from two different clients (different rule addresses, the
    second ranked below the first) is parked. Each has a claim of its own (different keys, so
    no collapse): while both are parked both are listed; when the first one's turn starts the
    second one's claim is still listed and the first one's is not; at the end none is
    registered. The turns start in rank order, holder's follow-up last. This is the case that
    already works with one parked request per client, so it must not regress."""
    async with build_world(tmp_path) as world:
        h1 = await start_holder(world)
        await park(world, h1, [
            ("CA", CLIENT_A_IP, "thread-CA", {}),
            ("CB", CLIENT_B_IP, "thread-CB", {}),
        ])
        ca, cb = world.slots["CA"], world.slots["CB"]
        assert world.has_claim("CA") and world.has_claim("CB"), (
            "each parked request of a different client must hold its own claim"
        )

        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)

        assert await world.started("CA", timeout=PROMPT_S), f"order {world.order()}"
        at_ca = world.snapshot_at_start[ca.slot_id]
        assert cb.slot_id in at_ca and ca.slot_id not in at_ca, (
            f"snapshot when the first turn started: {[world.labels.get(s, s) for s in at_ca]}"
        )

        world.release("CA")
        assert await world.started("CB", timeout=AFTER_GRACE_S), f"order {world.order()}"
        # Only the holder's own follow-up (a lower-ranked waiter) may still hold a claim.
        assert set(world.snapshot_at_start[cb.slot_id]) <= {h2.slot_id}, (
            f"unexpected claims when the second turn started: {world.snapshot_at_start[cb.slot_id]}"
        )

        world.release("CB")
        assert await world.started("H2", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert world.order() == ["H1", "CA", "CB", "H2"], world.order()
        world.release("H2")
        await world.turn_done("H2")
        assert world.claims_of_client(CLIENT_A_IP) == []
        assert world.claims_of_client(CLIENT_B_IP) == []


async def test_control_a_second_request_in_staging_not_parked_is_unchanged(tmp_path):
    """Control. The dispatcher is stopped, so two requests of one client stay in the staging
    queue (neither is parked). They collapse onto one claim as before: the snapshot lists
    exactly one claim for the client, the first request's, and none for the second. The dedup
    of waiting-in-staging requests must stay."""
    async with build_world(tmp_path) as world:
        await start_holder(world)
        await world.pause_dispatcher()
        c1 = await world.send("C1", CLIENT_A_IP, "thread-C1")
        c2 = await world.send("C2", CLIENT_A_IP, "thread-C2")
        assert c1 in world.mgr.queue._staging and c2 in world.mgr.queue._staging, (
            "setup: both requests must still be queued, not routed"
        )
        r = resident_for(world.mgr, MODEL)
        assert r.inbox.qsize() == 0, "setup: nothing may be parked"

        rows = [row for row in world.mgr.fastlane_claims_snapshot()
                if row["slot_id"] in (c1.slot_id, c2.slot_id)]
        assert [row["slot_id"] for row in rows] == [c1.slot_id], (
            "two staged requests of one client must collapse onto the first one's claim; "
            f"the snapshot listed {[world.labels.get(row['slot_id']) for row in rows]}"
        )
        assert world.claims_of_client(CLIENT_A_IP) == [c1.slot_id]
