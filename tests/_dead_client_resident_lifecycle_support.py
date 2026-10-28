"""Construction shared by the lifecycle tests of waiting requests whose client has gone away.

Two worlds, both built around the real manager with only the outside faked:

* ``Driver``: one resident whose real driver task runs over a fake engine and a fake per-turn
  serve. The driver can be brought up parked (idle, empty inbox) or busy with a held turn, which
  is the state the removal of a waiting request depends on.
* ``Box``: a manager with a fake engine process whose turns end only when told to, spies that
  call through to the real functions and return their results unchanged, and, optionally, the
  dispatcher running.

Nothing here decides an outcome. A spy records and calls through; a helper builds a request or
reads a registry. Claims are read from the registry directly, never through the snapshot method,
which prunes dead claims as a side effect and would change what it observes.
"""
from __future__ import annotations

import asyncio
import contextlib

from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    seed_manifest,
    wait_until,
)

from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db

MODEL = "lc-model"
IP_A = "10.0.0.4"      # rule index 0
IP_B = "10.0.0.5"      # rule index 1
SHORT_S = 5.0


def client_meta(ip):
    return {"ip": ip, "is_main": True}


def _match(ip):
    index = 0 if ip == IP_A else 1
    return FastLaneMatch(rule_index=index, raw_address=ip, label="", effective_tag="main", rank=1)


def _runtime(tmp_path, **kw):
    boot, runtime = boot_ranked_runtime(tmp_path, rules=ranked_rules((IP_A, 1), (IP_B, 1)), **kw)
    # The audit write that failing a request fires in the background needs an existing database.
    open_state_db(boot.storage.state_db_path).close()
    return boot, runtime


def waiting_slot(slot_id, created_at, *, ip=IP_A, event=None):
    """A request that has not started: STAGED, with a pending future and a rank, as the arrival path leaves one."""
    slot = Slot(
        slot_id=slot_id,
        model_tag=MODEL,
        state=SlotState.STAGED,
        created_at=created_at,
        fastlane=_match(ip),
        client_meta={},
        completion_future=asyncio.get_running_loop().create_future(),
    )
    slot.disconnect_event = event
    return slot


def claims_of(mgr, slot):
    """The registered claims that point at ``slot`` or list it among their waiters."""
    return [
        c for c in mgr._fastlane_claims.values()
        if c["slot"] is slot or slot in (c.get("waiters") or {}).values()
    ]


def install_release_spy(mgr):
    """Record (slot, reason) of every claim release; the real release still runs."""
    releases = []
    real = mgr._release_fastlane_claim_locked

    def spy(slot, reason):
        releases.append((slot, reason))
        return real(slot, reason)

    mgr._release_fastlane_claim_locked = spy
    return releases


def install_audit_spy(mgr):
    """Count the audit records written for a failed request; the real one still runs."""
    audited = []
    real = mgr._audit_unloaded_slot

    async def spy(slot):
        audited.append(slot)
        return await real(slot)

    mgr._audit_unloaded_slot = spy
    return audited


async def settle_background(mgr, timeout=SHORT_S):
    """Wait for the manager's background tasks (evictions, audit writes) to finish."""
    tasks = list(mgr._bg_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)


# --------------------------------------------------------------------------------------
# one resident, its real driver, a fake engine
# --------------------------------------------------------------------------------------

class _StoppedEngine:
    """An engine handle that reports itself as not running, so the driver skips saving the previous
    turn's context."""

    def is_alive(self):
        return False


class Driver:
    """A manager, one resident and the real driver task over it."""

    def __init__(self, mgr, resident):
        self.mgr = mgr
        self.r = resident
        self.served = []          # slots whose turn the driver began, in order
        self.gates = {}           # slot id -> event the fake turn waits on
        self.task = None
        self.releases = install_release_spy(mgr)
        self.audited = install_audit_spy(mgr)

    def hold(self, slot_id):
        return self.gates.setdefault(slot_id, asyncio.Event())

    def served_ids(self):
        return [s.slot_id for s in self.served]

    def in_inbox(self, slot):
        return self.mgr._inbox_holds_slot(self.r, slot)

    def park_claim(self, slot):
        """The request waits in the resident's inbox holding a claim, as the dispatcher leaves it."""
        self.mgr._register_fastlane_claim_locked(slot, "parked_in_inbox")
        self.mgr._park_fastlane_claim_locked(slot, self.r)

    async def produce(self, slot):
        """Exactly what the dispatcher's route does: under the lock, wake an idle engine, then put."""
        async with self.mgr._registry_lock:
            if self.r.state is ResidentState.IDLE_EVICTABLE:
                self.r.state = ResidentState.ACTIVE
            self.r.inbox.put_nowait(slot)

    async def stop(self):
        if self.task is not None:
            # A cancel that lands just as the driver's idle wait completes can be swallowed by the
            # wait, so ask again (a bounded number of times) until the driver has ended.
            for _ in range(10):
                if self.task.done():
                    break
                self.task.cancel()
                await asyncio.wait({self.task}, timeout=0.2)
            await asyncio.gather(self.task, return_exceptions=True)
        await asyncio.gather(*list(self.mgr._bg_tasks), return_exceptions=True)


async def bring_up_driver(tmp_path, monkeypatch, *, keep_alive_s, hold_anchor=False):
    """Run the real resident driver with only the engine start and the per-turn serve faked.

    With ``hold_anchor`` the first request's turn is held open: the driver is busy with a turn
    (state ACTIVE, an active request). Otherwise the first turn ends at once and the driver is left
    parked, idle, on an empty inbox. ``keep_alive_s`` is the resident's idle window.
    """
    boot, runtime = _runtime(tmp_path)
    mgr = TurbohaulManager(boot, runtime)
    r = Resident(resident_key=MODEL, model_tag=MODEL, inbox=asyncio.Queue())
    mgr._residents[MODEL] = r
    d = Driver(mgr, r)

    async def fake_spawn(r_, slot):
        return _StoppedEngine()

    async def fake_serve(r_, slot, handle):
        d.served.append(slot)
        gate = d.gates.get(slot.slot_id)
        if gate is not None:
            await gate.wait()

    monkeypatch.setattr(mgr, "_spawn_for_resident", fake_spawn)
    monkeypatch.setattr(mgr, "_serve_on_resident", fake_serve)
    r.latest_keep_alive_s = keep_alive_s
    r.sleep_idle_seconds = keep_alive_s
    anchor = waiting_slot("anchor", 0.0)
    anchor.state = SlotState.ACTIVE
    if hold_anchor:
        d.hold("anchor")
    r.inbox.put_nowait(anchor)
    d.task = asyncio.create_task(mgr._drive_resident(r))
    try:
        await wait_until(lambda: d.served_ids() == ["anchor"], timeout=SHORT_S)
        if hold_anchor:
            await wait_until(lambda: r.active_slot is anchor, timeout=SHORT_S)
        else:
            await wait_until(
                lambda: r.state is ResidentState.IDLE_EVICTABLE and r.inbox.empty(), timeout=SHORT_S,
            )
    except BaseException:
        await d.stop()
        raise
    return d


# --------------------------------------------------------------------------------------
# a manager over a fake engine process
# --------------------------------------------------------------------------------------

class Box:
    """One manager, a fake engine whose turns end only when told to, and recording spies."""

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns          # model tag of every engine process started
        self.slots = {}
        self.labels = {}
        self.events = []              # ("start" | "end", slot_id)
        self.reserve_calls = []       # the slot of every call to the engine reserve
        self.releases = []            # (slot, reason) of every claim release
        self.audited = []             # the slot of every audit record for a failed request
        self._gates = {}
        self._open = False

    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.events.append(("start", slot.slot_id))
        if not self._open:
            await self.gate(slot.slot_id).wait()
        self.events.append(("end", slot.slot_id))
        return {"ok": True, "model": handle.model_tag}

    def release(self, label):
        self.gate(self.slots[label].slot_id).set()

    def open_everything(self):
        self._open = True
        for ev in self._gates.values():
            ev.set()

    async def send(self, label, ip=IP_A, thread=None, event=None):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread or f"thread-{label}", client_meta=client_meta(ip),
            wait_for_completion=True, disconnect_event=event,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        """Labels of the requests whose engine turn has begun, in order."""
        return [self.labels.get(s, s) for kind, s in self.events if kind == "start"]

    async def started(self, label, timeout=SHORT_S):
        try:
            await wait_until(lambda: label in self.order(), timeout=timeout)
        except AssertionError:
            return False
        return True

    async def settled(self, label, timeout):
        """'started' | 'answered' (future done, never started) | 'pending' within ``timeout``."""
        slot = self.slots[label]
        try:
            await wait_until(lambda: label in self.order() or slot.completion_future.done(), timeout=timeout)
        except AssertionError:
            return "pending"
        return "started" if label in self.order() else "answered"

    def reserved(self, label):
        return any(s is self.slots[label] for s in self.reserve_calls)

    def release_reasons(self, label):
        return [reason for s, reason in self.releases if s is self.slots[label]]

    def start_dispatcher(self):
        self.mgr._worker_task = asyncio.create_task(self.mgr.worker_loop())

    async def pop(self):
        """Take the head of the queue the way the dispatcher does."""
        return await self.mgr.queue.pop_next()


@contextlib.asynccontextmanager
async def build_box(tmp_path, *, dispatcher=True, staging_depth=None, grace_seconds=0):
    """A manager with a one-engine budget over one model, a fake engine, and spies on the engine
    reserve, the claim release and the audit write for a failed request."""
    boot, runtime = _runtime(tmp_path, max_parallel_sidecars=1, grace_seconds=grace_seconds)
    if staging_depth is not None:
        runtime.queue.staging_queue_depth = staging_depth
    seed_manifest(boot, MODEL, main_gpu=0)
    spawns = []
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})

    def counting_spawn(binary, gguf, port, model_tag, argv, **kw):
        spawns.append(model_tag)
        return spawn_fn(binary, gguf, port, model_tag, argv, **kw)

    holder = {}

    async def complete(slot, handle):
        return await holder["box"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=counting_spawn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    box = Box(mgr, spawns)
    holder["box"] = box

    async def fake_save(port, model_tag, slot=None, **kwargs):
        return True

    mgr._save_slot_kv = fake_save
    mgr._other_active_residents = lambda handle: 1
    box.releases = install_release_spy(mgr)
    box.audited = install_audit_spy(mgr)

    real_reserve = mgr._reserve_and_start_locked

    async def spy_reserve(slot, *a, **kw):
        box.reserve_calls.append(slot)
        return await real_reserve(slot, *a, **kw)

    mgr._reserve_and_start_locked = spy_reserve
    if dispatcher:
        box.start_dispatcher()
    try:
        with high_vram():
            yield box
    finally:
        box.open_everything()
        await mgr.shutdown()
