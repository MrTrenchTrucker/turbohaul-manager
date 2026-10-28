"""Construction shared by the tests of waiting requests whose client has gone away.

Two worlds, both built around the real manager with only the outside faked:

* ``RecheckBox``: a real manager, a fake free-memory reading per card, a fake engine
  process and a fake re-check clock. It answers "does the periodic co-residence re-check
  start a further engine for a waiting request whose client is gone".
* ``World``: a real manager with ranked Fast Lane clients, a fake engine whose turns end
  only when told to, and recording spies that call through to the real functions and
  return their results unchanged. It answers the questions about grace, the choice of the
  engine to unload, the Fast Lane claim and the admission instant.

The client's departure is modelled the way the route layer models it: the request's
``disconnect_event`` is set and nothing else is touched (the evicted flag stays False).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from unittest.mock import MagicMock

import yaml
from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

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

# --------------------------------------------------------------------------------------
# the re-check box
# --------------------------------------------------------------------------------------

TAG = "tag-a"
NEED_MIB = 4000          # the model footprint (the manifest's expected VRAM)
FLOOR_MIB = 512          # the memory the card picker keeps free on every card
CARD_TOTAL_MIB = 5000    # one engine per card: 5000 - 512 >= 4000, but (5000 - 4000) - 512 < 4000
RECHECK_LINE = "CO_RESIDENCE_RECHECK"
SETTLE_TIMEOUT_S = 3.0
BOX_PORT_BASE = 59800


def _box_runtime(tmp_path, cap):
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
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=BOX_PORT_BASE),
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


def _write_box_manifest(boot, tag):
    """One card per engine: the manager picks the card (auto placement) and the engine uses one."""
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 1000 * MIB,
        "context_size": 512,
        "expected_vram_bytes": NEED_MIB * MIB,
        "auto_place": True,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


class RecheckBox:
    """The faked outside world around one real manager, and what it was asked."""

    def __init__(self, mgr, cards):
        self.mgr = mgr
        self.totals = [CARD_TOTAL_MIB] * cards
        self.clock = [time.monotonic()]
        self.gate = asyncio.Event()       # an engine's request handler waits on it: set = engines finish
        self.entered = []                 # pid of the engine, in the order each began serving a request
        self.spawned = []                 # pid of every engine process started
        self.picks = []                   # (tag, (card, split)) answers of the card picker
        self.reserved = []                # the slot of every call to the engine reserve
        self.pid = 93_000

    def free_mib(self):
        """Free MiB per card: total minus the footprint of every LOADED engine."""
        loaded = {}
        for r in list(self.mgr._model_residents()):
            if r.handle is not None:
                loaded[r.main_gpu] = loaded.get(r.main_gpu, 0) + NEED_MIB
        return [t - loaded.get(g, 0) for g, t in enumerate(self.totals)]

    def now(self):
        return self.clock[0]

    def cards_of(self, tag=TAG):
        return sorted(r.main_gpu for r in self.mgr._instances_for(tag))

    def reserved_ids(self):
        return [s.slot_id for s in self.reserved]

    def recheck_lines(self, caplog):
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith(RECHECK_LINE)]


def _engine_process(box):
    def spawn(binary, gguf, port, model_tag, argv, **kw):
        box.pid += 1
        box.spawned.append(box.pid)
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


async def build_recheck_box(tmp_path, monkeypatch, caplog, *, cap=1, cards=2):
    """A manager with ``cards`` cards (one engine each) and a box budget of ``cap`` engines."""
    monkeypatch.setattr("turbohaul.manager.boot_orphan_reaper", lambda **k: {"reaped": 0})
    monkeypatch.setattr("turbohaul.manager.intra_lifetime_orphan_scan", lambda **k: {"reaped": 0})
    boot, runtime = _box_runtime(tmp_path, cap)
    _write_box_manifest(boot, TAG)
    box = RecheckBox(None, cards)
    mgr = TurbohaulManager(boot, runtime, **_engine_process(box))
    box.mgr = mgr
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", box.free_mib)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", box.free_mib)
    monkeypatch.setattr(mgr, "_co_residence_now", box.now, raising=False)

    real_pick = mgr._pick_card_for_new_instance

    def pick(tag, need):
        answer = real_pick(tag, need)
        box.picks.append((tag, answer))
        return answer

    mgr._pick_card_for_new_instance = pick

    real_reserve = mgr._reserve_and_start_locked

    async def reserve(slot, *a, **kw):
        box.reserved.append(slot)
        return await real_reserve(slot, *a, **kw)

    mgr._reserve_and_start_locked = reserve
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="turbohaul")
    return mgr, box


def parked_slot(box, thread_id="", *, event=None, age=0.0):
    """A request built the way the arrival path leaves one that waits: STAGED, with a pending future,
    that has been waiting ``age`` seconds on the re-check clock."""
    slot = Slot.new(TAG, prompt="p", thread_id=thread_id)
    slot.client_meta = {}
    slot.completion_future = asyncio.get_running_loop().create_future()
    slot.disconnect_event = event
    fsm_transition(slot, SlotState.STAGED)
    slot.created_at = box.clock[0] - age
    return slot


async def staged_slot(box, thread_id, *, event=None):
    """A request submitted through the real arrival path while nothing pops the queue: it stays in staging."""
    slot = await box.mgr.submit(
        TAG, prompt="p", thread_id=thread_id, client_meta={}, wait_for_completion=True,
        disconnect_event=event,
    )
    assert slot in box.mgr.queue.staging_snapshot(), "the request must wait in the staging queue"
    return slot


async def settle(box, timeout_s=SETTLE_TIMEOUT_S):
    """Wait until every engine the manager holds is loaded and has begun serving its first request."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        residents = list(box.mgr._model_residents())
        if all(r.handle is not None for r in residents) and len(box.entered) >= len(residents):
            return True
        await asyncio.sleep(0.005)
    return False


async def start_busy_engine(box):
    """The first engine of the tag, started the normal way and busy on a request."""
    first = parked_slot(box)
    await box.mgr._route_or_reserve(first)
    assert await settle(box), "the first engine never began serving"
    (r,) = box.mgr._instances_for(TAG)
    assert r.state is ResidentState.ACTIVE and r.inbox is not None
    box.reserved.clear()      # from here on, only the reserves the re-check makes are recorded
    return r


async def recheck_pass(box, advance_s=11.0):
    """Move the clock past the debounce and the demand age, run one re-check pass, let a started engine load."""
    from turbohaul.co_residence import run_recheck

    box.clock[0] += advance_s
    await run_recheck(box.mgr)
    await settle(box)


async def finish_box(box):
    box.gate.set()
    await box.mgr.shutdown()


# --------------------------------------------------------------------------------------
# the ranked-client world
# --------------------------------------------------------------------------------------

HOLDER_MODEL = "model-holder"
OTHER_MODEL = "model-other"
CLAIMANT_IP = "10.0.0.4"      # rule index 0: outranks the holder
HOLDER_IP = "10.0.0.5"        # rule index 1
HOLDER_THREAD = "thread-holder"
CLAIMANT_THREAD = "thread-claimant"
PROMPT_S = 1.5


def client_meta(ip):
    return {"ip": ip, "is_main": True}


class World:
    """One manager, a fake engine whose turns end only when told to, and recording spies."""

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns              # model tag of every engine process started
        self.slots = {}
        self.labels = {}
        self.events = []                  # ("start" | "end", slot_id)
        self.reserve_calls = []           # the slot of every call to the engine reserve
        self.releases = []                # (slot, reason) of every claim release
        self._gates = {}
        self._open = False

    # -- fake engine turn -------------------------------------------------------------
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

    # -- requests ---------------------------------------------------------------------
    async def send(self, label, ip, thread, model=HOLDER_MODEL, **kw):
        slot = await self.mgr.submit(
            model, prompt=label, thread_id=thread, client_meta=client_meta(ip),
            wait_for_completion=True, **kw,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def label_of(self, slot_id):
        return self.labels.get(slot_id, slot_id)

    def order(self):
        """Labels of the requests whose engine turn has begun, in order."""
        return [self.label_of(s) for kind, s in self.events if kind == "start"]

    async def started(self, label, timeout=PROMPT_S):
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

    async def parked(self, label, model=HOLDER_MODEL, timeout=PROMPT_S):
        slot = self.slots[label]
        r = resident_for(self.mgr, model)
        await wait_until(lambda: self.mgr._inbox_holds_slot(r, slot), timeout=timeout)
        return r

    async def turn_done(self, label, timeout=PROMPT_S):
        slot = self.slots[label]
        await wait_until(lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout)

    # -- observation ------------------------------------------------------------------
    def claims_of(self, slot):
        """The registered claims that point at ``slot`` or list it among their waiters."""
        return [
            c for c in self.mgr._fastlane_claims.values()
            if c["slot"] is slot or slot in (c.get("waiters") or {}).values()
        ]

    def claim_points_at(self, slot):
        return any(c["slot"] is slot for c in self.mgr._fastlane_claims.values())

    def reserved(self, label):
        """Was the engine reserve ever called for this request?"""
        return any(s is self.slots[label] for s in self.reserve_calls)

    def release_reasons(self, label):
        return [reason for s, reason in self.releases if s is self.slots[label]]


async def pause_dispatcher(mgr):
    """Stop the dispatcher task: nothing leaves staging, the way a dispatcher stuck on another await would behave."""
    task = mgr._worker_task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=30, max_parallel_sidecars=1):
    """A manager over two ranked clients (the claimant outranks the holder), two models on two cards,
    and the dispatcher running."""
    rules = ranked_rules((CLAIMANT_IP, 1), (HOLDER_IP, 1))
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=max_parallel_sidecars, grace_seconds=grace_seconds,
        max_grace_extensions=50, idle_hot_load_seconds=600,
    )
    seed_manifest(boot, HOLDER_MODEL, main_gpu=0)
    seed_manifest(boot, OTHER_MODEL, main_gpu=1)
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

    async def fake_save(port, model_tag, slot=None, **kwargs):
        return True

    mgr._save_slot_kv = fake_save
    mgr._other_active_residents = lambda handle: 1

    real_release = mgr._release_fastlane_claim_locked

    def spy_release(slot, reason):
        world.releases.append((slot, reason))
        return real_release(slot, reason)

    mgr._release_fastlane_claim_locked = spy_release

    real_reserve = mgr._reserve_and_start_locked

    async def spy_reserve(slot, *a, **kw):
        world.reserve_calls.append(slot)
        return await real_reserve(slot, *a, **kw)

    mgr._reserve_and_start_locked = spy_reserve

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


async def hold_engine(world, label="H1", ip=HOLDER_IP, thread=HOLDER_THREAD):
    """The holder's turn on the only engine, mid-turn."""
    slot = await world.send(label, ip, thread)
    assert await world.started(label, timeout=5.0), "setup: the holder's turn never started"
    r = resident_for(world.mgr, HOLDER_MODEL)
    assert r.state is ResidentState.ACTIVE and r.active_slot is slot, "setup: the holder must be mid-turn"
    return slot, r
