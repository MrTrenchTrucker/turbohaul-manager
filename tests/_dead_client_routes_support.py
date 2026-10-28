"""Support for the dead-client route tests.

Builds, inside ONE process and one event loop:

* a real ``TurbohaulManager`` with exactly one engine slot (``max_parallel_sidecars=1``),
  whose fake engine finishes a turn only when the test says so;
* the real chat routes (``/v1/chat/completions`` and ``/api/chat``) mounted on a small
  FastAPI app, served by an in-process uvicorn server bound to a loopback ephemeral port;
* a raw-socket HTTP client, so a test can hang up (close its socket) exactly like a client
  that gives up on a request that is still waiting.

Nothing here sets a slot's ``disconnect_event`` or ``is_evicted`` flag, evicts a slot, or
releases a claim: every observation is a read (the queue's staging snapshot, the accept
buffer, the resident inbox, the claim registry, the slot object, spies that only record).
``fastlane_claims_snapshot()`` is deliberately NOT used: it prunes dead claims as a side
effect and would change what it is asked to observe.

The server and every socket are closed in ``finally`` blocks, every wait is bounded.
"""

import asyncio
import contextlib
import json
import time

import uvicorn
from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)
from fastapi import FastAPI

import turbohaul.manager as manager_module
from turbohaul.api import chat_completion
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState

MODEL = "dc-model"
LOOPBACK_IP = "127.0.0.1"
HOLDER_IP = "10.0.0.5"
HOLDER_THREAD = "thread-holder"

# How long a test waits for something that is supposed to happen, and how long it
# waits to be sure something does NOT happen.
OBSERVE_WINDOW_S = 5.0
NO_TURN_WINDOW_S = 2.5
SERVED_WINDOW_S = 6.0

_STARTED_STATES = (SlotState.LOADING, SlotState.ACTIVE, SlotState.GRACE, SlotState.POPPED)


# --------------------------------------------------------------------------- server
class LoopbackServer:
    """The app served by uvicorn in this process, on 127.0.0.1 with an ephemeral port."""

    def __init__(self, app):
        self.config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", lifespan="off",
            access_log=False, timeout_graceful_shutdown=2,
        )
        self.server = uvicorn.Server(self.config)
        self.port = None

    async def start(self):
        self.config.load()
        self.server.lifespan = self.config.lifespan_class(self.config)
        await asyncio.wait_for(self.server.startup(), timeout=10)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self.port

    async def stop(self):
        self.server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.server.shutdown(), timeout=10)


class RawClient:
    """One raw-socket HTTP/1.1 client connection that has sent a JSON POST."""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.slot = None
        self.closed = False

    @classmethod
    async def post(cls, port, path, payload):
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=5)
        body = json.dumps(payload).encode()
        head = (
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nAccept: */*\r\n\r\n"
        ).encode()
        writer.write(head + body)
        await asyncio.wait_for(writer.drain(), timeout=5)
        return cls(reader, writer)

    async def hang_up(self):
        """Close the socket without reading anything: the client has gone away."""
        if self.closed:
            return
        self.closed = True
        self.writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.writer.wait_closed(), timeout=2)

    async def status_line(self, timeout):
        """First line of the response, or None if nothing arrives within ``timeout``."""
        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
        except (TimeoutError, ConnectionError):
            return None
        return line.decode("latin-1").strip() or None


# --------------------------------------------------------------------------- world
class RouteWorld:
    """A manager with one busy engine slot plus the real routes behind a loopback server."""

    def __init__(self, mgr, spawns, port):
        self.mgr = mgr
        self.spawns = spawns
        self.port = port
        self.turn_starts = []      # slot ids whose fake engine turn began, in order
        self.http_slots = []       # slots created for loopback (HTTP) clients, in order
        self.state_history = {}    # slot id -> list of states entered (HTTP slots only)
        self.holder = None
        self._gates = {}
        self._open = False
        self._clients = []

    # fake engine ---------------------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.turn_starts.append(slot.slot_id)
        # Only the busy request waits for the test; a request from a loopback client
        # finishes at once, so a served client gets its answer.
        is_http_client = (slot.client_meta or {}).get("ip") == LOOPBACK_IP
        if not self._open and not is_http_client:
            await self.gate(slot.slot_id).wait()
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": handle.model_tag,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def open_everything(self):
        self._open = True
        for ev in self._gates.values():
            ev.set()

    # the one busy slot -----------------------------------------------------------
    async def start_holder(self):
        """A live request occupies the only engine; its turn ends only on release_holder()."""
        self.holder = await self.mgr.submit(
            MODEL, prompt="holder", thread_id=HOLDER_THREAD,
            client_meta={"ip": HOLDER_IP, "is_main": True}, wait_for_completion=True,
        )
        await wait_until(lambda: self.holder.slot_id in self.turn_starts, timeout=5.0)
        return self.holder

    def release_holder(self):
        self.gate(self.holder.slot_id).set()

    # clients ---------------------------------------------------------------------
    async def open_waiting_client(self, path, payload):
        """Send a request through the real route and wait until it sits in the manager,
        parked behind the busy engine with its claim registered."""
        before = len(self.http_slots)
        client = await RawClient.post(self.port, path, payload)
        self._clients.append(client)
        await wait_until(lambda: len(self.http_slots) > before, timeout=5.0)
        client.slot = self.http_slots[before]
        resident = resident_for(self.mgr, MODEL)
        await wait_until(lambda: bool(self.places(client.slot)), timeout=5.0)
        # Setup guards: the request is genuinely waiting (not started) and holds a claim.
        assert client.slot.state not in _STARTED_STATES, "setup: the request already started"
        assert self.mgr._inbox_holds_slot(resident, client.slot), (
            "setup: the request is expected to be parked in the busy engine's inbox, "
            f"it waits in {self.places(client.slot)}"
        )
        assert self.claim_rows(client.slot), "setup: the waiting request holds no claim"
        assert not client.slot.disconnect_event.is_set(), "setup: flag already set"
        assert client.slot.is_evicted is False, "setup: slot already evicted"
        return client

    async def close_clients(self):
        for c in self._clients:
            await c.hang_up()

    # read-only observations ------------------------------------------------------
    def places(self, slot):
        """Where the slot waits right now: staging, accept buffer, a resident's inbox."""
        out = []
        if any(s is slot for s in self.mgr.queue.staging_snapshot()):
            out.append("staging")
        if any(s is slot for s in tuple(self.mgr.queue._accept_buf)):
            out.append("accept_buffer")
        if any(self.mgr._inbox_holds_slot(r, slot) for r in self.mgr._model_residents()):
            out.append("inbox")
        return out

    def claim_rows(self, slot):
        """Registered claims that point at the slot or list it among their waiters."""
        rows = []
        for claim in self.mgr._fastlane_claims.values():
            if claim["slot"] is slot or slot in (claim.get("waiters") or {}).values():
                rows.append({"points_at_it": claim["slot"] is slot, "parked_on": claim.get("parked_on")})
        return rows

    def turn_started(self, slot):
        """Did an engine turn (or an engine handle) get assigned to the slot?"""
        if slot.slot_id in self.turn_starts:
            return True
        if getattr(slot, "stream_handle", None) is not None:
            return True
        return any(s in _STARTED_STATES for s in self.state_history.get(slot.slot_id, ()))


@contextlib.asynccontextmanager
async def route_world(tmp_path):
    """Manager (one engine slot, dispatcher running) + routes + loopback server."""
    rules = ranked_rules((LOOPBACK_IP, 1), (HOLDER_IP, 1))
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=1, grace_seconds=1,
        max_grace_extensions=50, idle_hot_load_seconds=600,
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

    async def no_kv_save(port, model_tag, slot=None, **kwargs):
        return True

    # The fake engine has no saved context to write: skip the save step only.
    mgr._save_slot_kv = no_kv_save
    mgr._other_active_residents = lambda handle: 1

    app = FastAPI()
    app.include_router(chat_completion.router)
    app.state.manager = mgr
    server = LoopbackServer(app)

    real_submit = mgr.submit
    real_transition = manager_module.transition
    world_ref = {}

    async def recording_submit(*a, **kw):
        slot = await real_submit(*a, **kw)
        if (slot.client_meta or {}).get("ip") == LOOPBACK_IP:
            world_ref["w"].http_slots.append(slot)
        return slot

    def recording_transition(slot, new_state, *a, **k):
        res = real_transition(slot, new_state, *a, **k)
        w = world_ref.get("w")
        if w is not None and any(slot is s for s in w.http_slots):
            w.state_history.setdefault(slot.slot_id, []).append(new_state)
        return res

    port = None
    worker = None
    world = None
    try:
        with high_vram():
            port = await server.start()
            world = RouteWorld(mgr, spawns, port)
            holder["world"] = world
            world_ref["w"] = world
            mgr.submit = recording_submit
            manager_module.transition = recording_transition
            worker = asyncio.create_task(mgr.worker_loop())
            mgr._worker_task = worker
            yield world
    finally:
        manager_module.transition = real_transition
        mgr.submit = real_submit
        if world is not None:
            world.open_everything()
            await world.close_clients()
        await server.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(mgr.shutdown(), timeout=15)


# --------------------------------------------------------------------------- scenario
async def hang_up_and_observe(world, client, *, window_s=OBSERVE_WINDOW_S):
    """Hang the client up, then sample the slot every 50 ms for up to ``window_s``.

    Returns ``(first, at_end)``. ``first`` maps a property to the seconds after the hang-up at
    which it first held; a property that never held within the window is absent.
    ``at_end`` is what the slot waits in when the window closes (places, claim count).
    Properties:
      flag_set        the slot's disconnect_event is set
      marked_evicted  slot.is_evicted is True
      left_queue      the slot waits in no staging queue, accept buffer or inbox
      claim_released  no registered claim points at the slot or lists it as a waiter
    """
    slot = client.slot
    checks = {
        "flag_set": lambda: slot.disconnect_event.is_set(),
        "marked_evicted": lambda: slot.is_evicted is True,
        "left_queue": lambda: not world.places(slot),
        "claim_released": lambda: not world.claim_rows(slot),
    }
    t0 = time.monotonic()
    await client.hang_up()
    first = {}
    while True:
        now = time.monotonic() - t0
        for name, check in checks.items():
            if name not in first and check():
                first[name] = round(now, 2)
        if len(first) == len(checks) or now > window_s:
            return first, {"places": world.places(slot), "claims": len(world.claim_rows(slot))}
        await asyncio.sleep(0.05)


async def end_holder_turn_and_watch(world, slot, *, window_s=NO_TURN_WINDOW_S):
    """Let the busy engine's turn end, then watch ``window_s`` for a turn for ``slot``.

    Returns True as soon as an engine turn (or engine handle) exists for the slot, False if
    none appeared within the window.
    """
    world.release_holder()
    deadline = time.monotonic() + window_s
    while time.monotonic() < deadline:
        if world.turn_started(slot):
            return True
        await asyncio.sleep(0.05)
    return world.turn_started(slot)


def describe(first, expected=("flag_set", "marked_evicted", "left_queue", "claim_released"), window_s=OBSERVE_WINDOW_S):
    """One-line account of when each property held after the hang-up."""
    return ", ".join(
        f"{name}={'+' + str(first[name]) + 's' if name in first else 'never within ' + str(window_s) + 's'}"
        for name in expected
    )
