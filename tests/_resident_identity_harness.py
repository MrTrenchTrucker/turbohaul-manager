"""Worlds and drivers for the resident identity equivalence tests.

Everything here drives the REAL ``TurbohaulManager`` through methods it already has. The
test file holds the expected values; this file only builds the situations and reads back what the
manager did. Not a test module itself.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "tests"), str(_TREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _fastlane_fixture import wait_until  # noqa: E402
from _multiinstance_support import _manifest, _new_manager, _probe  # noqa: E402
from test_multislot_concurrency import _fake_handle  # noqa: E402

from turbohaul.live_monitor import compute_generation_id  # noqa: E402
from turbohaul.manager import _SINGLETON_RESIDENT_KEY, Resident, ResidentState  # noqa: E402
from turbohaul.slot import Slot  # noqa: E402

MODEL = "duo"
SECOND = f"{MODEL}#1"
SETUP_S = 5.0

# A request is (session, thread, ip, extra client_meta, fast lane object or None).
FASTLANE_LABELLED = SimpleNamespace(label="Gateway", raw_address="edge-gw")
REQUESTS = {
    "a": ("s-a", "t-a", "10.0.0.5", {"is_main": True}, None),
    "a2": ("s-a2", "t-a", "10.0.0.5", {"is_main": True}, None),
    "b": (
        "s-b", "t-b", "10.0.0.6",
        {"is_main": False, "is_sub_agent": True, "req_id": "req-b"}, FASTLANE_LABELLED,
    ),
}


def snap(value):
    """A detached, comparable copy of a plain value (dicts, lists, scalars)."""
    return json.loads(json.dumps(value))


def make_slot(name, model_tag=MODEL):
    session, thread, ip, extra, fastlane = REQUESTS[name]
    meta = {"ip": ip, "session_id": session, **extra}
    slot = Slot.new(model_tag, thread_id=thread, client_meta=meta)
    slot.fastlane = fastlane
    return slot


def make_resident(key, *, model_tag=MODEL, idx=0, pid=None, port=None, spawn_seq=0,
                  state=ResidentState.IDLE_EVICTABLE):
    handle = None if pid is None else _fake_handle(model_tag, port, pid)
    return Resident(
        model_tag=model_tag, resident_key=key, instance_idx=idx, handle=handle, port=port,
        spawn_seq=spawn_seq, state=state, last_active_monotonic=1.0,
    )


def new_manager(tmp_path, budget=2):
    mgr, _boot, _calls = _new_manager(tmp_path, budget=budget)
    return mgr


# ---- worlds -----------------------------------------------------------------------------

class RecordingDict(dict):
    """A dict that records every pop call as (key, defaults...), then does the real pop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pops = []

    def pop(self, key, *default):
        self.pops.append([key, *default])
        return super().pop(key, *default)


class IdentityWorld:
    """Two residents of one model plus a manager, with the identity dict still empty."""

    def __init__(self, tmp_path):
        self.mgr = new_manager(tmp_path)
        self.a = make_resident(MODEL, idx=0, pid=41001, port=59500, spawn_seq=1)
        self.b = make_resident(SECOND, idx=1, pid=41002, port=59501, spawn_seq=2)
        self.mgr._residents[MODEL] = self.a
        self.mgr._residents[SECOND] = self.b

    def stash_both(self):
        self.mgr._stash_resident_identity(self.a, make_slot("a"))
        self.mgr._stash_resident_identity(self.b, make_slot("b"))

    def dict_now(self):
        return snap(self.mgr._last_request_identity_by_model)


class LookupWorld:
    """Registry for the handle lookups.

    duo, duo#1: two live residents of one model, each with its own engine handle and spawn_seq.
    singleton: the legacy single-engine entry (never matched by a handle lookup).
    duo#2: a DEAD resident that still owns a handle.
    duo#3: a resident with no handle yet (still loading).
    other, other#1: two residents of ANOTHER model that share one handle object.
    """

    def __init__(self, tmp_path):
        self.mgr = new_manager(tmp_path)
        live = ResidentState.IDLE_EVICTABLE
        self.residents = {
            MODEL: make_resident(MODEL, idx=0, pid=41001, port=59500, spawn_seq=1),
            SECOND: make_resident(SECOND, idx=1, pid=41002, port=59501, spawn_seq=2),
            _SINGLETON_RESIDENT_KEY: make_resident(
                _SINGLETON_RESIDENT_KEY, pid=41003, port=59502, spawn_seq=9, state=live),
            "duo#2": make_resident("duo#2", idx=2, pid=41004, port=59503, spawn_seq=5,
                                   state=ResidentState.DEAD),
            "duo#3": make_resident("duo#3", idx=3, spawn_seq=7,
                                   state=ResidentState.RESERVED_LOADING),
            "other": make_resident("other", model_tag="other", pid=41005, port=59504, spawn_seq=3),
            "other#1": make_resident("other#1", model_tag="other", idx=1, spawn_seq=4),
        }
        self.residents["other#1"].handle = self.residents["other"].handle
        for key, r in self.residents.items():
            self.mgr._residents[key] = r
        self.stray = _fake_handle(MODEL, 59999, 88100)
        self.handles = {
            "own_duo": self.residents[MODEL].handle,
            "own_duo_1": self.residents[SECOND].handle,
            "none": None,
            "stray": self.stray,
            "singleton": self.residents[_SINGLETON_RESIDENT_KEY].handle,
            "dead": self.residents["duo#2"].handle,
            "shared": self.residents["other"].handle,
        }

    def key_of(self, resident):
        """The registry key of ``resident`` (matched by identity), None for None."""
        if resident is None:
            return None
        for key, r in self.residents.items():
            if r is resident:
                return key
        return "<not in registry>"


# ---- scenario drivers: stash, skip, swallow ---------------------------------------------

def run_stash_sequence(world, fresh_dict=False):
    """Bind a to duo, b to duo#1, then a second request to duo. The dict after each bind.

    With ``fresh_dict`` a bind of ``a`` is made first, and then a new dict replaces the old one."""
    mgr = world.mgr
    original = mgr._last_request_identity_by_model
    if fresh_dict:
        # One bind lands on the original dict first, then a new dict replaces it: the later binds
        # must follow the attribute, not whichever dict the first call saw.
        mgr._stash_resident_identity(world.a, make_slot("a"))
        mgr._last_request_identity_by_model = {}
    steps = []
    for resident, name in ((world.a, "a"), (world.b, "b"), (world.a, "a2")):
        mgr._stash_resident_identity(resident, make_slot(name))
        steps.append(snap(mgr._last_request_identity_by_model))
    return steps, snap(original)


def run_skipped_key(mgr, key, preseed):
    mgr._last_request_identity_by_model.update(preseed)
    resident = make_resident(key)
    returned = mgr._stash_resident_identity(resident, make_slot("a"))
    return returned, snap(mgr._last_request_identity_by_model)


class _RaisesOnRead:
    """A slot whose ``which`` field raises when it is read; every other field is real."""

    def __init__(self, which):
        real = make_slot("b")
        for name in ("client_meta", "model_tag", "thread_id", "fastlane"):
            if name != which:
                setattr(self, name, getattr(real, name))
        self._which = which

    def __getattr__(self, name):
        if name == self._which:
            raise RuntimeError(f"cannot read {name}")
        raise AttributeError(name)


class _BadLabel:
    @property
    def label(self):
        raise RuntimeError("cannot read label")


class _KeylessResident:
    model_tag = MODEL

    @property
    def resident_key(self):
        raise RuntimeError("cannot read resident_key")


SWALLOW_CASES = ["client_meta", "model_tag", "thread_id", "fastlane_label", "resident_key"]


def run_swallow(mgr, case):
    seen = {"a": {"session_id": "s-keep", "model_tag": MODEL}}
    mgr._last_request_identity_by_model.update(seen)
    resident = make_resident(MODEL)
    if case == "fastlane_label":
        slot = make_slot("a")
        slot.fastlane = _BadLabel()
    elif case == "resident_key":
        slot, resident = make_slot("a"), _KeylessResident()
    else:
        slot = _RaisesOnRead(case)
    error = None
    returned = None
    try:
        returned = mgr._stash_resident_identity(resident, slot)
    except Exception as exc:  # noqa: BLE001 - recorded so the test can assert it is None
        error = type(exc).__name__
    return error, returned, snap(mgr._last_request_identity_by_model)


# ---- scenario drivers: status read ---------------------------------------------------

def build_status_world(tmp_path):
    """duo and duo#1 (own entries), solo (empty resident_key, entry under its model_tag), lone (no entry)."""
    world = IdentityWorld(tmp_path)
    mgr = world.mgr
    world.stash_both()
    solo = make_resident("", model_tag="solo", pid=41010, port=59510, spawn_seq=4)
    lone = make_resident("lone", model_tag="lone", pid=41011, port=59511, spawn_seq=5)
    mgr._residents["solo"] = solo
    mgr._residents["lone"] = lone
    mgr._last_request_identity_by_model["solo"] = {
        "ip": "10.0.0.9", "model_tag": "solo", "session_id": "s-solo", "fastlane_rule": None,
    }
    world.solo, world.lone = solo, lone
    return world


def status_request_identity_by_pid(mgr):
    out = {}
    for entry in mgr.status_snapshot()["residents"]:
        out[entry["pid"]] = snap(entry["request_identity"])
    return out


# ---- scenario drivers: eviction -------------------------------------------------------

def run_unload(world, evict):
    """Evict one resident through the unload path with a recording dict already in place."""
    mgr = world.mgr
    world.stash_both()
    recorder = RecordingDict(mgr._last_request_identity_by_model)
    mgr._last_request_identity_by_model = recorder
    mgr._begin_unload_locked(world.a if evict == MODEL else world.b)
    return [list(p) for p in recorder.pops], snap(dict(recorder))


class _Gated:
    """Fake completion: a request whose session is in ``block`` holds its turn until released."""

    def __init__(self, mgr, block=()):
        self.gates = {s: asyncio.Event() for s in block}
        self.started, self.done, self.tasks = [], [], []
        mgr._complete_fn = self._complete

    async def _complete(self, slot, handle):
        session = (slot.client_meta or {}).get("session_id")
        self.started.append(session)
        gate = self.gates.get(session)
        if gate is not None:
            await gate.wait()
        self.done.append(session)
        return {"ok": True, "model": handle.model_tag}

    def release_all(self):
        for gate in self.gates.values():
            gate.set()


async def run_driver_finally(tmp_path):
    """Two real engines of one model serve one request each; then the idle driver of duo#1 is
    cancelled so it leaves through its own finally. Returns what the identity dict held after
    both admissions, and what the finally did to it."""
    mgr, boot, _ = _new_manager(tmp_path, budget=2)
    _manifest(boot, MODEL, auto_place=True, llama_server_flags={"split_mode": "none"})
    mgr.runtime.queue.grace_seconds = 2
    mgr.runtime.queue.idle_hot_load_seconds = 600
    gated = _Gated(mgr, block={"s-a", "s-b"})
    p_free, p_probe = _probe([20000, 20000])
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    out = {}
    with p_free, p_probe:
        try:
            for name in ("a", "b"):
                session, thread, ip, _extra, _fl = REQUESTS[name]
                gated.tasks.append(asyncio.create_task(mgr.submit_and_wait(
                    MODEL, prompt=session, thread_id=thread,
                    client_meta={"ip": ip, "is_main": True, "session_id": session},
                )))
                await wait_until(lambda s=session: s in gated.started, timeout=SETUP_S)
            await wait_until(lambda: len(mgr._last_request_identity_by_model) == 2, timeout=SETUP_S)
            out["after_admissions"] = snap(mgr._last_request_identity_by_model)
            gated.release_all()
            await wait_until(lambda: {"s-a", "s-b"} <= set(gated.done), timeout=SETUP_S)
            resident = mgr._residents[SECOND]
            await wait_until(lambda: resident.state is ResidentState.IDLE_EVICTABLE, timeout=SETUP_S)
            driver = resident.driver_task
            recorder = RecordingDict(mgr._last_request_identity_by_model)
            mgr._last_request_identity_by_model = recorder
            real_begin = mgr._begin_unload_locked
            begin_states = []

            def _spy(rr):
                begin_states.append(rr.state is ResidentState.DEAD)
                return real_begin(rr)

            mgr._begin_unload_locked = _spy
            driver.cancel()
            await asyncio.wait({driver}, timeout=SETUP_S)
            await wait_until(lambda: SECOND not in mgr._residents, timeout=SETUP_S)
            out["pops"] = [list(p) for p in recorder.pops]
            out["after_finally"] = snap(dict(recorder))
            out["unload_calls_on_live_resident"] = begin_states.count(False)
        finally:
            gated.release_all()
            await asyncio.wait_for(asyncio.gather(*gated.tasks, return_exceptions=True), timeout=SETUP_S)
            await mgr.shutdown()
    return out


# ---- scenario drivers: finished-request outcome line ----------------------------------

def _block(handle, spawn_seq, slot, cache):
    return {
        "generation_id": compute_generation_id(handle.pid, spawn_seq, slot.slot_id),
        "n_prompt_cache": cache,
    }


def run_outcome(world, case, caplog):
    """The KV_REUSE_OUTCOME line of one finished request on duo#1 (or on an unknown handle).

    Returns the part of the line after ``wait_state=completed``."""
    mgr = world.mgr
    slot = make_slot("b")
    a, b = world.residents[MODEL], world.residents[SECOND]
    if case == "sibling":
        handle = b.handle
        mgr.live_generations[MODEL] = _block(a.handle, 1, slot, 11111)
        mgr.live_generations[SECOND] = _block(b.handle, 2, slot, 77777)
    elif case == "unknown_handle":
        handle = world.stray
        mgr.live_generations[MODEL] = _block(handle, 1, slot, 11111)
        mgr.live_generations[SECOND] = _block(b.handle, 2, slot, 77777)
    elif case == "sibling_only_model_block":
        handle = b.handle
        mgr.live_generations[MODEL] = _block(b.handle, 2, slot, 55555)
    else:
        raise ValueError(case)
    caplog.set_level("INFO")
    mgr._log_kv_reuse_outcome(slot, handle, wait_state="completed")
    lines = [r.message for r in caplog.records if r.message.startswith("KV_REUSE_OUTCOME")]
    if len(lines) != 1:
        return f"<{len(lines)} lines>"
    return lines[0].split("wait_state=completed ", 1)[1]
