"""The per-resident LAST request identity when two residents share one model_tag.

The entry ``_last_request_identity_by_model`` is written at ADMISSION (the driver binds the request to
its resident) under that resident's ``resident_key`` (not at enqueue under ``model_tag``), and it is popped
only for that resident, so two residents of one model never read one shared entry. I1-I7 below are the
cases; I7 is the one static check that names the helper.

The seam is the real ``submit_and_wait`` -> dispatcher -> driver path with two engines of one model (the same
auto_place route as tests/test_parked_claim_other_routes.py), except I3 (the enqueue emit called
directly), I5 (the real teardown called white-box, as the stale-reap test in test_per_resident_identity.py does) and
I7 (the source read with ``ast``).
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from _fastlane_fixture import wait_until
from _multiinstance_support import _manifest, _new_manager, _probe

from turbohaul.manager import _SINGLETON_RESIDENT_KEY, Resident, ResidentState

MODEL = "duo"
SECOND = f"{MODEL}#1"
SETUP_S = 5.0
ENGINE_A_IP = "10.0.0.5"
ENGINE_B_IP = "10.0.0.6"


def _meta(session, ip):
    return {"ip": ip, "is_main": True, "session_id": session}


class _Gated:
    """Fake completion. A request whose session is in ``block`` holds its turn until released."""

    def __init__(self, mgr, block=()):
        self.gates = {session: asyncio.Event() for session in block}
        self.started = []
        self.done = []
        self.tasks = []
        self.open = False
        mgr._complete_fn = self._complete

    async def _complete(self, slot, handle):
        session = (slot.client_meta or {}).get("session_id")
        self.started.append(session)
        gate = self.gates.get(session)
        if gate is not None and not self.open:
            await gate.wait()
        self.done.append(session)
        return {"ok": True, "model": handle.model_tag}

    def release(self, session):
        self.gates[session].set()

    def open_all(self):
        self.open = True
        for gate in self.gates.values():
            gate.set()


async def _admit(mgr, gated, session, thread, ip):
    """Start one request through submit_and_wait as a task, and wait until its turn has started.

    submit_and_wait is the entry that emits the request identity (the bare submit does not), so the
    tests go through it. The task is kept and awaited at teardown; the fake completion decides when
    the turn ends.
    """
    gated.tasks.append(asyncio.create_task(mgr.submit_and_wait(
        MODEL, prompt=session, thread_id=thread, client_meta=_meta(session, ip),
    )))
    await wait_until(lambda: session in gated.started, timeout=SETUP_S)


async def _finish(mgr, gated):
    """Release every held turn, let the request tasks end, then shut the manager down."""
    gated.open_all()
    await asyncio.wait_for(asyncio.gather(*gated.tasks, return_exceptions=True), timeout=SETUP_S)
    await mgr.shutdown()


def _two_engines(mgr):
    keys = sorted(r.resident_key for r in mgr._instances_for(MODEL))
    assert keys == [MODEL, SECOND], f"per-resident setup: expected two engines of {MODEL}, got {keys}"


def _identities(mgr):
    """resident_key -> the request_identity /status shows on that resident's card."""
    by_pid = {entry["pid"]: entry for entry in mgr._residents_snapshot()}
    out = {}
    for key, r in list(mgr._residents.items()):
        if r.handle is not None and r.handle.pid in by_pid:
            out[key] = by_pid[r.handle.pid]["request_identity"]
    return out


def _session(identity):
    return None if identity is None else identity.get("session_id")


def _expect_card(mgr, key, session, when):
    got = _session(_identities(mgr).get(key))
    assert got == session, (
        f"per-resident {when}: resident {key!r} must show its OWN last request {session!r}, got {got!r}"
    )


async def _wait_for(read, expected, message):
    """Bounded wait (SETUP_S) until ``read()`` equals ``expected``, then assert it exactly.

    The admission stashes the identity, and that can finish after the turn has started (see
    _admit), so an immediate read can still see the previous request. The wait is on the value
    itself. If the value never arrives, the assertion fails with the LAST value seen.
    """
    last = {"got": None}

    def same():
        last["got"] = read()
        return last["got"] == expected

    try:
        await wait_until(same, timeout=SETUP_S)
    except AssertionError:
        pass
    assert last["got"] == expected, f"{message}, got {last['got']!r}"


async def _wait_card(mgr, key, session, when):
    """_expect_card, but first waits (bounded) until the card shows ``session``."""
    await _wait_for(
        lambda: _session(_identities(mgr).get(key)), session,
        f"per-resident {when}: resident {key!r} must show its OWN last request {session!r}",
    )


class TestSameModelResidentsKeepTheirOwnIdentity:
    async def test_two_same_model_residents_keep_their_own_last_identity(self, tmp_path):
        """I1: residents "duo" and "duo#1" (one model_tag, two engines) each serve one request with
        a different session. /status must show each card ITS OWN request, not the other's."""
        mgr, boot, _ = _new_manager(tmp_path, budget=2)
        _manifest(boot, MODEL, auto_place=True, llama_server_flags={"split_mode": "none"})
        mgr.runtime.queue.grace_seconds = 2
        mgr.runtime.queue.idle_hot_load_seconds = 600
        gated = _Gated(mgr, block={"s-a1", "s-b1"})
        p_free, p_probe = _probe([20000, 20000])
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        with p_free, p_probe:
            try:
                await _admit(mgr, gated, "s-a1", "t-a", ENGINE_A_IP)
                await _admit(mgr, gated, "s-b1", "t-b", ENGINE_B_IP)
                _two_engines(mgr)
                await _wait_card(mgr, MODEL, "s-a1", "after both admissions")
                await _wait_card(mgr, SECOND, "s-b1", "after both admissions")
            finally:
                await _finish(mgr, gated)

    async def test_interleaved_admissions_never_overwrite_the_other_residents_identity(self, tmp_path):
        """I2: admissions A1, B1, A2, B2 in that order (A2 and B2 are same-thread follow-ups of A1
        and B1, served by the resident that holds them). After EACH admission the admitted resident
        shows the new request and the other resident still shows its own latest one."""
        mgr, boot, _ = _new_manager(tmp_path, budget=2)
        _manifest(boot, MODEL, auto_place=True, llama_server_flags={"split_mode": "none"})
        mgr.runtime.queue.grace_seconds = 2
        mgr.runtime.queue.idle_hot_load_seconds = 600
        gated = _Gated(mgr, block={"s-a1", "s-b1", "s-a2", "s-b2"})
        p_free, p_probe = _probe([20000, 20000])
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        with p_free, p_probe:
            try:
                await _admit(mgr, gated, "s-a1", "t-a", ENGINE_A_IP)
                await _wait_card(mgr, MODEL, "s-a1", "after A1")
                await _admit(mgr, gated, "s-b1", "t-b", ENGINE_B_IP)
                _two_engines(mgr)
                _expect_card(mgr, MODEL, "s-a1", "after B1: the first resident keeps A1")
                await _wait_card(mgr, SECOND, "s-b1", "after B1")
                gated.release("s-a1")
                await _admit(mgr, gated, "s-a2", "t-a", ENGINE_A_IP)
                await _wait_card(mgr, MODEL, "s-a2", "after A2")
                _expect_card(mgr, SECOND, "s-b1", "after A2: the second resident keeps B1")
                gated.release("s-b1")
                await _admit(mgr, gated, "s-b2", "t-b", ENGINE_B_IP)
                await _wait_card(mgr, SECOND, "s-b2", "after B2")
                _expect_card(mgr, MODEL, "s-a2", "after B2: the first resident keeps A2")
            finally:
                await _finish(mgr, gated)

    def test_enqueue_emit_does_not_write_the_per_resident_dict(self, tmp_path):
        """I3: the enqueue-time emit keeps the log line and the global alias, but it no longer writes
        the per-resident dict (the admission binds are its only writers)."""
        mgr, _boot, _ = _new_manager(tmp_path, budget=1)
        mgr._emit_request_identity(
            ip=ENGINE_A_IP, model_tag=MODEL, client_meta=_meta("s-enq", ENGINE_A_IP),
            thread_id="t-enq", fastlane=None,
        )
        assert mgr._last_request_identity is not None, "per-resident guard: the global alias must still be set"
        assert mgr._last_request_identity["session_id"] == "s-enq", (
            "per-resident guard: the global alias must still carry the admitted request"
        )
        assert mgr._last_request_identity_by_model == {}, (
            "the enqueue emit must not write the per-resident dict, it holds "
            f"{mgr._last_request_identity_by_model!r}"
        )

    async def test_follow_up_admission_updates_the_identity(self, tmp_path):
        """I4: a same-thread grace follow-up matched to the resident (the `r.active_slot = matched`
        bind in _serve_on_resident) becomes the resident's latest request: its identity replaces the
        anchor's while it is served, and it stays after it finishes. Guard."""
        mgr, boot, _ = _new_manager(tmp_path, budget=1)
        _manifest(boot, MODEL)
        mgr.runtime.queue.grace_seconds = 5
        mgr.runtime.queue.idle_hot_load_seconds = 600
        gated = _Gated(mgr, block={"s-follow"})
        p_free, p_probe = _probe([20000, 20000])
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        with p_free, p_probe:
            try:
                await _admit(mgr, gated, "s-anchor", "t-x", ENGINE_A_IP)
                await _wait_card(mgr, MODEL, "s-anchor", "sanity: the anchor was admitted")
                await _admit(mgr, gated, "s-follow", "t-x", ENGINE_A_IP)
                await _wait_card(mgr, MODEL, "s-follow", "while the same-thread follow-up is served")
                gated.release("s-follow")
                await wait_until(lambda: "s-follow" in gated.done, timeout=SETUP_S)
                _expect_card(mgr, MODEL, "s-follow", "after the follow-up finished: last stays")
            finally:
                await _finish(mgr, gated)


class TestEvictionPrunesOnlyItsOwnIdentity:
    async def test_eviction_pops_only_that_residents_identity(self, tmp_path):
        """I5: tearing a resident down pops ITS OWN entry only (no "other live resident of
        the tag" guard remains). Part 1: "duo#1" goes while "duo" lives, so "duo" keeps its entry,
        and a NEW resident under "duo#1" starts with no entry. Part 2: "duo" goes while "duo#1"
        lives, so "duo#1" keeps its entry. White-box: _begin_unload_locked is the real teardown and
        the residents are registered by hand, as the stale-reap test in test_per_resident_identity.py does."""
        mgr, _boot, _ = _new_manager(tmp_path, budget=2)
        failures = []
        id_a = {"session_id": "s-a", "model_tag": MODEL}
        id_b = {"session_id": "s-b", "model_tag": MODEL}
        ra = Resident(model_tag=MODEL, resident_key=MODEL, state=ResidentState.IDLE_EVICTABLE)
        rb = Resident(model_tag=MODEL, resident_key=SECOND, instance_idx=1,
                      state=ResidentState.IDLE_EVICTABLE)
        mgr._residents[MODEL] = ra
        mgr._residents[SECOND] = rb
        by_model = mgr._last_request_identity_by_model
        by_model[MODEL] = id_a
        by_model[SECOND] = id_b
        try:
            mgr._begin_unload_locked(rb)
            if SECOND in by_model:
                failures.append(f"part 1: '{SECOND}' still has an entry after its teardown")
            if by_model.get(MODEL) != id_a:
                failures.append(f"part 1: '{MODEL}' lost or changed its entry, got {by_model.get(MODEL)!r}")
            mgr._residents[SECOND] = Resident(
                model_tag=MODEL, resident_key=SECOND, instance_idx=1,
                state=ResidentState.IDLE_EVICTABLE,
            )
            if SECOND in by_model:
                failures.append(f"part 1: a NEW resident under '{SECOND}' starts with an entry")
            mgr._residents[MODEL] = ra
            by_model[SECOND] = id_b
            mgr._begin_unload_locked(ra)
            if MODEL in by_model:
                failures.append(f"part 2: '{MODEL}' still has an entry after its teardown")
            if by_model.get(SECOND) != id_b:
                failures.append(f"part 2: '{SECOND}' lost or changed its entry, got {by_model.get(SECOND)!r}")
        finally:
            await mgr.shutdown()
        assert not failures, "per-resident eviction must pop only the torn-down resident: " + "; ".join(failures)


class TestCapOneAndStaticAdmissionSites:
    async def test_cap1_singleton_identity_unchanged(self, tmp_path):
        """I6 (guard): one resident at cap 1. Its entry is keyed by its model_tag (its resident_key),
        the global alias carries the same request, and nothing is stored under the singleton key."""
        mgr, boot, _ = _new_manager(tmp_path, budget=1)
        _manifest(boot, MODEL)
        mgr.runtime.queue.grace_seconds = 2
        mgr.runtime.queue.idle_hot_load_seconds = 600
        gated = _Gated(mgr)
        p_free, p_probe = _probe([20000, 20000])
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        with p_free, p_probe:
            try:
                await _admit(mgr, gated, "s-solo", "t-solo", ENGINE_A_IP)
                await wait_until(lambda: "s-solo" in gated.done, timeout=SETUP_S)
                await _wait_for(
                    lambda: (mgr._last_request_identity_by_model.get(MODEL) or {}).get("session_id"),
                    "s-solo", "per-resident cap1: the entry must hold the request",
                )
                by_model = mgr._last_request_identity_by_model
                assert set(by_model) == {MODEL}, (
                    f"per-resident cap1: the entry must be keyed by the model_tag, got {sorted(by_model)}"
                )
                assert _SINGLETON_RESIDENT_KEY not in by_model, (
                    "per-resident cap1: nothing may be stored under the singleton key"
                )
                assert by_model[MODEL]["session_id"] == "s-solo", (
                    f"per-resident cap1: the entry must hold the request, got {by_model[MODEL]!r}"
                )
                assert mgr._last_request_identity["session_id"] == "s-solo", (
                    "per-resident cap1: the global alias must carry the same request"
                )
            finally:
                await _finish(mgr, gated)

    def test_static_every_new_admission_site_calls_the_helper(self):
        """I7 (STATIC, the only test that names the helper): the ADMISSION binds in
        _drive_resident (the swap-path bind and the turn-start bind, 2 calls) and the matched
        follow-up bind in _serve_on_resident (1 call, on `matched`) call _stash_resident_identity;
        the enqueue emit and the legacy _process_slot do not. The source is read with ast, so a
        missing helper is a count failure, not an AttributeError."""
        src = (Path(__file__).resolve().parents[1] / "src" / "turbohaul" / "manager.py").read_text(
            encoding="utf-8")
        fns = {n.name: n for n in ast.walk(ast.parse(src))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def stash_calls(name):
            return [n for n in ast.walk(fns[name]) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_stash_resident_identity"]

        drive = stash_calls("_drive_resident")
        serve = stash_calls("_serve_on_resident")
        assert len(drive) == 2, (
            f"per-resident static: _drive_resident must call the stash helper at its 2 admission binds, "
            f"found {len(drive)}"
        )
        assert len(serve) == 1 and "matched" in ast.unparse(serve[0]), (
            "per-resident static: _serve_on_resident must stash the matched follow-up, found "
            f"{[ast.unparse(c) for c in serve]}"
        )
        assert not stash_calls("_emit_request_identity"), (
            "per-resident static: the enqueue emit must not stash per resident"
        )
        assert not stash_calls("_process_slot"), "per-resident static: the legacy singleton path must not stash"


def _resident(key, port, instance_idx):
    return Resident(
        model_tag=MODEL, resident_key=key, instance_idx=instance_idx, port=port,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
    )


class TestSnapshotCarriesResidentKey:
    def test_residents_snapshot_carries_resident_key_after_model_tag(self, tmp_path):
        """Additional test (its label I7 is also used by the static check
        above): every _residents_snapshot() entry carries "resident_key" equal to that resident's own
        resident_key, directly after "model_tag". Two residents of one model_tag ("duo" and "duo#1")
        are registered by hand (the recipe used by the resident-with-no-requests status test in test_per_resident_identity.py); the
        entries are told apart by their port. A single instance-0 resident is checked as well.
        The field must be present."""
        mgr, _boot, _ = _new_manager(tmp_path, budget=2)
        failures = []
        expected = {59500: MODEL, 59501: SECOND}
        mgr._residents[MODEL] = _resident(MODEL, 59500, 0)
        mgr._residents[SECOND] = _resident(SECOND, 59501, 1)
        seen = set()
        for entry in mgr._residents_snapshot():
            if entry.get("port") not in expected:
                continue
            key = expected[entry["port"]]
            seen.add(entry["port"])
            if entry.get("resident_key") != key:
                failures.append(
                    f"two residents: the entry for '{key}' carries resident_key "
                    f"{entry.get('resident_key')!r}, expected {key!r}"
                )
            if list(entry)[:2] != ["model_tag", "resident_key"]:
                failures.append(
                    f"two residents: the first keys of the entry for '{key}' are {list(entry)[:2]}, "
                    "expected model_tag then resident_key"
                )
        if seen != set(expected):
            failures.append(f"two residents: expected entries for ports {sorted(expected)}, saw {sorted(seen)}")
        mgr._residents.pop(SECOND, None)
        single = mgr._residents_snapshot()
        if len(single) != 1 or single[0].get("resident_key") != MODEL:
            failures.append(
                "single instance-0 resident: expected one entry with resident_key "
                f"{MODEL!r}, got {[e.get('resident_key') for e in single]!r}"
            )
        assert not failures, "per-resident snapshot must carry resident_key after model_tag: " + "; ".join(failures)
