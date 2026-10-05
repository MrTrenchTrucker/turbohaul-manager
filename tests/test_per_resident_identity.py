"""Per-resident request_identity in /status.residents[].

Each resident now carries its OWN last-served request identity, additive to the
existing top-level singleton `self._last_request_identity` (Queue.tsx's contract,
untouched). Keyed by model_tag, identical to how `generation` is already keyed
one line above it in `_residents_snapshot`.

LIFETIME (a design choice): `_last_request_identity_by_model`
is pruned on FULL EVICTION only -- mirroring `live_generations.pop(tag, None)`
(live_monitor.py) -- never on an idle transition. An idle-but-still
-resident engine must keep showing what it last served; only a resident that is
actually torn down and removed from `self._residents` loses its entry. The pop
lives INSIDE the existing `if self._residents.get(r.model_tag) is r:` guard at
both teardown sites (manager.py `_begin_unload_locked` and `_drive_resident`'s own
`finally` block) -- a stale reap for an already-superseded resident must never
delete a NEWER resident's identity that has since taken over the same model_tag.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

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
from turbohaul.subprocess_mgr import SidecarHandle

# NOTE: pyproject.toml sets asyncio_mode = "auto" -- async def tests need no
# @pytest.mark.asyncio marker, matching every other file in this suite.


# ---------------------------------------------------------------------------
# Fixtures -- local copies, matching the established per-file convention (see
# test_multislot_concurrency.py, the idle-park tip-freshness test,
# etc: every test file that needs these keeps its own copy rather than
# importing another test module's private helpers).
# ---------------------------------------------------------------------------

def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2,
                            grace_seconds=0, idle_hot_load_seconds=0):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _high_vram():
    return patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000])


def _mocks(spawn_calls, sigterm_calls=None, complete_gate=None,
           gate_model=None, raise_model=None):
    pid = [90000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        if sigterm_calls is not None:
            sigterm_calls.append(handle.model_tag)
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        if complete_gate is not None and (
            gate_model is None or handle.model_tag == gate_model
        ):
            await complete_gate.wait()
        if raise_model is not None and handle.model_tag == raise_model:
            raise RuntimeError("boom in complete")
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _snapshot_by_tag(mgr) -> dict[str, dict]:
    return {entry["model_tag"]: entry for entry in mgr._residents_snapshot()}


# ---------------------------------------------------------------------------
# Core wiring: no-flip, None-not-missing, top-level global untouched
# ---------------------------------------------------------------------------

class TestPerResidentIdentityWiring:
    async def test_two_residents_carry_their_own_identity_no_flip(self, tmp_path):
        """The no-flip property at the API level: with 2 residents, each
        resident's request_identity must be ITS OWN, not the other's, and the
        two must NOT be equal."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", main_gpu=0)
        _seed_manifest(boot, "m2", main_gpu=1)
        mgr = _mk(boot, runtime, **_mocks([]))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-m1"},
                    ), timeout=5,
                )
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m2", "b", thread_id="t2",
                        client_meta={"session_id": "sess-m2"},
                    ), timeout=5,
                )
                snap = _snapshot_by_tag(mgr)
                id1 = snap["m1"]["request_identity"]
                id2 = snap["m2"]["request_identity"]
                assert id1 is not None and id2 is not None
                assert id1["session_id"] == "sess-m1"
                assert id2["session_id"] == "sess-m2"
                assert id1 != id2, "no-flip: two residents must not show the same identity"
            finally:
                await mgr.shutdown()

    async def test_resident_with_no_requests_has_none_not_missing_not_borrowed(self, tmp_path):
        """A resident that has served nothing -> request_identity is None,
        never absent from the dict and never another resident's value."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", main_gpu=0)
        _seed_manifest(boot, "m2", main_gpu=1)
        mgr = _mk(boot, runtime, **_mocks([]))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-m1"},
                    ), timeout=5,
                )
                # m2 never served -- reserve it directly into the registry
                # without a request, mirroring how a loading/idle resident
                # can exist with zero served traffic.
                mgr._residents["m2"] = Resident(
                    model_tag="m2", state=ResidentState.IDLE_EVICTABLE,
                )
                snap = _snapshot_by_tag(mgr)
                assert "request_identity" in snap["m2"], "key present, not missing"
                assert snap["m2"]["request_identity"] is None
                assert snap["m1"]["request_identity"] is not None
            finally:
                await mgr.shutdown()

    async def test_top_level_global_still_present_and_unchanged(self, tmp_path):
        """The existing self._last_request_identity singleton (Queue.tsx's
        contract) stays exactly as it was -- additive only."""
        boot, runtime = _boot_runtime_multislot(tmp_path, idle_hot_load_seconds=120)
        _seed_manifest(boot, "m1", main_gpu=0)
        mgr = _mk(boot, runtime, **_mocks([]))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-global"},
                    ), timeout=5,
                )
                status = mgr.status_snapshot()
                assert status["request_identity"] is not None
                assert status["request_identity"]["session_id"] == "sess-global"
                assert status["request_identity"] == mgr._last_request_identity
            finally:
                await mgr.shutdown()


# ---------------------------------------------------------------------------
# Lifetime: prune on eviction, NEVER on idle -- the settled lifetime rule
# ---------------------------------------------------------------------------

class TestPerResidentIdentityLifetime:
    async def test_idle_but_still_resident_keeps_its_identity(self, tmp_path):
        """An engine sitting IDLE (not evicted, still in self._residents) must
        keep showing its last-served identity -- popping here would blank
        every idle card's last-request strip (the exact regression this behaviour
        exists to prevent)."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", main_gpu=0)
        mgr = _mk(boot, runtime, **_mocks([]))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-idle"},
                    ), timeout=5,
                )
                await asyncio.sleep(0.2)  # settle into IDLE_EVICTABLE/IDLE_HOT
                r = mgr._residents.get("m1")
                assert r is not None, "still resident, not evicted"
                assert mgr._last_request_identity_by_model.get("m1") is not None
                assert mgr._last_request_identity_by_model["m1"]["session_id"] == "sess-idle"
            finally:
                await mgr.shutdown()

    async def test_lru_evicted_resident_identity_is_pruned(self, tmp_path):
        """LRU-evict path (_begin_unload_locked): once a resident is fully torn
        down and removed from self._residents, its by-model identity entry is
        gone too -- mirrors live_generations.pop(tag, None)."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", main_gpu=0)
        _seed_manifest(boot, "m2", main_gpu=1)
        _seed_manifest(boot, "m3", main_gpu=1)
        gate = asyncio.Event()
        # m1 stays busy (blocks on the gate) so it is never eviction-eligible;
        # m2 completes fast -> IDLE_EVICTABLE -> the one m3 evicts for capacity.
        mgr = _mk(boot, runtime, **_mocks(
            [], complete_gate=gate, gate_model="m1",
        ))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                await asyncio.sleep(0.2)
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m2", "b", thread_id="t2",
                        client_meta={"session_id": "sess-m2-evicted"},
                    ), timeout=5,
                )
                await asyncio.sleep(0.15)  # m2 -> IDLE_EVICTABLE
                assert mgr._last_request_identity_by_model.get("m2") is not None, (
                    "sanity: identity populated before eviction"
                )
                f3 = asyncio.create_task(
                    mgr.submit_and_wait("m3", "c", thread_id="t3")
                )
                await asyncio.sleep(0.4)
                assert mgr._residents.get("m2") is None, "m2 evicted"
                assert mgr._last_request_identity_by_model.get("m2") is None, (
                    "evicted resident's identity must be pruned"
                )
                gate.set()
                await asyncio.wait_for(asyncio.gather(f1, f3), timeout=5)
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_driver_death_resident_identity_is_pruned(self, tmp_path):
        """The OTHER teardown path (_drive_resident's own finally block, driver
        -death reap) prunes identically -- both sites carry the same 1-line
        addition, this proves the second one independently."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m1", main_gpu=0)
        mgr = _mk(boot, runtime, **_mocks([], raise_model="m1"))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(
                        mgr.submit_and_wait(
                            "m1", "a", thread_id="t1",
                            client_meta={"session_id": "sess-m1-died"},
                        ), timeout=5,
                    )
                await asyncio.sleep(0.2)
                assert mgr._residents.get("m1") is None, "dead resident deregistered"
                assert mgr._last_request_identity_by_model.get("m1") is None, (
                    "driver-death teardown must prune too, not only LRU-evict"
                )
            finally:
                await mgr.shutdown()


# ---------------------------------------------------------------------------
# The guard: a stale reap must never delete a NEWER resident's identity
# ---------------------------------------------------------------------------

class TestTagReuseGuard:
    async def test_stale_reap_does_not_delete_a_newer_residents_identity(self, tmp_path):
        """Correctness point: the pop lives INSIDE the
        `if self._residents.get(r.model_tag) is r:` guard. Constructs the
        exact race directly (white-box, no timing dependency): an old
        resident object r_old that has ALREADY been superseded by r_new under
        the same model_tag, then calls the real _begin_unload_locked(r_old).
        If the pop were placed OUTSIDE that guard (the bug this test was written to catch before
        it shipped), this test would fail -- r_new's identity would vanish.
        async def (not sync): _begin_unload_locked's own _spawn_bg(...) calls
        asyncio.create_task internally and needs a running loop even though
        this test never awaits that background teardown itself."""
        boot, runtime = _boot_runtime_multislot(tmp_path)
        mgr = _mk(boot, runtime, **_mocks([]))
        r_old = Resident(model_tag="shared-tag", state=ResidentState.ACTIVE)
        r_new = Resident(model_tag="shared-tag", state=ResidentState.ACTIVE)
        try:
            # r_new has already taken over the registry slot for this tag.
            mgr._residents["shared-tag"] = r_new
            mgr._last_request_identity_by_model["shared-tag"] = {"session_id": "sess-new"}
            # A stale reap for the OLD object (e.g. a delayed driver-death path
            # for an instance already superseded) must be a no-op on both dicts.
            mgr._begin_unload_locked(r_old)
            assert mgr._residents.get("shared-tag") is r_new, (
                "existing guard: newer resident must not be deregistered"
            )
            assert mgr._last_request_identity_by_model.get("shared-tag") == {"session_id": "sess-new"}, (
                "newer resident's identity must survive a stale reap for the old instance"
            )
        finally:
            await mgr.shutdown()


# ---------------------------------------------------------------------------
# current_request_identity -- CURRENT, not LAST.
# Arm 1 is the feature: a `current` that merely mirrors `last` passes any
# test that only checks "the key exists and has a value" -- the null-when
# -idle case is what actually proves it tracks the LIVE request, not the
# most recent one. Arm 3 is LOAD-BEARING: it drives status_snapshot()
# directly on the cap<=1 path, which is where residents is EMPTY -- catching
# whether steps 2 AND 3 (per-resident AND singleton wiring) both landed, not
# just one of them.
# ---------------------------------------------------------------------------

class TestCurrentRequestIdentity:
    async def test_arm1_idle_current_is_none_last_stays_populated(self, tmp_path):
        """THE FEATURE: once a request FINISHES, current_request_identity
        must go back to None -- while request_identity (LAST) keeps showing
        it. This is the exact defect being guarded against: a finished session
        must never keep looking live. A `current` that just mirrors `last`
        would pass a test that only asserts "not None" -- this asserts NONE.

        idle_hot_load_seconds=120 (matching this file's own established
        pattern, e.g. test_idle_but_still_resident_keeps_its_identity):
        with the default 0, grace-expired triggers IMMEDIATE full teardown,
        which prunes _last_request_identity_by_model too (per this file's
        own eviction-pruning tests) -- that would defeat this arm's premise
        by removing the resident from the snapshot entirely rather than
        leaving it idle-but-present."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=2, idle_hot_load_seconds=120,
        )
        _seed_manifest(boot, "m1", main_gpu=0)
        mgr = _mk(boot, runtime, **_mocks([]))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-finished"},
                    ), timeout=5,
                )
                # _process_slot's own finally (where _set_active_slot(None)
                # lives) reaches AFTER the completion future is already
                # resolved -- same async settle window every other test in
                # this file already accounts for (see the 0.15-0.4s sleeps
                # above); confirmed by first observing the race, not assumed.
                await asyncio.sleep(0.2)
                snap = _snapshot_by_tag(mgr)
                assert snap["m1"]["request_identity"] is not None, (
                    "sanity: last-served identity must still be populated"
                )
                assert snap["m1"]["request_identity"]["session_id"] == "sess-finished"
                assert snap["m1"]["current_request_identity"] is None, (
                    "current must be NULL once the request has finished -- "
                    "a value here means it is mirroring `last`, not tracking "
                    "what is actually live"
                )
            finally:
                await mgr.shutdown()

    async def test_arm2_live_request_current_matches_the_active_slot(self, tmp_path):
        """A request genuinely IN FLIGHT (held open via complete_gate) must
        have current_request_identity populated and matching that request's
        own identity fields -- not None, not some other resident's."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate, gate_model="m1"))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-live", "ip": "10.0.0.9"},
                    )
                )
                await asyncio.sleep(0.2)  # let it reach ACTIVE and block on the gate
                snap = _snapshot_by_tag(mgr)
                current = snap["m1"]["current_request_identity"]
                assert current is not None, "a genuinely in-flight request must show CURRENT"
                assert current["session_id"] == "sess-live"
                assert current["ip"] == "10.0.0.9"
                # And request_identity (LAST) agrees -- both describe the SAME
                # request while it's the one being served, they just answer
                # different questions (is it still running vs. what ran last).
                assert snap["m1"]["request_identity"]["session_id"] == "sess-live"
            finally:
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
                await mgr.shutdown()

    async def test_arm3_cap_le_1_status_snapshot_carries_current_when_residents_empty(self, tmp_path):
        """LOAD-BEARING: the cap<=1 path. residents is EMPTY here (the
        per-resident snapshot only populates at cap>=2) -- so this proves
        status_snapshot() ITSELF was wired (step 3), not just
        _residents_snapshot() (step 2). Exercises BOTH the live case and the
        idle-after-finish case, on the singleton self._active_slot."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=1)
        _seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate, gate_model="m1"))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                assert mgr._residents_snapshot() == [], (
                    "sanity: residents must be EMPTY at cap<=1"
                )
                f1 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-one-resident-cap-live"},
                    )
                )
                await asyncio.sleep(0.2)
                status_live = mgr.status_snapshot()
                assert status_live["residents"] == [], (
                    "residents stays EMPTY at cap<=1 even while busy -- "
                    "current_request_identity MUST be visible via "
                    "status_snapshot directly, or the feature is invisible "
                    "on this common configuration"
                )
                assert status_live["current_request_identity"] is not None
                assert status_live["current_request_identity"]["session_id"] == "sess-one-resident-cap-live"

                gate.set()
                await asyncio.wait_for(f1, timeout=5)
                # same post-completion settle window as arm 1 -- _set_active_slot(None)
                # lives in _process_slot's finally, which reaches after the
                # completion future is already resolved.
                await asyncio.sleep(0.2)
                status_idle = mgr.status_snapshot()
                assert status_idle["current_request_identity"] is None, (
                    "back to idle -- current must be null again"
                )
                assert status_idle["request_identity"] is not None, (
                    "last must still show the finished request"
                )
                assert status_idle["request_identity"]["session_id"] == "sess-one-resident-cap-live"
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_arm4_active_match_streaming_followup_is_current_then_none_on_grace(
        self, tmp_path,
    ):
        """Regression guard: the
        obvious null-check on `_active_slot` is NOT enough. A same-thread
        follow-up served warm via ACTIVE_MATCH during the grace window (the
        real shape of a streaming generation, as observed:
        n_decoded climbing while resident.current stayed correct and
        top.current stayed wrongly null) writes `r.active_slot` directly at
        the ACTIVE_MATCH sites (manager.py, two places) -- the exact writer
        that bypassed `_set_active_slot` and left `self._active_slot`
        structurally blind to this path. Neither of arms 1-3 nor any
        other test exercises ACTIVE_MATCH at all --
        that is precisely why they all passed while the top-level
        feature was still broken on the main path.

        Asserts BOTH halves of that observation:
          (a) while the matched follow-up is ACTIVE, current_request_identity
              matches the FOLLOW-UP's identity, not the (finished) anchor's.
          (b) the instant it pops back to GRACE with no further follow-up,
              current_request_identity is None again -- the half that the original
              three mutations could not reach, because a `current` built from
              `self._active_slot is not None` alone would still show the
              anchor as "current" throughout the grace window even with the
              ACTIVE_MATCH mirror fix alone (self._active_slot never clears
              until full teardown) -- this is what the state-gate inside
              `_current_request_identity` exists to catch.
        """
        boot, runtime = _boot_runtime_multislot(
            tmp_path, max_parallel_sidecars=1, grace_seconds=5,
        )
        _seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        # Both requests share model_tag="m1" by design (ACTIVE_MATCH requires
        # it) -- _mocks' gate_model can only key on model_tag, which would
        # gate the ANCHOR too. Gate on the SLOT's own identity instead, so
        # only the follow-up blocks.
        mocks = _mocks([])

        async def fake_complete_gated(slot, handle):
            if (slot.client_meta or {}).get("session_id") == "sess-followup":
                await gate.wait()
            return {"ok": True, "model": handle.model_tag}

        mocks["complete_fn"] = fake_complete_gated
        mgr = _mk(boot, runtime, **mocks)
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                # Anchor: completes fast (no gate held for it), enters GRACE.
                await asyncio.wait_for(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t-match",
                        client_meta={"session_id": "sess-anchor"},
                    ), timeout=5,
                )
                # Same thread_id, submitted while the anchor is in its grace
                # window -> pop_matched_thread finds it, served warm via
                # ACTIVE_MATCH on the SAME resident/handle. Held open on the
                # gate so we can observe status_snapshot mid-flight.
                f2 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m1", "b", thread_id="t-match",
                        client_meta={"session_id": "sess-followup"},
                    )
                )
                await asyncio.sleep(0.3)  # let it reach ACTIVE_MATCH -> ACTIVE
                status_matched = mgr.status_snapshot()
                current = status_matched["current_request_identity"]
                assert current is not None, (
                    "a live ACTIVE_MATCH follow-up must show as CURRENT at "
                    "the top level, not just on the per-resident surface"
                )
                assert current["session_id"] == "sess-followup", (
                    "current must describe the FOLLOW-UP actually being "
                    "served, not the finished anchor self._active_slot used "
                    "to be stuck on"
                )

                gate.set()
                await asyncio.wait_for(f2, timeout=5)
                await asyncio.sleep(0.05)  # settle back onto the anchor/GRACE
                status_after = mgr.status_snapshot()
                assert status_after["current_request_identity"] is None, (
                    "back in GRACE with no further follow-up -- current must "
                    "be null again immediately, not held for the rest of "
                    "the grace window"
                )
                assert status_after["request_identity"]["session_id"] == "sess-followup", (
                    "sanity: LAST still correctly updated to the follow-up"
                )
            finally:
                gate.set()
                await mgr.shutdown()

    async def test_arm5_cap_ge_2_status_snapshot_current_is_not_permanently_null(
        self, tmp_path,
    ):
        """ACCEPTANCE BAR: setups commonly
        run cap>=2 (the effective cap can be 2 despite the yaml saying 1), so
        the top-level surface must work THERE, not just at cap<=1.

        Before this rework, self._active_slot was PERMANENTLY unwritten at
        cap>=2 (worker_loop branches straight into _dispatch_loop, which
        never calls _process_slot -- the only place any of the five
        _set_active_slot call sites live) -- so status_snapshot's
        current_request_identity would have been null for EVERY request at
        cap>=2, live or not. This is the failure mode arm 3 (cap<=1) cannot
        reach: it drives status_snapshot() directly, but residents is EMPTY
        there BY DESIGN, so it never exercises _resolve_top_level_active_slot's
        _model_residents() branch at all.

        Asserts both halves at cap>=2 directly:
          (a) a genuinely in-flight request (held via complete_gate) ->
              status_snapshot()["current_request_identity"] is populated and
              matches it -- proving the top level is no longer permanently
              null under multi-residency.
          (b) once it finishes -> null again, while request_identity (LAST)
              still shows it -- the same null-when-idle bar as every other
              arm, now proven at the cap where it actually has to hold.
        """
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate, gate_model="m1"))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                assert mgr._active_slot is None, (
                    "sanity: the legacy singleton scalar is never touched at "
                    "cap>=2 -- confirming the mechanism this test exists to "
                    "route around, not just its symptom"
                )
                f1 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="t1",
                        client_meta={"session_id": "sess-two-resident-cap-live"},
                    )
                )
                await asyncio.sleep(0.2)  # reach ACTIVE, block on the gate
                status_live = mgr.status_snapshot()
                assert mgr._active_slot is None, (
                    "still never touched -- proves the fix does NOT route "
                    "through self._active_slot at cap>=2"
                )
                current = status_live["current_request_identity"]
                assert current is not None, (
                    "a live request at cap>=2 must show as CURRENT at the "
                    "top level -- this was UNCONDITIONALLY null before the "
                    "rework, live or not"
                )
                assert current["session_id"] == "sess-two-resident-cap-live"

                gate.set()
                await asyncio.wait_for(f1, timeout=5)
                await asyncio.sleep(0.2)  # settle, same convention as arm 1/3
                status_idle = mgr.status_snapshot()
                assert status_idle["current_request_identity"] is None, (
                    "idle again -- current must be null at cap>=2 too, not "
                    "just at cap<=1"
                )
                assert status_idle["request_identity"]["session_id"] == "sess-two-resident-cap-live", (
                    "sanity: LAST still populated"
                )
            finally:
                gate.set()
                await mgr.shutdown()


class TestLiveProtectedThreadHashesTwoResidentCap:
    """RED->GREEN: _live_protected_thread_hashes must scan residents
    at cap>=2 (not the dead self._active_slot / self._inflight scalars)."""

    async def test_two_resident_cap_active_resident_thread_is_protected(self, tmp_path):
        """Pre-fix (RED): _live_protected_thread_hashes read
        self._active_slot, which is PERMANENTLY unwritten at cap>=2 -- so the
        active thread's hash was MISSING from the protected set. KV-GC could
        then evict the live thread's KV .bin mid-serve.

        Post-fix (GREEN): it scans _model_residents(), so the active
        thread's hash IS in the protected set."""
        boot, runtime = _boot_runtime_multislot(tmp_path, max_parallel_sidecars=2)
        _seed_manifest(boot, "m1", main_gpu=0)
        gate = asyncio.Event()
        mgr = _mk(boot, runtime, **_mocks([], complete_gate=gate, gate_model="m1"))
        with _high_vram():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                f1 = asyncio.create_task(
                    mgr.submit_and_wait(
                        "m1", "a", thread_id="ref-live",
                        client_meta={"session_id": "sess-ref"},
                    )
                )
                await asyncio.sleep(0.2)
                assert mgr._active_slot is None, (
                    "sanity: legacy singleton scalar is never touched at cap>=2"
                )
                hashes = mgr._live_protected_thread_hashes()
                expected_hash = mgr._thread_hash("ref-live")
                assert expected_hash in hashes, (
                    "active resident thread hash must be in protected set at "
                    "cap>=2 -- pre-fix this was MISSING (KV-GC safety gap)"
                )
            finally:
                gate.set()
                await asyncio.wait_for(f1, timeout=5)
                await mgr.shutdown()
