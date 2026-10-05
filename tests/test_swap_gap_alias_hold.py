"""The live-residents supervisor must not report the
reserve window as idle.

_update_primary_alias mirrors the most-recently-active resident's generation
into the mgr.live_generation back-compat alias every ~1Hz tick. During a model
swap, residents[] can be genuinely empty for one tick -- the outgoing resident
already evicted, the incoming one not yet registered -- because a recent change
moved the VRAM probes off-loop, making this window observable to this
supervisor for the first time. Nulling the alias in that window is
indistinguishable, to every real consumer, from "nothing is loaded": the FE's
SSE anchor stream (api/live_stream.py) reads None and fires idle:true,
reset:true -- the exact symptom this fix exists to cure, reproduced by the fix
that unblocked the loop.

mgr._reserving_model_tag (from an earlier fix) already names this exact window.
The fix: hold the alias, don't null it, while a reservation is in flight.
(a) HOLD was chosen over (b) publish-the-incoming or (c) a third surface state
-- every consumer treats None as idle, and manager.py already has a
precedent (see the comment there) for holding the sibling
top-level surface the same way.
"""
from __future__ import annotations

import asyncio

import pytest

from turbohaul.live_monitor import LiveResidentsSupervisor


# ---------------------------------------------------------------------------
# GROUP A -- the subject fix, plus its two green controls. Local duck-type
# stubs, matching the established per-file convention (see
# test_live_monitor_prefill_pct.py: no shared TurbohaulManager needed to
# exercise _update_primary_alias in isolation).
# ---------------------------------------------------------------------------

class _FakeResident:
    def __init__(self, model_tag, last_active_monotonic=0.0):
        self.model_tag = model_tag
        self.last_active_monotonic = last_active_monotonic


class _FakeMgr:
    def __init__(self, *, reserving_model_tag=None, live_generation=None,
                 live_generations=None):
        self._reserving_model_tag = reserving_model_tag
        self.live_generation = live_generation
        self.live_generations = live_generations if live_generations is not None else {}


def _supervisor(mgr):
    # interval_s is irrelevant here -- _update_primary_alias is called directly,
    # never through run()/_tick()'s ~1Hz loop.
    return LiveResidentsSupervisor(mgr, interval_s=1.0)


class TestSwapGapHoldsAliasWhileReserving:
    def test_RED_empty_registry_with_reservation_in_flight_holds_alias(self):
        """THE SUBJECT. Registry empty (residents=[]) exactly as it is between
        the outgoing resident's eviction and the incoming one's registration.
        A reservation IS in flight (_reserving_model_tag set). PRE-FIX:
        best stays None, `mgr.live_generation = best` unconditionally nulls it
        -- this assertion fails. POST-FIX: the guard sees best is None AND a
        reservation is in flight, and returns before the write -- the prior
        value survives."""
        held_value = {"state": "idle", "generation_id": "old-gen-123"}
        mgr = _FakeMgr(
            reserving_model_tag="incoming-model",
            live_generation=held_value,
            live_generations={},   # nothing published yet for any tag
        )
        sup = _supervisor(mgr)

        sup._update_primary_alias([])   # residents=[] -- the dark window itself

        assert mgr.live_generation == held_value, (
            f"live_generation was nulled during an in-flight reservation "
            f"(_reserving_model_tag='incoming-model'); got {mgr.live_generation!r}, "
            f"expected the held prior value {held_value!r} -- this IS the FE "
            f"idle+reset symptom this fix exists to cure, reproduced by moving the "
            f"VRAM probes off-loop."
        )

    def test_GREEN_CONTROL_empty_registry_with_no_reservation_still_goes_idle(self):
        """Negative control, must pass on BOTH old and new code: when NOTHING is
        reserving (_reserving_model_tag is None) and the registry is genuinely
        empty, the alias MUST still null to idle -- this is the correct,
        unaffected 'truly nothing loaded' case. Proves the guard is scoped to
        the marker, not a blanket 'never null' hack that would just move the
        bug to a different, real-idle scenario."""
        mgr = _FakeMgr(
            reserving_model_tag=None,
            live_generation={"state": "idle", "generation_id": "stale-should-clear"},
            live_generations={},
        )
        sup = _supervisor(mgr)

        sup._update_primary_alias([])

        assert mgr.live_generation is None, (
            f"genuinely idle (no reservation, no residents) must still null "
            f"the alias; got {mgr.live_generation!r} -- a fix that never nulls "
            f"is just the same bug pointed the other way."
        )

    def test_GREEN_CONTROL_nonempty_registry_updates_normally_regardless_of_flag(self):
        """Negative control, must pass on BOTH old and new code: when a resident
        genuinely IS live and has a published generation, the alias updates
        normally to that generation -- whether or not _reserving_model_tag
        happens to be set for some OTHER tag at the same instant. Proves the
        guard only ever intercepts the best-is-None case; it never shadows a
        real update."""
        real_gen = {"state": "generating", "generation_id": "real-gen-456"}
        mgr = _FakeMgr(
            reserving_model_tag="some-other-incoming-model",
            live_generation={"state": "idle", "generation_id": "should-be-replaced"},
            live_generations={"active-model": real_gen},
        )
        sup = _supervisor(mgr)

        sup._update_primary_alias([_FakeResident("active-model", last_active_monotonic=5.0)])

        assert mgr.live_generation == real_gen, (
            f"a genuinely live resident's generation must always win, "
            f"reservation flag or not; got {mgr.live_generation!r}"
        )


# ---------------------------------------------------------------------------
# GROUP B -- a target-state invariant, pinned by an assertion (not a
# comment): the marker's window must cover the case where _route_or_reserve
# hands the actual spawn to a background task. Real TurbohaulManager fixture,
# matching the stranded-credit-release test's
# _reserve_and_start_locked-direct-call convention. This does NOT go red on
# today's code -- it is a regression pin on an ordering the fix above assumes
# holds, not the fix itself.
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(tmp_path):
    from turbohaul.config import (
        BootConfig, PullConfig, QueueConfig, RuntimeConfig,
        RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
    )
    from turbohaul.manager import TurbohaulManager

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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.mark.asyncio
class TestResidentRegisteredBeforeBackgroundHandoff:
    async def test_resident_registered_synchronously_before_driver_task_created(
        self, mgr, monkeypatch,
    ):
        """Pin the ordering _update_primary_alias's fix depends on: by the time
        _route_or_reserve returns (and _reserving_model_tag clears in its
        finally), the new Resident is ALREADY registered in _residents --
        inserted synchronously, with no await between the insert and
        asyncio.create_task(self._drive_resident(r)). If a future refactor
        moved the registration to AFTER the background hand-off, the dark
        window this fix closes would reopen by a different route: the marker
        clears, the supervisor's next tick sees nothing registered, and the
        just-added guard legitimately stops holding on a registry that still
        has nothing to publish.

        Verified live at the actual create_task call (matched by the
        coroutine's own code name, so any OTHER create_task in this path is
        left alone), not inferred after the fact from the final state.
        """
        from turbohaul.slot import Slot

        mgr._vram_admits_locked = lambda *a, **k: True
        mgr._resolve_placement_locked = lambda *a, **k: (1024, 1, 0, "none", 0, False, False)

        # Neutralise the real spawn (no llama-server binary exists in this
        # fixture) with a no-op coroutine NAMED _drive_resident -- the spy
        # below matches on the coroutine's own code name, so naming this
        # replacement identically to the real method is what lets the match
        # keep working under the mock, not a coincidence.
        async def _drive_resident(r):
            return None

        monkeypatch.setattr(mgr, "_drive_resident", _drive_resident)

        observed = {}
        real_create_task = asyncio.create_task

        def spy_create_task(coro, *a, **k):
            name = getattr(getattr(coro, "cr_code", None), "co_name", None)
            if name == "_drive_resident" and "snapshot" not in observed:
                observed["snapshot"] = list(mgr._instances_for("some-model"))
            return real_create_task(coro, *a, **k)

        monkeypatch.setattr(asyncio, "create_task", spy_create_task)

        # _route_or_reserve acquires _registry_lock ITSELF internally (unlike
        # _reserve_and_start_locked, which requires the caller to already hold
        # it) -- do not pre-acquire it here, that would deadlock on the
        # non-reentrant asyncio.Lock. Matches the existing convention in e.g.
        # the unlisted-candidate-veto test, which calls
        # _route_or_reserve directly with no outer lock.
        slot = Slot.new("some-model")
        await mgr._route_or_reserve(slot)

        assert "snapshot" in observed, (
            "the _drive_resident hand-off never fired -- fixture did not reach "
            "the reserve branch, this test proves nothing"
        )
        assert len(observed["snapshot"]) == 1, (
            f"the resident for the incoming tag must already be registered at "
            f"the exact instant the driver is handed to its background task "
            f"(create_task), got {observed['snapshot']!r} -- if this is empty, "
            f"a reorder has reopened the swap-gap by a different route than "
            f"the one this fix closed"
        )

        # The (now no-op) driver task completes on its own; let it settle so
        # no task is left pending at test teardown.
        r = mgr._instances_for("some-model")[0]
        if r.driver_task is not None:
            await r.driver_task
