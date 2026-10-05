"""Rule under test: a claim-holding slot never counts
as a 'starved other model' for a resident that is not the designated
victim -- a claim can never shorten a non-victim's grace.
Ordinary FIFO starvation breakout is unchanged.

Behaviour covered: manager.py's two grace loops (_serve_on_resident
"Loop A" and _process_slot "Loop B") each had an UNGATED
`starved_other_model` exit -- no priority/claim check whatsoever -- while
their sibling exits in the same two loops (`grace_fastlane_breakout`,
and its peers) DO gate on `has_strictly_higher_priority_waiting`. A
LOWER-priority Fast Lane claimant could strip a HIGHER-priority resident's
grace purely by aging past `max_other_model_wait_s`, because the
starvation exit did not consult the claim.

THE FIX: one new manager-level helper, `_starvation_breakout_candidate`,
called at BOTH sites instead of `queue.starved_other_model` directly.
Deliberately does NOT check `_is_designated_unload_target_locked` -- see that
helper's own docstring in manager.py for the equivalence argument
behind this: a victim structurally can never reach either
call site (Loop A: only entered from `skip_grace`'s `else` branch; Loop B:
`_is_designated_unload_target_locked` is vacuous for the legacy singleton, always
False), so gating unconditionally on "is the candidate itself a claim" is
exactly equivalent to gating on the full clause.

⛔⛔ MANDATORY CONTROL, the key one: an UNREGISTERED
(non-Fast-Lane) starved slot must STILL break grace -- the fairness floor
stays unchanged. See test_ordinary_fifo_waiter_still_breaks_grace_*
in each class below; without it the fix would be indistinguishable from
having deleted the fairness floor.
"""
import time

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
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.slot import Slot, SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


def _boot_runtime(tmp_path, *, grace_seconds=5, max_other_model_wait_s=0.2):
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
            safety_enabled=False,
            grace_seconds=grace_seconds,
            max_other_model_wait_s=max_other_model_wait_s,
            max_grace_extensions=50,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _audit_events(boot, slot_id):
    conn = open_state_db(boot.storage.state_db_path)
    cur = conn.execute(
        "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
        (slot_id,),
    )
    events = [r["event_type"] for r in cur.fetchall()]
    conn.close()
    return events


# === Group 1: the helper itself, isolated ===================================

@pytest.mark.asyncio
class TestStarvationBreakoutCandidateUnit:
    """Direct unit tests of `_starvation_breakout_candidate`, no manager
    driver loop needed -- exercises the queue interaction the helper wraps."""

    def _mgr(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        return TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=lambda *a, **k: {"ok": True},
        )

    async def test_fastlane_claim_candidate_is_suppressed(self, tmp_path):
        """THE defect fixed here, isolated to the helper: a staged
        Fast-Lane-claiming request aged past the threshold must NOT be
        handed back as a starvation-breakout candidate.

        MUST FAIL WITHOUT THE FIX -- before it, this was exactly
        `await self.queue.starved_other_model(slot.model_tag) is not None`,
        which is True here (the candidate IS aged past threshold); the
        fastlane claim was never consulted."""
        mgr = self._mgr(tmp_path)
        holder = Slot.new("m1", prompt="hi", thread_id="t1")
        claimant = Slot.new("m2", prompt="hi", thread_id="t2")
        claimant.fastlane = _match(rule_index=9)  # LOWER priority than any real waiter
        claimant.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(claimant)

        result = await mgr._starvation_breakout_candidate(holder)

        assert result is None, (
            "a Fast-Lane claim was allowed to break a non-victim's grace via "
            "the ordinary starvation path -- grace-priority violation"
        )

    async def test_ordinary_fifo_waiter_still_breaks_grace_MANDATORY_CONTROL(
        self, tmp_path
    ):
        """⛔⛔ THE KEY CONTROL. An UNREGISTERED
        (non-Fast-Lane) starved slot must STILL be handed back -- identical
        to the existing behaviour. Passes with and without the fix;
        without this control the fix is indistinguishable from having
        deleted the fairness floor entirely."""
        mgr = self._mgr(tmp_path)
        holder = Slot.new("m1", prompt="hi", thread_id="t1")
        waiter = Slot.new("m2", prompt="hi", thread_id="t2")
        assert waiter.fastlane is None  # ordinary FIFO request, not listed
        waiter.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(waiter)

        result = await mgr._starvation_breakout_candidate(holder)

        assert result is not None and result.slot_id == waiter.slot_id, (
            "an ordinary (non-claim) FIFO waiter must still break grace -- "
            "the fairness floor must not be touched by this fix"
        )

    async def test_none_when_nothing_starved(self, tmp_path):
        mgr = self._mgr(tmp_path)
        holder = Slot.new("m1", prompt="hi", thread_id="t1")
        assert await mgr._starvation_breakout_candidate(holder) is None


# === Group 2: Loop B (_process_slot, single-sidecar/legacy worker_loop) ====

@pytest.mark.asyncio
class TestProcessSlotLoopB:
    async def test_fastlane_claimant_does_not_strip_grace(self, tmp_path):
        """MUST FAIL WITHOUT THE FIX: a Fast-Lane
        claimant aged past max_other_model_wait_s strips slot_a's grace
        exactly like an ordinary waiter would -- grace_starvation_breakout
        fires and the loop exits well before grace_seconds."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(model_tag="m1", prompt="hi", thread_id="t1")

        claimant = Slot.new("m2", prompt="hi", thread_id="t2")
        claimant.fastlane = _match(rule_index=9)  # LOW priority -- ageing alone must not win
        claimant.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(claimant)

        mgr._worker_task = __import__("asyncio").create_task(mgr.worker_loop())

        started = time.monotonic()
        deadline = started + grace_seconds - 0.5
        while time.monotonic() < deadline:
            if "grace_starvation_breakout" in _audit_events(boot, slot_a.slot_id):
                break
            await __import__("asyncio").sleep(0.05)
        elapsed = time.monotonic() - started

        await mgr.shutdown()

        events = _audit_events(boot, slot_a.slot_id)
        assert "grace_starvation_breakout" not in events, (
            "a Fast-Lane claim stripped a non-victim resident's grace via "
            "the ordinary starvation path -- exactly the grace-priority violation "
            "fixed here"
        )
        assert elapsed >= grace_seconds - 1.0, (
            f"loop exited after only {elapsed:.2f}s -- something still broke "
            f"out early despite the fastlane claim"
        )

    async def test_ordinary_fifo_waiter_still_breaks_grace_MANDATORY_CONTROL(
        self, tmp_path
    ):
        """⛔⛔ THE CONTROL. Same scenario, but the staged waiter is
        ordinary (non-Fast-Lane) -- must still break grace, exactly as
        test_grace_starvation_breakout.py's own existing discriminator
        proves for the unfixed code. Fails identically on the unfixed code and on a
        broken 'always suppress' mutant."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot_a = await mgr.submit(model_tag="m1", prompt="hi", thread_id="t1")

        waiter = Slot.new("m2", prompt="hi", thread_id="t2")
        assert waiter.fastlane is None
        waiter.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(waiter)

        mgr._worker_task = __import__("asyncio").create_task(mgr.worker_loop())

        started = time.monotonic()
        deadline = started + grace_seconds - 0.5
        breakout_seen = False
        while time.monotonic() < deadline:
            if "grace_starvation_breakout" in _audit_events(boot, slot_a.slot_id):
                breakout_seen = True
                break
            await __import__("asyncio").sleep(0.05)
        elapsed = time.monotonic() - started

        await mgr.shutdown()

        assert breakout_seen, (
            "an ordinary FIFO waiter no longer breaks grace -- the fairness "
            "floor was damaged by this fix"
        )
        assert elapsed < grace_seconds - 1.0


# === Group 3: Loop A (_serve_on_resident, multi-slot dispatcher path) ======

@pytest.mark.asyncio
class TestServeOnResidentLoopA:
    async def test_fastlane_claimant_does_not_strip_grace(self, tmp_path):
        """MUST FAIL WITHOUT THE FIX: same defect, the other loop.
        Direct call, mirrors test_grace_starvation_breakout.py's own
        TestServeOnResidentGraceStarvationBreakout fixture shape."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        claimant = Slot.new("m2", prompt="hi", thread_id="t2")
        claimant.fastlane = _match(rule_index=9)
        claimant.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(claimant)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        import asyncio

        started = time.monotonic()
        # Unlike the ordinary-waiter discriminator, the fixed loop must NOT
        # exit early -- it runs to the full grace deadline. Expect the
        # TimeoutError from wait_for as the (correct) outcome here.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mgr._serve_on_resident(r, anchor, handle),
                timeout=grace_seconds - 0.5,
            )
        elapsed = time.monotonic() - started

        events = _audit_events(boot, anchor.slot_id)
        assert "grace_starvation_breakout" not in events, (
            "a Fast-Lane claim stripped a non-victim resident's grace via "
            "the ordinary starvation path (Loop A) -- grace-priority violation"
        )
        assert elapsed >= grace_seconds - 1.0

        await mgr.shutdown()

    async def test_ordinary_fifo_waiter_still_breaks_grace_MANDATORY_CONTROL(
        self, tmp_path
    ):
        """⛔⛔ THE CONTROL, Loop A side. Identical to
        test_grace_starvation_breakout.py's own
        test_discriminator_and_non_vacuity -- reproduced here so this file
        is a self-contained proof the fairness floor survives this fix."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=lambda *a, **k: None,
            health_fn=lambda *a, **k: True,
            sigterm_fn=lambda *a, **k: (True, "clean"),
            vram_fn=lambda **k: (True, 100),
            complete_fn=fake_complete,
        )

        waiter = Slot.new("m2", prompt="hi", thread_id="t2")
        assert waiter.fastlane is None
        waiter.created_at = time.monotonic() - 100.0
        await mgr.queue.enqueue(waiter)

        handle = _make_fake_handle("m1", 59500)
        r = Resident(
            model_tag="m1",
            handle=handle,
            port=handle.port,
            grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
        )
        anchor = Slot.new("m1", prompt="hi", thread_id="t1")
        anchor.state = SlotState.STAGED

        import asyncio

        started = time.monotonic()
        await asyncio.wait_for(
            mgr._serve_on_resident(r, anchor, handle),
            timeout=grace_seconds - 0.5,
        )
        elapsed = time.monotonic() - started

        assert elapsed < grace_seconds - 1.0
        events = _audit_events(boot, anchor.slot_id)
        assert "grace_starvation_breakout" in events

        await mgr.shutdown()


# === Group 4: two-call-site drift pin ===================

class TestBothCallSitesPinned:
    def test_neither_grace_loop_calls_queue_starved_other_model_directly(self):
        """Both real sites must route through `_starvation_breakout_candidate`
        exclusively -- the whole point of a shared helper is that there is
        only ONE entrance to get right. If a future edit reintroduces a
        direct `queue.starved_other_model` call in either loop, this must
        fail."""
        import inspect

        from turbohaul import manager as manager_mod

        src_serve = inspect.getsource(manager_mod.TurbohaulManager._serve_on_resident)
        src_process = inspect.getsource(manager_mod.TurbohaulManager._process_slot)

        for name, src in (("_serve_on_resident", src_serve), ("_process_slot", src_process)):
            assert "self.queue.starved_other_model(" not in src, (
                f"{name} calls queue.starved_other_model directly -- must go "
                f"through _starvation_breakout_candidate instead"
            )
            assert "self._starvation_breakout_candidate(" in src, (
                f"{name} does not call the shared starvation-breakout gate at all"
            )

    def test_helper_exists_and_is_the_sole_caller_of_starved_other_model(self):
        """Deliberately scoped to the three individual functions involved
        (not a whole-module/whole-class inspect.getsource -- that risks the
        ast/tokenize recursion fragility on a very large module) -- each is
        independently bounded and cheap to source. The scope is kept to
        those three functions only."""
        import inspect

        from turbohaul import manager as manager_mod

        assert hasattr(manager_mod.TurbohaulManager, "_starvation_breakout_candidate")
        src_helper = inspect.getsource(
            manager_mod.TurbohaulManager._starvation_breakout_candidate
        )
        src_serve = inspect.getsource(manager_mod.TurbohaulManager._serve_on_resident)
        src_process = inspect.getsource(manager_mod.TurbohaulManager._process_slot)

        total = sum(
            s.count("self.queue.starved_other_model(")
            for s in (src_helper, src_serve, src_process)
        )
        assert total == 1, (
            f"expected exactly one call to queue.starved_other_model across "
            f"the helper + both grace loops (got {total}) -- inside "
            f"_starvation_breakout_candidate itself; any other count means "
            f"a site bypassed the gate or the helper is missing/duplicated"
        )
