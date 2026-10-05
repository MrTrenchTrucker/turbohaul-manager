"""The spawn-time reclaim barrier.

The shape of these tests follows the Fast Lane identity work. Three things
these tests cover:

(a) `_spawn_gate_placement` mirrors `_run_spawn_safety_gate`'s OWN
    widened condition -- `(m.auto_place and msm == "none") or
    r.placement_overridden` -- not an auto_place-only test, which would
    miss relocated claimants. See TestWrongCard's
    relocated-claimant tests below for the proof.
(b) A relocated-claimant watch test (TestWrongCard.
    test_relocated_claimant_*) proves the disjunction actually matters: a
    claimant with placement_overridden=True and a live release task on its
    RELOCATED card must be watched -- a two-term (auto_place-only) condition would
    have watched the manifest's stale card instead and missed it.
(c) The evict->unload rename applies throughout: this file's
    barrier calls `_handle_unloaded_slot` on disconnect (there is no
    `_handle_evicted_slot`).

On a VRAM-only safety-gate refusal with a bounded reclaim ALREADY in flight
on the gated card(s), wait up to `spawn_reclaim_wait_max_s` (re-running the
FULL safety gate every tick or on task completion, whichever first) before
the terminal refusal. Everything else -- non-VRAM, VRAM with no live release
task, or an unclassified gate -- stays terminal on the first gate run,
byte-identical to today.

NOTE: the defect-reproduction test stubs _drive_resident and cannot see this
change. A green run there is a regression check, NEVER evidence.

=== Fast Lane claim-registry tests (the registry is in this tree) ===
These tests use the real claim registry:
- TEST 7 (RIDER PROTECTION, literal): TestRiderProtectionLiteral -- Fast Lane
  on, a real claim registered via the real _defer_unroutable funnel (not
  mocked), asserting `_fastlane_claims[K]` survives untouched and
  `_release_fastlane_claim_locked` is never called for the rider.
- TEST 8 (ANCHOR CLAIM NEUTRALITY): TestAnchorClaimNeutrality -- the barrier's
  OWN anchor slot carries a Fast Lane match; `_fastlane_claims` stays `{}`
  and both claim helpers are spied zero-called for it across a full
  barrier-enter-then-terminal-refuse cycle.
- TEST 9 (FAST LANE OFF/ON PARITY): TestFastLaneOnOffParity -- the
  `_fastlane_claims == {}` assertion is checked (the claim
  registry exists in this tree).
Related tests: the busy-defer budget test,
the Fast Lane claims test (it exists and is
exercised in this tree),
test_multislot_concurrency.py, test_safety.py, test_hybrid_kv.py,
the defect-reproduction test (regression check ONLY, never evidence).
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import yaml

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.safety import GateResult
from turbohaul.slot import Slot, SlotEvictedError, SlotState, VramOverCommitError
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle

_FASTLANE_RULES = [FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1))]


def _boot_runtime(tmp_path, *, spawn_reclaim_wait_max_s=10.0,
                   max_parallel_sidecars=2, fastlane_enabled=False):
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
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            spawn_reclaim_wait_max_s=spawn_reclaim_wait_max_s,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=fastlane_enabled,
            rules=_FASTLANE_RULES if fastlane_enabled else [],
        ),
    )
    # Pre-warm state.sqlite (WAL init + schema) synchronously, once, before any
    # test triggers two near-simultaneous audit writers (the barrier's own
    # spawn_reclaim_wait event racing _handle_unloaded_slot's pre-existing
    # fire-and-forget slot_evicted audit on the disconnect path). Observed
    # flaky WITHOUT this: sqlite3.OperationalError: database is locked at
    # `PRAGMA journal_mode=WAL` -- two connections both doing FIRST-EVER WAL
    # setup on a brand-new file is the race; busy_timeout=5000 does not cover
    # it because it is set on the line AFTER journal_mode in open_state_db,
    # so it isn't armed yet on either connection at the moment they collide.
    # Pre-warming here means every later open_state_db call in a test hits an
    # ALREADY-WAL file -- a no-op re-assert of journal_mode, not a race.
    open_state_db(boot.storage.state_db_path).close()
    return boot, runtime


def _seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0,
                    auto_place=False):
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 4096,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mk(boot, runtime, *, spawn_calls=None):
    spawn_calls = spawn_calls if spawn_calls is not None else []
    pid = [80000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append(model_tag)
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )
    return mgr


def _resident(tag, *, state=ResidentState.RESERVED_LOADING, main_gpu=0,
              split_mode="none", reserved_need_mib=0, placement_overridden=False):
    return Resident(
        # A first-instance resident is keyed by its model tag, as the manager does.
        resident_key=tag,
        model_tag=tag, state=state, main_gpu=main_gpu, split_mode=split_mode,
        reserved_need_mib=reserved_need_mib, port=59500,
        placement_overridden=placement_overridden,
    )


def _match(rule_index=0, rank=1, raw_address="10.0.0.1"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=raw_address, label="",
        effective_tag="main", rank=rank,
    )


def _slot(model_tag, *, thread_id="t1", fastlane=None):
    slot = Slot(
        slot_id=f"slot-{model_tag}-{thread_id}", model_tag=model_tag,
        state=SlotState.LOADING, thread_id=thread_id, created_at=time.monotonic(),
        fastlane=fastlane,
    )
    slot.completion_future = asyncio.get_running_loop().create_future()
    return slot


def _vram_refusal(**kw):
    defaults = dict(name="vram", ok=False, detail="refused", blocked_on="vram",
                     required_mib=9000, available_mib=1000)
    defaults.update(kw)
    return GateResult(**defaults)


async def _release_task(hold_s):
    await asyncio.sleep(hold_s)


# === 1 + 2. BARRIER ENTRY AND CLEAR + GREEN CONTROL ==============

class TestBarrierEntryAndClear:
    async def test_1_red_then_cleared_once_the_watched_task_completes(
        self, tmp_path,
    ):
        """Without the new QueueConfig field every test in this file fails at
        fixture construction with
        `pydantic_core.ValidationError: Extra inputs are not permitted
        [type=extra_forbidden]` -- QueueConfig has `extra="forbid"`
        and spawn_reclaim_wait_max_s would not exist.

        With it: call 1 refuses (vram, a live
        release task is on the same card), the barrier enters, waits for
        the task, re-gates on completion, call 2 passes -> spawns."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.12))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
            r = _resident("m1")
            mgr._residents["m1"] = r
            slot = _slot("m1")
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=3)
        await task
        assert handle is not None, (
            "must spawn once the watched release completes and the re-gate clears"
        )
        assert len(calls) == 2, f"initial refusal + exactly one clearing re-gate, got {len(calls)}"
        assert r.spawn_barrier_active is False, "cleared in the finally, never left set"

    async def test_2_green_control_that_could_have_gone_red(self, tmp_path):
        """GREEN CONTROL: identical fixture, `_card_release_tasks` EMPTY.
        Without this control, test 1 above would also pass on a build where
        the barrier ALWAYS waits regardless of evidence -- this proves the
        entry predicate's second half (POSITIVE EVIDENCE) actually gates
        entry: refuses on the FIRST gate run, `all_safety_gates` called
        exactly once, and `_spawn_barrier_watch_locked` -- spied, not
        mocked -- was entered (eligible=True, blocked_on all "vram") and
        itself returned an EMPTY watch ("no-evidence")."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        watch_results = []
        real_watch_locked = mgr._spawn_barrier_watch_locked

        # _spawn_barrier_watch_locked takes a 3rd param (split_card_count,
        # supplied by the caller's own off-loop read) -- this spy's
        # signature follows the helper's signature (a spy must accept every
        # parameter the real helper takes). No assertion below
        # depends on it.
        def spy_watch_locked(msm, mmg, split_card_count):
            result = real_watch_locked(msm, mmg, split_card_count)
            watch_results.append(result)
            return result

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
             patch.object(mgr, "_spawn_barrier_watch_locked", side_effect=spy_watch_locked):
            r = _resident("m1")
            mgr._residents["m1"] = r
            slot = _slot("m1")
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=3)
        assert handle is None, "no live release task -- terminal on the first gate run"
        assert len(calls) == 1, f"no re-gate loop entered, got {len(calls)} gate calls"
        assert len(watch_results) == 1, "the entry predicate's watch-lock step WAS entered"
        watch, cards, pledged = watch_results[0]
        assert watch == set(), "no-evidence: nothing in _card_release_tasks"
        assert cards == [0]
        assert r.spawn_barrier_active is False, "never set -- watch came back empty"
        exc = slot.completion_future.exception()
        assert isinstance(exc, RuntimeError) and not isinstance(exc, VramOverCommitError)
        assert "vram: refused" in str(exc)


# === 3. PERMANENT FAILS FAST, three classifier cases + mutation control =====

class TestPermanentClassesFailFast:
    """Each case: a live release task IS present on the gated card (so only
    the classifier -- blocked_on != "vram" -- can save it from an eligible
    barrier entry). Each also carries the mutation control that
    is needed: force blocked_on="vram" on the SAME GateResult and watch the
    test's own terminal-on-first-call assertion go RED -- proving the
    classifier is doing the work, not the fixture."""

    @pytest.mark.parametrize("blocked_on,name,detail", [
        ("config", "tensor_split_devices", "element count mismatch"),
        ("host", "kv_cache_fit", "needs host RAM (--no-kv-offload)"),
        ("probe", "kv_cache_fit", "requires a VRAM probe"),
    ])
    async def test_classified_non_vram_refuses_immediately(
        self, tmp_path, blocked_on, name, detail,
    ):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(2.0))  # long-lived; must NOT be waited on
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [GateResult(name, False, detail, blocked_on=blocked_on)]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.5,
                )
            assert handle is None, f"blocked_on={blocked_on!r} must be terminal, not eligible"
            assert len(calls) == 1, (
                f"blocked_on={blocked_on!r} must never re-gate, got {len(calls)} calls"
            )
            assert r.spawn_barrier_active is False
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_mixed_vram_and_host_failures_is_not_eligible(self, tmp_path):
        """The entry predicate is `all(g.blocked_on == "vram" for g in
        failed)`, not `any(...)` -- for a single-element `failed` list (every
        other test in this file) the two are indistinguishable, so this is
        the ONE test that actually discriminates them: two gates fail at
        once, one vram (would be eligible alone) and one host (never
        eligible). `all` correctly refuses to wait on partial evidence --
        waiting could never make the host refusal go away regardless of
        what the vram side does."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(2.0))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal(), GateResult("ram", False, "low ram", blocked_on="host")]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.5,
                )
            assert handle is None, "one non-vram failure among several must veto eligibility"
            assert len(calls) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.parametrize("blocked_on", ["config", "host", "probe"])
    async def test_MUTATION_forcing_vram_classification_makes_it_eligible(
        self, tmp_path, blocked_on,
    ):
        """Mutation control: the ONLY change from the test above is
        blocked_on="vram" on the same refusal. If this does NOT go red
        relative to the assertion above (i.e. it does NOT become eligible),
        the classifier isn't the thing doing the discrimination."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.05))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [GateResult("g", False, "d", blocked_on="vram")]
            return [GateResult("g", True, "ok")]

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
            r = _resident("m1")
            mgr._residents["m1"] = r
            slot = _slot("m1")
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=2)
        await task
        assert handle is not None, (
            f"forcing blocked_on='vram' (was {blocked_on!r}) must make this ELIGIBLE "
            "-- a fixed-classification variant would pass the test above but not this one"
        )
        assert len(calls) == 2


# === 4. TERMINATION: bounded wait, shrink-only watch set ====================

class TestTermination:
    async def test_never_completing_task_terminates_within_budget_bound(
        self, tmp_path,
    ):
        budget, tick = 0.3, 0.1
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=budget)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        never_done = asyncio.create_task(_release_task(999))
        mgr._card_release_tasks.setdefault(0, set()).add(never_done)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", tick):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                t0 = time.monotonic()
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=budget + 5 * tick + 2.0,
                )
                elapsed = time.monotonic() - t0
            assert handle is None
            import math
            max_calls = math.ceil(budget / tick) + 1
            assert 1 < len(calls) <= max_calls, (
                f"expected 2..{max_calls} gate calls, got {len(calls)}"
            )
            assert elapsed < budget + tick + 2.0, (
                f"worst case ~= budget + one tick + one gate run, got {elapsed:.2f}s"
            )
            exc = slot.completion_future.exception()
            assert isinstance(exc, RuntimeError) and not isinstance(exc, VramOverCommitError)
            assert r.spawn_barrier_active is False
        finally:
            never_done.cancel()
            with pytest.raises(asyncio.CancelledError):
                await never_done

    async def test_task_registered_during_wait_is_not_chased(self, tmp_path):
        """Watch set is the ENTRY snapshot. A task that starts mid-wait must
        not extend the barrier even though it lands on the same watched
        card. The deterministic discriminator: spy on every
        `asyncio.wait(...)` call `_await_reclaim_barrier` makes and assert
        the late-registered task is NEVER a member of any of them."""
        budget, tick = 0.25, 0.08
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=budget)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        original = asyncio.create_task(_release_task(999))
        mgr._card_release_tasks.setdefault(0, set()).add(original)

        late_box: dict = {}

        async def _register_late():
            await asyncio.sleep(0.02)
            late = asyncio.create_task(_release_task(0.10))
            late_box["late"] = late
            mgr._card_release_tasks.setdefault(0, set()).add(late)
            await late

        late_registrar = asyncio.create_task(_register_late())
        calls = []
        watch_snapshots = []
        real_wait = asyncio.wait

        async def spy_wait(fs, **kw):
            watch_snapshots.append(set(fs))
            return await real_wait(fs, **kw)

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
                 patch("turbohaul.manager.asyncio.wait", spy_wait), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", tick):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                t0 = time.monotonic()
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=budget + 5 * tick + 2.0,
                )
                elapsed = time.monotonic() - t0
            assert handle is None, (
                "the late-registered task completing must NOT clear the barrier -- "
                "only the entry snapshot is awaited"
            )
            assert elapsed >= budget, "must run the full budget, not shortcut on the late task"
            late = late_box["late"]
            assert not any(late in snap for snap in watch_snapshots), (
                "the late-registered task was passed to asyncio.wait() -- watch "
                "was re-populated from a fresh scan instead of only ever "
                f"shrinking the entry snapshot. watch sizes were "
                f"{[len(s) for s in watch_snapshots]}"
            )
        finally:
            original.cancel()
            late_registrar.cancel()
            for t in (original, late_registrar):
                try:
                    await t
                except asyncio.CancelledError:
                    pass


# === 5. WRONG CARD ============================================================

class TestWrongCard:
    async def test_split_none_watches_only_main_gpu_not_other_cards(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        # Live task on card 1 only -- msm='none'/mmg=0 must watch {0}, not this.
        task_wrong_card = asyncio.create_task(_release_task(2.0))
        mgr._card_release_tasks.setdefault(1, set()).add(task_wrong_card)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000, 10_000]):
                r = _resident("m1", main_gpu=0, split_mode="none")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.5,
                )
            assert handle is None, "card 1's task must not satisfy card 0's barrier"
            assert len(calls) == 1, "no else-fallback to 'all cards' -- not entered at all"
        finally:
            task_wrong_card.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task_wrong_card

    async def test_control_task_moved_to_the_gated_card_is_watched(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task_right_card = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(0, set()).add(task_right_card)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000, 10_000]), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
            r = _resident("m1", main_gpu=0, split_mode="none")
            mgr._residents["m1"] = r
            slot = _slot("m1")
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=2)
        await task_right_card
        assert handle is not None, "positive control: same-card task IS watched and clears"

    # === relocated-claimant watch (a)+(b) ===========================================
    # These two tests show the disjunction matters: a run without
    # the disjunction fails, and a run with it passes.
    # A green that could never have been red would prove nothing.
    # They use an identical fixture; the ONLY variable is whether
    # _spawn_gate_placement uses the OLD auto_place-only test (mutant, via
    # the parametrize below) or the real widened disjunction it ships with.
    @pytest.mark.parametrize("use_widened_condition,expect_watched", [
        (True, True),    # real code: r.placement_overridden alone is enough
        (False, False),  # auto_place-only variant misses it
    ])
    async def test_relocated_claimant_watch_depends_on_the_disjunction(
        self, tmp_path, use_widened_condition, expect_watched,
    ):
        """A Fast-Lane-relocated claimant: manifest says main_gpu=0 (its
        NOMINAL card), but r.placement_overridden=True and r.main_gpu=1 (its
        ACTUAL, relocated card) -- manifest.auto_place is False throughout,
        so the auto_place-only test can NEVER see this relocation. A live
        release task sits on card 1 (the RELOCATED card) only.

        With the real widened condition, `_spawn_gate_placement` must return
        (msm, mmg) = (r.split_mode, r.main_gpu) = ("none", 1) -- card 1 is
        watched, the task is found, GREEN.

        With the auto_place-only mutant -- installed here via a
        monkeypatched `_spawn_gate_placement` that reproduces exactly the old
        two-term test -- placement_overridden is ignored, msm/mmg fall back
        to the manifest's own (which has auto_place=False so the override
        never fires at all) -- ("none", 0). Card 0 has no task. RED: the
        barrier is never entered (no evidence found on the wrong card), the
        request is refused, and the real relocated release on card 1 is
        never seen."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0, auto_place=False)
        mgr = _mk(boot, runtime)
        task_relocated_card = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(1, set()).add(task_relocated_card)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        def old_two_term_placement(r, slot):
            """Reproduces the auto_place-only mutant: `if m.auto_place and msm ==
            'none':` only -- no `or r.placement_overridden` disjunction."""
            from turbohaul.manifest import read_manifest as _rm
            msm, mmg = "layer", 0
            try:
                m = _rm(mgr.boot.storage.manifests_path, slot.model_tag)
                msm = str(m.llama_server_flags.get("split_mode", "layer") or "layer")
                mmg = int(m.llama_server_flags.get("main_gpu", 0) or 0)
                if m.auto_place and msm == "none":
                    msm = r.split_mode
                    mmg = r.main_gpu
            except FileNotFoundError:
                pass
            return msm, mmg

        try:
            with ExitStack() as stack:
                stack.enter_context(
                    patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates)
                )
                stack.enter_context(
                    patch("turbohaul.manager._read_free_vram_all_mib",
                          return_value=[10_000, 10_000])
                )
                stack.enter_context(
                    patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05)
                )
                if not use_widened_condition:
                    stack.enter_context(
                        patch.object(mgr, "_spawn_gate_placement",
                                     side_effect=old_two_term_placement)
                    )
                r = _resident(
                    "m1", main_gpu=1, split_mode="none", placement_overridden=True,
                )
                mgr._residents["m1"] = r
                slot = _slot("m1")
                if expect_watched:
                    handle = await asyncio.wait_for(
                        mgr._spawn_for_resident(r, slot), timeout=2,
                    )
                    await task_relocated_card
                    assert handle is not None, (
                        "widened disjunction must watch the relocated card and clear"
                    )
                    assert len(calls) == 2
                else:
                    handle = await asyncio.wait_for(
                        mgr._spawn_for_resident(r, slot), timeout=1.5,
                    )
                    assert handle is None, (
                        "old auto_place-only test must miss the relocation entirely "
                        "-- watching the manifest's stale card 0, not the actual "
                        "relocated card 1"
                    )
                    assert len(calls) == 1, (
                        "no evidence found on the wrong card -- never entered the wait"
                    )
        finally:
            if not task_relocated_card.done():
                task_relocated_card.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task_relocated_card


# === 6. DISCONNECT ============================================================

class TestDisconnect:
    async def test_disconnect_mid_barrier_evicts_within_one_tick(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(2.0))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        disconnect_event = asyncio.Event()

        async def _disconnect_soon():
            await asyncio.sleep(0.06)
            disconnect_event.set()

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                slot.disconnect_event = disconnect_event
                asyncio.create_task(_disconnect_soon())
                t0 = time.monotonic()
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.5,
                )
                elapsed = time.monotonic() - t0
            assert handle is None
            assert elapsed < 0.3, f"must evict within ~one tick of disconnect, got {elapsed:.2f}s"
            exc = slot.completion_future.exception()
            assert isinstance(exc, SlotEvictedError), f"expected SlotEvictedError, got {exc!r}"
            assert r.spawn_barrier_active is False
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# === RIDER PROTECTION -- real Fast Lane claim registry ======================

class TestRiderProtectionLiteral:
    """Rider protection against the real Fast Lane claim registry. Fast Lane on; a barriered resident's same-tag rider slot B
    (Fast-Lane-listed) is routed via the REAL `_defer_unroutable` funnel (not
    mocked) so it registers a REAL claim K. Assert: B is never inboxed, K
    survives with an UNCHANGED registered_at_monotonic across a repeat
    barriered pop, and `_release_fastlane_claim_locked` is never called for
    B while the resident stays barriered."""

    async def test_barriered_resident_rider_claim_survives_untouched(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, fastlane_enabled=True)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident("m1", state=ResidentState.RESERVED_LOADING)
        r.spawn_barrier_active = True
        r.inbox = asyncio.Queue()
        mgr._residents["m1"] = r
        slot_b = _slot("m1", thread_id="rider", fastlane=_match(rule_index=0, rank=1))

        released_for_b = []
        real_release = mgr._release_fastlane_claim_locked

        def spy_release(slot, reason):
            if slot is slot_b:
                released_for_b.append(reason)
            return real_release(slot, reason)

        with patch.object(mgr, "_release_fastlane_claim_locked", side_effect=spy_release):
            await mgr._route_or_reserve(slot_b)
            key = mgr._fastlane_claim_key(slot_b)
            assert key is not None, "a Fast-Lane-listed slot must register a claim key"
            assert key in mgr._fastlane_claims, (
                "the rider's defer through _defer_unroutable must register a real claim"
            )
            first_ts = mgr._fastlane_claims[key]["registered_at_monotonic"]
            assert r.inbox.empty(), "a barriered resident must never receive an inbox arrival"

            # Second pop while still barriered: same claim, no reset, no release.
            for t in list(mgr._bg_tasks):
                if not t.done():
                    t.cancel()
            await mgr._route_or_reserve(slot_b)
            assert key in mgr._fastlane_claims, "the claim must still be present"
            assert mgr._fastlane_claims[key]["registered_at_monotonic"] == first_ts, (
                "a repeat defer while barriered must not reset registered_at_monotonic"
            )
            assert r.inbox.empty()
            assert released_for_b == [], (
                f"_release_fastlane_claim_locked must never fire for the rider while "
                f"barriered, got {released_for_b}"
            )

        for t in list(mgr._bg_tasks):
            t.cancel()
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)

    async def test_control_unbarriered_resident_still_admits_and_keeps_the_claim_parked_on_the_resident(self, tmp_path):
        """Control: the SAME fixture with spawn_barrier_active=False takes
        the ordinary HIT path (inboxed, claim kept and marked as parked on
        that resident), proving the guard above is conditional, not a
        blanket block. Release at the slot's own turn start is proven in
        the parked-claim same-resident test (this fixture has no turn
        loop)."""
        boot, runtime = _boot_runtime(tmp_path, fastlane_enabled=True)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident("m1", state=ResidentState.ACTIVE)
        r.inbox = asyncio.Queue()
        mgr._residents["m1"] = r
        slot_b = _slot("m1", thread_id="rider", fastlane=_match(rule_index=0, rank=1))
        await mgr._route_or_reserve(slot_b)
        assert r.inbox.qsize() == 1
        assert r.inbox.get_nowait() is slot_b
        key = mgr._fastlane_claim_key(slot_b)
        # a parked request keeps its claim until its own turn starts
        assert key in mgr._fastlane_claims, (
            "an ordinary HIT hand-off must keep the request's claim registered "
            "until its own turn starts"
        )
        claim = mgr._fastlane_claims[key]
        assert claim["slot"] is slot_b
        assert claim.get("parked_on") == r.resident_key, (
            "the kept claim must be marked as parked on the resident whose "
            "inbox holds the request"
        )

    async def test_barriered_resident_defers_via_legacy_single_instance_hit(
        self, tmp_path,
    ):
        """The legacy single-instance HIT guard is a SEPARATE site from the
        _route_to closure above -- `_resident()` must set resident_key to the
        tag (instance_idx==0 convention) to route through it rather than the
        multi-instance branch."""
        boot, runtime = _boot_runtime(tmp_path, fastlane_enabled=True)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident("m1", state=ResidentState.RESERVED_LOADING)
        r.resident_key = "m1"  # instance_idx==0 convention
        r.spawn_barrier_active = True
        r.inbox = asyncio.Queue()
        mgr._residents["m1"] = r
        slot_b = _slot("m1", thread_id="rider-legacy", fastlane=_match(rule_index=0, rank=1))

        await mgr._route_or_reserve(slot_b)
        key = mgr._fastlane_claim_key(slot_b)
        assert key in mgr._fastlane_claims
        assert r.inbox.empty(), (
            "a barriered resident must never receive an inbox arrival via "
            "the legacy single-instance HIT path either"
        )
        for t in list(mgr._bg_tasks):
            t.cancel()
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)


# === 8. ANCHOR CLAIM NEUTRALITY ==============================================

class TestAnchorClaimNeutrality:
    """Test 8: the barrier's own ANCHOR slot (the one actually
    attempting to spawn, going through `_spawn_for_resident` directly, never
    through `_route_or_reserve`/`_defer_unroutable`) must never touch the
    Fast Lane claim registry -- registering or releasing a claim for the
    anchor would be a NEW side effect this change does not intend. The anchor
    slot deliberately CARRIES a Fast Lane match, so it COULD be claimed if
    the barrier code mistakenly reached into the registry."""

    async def test_anchor_never_registers_or_releases_a_claim(self, tmp_path):
        boot, runtime = _boot_runtime(
            tmp_path, spawn_reclaim_wait_max_s=0.2, fastlane_enabled=True,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        never_done = asyncio.create_task(_release_task(999))
        mgr._card_release_tasks.setdefault(0, set()).add(never_done)

        def fake_gates(**kw):
            return [_vram_refusal()]

        register_calls = []
        release_calls = []
        real_register = mgr._register_fastlane_claim_locked
        real_release = mgr._release_fastlane_claim_locked

        def spy_register(slot, reason):
            register_calls.append(slot)
            return real_register(slot, reason)

        def spy_release(slot, reason):
            release_calls.append(slot)
            return real_release(slot, reason)

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05), \
                 patch.object(mgr, "_register_fastlane_claim_locked", side_effect=spy_register), \
                 patch.object(mgr, "_release_fastlane_claim_locked", side_effect=spy_release):
                r = _resident("m1")
                mgr._residents["m1"] = r
                anchor = _slot("m1", thread_id="anchor", fastlane=_match(rule_index=0, rank=1))
                assert mgr._fastlane_claims == {}, "sanity: empty before the barrier runs"
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, anchor), timeout=2.0,
                )
            assert handle is None, "deadline reached, no clearing re-gate -- terminal refusal"
            assert mgr._fastlane_claims == {}, (
                "the barrier must never leave a claim registered for its own anchor"
            )
            assert not any(s is anchor for s in register_calls), (
                f"_register_fastlane_claim_locked must never be called for the anchor, "
                f"called for {register_calls}"
            )
            assert not any(s is anchor for s in release_calls), (
                f"_release_fastlane_claim_locked must never be called for the anchor, "
                f"called for {release_calls}"
            )
        finally:
            never_done.cancel()
            with pytest.raises(asyncio.CancelledError):
                await never_done


# === 9. FAST LANE ON/OFF PARITY ==============================================

class TestFastLaneOnOffParity:
    """Test 9's INTENT (barrier behaviour byte-identical with Fast
    Lane enabled vs disabled). The `_fastlane_claims == {}`
    assertion is checked -- the registry exists in this
    tree (it does not touch the anchor either way, see TestAnchorClaimNeutrality)."""

    @pytest.mark.parametrize("fastlane_enabled", [False, True])
    async def test_barrier_outcome_identical_regardless_of_fastlane(
        self, tmp_path, fastlane_enabled,
    ):
        boot, runtime = _boot_runtime(
            tmp_path, spawn_reclaim_wait_max_s=3.0, fastlane_enabled=fastlane_enabled,
        )
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
            r = _resident("m1")
            mgr._residents["m1"] = r
            slot = _slot("m1")  # no fastlane match -- nothing to claim either way
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=2)
        await task
        assert handle is not None
        assert len(calls) == 2
        assert mgr._fastlane_claims == {}, (
            "the barrier's own spawn path never touches the claim registry -- "
            "true regardless of fastlane_enabled"
        )


# === 10. NO-RETYPE REGRESSION GUARD ==========================================

class TestNoRetypeRegressionGuard:
    async def test_terminal_error_is_plain_runtimeerror_not_vramovercommit(
        self, tmp_path,
    ):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)

        def fake_gates(**kw):
            return [GateResult("ram", False, "only 10 MiB free", blocked_on="host")]

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates):
            r = _resident("m1")
            mgr._residents["m1"] = r
            slot = _slot("m1")
            handle = await asyncio.wait_for(mgr._spawn_for_resident(r, slot), timeout=1.5)
        assert handle is None
        exc = slot.completion_future.exception()
        assert type(exc) is RuntimeError, f"expected plain RuntimeError, got {type(exc)}"
        assert not isinstance(exc, VramOverCommitError), (
            "retyping to VramOverCommitError (503+Retry-After) was considered and "
            "rejected -- a permanent refusal handed a retry instruction "
            "turns a clean 500 into an unbounded client loop"
        )


# === 11. FAILS CLOSED ========================================================

class TestFailsClosed:
    async def test_unclassified_gate_result_is_not_eligible(self, tmp_path):
        """GateResult constructed POSITIONALLY (name, ok, detail) -- no
        blocked_on kwarg at all, exactly how the passing constructions and
        any future gate that forgets to tag itself will look. A live
        release task IS present, so only the fail-closed default (None !=
        'vram') can save it."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(2.0))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [GateResult("some_future_gate", False, "refused")]  # no blocked_on

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.5,
                )
            assert handle is None, "an unclassified refusal must fail CLOSED, never treated as vram"
            assert len(calls) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# === Kill switch (spawn_reclaim_wait_max_s=0.0) ==============================

class TestKillSwitch:
    async def test_zero_budget_is_byte_identical_to_terminal_without_reclaim_wait(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=0.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.05))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            return [_vram_refusal()]

        watch_calls = []
        real_watch_locked = mgr._spawn_barrier_watch_locked

        # Signature follow-through, see TestBarrier-
        # EntryAndClear.test_2's identical note. Never actually invoked here
        # (budget=0.0 skips the eligible branch) -- kept consistent anyway.
        def spy_watch_locked(msm, mmg, split_card_count):
            watch_calls.append((msm, mmg, split_card_count))
            return real_watch_locked(msm, mmg, split_card_count)

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
                 patch.object(mgr, "_spawn_barrier_watch_locked", side_effect=spy_watch_locked):
                r = _resident("m1")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=1.0,
                )
            assert handle is None
            assert len(calls) == 1, "budget=0.0 must never even check for evidence"
            assert r.spawn_barrier_active is False
            assert watch_calls == [], (
                "budget=0.0 must never call _spawn_barrier_watch_locked at all -- "
                f"got {watch_calls}"
            )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# === nvidia-smi off the event loop ===========================================

class TestVramReadOffEventLoop:
    """_spawn_barrier_watch_locked must not call _read_free_vram_all_mib()
    (a synchronous nvidia-smi, safety.py, 5s timeout) directly under
    _registry_lock, on the event loop -- a known class of bug
    (every OTHER loop-side nvidia-smi call in manager.py is wrapped
    in asyncio.to_thread). The read happens off-loop, before the lock, only
    on the split (non-'none') arm, and threading the resulting card COUNT
    into _spawn_barrier_watch_locked, which does zero I/O.

    Two assertions, both required (non-vacuity):
    (a) when the reader runs, it must NOT run on the event loop's own
        thread; (b) the reader must ACTUALLY be called during a real
        split-mode barrier entry -- (a) alone would pass vacuously if the
        reader were simply never invoked at all.
    """
    async def test_split_mode_entry_reads_vram_off_the_event_loop(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="layer", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        call_threads = []

        def fake_read_vram():
            call_threads.append(threading.get_ident())
            return [10_000, 10_000]

        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        loop_thread_id = threading.get_ident()
        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib",
                       side_effect=fake_read_vram), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
                r = _resident("m1", main_gpu=0, split_mode="layer")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=2,
                )
            await task
            assert handle is not None, (
                "positive control: card 0's release task must still be watched "
                "and clear the barrier -- the off-loop move must not change "
                "which card is watched, only HOW the count is read"
            )
            # (b) non-vacuity: the reader must actually have run.
            assert call_threads, (
                "split-mode entry never called _read_free_vram_all_mib at all -- "
                "assertion (a) below would be vacuous without this"
            )
            # (a) the actual MUST: never on the event loop's own thread.
            assert all(tid != loop_thread_id for tid in call_threads), (
                f"_read_free_vram_all_mib ran ON the event loop thread "
                f"(loop={loop_thread_id}, calls={call_threads}) -- it must run "
                f"via asyncio.to_thread, off-loop, before the lock is taken"
            )
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    async def test_control_single_gpu_pin_never_reads_vram_at_all(self, tmp_path):
        """Negative control for the above: split_mode='none' (single-GPU
        pin) must NEVER call _read_free_vram_all_mib -- proves the `!=
        'none'` guard actually gates the off-loop read, not just always-on."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        task = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(0, set()).add(task)
        call_threads = []

        def fake_read_vram():
            call_threads.append(threading.get_ident())
            return [10_000]

        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._read_free_vram_all_mib",
                       side_effect=fake_read_vram), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
                r = _resident("m1", main_gpu=0, split_mode="none")
                mgr._residents["m1"] = r
                slot = _slot("m1")
                handle = await asyncio.wait_for(
                    mgr._spawn_for_resident(r, slot), timeout=2,
                )
            await task
            assert handle is not None
            assert call_threads == [], (
                f"single-GPU pin (split_mode='none') must never read vram for "
                f"the barrier watch -- got {call_threads}"
            )
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


# === rider TOCTOU window =====================================================

class TestRiderToctouWindowClosed:
    """spawn_barrier_active only flips True under _registry_lock, AFTER the
    gate's own await points (_run_spawn_safety_gate, which includes the
    off-loop vram read). A same-tag rider whose _route_or_reserve
    interleaves with that evaluation sees spawn_barrier_active still False
    and lands in r.inbox normally -- exactly that window.

    Non-vacuity: this test constructs the race
    DETERMINISTICALLY (a real time.sleep inside the first fake_gates call,
    which runs on asyncio.to_thread's worker thread and so genuinely yields
    the event loop) and asserts, after the whole spawn attempt completes,
    that the rider is NOT left in r.inbox and WAS routed through
    _defer_unroutable -- proving both that the window is real and that the
    drain closes it, not just that some assertion never got exercised.
    """

    async def test_rider_arriving_during_gate_evaluation_is_deferred_not_inboxed(
        self, tmp_path,
    ):
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident(
            "m1", state=ResidentState.RESERVED_LOADING, main_gpu=0, split_mode="none",
        )
        r.inbox = asyncio.Queue()
        mgr._residents["m1"] = r
        release_task = asyncio.create_task(_release_task(1.0))
        mgr._card_release_tasks.setdefault(0, set()).add(release_task)

        gate_calls = []

        def fake_gates(**kw):
            gate_calls.append(kw)
            if len(gate_calls) == 1:
                # Runs on the to_thread worker thread -- a real sleep here
                # genuinely yields the event loop for long enough for a
                # concurrently-scheduled rider to run _route_or_reserve to
                # completion, landing (without the drain) in r.inbox.
                time.sleep(0.15)
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        defer_calls = []
        real_defer = mgr._defer_unroutable

        def spy_defer(slot, **kw):
            defer_calls.append(slot)
            return real_defer(slot, **kw)

        anchor_slot = _slot("m1", thread_id="anchor")
        rider_slot = _slot("m1", thread_id="rider")

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05), \
                 patch.object(mgr, "_defer_unroutable", side_effect=spy_defer):
                spawn_task = asyncio.create_task(
                    mgr._spawn_for_resident(r, anchor_slot)
                )
                # Let the gate call actually enter its to_thread sleep before
                # the rider arrives -- this is the window itself, not padding.
                await asyncio.sleep(0.03)
                await mgr._route_or_reserve(rider_slot)
                handle = await asyncio.wait_for(spawn_task, timeout=3)

            for t in list(mgr._bg_tasks):
                if not t.done():
                    t.cancel()
            await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)

            assert handle is not None, (
                "anchor must still spawn once the release task clears the "
                "re-gate -- this test is about the RIDER, not the anchor"
            )
            inbox_contents = []
            while not r.inbox.empty():
                inbox_contents.append(r.inbox.get_nowait())
            assert rider_slot not in inbox_contents, (
                "TOCTOU: the rider landed in r.inbox and was never drained -- "
                "it would sit there for the whole barrier hold with no "
                "disconnect interception (this is the REAL window hit)"
            )
            assert rider_slot in defer_calls, (
                "the rider must be routed through _defer_unroutable once the "
                "window closes, not silently dropped from the inbox"
            )
        finally:
            if not release_task.done():
                release_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await release_task

    async def test_control_pre_existing_rider_is_left_alone(self, tmp_path):
        """Negative control: a rider ALREADY in r.inbox before the gate ever
        starts (ordinary Layer-2 queuing, unrelated to the barrier) must NOT
        be swept up by the drain -- only the TAIL beyond the pre-gate
        baseline is ever touched."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident(
            "m1", state=ResidentState.RESERVED_LOADING, main_gpu=0, split_mode="none",
        )
        r.inbox = asyncio.Queue()
        pre_existing = _slot("m1", thread_id="pre-existing")
        r.inbox.put_nowait(pre_existing)
        mgr._residents["m1"] = r
        release_task = asyncio.create_task(_release_task(0.08))
        mgr._card_release_tasks.setdefault(0, set()).add(release_task)

        calls = []

        def fake_gates(**kw):
            calls.append(kw)
            if len(calls) == 1:
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        defer_calls = []
        real_defer = mgr._defer_unroutable

        def spy_defer(slot, **kw):
            defer_calls.append(slot)
            return real_defer(slot, **kw)

        anchor_slot = _slot("m1", thread_id="anchor")
        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05), \
             patch.object(mgr, "_defer_unroutable", side_effect=spy_defer):
            handle = await asyncio.wait_for(
                mgr._spawn_for_resident(r, anchor_slot), timeout=2,
            )
        assert handle is not None
        remaining = []
        while not r.inbox.empty():
            remaining.append(r.inbox.get_nowait())
        assert remaining == [pre_existing], (
            f"the pre-existing (non-TOCTOU) rider must be put back untouched, "
            f"in place -- got {remaining}"
        )
        assert defer_calls == [], (
            f"nothing arrived during the window this time -- the drain must "
            f"not fire on pre-existing occupants, got {defer_calls}"
        )

    async def test_mixed_baseline_and_racing_rider_preserves_order_defers_only_rider(
        self, tmp_path,
    ):
        """Shortened kept slice: the two tests
        above each exercise only ONE dimension -- zero-baseline+racing-rider,
        or nonzero-baseline+no-racing-rider. Neither can catch it (shortening
        `drained[:pre_gate_inbox_n]` by one), which only drops a baseline
        slot when BOTH >=1 baseline slot AND a racing arrival are present in
        the SAME window. Two baseline slots (not one) aren't needed to
        CATCH it -- even a single-slot count goes 1 -> 0 under it and a
        plain count check would already fail. They exist to satisfy the
        intended assertion (a): proving the survivors keep their
        ORIGINAL ORDER, a property one slot can't express at all. That mutant
        actually drops baseline_2 (the slot adjacent to the boundary,
        not the front) while leaving baseline_1 -- and a bare
        `len(inbox_contents) == 2` count check already catches that on
        its own; the identity/order asserts below add rigor about WHICH
        slot survives and in what sequence, not sensitivity the count
        check lacks."""
        boot, runtime = _boot_runtime(tmp_path, spawn_reclaim_wait_max_s=3.0)
        _seed_manifest(boot, "m1", split_mode="none", main_gpu=0)
        mgr = _mk(boot, runtime)
        r = _resident(
            "m1", state=ResidentState.RESERVED_LOADING, main_gpu=0, split_mode="none",
        )
        r.inbox = asyncio.Queue()
        baseline_1 = _slot("m1", thread_id="baseline-1")
        baseline_2 = _slot("m1", thread_id="baseline-2")
        r.inbox.put_nowait(baseline_1)
        r.inbox.put_nowait(baseline_2)
        mgr._residents["m1"] = r
        release_task = asyncio.create_task(_release_task(1.0))
        mgr._card_release_tasks.setdefault(0, set()).add(release_task)

        gate_calls = []

        def fake_gates(**kw):
            gate_calls.append(kw)
            if len(gate_calls) == 1:
                # Same real-sleep-on-the-to_thread-worker-thread window as
                # the single-rider test above -- the mixed case is this same
                # window, just with baseline occupants already queued.
                time.sleep(0.15)
                return [_vram_refusal()]
            return [GateResult("vram", True, "ok")]

        defer_calls = []
        real_defer = mgr._defer_unroutable

        def spy_defer(slot, **kw):
            defer_calls.append(slot)
            return real_defer(slot, **kw)

        anchor_slot = _slot("m1", thread_id="anchor")
        rider_slot = _slot("m1", thread_id="rider")

        try:
            with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
                 patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05), \
                 patch.object(mgr, "_defer_unroutable", side_effect=spy_defer):
                spawn_task = asyncio.create_task(
                    mgr._spawn_for_resident(r, anchor_slot)
                )
                await asyncio.sleep(0.03)
                await mgr._route_or_reserve(rider_slot)
                handle = await asyncio.wait_for(spawn_task, timeout=3)

            for t in list(mgr._bg_tasks):
                if not t.done():
                    t.cancel()
            await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)

            assert handle is not None

            inbox_contents = []
            while not r.inbox.empty():
                inbox_contents.append(r.inbox.get_nowait())

            # (a) IDENTITY in order (tightened check) --
            # `is`, not `==`: Slot is a plain @dataclass with field-based
            # equality, so an equality check alone could not distinguish
            # "the actual queued baseline object, preserved" from "a new
            # Slot reconstructed with the same field values." `is` pins
            # that the drain returns the SAME objects, in the SAME order.
            assert len(inbox_contents) == 2, (
                f"both baseline slots must survive -- got {inbox_contents} "
                f"(a shortened kept-slice mutant silently DROPS baseline_2 "
                f"here -- never requeued, never deferred, just lost)"
            )
            assert inbox_contents[0] is baseline_1, (
                f"baseline_1 (front of the FIFO) must be first -- got "
                f"{inbox_contents[0]!r}"
            )
            assert inbox_contents[1] is baseline_2, (
                f"baseline_2 (second in the FIFO) must be second, in "
                f"original order -- got {inbox_contents[1]!r}"
            )

            # (b) the deferred one IS the rider, identity-checked, and NO
            # baseline slot is in the defer log (tightened check).
            assert len(defer_calls) == 1 and defer_calls[0] is rider_slot, (
                f"exactly the rider, by identity, must be deferred -- got "
                f"{defer_calls}"
            )
            assert not any(d is baseline_1 or d is baseline_2 for d in defer_calls), (
                f"no baseline slot may ever reach _defer_unroutable -- got "
                f"{defer_calls}"
            )

            # (c) accounting closes: nothing lost, nothing duplicated across
            # the 2 baseline + 1 rider = 3 slots queued this test.
            assert len(inbox_contents) + len(defer_calls) == 3, (
                f"total accounted-for slots must equal everything queued "
                f"(2 baseline + 1 rider = 3) -- got "
                f"{len(inbox_contents)} in inbox + {len(defer_calls)} deferred"
            )
        finally:
            if not release_task.done():
                release_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await release_task


# === spawn_reclaim_wait_max_s default and bounds ============================

class TestSpawnReclaimWaitMaxSBounds:
    """QueueConfig's own default and Field bounds for spawn_reclaim_wait_max_s
    are checked directly: every barrier test constructs QueueConfig with an
    EXPLICIT spawn_reclaim_wait_max_s=, so a changed default (for example
    10.0 -> 0.0) would be invisible to all of them."""

    def test_default_is_ten_seconds(self):
        assert QueueConfig().spawn_reclaim_wait_max_s == 10.0

    def test_bounds_reject_outside_ge_0_le_60(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueueConfig(spawn_reclaim_wait_max_s=-0.1)
        with pytest.raises(ValidationError):
            QueueConfig(spawn_reclaim_wait_max_s=60.1)

    def test_bounds_accept_the_edges(self):
        assert QueueConfig(
            spawn_reclaim_wait_max_s=0.0
        ).spawn_reclaim_wait_max_s == 0.0
        assert QueueConfig(
            spawn_reclaim_wait_max_s=60.0
        ).spawn_reclaim_wait_max_s == 60.0
