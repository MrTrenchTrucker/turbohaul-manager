"""OOM load failure goes back in the queue (resident-driver path) —
_spawn_for_resident's failure branch ("if not healthy:"), the ONE
live spawn path: worker_loop
unconditionally delegates to _dispatch_loop at every max_parallel_sidecars
value (config.py pins it ge=1), so manager.py's OTHER "if not healthy:"
branch (_process_slot) is unreachable dead code (a stack-trace
instrument shows it is never reached). See
the single-slot OOM-requeue test's own module docstring
for the full citation and for why THAT file, despite its name and its
max_parallel_sidecars=1 config, exercises this exact same path too --
there is only this one path in this test suite;
it is not a second, independent code path to prove.
Nothing in _process_slot is exercised or modified here: the change under
test touches only the live resident-driver path. The unreachable branch
in _process_slot cannot be driven by any test and is not covered by this
file; it is dead code, not a second path to prove (see the paragraph
above).

Harness mirrors the spec-downgrade test's
TestResidentDriverPathPicksUpEngineDowngrade (the existing precedent that
drives THIS path via max_parallel_sidecars=2 + a split_mode:none manifest,
through the constructor-DI style TurbohaulManager(boot, runtime, spawn_fn=...,
health_fn=..., ...)) -- the same shape the resident-driver
path's errors-scan needs.
"""
import asyncio
from unittest.mock import MagicMock

import pytest
import yaml

from turbohaul import load_verify_log
from turbohaul import manager as manager_module
from turbohaul import safety
from turbohaul.config import (
    BootConfig, FastLaneConfig, FastLaneRule, FastLaneTagRanks, PullConfig,
    QueueConfig, RuntimeConfig, RuntimePathsConfig, ServerConfig,
    StorageConfig, UIConfig,
)
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import SlotEvictedError
from turbohaul.subprocess_mgr import SidecarHandle

pytestmark = pytest.mark.asyncio

_OOM_SCAN = {
    "engine_errors_detected": True,
    "engine_error_lines": [
        "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 884.62 MiB "
        "on device 0: cudaMalloc failed: out of memory"
    ],
    "reason": None,
}
_OTHER_SCAN = {
    "engine_errors_detected": True,
    "engine_error_lines": ["E load_model: failed to open GGUF file: No such file or directory"],
    "reason": None,
}

# A representative OOM engine-log line (exact text of a real failure,
# same text baked into _OOM_SCAN above), but here it is
# written to a REAL temp file and read by the REAL scan_engine_log_for_errors
# + classify_load_failure (neither is stubbed).
_INCIDENT_LINE = (
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 884.62 MiB "
    "on device 0: cudaMalloc failed: out of memory"
)


def _boot_runtime(tmp_path, *, fastlane_rules=None, **queue_kwargs):
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
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_min_free_vram_mib=1000, safety_enabled=False, **queue_kwargs),
        pull=PullConfig(),
        # A real, matched FastLaneMatch is what
        # makes claim registration a non-no-op (_fastlane_claim_key returns
        # None otherwise) -- fastlane=None slots silently skip claim
        # registration entirely (the FIXTURE TRAP documented in
        # the readmission-rank test).
        fastlane=FastLaneConfig(
            enabled=bool(fastlane_rules), rules=list(fastlane_rules or []),
        ),
    )
    return boot, runtime


_SHUTDOWN_TIMEOUT_S = 1.5


async def _bounded_shutdown(mgr, *, timeout=_SHUTDOWN_TIMEOUT_S):
    """``mgr.shutdown()`` drains every
    ``_spawn_bg``'d background task (``_drain_bg_tasks``), INCLUDING an
    ``_oom_requeue_wait_then_readmit`` companion -- that drain is a bare
    ``await``, not itself timeout-bounded. If the companion's loop never observes
    its own exit condition (a mutant that ignores disconnect, widens the
    timeout exemption, or requeues forever on no-disconnect-
    signal), ``shutdown()`` hangs draining it FOREVER
    rather than raising anything -- a hang, not a failure, is exactly what a
    bare ``await mgr.shutdown()`` in a test's ``finally`` would produce
    without this helper (a hang is not a named failure). Wrapping the
    SAME call in ``asyncio.wait_for`` turns that hang into ``asyncio.
    TimeoutError`` -- a named, bounded, reportable test failure -- with zero
    change to what shutdown() itself does or asserts."""
    await asyncio.wait_for(mgr.shutdown(), timeout=timeout)


def _seed_manifest(boot, model_tag, *, expected_vram_bytes=3000 * 1024 * 1024,
                    split_mode="none", main_gpu=0):
    payload = {
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_bytes,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_bytes,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump(payload))


def _make_manager(tmp_path, monkeypatch, *, scan_result, fastlane_rules=None,
                   settle_s=0.0, poll_s=0.01):
    boot, runtime = _boot_runtime(
        tmp_path, max_parallel_sidecars=2, fastlane_rules=fastlane_rules,
    )
    _seed_manifest(boot, "m1")

    call_count = [0]
    pid = [91000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        call_count[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        pid[0] += 1
        proc.poll.return_value = None  # alive
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        call_count[0] += 0  # counted at spawn already; kept for symmetry/clarity
        return False

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )

    monkeypatch.setattr(
        load_verify_log, "scan_engine_log_for_errors", lambda *a, **k: scan_result
    )
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_POLL_S", poll_s)
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_SETTLE_S", settle_s)
    # Hermeticity:
    # safety._read_free_vram_all_mib is a REAL nvidia-smi subprocess call.
    # Unpatched, every reservation-time admission (_vram_admits_locked ->
    # _vram_budget -> this) depends on the HOST's actual GPU state -- on a
    # GPU-void box it raises and degrades to None (the "unreadable"
    # doctrine), so this file's tests pass on such a machine, but a
    # GPU-visible box gets REAL free-MiB numbers instead, changing outcomes
    # for tests whose manifests happen to interact with that value. Patched
    # to a large, always-sufficient constant here so every test's OWN
    # (unrelated) admission phase is deterministic; a test that needs to
    # drive the free-VRAM value ITSELF re-patches this same name
    # afterward with its own scripted sequence.
    monkeypatch.setattr(
        safety, "_read_free_vram_all_mib", lambda: [999_999_999, 999_999_999]
    )

    return mgr, call_count, boot


def _make_manager_real_scan(tmp_path, monkeypatch, *, log_lines,
                             poll_s=0.01, settle_s=0.0):
    """Like ``_make_manager``, but
    ``scan_engine_log_for_errors`` and ``classify_load_failure`` are the REAL
    functions -- neither is stubbed. Each fake spawn writes ``log_lines`` to a
    genuine temp file and threads it onto the handle via
    ``SidecarHandle(engine_log_path=...)``, the exact field the real scan
    function reads (``load_verify_log.py``: base path, then its ``.stdio``
    sibling). This is what proves the REAL scan's output actually reaches
    ``_requeue_on_oom_load_failure`` at the live call site in manager.py,
    in the resident-driver failure branch, not just a
    mocked stand-in for it."""
    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=2)
    _seed_manifest(boot, "m1")

    call_count = [0]
    pid = [92000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        call_count[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        pid[0] += 1
        proc.poll.return_value = None  # alive
        log_path = tmp_path / f"engine-{pid[0]}.log"
        log_path.write_text("\n".join(log_lines) + "\n")
        return SidecarHandle(
            proc=proc, port=port, model_tag=model_tag,
            engine_log_path=str(log_path),
        )

    async def fake_health(*a, **k):
        return False

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )

    # scan_engine_log_for_errors and classify_load_failure are DELIBERATELY
    # left unpatched -- only the retry/settle cadence constants are injected,
    # exactly as in every other test in this file.
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_POLL_S", poll_s)
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_SETTLE_S", settle_s)
    # This helper applies the same hermeticity patch as _make_manager: the
    # failure path takes an unconditional pre-park seed reading, so on a
    # GPU-visible box an unpatched run would reach the REAL
    # safety._read_free_vram_all_mib nvidia-smi subprocess call (see
    # _make_manager's comment above) and
    # depend on the host's actual GPU state.
    monkeypatch.setattr(
        safety, "_read_free_vram_all_mib", lambda: [999_999_999, 999_999_999]
    )

    return mgr, call_count, boot


async def _run_worker(mgr):
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())


async def _pump_until_retried(mgr, calls, start_count, *, timeout=2.0):
    """Poll-based, not sleep-based -- see the single-slot test file's own
    copy of this helper for the ambient-load rationale and for
    why phase 1 (wait for the INITIAL attempt to land before taking the
    baseline) is needed: ``start_count`` is captured before the freshly
    created worker task has run at all, so without phase 1 the helper's
    first notify+sleep observes the initial attempt itself and reports a
    false "retried"."""
    deadline = asyncio.get_running_loop().time() + timeout
    while (
        asyncio.get_running_loop().time() < deadline
        and calls[0] <= start_count
    ):
        await asyncio.sleep(0.01)
    baseline = calls[0]
    while asyncio.get_running_loop().time() < deadline:
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await asyncio.sleep(0.01)
        if calls[0] > baseline:
            return True
    return False


async def _pump_backstock_only_until_retried(mgr, calls, start_count, *, timeout=2.0):
    """Unlike ``_pump_until_retried``, this must NOT
    notify ``_make_room_signal`` (a REAL wake drives the existing
    wake-then-settle path instead of the backstock-timeout path these
    tests add), so it purely waits on the companion's own
    ``_OOM_REQUEUE_POLL_S`` cadence (tiny in tests -- see
    ``_make_manager``'s ``poll_s``). Single-phase (no "wait for the initial
    attempt" pre-step): callers of THIS helper capture ``start_count``
    AFTER the initial OOM-park already happened (``_wait_until_parked``),
    since only the retry attempt matters here and the companion attempts
    at most once before ending -- a two-phase wait would wait forever for
    a SECOND retry that never comes."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if calls[0] > start_count:
            return True
        await asyncio.sleep(0.01)
    return False


async def _wait_until_parked(slot, *, timeout=2.0):
    """Wait for the INITIAL admission attempt to OOM
    and park (``oom_requeue_pending`` flips True) before a test starts
    controlling ``_vram_budget`` -- the initial reservation-time gate
    (``_vram_admits_locked``) ALSO calls ``_vram_budget`` for its own
    unrelated pre-check, so patching it before this point would corrupt
    the very first admission attempt (e.g. a low canned reading could
    defer/refuse it outright, never reaching the OOM-classified spawn
    failure these tests are about at all)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if slot.oom_requeue_pending:
            return True
        await asyncio.sleep(0.005)
    return False


def _make_scripted_free_vram_all_mib(sequences):
    """Hermeticity helper: a scripted,
    call-ordered stand-in for
    ``turbohaul.safety._read_free_vram_all_mib`` -- returns
    ``sequences[i]`` (a per-card MiB list, or ``None`` for an unreadable
    probe) on the i-th call, clamped to the last entry once exhausted.
    Patched at THIS level (not ``_vram_budget`` itself, which this
    replaced) so ``_vram_budget``'s own REAL split_mode/bounds-check math
    still runs against the scripted numbers -- exercising production code,
    not a reimplementation of it, and never touching the real nvidia-smi
    subprocess. ``calls`` lets a test assert the probe was actually reached
    (not a vacuous 0-attempts from some OTHER short-circuit)."""
    calls = {"n": 0}

    def fake():
        idx = min(calls["n"], len(sequences) - 1)
        calls["n"] += 1
        return sequences[idx]

    return fake, calls


async def test_oom_with_no_disconnect_event_fails_once_unchanged(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    try:
        slot = await mgr.submit(model_tag="m1", prompt="x", wait_for_completion=True)
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert slot.oom_requeue_pending is False
    finally:
        await _bounded_shutdown(mgr)


async def test_oom_with_live_disconnect_is_requeued_not_failed(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, "must have made more than one spawn attempt"
        assert not slot.completion_future.done(), (
            "resident-driver path: an OOM failure with a live disconnect "
            "signal must never fail the request while the caller stays "
            "connected"
        )
        assert calls[0] > 1, "must have made more than one spawn attempt"
        assert slot.oom_requeue_pending is True
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_disconnect_while_requeued_stops_the_retry_loop(tmp_path, monkeypatch):
    """Asserts more
    than "no further re-admission calls" -- that alone is also satisfied
    by a LEAKED future, so it would not prove the
    request finished. Also asserts completion_future resolves with
    SlotEvictedError and oom_requeue_pending clears, same as the dedicated
    disconnect-release tests, on THIS scenario (loop-top disconnect, no real wake ever
    happens)."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        await asyncio.sleep(0.03)
        assert calls[0] >= 1
        disconnect_event.set()
        calls_at_disconnect = calls[0]
        await asyncio.sleep(0.15)
        assert calls[0] == calls_at_disconnect, (
            "a disconnected slot must not be re-admitted from the OOM "
            "retry-wait companion (resident-driver path)"
        )
        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=2.0)
        assert slot.oom_requeue_pending is False, (
            "oom_requeue_pending must be cleared once "
            "the future resolves on disconnect-while-parked"
        )
    finally:
        await _bounded_shutdown(mgr)


async def test_non_oom_error_with_disconnect_still_fails_once(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OTHER_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert slot.oom_requeue_pending is False
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_never_fits_known_capacity_fails_once(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(boot, "huge", expected_vram_bytes=999_000 * 1024 * 1024)
    mgr._vram_total_mib = [24000, 24000]
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="huge", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_unknown_capacity_requeues_instead_of_never_fits(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(boot, "huge", expected_vram_bytes=999_000 * 1024 * 1024)
    assert mgr._vram_total_mib is None
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="huge", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried
        assert not slot.completion_future.done()
        assert calls[0] > 1
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# The timeout-exemption tests, duplicated onto the resident-driver
# path. submit_and_wait's timeout_s exemption (_await_completion_with_oom_
# exemption) is a manager-level, spawn-path-INDEPENDENT mechanism -- it reads
# slot.oom_requeue_pending, never which spawn path set it -- but it is
# checked once per spawn path,
# so these mirror the single-slot file's own three timeout-exemption tests exactly, on
# THIS path's _make_manager.
# ---------------------------------------------------------------------------
async def test_requeued_connected_survives_past_injected_timeout(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        await _run_worker(mgr)
        task = asyncio.create_task(
            mgr.submit_and_wait(
                "m1", "x", disconnect_event=disconnect_event, timeout_s=0.05,
            )
        )
        await asyncio.sleep(0.2)  # well past the 0.05s timeout_s
        assert not task.done(), (
            "timeout_s must not end an OOM-requeued+connected wait, even "
            "well past its own value (resident-driver path)"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_ordinary_connected_wait_still_times_out(tmp_path, monkeypatch):
    # The real control for a mutant that widens the timeout exemption to "any
    # connected wait" instead of "oom_requeue_pending AND connected": a
    # connected slot that never became OOM-requeued (no worker running at
    # all) must still time out at timeout_s, resident-driver path included.
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        # See the single-slot file's own copy of
        # this comment -- the "exemption widened" mutant makes this
        # slot's wait genuinely UNBOUNDED (per_wait=None), so
        # pytest.raises(asyncio.TimeoutError) alone cannot tell that apart
        # from this outer 2.0s safety bound also raising TimeoutError; the
        # elapsed-time assertion is what actually distinguishes them.
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mgr.submit_and_wait(
                    "m1", "x", disconnect_event=disconnect_event, timeout_s=0.05,
                ),
                timeout=2.0,
            )
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 1.0, (
            f"must time out via its OWN timeout_s (~0.05s), not this test's "
            f"outer 2.0s safety bound -- {elapsed:.2f}s elapsed means the "
            f"exemption is wrongly suspending an ORDINARY connected wait"
        )
    finally:
        await _bounded_shutdown(mgr)


async def test_ordinary_wait_still_times_out_unchanged(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await mgr.submit_and_wait("m1", "x", timeout_s=0.05)
    finally:
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# The burst-of-frees ordering mutant.
# _OOM_REQUEUE_SETTLE_S is 0 in every OTHER test in this file (tests use ~0
# by design) -- a debounce window of ZERO length is not
# exercised by any of them, so a mutant deleting the
# _settle_after_make_room_wake() call inside _oom_requeue_wait_then_readmit
# is indistinguishable from the passing suite without a test that uses a
# genuinely NONZERO settle window and fires a real burst inside it.
# ---------------------------------------------------------------------------
async def test_burst_of_frees_collapses_to_one_admission_attempt(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN,
        settle_s=0.2, poll_s=5.0,
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        # Phase 1 (see _pump_until_retried's own docstring for why): wait for
        # the INITIAL (non-retry) admission attempt to land before treating
        # calls[0] as a baseline.
        deadline = asyncio.get_running_loop().time() + 2.0
        while calls[0] < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert calls[0] == 1, "sanity: the initial OOM attempt happened"
        baseline = calls[0]

        # A BURST: several near-simultaneous frees, all inside the 0.2s
        # settle window -- the real production shape this debounce exists
        # for (several residents' turns ending together), not one notify
        # repeated by an external poll loop.
        for _ in range(6):
            async with mgr._make_room_signal:
                mgr._make_room_signal.notify_all()
            await asyncio.sleep(0.02)  # 6 * 0.02s = 0.12s, still < 0.2s settle

        # Wait out the settle window fully, with margin, so a second
        # attempt (if the debounce were absent) would have had time to land.
        await asyncio.sleep(0.5)
        assert calls[0] == baseline + 1, (
            f"a burst of 6 near-simultaneous frees inside the settle window "
            f"must collapse to exactly ONE admission attempt, not "
            f"{calls[0] - baseline}"
        )
    finally:
        disconnect_event.set()
        # This test's own poll_s=5.0 (so the burst's timing margin is not
        # racing a fast backstock) means a parked companion would otherwise
        # only notice the disconnect on its own 5s backstock -- a REAL
        # (non-mutant) slow cleanup, not a hang, but one that would trip
        # _bounded_shutdown's much tighter timeout. A real wake, not a
        # longer bound, is the fix: the companion checks disconnect
        # immediately upon waking, same as it always does.
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# The claim-not-released-on-disconnect ordering
# mutant. Claim release on disconnect is existing
# machinery -- _claim_is_live's own disconnect check, consumed by
# fastlane_claims_snapshot's own inline prune.
# This proves that machinery actually reaches a slot that is currently
# OOM-requeued (not just an ordinary waiter, which existing tests already
# cover).
# ---------------------------------------------------------------------------
async def test_claim_released_on_disconnect_while_requeued(tmp_path, monkeypatch):
    rules = [FastLaneRule(address="10.0.0.5", tag_ranks=FastLaneTagRanks(main=1))]
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, fastlane_rules=rules,
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
            client_meta={"ip": "10.0.0.5"},
        )
        assert slot.fastlane is not None, (
            "fixture trap: an unmatched slot makes claim registration a "
            "silent no-op and this whole test vacuous"
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, "sanity: genuinely OOM-requeued, not failed"

        key = mgr._fastlane_claim_key(slot)
        # Verified, not assumed: the staging-arrival claim (_register_
        # staged_claim) is released the moment the slot is
        # ADMITTED for a spawn attempt (the "admitted" release call sites
        # in manager.py) -- success or
        # failure of that attempt is irrelevant to the release, so by the
        # time an OOM failure is even classified the staging claim is
        # ALREADY gone (confirmed: mgr._fastlane_claims is {} here without
        # this re-registration). A claim only exists WHILE waiting, never
        # while an attempt is in flight -- this re-registers one directly,
        # via the exact same primitive an ordinary concurrent MISS/defer on
        # this still-requeued slot would call, so the disconnect-release
        # check below is exercised against a claim that is actually live.
        async with mgr._registry_lock:
            mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        assert key in mgr._fastlane_claims, (
            "the registration primitive itself failed to register a claim "
            "for an eligible, still-connected, still-requeued slot"
        )

        disconnect_event.set()
        # The SAME prune a production /status read performs
        # (in manager.py), exercised directly.
        pruned = mgr.fastlane_claims_snapshot()
        assert not any(c.get("slot") is slot for c in pruned), (
            "a disconnected slot's claim must be classified dead by "
            "fastlane_claims_snapshot's own pruning pass"
        )
        assert key not in mgr._fastlane_claims, (
            "a disconnected requeued slot's claim must actually be REMOVED "
            "from the registry, not merely absent from the snapshot's "
            "return value"
        )
    finally:
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# TTL re-arm: a
# requeue that outlives one TTL re-arms exactly as an ordinary waiter does
# (an injected small TTL is used). No new claim-lifetime logic is needed
# -- this proves the EXISTING claim machinery
# (_register_fastlane_claim_locked's own no-op-if-present guard,
# _claim_is_live's own ttl_deadline_monotonic check, fastlane_claims_
# snapshot's own inline prune) carries a still-OOM-requeued slot across a
# TTL boundary identically to how it carries an ordinary waiter: the old
# claim goes ttl_expired, the next natural prune removes it, and the next
# natural claim-check registers a fresh one -- never zero live claims for a
# still-waiting request, never two.
# ---------------------------------------------------------------------------
async def test_ttl_expiry_while_requeued_rearms_like_an_ordinary_waiter(
    tmp_path, monkeypatch,
):
    tiny_ttl = 0.05
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", tiny_ttl)
    rules = [FastLaneRule(address="10.0.0.5", tag_ranks=FastLaneTagRanks(main=1))]
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, fastlane_rules=rules,
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
            client_meta={"ip": "10.0.0.5"},
        )
        assert slot.fastlane is not None, (
            "fixture trap: an unmatched slot makes claim registration a "
            "silent no-op and this whole test vacuous"
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, "sanity: genuinely OOM-requeued, not failed"
        assert slot.oom_requeue_pending is True

        key = mgr._fastlane_claim_key(slot)
        # Verified, not assumed: the staging-arrival claim is released the
        # instant the slot is ADMITTED for a spawn attempt, success or
        # failure (see the sibling disconnect-release test's own comment for
        # the full citation) -- by the time an OOM failure is classified,
        # mgr._fastlane_claims is already {}. Re-registered directly here,
        # via the exact primitive an ordinary concurrent MISS/defer on this
        # still-requeued slot would call, so the TTL/re-arm mechanism below
        # is exercised against a claim that is actually live.
        async with mgr._registry_lock:
            mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        assert key in mgr._fastlane_claims, (
            "the registration primitive itself failed to register a claim "
            "for an eligible, still-connected, still-requeued slot"
        )
        original_deadline = mgr._fastlane_claims[key]["ttl_deadline_monotonic"]

        # Outlive the (injected, tiny) TTL while STILL requeued+connected --
        # never a real 1800s wait.
        await asyncio.sleep(tiny_ttl * 3)
        assert not slot.completion_future.done(), (
            "outliving the claim TTL must never itself fail an OOM-requeued "
            "+ connected request -- the completion-future timeout rule and the claim "
            "TTL rule are independent, and neither may leak into the other"
        )

        # The SAME prune a production /status read performs
        # (in manager.py), exercised directly here.
        pruned = mgr.fastlane_claims_snapshot()
        assert not any(c.get("slot") is slot for c in pruned), (
            "a ttl_expired claim must be pruned like any other dead claim"
        )
        assert key not in mgr._fastlane_claims, (
            "the expired claim must actually be gone from the registry, "
            "not merely absent from the snapshot's return value"
        )

        # The SAME registration primitive an ordinary still-waiting MISS
        # re-arms through (_defer_unroutable calls this exact method,
        # manager.py) -- called directly, under the same lock it
        # always requires, rather than orchestrating a full dispatch-loop
        # admission cycle to reach it.
        async with mgr._registry_lock:
            mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        assert key in mgr._fastlane_claims, (
            "an ordinary still-waiting slot re-arms its claim on the next "
            "defer -- a requeued one must too (same rules, no new logic)"
        )
        new_deadline = mgr._fastlane_claims[key]["ttl_deadline_monotonic"]
        assert new_deadline > original_deadline, (
            "the re-armed claim must carry a FRESH deadline, not the stale "
            "one -- a refreshed timestamp is what distinguishes a genuine "
            "re-arm from the dead entry simply never having been pruned"
        )
        claims_for_slot = [
            c for c in mgr._fastlane_claims.values() if c.get("slot") is slot
        ]
        assert len(claims_for_slot) == 1, (
            "never two live claims for the same requeued request "
            "(one claim per still-waiting request)"
        )
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# The LIVE
# path is _spawn_for_resident alone (worker_loop unconditionally delegates to
# _dispatch_loop -- config.py pins max_parallel_sidecars >= 1 -- so
# _process_slot's only call site is unreachable, per a comment in manager.py).
# A defect at the live call
# site in manager.py (passing None instead of the real _errs into
# _requeue_on_oom_load_failure) goes undetected by any test that stubs the scan
# (the mocked-scan tests),
# even though the wiring is broken, because a monkeypatched
# scan_engine_log_for_errors returns its canned dict regardless of what the
# real function would have read, so a defect that discards the real scan's
# OUTPUT is invisible to a test that never lets the real scan produce one.
# This test writes a representative OOM engine-log line to a REAL temp
# file and runs the REAL scan_engine_log_for_errors + classify_load_failure
# (neither stubbed) so such a defect has something real to discard.
# ---------------------------------------------------------------------------
async def test_real_engine_log_scan_reaches_the_requeue_decision(tmp_path, monkeypatch):
    mgr, calls, boot = _make_manager_real_scan(
        tmp_path, monkeypatch, log_lines=[_INCIDENT_LINE],
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, (
            "the REAL engine log's own OOM line, read by the REAL scan + "
            "classifier (neither stubbed), must drive a genuine retry -- "
            "not just a mocked scan result standing in for it"
        )
        assert not slot.completion_future.done(), (
            "a real, unstubbed OOM classification must requeue, not fail, "
            "while the caller stays connected"
        )
        assert calls[0] > 1
        assert slot.oom_requeue_pending is True
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# Tests for the disconnect-while-parked release and the row-split
# capacity check. Both run through this file's own harness against
# the real production code paths (no reimplementation of the logic
# under test).
# ---------------------------------------------------------------------------
async def test_disconnect_while_parked_releases_completion_future(
    tmp_path, monkeypatch
):
    """A caller who disconnects while OOM-requeue-parked must have
    its completion_future resolved (SlotEvictedError) and oom_requeue_pending
    cleared, not leaked forever. Both disconnect checks in
    _oom_requeue_wait_then_readmit must resolve the future, not do a bare
    ``return`` -- this asserts the future actually resolves, which
    test_disconnect_while_requeued_stops_the_retry_loop (above) does not check
    (it only asserts no re-admission attempts, which a leaked future also satisfies)."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        # Poll-based, not sleep-based (same rationale as _pump_until_retried
        # above): the first spawn attempt landing and failing/classifying
        # OOM/setting the flag isn't instantaneous, and a fixed short sleep
        # is flaky under load.
        deadline = asyncio.get_running_loop().time() + 2.0
        while (
            not slot.oom_requeue_pending
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.005)
        assert slot.oom_requeue_pending is True, (
            "precondition: the slot must actually be OOM-requeue-parked "
            "before disconnecting, or this test proves nothing"
        )
        disconnect_event.set()
        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=2.0)
        assert slot.oom_requeue_pending is False, (
            "oom_requeue_pending must be cleared once the future "
            "is resolved on disconnect-while-parked"
        )
    finally:
        await _bounded_shutdown(mgr)


async def test_disconnect_during_settle_releases_completion_future(
    tmp_path, monkeypatch
):
    """The SECOND disconnect check (after a real _make_room_signal wake
    + _settle_after_make_room_wake, just before enqueue_head) --
    reverting ONLY this check
    to a bare ``return`` would still pass the rest of the suite, because
    test_disconnect_while_parked_releases_completion_future (above)
    disconnects BEFORE any real wake ever happens, so it only ever drives
    the LOOP-TOP check. This test drives a REAL wake first (a genuinely
    nonzero settle window, per test_burst_of_frees_collapses_to_one_
    admission_attempt's own pattern), then disconnects DURING that settle
    window -- exactly the gap between "woken" and "enqueue_head" the second
    check exists to guard."""
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN,
        settle_s=0.3, poll_s=5.0,
    )
    disconnect_event = asyncio.Event()

    enqueue_calls = [0]
    orig_enqueue_head = mgr.queue.enqueue_head

    async def spying_enqueue_head(*a, **k):
        enqueue_calls[0] += 1
        return await orig_enqueue_head(*a, **k)

    mgr.queue.enqueue_head = spying_enqueue_head

    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        deadline = asyncio.get_running_loop().time() + 2.0
        while calls[0] < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert calls[0] == 1, "sanity: the initial OOM attempt happened"
        assert slot.oom_requeue_pending is True

        # A REAL wake starts the (0.3s) settle window -- the companion is
        # now inside _settle_after_make_room_wake, past the loop-top check,
        # not yet at the second one.
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await asyncio.sleep(0.05)  # inside the settle window, well before it elapses

        # Disconnect arrives DURING the settle window.
        disconnect_event.set()

        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=2.0)
        assert slot.oom_requeue_pending is False, (
            "second check: oom_requeue_pending must be cleared "
            "when disconnect arrives during the settle window"
        )
        assert enqueue_calls[0] == 0, (
            "second check: a disconnect during the settle window "
            "must NOT re-enqueue the slot"
        )
    finally:
        disconnect_event.set()
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await _bounded_shutdown(mgr)


async def test_row_split_never_fits_uses_summed_capacity(
    tmp_path, monkeypatch
):
    """Row/tensor split models span every visible card just like
    layer-split (per _vram_admits_locked's own docstring: "Layer-split
    (layer/row/tensor) models are still refuse-blind for co-residence: a
    layer-split sibling spans every visible GPU"), so
    _model_never_fits_known_capacity must SUM per-card capacity for them
    too, not max() them like a tensor-isolated (split_mode='none') model. A
    row-split model bigger than any single card but within the SUM must
    still requeue (a fit exists), not fail-once."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    # 30000 MiB: bigger than either single 24000 MiB card, well within the
    # 48000 MiB two-card sum a row-split model is actually allowed to use.
    _seed_manifest(
        boot, "rowsplit", expected_vram_bytes=30000 * 1024 * 1024,
        split_mode="row",
    )
    mgr._vram_total_mib = [24000, 24000]
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="rowsplit", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, (
            "a row-split model that fits the SUM of cards must requeue, "
            "not fail-once on a max(card)-only capacity check"
        )
        assert not slot.completion_future.done()
        assert calls[0] > 1
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# Resolved-placement capacity check:
# _model_never_fits_known_capacity must check the RESOLVED placement
# (_spawn_for_resident's own spawn_flags, after any auto-place/Fast-Lane-
# relocation override) -- not the manifest's raw main_gpu/split_mode -- and
# for split_mode=='none' must bound by the SPECIFIC resolved card, not
# max(capacity) globally. Tests 2a/2b/2c/2d below.
# ---------------------------------------------------------------------------
async def test_widened_2a_none_split_uses_specific_card_not_max(
    tmp_path, monkeypatch
):
    """2a: guards the "max(card)-for-none-reverted-wrong" case. manifest
    (and here, unresolved -- no override in play) pins main_gpu=0,
    split_mode='none'. capacity=[8000, 24000]: max(capacity)=24000 would
    wrongly say "fits"; the SPECIFIC card 0 (8000 MiB) correctly says
    never-fits. A model whose need sits strictly between the two cards
    distinguishes the two implementations."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(
        boot, "narrowcard", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none",
    )
    mgr._vram_total_mib = [8000, 24000]
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="narrowcard", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_widened_2b_resolved_placement_not_manifest(
    tmp_path, monkeypatch
):
    """2b: guards the "manifest-placement-instead-of-resolved" case, 'none'
    arm. Manifest pins main_gpu=1 (the LARGE card, a deliberately awkward
    raw value), but the reservation resolves (relocation, stubbed via
    _resolve_placement_locked) to main_gpu=0 -- the SMALL card, and the
    value _spawn_for_resident's spawn_flags actually carries into argv.
    capacity=[8000, 24000]: this is the inverse of 2a's pairing on purpose
    -- if an implementation used the manifest's card (1, 24000, "fits") instead of
    the resolved one (0, 8000, "never-fits"), it would give the SAME wrong
    answer max(capacity) itself gives for that card layout, so a resolved
    card that DOUBLES as the argmax can't distinguish the two; here the
    resolved (correct) card is the SMALLER one, so only a genuinely
    resolved-placement read fails this once instead of wrongly requeuing
    it forever."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(
        boot, "relocatednone", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none", main_gpu=1,
    )
    mgr._vram_total_mib = [8000, 24000]
    mgr._resolve_placement_locked = lambda tag: (15200, 1, 0, "none", 0, False, True)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="relocatednone", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_widened_2c_resolved_split_mode_not_manifest(
    tmp_path, monkeypatch
):
    """2c: guards the "manifest-placement-instead-of-resolved" case, sum
    arm. Manifest pins split_mode='none' (single-card), but the reservation
    resolves to split_mode='row' (stubbed via _resolve_placement_locked) --
    a layer/row/tensor split spans every card, so the check must SUM
    capacity, not bound by a single card, once it is told the RESOLVED
    split_mode. capacity=[8000, 24000] (sum 32000): a need bigger than
    either single card but within the sum only requeues if the resolved
    'row' (not the manifest's 'none') governs the math."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(
        boot, "relocatedrow", expected_vram_bytes=28000 * 1024 * 1024,
        split_mode="none",
    )
    mgr._vram_total_mib = [8000, 24000]
    mgr._resolve_placement_locked = lambda tag: (28200, 1, 0, "row", 0, False, True)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="relocatedrow", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, (
            "the RESOLVED split_mode='row' sums both cards (32000 MiB), "
            "which fits -- using the manifest's raw split_mode='none' "
            "(bounded to a single 8000 or 24000 MiB card) would wrongly "
            "fail it once as never-fits"
        )
        assert not slot.completion_future.done()
        assert calls[0] > 1
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_widened_2d_resolved_main_gpu_nonzero_binding_card(
    tmp_path, monkeypatch
):
    """Guards the
    "resolved main_gpu ignored, hardcoded to 0" case, which 2a/2b never
    could -- 2a never overrides main_gpu (manifest's own 0 flows straight
    through), and 2b's RESOLVED main_gpu, even though it differs from the
    manifest's, is ALSO 0, so an implementation that just hardcodes 0 is invisible
    to both. Here main_gpu=1 (no override in play -- the manifest's own
    value IS what flows through, same shape as 2a), and card 1 is the
    SMALLER/binding one: capacity=[24000, 8000]. A need that fits card 0
    (24000) but not card 1 (8000) distinguishes "uses main_gpu=1" (never-
    fits, fails once) from "hardcodes main_gpu=0" (wrongly fits, requeues
    forever)."""
    mgr, calls, boot = _make_manager(tmp_path, monkeypatch, scan_result=_OOM_SCAN)
    _seed_manifest(
        boot, "cardonebinding", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none", main_gpu=1,
    )
    mgr._vram_total_mib = [24000, 8000]
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="cardonebinding", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert calls[0] == 1, (
            "main_gpu=1 (8000 MiB, the binding card) must never fit this "
            "need -- a mutant hardcoding main_gpu=0 would wrongly check "
            "card 0 (24000 MiB, fits) and requeue forever instead"
        )
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# Backstock-tick retry for a resident-less box (an off-loop
# to_thread(_vram_budget, ...) reading): a resident-less box parked via
# OOM-requeue never gets a _make_room_signal wake, so a backstock tick must
# take a fresh off-loop reading and attempt exactly once on a confirmed
# rise -- never on a bare timer, never with a resident present, never on an
# unreadable probe.
# ---------------------------------------------------------------------------
async def test_empty_box_no_rise_never_attempts(tmp_path, monkeypatch):
    """A CONSTANT reading that already fits (20000 MiB free against a
    ~15200 MiB need) must still make ZERO attempts -- proves growth, not
    bare "fits", gates the attempt (kills the "bare-timer-attempt-with-no-
    rise" mutant: a fits-only check would attempt on the very first tick)."""
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, poll_s=0.01,
    )
    _seed_manifest(
        boot, "emptynorise", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none",
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="emptynorise", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "sanity: must OOM-park first"
        fake_probe, probe_calls = _make_scripted_free_vram_all_mib([[20000]])
        monkeypatch.setattr(safety, "_read_free_vram_all_mib", fake_probe)
        retried = await _pump_backstock_only_until_retried(
            mgr, calls, calls[0], timeout=0.3,
        )
        assert not retried, (
            "a reading that never GROWS must never trigger an admission "
            "attempt, even if it already numerically fits -- a bare-timer "
            "fits-only check would wrongly attempt here"
        )
        assert probe_calls["n"] > 1, (
            "sanity: the probe must actually have been reached more than "
            "once for 'no rise across ticks' to mean anything"
        )
        assert not slot.completion_future.done()
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_empty_box_rise_makes_exactly_one_attempt(
    tmp_path, monkeypatch
):
    """Readings start insufficient (8000 MiB, need ~15200) then GROW past
    that baseline to a fitting value (25000 MiB) -- must make EXACTLY one
    admission attempt on the tick where growth is first observed."""
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, poll_s=0.01,
    )
    _seed_manifest(
        boot, "emptyrise", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none",
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="emptyrise", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "sanity: must OOM-park first"
        fake_probe, probe_calls = _make_scripted_free_vram_all_mib(
            [[8000], [8000], [25000], [25000]]
        )
        monkeypatch.setattr(safety, "_read_free_vram_all_mib", fake_probe)
        retried = await _pump_backstock_only_until_retried(
            mgr, calls, calls[0], timeout=2.0,
        )
        assert retried, (
            "a reading that GROWS past the last insufficient reading and "
            "now fits must trigger exactly one admission attempt"
        )
        assert not slot.completion_future.done()
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_resident_present_keeps_wake_only_behavior(
    tmp_path, monkeypatch
):
    """A box that is NOT empty (some other resident live) must keep the EXISTING
    wake-only behavior unchanged -- zero attempts on a backstock tick even
    when the probe would say "fits" (25000 MiB, need ~15200), because the
    empty-box check must be skipped entirely, not merely fail to trigger by
    coincidence (guards against the "attempt-on-box-with-a-resident-on-a-backstock-
    tick" regression)."""
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, poll_s=0.01,
    )
    _seed_manifest(
        boot, "notemptybox", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none",
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="notemptybox", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "sanity: must OOM-park first"
        # Injected AFTER parking (not before): the test proves what the
        # BACKSTOCK TICK sees, not the ORIGINAL admission -- an "other"
        # resident live at reservation time is a different scenario (the
        # cross-resident co-residence gate, not this one).
        mgr._residents["other-resident"] = Resident(
            model_tag="other-resident", state=ResidentState.ACTIVE, main_gpu=0,
            split_mode="none", reserved_need_mib=100, port=59999,
        )
        fake_probe, probe_calls = _make_scripted_free_vram_all_mib([[25000]])
        monkeypatch.setattr(safety, "_read_free_vram_all_mib", fake_probe)
        retried = await _pump_backstock_only_until_retried(
            mgr, calls, calls[0], timeout=0.3,
        )
        assert not retried, (
            "a box with ANY live resident must never make a VRAM-probe-"
            "driven admission attempt on a backstock tick -- only a REAL "
            "_make_room_signal wake may ever retry it"
        )
        assert probe_calls["n"] == 0, (
            "the empty-box probe must not even be REACHED when a resident "
            "exists -- proves the check is skipped, not just unlucky"
        )
        assert not slot.completion_future.done()
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_unreadable_probe_never_attempts(tmp_path, monkeypatch):
    """A fresh reading that is unreadable (``None``, nvidia-smi failed)
    must never trigger an attempt and must never establish a baseline
    either -- stays parked indefinitely, exactly like the
    "unknown capacity never fails once" doctrine applied
    to this retry path too."""
    mgr, calls, boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, poll_s=0.01,
    )
    _seed_manifest(
        boot, "unreadableprobe", expected_vram_bytes=15000 * 1024 * 1024,
        split_mode="none",
    )
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="unreadableprobe", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "sanity: must OOM-park first"
        fake_probe, probe_calls = _make_scripted_free_vram_all_mib([None])
        monkeypatch.setattr(safety, "_read_free_vram_all_mib", fake_probe)
        retried = await _pump_backstock_only_until_retried(
            mgr, calls, calls[0], timeout=0.3,
        )
        assert not retried, (
            "an unreadable probe must never attempt admission -- "
            "'unknown' is not license to act, same doctrine as "
            "_model_never_fits_known_capacity's own unknown-capacity arm"
        )
        assert probe_calls["n"] > 1, (
            "sanity: the probe was actually reached repeatedly, not "
            "short-circuited some other way"
        )
        assert not slot.completion_future.done()
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)
