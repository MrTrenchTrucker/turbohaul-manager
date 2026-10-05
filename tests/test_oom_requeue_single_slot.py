"""OOM requeue, single-slot config (TurboHaul: OOM load failure goes back in the queue) —
written against max_parallel_sidecars=1, which might look like it drives
manager.py's _process_slot failure branch ("if not healthy:").

⛔ NOTE: IT DOES
NOT. worker_loop unconditionally delegates to
_dispatch_loop at ANY max_parallel_sidecars value -- config.py pins the
field ge=1, so the ">= 1" branch is always taken and _process_slot's one
call site (inside worker_loop's own dead while-loop, per the
comment there) is UNREACHABLE. A stack trace taken inside
_requeue_on_oom_load_failure, run
against this file's own test_oom_with_live_disconnect_is_requeued_not_failed,
shows the call arriving from _drive_resident ->
_spawn_for_resident -- never from _process_slot -- and nothing in this
file reaches _process_slot's own failure line at all.
Another test file independently documents the identical fact.
(The resident-driver test file covers the same path.)

Every test in this file therefore exercises the SAME live path as
the resident-driver OOM-requeue tests
(_drive_resident -> _spawn_for_resident), NOT _process_slot, regardless of
this file's max_parallel_sidecars=1 config. Nothing here is evidence about
_process_slot's own failure branch. That branch
is deliberately left unedited: a change should touch only reachable code,
since editing dead code ships lines no test can drive through the live loop:
a green that could never be red. The fix therefore needs no hunk
inside _process_slot.
The single-slot _process_slot load-failure branch is dead code (its only
call site is in worker_loop's dead while-loop); the only live load-failure
path is _spawn_for_
resident, and the change touches only that. This file is NOT redundant with
the resident-driver file's own tests: it independently confirms the SAME
shared decision (_requeue_on_oom_load_failure) is reached correctly at
max_parallel_sidecars=1, a config the resident-driver file does not cover
(it uses max_parallel_sidecars=2) -- it just is not proof of a SECOND,
independent code path, because there isn't one.

Harness mirrors tests/test_lifecycle_hardening.py's
TestHealthTimeoutBounded (max_parallel_sidecars=1, an injected
_wait_healthy/_spawn/_sigterm/_vram_verify, safety_enabled=False) -- that
file's own "mgr._active_handle is None" assertion is ALSO consistent with
never having gone through _process_slot (that scalar is written only by
_process_slot's own spawn line, so it reads None from __init__ regardless);
flagged as a reading of that file, not a measurement of it -- it has not
been touched here and its pass/fail is not judged here.

load_verify_log.scan_engine_log_for_errors is monkeypatched directly (mock
at the seam, keeping the test hermetic) so classification is
driven by an in-memory dict, never a real engine-log file. (The
REAL, unstubbed-scan proof lives in the resident-driver test
file instead, since that is the file exercising the one live call site.)
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from turbohaul import load_verify_log
from turbohaul import manager as manager_module
from turbohaul import safety
from turbohaul.config import PullConfig, QueueConfig, RuntimeConfig
from turbohaul.manager import TurbohaulManager
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
_UNKNOWN_SCAN = {
    "engine_errors_detected": None,
    "engine_error_lines": [],
    "reason": "neither log file was readable",
}

_SHUTDOWN_TIMEOUT_S = 1.5


async def _bounded_shutdown(mgr, *, timeout=_SHUTDOWN_TIMEOUT_S):
    """See the resident-driver test file's own copy
    of this helper for the full rationale -- ``mgr.shutdown()`` drains every
    ``_spawn_bg``'d background task with a bare, unbounded ``await``, so a
    mutant that makes the OOM retry-wait companion's loop never observe its
    own exit condition turns a test's ``finally: await _bounded_shutdown(mgr)`` into
    a hang, not a failure. Wrapping the same call in ``asyncio.wait_for``
    turns that hang into a named, reportable ``asyncio.TimeoutError``."""
    await asyncio.wait_for(mgr.shutdown(), timeout=timeout)


@pytest.fixture
def boot_and_runtime(tmp_path):
    from turbohaul.config import (
        BootConfig, RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
    )
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, loading_health_timeout_s=10, max_parallel_sidecars=1),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_manager(boot_and_runtime, monkeypatch, *, scan_result, always_unhealthy=True):
    """A manager whose spawn attempts always report unhealthy, classified by
    ``scan_result``. Returns (mgr, call_count) where call_count[0] increments
    on every _wait_healthy call, so a test can prove more than one attempt
    was made (i.e. the retry loop actually re-attempted admission)."""
    boot, runtime = boot_and_runtime
    mgr = TurbohaulManager(boot, runtime)

    call_count = [0]

    async def fake_wait_healthy(port, timeout_s, **kw):
        call_count[0] += 1
        return not always_unhealthy if call_count[0] > 1 and not always_unhealthy else False

    mgr._wait_healthy = fake_wait_healthy

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = 12345 + call_count[0]
        proc.poll.return_value = None  # alive
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    mgr._spawn = fake_spawn

    async def fake_sigterm(handle, **kw):
        return True, "sigterm-clean"

    mgr._sigterm = fake_sigterm

    async def fake_vram(**kw):
        return True, 0

    mgr._vram_verify = fake_vram

    monkeypatch.setattr(
        load_verify_log, "scan_engine_log_for_errors", lambda *a, **k: scan_result
    )
    # Tests never wait the real production cadence (by design: "tests
    # use ~0").
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_POLL_S", 0.01)
    monkeypatch.setattr(manager_module, "_OOM_REQUEUE_SETTLE_S", 0.0)
    # Host-independence of the VRAM probe:
    # this file's own _make_manager pins
    # safety._read_free_vram_all_mib below; left as a REAL
    # nvidia-smi subprocess call, every reservation-time admission here
    # would be host-GPU-dependent (GPU-hidden: degrades to None, "unreadable" ->
    # the documented degrade-open/no-attempt arms; GPU-visible: real free-
    # MiB numbers instead, changing outcomes). Same stub, same constant, as
    # the resident-driver OOM-requeue tests' _make_manager.
    monkeypatch.setattr(
        safety, "_read_free_vram_all_mib", lambda: [999_999_999, 999_999_999]
    )

    return mgr, call_count


def _write_manifest(boot, model_tag="m", expected_vram_bytes=0):
    (boot.storage.manifests_path / f"{model_tag}.yaml").write_text(
        f"model_tag: {model_tag}\n"
        "gguf_blob_sha256: " + "a" * 64 + "\n"
        f"display_name: \"{model_tag}\"\n"
        "description: test\n"
        "context_size: 2048\n"
        f"expected_vram_bytes: {expected_vram_bytes}\n"
        "llama_server_flags: {}\n"
    )


async def _run_worker(mgr):
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())


async def _pump_until_retried(mgr, calls, start_count, *, timeout=2.0):
    """Poll-based, not sleep-based: notify _make_room_signal repeatedly and
    yield, until a NEW admission attempt is observed (calls[0] grows past
    start_count) or timeout elapses. Immune to ambient host CPU load, unlike
    a fixed sleep count (a lesson this codebase already learned) -- the
    predicate is checked, not assumed.

    ``start_count`` is captured right after ``_run_worker`` returns, before
    the freshly-created worker task has had a chance to run at all -- so it
    is usually 0, i.e. BEFORE the very first (non-retry) admission attempt.
    Phase 1 waits for that initial attempt to land and treats its count as
    the real baseline; only phase 2, pumping notifies past that baseline,
    proves a genuine RETRY happened rather than just the first attempt.
    """
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


# ---------------------------------------------------------------------------
# 2.b (unchanged): no disconnect_event -> fails once. RED-on-base-revision
# precedent already exists (test_lifecycle_hardening.py); this is the same
# shape with the OOM scan wired in, proving the classifier does not change
# this path at all.
# ---------------------------------------------------------------------------
async def test_oom_with_no_disconnect_event_fails_once_unchanged(boot_and_runtime, monkeypatch):
    boot, runtime = boot_and_runtime
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    try:
        slot = await mgr.submit(model_tag="m", prompt="x", wait_for_completion=True)
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert calls[0] == 1, "must not retry when there is no disconnect signal"
        assert slot.oom_requeue_pending is False
    finally:
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# a/b: OOM + live disconnect_event -> requeued, NOT failed, and genuinely
# retried more than once (proves the loop is real, not a one-shot).
# ---------------------------------------------------------------------------
async def test_oom_with_live_disconnect_is_requeued_not_failed(boot_and_runtime, monkeypatch):
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, "must have made more than one admission attempt"
        assert not slot.completion_future.done(), (
            "an OOM failure with a live disconnect signal must never fail "
            "the request while the caller stays connected"
        )
        assert calls[0] > 1, "must have made more than one admission attempt"
        assert slot.oom_requeue_pending is True
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# disconnect while requeued -> the wait-companion returns without
# re-enqueuing; the request ends (existing disconnect machinery owns the
# finish, as designed -- no new claim/eviction logic here).
# ---------------------------------------------------------------------------
async def test_disconnect_while_requeued_stops_the_retry_loop(boot_and_runtime, monkeypatch):
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        await asyncio.sleep(0.03)  # let the first OOM+requeue happen
        assert calls[0] >= 1
        disconnect_event.set()
        calls_at_disconnect = calls[0]
        await asyncio.sleep(0.15)  # give the retry-wait companion a chance to notice
        # No NEW admission attempt should occur once disconnected -- the
        # companion must return, not re-enqueue.
        assert calls[0] == calls_at_disconnect, (
            "a disconnected slot must not be re-admitted from the OOM "
            "retry-wait companion"
        )
    finally:
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# d: a non-OOM engine error, with a live disconnect signal, still fails
# once -- classification, not the disconnect signal, gates the requeue.
# ---------------------------------------------------------------------------
async def test_non_oom_error_with_disconnect_still_fails_once(boot_and_runtime, monkeypatch):
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OTHER_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert calls[0] == 1
        assert slot.oom_requeue_pending is False
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# unknown-cause (unreadable/empty engine log), with a live disconnect
# signal, still fails once -- an unprovable cause never requeues.
# ---------------------------------------------------------------------------
async def test_unknown_cause_with_disconnect_still_fails_once(boot_and_runtime, monkeypatch):
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_UNKNOWN_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert calls[0] == 1
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# g: a never-fits model (need provably exceeds a KNOWN empty-box capacity)
# fails once even though the failure is OOM-classified and disconnect is
# live -- requeuing it would wait forever for a change that can never come.
# ---------------------------------------------------------------------------
async def test_never_fits_known_capacity_fails_once(boot_and_runtime, monkeypatch):
    boot, runtime = boot_and_runtime
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    # A model whose manifest declares more VRAM than any card could ever
    # hold, on a FULLY KNOWN (non-None, all-positive) capacity reading.
    _write_manifest(boot, model_tag="huge", expected_vram_bytes=999_000 * 1024 * 1024)
    mgr._vram_total_mib = [24000, 24000]  # two known 24GB cards; need >> both
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="huge", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        with pytest.raises(RuntimeError, match="loading-fail"):
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
        assert calls[0] == 1, "a never-fits model must never be requeued"
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# Additional case: UNKNOWN capacity (_vram_total_mib None) is UNDECIDED,
# never never-fits -- the same "huge" manifest requeues instead of failing
# once when capacity has not been read yet.
# ---------------------------------------------------------------------------
async def test_unknown_capacity_requeues_instead_of_never_fits(boot_and_runtime, monkeypatch):
    boot, runtime = boot_and_runtime
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    _write_manifest(boot, model_tag="huge", expected_vram_bytes=999_000 * 1024 * 1024)
    assert mgr._vram_total_mib is None  # capacity genuinely not yet polled
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="huge", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried
        assert not slot.completion_future.done(), (
            "unknown capacity must never be treated as never-fits -- "
            "an error here would end a legitimate caller's request"
        )
        assert calls[0] > 1
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# A requeued+connected wait survives past a small injected timeout_s
# (never a real 2h wait); an ORDINARY (never-requeued) wait still times out
# at timeout_s, unchanged -- the control for the same mutant class.
# ---------------------------------------------------------------------------
async def test_requeued_connected_survives_past_injected_timeout(boot_and_runtime, monkeypatch):
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        await _run_worker(mgr)
        task = asyncio.create_task(
            mgr.submit_and_wait(
                "m", "x", disconnect_event=disconnect_event, timeout_s=0.05,
            )
        )
        await asyncio.sleep(0.2)  # well past the 0.05s timeout_s
        assert not task.done(), (
            "timeout_s must not end an OOM-requeued+connected wait, even "
            "well past its own value"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_ordinary_connected_wait_still_times_out(boot_and_runtime, monkeypatch):
    # The real control for a mutant that widens the requeue exemption to "any
    # connected wait" instead of "oom_requeue_pending AND connected": a
    # slot that is connected (disconnect_event live) but never became
    # OOM-requeued (no worker running at all, so it never even attempts
    # admission) must still time out at timeout_s exactly like it does without
    # the exemption. Only oom_requeue_pending=True may suspend the clock.
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        # The "requeue exemption widened to ordinary
        # waits" mutant makes _is_exempt() true for THIS slot too (it is
        # connected), so its inner asyncio.wait_for gets per_wait=None --
        # truly unbounded, not just slow. pytest.raises(asyncio.TimeoutError)
        # alone cannot tell a genuine ~0.05s timeout_s expiry apart from this
        # outer 2.0s safety bound also raising asyncio.TimeoutError (both are
        # the same exception type) -- the elapsed-time assertion below is
        # what actually distinguishes them; a hang is a NAMED failure now
        # instead of hanging the whole suite.
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mgr.submit_and_wait(
                    "m", "x", disconnect_event=disconnect_event, timeout_s=0.05,
                ),
                timeout=2.0,
            )
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 1.0, (
            f"must time out via its OWN timeout_s (~0.05s), not this test's "
            f"outer 2.0s safety bound -- {elapsed:.2f}s elapsed means the requeue "
            f"exemption is wrongly suspending an ORDINARY connected wait"
        )
    finally:
        await _bounded_shutdown(mgr)


async def test_backstock_alone_never_makes_an_admission_attempt(boot_and_runtime, monkeypatch):
    # Rule: a still-not-fitting request is never retried on a
    # timer alone. With the poll interval tiny and NO real
    # _make_room_signal notify ever fired, several backstock intervals must
    # elapse with exactly ONE admission attempt (the original OOM failure)
    # -- zero retries driven by the backstock alone.
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        await _run_worker(mgr)
        await asyncio.sleep(0.15)  # >> several _OOM_REQUEUE_POLL_S (0.01s) backstocks
        assert calls[0] == 1, (
            "a lost-wake backstock timeout must never by itself cause a "
            "re-admission attempt"
        )
        assert not slot.completion_future.done()
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)


async def test_ordinary_wait_still_times_out_unchanged(boot_and_runtime, monkeypatch):
    # A slot that never reaches LOADING at all (queue kept artificially full
    # is overkill here -- simplest control: no worker_loop running, so the
    # slot just sits STAGED forever, never becomes OOM-requeued, and
    # timeout_s must still end the wait exactly as it does without the exemption).
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await mgr.submit_and_wait("m", "x", timeout_s=0.05)
    finally:
        await _bounded_shutdown(mgr)


# ---------------------------------------------------------------------------
# No second claim is ever registered across a requeue cycle.
# ---------------------------------------------------------------------------
async def test_no_second_claim_after_requeue(boot_and_runtime, monkeypatch):
    boot, runtime = boot_and_runtime
    mgr, calls = _make_manager(boot_and_runtime, monkeypatch, scan_result=_OOM_SCAN)
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event,
            client_meta={"ip": "10.0.0.5"},
        )
        await _run_worker(mgr)
        retried = await _pump_until_retried(mgr, calls, calls[0])
        assert retried, "sanity: the slot was really re-attempted"
        claims_for_slot = [
            c for c in mgr._fastlane_claims.values() if c.get("slot") is slot
        ]
        assert len(claims_for_slot) <= 1, (
            "a requeue must never leave more than one live claim for the "
            "same slot"
        )
    finally:
        disconnect_event.set()
        await _bounded_shutdown(mgr)
