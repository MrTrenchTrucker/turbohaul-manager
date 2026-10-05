"""Causal proof for the dead-idle sweep NAMING
what it caught, instead of silently recovering from it.

Two things are proven separately:

1. ``TestClassifyIdleDeadHolderPure`` -- the new, standalone
   ``load_verify_log.classify_idle_dead_holder`` read helper, in isolation,
   against crafted engine-log fixtures covering both named cases (clean
   shutdown / the mid-serving fault) and the two
   "we genuinely don't know" cases (could not check / readable but neither
   signature present) that must NOT collapse into either named verdict.

2. ``TestDeadIdleSweepClassifyWiring`` -- drives the REAL worker_loop sweep
   (manager.py) end to end with an actually-dead fake
   handle, and proves: (a) the new IDLE_DEAD_CLASSIFY line fires with the
   right verdict for both named cases, (b) it STILL fires (with a non-None,
   honest "could not classify" verdict) when there is nothing to read --
   satisfying the "empty outcome must be visible" constraint one level below
   the sweep's own top-level warning, and (c) the pre-existing "Idle-hot
   holder DEAD..." warning and the actual teardown/recovery are BOTH
   completely unchanged -- this change is attribution only.

Module-level import is ``from turbohaul import load_verify_log`` (the module
itself already exists) and every call site below reaches the new
function via attribute access (``load_verify_log.classify_idle_dead_holder``),
never ``from turbohaul.load_verify_log import classify_idle_dead_holder``.
This is deliberate: without the change the new function does not exist, and a
top-level symbol import would fail at COLLECTION time, cascading a false
"every test in this file failed" signal across tests that have nothing to do
with each other (a known collection-time trap, cited
again here). Reaching the symbol via the module keeps each test's
failure -- an AttributeError for the pure tests, or a real assertion failure
for the wiring tests, since the wiring tests' new log line simply never
appears on unchanged code -- scoped to that one test, and behaviorally
honest about what the unchanged code actually does.
"""
import logging
import time
from unittest.mock import MagicMock

import pytest

from turbohaul import load_verify_log
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
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


# ===========================================================================
# Part 1 -- the pure classifier, in isolation
# ===========================================================================
class TestClassifyIdleDeadHolderPure:
    def test_no_engine_log_path_given(self):
        out = load_verify_log.classify_idle_dead_holder(None)
        assert out["death_class"] is None
        assert "no engine_log_path supplied" in out["reason"]
        assert out["matched_line"] is None

    def test_neither_file_exists(self, tmp_path):
        missing = str(tmp_path / "never_written.log")
        out = load_verify_log.classify_idle_dead_holder(missing)
        assert out["death_class"] is None
        assert "not found" in out["reason"]

    def test_file_present_but_empty(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] is None
        assert "empty" in out["reason"]

    def test_fault_signature_no_announcement(self, tmp_path):
        """The mid-serving fault shape: active mid-serving, CUDA error, file ends
        cold -- zero further lines, no exit announcement anywhere."""
        p = tmp_path / "fault.log"
        p.write_text(
            "H.01.000.000 I main: loading model\n"
            "H.01.480.000 I slot: context checkpoint eviction\n"
            "H.01.481.000 E ggml_backend_cuda_cpy_tensor_async: "
            "CUDA error: unspecified launch failure\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "fault_signature"
        assert "CUDA error" in out["matched_line"]

    def test_destructor_fault_after_announced_shutdown_is_clean(self, tmp_path):
        """The BENIGN shape: the engine already announced its own
        exit; the CUDA error that follows is a teardown-side-effect
        (~ggml_backend_cuda_buffer_context destructor), not the cause."""
        p = tmp_path / "benign.log"
        p.write_text(
            "H.02.000.000 I main: task cancelled\n"
            "H.02.001.000 I main: cleaning up before exit\n"
            "H.02.002.000 E ~ggml_backend_cuda_buffer_context: "
            "CUDA error: an illegal memory access was encountered\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "clean_shutdown"

    def test_plain_clean_shutdown_no_fault_at_all(self, tmp_path):
        p = tmp_path / "plain_clean.log"
        p.write_text(
            "H.01.000.000 I main: serving request\n"
            "H.01.900.000 I main: cleaning up before exit\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "clean_shutdown"

    def test_readable_nonempty_but_no_known_signature(self, tmp_path):
        p = tmp_path / "mystery.log"
        p.write_text(
            "H.01.000.000 I main: loading model\n"
            "H.01.100.000 I slot: prefill 512 tokens\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "no_signature_found"
        assert "neither the clean-exit mark" in out["reason"]

    def test_stdio_sibling_alone_carries_the_fault(self, tmp_path):
        """Mirrors scan_engine_log_for_errors's own documented reason for
        checking both sinks: some fatal output only reaches .stdio."""
        base = tmp_path / "split.log"
        base.write_text("H.01.000.000 I main: loading model\n")
        stdio = tmp_path / "split.log.stdio"
        stdio.write_text(
            "GGML_ASSERT: CUDA error: an illegal memory access was encountered\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(base))
        assert out["death_class"] == "fault_signature"

    def test_conflicting_files_fault_wins(self, tmp_path):
        """.log looks like an ordinary clean shutdown; .stdio independently
        shows the fault with nothing after it. The merge must not let a
        clean-looking structured log paper over a fault only .stdio saw --
        errs toward NOT missing a real fault."""
        base = tmp_path / "conflict.log"
        base.write_text(
            "H.01.000.000 I main: serving request\n"
            "H.01.900.000 I main: cleaning up before exit\n"
        )
        stdio = tmp_path / "conflict.log.stdio"
        stdio.write_text("CUDA error: unspecified launch failure\n")
        out = load_verify_log.classify_idle_dead_holder(str(base))
        assert out["death_class"] == "fault_signature"

    def test_never_raises_on_unreadable_path(self, tmp_path):
        # A directory, not a file -- open() raises IsADirectoryError (an
        # OSError subclass); must be swallowed same as scan_engine_log_for_errors.
        d = tmp_path / "a_directory.log"
        d.mkdir()
        out = load_verify_log.classify_idle_dead_holder(str(d))
        assert out["death_class"] is None

    def test_verbatim_measured_signature_is_clean(self, tmp_path):
        """The engine's REAL shutdown wording (a CUDA error during
        shutdown, with no exception raised) differs from
        this file's earlier H.MM.SSS.uuu-style synthetic fixtures.
        Reproduced with the exact quoted text (timestamp prefixes substituted;
        none were quoted originally) to prove this classifier -- built
        against the synthetic corpus -- still holds against a
        differently-worded real signature, not just
        the original synthetic shape."""
        p = tmp_path / "idle_benign.log"
        p.write_text(
            "12.34.567.890 I srv    operator(): cleaning up before exit...\n"
            "12.34.568.001 E CUDA error: an illegal memory access was encountered\n"
            "12.34.568.002 E   current device: 1, in function "
            "~ggml_backend_cuda_buffer_context at "
            "/opt/turboquant/ggml/src/ggml-cuda/ggml-cuda.cu:660\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "clean_shutdown", out

    def test_same_message_text_without_announcement_is_fault(self, tmp_path):
        """The other required direction (a
        classifier only ever shown to say benign is not a classifier) -- the
        EXACT SAME error text as the test above, but with no clean-exit mark
        anywhere in the file (this destructor also fires mid-serving, e.g. on
        checkpoint/buffer eviction, not only at shutdown -- see
        test_fault_signature_no_announcement above for that same distinction
        with a different call site). Proves the verdict comes from context
        (was an exit announced in THIS file), never from the message text
        alone -- the identical string is fault_signature here and
        clean_shutdown in the test above."""
        p = tmp_path / "idle_fatal.log"
        p.write_text(
            "12.34.560.000 I srv    task: dispatching to slot 0\n"
            "12.34.567.890 E CUDA error: an illegal memory access was encountered\n"
            "12.34.567.891 E   current device: 1, in function "
            "~ggml_backend_cuda_buffer_context at "
            "/opt/turboquant/ggml/src/ggml-cuda/ggml-cuda.cu:660\n"
        )
        out = load_verify_log.classify_idle_dead_holder(str(p))
        assert out["death_class"] == "fault_signature", out
        assert "CUDA error" in out["matched_line"]


# ===========================================================================
# Part 2 -- the real sweep, wired end to end
# ===========================================================================
def _boot_runtime(tmp_path, grace_seconds=0, idle_hot_load_seconds=120):
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 77_777
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


async def _boot_manager_with_one_idle_hot_holder(tmp_path):
    """Shared setup: real worker_loop, DI fakes for spawn/health/sigterm/
    vram/complete (no real llama-server), one request served, holder now
    idle-hot and ALIVE. Returns (mgr, sigterm_calls)."""
    boot, runtime = _boot_runtime(tmp_path)
    sigterm_calls = []

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_fake_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        sigterm_calls.append(handle.model_tag)
        return True, "sigterm-clean"

    async def fake_vram(*a, **kw):
        return True, None

    async def fake_complete(slot, handle):
        return {"ok": True}

    manifests_dir = boot.storage.manifests_path
    (manifests_dir / "real-model.yaml").write_text(
        "model_tag: real-model\n"
        "gguf_blob_sha256: " + "a" * 64 + "\n"
        "display_name: \"Real Model\"\n"
        "description: test\n"
        "context_size: 2048\n"
        "expected_vram_bytes: 0\n"
        "llama_server_flags: {}\n"
    )

    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram,
        complete_fn=fake_complete,
    )
    import asyncio
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    await mgr.submit_and_wait("real-model", "prompt", thread_id="t1")
    await asyncio.sleep(0.2)  # grace + idle-hold
    assert mgr._idle_handle is not None, "idle holder should exist"
    return mgr, sigterm_calls


def _classify_lines(caplog):
    return [r.message for r in caplog.records if "IDLE_DEAD_CLASSIFY" in r.message]


def _dead_warning_lines(caplog):
    return [
        r.message for r in caplog.records
        if "Idle-hot holder DEAD between requests" in r.message
    ]


async def _wait_for(predicate, timeout_s=5.0, interval_s=0.05):
    import asyncio
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


@pytest.mark.asyncio
class TestDeadIdleSweepClassifyWiring:
    async def test_fault_signature_fires_and_recovery_is_unchanged(
        self, tmp_path, caplog,
    ):
        mgr, sigterm_calls = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        engine_log = tmp_path / "engine_fault.log"
        engine_log.write_text(
            "H.01.000.000 I main: loading model\n"
            "H.01.480.000 I slot: context checkpoint eviction\n"
            "H.01.481.000 E ggml_backend_cuda_cpy_tensor_async: "
            "CUDA error: unspecified launch failure\n"
        )
        held.engine_log_path = str(engine_log)
        held.proc.poll.return_value = -11  # child is gone

        try:
            with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                found = await _wait_for(lambda: _classify_lines(caplog))
            assert found, "sweep never emitted IDLE_DEAD_CLASSIFY on a dead holder"
            lines = _classify_lines(caplog)
            assert any("class=fault_signature" in l for l in lines)
            assert any("CUDA error" in l for l in lines)
            assert any(f"port={held.port}" in l for l in lines)
            assert any("model=real-model" in l for l in lines)

            # The pre-existing top-level alert is byte-for-byte unchanged --
            # this change added a new line, it did not touch the old one.
            assert _dead_warning_lines(caplog), (
                "the original 'Idle-hot holder DEAD...' warning must still fire"
            )

            # Recovery itself is unchanged: the dead holder still gets torn
            # down and the manager still attempts the sigterm-side teardown
            # seam (DI'd fake, since the process is fake to begin with).
            recovered = await _wait_for(lambda: mgr._idle_handle is None)
            assert recovered, "dead holder was never torn down -- recovery broke"
            assert "real-model" in sigterm_calls
        finally:
            await mgr.shutdown()

    async def test_clean_shutdown_fires_distinctly_from_fault(
        self, tmp_path, caplog,
    ):
        mgr, _ = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        engine_log = tmp_path / "engine_clean.log"
        engine_log.write_text(
            "H.01.000.000 I main: serving request\n"
            "H.01.900.000 I main: cleaning up before exit\n"
        )
        held.engine_log_path = str(engine_log)
        held.proc.poll.return_value = 0

        try:
            with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                found = await _wait_for(lambda: _classify_lines(caplog))
            assert found
            lines = _classify_lines(caplog)
            assert any("class=clean_shutdown" in l for l in lines)
            assert not any("class=fault_signature" in l for l in lines), (
                "clean-shutdown holder must not be classified as the fault"
            )
        finally:
            await mgr.shutdown()

    async def test_undetermined_outcome_still_logs_not_silent(
        self, tmp_path, caplog,
    ):
        """Visibility requirement, one level below the sweep's own top-level warning:
        when the classifier genuinely cannot tell (no engine_log_path on
        this fake handle at all), that must still produce a visible,
        honest line -- not silence indistinguishable from 'never ran'."""
        mgr, _ = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        held.engine_log_path = None
        held.proc.poll.return_value = 1

        try:
            with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                found = await _wait_for(lambda: _classify_lines(caplog))
            assert found, (
                "an unclassifiable dead holder produced NO line at all -- "
                "this is exactly the silent-empty-outcome trap"
            )
            lines = _classify_lines(caplog)
            assert any("class=None" in l for l in lines)
            assert any("reason=no engine_log_path supplied" in l for l in lines)
        finally:
            await mgr.shutdown()


# ===========================================================================
# Part 3 -- verification-only: the BROADER "swept and found nothing across
# an idle-hot holder's whole life" case. This is pre-existing behavior (the
# IDLE_TEARDOWN line at the teardown seam already fires for every reason,
# alive=True or alive=False, not something this change adds) -- included to
# prove the claim directly rather than assert it from memory.
# ===========================================================================
@pytest.mark.asyncio
class TestBroaderEmptyOutcomeAlreadyCovered:
    async def test_holder_that_survives_to_idle_expired_still_logs_alive_true(
        self, tmp_path, caplog,
    ):
        boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=1)
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            sigterm_calls.append(handle.model_tag)
            return True, "sigterm-clean"

        async def fake_vram(*a, **kw):
            return True, None

        async def fake_complete(slot, handle):
            return {"ok": True}

        manifests_dir = boot.storage.manifests_path
        (manifests_dir / "real-model.yaml").write_text(
            "model_tag: real-model\n"
            "gguf_blob_sha256: " + "a" * 64 + "\n"
            "display_name: \"Real Model\"\n"
            "description: test\n"
            "context_size: 2048\n"
            "expected_vram_bytes: 0\n"
            "llama_server_flags: {}\n"
        )
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        import asyncio
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            with caplog.at_level(logging.INFO, logger="turbohaul.manager"):
                await mgr.submit_and_wait("real-model", "prompt", thread_id="t1")
                assert await _wait_for(lambda: "real-model" in sigterm_calls)
            lines = [
                r.message for r in caplog.records
                if r.message.startswith("IDLE_TEARDOWN")
                and "reason=idle_expired" in r.message
            ]
            assert lines, "no IDLE_TEARDOWN line for a holder that lived to expiry"
            assert any("alive=True" in l for l in lines), (
                "a holder that survived its whole idle-hot life must still "
                "produce a positive, non-silent record of that at teardown"
            )
        finally:
            await mgr.shutdown()
