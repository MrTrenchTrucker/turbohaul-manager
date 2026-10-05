"""Observability: KV_REUSE_OUTCOME.

KV_REUSE (see the KV_REUSE observability test) fires at decision time and
reports what the manager offered to restore -- it can never know what the
engine actually kept, because the engine's own counter (n_prompt_tokens_cache,
surfaced by the live poller into self.live_generations / self.live_generation)
is only populated once generation has actually run. KV_REUSE_OUTCOME is the
companion line, fired once per request from the point each completion path
knows generation is over: the non-streaming _complete_fn wrap, and each of
the streaming paths' stream_done_event resolution.

A reuse figure is reported ONLY when the wait genuinely resolved via the
route's own signal AND the poller's sample is proven -- by recomputing
compute_generation_id(pid, spawn_seq, slot_id) and matching it against the
poller's own generation_id -- to belong to THIS exact request. Every other
outcome (forced unwind, timeout, cancelled, exception, no matching sample)
reports an explicit unavailable + reason, never a guess.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_kv_reuse_outcome.py -v
"""
from __future__ import annotations

import asyncio
import logging
import re

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
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
from turbohaul.live_monitor import compute_generation_id
from turbohaul.manager import Resident, TurbohaulManager, _bin_identity
from turbohaul.slot import Slot, SlotState

pytestmark = pytest.mark.asyncio

_QWEN = "qwen3.6-27b"
_PID = 4242


@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
            default_port_base=60200,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _FakeHandle:
    def __init__(self, pid=_PID):
        self.pid = pid


def _outcome_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("KV_REUSE_OUTCOME")]


def _prime_heartbeat(mgr, slot_id, n_prompt_cache, *, pid=_PID, spawn_seq=0, model_tag=_QWEN):
    """Make the poller's heartbeat look like it already observed THIS exact
    request's generation -- mirrors what LiveSlotsPoller/LiveResidentsSupervisor
    would have written, without needing a real engine or a real poller."""
    gen_id = compute_generation_id(pid, spawn_seq, slot_id)
    heartbeat = {"generation_id": gen_id, "n_prompt_cache": n_prompt_cache}
    mgr.live_generations[model_tag] = heartbeat
    mgr.live_generation = heartbeat
    return heartbeat


class TestLogKvReuseOutcomeDirect:
    """Direct tests of the shared helper -- every wait_state, every guard."""

    async def test_completed_with_matching_sample_reports_real_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t1")
        _prime_heartbeat(mgr, slot.slot_id, 98765)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="completed")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=completed" in lines[0]
        assert "engine_reused_tokens=98765" in lines[0]
        assert f"slot_id={slot.slot_id}" in lines[0]

    async def test_completed_but_no_matching_sample_is_honest_unavailable(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t2")
        # Heartbeat present, but for a DIFFERENT slot_id (e.g. a concurrent
        # request on the same model_tag) -- must not be trusted for this one.
        _prime_heartbeat(mgr, "slot-someone-elses-request", 55555)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="completed")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "reason=no_matching_sample" in lines[0], lines[0]
        assert "98765" not in lines[0]
        assert "55555" not in lines[0]

    async def test_completed_no_heartbeat_at_all_is_honest_unavailable(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t3")
        # live_generations / live_generation left at their fixture defaults ({} / None).
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="completed")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "reason=no_matching_sample" in lines[0], lines[0]

    async def test_completed_no_handle_is_honest_unavailable(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t4")
        _prime_heartbeat(mgr, slot.slot_id, 12345)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, None, wait_state="completed")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "reason=no_handle" in lines[0], lines[0]

    async def test_completed_no_slot_id_is_honest_unavailable(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t5")
        slot.slot_id = None  # defensive: should never happen via Slot.new, but guard it anyway
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="completed")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "reason=no_slot_id" in lines[0], lines[0]

    async def test_forced_never_reports_a_figure_even_with_a_valid_sample(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t6")
        # A genuinely matching, real-looking sample IS present -- prove it's
        # ignored anyway, because wait_state says the wait was force-ended,
        # not genuinely resolved by the route.
        _prime_heartbeat(mgr, slot.slot_id, 77777)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(
                slot, _FakeHandle(), wait_state="forced", reason="fanout_drain_unwind")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=forced" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "reason=fanout_drain_unwind" in lines[0], lines[0]
        assert "77777" not in lines[0]

    async def test_timeout_never_reports_a_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t7")
        _prime_heartbeat(mgr, slot.slot_id, 11111)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="timeout")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=timeout" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "11111" not in lines[0]

    async def test_cancelled_never_reports_a_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t8")
        _prime_heartbeat(mgr, slot.slot_id, 22222)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="cancelled")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=cancelled" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]

    async def test_exception_never_reports_a_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="t9")
        _prime_heartbeat(mgr, slot.slot_id, 33333)
        with caplog.at_level(logging.INFO):
            mgr._log_kv_reuse_outcome(slot, _FakeHandle(), wait_state="exception")
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=exception" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]


class TestNonStreamingCompleteFnWrap:
    """The single DI-seam wrap: exercises the REAL wiring, not just the helper."""

    async def test_successful_completion_logs_outcome_and_preserves_return_value(
        self, mgr, caplog
    ):
        slot = Slot.new(model_tag=_QWEN, thread_id="nt1")
        _prime_heartbeat(mgr, slot.slot_id, 44444)
        _sentinel = {"choices": [{"message": {"content": "hi"}}]}

        async def _fake_real_complete(s, h):
            return _sentinel

        # The wrap is installed once, in __init__, around whatever complete_fn
        # is passed in -- so exercising it means constructing a manager WITH
        # our fake injected, the same way production injects the real httpx
        # forwarder. This runs the actual wrap code path, not a re-implementation.
        mgr2 = TurbohaulManager(mgr.boot, mgr.runtime, complete_fn=_fake_real_complete)
        _prime_heartbeat(mgr2, slot.slot_id, 44444)
        handle = _FakeHandle()
        with caplog.at_level(logging.INFO):
            result = await mgr2._complete_fn(slot, handle)
        assert result is _sentinel, "wrap must not alter the return value"
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=completed" in lines[0], lines[0]
        assert "engine_reused_tokens=44444" in lines[0], lines[0]

    async def test_cancelled_completion_logs_outcome_and_still_propagates_cancellederror(
        self, mgr, caplog
    ):
        slot = Slot.new(model_tag=_QWEN, thread_id="nt2")

        async def _fake_real_complete(s, h):
            raise asyncio.CancelledError()

        mgr2 = TurbohaulManager(mgr.boot, mgr.runtime, complete_fn=_fake_real_complete)
        with caplog.at_level(logging.INFO):
            with pytest.raises(asyncio.CancelledError):
                await mgr2._complete_fn(slot, _FakeHandle())
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=cancelled" in lines[0], lines[0]

    async def test_exception_completion_logs_outcome_and_still_propagates_same_exception(
        self, mgr, caplog
    ):
        slot = Slot.new(model_tag=_QWEN, thread_id="nt3")

        class _BoomError(RuntimeError):
            pass

        async def _fake_real_complete(s, h):
            raise _BoomError("sidecar 500")

        mgr2 = TurbohaulManager(mgr.boot, mgr.runtime, complete_fn=_fake_real_complete)
        with caplog.at_level(logging.INFO):
            with pytest.raises(_BoomError, match="sidecar 500"):
                await mgr2._complete_fn(slot, _FakeHandle())
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=exception" in lines[0], lines[0]


class TestStreamingAwaitStreamedSlot:
    """_await_streamed_slot is the cleanest standalone streaming call site --
    exercises the REAL wiring for the completed/forced/timeout/cancelled
    branches without needing a full worker_loop simulation."""

    async def test_route_genuine_completion_reports_real_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="s1")
        slot.stream_done_event = asyncio.Event()
        slot.stream_handle = _FakeHandle()
        slot.completion_future = asyncio.get_running_loop().create_future()
        _prime_heartbeat(mgr, slot.slot_id, 66666)
        slot.stream_done_event.set()  # the route "finished" before we even wait
        with caplog.at_level(logging.INFO):
            await mgr._await_streamed_slot(slot)
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=completed" in lines[0], lines[0]
        assert "engine_reused_tokens=66666" in lines[0], lines[0]

    async def test_manager_forced_never_reports_a_figure(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="s2")
        slot.stream_done_event = asyncio.Event()
        slot.stream_handle = _FakeHandle()
        slot.completion_future = asyncio.get_running_loop().create_future()
        _prime_heartbeat(mgr, slot.slot_id, 88888)
        # Simulate what a force-set site does: reason FIRST, then set().
        slot.stream_done_forced_reason = "fanout_drain_unwind"
        slot.stream_done_event.set()
        with caplog.at_level(logging.INFO):
            await mgr._await_streamed_slot(slot)
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=forced" in lines[0], lines[0]
        assert "reason=fanout_drain_unwind" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]
        assert "88888" not in lines[0]

    async def test_timeout_never_reports_a_figure(self, mgr, caplog, monkeypatch):
        monkeypatch.setattr(manager_mod, "_STREAM_TIMEOUT_S", 0.05)
        slot = Slot.new(model_tag=_QWEN, thread_id="s3")
        slot.stream_done_event = asyncio.Event()  # never set
        slot.stream_handle = _FakeHandle()
        slot.completion_future = asyncio.get_running_loop().create_future()
        _prime_heartbeat(mgr, slot.slot_id, 99999)
        with caplog.at_level(logging.INFO):
            await mgr._await_streamed_slot(slot)
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=timeout" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]

    async def test_cancelled_never_reports_a_figure_and_still_propagates(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="s4")
        slot.stream_done_event = asyncio.Event()  # never set
        slot.stream_handle = _FakeHandle()
        slot.completion_future = asyncio.get_running_loop().create_future()
        _prime_heartbeat(mgr, slot.slot_id, 10101)
        task = asyncio.create_task(mgr._await_streamed_slot(slot))
        await asyncio.sleep(0)  # let it reach the await
        task.cancel()
        with caplog.at_level(logging.INFO):
            with pytest.raises(asyncio.CancelledError):
                await task
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=cancelled" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]

    async def test_non_streaming_slot_through_this_function_emits_nothing(self, mgr, caplog):
        """Defensive branch: stream_done_event is None (not actually a
        streaming slot). Must not emit a KV_REUSE_OUTCOME line at all -- there
        is nothing to report an outcome ABOUT."""
        slot = Slot.new(model_tag=_QWEN, thread_id="s5")
        slot.stream_done_event = None
        slot.completion_future = asyncio.get_running_loop().create_future()
        with caplog.at_level(logging.INFO):
            await mgr._await_streamed_slot(slot)
        assert _outcome_lines(caplog) == []


class TestServeOnResidentCancellation:
    """Regression guard: _serve_on_resident's serial streaming branch
    (the plain ACTIVE path, no fan-out, no ACTIVE_MATCH) must handle
    CancelledError around its stream_done_event wait -- without a handler a
    cancellation there (which manager.shutdown() triggers routinely, via
    driver_task.cancel()) would skip _log_kv_reuse_outcome entirely and
    propagate straight out of the function. Exercises the REAL function,
    not a reimplementation of its shape."""

    async def test_cancelled_mid_wait_logs_outcome_and_still_propagates(self, mgr, caplog):
        slot = Slot.new(model_tag=_QWEN, thread_id="ser1")
        slot.state = SlotState.LOADING  # legal predecessor of ACTIVE
        slot.stream_ready_event = asyncio.Event()
        slot.stream_done_event = asyncio.Event()  # never set
        slot.client_meta = {"stream": True}
        slot.completion_future = asyncio.get_running_loop().create_future()
        handle = _FakeHandle()
        resident = Resident(model_tag=_QWEN, handle=handle)

        task = asyncio.create_task(mgr._serve_on_resident(resident, slot, handle))
        # _audit_async offloads to a real thread pool (asyncio.to_thread), so a
        # single bare tick is not always enough to reach the streaming wait --
        # poll with a bound instead of assuming a fixed number of ticks.
        await asyncio.wait_for(slot.stream_ready_event.wait(), timeout=5.0)
        task.cancel()
        with caplog.at_level(logging.INFO):
            with pytest.raises(asyncio.CancelledError):
                await task
        lines = _outcome_lines(caplog)
        assert len(lines) == 1, lines
        assert "wait_state=cancelled" in lines[0], lines[0]
        assert "engine_reused_tokens=unavailable" in lines[0], lines[0]


def _decision_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("KV_REUSE ")]


def _extract_thread_hash(line):
    m = re.search(r"thread_hash=(\S+)", line)
    assert m, f"no thread_hash field found in: {line!r}"
    return m.group(1)


class TestThreadHashJoinsAcrossBothLines:
    """Consistency check: the decision-time KV_REUSE line
    and the outcome-time KV_REUSE_OUTCOME line must not carry DIFFERENT thread_hash
    values for the same request whenever the client sends a session_id (the
    normal case) -- the decision-time emitters hash _bin_identity(thread_id,
    client_meta, chain), and the outcome line must not hash raw thread_id directly.
    Two fields both named thread_hash that can silently diverge for the same
    request is itself a lying instrument, independent of whether either line
    is individually correct. Each line is tested and correct in isolation
    above -- a defect would live entirely in the RELATIONSHIP between them, so
    this test asserts that relationship directly."""

    async def test_decision_and_outcome_lines_share_thread_hash_for_the_same_request(
        self, mgr, kv_dir, caplog
    ):
        client_meta = {"session_id": "sess-followup-relationship-test", "is_main": True}
        slot = Slot.new(
            model_tag=_QWEN,
            thread_id="raw-thread-id-must-not-appear-in-either-hash",
            client_meta=client_meta,
        )
        with caplog.at_level(logging.INFO):
            # Decision-time: no saved bins -> the cheapest real call path that
            # still fires the shared _emit_classifier_decision chokepoint.
            await mgr._restore_slot_kv(60200, _QWEN, slot)

            # Outcome-time: real streaming completion wiring, the SAME slot
            # object (same client_meta / thread_id / admission_hash_chain a
            # real request carries end to end from admission through decode).
            slot.stream_done_event = asyncio.Event()
            slot.stream_handle = _FakeHandle()
            slot.completion_future = asyncio.get_running_loop().create_future()
            slot.stream_done_event.set()
            await mgr._await_streamed_slot(slot)

        decision_lines = _decision_lines(caplog)
        outcome_lines = _outcome_lines(caplog)
        assert len(decision_lines) == 1, decision_lines
        assert len(outcome_lines) == 1, outcome_lines

        decision_hash = _extract_thread_hash(decision_lines[0])
        outcome_hash = _extract_thread_hash(outcome_lines[0])
        assert decision_hash == outcome_hash, (
            f"decision line thread_hash={decision_hash!r} != "
            f"outcome line thread_hash={outcome_hash!r} for the SAME request "
            "-- the two halves of the instrument do not join"
        )

        # Non-vacuity control: prove the match is the _bin_identity-derived
        # hash, not a coincidence of both sides hashing raw thread_id (which
        # would make this test pass even with such a bug present,
        # since it's specifically the session_id transform that makes the
        # two derivations diverge).
        raw_thread_id_hash = TurbohaulManager._thread_hash(slot.thread_id)
        assert decision_hash != raw_thread_id_hash, (
            "the shared hash equals sha256(raw thread_id) -- this test isn't "
            "exercising the session-scoped _bin_identity transform, so it "
            "would pass even with the original bug still present"
        )
        expected_hash = TurbohaulManager._thread_hash(
            _bin_identity(slot.thread_id, slot.client_meta, slot.admission_hash_chain))
        assert decision_hash == expected_hash
