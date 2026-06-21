"""Tests for the live inference monitor (tok/s + progress + live output text).

Covers: the tok/s algorithm (rate/EWMA/pct/eta + all honesty resets), the
LiveOutputBuffer SSE reframing + per-generation isolation, the poller's
concurrency guards (torn snapshot, port-reuse revalidate, loading/idle), and the
API surface (/status generation block, the output SSE endpoint, the contentless
/ws/state ping, CSP).
"""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import _CSP_HEADER, create_app
from turbohaul.config import (
    BootConfig,
    MonitorConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.live_monitor import (
    EWMA_ALPHA,
    MAX_LIVE_KEYS,
    MAX_SUBS_PER_GEN,
    LiveOutputBuffer,
    LiveSlotsPoller,
    compute_generation_id,
    idle_generation,
)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _proc_slot(*, id_task=7, n_decoded=0, n_remain=None, has_next=True,
               n_prompt=100, n_prompt_proc=100, max_tokens=1000, n_predict=1000,
               stream=True, is_processing=True):
    return {
        "id": 0, "is_processing": is_processing, "id_task": id_task,
        "n_prompt_tokens": n_prompt, "n_prompt_tokens_processed": n_prompt_proc,
        "params": {"max_tokens": max_tokens, "n_predict": n_predict, "stream": stream},
        "next_token": [{"has_next_token": has_next, "n_remain": n_remain, "n_decoded": n_decoded}],
    }


class _Bus:
    def __init__(self):
        self.events = []

    def publish_nowait(self, e):
        self.events.append(e)


class _State:
    def __init__(self, value):
        self.value = value


class _Slot:
    def __init__(self, state="ACTIVE", slot_id="slot-aaaa", thread_id="thr-xyz"):
        self.state = _State(state)
        self.slot_id = slot_id
        self.thread_id = thread_id


class _Handle:
    def __init__(self, pid=1582, port=11500):
        self.pid = pid
        self.port = port


class _Mgr:
    def __init__(self):
        self._active_slot = None
        self._active_handle = None
        self._spawn_seq = 0
        self.live_generation = None
        self.event_bus = _Bus()
        self._stop_event = asyncio.Event()


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _Client:
    """Fake httpx client; optionally mutates the mgr mid-get to simulate a swap."""

    def __init__(self, data, *, exc=None, on_get=None):
        self._data = data
        self._exc = exc
        self._on_get = on_get
        self.calls = 0

    async def get(self, url):
        self.calls += 1
        if self._on_get:
            self._on_get()
        if self._exc:
            raise self._exc
        return _Resp(self._data)

    async def aclose(self):
        pass


def _poller(mgr=None):
    p = LiveSlotsPoller(mgr or _Mgr())
    return p


# --------------------------------------------------------------------------- #
# tok/s algorithm (_compute) — pure, deterministic
# --------------------------------------------------------------------------- #
class TestTokSAlgorithm:
    def test_rate_pct_eta_after_two_samples(self):
        p = _poller()
        g1 = p._compute([_proc_slot(n_decoded=10, n_remain=990)], 100.0, 1, 0, "slot-a")
        assert g1["state"] == "generating"
        assert g1["tok_s"] is None  # first-decode pending (no 2nd sample yet)
        g2 = p._compute([_proc_slot(n_decoded=50, n_remain=950)], 101.0, 1, 0, "slot-a")
        assert g2["tok_s_instant"] == 40.0
        assert g2["tok_s"] == 40.0          # EWMA seeded = instantaneous
        assert g2["n_decoded"] == 50
        assert g2["pct"] == 5.0             # 50/1000
        assert g2["eta_s"] == round(950 / 40.0, 1)
        assert g2["riders"] == 1
        assert g2["streaming"] is True

    def test_ewma_smoothing(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=0)], 100.0, 1, 0, "slot-a")
        p._compute([_proc_slot(n_decoded=10)], 101.0, 1, 0, "slot-a")  # ewma=10
        g = p._compute([_proc_slot(n_decoded=30)], 102.0, 1, 0, "slot-a")  # inst=20
        expected = EWMA_ALPHA * 20 + (1 - EWMA_ALPHA) * 10
        assert g["tok_s"] == round(expected, 1)
        assert g["tok_s_instant"] == 20.0

    def test_clamp_on_missed_poll_spike(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=0)], 100.0, 1, 0, "slot-a")
        # huge jump over a realistic dt (1e6 tokens in 10ms -> 1e8 tok/s) clamps to 10000
        g = p._compute([_proc_slot(n_decoded=1_000_000)], 100.01, 1, 0, "slot-a")
        assert g["tok_s_instant"] == 10000.0

    def test_growing_n_prompt_does_not_stall_tok_s(self):
        # REGRESSION (caught live): llama.cpp /slots n_prompt_tokens grows with
        # generation (prompt + tokens decoded). It must NOT trigger a reset, or
        # every tick rebaselines and tok/s is stuck at None forever.
        p = _poller()
        p._compute([_proc_slot(id_task=7, n_decoded=10, n_prompt=110)], 100.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(id_task=7, n_decoded=50, n_prompt=150)], 101.0, 1, 0, "slot-a")
        assert g["tok_s"] == 40.0          # rate computed, NOT stuck pending
        assert g["tok_s_instant"] == 40.0

    def test_reset_on_max_tokens_change(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=10, max_tokens=1000)], 100.0, 1, 0, "slot-a")
        p._compute([_proc_slot(n_decoded=50, max_tokens=1000)], 101.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(n_decoded=60, max_tokens=2048)], 102.0, 1, 0, "slot-a")
        assert g["tok_s"] is None

    def test_reset_on_n_decoded_regression(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=500)], 100.0, 1, 0, "slot-a")
        p._compute([_proc_slot(n_decoded=600)], 101.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(n_decoded=5)], 102.0, 1, 0, "slot-a")  # regressed
        assert g["tok_s"] is None
        assert g["n_decoded"] == 5

    def test_stalled_when_frozen_with_more_to_come(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=50, has_next=True)], 100.0, 1, 0, "slot-a")
        p._compute([_proc_slot(n_decoded=50, has_next=True)], 101.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(n_decoded=50, has_next=True)], 103.5, 1, 0, "slot-a")
        assert g["state"] == "stalled"
        assert g["tok_s"] == 0.0
        assert g["stalled"] is True

    def test_finishing_when_frozen_but_no_more(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=50, has_next=False)], 100.0, 1, 0, "slot-a")
        p._compute([_proc_slot(n_decoded=50, has_next=False)], 101.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(n_decoded=50, has_next=False)], 103.5, 1, 0, "slot-a")
        assert g["state"] == "finishing"
        assert g["tok_s"] == 0.0
        assert g["stalled"] is False  # NOT a red alarm

    def test_prefill(self):
        p = _poller()
        g = p._compute(
            [_proc_slot(n_decoded=0, n_prompt=70000, n_prompt_proc=40000)],
            100.0, 1, 0, "slot-a",
        )
        assert g["state"] == "prefill"
        assert g["prompt_progress"] == round(40000 / 70000, 3)
        assert g["tok_s"] is None

    def test_eta_and_pct_null_when_unbounded(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=10, max_tokens=0, n_predict=-1, n_remain=-1)], 100.0, 1, 0, "slot-a")
        g = p._compute([_proc_slot(n_decoded=60, max_tokens=0, n_predict=-1, n_remain=-1)], 101.0, 1, 0, "slot-a")
        assert g["pct"] is None
        assert g["eta_s"] is None
        assert g["max_tokens"] is None
        assert g["n_remain"] is None  # negative sentinel scrubbed

    def test_starvation_keeps_generation_id(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=10)], 100.0, 1, 0, "slot-a")
        g_before = p._compute([_proc_slot(n_decoded=50)], 101.0, 1, 0, "slot-a")
        # >5s gap on the SAME generation -> rate-only reset, gen_id unchanged
        g_after = p._compute([_proc_slot(n_decoded=90)], 110.0, 1, 0, "slot-a")
        assert g_after["generation_id"] == g_before["generation_id"]

    def test_idle_when_no_processing_slot(self):
        p = _poller()
        p._compute([_proc_slot(n_decoded=10)], 100.0, 1, 0, "slot-a")
        g = p._compute([], 101.0, 1, 0, "slot-a")
        assert g["state"] == "idle"
        assert g["tok_s"] == 0.0
        assert p._samples == {}

    def test_parallel_two_slots_sum_rate(self):
        p = _poller()
        p._compute(
            [_proc_slot(id_task=1, n_decoded=10), _proc_slot(id_task=2, n_decoded=20)],
            100.0, 1, 0, "slot-a",
        )
        g = p._compute(
            [_proc_slot(id_task=1, n_decoded=50), _proc_slot(id_task=2, n_decoded=80)],
            101.0, 1, 0, "slot-a",
        )
        assert g["riders"] == 2
        assert g["tok_s_instant"] == 100.0  # 40 + 60
        assert g["n_decoded"] == 80          # headline = max-decoded rider

    def test_vanished_task_garbage_collected(self):
        p = _poller()
        p._compute(
            [_proc_slot(id_task=1, n_decoded=10), _proc_slot(id_task=2, n_decoded=20)],
            100.0, 1, 0, "slot-a",
        )
        p._compute([_proc_slot(id_task=1, n_decoded=50)], 101.0, 1, 0, "slot-a")
        assert set(p._samples.keys()) == {1}

    def test_schema_drift_does_not_crash(self):
        p = _poller()
        # next_token missing entirely
        g = p._compute([{"is_processing": True, "id_task": 7, "params": {}}], 100.0, 1, 0, "slot-a")
        assert g["state"] in ("transitioning", "idle")


# --------------------------------------------------------------------------- #
# poller _tick — concurrency guards
# --------------------------------------------------------------------------- #
class TestPollerTick:
    async def test_loading_state_skips_slots_fetch(self):
        mgr = _Mgr()
        mgr._active_slot = _Slot(state="LOADING")
        mgr._active_handle = _Handle()
        p = _poller(mgr)
        p._client = _Client([_proc_slot()])
        await p._tick()
        assert mgr.live_generation["state"] == "loading"
        assert p._client.calls == 0

    async def test_torn_snapshot_handle_without_slot(self):
        mgr = _Mgr()
        mgr._active_slot = None
        mgr._active_handle = _Handle()  # lingering handle, slot already nulled
        p = _poller(mgr)
        p._client = _Client([_proc_slot()])
        await p._tick()
        assert mgr.live_generation["state"] == "transitioning"
        assert p._client.calls == 0

    async def test_idle_when_both_none(self):
        mgr = _Mgr()
        p = _poller(mgr)
        p._client = _Client([_proc_slot()])
        await p._tick()
        assert mgr.live_generation["state"] == "idle"
        assert p._client.calls == 0

    async def test_port_reuse_revalidate_skips(self):
        mgr = _Mgr()
        mgr._active_slot = _Slot(state="ACTIVE")
        original = _Handle(pid=1, port=11500)
        mgr._active_handle = original

        def swap():
            # a DIFFERENT sidecar grabbed port 11500 across the await
            mgr._active_handle = _Handle(pid=999, port=11500)
            mgr._spawn_seq += 1

        p = _poller(mgr)
        p._client = _Client([_proc_slot(n_decoded=123)], on_get=swap)
        await p._tick()
        assert mgr.live_generation["state"] == "transitioning"

    async def test_happy_path_publishes_tick(self):
        mgr = _Mgr()
        mgr._active_slot = _Slot(state="ACTIVE")
        mgr._active_handle = _Handle(pid=1, port=11500)
        p = _poller(mgr)
        p._client = _Client([_proc_slot(n_decoded=10)])
        await p._tick()
        await p._tick()
        assert mgr.live_generation["state"] in ("generating", "prefill")
        assert any(e.get("event") == "generation_tick" for e in mgr.event_bus.events)

    async def test_slots_error_survives_as_transitioning(self):
        mgr = _Mgr()
        mgr._active_slot = _Slot(state="ACTIVE")
        mgr._active_handle = _Handle()
        p = _poller(mgr)
        p._client = _Client(None, exc=httpx.ConnectError("boom"))
        await p._tick()
        assert mgr.live_generation["state"] == "transitioning"


# --------------------------------------------------------------------------- #
# LiveOutputBuffer — text plane
# --------------------------------------------------------------------------- #
class TestLiveOutputBuffer:
    def test_reframing_at_every_split_offset(self):
        frame = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        for cut in range(1, len(frame)):
            b = LiveOutputBuffer()
            b.feed("g", frame[:cut])
            b.feed("g", frame[cut:])
            _, tail, _ = b.subscribe("g")
            assert tail == "hello", f"cut={cut} -> {tail!r}"

    def test_multi_frame_and_done_and_keepalive(self):
        b = LiveOutputBuffer()
        chunk = (
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            b': keep-alive\n\n'
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        b.feed("g", chunk)
        _, tail, _ = b.subscribe("g")
        assert tail == "ab"

    def test_reasoning_content_captured(self):
        b = LiveOutputBuffer()
        b.feed("g", b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n')
        _, tail, _ = b.subscribe("g")
        assert tail == "think"

    def test_tool_call_string_arguments_captured(self):
        # The canonical OpenAI/llama.cpp streaming shape: function.name once,
        # function.arguments as a JSON STRING fragment.
        b = LiveOutputBuffer()
        b.feed(
            "g",
            b'data: {"choices":[{"delta":{"tool_calls":[{"function":'
            b'{"name":"search","arguments":"{\\"q\\":1}"}}]}}]}\n\n',
        )
        _, tail, _ = b.subscribe("g")
        assert "tool_call: search" in tail
        assert '{"q":1}' in tail

    def test_tool_call_dict_arguments_does_not_drop_frame(self):
        # llama.cpp issue #20198: some builds emit function.arguments as an
        # already-parsed JSON OBJECT (dict). A dict must NOT make the join raise
        # (which the fail-open except would silently swallow, dropping the whole
        # frame — the live symptom "tool calls show nothing"); it must render as
        # JSON text instead.
        b = LiveOutputBuffer()
        b.feed(
            "g",
            b'data: {"choices":[{"delta":{"tool_calls":[{"function":'
            b'{"name":"search","arguments":{"q":"hi"}}}]}}]}\n\n',
        )
        _, tail, _ = b.subscribe("g")
        assert "tool_call: search" in tail
        assert '"q": "hi"' in tail  # json.dumps default rendering

    def test_dict_tool_argument_does_not_eat_sibling_content(self):
        # A dict-arg tool_call sharing a frame with content must not drop the
        # content via a join TypeError on the whole pieces list.
        b = LiveOutputBuffer()
        b.feed(
            "g",
            b'data: {"choices":[{"delta":{"content":"keep","tool_calls":[{"function":'
            b'{"name":"f","arguments":{"x":1}}}]}}]}\n\n',
        )
        _, tail, _ = b.subscribe("g")
        assert "keep" in tail
        assert "tool_call: f" in tail

    def test_parallel_keys_isolated(self):
        b = LiveOutputBuffer()
        b.feed("g1", b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n')
        b.feed("g2", b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n')
        assert b.subscribe("g1")[1] == "one"
        assert b.subscribe("g2")[1] == "two"

    def test_lru_cap(self):
        b = LiveOutputBuffer()
        for i in range(MAX_LIVE_KEYS + 4):
            b.feed(f"g{i}", b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        assert len(b._buffers) <= MAX_LIVE_KEYS

    def test_tail_is_bounded(self):
        b = LiveOutputBuffer(tail_bytes=10)
        for _ in range(50):
            b.feed("g", b'data: {"choices":[{"delta":{"content":"abcde"}}]}\n\n')
        _, tail, _ = b.subscribe("g")
        assert len(tail) <= 10

    def test_garbage_never_raises(self):
        b = LiveOutputBuffer()
        b.feed("g", b"not an sse frame at all")
        b.feed("g", b"data: not-json\n\n")
        b.feed("g", b'data: {"choices": "wrong-shape"}\n\n')
        _, tail, _ = b.subscribe("g")
        assert tail == ""

    async def test_subscribe_delivers_deltas(self):
        b = LiveOutputBuffer()
        q, tail, done = b.subscribe("g", allow_create=True)
        assert tail == "" and done is False
        b.feed("g", b'data: {"choices":[{"delta":{"content":"X"}}]}\n\n')
        piece = await asyncio.wait_for(q.get(), timeout=1.0)
        assert piece == "X"

    async def test_mark_done_pushes_sentinel(self):
        b = LiveOutputBuffer()
        q, _, _ = b.subscribe("g", allow_create=True)
        b.mark_done("g")
        piece = await asyncio.wait_for(q.get(), timeout=1.0)
        assert piece is None

    def test_subscribe_unknown_gid_does_not_create(self):
        # DoS guard: an arbitrary client gid can NOT allocate a buffer
        b = LiveOutputBuffer()
        q, tail, done = b.subscribe("never-fed")
        assert q is None and done is True
        assert len(b._buffers) == 0

    def test_subscribe_allow_create(self):
        b = LiveOutputBuffer()
        q, _, _ = b.subscribe("anchor", allow_create=True)
        assert q is not None
        assert "anchor" in b._buffers

    def test_subscriber_cap(self):
        b = LiveOutputBuffer()
        b.feed("g", b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        qs = [b.subscribe("g")[0] for _ in range(MAX_SUBS_PER_GEN)]
        assert all(q is not None for q in qs)
        assert b.subscribe("g")[0] is None  # cap hit

    def test_unsubscribe_frees_completed_buffer(self):
        b = LiveOutputBuffer()
        q, _, _ = b.subscribe("g", allow_create=True)
        b.feed("g", b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        b.mark_done("g")
        assert "g" in b._buffers       # kept while a watcher is present
        b.unsubscribe("g", q)
        assert "g" not in b._buffers   # freed once the last watcher leaves

    def test_carry_is_bounded(self):
        b = LiveOutputBuffer()
        b.feed("g", b"x" * 200_000)  # no frame delimiter
        assert len(b._buffers["g"].carry) <= 65536


# --------------------------------------------------------------------------- #
# identity unification across the two planes
# --------------------------------------------------------------------------- #
def test_generation_id_unified_and_deterministic():
    a = compute_generation_id(1582, 3, "slot-abc")
    b = compute_generation_id(1582, 3, "slot-abc")
    assert a == b
    assert len(a) == 8
    # spawn_seq disambiguates a fixed-port reuse
    assert compute_generation_id(1582, 4, "slot-abc") != a


# --------------------------------------------------------------------------- #
# API integration
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_test(tmp_path):
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
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestStatusGenerationBlock:
    def test_status_has_idle_generation_by_default(self, app_test):
        _, client = app_test
        body = client.get("/status").json()
        assert "generation" in body
        assert body["generation"]["state"] == "idle"
        assert body["generation"]["tok_s"] == 0.0

    def test_status_reflects_poller_written_generation(self, app_test):
        app, client = app_test
        app.state.manager.live_generation = {
            "state": "generating", "tok_s": 42.7, "n_decoded": 1820,
            "generation_id": "abc12345",
        }
        body = client.get("/status").json()
        assert body["generation"]["state"] == "generating"
        assert body["generation"]["tok_s"] == 42.7

    def test_status_snapshot_is_await_free(self, app_test):
        # status_snapshot() must be callable synchronously (no await) from any context
        app, _ = app_test
        snap = app.state.manager.status_snapshot()
        assert "generation" in snap


class TestOutputSSE:
    def test_no_active_generation_immediate_done(self, app_test):
        _, client = app_test
        with client.stream("GET", "/ui/live/output/stream") as r:
            body = r.read().decode()
        assert '"done": true' in body

    def test_replay_tail_and_done(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        gid = "feed1234"
        mgr.live_output.feed(gid, b'data: {"choices":[{"delta":{"content":"hello world"}}]}\n\n')
        mgr.live_output.mark_done(gid)
        with client.stream("GET", f"/ui/live/output/stream?generation_id={gid}") as r:
            body = r.read().decode()
        assert "hello world" in body
        assert '"done": true' in body


class TestWsGenerationTick:
    def test_generation_tick_is_contentless(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_json()  # connected
            mgr.event_bus.publish_nowait({"event": "generation_tick"})
            ev = ws.receive_json()
            assert ev == {"event": "generation_tick"}
            for k in ("prompt", "response", "context", "messages", "text"):
                assert k not in ev


def test_csp_allows_same_origin_sse():
    # EventSource('/ui/live/output/stream') is a same-origin 'self' connect.
    assert "connect-src 'self'" in _CSP_HEADER


class TestPollerLifecycle:
    def test_poller_starts_when_enabled(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, monitor_enabled=True)
        app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
        with TestClient(app):
            assert app.state.manager._live_poller_task is not None
        # after context exit the poller task is cancelled cleanly (no leak)

    def test_poller_absent_when_disabled(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, monitor_enabled=False)
        app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
        with TestClient(app):
            assert app.state.manager._live_poller_task is None


def _boot_runtime(tmp_path, *, monitor_enabled):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True, exist_ok=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake", default_port_base=59600),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(), pull=PullConfig(),
        monitor=MonitorConfig(enabled=monitor_enabled, poll_interval_s=1.0),
    )
    return boot, runtime
