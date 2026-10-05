"""THE TOOL-CALL LATENCY REGRESSION GUARD.

⛔⛔ READ THIS BEFORE YOU CHANGE ANYTHING IN THIS FILE ⛔⛔

THE PROTECTION CLAUSE:
    This test may be changed ONLY to make the system FASTER, and ONLY with
    proven results. It may NOT be changed to make a build pass. It may NOT be
    changed because it is inconvenient. IF THIS TEST GOES RED, THE CHANGE IS
    THE SUSPECT, NOT THE TEST.

Weakening this file gives back tens of seconds per agent turn and nobody will
notice for a long time, because the thing it protects is invisible from a green
suite. The full story, in prose, for a reader new to the problem, is in the
tool-call latency guard design notes. Read them before editing this file.

-----------------------------------------------------------------------------
THE REGRESSION THIS GUARDS AGAINST, IN FOUR LINES
-----------------------------------------------------------------------------
A latency regression of this kind shows up as a large rise in the median time
an agent waits for its first token while prefill compute does NOT change.
The time is MANAGER-SIDE ADMISSION, ahead of the engine, so it is not a
host-wide effect: a control model on the same host stays at the same
latency before and after.

-----------------------------------------------------------------------------
⭐ WHY THIS FILE COUNTS THINGS INSTEAD OF TIMING THEM
-----------------------------------------------------------------------------
This is the single most important design decision here, and it is the one a
future reader is most likely to want to "fix". Do not.

The manager cannot burn tens of seconds by being slightly slow. Every park on the
admission path is bounded by `_DISPATCH_DEFER_BACKOFF_S = 0.05` (manager.py).
To lose that long at 50ms a request would have to go round hundreds of times.
It loses that time by doing UNNECESSARY WORK, and in this setup that work has a
known unit price: ONE COLD MODEL LOAD, tens of seconds for a large GGUF.

Measuring the two populations directly says exactly this:
    COLD requests (needed a fresh slot assign) : high median
    WARM requests (reused a live slot)         : much lower median
    cold traffic share                         : small
The gap between cold and warm is about one cold model load, the whole
regression in a single unit. The improvement was NOT achieved by making
anything faster. It was achieved by STOPPING DOING SOMETHING: residents being
torn down and reloaded. The mechanism indicator agrees --
`grace_designated_victim_skip` (a counter of teardowns) drops sharply once the
fix is in place: a very large drop in teardowns.

So the guard counts units of work. Not milliseconds. Two consequences:

  1. It cannot flake. There is no clock in any assertion, so a loaded CI machine,
     a slow disk and a fast laptop all produce the same integers.

  2. ⭐ IT CANNOT BE QUIETLY WEAKENED. THERE IS NO CONSTANT IN IT TO BUMP.
     YOU CANNOT LOOSEN ZERO. A duration budget would have a number in it, and
     the first time it flaked someone would raise that number, truthfully
     report "the test still passes", and the protection would be gone with
     nobody having lied. That is the exact failure mode this file exists to be
     immune to. (Precedent: a 3.0s wall-clock
     margin PASSED against unfixed, buggy code because environment overhead
     beat the margin. A timing guard is the shape most likely to end up
     weakened, which makes it the wrong shape for the one test that must never
     be weakened.)

-----------------------------------------------------------------------------
THE FOUR COUNTS
-----------------------------------------------------------------------------
C4  the TTFT yardstick fires on the path production actually runs.
C1  a request that can be served warm triggers ZERO cold loads.
C2  a request that can be served warm triggers ZERO teardowns.
C3  the warm admission path executes ZERO unconditional sleeps.

C4 is FIRST on purpose. Without this guard, `on_first_token` -- the emitter that
produced the first-token measurement, and the only instrument that can DATE a
future regression -- would have ZERO references anywhere in the test suite. Nothing
would go red if it were deleted, renamed, made conditional, or moved out of the
streaming loop. Every other assertion here protects the win; C4 protects our
ability to KNOW we lost it.

-----------------------------------------------------------------------------
⚠ THE ANTI-VACUITY RULE THAT MAKES THE ZEROS MEAN ANYTHING
-----------------------------------------------------------------------------
"No work happened" is also true when NOTHING happened -- a refused request, a
fixture that never reached the code, a typo in a model tag. Every zero-count
assertion in this file is therefore paired with a POSITIVE assertion that the
work which was supposed to happen did happen (the slot was actually routed to
the warm resident / the stream actually produced a token). Never assert a zero
here without its positive partner. A guard that passes because it did nothing
is worse than no guard, because it reports safety.

-----------------------------------------------------------------------------
WHAT THIS FILE DELIBERATELY DOES NOT ASSERT
-----------------------------------------------------------------------------
The dispatcher's wake-on-enqueue behaviour ("woken, not timed out") is ALREADY
covered and this file does not restate it. That coverage lives in:
    tests/test_manager.py                          :: TestDispatchLoopWake
    the eviction-teardown wake test                :: TestVramOvercommitAdmissionWake
    the third-moment wake test                       :: TestThirdMomentWake
    the turn-boundary handoff test                 :: TestParkFiresNotifyAndDeferredRetryWakesEarly
⛔ THOSE FOUR ARE PART OF THIS PROTECTION. Deleting one of them removes wake
coverage that this file was deliberately not written to duplicate.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest \
        <this file> -v
"""
from __future__ import annotations

import asyncio
import traceback
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import turbohaul.telemetry as telemetry_module
from turbohaul.api.main import create_app
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
from turbohaul.subprocess_mgr import SidecarHandle

_MODEL = "test-model"

# An agent's conversation, modelled honestly. thread_id is auto-derived from a
# hash of the first 256 PROMPT TOKENS (slot.derive_thread_id_prefix_hash), so a
# conversation that keeps its system prompt and appends turns keeps ONE
# thread_id and can match a warm resident's grace window. A test that sent a
# different short prompt each turn would derive a DIFFERENT thread_id each
# turn, never match, and would be measuring a crowd of strangers rather than
# one agent's tool-call loop. The prefix is deliberately >256 tokens.
_AGENT_PREFIX = " ".join(f"sysword{i}" for i in range(300))


def _agent_turn_prompt(n: int) -> str:
    """Turn N of ONE conversation: same prefix, appended tail."""
    return f"{_AGENT_PREFIX} tail{n}"


def _other_conversation_prompt(k: int, n: int = 0) -> str:
    """Turn N of conversation K -- a DIFFERENT agent, same model.

    MEASURED, and it is why both shapes are exercised below: a
    warm turn reaches the resident by one of TWO different routes.
      * A CONTINUING conversation (same thread_id) is picked up by the
        resident's own grace window and never re-enters the admission
        decision at all -- `_route_or_reserve` runs ZERO times.
      * A NEW conversation on an already-resident model DOES go through
        `_route_or_reserve`, which finds the live resident and routes to it.
    Both must cost zero work. Only the second exercises admission, so C3
    (which is about the admission path) must use this one or it measures
    nothing.
    """
    return " ".join(f"conv{k}word{i}" for i in range(300)) + f" tail{n}"

# One SSE frame carrying real content, then the terminator. The first non-empty
# chunk is what makes `on_first_token` fire in chat_completion.py's byte loop.
_SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


# ---------------------------------------------------------------------------
# A fake sidecar stream. The real route opens httpx.AsyncClient(...).stream(...)
# to llama-server; there is no llama-server in a unit test, so the open would
# fail and the route would emit an error frame -- which never reaches the
# first-token branch. This fake makes real bytes arrive so the emitter is
# genuinely REACHED rather than merely present.
# ---------------------------------------------------------------------------
class _FakeStreamResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b""


class _FakeStreamCM:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return _FakeStreamResponse(self._chunks)

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *a, **k):
        return _FakeStreamCM(_SSE_CHUNKS)


# The frames that DEFINE "the admission path" for C3. `_route_or_reserve` is
# the admission decision itself (manager.py): HIT-route / evict-and-reserve /
# defer, all under one _registry_lock critical section. Because an `await`
# chain inside a single task is a real Python call stack, any sleep executed
# BENEATH this frame -- in it or in anything it calls -- is attributed to
# admission. A sleep in a DETACHED background task it spawned is not (a known
# limit of this approach).
_ADMISSION_FRAMES = frozenset({"_route_or_reserve"})


def _count_admission_entries(monkeypatch, mgr):
    """Count real entries into `_route_or_reserve`.

    ⚠ THIS EXISTS BECAUSE C3 WAS VACUOUS WITHOUT IT. A naive version of C3 drove
    a continuing conversation and asserted "no sleep ran under admission" --
    which passed, because on a continuing conversation `_route_or_reserve`
    NEVER RUNS. The recorder was correct and the scenario was wrong, and the
    two are indistinguishable from a green. Any C3-style assertion must prove
    the mechanism it is judging actually executed.
    """
    calls = {"n": 0}
    cls = type(mgr)
    real = cls._route_or_reserve

    async def counting(self, *args, **kwargs):
        calls["n"] += 1
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_route_or_reserve", counting)
    return calls


class _SleepRecorder:
    """Records every positive-delay asyncio.sleep executed under admission.

    A count, not a clock: this never asserts how LONG anything took, only how
    MANY unconditional sleeps ran where none should. `asyncio.sleep(0)` is a
    bare yield to the event loop, costs nothing, and is deliberately ignored.
    """

    def __init__(self):
        self.calls: list[tuple[float, str]] = []

    def install(self, monkeypatch):
        real_sleep = asyncio.sleep

        async def recording_sleep(delay=0, *args, **kwargs):
            if delay and delay > 0:
                stack = traceback.extract_stack()
                names = {frame.name for frame in stack}
                if names & _ADMISSION_FRAMES:
                    where = " <- ".join(
                        f"{f.name}:{f.lineno}"
                        for f in stack
                        if f.name in _ADMISSION_FRAMES or f.name == "recording_sleep"
                    )
                    self.calls.append((delay, where))
            return await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", recording_sleep)


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


class _WorkCounters:
    """The instrument. Every field is an integer count of a unit of work.

    `spawns` is the cold-load counter: `mgr._spawn` is the injected seam every
    sidecar launch goes through (manager.py `self._spawn = spawn_fn or
    spawn_sidecar`). `sigterms` and `vram_verifies` are the teardown counters,
    on the same injected-seam principle.
    """

    def __init__(self):
        self.spawns = 0
        self.sigterms = 0
        self.vram_verifies = 0

    @property
    def teardowns(self) -> int:
        return self.sigterms + self.vram_verifies

    def __repr__(self):
        return (
            f"<work spawns={self.spawns} sigterms={self.sigterms} "
            f"vram_verifies={self.vram_verifies}>"
        )


def _boot_and_runtime(tmp_path, *, grace_seconds=30):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    _write_manifest_yaml(storage_root / "manifests", _MODEL)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            # ⛔ max_parallel_sidecars=2 IS LOAD-BEARING, NOT INCIDENTAL.
            # QueueConfig's default is 1, and worker_loop forks on
            # `max_parallel_sidecars >= 2` into _dispatch_loop. Production runs
            # cap>=2; the cap<=1 singleton path is deprecated.
            # Leaving the default here would drive the whole
            # guard down the deprecated path -- which is EXACTLY the trap that
            # can leave on_slot_assign / on_prefill_start / on_completion stranded
            # off the production path. A guard that
            # runs on the path production does not use guards nothing.
            # If you lower this to 1, this file stops protecting anything.
            max_parallel_sidecars=2,
            safety_enabled=False,
            # ⛔ grace_seconds AND idle_hot_load_seconds ARE ALSO LOAD-BEARING.
            # These are what keep a resident alive between an agent's tool
            # calls. MEASURED on this harness: with
            # grace_seconds=0 every single turn spawns a fresh sidecar and
            # tears the last one down -- spawns 1,2,3,4 across four turns,
            # which IS the thrash that costs tens of seconds a turn. With grace
            # enabled the same four turns spawn exactly ONCE.
            # So a config of 0 here does not "simplify the fixture", it
            # reproduces the regression and makes C1/C2 unsatisfiable.
            # MEASURED: grace_seconds is ALSO how long a NEW
            # conversation waits before it can be admitted to a resident whose
            # current grace window still protects the incumbent thread -- one
            # wait of grace_seconds per new conversation. At 30s the two
            # cross-conversation tests took a minute or more. They therefore use a
            # SHORT grace (see app_short_grace); the same-conversation tests
            # keep a long one, because for THEM a short grace would be a
            # timing dependency: a follow-up turn arriving after the window
            # lapsed would be a genuine cold load and the test would flake on
            # a slow machine. Each fixture takes the value its scenario needs.
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=300,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _app_with_work_counters(tmp_path, *, grace_seconds):
    """The app, with every unit of work counted at its injected seam.

    ⚠ `turbohaul.telemetry.init_telemetry` is a MODULE-LEVEL SINGLETON
    (`global _telemetry`, idempotent), so every manager built in one pytest
    process shares ONE FlapTelemetry unless it is reset first. Reset BEFORE
    building the manager and AFTER the test, and filter reads by slot_id --
    otherwise this test reads another test's events. Same discipline as
    the telemetry-parity test.
    """
    telemetry_module._telemetry = None
    boot, runtime = _boot_and_runtime(tmp_path, grace_seconds=grace_seconds)
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager
    work = _WorkCounters()

    def counting_spawn(binary, gguf, port, model_tag, argv, **_kw):
        work.spawns += 1
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def counting_sigterm(handle, **kwargs):
        work.sigterms += 1
        return True, "sigterm-clean"

    async def counting_vram(**kwargs):
        work.vram_verifies += 1
        return True, 100

    async def fake_complete(slot, handle):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": slot.model_tag,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = counting_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = counting_sigterm
    mgr._vram_verify = counting_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client, mgr, work

    telemetry_module._telemetry = None


@pytest.fixture
def app_with_work_counters(tmp_path):
    """Long grace. For CONTINUING-conversation tests and C4.

    A long window here is the anti-flake choice: a follow-up turn that arrives
    after the window lapsed is a genuine cold load, so a short grace would make
    these tests depend on how fast the machine is.
    """
    yield from _app_with_work_counters(tmp_path, grace_seconds=30)


@pytest.fixture
def app_short_grace(tmp_path):
    """Short grace. For NEW-conversation tests only.

    A new conversation cannot be admitted to a resident while that resident's
    grace window still protects the incumbent thread, so it waits out the
    window -- measured at exactly grace_seconds per new conversation. The
    counts under test (spawns, teardowns) do not depend on the length of that
    wait, only on what happens after it, so a short window measures the same
    property in 2 seconds instead of 30. This is a RUNTIME choice, not a
    weakening: no assertion anywhere in this file reads a clock.
    """
    yield from _app_with_work_counters(tmp_path, grace_seconds=2)


def _stream_once(client, *, prompt="say hi"):
    """Drive ONE real streaming request through the real route."""
    body = b""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
    ) as r:
        status = r.status_code
        for chunk in r.iter_bytes():
            body += chunk
    return status, body


def _first_token_events(mgr):
    """Every `first_token` event this manager's telemetry holds.

    Filtered by event_type at the source; the fixture's singleton reset is what
    keeps another test's events out of the ring buffer.
    """
    result = mgr._telemetry.get_events(event_type="first_token", limit=1000)
    return result.get("events", [])


# ===========================================================================
# C4 -- THE YARDSTICK. FIRST, because nothing else enforces it.
# ===========================================================================
class TestC4TheTtftYardstickFiresOnTheProductionPath:
    """`telemetry.on_first_token` is the instrument that produced the
    first-token measurement, and the ONLY one that can date a future
    regression.

    ⛔ Without this guard it would have ZERO references in the entire test suite. Its
    three siblings (on_slot_assign / on_prefill_start / on_completion) are
    covered by the telemetry-parity test; this one is not.
    Nothing would go red if it were deleted, renamed, made conditional, or moved
    out of the streaming byte loop.

    An emitter EXISTING is not the same as an emitter FIRING on the path
    production runs -- sibling emitters can sit stranded on the cap<=1 path
    while cap>=2 is what actually serves traffic.
    That is why these tests drive the REAL route end to end and read the
    REAL event, rather than asserting the call site is present in the source.
    """

    def test_a_streaming_request_emits_exactly_one_first_token_event(
        self, app_with_work_counters, monkeypatch
    ):
        """THE YARDSTICK FIRES. Drive a real streaming request whose sidecar
        returns real bytes, and require the `first_token` event to exist.

        If this goes red, the TTFT instrument is broken or unreachable and we
        have lost the ability to date any future latency regression at all.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        status, body = _stream_once(client)

        # POSITIVE PARTNER (anti-vacuity): the stream really produced content.
        # Without this, a route that 500s or emits only an error frame would
        # leave zero events and the assertion below would be meaningless.
        assert status == 200, f"streaming route did not return 200: {status}"
        assert b"hi" in body, (
            "the fake sidecar's token never reached the client, so the "
            f"first-token branch was never entered; body={body!r}"
        )

        events = _first_token_events(mgr)
        assert len(events) == 1, (
            "expected exactly ONE first_token telemetry event for one "
            f"streaming request, got {len(events)}: {events!r}. "
            "This is the yardstick that dates a latency step change; "
            "if it stops firing we cannot date the next regression."
        )

    def test_the_first_token_event_carries_a_usable_ttft_ms(
        self, app_with_work_counters, monkeypatch
    ):
        """The event must carry the NUMBER, not just exist.

        An event that fires with no `ttft_ms` is a heartbeat, not a
        measurement -- it would keep this file green while making the
        telemetry useless for exactly the comparison it was used for.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        status, body = _stream_once(client)
        assert status == 200 and b"hi" in body, "positive partner failed"

        events = _first_token_events(mgr)
        assert events, "no first_token event at all"
        payload = events[0]
        assert "ttft_ms" in payload, (
            f"first_token event has no ttft_ms field: {payload!r}"
        )
        assert isinstance(payload["ttft_ms"], (int, float)), (
            f"ttft_ms is not a number: {payload['ttft_ms']!r}"
        )
        assert payload["ttft_ms"] >= 0, f"negative ttft_ms: {payload!r}"

    def test_the_event_identifies_which_slot_it_measured(
        self, app_with_work_counters, monkeypatch
    ):
        """A TTFT number that cannot be attributed to a slot cannot be split
        into the COLD and WARM populations, and that split (cold vs warm)
        is the entire justification for this file counting work instead of
        timing it. The join key has to survive too.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        status, body = _stream_once(client)
        assert status == 200 and b"hi" in body, "positive partner failed"

        events = _first_token_events(mgr)
        assert events, "no first_token event at all"
        payload = events[0]
        assert payload.get("slot_id"), (
            f"first_token event carries no slot_id: {payload!r}"
        )
        assert payload.get("model_tag") == _MODEL, (
            f"first_token event mis-attributes the model: {payload!r}"
        )


# ===========================================================================
# C1 + C2 -- THE WIN ITSELF. A warm turn does no work.
# ===========================================================================
class TestC1C2AWarmTurnCostsNoColdLoadAndNoTeardown:
    """The agent tool-call loop: one conversation, several turns, one model.

    Turn 0 pays for ONE cold load, because the model genuinely was not
    resident. Every turn after it must reuse that resident and cost NOTHING:
    no second spawn, no teardown. That difference -- a cold request vs a
    warm one, i.e. one model load -- is the whole improvement.

    ⭐ TURN 0 IS THE INSTRUMENT CONTROL AND IT IS NOT DECORATION. The exact
    counter that must read 0 across turns 1..N is PROVEN to reach 1 on turn 0,
    in the same test, on the same fixture. Without it, `spawns == 0` would
    also pass if the seam were never wired, if the fixture never reached the
    manager, or if the model tag were misspelled -- a green from an unwired
    probe is byte-identical to a green from real warm reuse.
    """

    def _run_turns(self, client, mgr, work, n_turns):
        seen = []
        for i in range(n_turns):
            status, body = _stream_once(client, prompt=_agent_turn_prompt(i))
            # POSITIVE PARTNER, every single turn: the request was actually
            # SERVED. A refused or 500-ing turn also does no work.
            assert status == 200, f"turn {i} did not return 200: {status}"
            assert b"hi" in body, (
                f"turn {i} produced no token, so it was never really served; "
                f"body={body!r}"
            )
            seen.append((i, work.spawns, work.sigterms, work.vram_verifies))
        return seen

    def test_turn_zero_really_does_cold_load_so_the_counter_is_proven_live(
        self, app_with_work_counters, monkeypatch
    ):
        """THE INSTRUMENT CONTROL. Run this before believing any zero below."""
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        self._run_turns(client, mgr, work, 1)

        assert work.spawns >= 1, (
            "turn 0 did NOT cold-load. The spawn counter is not wired to "
            "anything, so every `spawns == 0` assertion in this file would "
            "pass vacuously. Fix the harness before trusting this file."
        )

    def test_c1_a_warm_turn_triggers_zero_cold_loads(
        self, app_with_work_counters, monkeypatch
    ):
        """⭐ C1. THE COLD-LOAD COST. A cold load costs tens of seconds for a large GGUF; that
        single unit is the entire cold/warm gap. If this goes red, agents are
        paying a model load per tool call again.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        rows = self._run_turns(client, mgr, work, 4)
        spawns_after_turn_zero = rows[0][1]
        spawns_at_end = rows[-1][1]

        assert spawns_at_end == spawns_after_turn_zero, (
            "A WARM TURN COLD-LOADED. Turns 1..3 of ONE conversation on ONE "
            f"model added {spawns_at_end - spawns_after_turn_zero} sidecar "
            f"spawn(s) on top of turn 0's {spawns_after_turn_zero}. "
            f"Per-turn counts: {rows}. "
            "Each of those is tens of seconds of model load an agent now pays between "
            "tool calls. This is the latency regression returning. "
            "READ the latency guard design notes -- DO NOT WEAKEN THIS TEST."
        )

    def test_c2_a_warm_turn_triggers_zero_teardowns(
        self, app_with_work_counters, monkeypatch
    ):
        """⭐ C2. The mechanism indicator. `grace_designated_victim_skip` -- a
        resident denied its grace window and torn down -- is the counter that
        drops sharply once this is fixed. Teardowns are what CAUSE
        the cold loads C1 counts, so this catches the same regression one step
        earlier, at its source.
        """
        app, client, mgr, work = app_with_work_counters
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        rows = self._run_turns(client, mgr, work, 4)
        teardowns_after_turn_zero = rows[0][2] + rows[0][3]
        teardowns_at_end = rows[-1][2] + rows[-1][3]

        assert teardowns_at_end == teardowns_after_turn_zero, (
            "A WARM TURN TORE DOWN A RESIDENT. Turns 1..3 of one conversation "
            f"added {teardowns_at_end - teardowns_after_turn_zero} teardown(s). "
            f"Per-turn counts (turn, spawns, sigterms, vram_verifies): {rows}. "
            "A resident torn down between an agent's tool calls has to be "
            "reloaded for the next one -- this is the thrash that costs tens of seconds a "
            "turn. READ the latency guard design notes."
        )


# ===========================================================================
# C3 -- NO UNCONDITIONAL SLEEP ON THE WARM ADMISSION PATH.
# ===========================================================================
class TestC3TheWarmAdmissionPathSleepsForNothing:
    """The "somebody quietly adds an await" shape.

    ⛔ SCOPE, DELIBERATELY NARROW. This class asserts ONLY that no
    unconditional positive-delay sleep runs under `_route_or_reserve` on a
    WARM turn. It does NOT assert the dispatcher's wake-on-enqueue behaviour
    ("woken, not timed out") -- that is ALREADY covered by
    tests/test_manager.py::TestDispatchLoopWake and the three other wake classes
    named in this module's docstring, and restating their coverage here would
    let this file claim protection it did not actually add. Those four tests
    are part of this protection; do not delete them.

    ⛔ WARM ONLY, and that is a real constraint, not laziness. The eviction
    path legitimately sleeps: `verify_vram_cleared` takes a one-time
    `settle_floor_s` (default 5.0, subprocess_mgr.py) before polling, because
    VRAM does not physically begin dropping for >2s after an unload, so those
    samples are guaranteed misses. Three of the four manager call sites pass
    0.0; the default is kept at exactly one (_unload_teardown) where it gates a
    notify a waiting claimant depends on. Widening this class to the eviction
    path would red on landed, deliberate code -- a guard that fails on correct
    work teaches people to disable guards.
    """

    def test_c3_a_warm_turn_executes_no_unconditional_sleep(
        self, app_short_grace, monkeypatch
    ):
        """⭐ C3. Recording starts AFTER the cold load, because turn 0 is
        allowed to sleep -- it is loading a model.
        """
        app, client, mgr, work = app_short_grace
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        # Turn 0: the cold load. Not measured -- loading a model is allowed to
        # take time; that is the work C1 exists to keep rare, not to forbid.
        status, body = _stream_once(client, prompt=_agent_turn_prompt(0))
        assert status == 200 and b"hi" in body, "turn 0 (cold) was not served"

        entries = _count_admission_entries(monkeypatch, mgr)
        recorder = _SleepRecorder()
        recorder.install(monkeypatch)

        # NEW conversations on the now-resident model: these DO go through
        # `_route_or_reserve`. A continuing conversation would not, and this
        # test would then be judging a function that never ran.
        for k in (1, 2):
            status, body = _stream_once(client, prompt=_other_conversation_prompt(k))
            # POSITIVE PARTNER: a turn that never ran also never sleeps.
            assert status == 200, f"warm turn {k} did not return 200: {status}"
            assert b"hi" in body, (
                f"warm turn {k} produced no token, so admission was never "
                f"exercised and this assertion would be vacuous; body={body!r}"
            )

        # ⭐ SECOND POSITIVE PARTNER, and the one that keeps C3 honest: the
        # admission decision genuinely executed inside the recorded window.
        assert entries["n"] >= 1, (
            "`_route_or_reserve` never ran during the recorded window, so "
            "'no sleep under admission' is trivially true and this test is "
            f"measuring nothing. entries={entries['n']}. Fix the scenario, "
            "not the assertion."
        )

        assert recorder.calls == [], (
            "AN UNCONDITIONAL SLEEP RAN ON THE WARM ADMISSION PATH. "
            f"Recorded: {recorder.calls!r}. "
            "Every one of these is latency added to every warm tool call, "
            "ahead of the engine, invisible to prefill metrics. If you added "
            "it deliberately, it does not belong under _route_or_reserve on "
            "the warm path -- move it off the admission path or make it "
            "conditional. READ the latency guard design notes."
        )

    def test_the_sleep_recorder_actually_records_so_the_empty_list_means_something(
        self, app_with_work_counters, monkeypatch
    ):
        """INSTRUMENT CONTROL for C3, and it is essential.

        `recorder.calls == []` is exactly what a BROKEN recorder produces. So
        prove the recorder fires: call asyncio.sleep from inside a frame named
        `_route_or_reserve` and require it to be caught. This shares no code
        path with the manager -- it is a pure instrument check.
        """
        recorder = _SleepRecorder()
        recorder.install(monkeypatch)

        async def _route_or_reserve():  # noqa: N802 - named to match the real frame
            await asyncio.sleep(0.01)

        async def _not_an_admission_frame():
            await asyncio.sleep(0.01)

        asyncio.run(_route_or_reserve())
        assert len(recorder.calls) == 1, (
            "the sleep recorder did not catch a sleep executed inside a frame "
            f"named _route_or_reserve; calls={recorder.calls!r}. Every C3 "
            "green is vacuous until this passes."
        )

        # ...and does not fire for a sleep OUTSIDE the admission frames, so it
        # is a filter rather than a blanket "any sleep anywhere" alarm.
        asyncio.run(_not_an_admission_frame())
        assert len(recorder.calls) == 1, (
            "the recorder attributed a NON-admission sleep to admission; "
            f"calls={recorder.calls!r}"
        )


# ===========================================================================
# C1 + C2, SECOND SHAPE -- a new conversation joining an already-resident model.
# ===========================================================================
class TestC1C2ANewConversationJoinsAWarmResidentForFree:
    """The other way a warm turn happens, and the one that goes through the
    admission decision.

    A second agent (or the same agent starting a new conversation) asks for a
    model that is ALREADY resident. That request is routed by
    `_route_or_reserve`, which must find the live resident and hand the slot
    to it -- no new sidecar, no teardown of the one already there.

    Getting this wrong is exactly the thrash: two conversations on
    one model taking turns evicting each other, each paying a cold load of tens of seconds
    to get back what the other just threw away.
    """

    def test_a_second_conversation_costs_no_spawn_and_no_teardown(
        self, app_short_grace, monkeypatch
    ):
        app, client, mgr, work = app_short_grace
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        status, body = _stream_once(client, prompt=_agent_turn_prompt(0))
        assert status == 200 and b"hi" in body, "the cold load turn failed"
        spawns_after_cold = work.spawns
        teardowns_after_cold = work.teardowns
        assert spawns_after_cold >= 1, (
            "INSTRUMENT CONTROL: the cold turn did not spawn, so the counter "
            "is unwired and every zero below is vacuous."
        )

        entries = _count_admission_entries(monkeypatch, mgr)
        for k in (1, 2, 3):
            status, body = _stream_once(client, prompt=_other_conversation_prompt(k))
            assert status == 200, f"conversation {k} did not return 200: {status}"
            assert b"hi" in body, (
                f"conversation {k} produced no token; body={body!r}"
            )

        assert entries["n"] >= 1, (
            "no new conversation reached `_route_or_reserve`, so this test is "
            "not exercising the admission decision it claims to guard"
        )
        assert work.spawns == spawns_after_cold, (
            "A NEW CONVERSATION COLD-LOADED A MODEL THAT WAS ALREADY RESIDENT. "
            f"spawns went {spawns_after_cold} -> {work.spawns} across three "
            "conversations on one resident model. Each extra spawn is tens of seconds "
            "an agent waits for a model that was already in memory. "
            "READ the latency guard design notes -- DO NOT WEAKEN THIS TEST."
        )
        assert work.teardowns == teardowns_after_cold, (
            "A NEW CONVERSATION TORE DOWN A LIVE RESIDENT INSTEAD OF SHARING "
            f"IT. teardowns went {teardowns_after_cold} -> {work.teardowns}. "
            "This is two conversations evicting each other -- the thrash that "
            "costs tens of seconds a turn. READ the latency guard design notes."
        )
