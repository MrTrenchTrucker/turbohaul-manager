"""OpenAI `reasoning_effort` tests: a client that sends the field gets a
defined behaviour (an engine launch budget, or a 400) rather than having it
silently vanish.

This is a startup-time mechanism, not a per-request one: the engine is not designed
to take a changed think budget mid-conversation, and a request should not try to
fight that, because the think
budget can only be honoured at engine STARTUP. `reasoning_effort` does not
resolve to a per-request `thinking_budget_tokens` payload write at all --
it sets the engine's `--reasoning-budget` LAUNCH flag once, when a resident
is first spawned, and is LOCKED for that engine's entire life
(manager.py: `Resident.reasoning_budget_override`, `_reserve_and_start_
locked`, `_spawn_for_resident`).

FIVE rungs, not six -- `minimal` is not a rung and is not special-cased:
`low .25 / medium .50 / high .75 / xhigh 1.0 / max 1.25`. It falls into the
SAME 400 every other unrecognized value gets -- no new code path. See
`chat_completion.py`'s module-level comment above `_EFFORT_FRACTION` for the
full measured reasoning (why `minimal` could not be delivered safely on
every model, the `tbt=3 -> STOP` unexplained exception, the OpenAI-spec
deviation).

Split cleanly into TWO functions:
- `resolve_reasoning_effort(payload) -> str | None` -- validation ONLY.
  Case-folds, rejects non-string/unrecognized with 400, returns the
  validated ladder key (or None if absent). Never touches `payload`, never
  computes a token count.
- `resolve_reasoning_budget_for_spawn(effort_key, manifest_budget) -> int |
  None` -- the arithmetic, kept out of request-time entirely. Called ONCE,
  from `manager._reserve_and_start_locked`, against whichever request FIRST
  admits a new resident. No ceiling/floor concept at all (unlike a
  per-request design) -- there is no caller `max_tokens` in scope at spawn;
  the separate per-request clamp (`clamp_reasoning_budget_for_
  ceiling`, independent of this feature) still protects the answer floor on every
  individual request regardless of what got baked in at spawn.

⛔ NEVER `payload["thinking_budget_tokens"]` for reasoning_effort
-- that per-request write is the mechanism that runs away (measured, twice,
on two different levers -- see chat_completion.py's comment) and the entire
point of this design is to leave it alone. An explicit `thinking_budget_
tokens` the client sends is still forwarded verbatim via
`_COMMON_FORWARDED_KNOBS`, exactly as it is when reasoning_effort is absent --
independent of whatever `reasoning_effort` says. This design has NO
conflict-check between the two (there is no "resolved value" to
compare an explicit one against) -- noted explicitly at the tests below
whose premise this affects.

⛔ NEVER `payload["reasoning_budget"]` for reasoning_effort either -- that is
the per-request clamp's SEPARATE key.

⚠ FIRST CALLER WINS, and it must be said in the docs in plain
words, not just logged/status-surfaced: whoever's request
first spawns a model's engine fixes that model's `--reasoning-budget` for
the engine's entire life. A later request asking a DIFFERENT effort on the
same warm engine silently gets the first caller's value. Surfaced via a
WARNING log (`manager._warn_if_reasoning_effort_mismatch`, mirroring the
existing warn-once-and-cap style) and the per-resident status listing
(`reasoning_budget_override`, beside `main_gpu`/`split_mode`) -- but a
caller who isn't watching either needs telling outright, which is what this
comment and the docs are for.

Three tests below are pinned BY NAME: the dedup-ordering
pin, the two-key tripwire, and the forwarded-knobs-filter-is-`is not
None` tripwire (folded into the "never writes" tests below since `minimal`
does not resolve to an explicit 0 at all under this design -- there is
nothing for that specific tripwire to guard against; the omission is
deliberate).
"""
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import clamp_reasoning_budget_for_ceiling
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

SAMPLE_SHA = "a" * 64


@pytest.fixture(autouse=True)
def _clear_warn_cache():
    """The warn-once-per-(model,caller) cache is module-level global
    state (chat_completion._reasoning_budget_warned) shared across every test
    in this process. Same pattern as the reasoning-budget clamp test's
    fixture of the same name -- without it, an earlier test in THIS file using
    the same (model, caller) shape silently suppresses a later test's warning
    assertion."""
    from turbohaul.api import chat_completion as cc
    cache = getattr(cc, "_reasoning_budget_warned", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


# ============================================================================
# A1. resolve_reasoning_effort -- validation only, no arithmetic, no payload.
# ============================================================================


@pytest.fixture
def resolve_reasoning_effort():
    from turbohaul.api.chat_completion import resolve_reasoning_effort as f
    return f


class TestResolveReasoningEffortValidation:
    def test_absent_returns_none(self, resolve_reasoning_effort):
        assert resolve_reasoning_effort({}) is None

    def test_json_null_returns_none(self, resolve_reasoning_effort):
        # FastAPI decodes a JSON `null` body value to Python None -- same
        # .get() result and same code path as the field being absent.
        assert resolve_reasoning_effort({"reasoning_effort": None}) is None

    def test_non_string_raises_400(self, resolve_reasoning_effort):
        with pytest.raises(Exception) as exc_info:
            resolve_reasoning_effort({"reasoning_effort": 5})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "reasoning_effort_unsupported_value"
        assert exc_info.value.detail["received"] == "int"

    def test_minimal_is_no_longer_recognized_and_400s(self, resolve_reasoning_effort):
        """THE required RED test:
        `minimal` MUST 400 (a six-entry table would accept it). Not
        special-cased, not mapped to anything --
        it simply is not in the five-entry `_EFFORT_FRACTION`,
        so it falls into the exact same unrecognized-value 400
        every other bad string gets."""
        with pytest.raises(Exception) as exc_info:
            resolve_reasoning_effort({"reasoning_effort": "minimal"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "reasoning_effort_unsupported_value"
        assert exc_info.value.detail["received"] == "minimal"

    def test_unknown_string_raises_400(self, resolve_reasoning_effort):
        with pytest.raises(Exception) as exc_info:
            resolve_reasoning_effort({"reasoning_effort": "extreme"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "reasoning_effort_unsupported_value"

    def test_the_400_message_names_all_five_efforts_and_not_minimal(self, resolve_reasoning_effort):
        """Explicit per-name check, not a substring check (a substring check
        against a truncated list can pass for the wrong reason -- a known
        pitfall). Also asserts `minimal` is
        ABSENT from the message -- the strongest possible proof it was
        actually deleted, not merely left unlisted by accident."""
        with pytest.raises(Exception) as exc_info:
            resolve_reasoning_effort({"reasoning_effort": "extreme"})
        message = exc_info.value.detail["message"]
        for name in ("low", "medium", "high", "xhigh", "max"):
            assert name in message, (name, message)
        assert "minimal" not in message, message

    def test_case_fold_HIGH_accepted(self, resolve_reasoning_effort):
        # A case-sensitive match would itself be a silent-drop generator.
        # Returns the case-folded KEY, not a resolved number.
        assert resolve_reasoning_effort({"reasoning_effort": "HIGH"}) == "high"

    def test_all_five_recognized_values_round_trip_case_folded(self, resolve_reasoning_effort):
        for raw, expected_key in (
            ("low", "low"), ("MEDIUM", "medium"), ("High", "high"),
            ("XHIGH", "xhigh"), (" max ", "max"),
        ):
            assert resolve_reasoning_effort({"reasoning_effort": raw}) == expected_key, raw


# ============================================================================
# A2. resolve_reasoning_budget_for_spawn -- the arithmetic, moved to spawn
# time. Callable in isolation, no spawn/Resident/manager needed at all.
# ============================================================================


@pytest.fixture
def resolve_spawn_budget():
    from turbohaul.api.chat_completion import resolve_reasoning_budget_for_spawn as f
    return f


class TestResolveReasoningBudgetForSpawn:
    def test_each_rung_at_budget_800(self, resolve_spawn_budget):
        for effort_key, expected in (
            ("low", 200), ("medium", 400), ("high", 600), ("xhigh", 800), ("max", 1000),
        ):
            assert resolve_spawn_budget(effort_key, 800) == expected, effort_key

    def test_round_up_not_to_nearest(self, resolve_spawn_budget):
        """Rounding proof: 0.25*801=200.25
        -- round-half gives 200 (wrong), ceil gives 201. Exactly ONE token
        difference, asserted as the specific value (not >=200), so a
        regression back to round() -- which would silently pass a >= check
        -- fails this outright instead."""
        assert resolve_spawn_budget("low", 801) == 201

    def test_none_when_effort_key_is_none(self, resolve_spawn_budget):
        assert resolve_spawn_budget(None, 800) is None

    def test_none_when_effort_key_unrecognized(self, resolve_spawn_budget):
        # Includes "minimal" itself -- defence in depth: even if something
        # upstream ever let a deleted/unknown key reach this function
        # directly (bypassing resolve_reasoning_effort's own validation),
        # it still safely no-ops rather than resolving to a fabricated
        # number.
        for bad_key in ("minimal", "extreme", ""):
            assert resolve_spawn_budget(bad_key, 800) is None, bad_key

    def test_none_when_manifest_budget_non_positive(self, resolve_spawn_budget):
        assert resolve_spawn_budget("high", 0) is None
        assert resolve_spawn_budget("high", -1) is None

    def test_none_when_manifest_budget_not_an_int(self, resolve_spawn_budget):
        # bool is an int subclass in Python -- must be explicitly excluded,
        # not merely rely on isinstance(x, int) alone.
        assert resolve_spawn_budget("high", True) is None
        assert resolve_spawn_budget("high", "800") is None
        assert resolve_spawn_budget("high", None) is None

    def test_max_exceeds_the_manifest_budget_unconditionally(self, resolve_spawn_budget):
        """Unlike a per-request design, this function has
        NO ceiling/floor concept at all -- there is no caller max_tokens in
        scope at spawn time, a resident outlives any one request. `max`
        (1.25) therefore ALWAYS exceeds the manifest's own budget when
        resolved here; whatever protects an individual request's answer
        room is the entirely separate per-request clamp, applied
        downstream of this value, not by this function."""
        assert resolve_spawn_budget("max", 800) == 1000 > 800


# ============================================================================
# B. End-to-end: the ACTUAL client_meta a request produces, and (separately)
# the ACTUAL argv a first-caller spawn produces. Two different layers --
# capturing client_meta asserts what reaches the manager per request;
# capturing argv (see the class below) asserts what reaches the engine at
# spawn. Neither on its own proves the ENGINE honoured the value -- that is
# a real-engine smoke test's job, not this suite's.
# ============================================================================


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str, reasoning_budget, n_predict=None):
    flags_lines = [f"  reasoning_budget: {reasoning_budget}"]
    if n_predict is not None:
        flags_lines.append(f"  n_predict: {n_predict}")
    (manifests_root / f"{tag}.yaml").write_text(
        "model_tag: " + tag + "\n"
        'gguf_blob_sha256: "' + SAMPLE_SHA + '"\n'
        "gguf_size_bytes: 1000\n"
        "llama_server_flags:\n" + "\n".join(flags_lines) + "\n"
    )


@pytest.fixture
def effort_e2e_app(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    manifests = storage_root / "manifests"
    # no-ceiling shape: 800, no n_predict -> ceiling None w/o a caller cap.
    _write_manifest_yaml(manifests, "effort-model", reasoning_budget=800)
    # off / unbounded.
    _write_manifest_yaml(manifests, "off-model", reasoning_budget=0)
    _write_manifest_yaml(manifests, "unbounded-model", reasoning_budget=-1)
    # the exact pathological shape the per-request clamp targets (reused from the
    # reasoning-budget clamp test) for the two-key tripwire below.
    _write_manifest_yaml(manifests, "clamp-model", reasoning_budget=3072)

    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=manifests,
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
            # max_parallel_sidecars>=2 is REQUIRED, not incidental.
            # worker_loop branches on this at cap>=2 into _dispatch_loop ->
            # _drive_resident -> _spawn_for_resident, which is where this feature's
            # reasoning_budget_override lives. At the DEFAULT cap<=1,
            # worker_loop instead calls _process_slot directly -- a
            # completely separate argv-build with NO override of any kind
            # (not even the main_gpu/split_mode override), so this whole
            # feature would be silently inert under that path. Deployed
            # configurations commonly run an EFFECTIVE cap >=2
            # even when a config default/yaml might say otherwise -- but
            # this is called out regardless, not assumed silently.
            max_parallel_sidecars=2,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    captured_client_meta = {}
    captured_argv = {}

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        # Capture argv too (list, not just discarded) so spawn
        # tests below can inspect --reasoning-budget PROVENANCE.
        captured_argv.setdefault(model_tag, []).append(list(argv))
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        captured_client_meta["last"] = dict(slot.client_meta or {})
        messages = (slot.client_meta or {}).get("messages") or []
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1700000000,
            "model": slot.model_tag,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo: {last_user}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client, mgr, captured_client_meta, captured_argv


class TestOpenAIRouteEndToEnd:
    def test_minimal_400s_end_to_end(self, effort_e2e_app):
        """Route-level twin of test_minimal_is_no_longer_recognized_and_400s
        -- the RED this change is watched against: a design where minimal
        resolves to an explicit 0 would pass it; `minimal` must 400."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "minimal",
            },
        )
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"] == "reasoning_effort_unsupported_value"

    def test_reasoning_effort_never_writes_thinking_budget_tokens_into_payload(self, effort_e2e_app):
        """There are no per-value thinking_budget_tokens
        assertions (a per-value check would be vacuous) -- under
        this design NO value ever reaches client_meta's thinking_budget_
        tokens key via reasoning_effort, for any of the five rungs. The
        actual resolved value is observable on the SPAWNED engine's argv
        instead (see TestSpawnArgvObservesReasoningBudgetOverride below)."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "effort-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": effort,
                },
            )
            assert r.status_code == 200, r.text
            assert "thinking_budget_tokens" not in captured["last"], (effort, captured["last"])

    def test_reasoning_effort_key_carried_in_client_meta_case_folded(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "HIGH",
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"]["reasoning_effort"] == "high"

    def test_absent_field_reasoning_effort_is_none_not_missing(self, effort_e2e_app):
        """When the client sends no reasoning_effort, client_meta still
        carries the "reasoning_effort" key explicitly (so the manager can
        read it at spawn-admission time even when None), with value
        None -- the key is not missing from client_meta. It is
        always present by design;
        thinking_budget_tokens absence is the load-bearing half of this
        guard."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={"model": "effort-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert "thinking_budget_tokens" not in captured["last"], captured["last"]
        assert captured["last"]["reasoning_effort"] is None, captured["last"]

    def test_off_model_byte_identical(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "off-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200, r.text
        assert "thinking_budget_tokens" not in captured["last"], captured["last"]

    def test_unbounded_model_byte_identical(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "unbounded-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200, r.text
        assert "thinking_budget_tokens" not in captured["last"], captured["last"]

    def test_unknown_value_400(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "extreme",
            },
        )
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"] == "reasoning_effort_unsupported_value"

    def test_explicit_thinking_budget_tokens_not_overwritten(self, effort_e2e_app):
        """Unaffected by this feature: thinking_budget_tokens sent WITHOUT
        reasoning_effort is forwarded verbatim via _COMMON_FORWARDED_KNOBS,
        exactly as when reasoning_effort is absent."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking_budget_tokens": 777,
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"]["thinking_budget_tokens"] == 777

    def test_both_reasoning_effort_and_explicit_thinking_budget_tokens_coexist(self, effort_e2e_app):
        """The two fields are INDEPENDENT. reasoning_effort does not
        resolve to a comparable value at request time at all, so there
        is no "matching explicit value" case to treat as a no-op.
        The explicit thinking_budget_tokens value
        is forwarded verbatim (same as if reasoning_effort had not
        been sent), and reasoning_effort separately only affects a NEW
        resident's spawn-time argv. Both can be sent together; neither
        overwrites the other; there is no comparison between them.
        Sending both is not an error and neither value is dropped.
        """
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
                "thinking_budget_tokens": 12345,
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"]["thinking_budget_tokens"] == 12345
        assert captured["last"]["reasoning_effort"] == "high"

    def test_both_sent_no_longer_conflict_checked(self, effort_e2e_app):
        """⚠ NO CONFLICT CHECK, by design:
        a request with both reasoning_effort and an explicit
        thinking_budget_tokens is not rejected, even when they would
        resolve to DIFFERENT values. Such a comparison does not exist:
        reasoning_effort has no per-request resolved value to compare
        against. A client sending both together, even with wildly different
        intents, gets 200 -- the explicit value wins (forwarded
        verbatim) and reasoning_effort only ever affects a NEW resident's
        spawn. This is a deliberate behavioural narrowing compared with a
        per-request conflict check, a consequence of doing the arithmetic
        at spawn time."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
                "thinking_budget_tokens": 1,
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"]["thinking_budget_tokens"] == 1

    def test_never_writes_the_dead_reasoning_budget_key(self, effort_e2e_app):
        """reasoning_effort must never touch payload["reasoning_budget"] -- that
        stays the per-request clamp's key."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200, r.text
        assert "reasoning_budget" not in captured["last"], captured["last"]


class TestOllamaRouteEndToEnd:
    def test_reasoning_effort_never_writes_thinking_budget_tokens_either(self, effort_e2e_app):
        """Route parity: there is no per-request ceiling capping
        reasoning_effort's resolved value on this route either -- there is no ceiling
        concept in
        resolve_reasoning_effort/resolve_reasoning_budget_for_spawn, so a
        num_predict option cannot cap it. This
        proves route parity on the one guarantee that holds: the
        Ollama route also never writes thinking_budget_tokens per request."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/api/chat",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
                "options": {"num_predict": 100},
                "stream": False,
            },
        )
        assert r.status_code == 200, r.text
        assert "thinking_budget_tokens" not in captured["last"], captured["last"]
        assert captured["last"]["reasoning_effort"] == "high"

    def test_unknown_value_400_via_ollama_route(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/api/chat",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "extreme",
                "stream": False,
            },
        )
        assert r.status_code == 400, r.text


# ============================================================================
# C. Spawn-argv observability + first-caller-wins. ⛔ PROVENANCE, NOT EFFECT:
# these tests assert the manager COMPUTED and PASSED the right
# --reasoning-budget value into argv. They do NOT prove the engine honoured
# it -- asserting that would be conflating provenance with effect. A
# real-engine smoke test is what measures effect.
# ============================================================================


def _reasoning_budget_from_argv(argv: list) -> "str | None":
    if "--reasoning-budget" not in argv:
        return None
    idx = argv.index("--reasoning-budget")
    return argv[idx + 1] if idx + 1 < len(argv) else None


class TestSpawnArgvObservesReasoningBudgetOverride:
    def test_first_request_reasoning_effort_overrides_manifest_value_in_argv(self, effort_e2e_app):
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200, r.text
        argv = captured_argv["effort-model"][0]
        # PROVENANCE only: the manager computed and passed ceil(0.75*800)=600
        # into argv. Whether llama-server honours it is a real-engine smoke test's job.
        assert _reasoning_budget_from_argv(argv) == "600", argv

    def test_no_reasoning_effort_manifest_value_reaches_argv_unmodified(self, effort_e2e_app):
        """Byte-identical path: no reasoning_effort asked -> the manifest's
        own reasoning_budget (800) reaches argv completely untouched, same
        as without this feature."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={"model": "effort-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        argv = captured_argv["effort-model"][0]
        assert _reasoning_budget_from_argv(argv) == "800", argv

    def test_first_caller_wins_second_differing_request_does_not_respawn(self, effort_e2e_app):
        """⚠ FIRST CALLER WINS, structurally proven, not just inspected:
        first request asks "high" (600, spawns once). Second request to the
        SAME model asks "low" (200) -- the resident is already warm, so NO
        second spawn happens at all (captured_argv stays at exactly one
        entry for this model) and the locked-in argv value from the FIRST
        request is what's already running.

        Requires a positive idle-hot window: the shared fixture's
        grace_seconds=0/idle_hot_load_seconds=0 (tuned for OTHER tests'
        fast-teardown assertions) tears the resident down essentially
        instantly after its one request, so two sequential synchronous
        TestClient calls would otherwise each trigger their OWN spawn --
        that IS what a zero idle window looks like, not first-caller-wins
        breaking. Set here, locally, so the resident survives long
        enough for the second request to reach it warm."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        mgr.runtime.queue.idle_hot_load_seconds = 30
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r1.status_code == 200, r1.text
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi again"}],
                "reasoning_effort": "low",
            },
        )
        assert r2.status_code == 200, r2.text
        # Exactly ONE spawn for this model -- the second request was served
        # by the already-warm resident, never re-derived its own argv.
        assert len(captured_argv["effort-model"]) == 1, captured_argv["effort-model"]
        assert _reasoning_budget_from_argv(captured_argv["effort-model"][0]) == "600"

    def test_first_caller_wins_mismatch_is_logged(self, effort_e2e_app, caplog):
        """The log half of the disclosure: a second request
        asking a DIFFERENT effort than what's locked in gets a WARNING
        naming both values, mirroring the existing warn-once style.

        Same positive idle-hot window bump as the sibling test above, and
        for the same reason -- the second request must reach the FIRST
        request's still-warm resident to exercise the mismatch check at
        all."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        mgr.runtime.queue.idle_hot_load_seconds = 30
        client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
            r2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "effort-model",
                    "messages": [{"role": "user", "content": "hi again"}],
                    "reasoning_effort": "low",
                },
            )
        assert r2.status_code == 200, r2.text
        mismatch_lines = [
            rec for rec in caplog.records
            if "first-caller-wins" in rec.getMessage()
        ]
        assert len(mismatch_lines) == 1, caplog.text

    def test_resident_status_surfaces_reasoning_budget_override(self, effort_e2e_app):
        """The status-field half of the disclosure: an
        operator reads the same locked-in value from the per-resident
        status listing, beside main_gpu/split_mode -- same shape, same
        precedent."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        # The fixture's zero idle window unloads the resident right after the
        # response, before this test reads the registry; keep it warm instead.
        mgr.runtime.queue.idle_hot_load_seconds = 30
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "effort-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        assert r.status_code == 200, r.text
        resident = mgr._residents.get("effort-model")
        assert resident is not None
        assert resident.reasoning_budget_override == 600
        assert resident.reasoning_effort_locked == "high"


# ============================================================================
# D. Tests pinned BY NAME.
# ============================================================================


class TestOrderingPinsTheDedupKey:
    def test_two_requests_differing_only_in_reasoning_effort_do_not_collide(
        self, effort_e2e_app,
    ):
        """Pinned BY NAME: two requests identical except for
        reasoning_effort must not collide in the result dedup.

        The dedup keys differ because the "reasoning_effort" string differs
        in client_meta (no per-request thinking_budget_tokens value is
        involved). The test reads the
        actual client_meta each request produces and calls the real dedup-
        key method, not a hand-built stand-in. What
        differs in
        client_meta is the "reasoning_effort" string itself
        ("low" vs "high"), which is exactly what has to reach client_meta
        distinctly per request for manager._reserve_and_start_locked to
        resolve the FIRST caller's effort correctly at admission time. If
        this ever stopped differing, two different efforts would not just
        share a dedup key -- they'd also be indistinguishable to the
        spawn-time resolver."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        messages = [{"role": "user", "content": "same prompt, different effort"}]

        client.post("/v1/chat/completions", json={
            "model": "effort-model", "messages": messages, "reasoning_effort": "low",
        })
        meta_low = dict(captured["last"])
        assert meta_low["reasoning_effort"] == "low"

        client.post("/v1/chat/completions", json={
            "model": "effort-model", "messages": messages, "reasoning_effort": "high",
        })
        meta_high = dict(captured["last"])
        assert meta_high["reasoning_effort"] == "high"

        key_low = mgr._completion_cache_key("effort-model", "", messages, meta_low)
        key_high = mgr._completion_cache_key("effort-model", "", messages, meta_high)
        assert key_low is not None and key_high is not None
        assert key_low != key_high, (
            "two requests differing only in reasoning_effort produced the "
            "SAME dedup key -- the reasoning_effort key is not reaching "
            "client_meta distinctly per request, so a cached answer for one "
            "effort level would be silently served for another"
        )


class TestTwoKeyStateTripwire:
    def test_both_clamp_fires_and_effort_no_longer_auto_writes_the_second_key(
        self, effort_e2e_app,
    ):
        """Pinned BY NAME, and it is deliberately a TRIPWIRE aimed at
        whoever next changes the dead-key behaviour.

        The two-key state: the per-request clamp is real, per-request, and
        independent of this feature; there is no second per-request key for
        reasoning_effort to write at all, so the check is that the clamp
        fires and reasoning_effort does not also write a key. (A design that
        had reasoning_effort write
        `payload["thinking_budget_tokens"]` per request would need two
        DIFFERENT numbers on the same request, so a passing
        assertion couldn't be explained by one copying the other.) The
        tripwire is: the clamp fires and
        writes `reasoning_budget`, AND reasoning_effort's
        mechanism does NOT ALSO write into `thinking_budget_tokens` on this
        same pathological-ceiling request -- the two mechanisms are
        separated by LAYER (the per-request clamp vs. the one-time
        spawn flag), not merely by key. The moment someone re-introduces a
        per-request thinking_budget_tokens write for reasoning_effort, this
        test goes red, forcing the reconciliation with the clamp back in front
        of them instead of leaving it to their diligence -- the same
        protective intent, applied to the current architecture."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "clamp-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2048,
                "reasoning_effort": "low",
            },
        )
        assert r.status_code == 200, r.text
        last = captured["last"]
        assert "reasoning_budget" in last, last
        assert last["reasoning_budget"] == 1024, last
        assert "thinking_budget_tokens" not in last, last
        assert last["reasoning_effort"] == "low", last


# ============================================================================
# E. Warning coexistence + logging non-interference (regression guard).
# ============================================================================


class TestNoInterferenceWithReasoningBudgetCeilingWarning:
    def test_warning_still_fires_alongside_effort_resolution(self, caplog, effort_e2e_app):
        """Unaffected by this feature: reasoning_effort's validation call still
        runs after the existing warning in program order; nothing about moving
        the ARITHMETIC to spawn time changes that call order."""
        app, client, mgr, captured, captured_argv = effort_e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clamp-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2048,
                    "reasoning_effort": "low",
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text
