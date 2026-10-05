"""KV ANTI-REWIND -- token-based guard tests.

Covers the guard added ahead of every existing branch in
``_maybe_force_clean_restore``:
  * Token-based anti-rewind: never force-restore a bin whose
    TOKEN count is less than the live warm slot's CURRENT TOKEN position,
    sourced from a direct ``/slots`` GET at decision time (never
    ``mgr.live_generation`` / the ~1Hz poller cache, whose idle-state
    token count is not meaningful).
  * ``_find_clean_bin``'s plumbed-out ``prompt_len`` (CHARACTERS) and
    ``prompt_tokens`` (TOKENS) — both computed by the scan and
    surfaced to callers so the guard can use them.

Note on the spawn-age check: it (bin-age-vs-process-spawn) is intentionally NOT
part of the guard — it would decline a legitimately-valid restore on every ordinary
idle-out-then-resume, because bins are written only at unload so their mtime
always predates a fresh engine spawn. See the section near the bottom of this
file for the full written reasoning and the tests that guard its absence: a healthy-flow
acceptance proof plus a test that derives ``warm_chain`` through the real
``_engine_view_chain`` production function instead of hand-building it.

UNITS: ``prompt_len`` is a CHARACTER count (``compute_ctx_len``); the engine's
``/slots`` ``n_prompt_tokens`` and the bin meta's ``prompt_tokens`` are TOKEN
counts, roughly 3x smaller on real prompts. The token guard's comparison MUST use
``prompt_tokens``, never ``prompt_len`` — a version that
compared ``prompt_len`` against a token count would still pass every test
if every fixture set both to consistent-but-arbitrary values
instead of deliberately DIVERGENT, realistic ones. Every test below sets
``prompt_len`` and ``prompt_tokens`` to independent values (roughly the real
~3x ratio) chosen so that comparing the WRONG field would flip the verdict —
a discriminator that only proves "some number was compared" is vacuous in the
unit dimension even when it is non-vacuous in every other dimension.

Each real invariant gets a DISCRIMINATOR (proves the guard declines when it
should) and a CONTROL (proves force-restore still fires on a genuinely valid
case — a guard that declines universally is a kill-switch in a fix's
clothes, and passes a naive discriminator perfectly).
"""
import json
import os
import time
from types import SimpleNamespace

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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot


# --- fixtures (mirror test_classifier.py's own local copies) -----------------
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Isolated SLOT_SAVE_DIR (re-imported per call inside the manager methods)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _warm_force_on(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}
_QWEN = "qwen-27b"  # forced-restore is gated to the qwen MTP family


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain,
                      prompt_len=120000, prompt_tokens=40000):
    """Write a pinned clean bin (.bin + clean_prefix .json meta) for a thread.

    prompt_len (CHARACTERS) and prompt_tokens (TOKENS) are DELIBERATELY
    separate parameters with different default magnitudes (~3x apart,
    matching the real ratio) — never derive one from the other here. A
    fixture that quietly keeps them equal, or coupled, would hide exactly the
    unit bug this file exists to catch."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


def _think_strip_scenario(kv_dir, prompt_len=120000, prompt_tokens=40000, sid=0):
    """The warm-diverges win shape (test_classifier.py's own control case): warm KV
    holds the <think> turn, incoming is think-stripped -> clean bin IS a valid
    prefix, warm diverges -> the OLD gate alone would force. Used as the base
    scenario for every discriminator+control pair below, so a
    decline can ONLY be attributed to the new guard, never to the pre-existing
    NO-DOWNGRADE gate."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", sid, clean_chain,
                               prompt_len, prompt_tokens)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    return bin_fn, slot, warm


# --- fake httpx: GET /slots (anti-rewind) + POST action=restore (existing gate) ------
class _FakeGetResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakePostResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _FakeSlotsClient:
    def __init__(self, state):
        self._state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        self._state["get_calls"].append(url)
        return _FakeGetResp(self._state["slots"], self._state["slots_status"])

    async def post(self, url, json=None, **kw):
        self._state["posts"].append((url, json))
        return _FakePostResp()


@pytest.fixture
def net(monkeypatch):
    """Controls the fake engine's GET /slots payload + records restore POSTs."""
    state = {"slots": [], "slots_status": 200, "posts": [], "get_calls": []}

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _FakeSlotsClient(state))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return state


def _slots_payload(sid, n_prompt_tokens):
    return [{"id": sid, "n_prompt_tokens": n_prompt_tokens, "is_processing": False}]


# =====================================================================
# Token-based anti-rewind, sourced from a direct /slots GET
# =====================================================================
@pytest.mark.asyncio
async def test_discriminator_declines_when_bin_is_behind_the_live_slot(mgr, kv_dir, net):
    """DISCRIMINATOR (unit-honest): the bin has FEWER TOKENS than the live
    warm slot's CURRENT position -> the guard must decline and keep the warm
    slot, even though the pre-existing NO-DOWNGRADE gate alone would have
    forced here (see _think_strip_scenario). prompt_len (chars) is
    DELIBERATELY set to equal the live token count -- if the comparison ever
    regresses to reading prompt_len instead of prompt_tokens, this becomes
    `50000 < 50000` = False = wrongly NOT declining, so this test would catch
    that regression, not just pass vacuously either way."""
    bin_fn, slot, warm = _think_strip_scenario(kv_dir, prompt_len=50000, prompt_tokens=16000, sid=0)
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=50000)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is False
    assert d["resolved_from"] == "anti-rewind-token-skip"
    assert d["clean_bin_id"] == bin_fn
    assert net["posts"] == []  # the warm slot was never touched


@pytest.mark.asyncio
async def test_control_still_forces_on_a_genuinely_fresher_bin(mgr, kv_dir, net):
    """CONTROL: the bin has MORE tokens than the live warm slot's current
    position -> the guard must NOT decline; the pre-existing gate still forces
    exactly as before. Without this test, a guard that declines
    unconditionally would pass the discriminator above perfectly."""
    bin_fn, slot, warm = _think_strip_scenario(kv_dir, prompt_len=120000, prompt_tokens=40000, sid=0)
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=1000)  # bin is well ahead
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"
    assert len(net["posts"]) == 1
    url, body = net["posts"][0]
    assert "action=restore" in url and body == {"filename": bin_fn}


@pytest.mark.asyncio
async def test_live_shaped_regression_2turn_39433_vs_70325_warm_survives(mgr, kv_dir, net):
    """★ Realistic regression signature AND the
    unit discriminator that catches the char/token defect (numbers verified
    from a real bin: prompt_len ~3.01x prompt_tokens).
    A 2-turn / 39,433-TOKEN bin meets a 70,325-TOKEN warm slot -- the warm
    slot must survive. prompt_len is realistically ~3x inflated (118,299
    chars) -- reading THAT instead of prompt_tokens gives `118299 < 70325` =
    False = wrongly NOT declining, which is exactly the char/token defect and would let the 31k
    reprefill keep happening.
    THIS is the test that must fail against a char-based comparison --
    it was verified to fail against a char-based implementation."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    assert len(clean_chain) == 2
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain,
                               prompt_len=118299, prompt_tokens=39433)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=70325)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is False
    assert d["resolved_from"] == "anti-rewind-token-skip"
    assert d["clean_bin_id"] == bin_fn
    assert net["posts"] == []


@pytest.mark.asyncio
async def test_probe_failure_never_declines_safe_degrade(mgr, kv_dir, net):
    """A /slots probe failure (non-200) must NEVER fabricate a decline --
    safe-degrade to today's behavior, exactly as if the guard didn't exist."""
    bin_fn, slot, warm = _think_strip_scenario(kv_dir, prompt_len=120000, prompt_tokens=40000, sid=0)
    net["slots_status"] = 500
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"
    assert len(net["posts"]) == 1


@pytest.mark.asyncio
async def test_slot_id_not_found_never_declines_safe_degrade(mgr, kv_dir, net):
    """The engine's slot id in /slots doesn't match this bin's sid (e.g. a
    resident swap mid-flight) -> unknown position -> never declines."""
    bin_fn, slot, warm = _think_strip_scenario(kv_dir, prompt_len=120000, prompt_tokens=40000, sid=0)
    net["slots"] = _slots_payload(sid=7, n_prompt_tokens=99999)  # wrong id
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True


@pytest.mark.asyncio
async def test_missing_prompt_tokens_never_declines_safe_degrade(mgr, kv_dir, net):
    """MISSING/ZERO prompt_tokens (older bins predating the field) -> None ->
    fail-open, DO NOT decline -- and, critically, do NOT fall back to
    prompt_len (which would silently reintroduce the char/token bug under a
    different name). prompt_len is deliberately tiny here (1000, far below
    the live slot) so a prompt_len-fallback would (wrongly) decline; the
    correct fail-open behavior must NOT decline regardless."""
    bin_fn = _write_clean_bin(
        kv_dir, _QWEN, 59500, "t", 0, _prefix_hash_chain([_SYS, _U1]),
        prompt_len=1000, prompt_tokens=0,
    )
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=50000)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"


@pytest.mark.asyncio
async def test_live_slot_token_position_direct_helper(mgr, net):
    """Unit-level: _live_slot_token_position reads n_prompt_tokens off the
    /slots entry matching `id`, via a direct GET issued at call time."""
    net["slots"] = _slots_payload(sid=3, n_prompt_tokens=12345)
    got = await mgr._live_slot_token_position(59500, 3)
    assert got == 12345
    assert net["get_calls"] and net["get_calls"][0].endswith(":59500/slots")


@pytest.mark.asyncio
async def test_live_slot_token_position_missing_slot_is_none(mgr, net):
    net["slots"] = _slots_payload(sid=3, n_prompt_tokens=12345)
    assert await mgr._live_slot_token_position(59500, 99) is None


# --- guardrail: never re-source from the poller cache ---------------------------
def test_live_slot_tokens_source_never_reads_live_generation_cache():
    """PERMANENT GUARDRAIL: live_slot_tokens must be sourced from a direct
    /slots GET at decision time, never mgr.live_generation / the ~1Hz poller
    cache -- whose idle-state n_prompt_tokens is a hardcoded 0 (the poller
    cache reports 0 when idle): a 0 makes `bin_tokens < 0` always
    False, so the guard would silently never fire in exactly the between-tool-
    calls window this fix exists for. Someone will 'optimize' the direct GET
    away in six months -- this is the only thing that stops them."""
    import inspect
    src = inspect.getsource(TurbohaulManager._live_slot_token_position)
    assert "live_generation" not in src
    assert "live_generations" not in src


@pytest.mark.asyncio
async def test_live_slot_tokens_ignores_a_poisoned_live_generation_cache(mgr, net):
    """Behavioral twin of the guardrail above: even when mgr.live_generation
    is set to the EXACT poller-idle shape that broke the guard's premise
    (n_prompt_tokens=0), the direct-GET result is unaffected."""
    mgr.live_generation = {"state": "idle", "n_prompt_tokens": 0}
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=70325)
    got = await mgr._live_slot_token_position(59500, 0)
    assert got == 70325


# =====================================================================
# Note: the spawn-age check is intentionally ABSENT, because
# identity-scoped lookup + atomic
# writes + content-hash chain validity already rule out its stated
# threat model; nothing about "bin identity" closes that gap without also
# reintroducing it for the ordinary case. These tests guard that choice:
# (1) proves the healthy flow it would wrongly reject --
# ordinary idle-out-then-resume -- is accepted, and (2) closes the
# construct-vs-derive gap by deriving `warm_chain`
# through the REAL production function (`_engine_view_chain`) from a
# realistic completion/streaming shape, instead of hand-building it via
# `_prefix_hash_chain` on an arbitrary literal list the way every test
# above this section still does for its OWN chain (the token-check tests stay
# hand-built deliberately -- they are not what's being examined here, and
# touching their proven-sound tests would be needless churn).
# "What does this test CONSTRUCT that production DERIVES?" is the
# question behind both tests below.
# =====================================================================
@pytest.mark.asyncio
async def test_ordinary_idle_out_then_resume_now_accepted_nonstreaming(mgr, kv_dir, net):
    """THE HEALTHY-FLOW PROOF, driven through a REAL derivation, not a
    hand-built one. Simulates the ordinary case that the spawn-age check would decline on
    every occurrence: a clean bin written a long time ago (mtime artificially
    aged, exactly what an unload-time save looks like by the time a session
    resumes), no special engine-handle state assumed (the removed check's
    only input was `_active_handle`/spawn time -- proving this scenario
    force-restores with NO active handle set at all proves the removal left
    no residual dependency on it). `warm_chain` is DERIVED via the real
    `_engine_view_chain(slot, result)` from a realistic non-streaming
    completion result shape (the exact dict `_complete_fn` produces and the
    production call site at manager.py's non-streaming warm-followup branch
    passes to this same function), not constructed via `_prefix_hash_chain`
    on a literal list the way the pre-existing token-check tests do for warm_chain.

    CAUSAL: this test FAILS against an implementation that carries the spawn-age check (an aged-mtime bin
    with no active handle still safe-degrades under its `_spawn_wall
    is None` branch -- so to fail this specific way, see the
    streaming-derivation twin below, which uses an explicit stale handle;
    both directions are covered so the absence is proven from both angles)."""
    messages = [_SYS, _U1]
    clean_chain = _prefix_hash_chain(messages)
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain,
                              prompt_len=120000, prompt_tokens=40000)
    inc = _prefix_hash_chain(messages + [{"role": "assistant", "content": "A"}, _U2])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc,
                    client_meta={"messages": messages})
    old = time.time() - 3600 * 24 * 7  # a week old -- an ordinary idled-out session
    os.utime(kv_dir / bin_fn, (old, old))
    result = {"choices": [{"message": {"content": "<think>x</think>A"}}]}
    warm_chain = mgr._engine_view_chain(slot, result)
    assert warm_chain, "sanity: the realistic result must actually derive a non-empty chain"
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=1000)  # bin well ahead -- the anti-rewind guard lets it through
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm_chain)
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"
    assert len(net["posts"]) == 1
    url, body = net["posts"][0]
    assert "action=restore" in url and body == {"filename": bin_fn}


@pytest.mark.asyncio
async def test_ordinary_idle_out_then_resume_now_accepted_streaming_with_stale_handle(
        mgr, kv_dir, net):
    """Same healthy-flow proof, but (a) derives warm_chain via the STREAMING
    shape (`slot.streamed_assistant_text`, matching the OTHER real call site
    at the streaming warm-followup branch, `result=None`), and (b) sets an
    EXPLICIT active handle whose spawn time postdates the aged bin by a wide
    margin -- the exact configuration that makes the spawn-age check fire with
    `resolved_from='anti-rewind-stale-process'`. This is the test that FAILS
    against code that carries the spawn-age check and proves the
    absence of the behavior, not just of a symbol."""
    messages = [_SYS, _U1]
    clean_chain = _prefix_hash_chain(messages)
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain,
                              prompt_len=120000, prompt_tokens=40000)
    inc = _prefix_hash_chain(messages + [{"role": "assistant", "content": "A"}, _U2])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc,
                    client_meta={"messages": messages})
    old = time.time() - 3600 * 24 * 7
    os.utime(kv_dir / bin_fn, (old, old))
    slot.streamed_assistant_text = "<think>x</think>A"
    warm_chain = mgr._engine_view_chain(slot, None)
    assert warm_chain, "sanity: the streamed-text path must derive a non-empty chain too"
    mgr._active_handle = SimpleNamespace(port=59500, spawned_at_wall=time.time())
    net["slots"] = _slots_payload(sid=0, n_prompt_tokens=1000)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm_chain)
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"
    assert len(net["posts"]) == 1
    url, body = net["posts"][0]
    assert "action=restore" in url and body == {"filename": bin_fn}


def test_engine_spawn_wall_time_helper_removed():
    """Source-level guardrail: the spawn-age-only helper must be gone, not left
    as dead code nobody calls -- an orphaned "correct but unreachable"
    function is exactly the kind of citation-without-liveness debt that
    invites someone to wire it back in without re-deriving why it was wrong."""
    assert not hasattr(TurbohaulManager, "_engine_spawn_wall_time")


def test_sidecar_handle_has_no_spawned_at_wall_field():
    """Companion guardrail on the subprocess_mgr.py side: SidecarHandle must
    not carry the now-orphaned field either -- removed cleanly on both ends,
    not just made unreachable from one side."""
    handle = subprocess_mgr.SidecarHandle(proc=SimpleNamespace(pid=1, poll=lambda: None),
                                           port=1, model_tag="m")
    assert not hasattr(handle, "spawned_at_wall")


# =====================================================================
# _find_clean_bin: prompt_len plumbed out (computed by the scan and surfaced to callers)
# =====================================================================
def test_find_clean_bin_returns_prompt_len_and_prompt_tokens(mgr, kv_dir):
    """Both fields surface, distinctly, in their own positions -- prompt_len
    (chars) at index 3, prompt_tokens (tokens)
    at index 4. Deliberately different values so a swap or
    off-by-one in the tuple shape shows up as a wrong-value assertion, not a
    coincidental pass."""
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, "m", 59500, "t", 0, chain,
                               prompt_len=118299, prompt_tokens=39433)
    found = mgr._find_clean_bin(59500, "m", "t")
    assert found is not None
    assert len(found) == 5
    assert found[0] == bin_fn and found[1] == chain and found[2] == 0
    assert found[3] == 118299  # prompt_len (chars)
    assert found[4] == 39433   # prompt_tokens (tokens)
