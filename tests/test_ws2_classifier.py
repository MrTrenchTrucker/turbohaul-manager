"""KV RESTORE CLASSIFIER — manager-side prefix-classified restore.

Covers the pieces that live in manager.py (the gate itself is unit-tested in
test_kv_policy.py::test_restore_* — prefix-valid / diverged / physics-belt / empty-
chain-skip). Here:
  * _generated_assistant_msg  — reconstruct the engine's warm turn (with <think>),
    safe-degrade to None on tool-calls / noop / streamed.
  * _engine_view_chain        — warm-state chain = prompt + generated turn.
  * _classify_event           — the 4 events (+ guard-skip).
  * _find_clean_bin           — locate the pinned clean anchor.
  * _maybe_force_clean_restore — the NO-DOWNGRADE gate: FORCE only when it improves
    reuse; never regress a true continuation; POST action=restore exactly when force.
"""
import json

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


# --- fixtures --------------------------------------------------------------------
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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
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
    """The WARM forced clean-bin restore is DEFAULT OFF. This module exercises the
    force-GATE logic itself (WHEN it forces / no-downgrade / classification), which is
    the ROLLBACK path — so enable it for every test here. The dedicated default-OFF
    gate test overrides this with delenv()."""
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


class _FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _FakeClient:
    """Records every POST so tests can assert the forced restore fired (or not)."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _FakeResp()


@pytest.fixture
def posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _FakeClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


class _FakeRespErr:
    def raise_for_status(self):
        raise RuntimeError("engine 500")

    def json(self):
        return {}


class _FakeClientErr:
    """Records POSTs but raise_for_status() fails (engine 500 on restore)."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _FakeRespErr()


@pytest.fixture
def posts_failing(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _FakeClientErr(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


# turn primitives
_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}

# The forced clean-restore is gated to the qwen family (proven-safe scope).
# Force-path tests use a qwen tag so the gate passes; a dedicated test asserts a
# non-qwen tag safe-degrades to no-force.
_QWEN = "qwen3-27b"


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    """Write a pinned clean bin (.bin + clean_prefix .json meta) for a thread."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")  # non-empty bin so existence check passes
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


# --- _generated_assistant_msg ----------------------------------------------------
def test_generated_msg_text():
    r = {"choices": [{"message": {"role": "assistant", "content": "hello answer"}}]}
    assert TurbohaulManager._generated_assistant_msg(r) == {
        "role": "assistant", "content": "hello answer"}


def test_generated_msg_merges_reasoning_as_think():
    # The reconstruction uses the SINGLE-SOURCE wrap_reasoning_think (the exact
    # _merge_reasoning_into_content NEWLINE form the harness stores), NOT the old
    # no-newline `<think>{r}</think>{c}` form — so a shadow reconstruction byte-matches
    # the harness's think-stripped resend.
    r = {"choices": [{"message": {
        "role": "assistant", "content": "final", "reasoning_content": "deep thoughts"}}]}
    got = TurbohaulManager._generated_assistant_msg(r)
    assert got == {"role": "assistant", "content": "<think>\ndeep thoughts\n</think>\n\nfinal"}
    # and it byte-matches _merge_reasoning_into_content on the same inputs (single source)
    from turbohaul.api.chat_completion import _merge_reasoning_into_content
    merged = {"choices": [{"message": {
        "role": "assistant", "content": "final", "reasoning_content": "deep thoughts"}}]}
    _merge_reasoning_into_content(merged)
    assert got["content"] == merged["choices"][0]["message"]["content"]


def test_generated_msg_tool_call_is_none():
    """Tool-call turn -> None -> caller SAFE-DEGRADES (never force on tool loops)."""
    r = {"choices": [{"message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}}]}
    assert TurbohaulManager._generated_assistant_msg(r) is None


def test_generated_msg_noop_and_streamed_are_none():
    assert TurbohaulManager._generated_assistant_msg(None) is None
    assert TurbohaulManager._generated_assistant_msg({"_streamed": True}) is None
    assert TurbohaulManager._generated_assistant_msg({"choices": []}) is None


# --- _engine_view_chain ----------------------------------------------------------
def test_engine_view_chain_prompt_plus_generated(mgr):
    slot = Slot.new(model_tag="m", client_meta={"messages": [_SYS, _U1]})
    result = {"choices": [{"message": {"role": "assistant", "content": "A"}}]}
    chain = mgr._engine_view_chain(slot, result)
    assert chain == _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}])


def test_engine_view_chain_unknown_when_generated_unreconstructable(mgr):
    slot = Slot.new(model_tag="m", client_meta={"messages": [_SYS, _U1]})
    assert mgr._engine_view_chain(slot, None) == []  # noop -> unknown -> safe
    tool = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "x"}]}}]}
    assert mgr._engine_view_chain(slot, tool) == []


# --- _classify_event (the 4 events + guard-skip) ---------------------------------
def test_classify_guard_skip(mgr):
    assert mgr._classify_event([], [], []) == "guard-skip"


def test_classify_sub_agent_when_no_anchor(mgr):
    inc = _prefix_hash_chain([_SYS, _U1])
    assert mgr._classify_event(None, inc, []) == "sub-agent"


def test_classify_compression_when_diverged(mgr):
    clean = _prefix_hash_chain([_SYS, _U1])
    inc = _prefix_hash_chain([_SYS, {"role": "user", "content": "rewritten"}])
    assert mgr._classify_event(clean, inc, []) == "compression"


def test_classify_continuation_when_warm_covers_longer(mgr):
    clean = _prefix_hash_chain([_SYS, _U1])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}])
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    assert mgr._classify_event(clean, inc, warm) == "continuation"


def test_classify_user_message_when_warm_diverges(mgr):
    """Think-strip: warm holds <think>, incoming is stripped -> user-message."""
    clean = _prefix_hash_chain([_SYS, _U1])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>t</think>A"}])
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    assert mgr._classify_event(clean, inc, warm) == "user-message"


# --- _find_clean_bin -------------------------------------------------------------
def test_find_clean_bin(mgr, kv_dir):
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, "m", 59500, "t", 0, chain)
    found = mgr._find_clean_bin(59500, "m", "t")
    assert found is not None
    assert found[0] == bin_fn and found[1] == chain and found[2] == 0


def test_find_clean_bin_absent(mgr, kv_dir):
    assert mgr._find_clean_bin(59500, "m", "no-such-thread") is None


# --- _maybe_force_clean_restore : the NO-DOWNGRADE gate --------------------------
@pytest.mark.asyncio
async def test_force_on_think_strip_user_turn(mgr, kv_dir, posts):
    """THE KEY WIN: warm KV holds the <think> turn; incoming is think-stripped ->
    clean bin IS a valid prefix, warm diverges -> FORCE restore the clean bin."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True
    assert d["action"] == "restore" and d["event_type"] == "user-message"
    assert d["resolved_from"] == "warm-force-clean-restore"
    assert d["clean_bin_id"] == bin_fn and d["common_prefix_turns"] == 2
    # POSTed action=restore of the clean bin into the engine slot
    assert len(posts) == 1
    url, body = posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}
    # metric updated
    assert mgr._kv_classifier_forced == 1
    assert mgr._kv_classifier_counts.get("user-message") == 1


@pytest.mark.asyncio
async def test_no_force_on_true_continuation(mgr, kv_dir, posts):
    """NO-DOWNGRADE: warm state is an equal-or-LONGER valid prefix (a real tool/
    continuation turn kept verbatim) -> DO NOT force; native warm reuse wins."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, "m", 59500, "t", 0, clean_chain)
    a1 = {"role": "assistant", "content": "A"}
    warm = _prefix_hash_chain([_SYS, _U1, a1])
    inc = _prefix_hash_chain([_SYS, _U1, a1, _U2])   # incoming KEEPS a1 verbatim
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, warm)
    assert d["forced_clean_restore"] is False
    assert d["event_type"] == "continuation"
    assert d["resolved_from"] == "warm-native-reuse-longer"
    assert posts == []  # NO restore POST -> no regression


@pytest.mark.asyncio
async def test_no_force_when_warm_state_unknown(mgr, kv_dir, posts):
    """Safe-degrade: warm chain unknown ([] — noop/streamed/tool-call) -> no force."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, "m", 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, [])
    assert d["forced_clean_restore"] is False
    assert d["resolved_from"] == "warm-state-unknown-safe"
    assert posts == []


@pytest.mark.asyncio
async def test_no_force_when_compression(mgr, kv_dir, posts):
    """Clean anchor is NOT a prefix of incoming (early turn rewritten) -> compression
    -> FRESH, no forced restore."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, "m", 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, {"role": "user", "content": "summarized"}])
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, [])
    assert d["forced_clean_restore"] is False
    assert d["event_type"] == "compression"
    assert d["resolved_from"] == "warm-clean-not-prefix"
    assert posts == []


@pytest.mark.asyncio
async def test_no_force_physics_belt_clean_longer(mgr, kv_dir, posts):
    """PHYSICS BELT: clean bin has MORE turns than incoming -> not a valid prefix ->
    never force a bin longer than incoming (the engine would CLEAR)."""
    clean_chain = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    _write_clean_bin(kv_dir, "m", 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1])  # shorter than the clean bin
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, [])
    assert d["forced_clean_restore"] is False
    assert posts == []


@pytest.mark.asyncio
async def test_no_force_when_no_clean_bin(mgr, kv_dir, posts):
    inc = _prefix_hash_chain([_SYS, _U1])
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, [])
    assert d["forced_clean_restore"] is False
    assert d["event_type"] == "sub-agent"
    assert d["resolved_from"] == "warm-no-clean-bin"
    assert posts == []


@pytest.mark.asyncio
async def test_guard_skip_when_no_incoming_chain(mgr, kv_dir, posts):
    slot = Slot.new(model_tag="m", thread_id="t", admission_hash_chain=[])
    d = await mgr._maybe_force_clean_restore(59500, "m", slot, [])
    assert d["event_type"] == "guard-skip" and d["forced_clean_restore"] is False
    assert posts == []


# --- status_snapshot metric surface ----------------------------------------------
@pytest.mark.asyncio
async def test_status_snapshot_exposes_classifier_metric(mgr, kv_dir, posts):
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    snap = mgr.status_snapshot()
    assert "kv_classifier" in snap
    kc = snap["kv_classifier"]
    assert kc["forced_clean_restores"] == 1
    assert kc["events"].get("user-message") == 1
    assert kc["last"]["resolved_from"] == "warm-force-clean-restore"


# --- qwen-only scope on the force path -------------------------------------------
@pytest.mark.asyncio
async def test_no_force_non_qwen_family(mgr, kv_dir, posts):
    """The SAME think-strip signature that FORCES on qwen must NOT force on a
    non-qwen family (the turn-hash⊑ → token-prefix equivalence is unproven there;
    token divergence could exceed n_rs_seq=2 → CLEAR). Safe-degrade to no-force."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, "llama-70b", 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag="llama-70b", thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, "llama-70b", slot, warm)
    assert d["forced_clean_restore"] is False
    assert d["resolved_from"] == "warm-non-qwen-skip"
    assert posts == []
    assert mgr._kv_classifier_forced == 0


def test_is_qwen_family():
    assert TurbohaulManager._is_qwen_family("qwen3-27b")
    assert TurbohaulManager._is_qwen_family("engine_qwen3-27b-mtp")
    assert TurbohaulManager._is_qwen_family("QwQ-32B")
    assert not TurbohaulManager._is_qwen_family("llama-3-70b")
    assert not TurbohaulManager._is_qwen_family("")


# --- status-check the forced restore POST ----------------------------------------
@pytest.mark.asyncio
async def test_force_restore_post_failure_not_counted(mgr, kv_dir, posts_failing):
    """The engine 500s the restore POST -> the decode proceeds on the polluted
    warm KV, so the metric must NOT claim a clean restore. forced_clean_restore
    stays False; resolved_from records the attempt-but-failed."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["action"] == "restore"                 # we attempted
    assert d["forced_clean_restore"] is False        # but it did not land
    assert d["resolved_from"] == "warm-force-restore-failed"
    assert len(posts_failing) == 1                    # POST was made (then 500'd)
    assert mgr._kv_classifier_forced == 0             # NOT counted


# --- streaming warm-state reconstruction hook ------------------------------------
def test_engine_view_chain_from_streamed_text(mgr):
    """The streaming route stashes the generated text on slot.streamed_assistant_
    text; _engine_view_chain(slot, None) reconstructs the same engine-view warm_chain
    as the non-streaming path (prompt + the generated <think> turn)."""
    slot = Slot.new(model_tag=_QWEN, client_meta={"messages": [_SYS, _U1]})
    slot.streamed_assistant_text = "<think>t</think>A"
    chain = mgr._engine_view_chain(slot, None)
    assert chain == _prefix_hash_chain(
        [_SYS, _U1, {"role": "assistant", "content": "<think>t</think>A"}])
    # empty stash -> unknown -> [] (safe)
    slot2 = Slot.new(model_tag=_QWEN, client_meta={"messages": [_SYS, _U1]})
    assert mgr._engine_view_chain(slot2, None) == []


# --- near-match continuation (ACCEPTED perf-limitation) --------------------------
@pytest.mark.asyncio
async def test_near_match_continuation_forces_perf_limitation(mgr, kv_dir, posts):
    """ACCEPTED perf-limitation, documented: a TRUE continuation that re-sends
    the preserved-think assistant turn BYTE-DIFFERENTLY (trailing whitespace trimmed)
    is not an EXACT turn-hash prefix of the warm chain, so the gate cannot see that
    the engine would have seq_rm-reused it (≤ n_rs_seq=2 token divergence). The force
    fires and reprefills the reasoning block. PERF-only (no CLEAR); a typical client
    does NOT hit it (its win case is the unconditional think-STRIP). Turn-hash granularity
    can't distinguish tail-divergence (engine reuses) from lead-divergence (CLEAR).
    This test PINS the current documented behavior (force), not an ideal."""
    a1_warm = {"role": "assistant", "content": "<think>reason</think>answer\n"}  # trailing \n
    a1_inc = {"role": "assistant", "content": "<think>reason</think>answer"}     # \n trimmed
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    warm = _prefix_hash_chain([_SYS, _U1, a1_warm])
    inc = _prefix_hash_chain([_SYS, _U1, a1_inc, _U2])  # preserved-think, perturbed
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    # DOCUMENTED perf-limitation: warm not an exact prefix -> force fires (would-be
    # seq_rm reuse becomes a reprefill). Not ideal, accepted; the think-strip win case
    # is unaffected.
    assert d["forced_clean_restore"] is True
    assert d["resolved_from"] == "warm-force-clean-restore"


# --- bounded clean-bin pin in GC -------------------------------------------------
@pytest.mark.asyncio
async def test_gc_bounds_clean_bin_pin(mgr, kv_dir):
    """Pin only the NEWEST/LONGEST clean bin per (model_tag, thread_hash). A
    superseded clean bin (shorter, older-port) for the SAME thread is reclaimable —
    else one full-KV dump leaks per save forever."""
    ch = _prefix_hash_chain([_SYS, _U1])
    keep = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, ch, prompt_len=80000)   # longest -> pinned
    drop = _write_clean_bin(kv_dir, _QWEN, 59501, "t", 0, ch, prompt_len=40000)   # superseded (older port, shorter)
    # max_age_hours=0 -> every non-pinned bin is age-evictable; the pinned one is exempt.
    deleted = await mgr._gc_kv_cache(max_age_hours=0.0, max_files=100)
    assert (kv_dir / keep).exists()          # the anchor _find_clean_bin picks survives
    assert not (kv_dir / drop).exists()      # the superseded clean bin is reclaimed
    assert not (kv_dir / (drop[:-4] + ".json")).exists()  # + its meta sidecar
    assert deleted >= 1


@pytest.mark.asyncio
async def test_gc_keeps_sole_clean_anchor(mgr, kv_dir):
    """A thread's single live clean anchor is NEVER evicted (age or LRU)."""
    ch = _prefix_hash_chain([_SYS, _U1])
    anchor = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, ch, prompt_len=80000)
    deleted = await mgr._gc_kv_cache(max_age_hours=0.0, max_files=0)
    assert (kv_dir / anchor).exists()        # pinned despite age=0 + max_files=0
    assert deleted == 0


# ================================================================================
# WARM forced clean-bin restore — DEFAULT-ON since the single-copy flip; env
# '0'/'false'/'no'/'off' (or the /var/lib/turbohaul/.warm_force_clean_off sentinel)
# is the kill-switch
# ================================================================================
@pytest.mark.parametrize("val,expected", [
    (None, True), ("", True), ("0", False), ("false", False), ("no", False),
    ("off", False), ("  OFF ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), ("TRUE", True),
])
def test_warm_force_clean_restore_flag_reader(monkeypatch, val, expected):
    """The flag helper after the single-copy flip: unset/"" -> True
    (DEFAULT ON); 0/false/no/off -> False (kill-switch); 1/true/yes/on -> True
    (case/whitespace-insensitive)."""
    if val is None:
        monkeypatch.delenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", raising=False)
    else:
        monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", val)
    assert manager_mod._warm_force_clean_restore_enabled() is expected


@pytest.mark.asyncio
async def test_warm_force_gated_off_via_env_kill_switch(mgr, kv_dir, posts, monkeypatch):
    """Kill-switch (post default-ON flip): the EXACT scenario that now forces BY
    DEFAULT must NOT force when the env kill-switch is set — no restore POST fires;
    the warm follow-up rides the engine's native in-RAM get_common_prefix reuse.
    Proves the rollback lever stays real."""
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "0")  # kill-switch
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is False
    assert d["action"] == "fresh"
    assert d["resolved_from"] == "warm-force-gated-native-reuse"
    assert posts == []                          # NO restore POST -> native reuse
    assert mgr._kv_classifier_forced == 0
    # gate is observable on /status via the last classifier decision
    assert mgr.status_snapshot()["kv_classifier"]["last"][
        "resolved_from"] == "warm-force-gated-native-reuse"


@pytest.mark.asyncio
async def test_warm_force_still_fires_when_flag_on(mgr, kv_dir, posts, monkeypatch):
    """ROLLBACK path intact: with the flag explicitly ON, the SAME think-strip
    scenario STILL forces the clean-bin restore (byte-identical to pristine behavior)."""
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    warm = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "<think>x</think>A"}])
    slot = Slot.new(model_tag=_QWEN, thread_id="t", admission_hash_chain=inc)
    d = await mgr._maybe_force_clean_restore(59500, _QWEN, slot, warm)
    assert d["forced_clean_restore"] is True
    assert d["action"] == "restore" and d["resolved_from"] == "warm-force-clean-restore"
    assert len(posts) == 1
    url, body = posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}
    assert mgr._kv_classifier_forced == 1


@pytest.mark.asyncio
async def test_cold_wave_return_independent_of_warm_flag(mgr, kv_dir, posts, monkeypatch):
    """DECOUPLING: the COLD swap-back / wave-return (_restore_slot_kv) must NOT be
    coupled to the new WARM flag. With the WARM force DEFAULT-OFF, a cold clean restore
    STILL fires + tags wave-return (proves the cold path is left untouched)."""
    monkeypatch.delenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", raising=False)  # warm force OFF
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_clean_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain)
    inc = _prefix_hash_chain([_SYS, _U1, {"role": "assistant", "content": "A"}, _U2])
    slot = Slot.new(model_tag=_QWEN, thread_id="t",
                    admission_ctx_len=50000, admission_hash_chain=inc)
    await mgr._restore_slot_kv(59500, _QWEN, slot)
    assert mgr._kv_classifier_wave_return == 1        # cold restore fired
    assert mgr._kv_classifier_forced == 0             # warm counter untouched
    last = mgr._kv_classifier_last
    assert last["resolved_from"] == "wave-return-clean-restore"
    assert last["forced_clean_restore"] is False
    assert len(posts) == 1
    url, body = posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}
