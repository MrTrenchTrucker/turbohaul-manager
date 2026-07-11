"""Phase 1 — GOLDEN-MASTER characterization of TODAY's scattered
KV request classification.

These tests PIN the current behavior of the ~5 scattered sites / 3 mechanisms so
Phase 2 (which unifies them onto ``kv_classify.classify_request``) has a byte-level
safety net. They assert the SCATTERED logic directly (NOT the new module) — the new
module is proven to MATCH this baseline in ``test_kv_classify_scaffold.py``.

Baseline = LABELS ABSENT (today's traffic): ``client_meta`` identity labels are parsed
but dropped, so the byte-identical baseline never reads them.

Mechanisms pinned:
  1. event_type  -> manager.TurbohaulManager._classify_event  (exhaustive matrix).
  2. identity    -> chat_completion._shadow_recompose_identity + the _m2b_active gate
                    (default-OFF today) + slot.derive_thread_id_prefix_hash base rule.
  3. reuse (cold)-> kv_policy.resolve_kv("restore", ...) do_it  (unit, per-bin) AND the
                    manager._restore_slot_kv fresh-vs-restore AGGREGATE (integration,
                    with on-disk bin fixtures).
"""
import json
from dataclasses import dataclass

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
from turbohaul.api.chat_completion import _m2b_active, _shadow_recompose_identity
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
from turbohaul.kv_policy import (
    _prefix_hash_chain,
    compute_ctx_len,
    kv_meta_fn,
    resolve_kv,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, derive_thread_id_prefix_hash

# ============================================================================
# Shared classification matrix (imported by test_kv_classify_scaffold.py too)
# ============================================================================

# Turn primitives.
SYS = {"role": "system", "content": "system prompt long enough to matter"}
U1 = {"role": "user", "content": "first user turn"}
U1R = {"role": "user", "content": "first user turn REWRITTEN"}  # compression
A1 = {"role": "assistant", "content": "answer one"}
A1T = {"role": "assistant", "content": "<think>deep</think>answer one"}  # with-think warm
AX = {"role": "assistant", "content": "a DIFFERENT answer"}
U2 = {"role": "user", "content": "second user turn"}


def _chain(msgs):
    """Turn-hash chain of a message list, or None to represent 'no anchor'."""
    return None if msgs is None else _prefix_hash_chain(msgs)


@dataclass(frozen=True)
class Scenario:
    id: str
    clean_msgs: list | None   # saved clean-anchor turns (None => no anchor)
    inc_msgs: list            # incoming turns
    warm_msgs: list | None    # engine warm-state turns (None => unknown)
    expected_event: str       # manager._classify_event(...) result
    expected_reuse: str       # 'restore'|'fresh' — resolve_kv/cold aggregate outcome

    @property
    def clean_chain(self):
        return _chain(self.clean_msgs)

    @property
    def inc_chain(self):
        return _prefix_hash_chain(self.inc_msgs)

    @property
    def warm_chain(self):
        return _chain(self.warm_msgs)

    @property
    def size(self):
        return compute_ctx_len(self.inc_msgs)


# LABELS-ABSENT matrix covering all 5 events + both reuse outcomes + physics belt +
# warm-unknown cold path.
SCENARIOS = [
    Scenario("guard_skip_empty_inc", [SYS, U1], [], None, "guard-skip", "fresh"),
    Scenario("sub_agent_no_anchor", None, [SYS, U1], None, "sub-agent", "fresh"),
    Scenario("compression_diverged", [SYS, U1], [SYS, U1R], None, "compression", "fresh"),
    Scenario(
        "continuation_warm_covers_longer",
        [SYS, U1], [SYS, U1, A1, U2], [SYS, U1, A1], "continuation", "restore",
    ),
    Scenario(
        "user_message_warm_diverges",
        [SYS, U1], [SYS, U1, A1, U2], [SYS, U1, A1T], "user-message", "restore",
    ),
    Scenario("continuation_equal", [SYS, U1], [SYS, U1], [SYS, U1], "continuation", "restore"),
    Scenario(
        "user_message_warm_unknown_cold",
        [SYS, U1], [SYS, U1, U2], None, "user-message", "restore",
    ),
    Scenario(
        "physics_belt_clean_longer",
        [SYS, U1, A1, U2], [SYS, U1], None, "compression", "fresh",
    ),
    Scenario(
        "compression_partial_diverge",
        [SYS, U1, A1], [SYS, U1, AX], None, "compression", "fresh",
    ),
]
SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}


# ============================================================================
# Fixtures (mirrors tests/test_ws2_classifier.py conventions)
# ============================================================================

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
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _FakeClient:
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


def _write_clean_bin(kv_dir, model_tag, port, thread_id, sid, chain, prompt_len=40000):
    """Write a pinned clean bin (.bin + clean_prefix .json meta) for a thread."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 12345,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": True,
    }))
    return bin_fn


# ============================================================================
# 1. event_type — pin manager._classify_event over the full matrix
# ============================================================================

@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_classify_event_golden(mgr, sc):
    got = mgr._classify_event(sc.clean_chain, sc.inc_chain, sc.warm_chain)
    assert got == sc.expected_event, f"{sc.id}: {got!r} != {sc.expected_event!r}"


def test_classify_event_covers_all_five_events():
    """Sanity: the matrix exercises every one of the 5 event outcomes."""
    assert {s.expected_event for s in SCENARIOS} == {
        "guard-skip", "sub-agent", "compression", "continuation", "user-message",
    }


# ============================================================================
# 2a. reuse (cold, UNIT) — pin kv_policy.resolve_kv("restore") do_it per bin
# ============================================================================

def _resolve_restore(sc):
    """resolve_kv('restore', ...) for an owner-matched, positive-token baseline —
    isolates the decision to prefix-validity (today's cold per-bin rule)."""
    return resolve_kv(
        "restore",
        {"thread_id": "t", "model_tag": "m", "slot_id": 0, "port": 0},
        {
            "saved_tokens": 12345,
            "saved_len": compute_ctx_len(sc.clean_msgs or []),
            "incoming_len": sc.size,
            "cache_age_s": 0.0,
            "saved_thread_id": "t",
            "saved_chain": sc.clean_chain or [],
            "incoming_chain": sc.inc_chain,
        },
    )


@pytest.mark.parametrize(
    "sc",
    [s for s in SCENARIOS if s.clean_msgs is not None],
    ids=[s.id for s in SCENARIOS if s.clean_msgs is not None],
)
def test_resolve_kv_restore_golden(sc):
    """Unit-pin the cold per-bin restore decision. do_it True <=> reuse 'restore'."""
    dec = _resolve_restore(sc)
    expected_do_it = sc.expected_reuse == "restore"
    assert dec.do_it is expected_do_it, f"{sc.id}: {dec!r}"


# ============================================================================
# 2b. reuse (cold, INTEGRATION) — pin _restore_slot_kv fresh-vs-restore AGGREGATE
# ============================================================================

def _make_slot(sc, model_tag="m", thread_id="t"):
    return Slot.new(
        model_tag=model_tag,
        thread_id=thread_id,
        client_meta={"messages": sc.inc_msgs},
        admission_ctx_len=sc.size,
        admission_hash_chain=sc.inc_chain,
    )


async def test_restore_slot_kv_restores_on_valid_prefix(mgr, kv_dir, posts):
    """A valid-prefix clean anchor -> cold restore POST fires (reuse='restore').

    Default env: warm-force OFF (cold path unaffected), shadow-cold OFF (no shadow bin,
    keep clean anchor), tooltail-skip OFF (text tail, no skip)."""
    sc = SCENARIOS_BY_ID["user_message_warm_unknown_cold"]  # clean [SYS,U1] ⊑ inc [SYS,U1,U2]
    _write_clean_bin(kv_dir, "m", 0, "t", 0, sc.clean_chain)
    await mgr._restore_slot_kv(0, "m", _make_slot(sc))
    assert len(posts) == 1, f"expected one restore POST, got {posts}"
    url, _ = posts[0]
    assert "action=restore" in url


async def test_restore_slot_kv_fresh_on_physics_belt(mgr, kv_dir, posts):
    """A clean anchor LONGER than incoming -> physics belt -> fresh, NO restore POST."""
    sc = SCENARIOS_BY_ID["physics_belt_clean_longer"]  # clean 4 turns, inc 2 turns
    _write_clean_bin(kv_dir, "m", 0, "t", 0, sc.clean_chain)
    await mgr._restore_slot_kv(0, "m", _make_slot(sc))
    assert posts == [], f"expected NO restore POST (fresh), got {posts}"


async def test_restore_slot_kv_fresh_on_diverged(mgr, kv_dir, posts):
    """A clean anchor that diverges from incoming (compression) -> fresh, no POST."""
    sc = SCENARIOS_BY_ID["compression_diverged"]
    _write_clean_bin(kv_dir, "m", 0, "t", 0, sc.clean_chain)
    await mgr._restore_slot_kv(0, "m", _make_slot(sc))
    assert posts == [], f"expected NO restore POST (fresh), got {posts}"


async def test_restore_slot_kv_fresh_when_no_bins(mgr, kv_dir, posts):
    """No saved bin at all (sub-agent / first-seen identity) -> fresh, no POST."""
    sc = SCENARIOS_BY_ID["sub_agent_no_anchor"]
    await mgr._restore_slot_kv(0, "m", _make_slot(sc))
    assert posts == [], f"expected NO restore POST (fresh), got {posts}"


# ============================================================================
# 3. identity — pin the recompose rule + the m2b (default-OFF) gate + base rule
# ============================================================================

# (base, role, session_id) triples. Labels-absent (None,None) is today's baseline.
_IDENTITY_CASES = [
    ("agent-ip-1.2.3.4-auto-abc", None, None),
    ("agent-ip-1.2.3.4-auto-abc", "curator", None),
    ("agent-ip-1.2.3.4-auto-abc", None, "sess-9"),
    ("agent-ip-1.2.3.4-auto-abc", "main", "sess-9"),
    ("", None, None),
]


@pytest.mark.parametrize("base,role,session", _IDENTITY_CASES)
def test_shadow_recompose_identity_golden(base, role, session):
    """Pin _shadow_recompose_identity: append-only, hashed suffixes, no-fields=identity."""
    got = _shadow_recompose_identity(base, role, session)
    # Append-only: the base is always a prefix of the recomposed key (never a merge).
    assert got.startswith(base)
    if not role and not session:
        # Labels absent -> recompose is an IDENTITY (today's baseline value == base).
        assert got == base
    if session:
        assert "-s=" in got
    if role:
        assert "-r=" in got


def test_identity_labels_absent_is_base(monkeypatch):
    """The load-bearing baseline: with labels absent AND m2b default-OFF, today's
    identity is exactly the base thread_id (byte-identical). This is what Phase 1's
    classify_request must reproduce."""
    monkeypatch.delenv("TURBOHAUL_M2B_ACTIVE", raising=False)
    assert _m2b_active() is False  # default OFF today
    base = "agent-ip-10.0.0.5-auto-deadbeef"
    # role/session absent -> recompose == base; and m2b OFF means it never drives id.
    assert _shadow_recompose_identity(base, None, None) == base


def test_derive_thread_id_prefix_hash_extension_invariant():
    """Base identity rule (mechanism 2): a prefix extension (same first-N tokens, more
    appended) maps to the SAME auto- thread_id, so a conversation's follow-ups reuse KV."""
    base_prompt = "system preamble " + " ".join(f"tok{i}" for i in range(300))
    extended = base_prompt + " " + " ".join(f"more{i}" for i in range(50))
    tid_a = derive_thread_id_prefix_hash(base_prompt, "m", prefix_tokens=256)
    tid_b = derive_thread_id_prefix_hash(extended, "m", prefix_tokens=256)
    assert tid_a == tid_b  # same 256-token prefix -> same id
    assert tid_a.startswith("auto-")
    # Different model_tag -> different id (model is part of the keyed payload).
    assert derive_thread_id_prefix_hash(base_prompt, "other", prefix_tokens=256) != tid_a
