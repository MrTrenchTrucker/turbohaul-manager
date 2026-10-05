"""WARM stale-tail restore SKIP.

`_maybe_force_clean_restore`'s force gate (clean_valid AND warm_known AND NOT warm_covers) decides
WHETHER to force a disk clean-bin restore over the engine's native warm reuse, but does not ask
whether the CURRENTLY-RESIDENT warm state has MORE turns than the INCOMING request itself represents.
When it does (e.g. a background turn served on the same shared engine slot appended content past the
last turn the current conversation has ever produced), the restore loads the shorter clean bin but
leaves the surplus warm KV resident; the engine's trim budget for that surplus is small and fixed, so
any multi-turn surplus exceeds it and the engine CLEARs the slot and does a full reprefill instead of
the incremental reuse the restore was meant to provide — worse than the native reuse it was forcing
past.

Fix (restore-DECISION path only, WARM side): when `warm_turns > inc_turns`, SKIP the force/POST and
safe-degrade to the engine's native reuse, mirroring the existing TOOL-tail restore-skip guard's shape
(same file, same function, same placement before `if force:`).

Compared against the INCOMING turn count, not the clean bin's: comparing against `clean_turns`
would skip the force-restore for a normal conversation whose clean bin trails by one turn,
which the warm-path scenarios of test_tooltail_restore_skip.py cover. A clean bin trailing a
healthy, ordinary conversation by one turn (e.g. the WITH-reasoning turn the harness will resend
stripped) is completely normal and must still force-restore; only warm exceeding what the CURRENT
request itself represents is structurally foreign. `_is_prefix_match` (kv_policy) already refuses a
saved chain longer than incoming outright, so `warm_turns > inc_turns` is the precise signal.

Scenarios: (a) warm longer than incoming AND diverges (curator-pollution shape) -> SKIP; (b) warm
much SHORTER than everything -> restore STILL fires (low-end control); (c) warm turns EQUAL to
incoming's own turn count (diverged content) -> does NOT skip (boundary: strict `>`, not `>=`); (c2)
REGRESSION GUARD — warm longer than clean but NOT exceeding incoming (main's own natural next turn,
the tool-tail suite's own WARM_CHAIN shape) -> does NOT skip; (d) flag OFF, same fixture as (a),
opposite outcome -> restore fires. Plus a flag-reader unit test. Fixtures mirror
test_tooltail_restore_skip.py.
"""
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn, kv_save_fn
from turbohaul.manager import TurbohaulManager


# --- fixtures (mirror test_tooltail_restore_skip.py) ---------------------------------
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
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _flags_clean(monkeypatch):
    """Default: staletail skip unset (default ON per _staletail_restore_skip_enabled), a
    clean ambient env otherwise. Tests that need it OFF opt in via the `skip_off` fixture."""
    for f in ("TURBOHAUL_STALETAIL_RESTORE_SKIP", "TURBOHAUL_TOOLTAIL_RESTORE_SKIP",
              "TURBOHAUL_SHADOW_REPREFILL", "TURBOHAUL_SHADOW_RESTORE_PREFER",
              "TURBOHAUL_SHADOW_COLD_RESTORE"):
        monkeypatch.delenv(f, raising=False)
    # The staletail guard sits INSIDE the WARM force gate -- arm it, same as the
    # tooltail precedent does for its own warm scenarios.
    monkeypatch.setenv("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "1")


@pytest.fixture
def skip_off(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_STALETAIL_RESTORE_SKIP", "0")


# --- restore-POST capture (only /slots/{sid}?action=restore fires here) -------------
class _Resp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _RestoreClient:
    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _Resp()


@pytest.fixture
def capture_restore(monkeypatch):
    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


def _restores(posts):
    return [(u, b) for (u, b) in posts if "action=restore" in u]


def _the_restore(posts):
    r = _restores(posts)
    assert len(r) == 1, f"expected exactly one restore POST, got {posts}"
    return r[0]


# --- conversation shapes ------------------------------------------------------------
_QWEN = "qwen-27b"              # matches the model-family gate -> the force gate is armed
_PORT = 59500
_TID = "agent-ip-10.0.0.5"
_SID_CLEAN = 0


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _write_clean(kv_dir, sid, chain, tid=_TID, port=_PORT, prompt_len=40000):
    th = TurbohaulManager._thread_hash(tid)
    bin_fn = kv_save_fn(_QWEN, sid, th, port)
    meta_fn = kv_meta_fn(_QWEN, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"CLEAN_ANCHOR_BYTES")
    (kv_dir / meta_fn).write_text(__import__("json").dumps({
        "thread_id": tid, "thread_hash": th, "prompt_tokens": 500,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "prompt_hash": "", "model_tag": _QWEN, "slot_id": sid, "port": port,
        "clean_prefix": True,
    }))
    return bin_fn


def _warm_slot(inc_msgs, inc_chain, tid=_TID):
    return SimpleNamespace(
        thread_id=tid,
        admission_hash_chain=inc_chain,
        client_meta={"messages": inc_msgs},
    )


# Base = 5 settled turns. The clean anchor covers [0..4], a valid prefix of every INC below.
CLEAN_MSGS = _msgs(5)
CLEAN_CHAIN = _prefix_hash_chain(CLEAN_MSGS)

# incoming: clean's 5 turns + ONE new real turn (main's own next turn) = 6
INC_MSGS = CLEAN_MSGS + [{"role": "user", "content": "the next real user turn"}]
INC_CHAIN = _prefix_hash_chain(INC_MSGS)

# (a) warm LONGER than clean AND diverges from incoming — the curator-pollution shape:
# shares clean's 5 turns, then TWO turns neither matching incoming's own 6th turn.
WARM_LONGER_DIVERGED_MSGS = CLEAN_MSGS + [
    {"role": "assistant", "content": "a foreign turn appended on the shared slot"},
    {"role": "assistant", "content": "a second foreign turn"},
]
WARM_LONGER_DIVERGED_CHAIN = _prefix_hash_chain(WARM_LONGER_DIVERGED_MSGS)   # len 7

# (b) warm much SHORTER than everything — structurally cannot cover clean_chain's
# length either; force still fires, guard must NOT fire (len(warm) > len(inc) is False).
WARM_SHORTER_MSGS = _msgs(3)
WARM_SHORTER_CHAIN = _prefix_hash_chain(WARM_SHORTER_MSGS)                  # len 3

# (c) warm turns EQUAL to incoming's own turn count, but diverged content at the last
# turn — boundary: the guard's strict `>` must NOT fire on `==`, even though force still
# does (warm doesn't cover incoming either, since content diverges).
WARM_EQUAL_INC_DIVERGED_MSGS = CLEAN_MSGS + [
    {"role": "user", "content": "a turn that differs from incoming's own 6th turn"},
]
WARM_EQUAL_INC_DIVERGED_CHAIN = _prefix_hash_chain(WARM_EQUAL_INC_DIVERGED_MSGS)  # len 6 == len(INC_CHAIN)

# (c2) REGRESSION GUARD: warm has one MORE turn than the clean bin, but does NOT exceed
# the incoming request's own turn count — main's own natural next turn, generated WITH
# reasoning content the harness will resend WITHOUT. Real, common, already-shipped shape:
# test_tooltail_restore_skip.py's own WARM_CHAIN fixture is exactly this (CLEAN_MSGS +
# one with-think turn), and its warm-path "text tail still forces" scenarios would
# catch a clean_turns-based version of this guard over-triggering here.
WARM_NATURAL_CONTINUATION_MSGS = CLEAN_MSGS + [
    {"role": "assistant", "content": "<think>reasoning</think>the final answer"},
]
WARM_NATURAL_CONTINUATION_CHAIN = _prefix_hash_chain(WARM_NATURAL_CONTINUATION_MSGS)  # len 6

INC_STRIPPED_MSGS = CLEAN_MSGS + [
    {"role": "assistant", "content": "the final answer"},
    {"role": "user", "content": "the next real user turn"},
]
INC_STRIPPED_CHAIN = _prefix_hash_chain(INC_STRIPPED_MSGS)                  # len 7; warm(6) <= inc(7)


# ====================================================================================
# (e) pure flag-reader unit test
# ====================================================================================
@pytest.mark.parametrize("val,expected", [
    ("", True), ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("false", False), ("no", False), ("OFF", False),
])
def test_flag_default_on(monkeypatch, val, expected):
    # Default ON (unlike the tool-tail flag): UNSET -> ON; only explicit falsy -> OFF.
    if val == "":
        monkeypatch.delenv("TURBOHAUL_STALETAIL_RESTORE_SKIP", raising=False)
    else:
        monkeypatch.setenv("TURBOHAUL_STALETAIL_RESTORE_SKIP", val)
    assert manager_mod._staletail_restore_skip_enabled() is expected


# ====================================================================================
# WARM path — _maybe_force_clean_restore
# ====================================================================================
@pytest.mark.asyncio
async def test_warm_a_longer_diverged_warm_skips_force(mgr, kv_dir, capture_restore):
    """(a) warm longer than clean AND diverges (curator-shaped pollution) -> force SKIPPED,
    no restore POST, counter bumped, resolved_from set."""
    _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_MSGS, INC_CHAIN), WARM_LONGER_DIVERGED_CHAIN)

    assert _restores(capture_restore) == []              # no bin POSTed
    assert decision["forced_clean_restore"] is False
    assert decision["action"] != "restore"                # no instrument lie
    assert decision["resolved_from"] == "warm-staletail-skip"
    assert mgr._kv_staletail_skip_counts.get("warm") == 1


@pytest.mark.asyncio
async def test_warm_b_shorter_warm_still_forces(mgr, kv_dir, capture_restore):
    """(b) CONTROL: warm SHORTER than clean -> the clean anchor is STILL force-restored.
    Must fail (restore must fire) if the guard over-triggers on the wrong side."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_MSGS, INC_CHAIN), WARM_SHORTER_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_staletail_skip_counts == {}            # guard never tripped


@pytest.mark.asyncio
async def test_warm_c_equal_to_incoming_diverged_warm_does_not_skip(mgr, kv_dir, capture_restore):
    """(c) BOUNDARY: warm turns EQUAL to incoming's own turn count (but diverged content)
    -> force still fires, the staletail guard's strict `>` must NOT trip on `==`."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_MSGS, INC_CHAIN), WARM_EQUAL_INC_DIVERGED_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_staletail_skip_counts == {}


@pytest.mark.asyncio
async def test_warm_c2_natural_continuation_not_exceeding_incoming_does_not_skip(
        mgr, kv_dir, capture_restore):
    """(c2) REGRESSION: warm has one more turn than clean (main's own natural next turn,
    with reasoning the harness will resend without) but does NOT exceed the incoming
    request's own turn count -> must NOT skip. This is the exact shape a
    clean_turns-based version of this guard would break, as in test_tooltail_restore_skip.py's
    own WARM_CHAIN fixture."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_STRIPPED_MSGS, INC_STRIPPED_CHAIN),
        WARM_NATURAL_CONTINUATION_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_staletail_skip_counts == {}


@pytest.mark.asyncio
async def test_warm_d_flag_off_same_fixture_opposite_outcome(
        mgr, kv_dir, capture_restore, skip_off):
    """(d) SAME fixture as (a), flag OFF -> opposite outcome: the restore fires despite the
    warm-longer-and-diverged shape. Proves the flag is the lever, not incidental."""
    clean_bin = _write_clean(kv_dir, _SID_CLEAN, CLEAN_CHAIN)
    decision = await mgr._maybe_force_clean_restore(
        _PORT, _QWEN, _warm_slot(INC_MSGS, INC_CHAIN), WARM_LONGER_DIVERGED_CHAIN)

    url, body = _the_restore(capture_restore)
    assert body["filename"] == clean_bin
    assert decision["forced_clean_restore"] is True
    assert decision["resolved_from"] == "warm-force-clean-restore"
    assert mgr._kv_staletail_skip_counts == {}
