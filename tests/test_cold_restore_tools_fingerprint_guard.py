"""Tools-fingerprint guard (a backstop to the chain-hash check) — cold-
restore TOOLS-FINGERPRINT decline guard.

The kv_policy.py chain hash being blind to `tools` is HALF THE
MECHANISM of the frozen-prefix re-prefill: a client's tools payload changes mid-conversation, Qwen
renders `tools` at the FRONT of the prompt so the render diverges at a
constant early offset, and because the chain hash cannot see `tools` the
manager reports a MATCH while the prompt has already diverged.

DESIGN CHOICE (weighed against the alternative below):
NOT a kv_policy.py hash change. The minimal "fold
tools into the chain only when present" design was worked through and found
INSUFFICIENT for this failure specifically -- it leaves a real coincidental-
collision path (an old tools-bearing bin's hash, computed with no tools
segment at all, can structurally coincide with a later tools-LESS request's
hash under the same minimal design) that is not a rare edge case here: tools instability
is common across threads. A safe hash change
would need an unconditional format bump (every bin invalidated, tools or
not) -- strictly bigger than the alternative below, and both routes are
equally safe for bins saved AFTER the guard, so the choice comes down to
blast radius: this guard reuses the already-live, already-tested
`_tools_fingerprint` machinery, touches nothing every other restore guard
depends on, and needs no bin-format version marker (one new OPTIONAL meta
field, `tools_fingerprint`, read with `.get()` like `clean_prefix`/`stale`).

THE HEALTHY-FLOW QUESTION, since a decline is a relaxation in reverse (the
same scrutiny applies): which HEALTHY flow most resembles the bad one, and does the guard
reject it? A thread whose tools legitimately never change, restoring either
(a) an OLD bin with no recorded fingerprint at all, or (b) a NEW bin whose
recorded fingerprint matches the incoming request's current one. The guard
fails OPEN on (a) (absence is not evidence of instability -- never
manufacture a decline out of a metadata gap) and proceeds on (b) (an exact
match). It declines ONLY a POSITIVELY RECORDED mismatch. Neither healthy
flow pays anything.

Two tiers, same discipline as the cold-restore size-guard test (this file's
direct structural sibling -- same fixtures, same insertion point one guard
later in the chain):
  * TIER 1 -- `_classify_tools_fingerprint_restore_guard` in isolation.
  * TIER 2 -- drives `_restore_slot_kv` / `_restore_slot_kv_inner` end to end
    with a fake httpx client (also answering the size guard, which
    runs immediately before this one on the same path, so it must not
    spuriously confound these tests -- same trap the cold-restore hydration
    test's docstring already documents for this exact path).

CAUSAL BOTH DIRECTIONS -- shown by TARGETED
MUTATION (the stronger check,
not a whole-file revert, which vacuously ImportErrors this file's TIER 1
import list regardless of which line is actually wrong):
  * An "always declines" mutant (force `_classify_tools_fingerprint_restore_
    guard` to always return False) must fail
    `test_fingerprint_match_proceeds_and_restores` -- a guard that always
    declines destroys cache reuse for a thread whose tools never changed.
  * A "never declines" mutant (force it to always return True) must fail
    `test_fingerprint_mismatch_declines_restore` -- recreating exactly the
    tools-blindness this guard exists to close.

At least one test drives the real production path: what the tests construct
is a bin's `.json` meta (with or without `tools_fingerprint`) plus an
incoming Slot's `client_meta.tools`; what production DERIVES is the fresh
incoming fingerprint, the stored/incoming comparison, and whether the
restore POST fires.
"""
import json
import os

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
from turbohaul.manager import (
    TurbohaulManager,
    _classify_tools_fingerprint_restore_guard,
    _tools_fingerprint,
)
from turbohaul.slot import Slot


# =============================================================================
# TIER 1 — pure decision table
# =============================================================================
def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_classify_absent_bin_fingerprint_fails_open():
    """An old, pre-tools-fingerprint bin (or a genuinely toolless save -- .get()
    cannot and need not distinguish the two) -- must proceed, never
    manufacture a decline out of a metadata gap."""
    proceed, case = _classify_tools_fingerprint_restore_guard(None, "abc123")
    assert proceed is True
    assert case == "unknown_fail_open"


def test_classify_absent_bin_and_absent_incoming_fails_open():
    proceed, case = _classify_tools_fingerprint_restore_guard(None, None)
    assert proceed is True
    assert case == "unknown_fail_open"


def test_classify_matching_fingerprints_proceeds():
    h = _tools_fingerprint([_tool("x")])
    proceed, case = _classify_tools_fingerprint_restore_guard(h, h)
    assert proceed is True
    assert case == "match"


def test_classify_both_toolless_counts_as_match_shape():
    """Both sides None is handled by the fail-open branch (bin_fingerprint is
    None), not the equality branch -- same OUTCOME (proceed), verified
    explicitly so the two branches' overlap is intentional, not untested."""
    proceed, case = _classify_tools_fingerprint_restore_guard(None, None)
    assert proceed is True


def test_classify_mismatched_fingerprints_declines():
    old = _tools_fingerprint([_tool("a")])
    new = _tools_fingerprint([_tool("a"), _tool("extra_tool_recall")])
    proceed, case = _classify_tools_fingerprint_restore_guard(old, new)
    assert proceed is False
    assert case == "fingerprint_mismatch"


def test_classify_bin_fingerprint_present_incoming_none_declines():
    """A bin recorded WITH tools, incoming has none -- a real, recorded
    divergence (the exact shape of a tools toggle), not an
    absence -- must decline, not fail open."""
    old = _tools_fingerprint([_tool("a")])
    proceed, case = _classify_tools_fingerprint_restore_guard(old, None)
    assert proceed is False
    assert case == "fingerprint_mismatch"


# =============================================================================
# TIER 2 — wiring, driving _restore_slot_kv end to end
# =============================================================================
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


_QWEN = "qwen3.6-27b"
_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_A1 = {"role": "assistant", "content": "A"}
_U2 = {"role": "user", "content": "second user turn"}


def _write_bin(kv_dir, model_tag, port, thread_id, sid, chain, *, clean,
                tools_fingerprint="__omit__", prompt_len=40000, prompt_tokens=500):
    """Mirrors the cold-restore size-guard test's `_write_bin` (same repo
    convention) plus the new optional `tools_fingerprint` field.
    tools_fingerprint="__omit__" (the default) means "don't write the key at
    all" -- simulating an old, pre-tools-fingerprint bin, distinct from explicitly
    passing None (a genuinely toolless save, recorded as such)."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")
    meta = {
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }
    if tools_fingerprint != "__omit__":
        meta["tools_fingerprint"] = tools_fingerprint
    (kv_dir / meta_fn).write_text(json.dumps(meta))
    return bin_fn


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    """Fakes the freshly-spawned engine: POST /apply-template + /tokenize
    satisfy the size guard (which runs immediately before this one on
    the same path) with a generously large incoming token count so it never
    confounds these tests (mirrors the cold-restore hydration test's
    documented fix for the identical trap); POST action=restore records +
    succeeds; GET /slots (shadow-preference lookup) returns empty (no shadow
    bin, so the shadow-preference block never overrides r_bin away from the
    clean anchor these tests write)."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        return _Resp([])

    async def post(self, url, json=None, **kw):
        if "/apply-template" in url:
            return _Resp({"prompt": "x" * 64})
        if "/tokenize" in url:
            return _Resp({"tokens": list(range(20000))})
        self._posts.append((url, json))
        return _Resp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client(posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


def _restore_posts(posts):
    return [(u, b) for (u, b) in posts if "action=restore" in u]


def _slot(tid, chain, tools):
    return Slot.new(
        _QWEN, thread_id=tid, admission_ctx_len=50000,
        admission_hash_chain=chain,
        client_meta={"messages": [_SYS, _U1, _A1, _U2], "tools": tools})


_TOOLS_A = [_tool("a"), _tool("b")]
_TOOLS_B = _TOOLS_A + [_tool("extra_tool_recall")]


@pytest.mark.asyncio
async def test_fingerprint_mismatch_declines_restore(mgr, kv_dir, make_httpx):
    """THE core proof: a bin recorded with tools=A, incoming request now
    carries tools=B (the exact tools-toggle shape) -- the restore POST must
    never fire."""
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t", 0, chain, clean=True,
               tools_fingerprint=_tools_fingerprint(_TOOLS_A))
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])

    await mgr._restore_slot_kv(59500, _QWEN, _slot("t", inc, _TOOLS_B))

    assert _restore_posts(make_httpx) == [], (
        f"restore POSTed despite a recorded tools-fingerprint mismatch: {make_httpx}"
    )
    assert mgr._kv_classifier_last["resolved_from"] == "cold-tools-fingerprint-decline"
    assert mgr._kv_classifier_last["action"] == "fresh"
    assert mgr._kv_cold_tools_fp_decline_count == 1


@pytest.mark.asyncio
async def test_fingerprint_match_proceeds_and_restores(mgr, kv_dir, make_httpx):
    """THE ONE THAT EARNS ITS KEEP: a thread whose tools genuinely never
    changed (bin recorded with tools=A, incoming still tools=A) -- the
    restore POST must still fire. A guard that always declines would pass
    the mismatch test above while destroying every legitimate cold restore."""
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, chain, clean=True,
                        tools_fingerprint=_tools_fingerprint(_TOOLS_A))
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])

    await mgr._restore_slot_kv(59500, _QWEN, _slot("t", inc, _TOOLS_A))

    assert len(_restore_posts(make_httpx)) == 1
    url, body = _restore_posts(make_httpx)[0]
    assert body == {"filename": bin_fn}
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


@pytest.mark.asyncio
async def test_old_bin_no_recorded_fingerprint_still_restores(
        mgr, kv_dir, make_httpx):
    """THE HEALTHY-FLOW QUESTION, driven end to end: a bin saved BEFORE the guard
    (no `tools_fingerprint` key in its meta at all) for a thread whose tools
    are, in fact, currently stable -- must still restore. Absence of a
    recorded fingerprint is not evidence of instability; declining here would
    make every pre-existing bin pay a full reprefill on its very next use for
    no reason connected to actual tools behavior."""
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, chain, clean=True,
                        tools_fingerprint="__omit__")
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])

    await mgr._restore_slot_kv(59500, _QWEN, _slot("t", inc, _TOOLS_A))

    assert len(_restore_posts(make_httpx)) == 1, (
        "an old bin with no recorded tools fingerprint was wrongly declined "
        "instead of failing open"
    )
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


@pytest.mark.asyncio
async def test_old_toolless_bin_toolless_incoming_still_restores(
        mgr, kv_dir, make_httpx):
    """Belt case: an old bin (no field) AND a toolless incoming request --
    both signals are 'nothing recorded', must still proceed."""
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, chain, clean=True,
                        tools_fingerprint="__omit__")
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])

    await mgr._restore_slot_kv(59500, _QWEN, _slot("t", inc, None))

    assert len(_restore_posts(make_httpx)) == 1
    assert mgr._kv_classifier_last["resolved_from"] == "wave-return-clean-restore"


@pytest.mark.asyncio
async def test_guard_env_kill_switch_disables_decline(
        mgr, kv_dir, make_httpx, monkeypatch):
    """Same idiom as every other guard in this file: the env var fully
    disables the decline, restoring pre-guard (tools-blind) behavior."""
    monkeypatch.setenv("TURBOHAUL_COLD_RESTORE_TOOLS_FINGERPRINT_GUARD", "0")
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t", 0, chain, clean=True,
               tools_fingerprint=_tools_fingerprint(_TOOLS_A))
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])

    await mgr._restore_slot_kv(59500, _QWEN, _slot("t", inc, _TOOLS_B))

    assert len(_restore_posts(make_httpx)) == 1, (
        "kill switch did not disable the decline"
    )


# =============================================================================
# TIER 2b — the SAVE side, real path: does _save_slot_kv actually WRITE the
# field the restore-side guard above depends on reading back?
# =============================================================================
class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _SaveClient:
    """Mirrors test_wave_return.py's _SaveClient exactly (same repo
    convention): GET /slots reports one populated slot; action=save
    materializes the temp .bin so the manager's os.replace lands."""

    def __init__(self, slots_payload):
        self._slots_payload = slots_payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        if "action=save" in url and json and "filename" in json:
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        return _SaveResp({"status": "ok"})


@pytest.fixture
def save_httpx(monkeypatch):
    payload = [{"id": 0, "n_prompt_tokens": 100, "id_task": 0}]

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _SaveClient(payload))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return payload


@pytest.mark.asyncio
async def test_real_save_writes_tools_fingerprint_into_meta(mgr, kv_dir, save_httpx):
    """The REAL _save_slot_kv path, not a hand-built meta dict: a slot with
    tools=A saves a bin whose .json meta carries the SAME fingerprint
    _tools_fingerprint(tools=A) computes -- proves the save-side wiring
    actually persists what the restore-side guard reads back, not just that
    the two happen to agree in a hand-written fixture."""
    messages = [_SYS, _U1, _U2]
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": messages, "tools": _TOOLS_A})

    await mgr._save_slot_kv(59500, _QWEN, slot)

    meta_fn = kv_meta_fn(_QWEN, 0, mgr._thread_hash("t"), 59500)
    meta = json.loads((kv_dir / meta_fn).read_text())
    assert meta["tools_fingerprint"] == _tools_fingerprint(_TOOLS_A)


@pytest.mark.asyncio
async def test_real_save_toolless_writes_none_fingerprint(mgr, kv_dir, save_httpx):
    """A toolless save records tools_fingerprint=None explicitly (not the
    key omitted) -- the .get()-based read on the restore side treats this
    identically to an old bin missing the key, which is the intended,
    documented ambiguity (both cases must fail open)."""
    messages = [_SYS, _U1, _U2]
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": messages})

    await mgr._save_slot_kv(59500, _QWEN, slot)

    meta_fn = kv_meta_fn(_QWEN, 0, mgr._thread_hash("t"), 59500)
    meta = json.loads((kv_dir / meta_fn).read_text())
    assert meta["tools_fingerprint"] is None
