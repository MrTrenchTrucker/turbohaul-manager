"""Cold-restore hydrate fix -- the KV disk tier was write-only: a complete
persisted triplet was never read back on a cold restore.

ROOT CAUSE: `_hydrate_ram_from_persist_for_key`
(manager.py, the key-scoped hydrate)
works correctly in isolation -- see the gap-triggered hydrate test,
UNCHANGED by this fix and still passing, which already covers its fast-path
skip, gap-triggered copy, per-key latch, and stale-flag handling. The defect
is structural, one level up: `_restore_slot_kv_inner` -- the COLD-restore
path, the one that runs after a container restart wipes the tmpfs RAM tier
-- does its OWN independent `os.listdir(SLOT_SAVE_DIR)` bin discovery and
NEVER called the hydrate mechanism at all, on either hydrate function, from
anywhere in its call chain (`_scan_kvcache_if_stale` is the only caller of
both hydrate functions, and it has exactly 3 call sites, none inside
`_restore_slot_kv_inner`). A direct call to the hydrate function with the
correct key, against the exact same fixture data, succeeds immediately --
proving the hydrate function itself was never the problem.

These tests exercise the fix through the REAL production entry point,
`_restore_slot_kv` (not the internal hydrate function directly), closing the
routing gap: what does this test CONSTRUCT that
production DERIVES? Answer applied here: nothing about the hydrate key is
hand-picked -- it's read off the same Slot/identity shape a real cold
restore receives, the same way `_restore_slot_kv_inner` derives it
internally (`_bin_identity(...)` then `_thread_hash(...)`).

TRAP (read before touching this file again): a test that leaves ANY faster
path able to serve the request proves nothing. Every test below checks the
RAM-tier directory contents directly (which source actually answered), not
just whether `_restore_slot_kv` returned a token count -- a restore that
returned SOMETHING via a coincidental match is not evidence the hydrate
fired. No test in this file touches anything under /var/lib/turbohaul --
every path is a pytest tmp_path fixture.
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

_MODEL_TAG = "qwen3.6-27b-mtp"
_PORT = 11500
_SYS = {"role": "system", "content": "s" * 200}
_U1 = {"role": "user", "content": "first user turn"}
_A1 = {"role": "assistant", "content": "A"}
_U2 = {"role": "user", "content": "second user turn"}


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
            default_port_base=_PORT,
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


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache_persist"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(d))
    return d


class _RestoreResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _RestoreClient:
    """Fakes the sidecar for the cold-restore path: the action=restore POST
    this fix leads to, AND the /apply-template + /tokenize probe that the
    cold-restore size guard also issues on this same path (a
    realistic non-empty response so that guard doesn't itself decline and
    confound what THIS test is checking). Every call is recorded; tests
    filter by URL rather than assuming restore is the only POST -- two
    features' worth of real traffic share this call site now, on purpose."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "/apply-template" in url:
            return _RestoreResp2({"prompt": "x" * 64})
        if "/tokenize" in url:
            # The cold-restore size guard also runs on this path and
            # declines when the bin looks larger than what's incoming --
            # report a token count that's a genuine strict EXTENSION of any
            # bin these tests write (well above the 14440 _write_triplet
            # stamps), so that unrelated guard doesn't confound what THIS
            # suite is checking. A too-small fake count here would make
            # a real, correct SIZE DECLINE from that guard look
            # like a hydrate failure -- so the fake is kept realistic,
            # not by loosening either guard.
            return _RestoreResp2({"tokens": list(range(20000))})
        return _RestoreResp()


class _RestoreResp2:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _restore_action_posts(posts):
    return [(u, b) for (u, b) in posts if "action=restore" in u]


@pytest.fixture
def restore_posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _RestoreClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


def _write_triplet(root, model_tag, port, thread_id, sid, chain, *, clean=True, stale=None):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    ckpt_fn = bin_fn + ".ckpt"
    (root / bin_fn).write_bytes(b"\x00" * 64)
    (root / ckpt_fn).write_bytes(b"\x00" * 8)
    meta = {
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 14440,
        "prompt_len": 44000, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }
    if stale is not None:
        meta["stale"] = stale
    (root / meta_fn).write_text(json.dumps(meta))
    return bin_fn, ckpt_fn, meta_fn


def _cold_slot(thread_id):
    return Slot.new(
        model_tag=_MODEL_TAG, thread_id=thread_id,
        admission_hash_chain=_prefix_hash_chain([_SYS, _U1, _A1, _U2]),
        admission_ctx_len=50000,
        client_meta={"messages": [_SYS, _U1]},
    )


# ==============================================================================
# THE ROOT-CAUSE FIX, proven through the real production entry point
# ==============================================================================
@pytest.mark.asyncio
async def test_cold_restore_now_hydrates_from_disk_reproducing_production(
        mgr, kv_dir, persist_dir, restore_posts):
    """THE COLD-RESTORE DEFECT, reproduced and fixed. Scenario: after a
    container restart empties SLOT_SAVE_DIR, the very next
    restore logged resolved_from=restore-no-bins despite complete,
    valid triplets sitting in SLOT_PERSIST_DIR. This test recreates exactly
    that shape (RAM tier genuinely empty, one complete non-stale triplet on
    disk) and drives `_restore_slot_kv` -- not the internal hydrate function
    -- the real entry point a cold restore actually calls.

    CAUSAL: fails on pre-fix manager.py (resolved_from=restore-no-bins, RAM
    dir stays empty, zero
    restore POST). Checks WHICH SOURCE answered, not just that something did:
    the RAM directory listing before/after is the ground truth, not the
    return value alone (the trap this class of bug is known for)."""
    thread_id = "t-repro-1"
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn, ckpt_fn, meta_fn = _write_triplet(
        persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)
    assert list(kv_dir.iterdir()) == [], "sanity: RAM tier starts genuinely empty"

    slot = _cold_slot(thread_id)
    result = await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    ram_files_after = {p.name for p in kv_dir.iterdir()}
    assert {bin_fn, ckpt_fn, meta_fn} <= ram_files_after, (
        "the disk triplet was never copied into the RAM tier -- WHICH SOURCE "
        f"answered matters here, not just the return value; RAM contains: {ram_files_after}"
    )
    last = mgr._kv_classifier_last
    assert last["resolved_from"] != "restore-no-bins", (
        "reproduced the cold-restore defect: restore-no-bins despite a "
        "valid persisted triplet"
    )
    assert last["clean_bin_id"] == bin_fn
    # The restore POST must have been attempted against the hydrated bin --
    # proves the fix reaches all the way through, not just the hydrate step.
    action_posts = _restore_action_posts(restore_posts)
    assert len(action_posts) == 1
    url, body = action_posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}


@pytest.mark.asyncio
async def test_cold_restore_hydrate_uses_the_same_identity_derivation_as_save(
        mgr, kv_dir, persist_dir, restore_posts):
    """The hydrate key this fix constructs (model_tag, thread_hash,
    port) must be derived the SAME way `_find_clean_bin`'s callers derive it
    (`_bin_identity(...)` then `_thread_hash(...)`) -- not a parallel, only-
    accidentally-matching scheme. Proven by using a thread_id that only
    matches after going through the real derivation (a plain string, not a
    pre-hashed literal), so a hand-rolled key in the fix would silently
    mismatch and this test would catch it as a routing bug, not a hydrate
    bug (the gap-triggered hydrate test already proves the
    hydrate function itself is correct given a correct key)."""
    thread_id = "agent-production-shaped-identity-string"
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn, ckpt_fn, meta_fn = _write_triplet(
        persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)

    slot = _cold_slot(thread_id)
    await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    assert (kv_dir / bin_fn).exists(), (
        "hydrate ran but the key it computed did not match the meta's own "
        "thread_hash -- a derivation mismatch between this fix's key and "
        "the save-time key"
    )


# ==============================================================================
# CAUSAL BOTH DIRECTIONS at the NEW call site: a hydrate that always fires
# and a hydrate that never fires must each fail a DIFFERENT test.
# ==============================================================================
@pytest.mark.asyncio
async def test_cold_restore_hydrate_does_not_touch_disk_when_ram_already_has_it(
        mgr, kv_dir, persist_dir, restore_posts, monkeypatch):
    """A hydrate that ALWAYS copies (ignoring the fast path) must fail this
    test: it would hammer disk on every cold restore even when RAM already
    holds a matching file, defeating the whole point of the RAM tier. RAM
    already has a matching bin for this key -- the persist_dir listdir must
    never be touched, exactly mirroring
    test_ram_hit_never_touches_persist_dir's acceptance criterion, now
    checked from the NEW call site instead of the pre-existing one."""
    import os
    thread_id = "t-ram-hit-1"
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_triplet(kv_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)
    _write_triplet(persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)

    listed_dirs = []
    real_listdir = os.listdir

    def spy_listdir(path):
        listed_dirs.append(str(path))
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", spy_listdir)

    slot = _cold_slot(thread_id)
    await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    assert str(persist_dir) not in listed_dirs, (
        "a RAM hit at the cold-restore call site still touched persist_dir "
        "-- the fast path was bypassed from the new call site"
    )


@pytest.mark.asyncio
async def test_cold_restore_still_refuses_when_disk_also_has_nothing(
        mgr, kv_dir, persist_dir, restore_posts):
    """A hydrate that NEVER copies -- the exact pre-fix defect -- must fail
    a DIFFERENT test than the one above: this one proves the genuinely-
    empty case (no bin anywhere, RAM or disk) is still correctly refused,
    not accidentally papered over by always claiming success."""
    slot = _cold_slot("t-truly-nothing")
    result = await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    assert result is None
    assert mgr._kv_classifier_last["resolved_from"] == "restore-no-bins"
    assert list(kv_dir.iterdir()) == []
    assert restore_posts == []


@pytest.mark.asyncio
async def test_cold_restore_hydrate_respects_stale_flag(mgr, kv_dir, persist_dir, restore_posts):
    """The stale-flag case, re-checked from the new call site: a
    stale-marked persisted triplet must NOT be hydrated even though it is
    otherwise a complete, well-formed match for this identity."""
    thread_id = "t-stale-1"
    chain = _prefix_hash_chain([_SYS, _U1])
    _write_triplet(persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain,
                    clean=True, stale=True)

    slot = _cold_slot(thread_id)
    result = await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    assert result is None
    assert mgr._kv_classifier_last["resolved_from"] == "restore-no-bins"
    assert list(kv_dir.iterdir()) == [], "a stale-marked triplet was hydrated anyway"


# ==============================================================================
# OBSERVABILITY, non-negotiable regardless of which fix lands
# ==============================================================================
def test_kv_hydrate_log_line_on_successful_copy(mgr, kv_dir, persist_dir, caplog):
    """Every hydrate attempt must log key/matched/copied -- this is the
    line whose ABSENCE would let a silent failure like this
    defect go undetected."""
    import logging
    caplog.set_level(logging.INFO, logger=manager_mod.log.name)
    thread_id = "t-log-1"
    chain = _prefix_hash_chain([_SYS, _U1])
    th = TurbohaulManager._thread_hash(thread_id)
    _write_triplet(persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)

    key = (_MODEL_TAG, th, _PORT)
    mgr._hydrate_ram_from_persist_for_key(str(kv_dir), str(persist_dir), key)

    lines = [r.message for r in caplog.records if r.message.startswith("KV_HYDRATE")]
    assert lines, "no KV_HYDRATE log line was emitted at all"
    assert "matched=3" in lines[0] and "copied=3" in lines[0], (
        f"log line did not report the expected counts: {lines[0]!r}"
    )


def test_kv_hydrate_log_line_says_why_on_zero(mgr, kv_dir, persist_dir, caplog):
    """THE line that must exist: 'why zero if zero.' Nothing on
    disk for this key -> the log must say so explicitly, not just 'matched=0
    copied=0' with no reason a human can grep for."""
    import logging
    caplog.set_level(logging.INFO, logger=manager_mod.log.name)
    th = TurbohaulManager._thread_hash("t-nothing-here")
    key = (_MODEL_TAG, th, _PORT)

    mgr._hydrate_ram_from_persist_for_key(str(kv_dir), str(persist_dir), key)

    lines = [r.message for r in caplog.records if r.message.startswith("KV_HYDRATE")]
    assert lines, "no KV_HYDRATE log line was emitted for the zero case"
    assert "matched=0" in lines[0] and "copied=0" in lines[0]
    assert "reason=" in lines[0], (
        f"zero-match case did not name WHY it was zero: {lines[0]!r}"
    )


def test_kv_hydrate_log_line_on_ram_already_present(mgr, kv_dir, persist_dir, caplog):
    """The fast-path early-return must ALSO be
    observable, not just the two paths above -- confirmed here, and see the
    next test for the real risk this early-return carries."""
    import logging
    caplog.set_level(logging.INFO, logger=manager_mod.log.name)
    thread_id = "t-log-2"
    chain = _prefix_hash_chain([_SYS, _U1])
    th = TurbohaulManager._thread_hash(thread_id)
    _write_triplet(kv_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)

    key = (_MODEL_TAG, th, _PORT)
    mgr._hydrate_ram_from_persist_for_key(str(kv_dir), str(persist_dir), key)

    lines = [r.message for r in caplog.records if r.message.startswith("KV_HYDRATE")]
    assert lines and "reason=ram-already-present" in lines[0]


# ==============================================================================
# FAST-PATH early-return, confirmed as a REAL but SEPARATE risk from the
# root cause above -- not what fired originally (RAM was genuinely, fully
# empty in the original failure), but real: a single stray leftover file
# blocks the entire hydrate, silently, for a key that otherwise has nothing
# usable in RAM at all.
# ==============================================================================
@pytest.mark.asyncio
async def test_orphaned_ram_file_blocks_hydrate_even_with_no_usable_ram_copy(
        mgr, kv_dir, persist_dir, restore_posts):
    """A lone .json with NO matching .bin (a plausible half-write leftover)
    satisfies the fast path's filename-only check and blocks hydrate --
    even though RAM has nothing actually restorable for this key. This is a
    real risk (the fast-path early-return), reported separately
    from the root cause because it did not fire in the original failure (RAM
    was genuinely empty there) -- but it is real and this change does not fix
    it, which is out of scope (this fix targets the ROOT CAUSE of the
    defect above; this is a related, secondary risk, named explicitly rather
    than silently left for someone to rediscover)."""
    thread_id = "t-orphan-1"
    chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn, ckpt_fn, meta_fn = _write_triplet(
        persist_dir, _MODEL_TAG, _PORT, thread_id, sid=0, chain=chain, clean=True)
    # Orphan: only the .json half-write survives in RAM, no .bin.
    (kv_dir / meta_fn).write_text((persist_dir / meta_fn).read_text())

    slot = _cold_slot(thread_id)
    result = await mgr._restore_slot_kv(_PORT, _MODEL_TAG, slot)

    # Documents the confirmed-real gap: hydrate did NOT run (fast path saw
    # the orphaned .json and returned), so the .bin never arrived, so the
    # cold path's own os.listdir(SLOT_SAVE_DIR)-based bin discovery -- which
    # filters on .bin specifically -- finds nothing either.
    assert not (kv_dir / bin_fn).exists(), (
        "if this now passes, the fast-path gap has been fixed "
        "elsewhere -- update this test's assertion and its docstring, do "
        "not just delete it"
    )
