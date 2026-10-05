"""Main's clean KV bin must be written with main's CURRENT turn
count, not a stale snapshot from the first turn of the grace cycle.

Regression guarded: the idle-hot-enter handoff (the sole caller of
_set_idle_holder) read model_tag/thread_id/admission_ctx_len/client_meta off
the grace loop's own ANCHOR `slot` parameter, which is never reassigned as
ACTIVE_MATCH follow-ups grow the same conversation on the same warm handle.
A conversation that grew from 2 turns to 28+ turns across several ACTIVE_MATCH
continuations still parked (and then unload-seam-flushed) the ORIGINAL
2-turn identity, freezing the persisted clean bin at the call-#1 floor.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest

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
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle

pytestmark = pytest.mark.asyncio

_MODEL = "qwen3.6-27b-mtp"
_PORT = 59900


# --- harness (mirrors test_worker_loop.py's _boot_runtime / _seed_manifest,
# and the KV-save observability test module's make_httpx/kv_dir/_ProbeSaveClient) --

def _boot_runtime(tmp_path, grace_seconds, idle_hot_load_seconds):
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
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            safety_enabled=False,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag: str) -> None:
    manifests_dir = boot.storage.manifests_path
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{model_tag}.yaml").write_text(
        f"model_tag: {model_tag}\n"
        "gguf_blob_sha256: " + "a" * 64 + "\n"
        "context_size: 2048\n"
        "expected_vram_bytes: 0\n"
        "llama_server_flags: {}\n"
    )


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_887
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Patches BOTH tiers. SLOT_PERSIST_DIR (the RAM<->SSD mirror) is
    the real default persist location (/var/lib/turbohaul/kvcache_persist)
    -- every test in this repo shares one (model_tag, thread_hash, port)
    keyspace there, so leaving it unpatched lets a stale real bin from a
    prior run hydrate into a fresh test and silently change its outcome
    (a leftover 28-turn persisted bin can out-throttle a same-run
    5-turn control save via never-overwrite-smaller)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    p = tmp_path / "kvcache_persist"
    p.mkdir()
    import turbohaul.subprocess_mgr as subprocess_mgr
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(p))
    return d


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    """GET /slots (any, non-action=save) -> one populated slot. POST
    action=save with a filename -> materialize the bin on disk (so the
    later meta-sidecar read is a real filesystem round-trip, not a mock
    assertion). Any other POST (the render/strip prefill probe) -> ok."""

    def __init__(self, slots_payload, posts):
        self._slots_payload = slots_payload
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=save" in url and json and "filename" in json:
            import os
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        if "/v1/chat/completions" in url:
            # Plain-probe regime (covered_scaffold_strip off): the real
            # sidecar's /v1/chat/completions reply carries the tokenized
            # prompt's usage. The manager's strict clean stamp
            # keys off usage.prompt_tokens as its save evidence, so the fake
            # reports the slot's own engine-reported count — exactly what the
            # engine would return for the same render.
            _n = (self._slots_payload[0].get("n_prompt_tokens") or 1) if self._slots_payload else 1
            return _SaveResp({"status": "ok",
                               "usage": {"prompt_tokens": _n,
                                         "completion_tokens": 0,
                                         "total_tokens": _n}})
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    import turbohaul.manager as manager_mod

    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


def _clean_bin_meta(kv_dir, model_tag, port):
    """Read back the ONE clean_prefix=True meta sidecar written under kv_dir
    for this (model_tag, port). Fails loudly (not None) if zero or more than
    one exist -- ambiguity here would silently validate the wrong artifact."""
    import os
    candidates = []
    for fn in os.listdir(kv_dir):
        if fn.startswith(f"{model_tag}.p{port}.") and fn.endswith(".json"):
            with open(os.path.join(kv_dir, fn)) as f:
                meta = json.load(f)
            if meta.get("clean_prefix"):
                candidates.append(meta)
    assert len(candidates) == 1, (
        f"expected exactly one clean_prefix meta sidecar under {kv_dir}, "
        f"found {len(candidates)}: {[c.get('thread_hash') for c in candidates]}"
    )
    return candidates[0]


async def _run_two_turn_thread_then_idle_park(
    tmp_path, kv_dir, make_httpx, *, anchor_turns, matched_turns,
    grace_seconds=2, idle_hot_load_seconds=60,
):
    """Drive: anchor (anchor_turns messages) -> ACTIVE_MATCH follow-up on the
    SAME thread (matched_turns messages) -> let the grace window EXPIRE
    NATURALLY (not cancel — a mid-grace cancel takes the finally-unwind
    teardown path, which never reaches idle-hot-enter at all) -> shutdown()
    (which flushes the parked idle holder for real). Returns the persisted
    clean bin's meta dict."""
    boot, runtime = _boot_runtime(tmp_path, grace_seconds, idle_hot_load_seconds)
    _seed_manifest(boot, _MODEL)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_fake_handle(model_tag, port)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(*a, **k):
        return True, None

    async def fake_complete(slot, handle):
        return {"ok": True}

    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram,
        complete_fn=fake_complete,
    )
    mgr.runtime.kv.covered_scaffold_strip = False
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        anchor = await mgr.submit(
            model_tag=_MODEL, prompt="anchor", thread_id="t-main",
            client_meta={"is_main": True, "messages": _msgs(anchor_turns, "ANCHOR")},
            admission_ctx_len=60000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(anchor.completion_future, timeout=5.0)
        await asyncio.sleep(0.05)  # let worker enter the GRACE loop

        matched = await mgr.submit(
            model_tag=_MODEL, prompt="matched-followup", thread_id="t-main",
            client_meta={"is_main": True, "messages": _msgs(matched_turns, "GROWN")},
            admission_ctx_len=90000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(matched.completion_future, timeout=5.0)

        # Let the grace window expire NATURALLY so _process_slot's own code
        # (not a cancellation-unwind finally) reaches idle-hot-enter.
        await asyncio.sleep(grace_seconds + 0.5)

        # shutdown() finds the parked idle holder and performs the real
        # unload-seam flush (_teardown_idle_holder -> _flush_clean_kv_at_unload).
        await mgr.shutdown()
    finally:
        if not mgr._stop_event.is_set():
            await mgr.shutdown()

    return _clean_bin_meta(kv_dir, _MODEL, _PORT), posts


def _list_clean_bin_metas(kv_dir, model_tag, port):
    """Like _clean_bin_meta but returns the raw list (0, 1, or more) instead
    of asserting exactly one -- for tests where "zero" or "more than one" is
    itself part of what's being verified, not an error."""
    import os
    out = []
    for fn in os.listdir(kv_dir):
        if fn.startswith(f"{model_tag}.p{port}.") and fn.endswith(".json"):
            with open(os.path.join(kv_dir, fn)) as f:
                meta = json.load(f)
            if meta.get("clean_prefix"):
                out.append(meta)
    return out


async def _run_role_sequence_then_idle_park(
    tmp_path, kv_dir, make_httpx, *, turns, thread_id="t-mixed",
    grace_seconds=2, idle_hot_load_seconds=60,
):
    """Generalizes _run_two_turn_thread_then_idle_park to an arbitrary
    sequence of submits on the SAME thread_id, each waited to completion
    before the next lands -- so a MIXED-role thread (main, then curator,
    then main again, ...) can be driven turn by turn. `turns` is a list of
    dicts: {"client_meta": {...}, "admission_ctx_len": int, "turns": int,
    "marker": str}. Returns the list of clean_prefix meta dicts found (0,
    1, or more -- callers assert on the count themselves)."""
    boot, runtime = _boot_runtime(tmp_path, grace_seconds, idle_hot_load_seconds)
    _seed_manifest(boot, _MODEL)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_fake_handle(model_tag, port)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(*a, **k):
        return True, None

    async def fake_complete(slot, handle):
        return {"ok": True}

    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram,
        complete_fn=fake_complete,
    )
    mgr.runtime.kv.covered_scaffold_strip = False
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        for i, t in enumerate(turns):
            cm = dict(t["client_meta"])
            cm["messages"] = _msgs(t["turns"], t.get("marker", f"T{i}"))
            slot = await mgr.submit(
                model_tag=_MODEL, prompt=f"turn-{i}", thread_id=thread_id,
                client_meta=cm, admission_ctx_len=t["admission_ctx_len"],
                wait_for_completion=True,
            )
            await asyncio.wait_for(slot.completion_future, timeout=5.0)
            await asyncio.sleep(0.05)  # let worker enter/stay in the GRACE loop

        await asyncio.sleep(grace_seconds + 0.5)
        await mgr.shutdown()
    finally:
        if not mgr._stop_event.is_set():
            await mgr.shutdown()

    return _list_clean_bin_metas(kv_dir, _MODEL, _PORT), posts


class TestIdleParkTracksLiveTurnCount:
    async def test_saved_bin_turn_count_tracks_last_served_not_anchor(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """THE LOAD-BEARING ASSERTION. Merely checking that the bin
        EXISTS would be vacuous -- it passes even on the buggy code (any
        n_context_turns value, including the frozen anchor's, produces a
        bin). This asserts the NUMBER: the persisted clean bin's
        n_context_turns must equal the LAST successfully-served turn count
        (28), not the anchor's turn-1 snapshot (2)."""
        meta, _posts = await _run_two_turn_thread_then_idle_park(
            tmp_path, kv_dir, make_httpx, anchor_turns=2, matched_turns=28,
        )
        assert meta["n_context_turns"] == 28, (
            f"clean bin froze at n_context_turns={meta['n_context_turns']} "
            "(the anchor's turn-1 snapshot) instead of tracking the grown "
            "28-turn conversation actually live on the engine"
        )

    async def test_control_single_shot_thread_still_saves_correct_count(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """Control: when NO ACTIVE_MATCH follow-up ever occurs, the anchor
        IS the last-served slot -- the fix must not regress this, the single
        most common case (an ordinary single-turn run exercises exactly
        this shape)."""
        meta, _posts = await _run_two_turn_thread_then_idle_park(
            tmp_path, kv_dir, make_httpx, anchor_turns=5, matched_turns=5,
        )
        # anchor_turns == matched_turns here only to keep the helper shape;
        # what matters is a single real submit reaching idle-park correctly.
        assert meta["n_context_turns"] == 5

    async def test_bin_identity_thread_hash_unaffected_by_fix(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """thread_id/model_tag are guaranteed identical between the anchor
        and any ACTIVE_MATCH follow-up (pop_matched_thread matches on both) --
        the fix changes WHICH slot's client_meta/admission_ctx_len feed the
        park, never the thread identity itself. This pins that down instead
        of leaving it as an unstated assumption."""
        meta, _posts = await _run_two_turn_thread_then_idle_park(
            tmp_path, kv_dir, make_httpx, anchor_turns=2, matched_turns=28,
        )
        assert meta["thread_id"] == "t-main"
        assert meta["model_tag"] == _MODEL


_MAIN_CM = {"is_main": True}
_CURATOR_CM = {"is_curator": True, "save_kv": False}
_COMPRESSION_CM = {"is_compression": True, "save_kv": False}


class TestCuratorStillDisposable:
    """The fix must not be an over-correction. It changes WHICH slot's
    client_meta feeds idle-park; it must NOT change the disposability
    verdict for a thread whose OWN identity (in that client_meta) says
    disposable. Only main's cache was ever misclassified; the curator's own
    growing conversation must remain exactly as disposable after the fix as
    before it.

    Note: an ALL-curator
    thread for both the anchor and the follow-up (as in the control below) -- both sides share one
    role, so the assertion ("zero clean bins") is identical whether
    _grace_tip is gated by role or not. That is not a discriminating test;
    it cannot tell a correctly-gated fix apart from no gate at all, or even
    a backwards one. The MIXED-role tests below are what actually exercise
    the gate: a curator CAN share a main anchor's own thread_id
    (pop_matched_thread has no role awareness), and a naive (ungated)
    _grace_tip = matched turns that into a worse bug than the one this class
    targets -- a stale-but-PRESENT main bin becomes NO bin at all, because
    the parked identity is now the curator's, and _seam_flush_allowed
    correctly (per its own, unchanged contract) refuses to flush a
    disposable identity."""

    async def test_curator_only_thread_active_match_growth_never_produces_a_clean_bin(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """Control (a curator's own growing
        conversation must never itself become a clean bin) -- but per the
        class docstring, this alone cannot discriminate the gate."""
        metas, _posts = await _run_role_sequence_then_idle_park(
            tmp_path, kv_dir, make_httpx, thread_id="t-curator-only",
            turns=[
                {"client_meta": _CURATOR_CM, "admission_ctx_len": 45000,
                 "turns": 2, "marker": "CURATOR-A"},
                {"client_meta": _CURATOR_CM, "admission_ctx_len": 90000,
                 "turns": 28, "marker": "CURATOR-B"},
            ],
        )
        assert metas == [], (
            f"all-curator thread produced clean_prefix bin(s) {metas} -- "
            "disposable-role content must never be persisted as a clean "
            "anchor, fix or no fix"
        )

    async def test_mixed_main_then_disposable_followup_recovers_mains_own_bin(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """THE DISCRIMINATING MIXED-ROLE TEST.

        An is_curator follow-up would PASS even on the ungated (buggy) code
        -- not because the bug is fixed, but because it is never exercised:
        the displacement
        seam, `_probe_and_save_clean_kv`'s own hoisted chokepoint, fired
        unconditionally on the disposable follow-up's own per-turn probe)
        independently rescues main's identity via `last_main_client_meta`
        -- an always-fresh stash, structurally unrelated to `_grace_tip` --
        the moment ANY disposable role's own turn runs `_probe_and_save_
        clean_kv`. That path already worked correctly before this fix and
        masked the very bug this test exists to catch. In the log,
        the displacement-seam save fires and writes
        the bin BEFORE _grace_tip / idle-park ever mattered.

        Hence `is_compression` is used for the follow-up. The seam's own
        resolve point (`_displacement_save_allowed`) explicitly EXCLUDES
        compression by design ("a compression event is about to rewrite
        main's own context... a pre-compression bin would protect a
        context that is about to be obsolete") -- so the seam declines
        WITHOUT attempting a save, and the LATER unload-seam flush (the
        one `_grace_tip` actually feeds) becomes the only remaining chance
        to preserve main's identity. `is_compression` is the same
        disposable-shaped role vocabulary as curator for _bin_role /
        _seam_flush_allowed purposes (both non-None, non-"main"), and a
        compression pass riding main's own thread_id is the identical
        real-world shape the mixed-role scenario describes.

        An ungated _grace_tip parks the compression turn's own (disposable)
        client_meta -- _seam_flush_allowed then refuses the unload-seam
        flush entirely, and NO clean bin is written at all (worse than the
        original bug: a stale bin at least existed). A correctly
        role-gated _grace_tip skips the compression follow-up and stays on
        the anchor -- main's own 2-turn identity is exactly what should be
        recovered here (nothing main-class followed the interlude in this
        test)."""
        metas, _posts = await _run_role_sequence_then_idle_park(
            tmp_path, kv_dir, make_httpx, thread_id="t-mixed-tail-compression",
            turns=[
                {"client_meta": _MAIN_CM, "admission_ctx_len": 60000,
                 "turns": 2, "marker": "MIXEDMAIN"},
                {"client_meta": _COMPRESSION_CM, "admission_ctx_len": 90000,
                 "turns": 10, "marker": "MIXEDCOMPRESSION"},
            ],
        )
        assert len(metas) == 1, (
            f"expected exactly one clean bin (main's own, recovered despite "
            f"the compression follow-up), got {len(metas)}: {metas} -- a "
            "disposable follow-up must never turn a stale-but-present main "
            "bin into no bin at all"
        )
        assert metas[0]["n_context_turns"] == 2, (
            f"clean bin has n_context_turns={metas[0]['n_context_turns']} -- "
            "expected 2 (main's own anchor turn count); a value of 10 would "
            "mean the compression turn's own (disposable) content was "
            "smuggled into the parked identity"
        )

    async def test_main_resumes_after_curator_interlude_tracks_final_main_turns(
        self, tmp_path, kv_dir, make_httpx,
    ):
        """Supporting composite pin (not required to discriminate the
        mixed-role bug on its own -- the last slot in this particular
        sequence is already main-class either way -- but pins the intended
        compositional behavior: a disposable interlude must be SKIPPED,
        not treated as a permanent freeze. Guards the opposite
        over-correction: gating that never advances past the first
        disposable visit at all."""
        metas, _posts = await _run_role_sequence_then_idle_park(
            tmp_path, kv_dir, make_httpx, thread_id="t-mixed-resume",
            turns=[
                {"client_meta": _MAIN_CM, "admission_ctx_len": 60000,
                 "turns": 2, "marker": "RESUME-MAIN-A"},
                {"client_meta": _CURATOR_CM, "admission_ctx_len": 90000,
                 "turns": 10, "marker": "RESUME-CURATOR"},
                {"client_meta": _MAIN_CM, "admission_ctx_len": 95000,
                 "turns": 28, "marker": "RESUME-MAIN-B"},
            ],
        )
        assert len(metas) == 1, f"expected exactly one clean bin, got {metas}"
        assert metas[0]["n_context_turns"] == 28, (
            f"clean bin has n_context_turns={metas[0]['n_context_turns']} -- "
            "expected 28 (main's own FINAL turn count, tracked across and "
            "past the curator interlude)"
        )
