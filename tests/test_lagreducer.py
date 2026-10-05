"""SAFE lag-reducer (clean-bin) — SAVE-path manager delta.

Covers the 3 changes in manager.py (DESIGN_LAGREDUCER.md), SAVE path only:

  1. NEVER-DEMOTE (FM-2) — a force_clean save that resolves to
     _eff_force_clean=False (len(populated) != 1) must NOT overwrite/demote an
     existing clean_prefix=True anchor for the same (model_tag, thread_hash, port).
  2. THROTTLE (FM-5) — _probe_and_save_clean_kv re-saves the multi-GB clean
     anchor only when the incoming clean-prefix chain grew by
     >= LAGREDUCER_MIN_GROWTH_TURNS turns; skips below it; keeps the
     never-overwrite-with-smaller (prompt_len) belt.
  3. FM-7 degenerate — a 0-turn incoming chain never forces a save.

The single-series gate, the #103 clean stamp, and the RESTORE path are
NOT touched (still covered by test_classifier.py / test_wave_return.py /
test_kv_policy.py).
"""
import json
import logging
import os
import types

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
from turbohaul.manager import KV_DECLINE_NEVER_OVERWRITE_SMALLER, TurbohaulManager
from turbohaul.slot import Slot


# --- fixtures (mirror test_classifier.py / test_wave_return.py) --------------
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


# --- save/probe httpx fake: GET /slots + records every POST + materialises temp bin
class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    """Fakes the sidecar for _probe_and_save_clean_kv + _save_slot_kv:
      * GET /slots           -> the configured populated-slots payload
      * POST (probe / save)  -> recorded; action=save materialises the temp .bin so
                                the manager's os.replace lands.
    """

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
    """Factory: patch manager_mod.httpx with a fake reporting `payload` populated
    slots; returns the list that records every POST (probe + save)."""
    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(mgr):
    """The covered-scaffold strip ships default-ON, which swaps the clean-probe prefill
    TRANSPORT to /apply-template + /completion. This suite verifies the
    TRANSPORT-INDEPENDENT never-demote / throttle / grow-monotone / zero-turn-guard SAVE logic
    against the flag-OFF (byte-identical-to-today) messages transport; the default-ON
    render+strip transport + clean-bin write is covered in
    tests/test_covered_scaffold_strip.py."""
    mgr.runtime.kv.covered_scaffold_strip = False


# path-safe example model tag (matches the other suites)
_QWEN = "qwen3.6-27b"
_PORT = 59500


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
    return meta_fn


def _msgs(k):
    """k structured turns -> _prefix_hash_chain(...) is length k (1 hash per turn)."""
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _read_meta(kv_dir, model_tag, sid, thread_id, port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    return json.loads((kv_dir / kv_meta_fn(model_tag, sid, th, port)).read_text())


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port)


def _n_save_posts(posts):
    return sum(1 for (url, _j) in posts if "action=save" in url)


# ================================================================================
# 1. NEVER-DEMOTE (FM-2) — _save_slot_kv force_clean path
# ================================================================================
@pytest.mark.asyncio
async def test_never_demote_multislot_force_clean_preserves_anchor(mgr, kv_dir, make_httpx):
    """A force_clean save with >1 populated slot resolves _eff_force_clean=False and
    would stamp clean_prefix=False -> it MUST abort rather than demote the existing
    clean anchor (True->False), and must not write ANY non-clean bin under the anchor
    thread identity."""
    anchor_chain = _prefix_hash_chain(_msgs(12))
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, anchor_chain, prompt_len=100000)
    before = _read_meta(kv_dir, _QWEN, 0, "t")
    assert before["clean_prefix"] is True

    # engine reports TWO populated slots -> _eff_force_clean=False on a force_clean save
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100},
                        {"id": 1, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=_msgs(20), client_meta={})

    await mgr._save_slot_kv(_PORT, _QWEN, slot, force_clean=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["clean_prefix"] is True                 # NOT demoted
    assert after["n_context_turns"] == before["n_context_turns"]
    assert after["hash_chain"] == before["hash_chain"]   # anchor bytes untouched
    # co-resident slot1 KV was NOT written under the anchor thread identity
    th = TurbohaulManager._thread_hash("t")
    assert not (kv_dir / kv_meta_fn(_QWEN, 1, th, _PORT)).exists()
    assert _n_save_posts(posts) == 0                      # aborted before the save POST


@pytest.mark.asyncio
async def test_single_slot_force_clean_still_overwrites_clean_bin(mgr, kv_dir, make_httpx):
    """Precision guard: the never-demote abort fires ONLY on the multi-slot demote
    case. A legitimate single-slot force_clean re-save (_eff_force_clean=True) still
    (over)writes the clean bin -> the grow path is not blocked.

    A force_clean save must carry probe evidence (in production the
    only force_clean caller is the probe path, which records the clean-render
    token count first) — pre-seed it exactly as the probe would, then exercise
    the overwrite/grow semantics this test exists to pin."""
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, _prefix_hash_chain(_msgs(10)))
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])  # exactly ONE populated slot
    # The register holds CleanPrefillEvidence — the
    # probe's count BOUND to THIS context's turn-hash chain (coherent
    # extent: the slot reports the same 100, so the coherence gate passes by
    # construction; the chain is this 15-turn save context's).
    from turbohaul.manager import CleanPrefillEvidence, _chain_fp
    mgr._clean_prefill_tokens[_PORT] = CleanPrefillEvidence(
        count=100, chain_fp=_chain_fp(_prefix_hash_chain(_msgs(15))), arm="plain")
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(15)})

    await mgr._save_slot_kv(_PORT, _QWEN, slot, force_clean=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["clean_prefix"] is True
    assert after["n_context_turns"] == 15                # grew via the legit clean save
    assert _n_save_posts(posts) == 1


# ================================================================================
# 2. THROTTLE (FM-5) — _probe_and_save_clean_kv pre-check
# ================================================================================
@pytest.mark.asyncio
async def test_throttle_skips_below_min_growth(mgr, kv_dir, make_httpx):
    """A clean anchor exists at 10 turns; incoming grew only +3 (< MIN_GROWTH=4) ->
    the throttle SKIPS the multi-GB re-save (anchor untouched, no probe/save POST)."""
    # SPEC-V2 REWORK: probe defers disk to unload; this test pins the flush/deferred path
    # (spec-v2 default threshold is 1 — pin 4 on the instance so the skip branch is exercised)
    mgr.LAGREDUCER_MIN_GROWTH_TURNS = 4
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, _prefix_hash_chain(_msgs(10)),
                     prompt_len=40000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(13)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot)

    assert _read_meta(kv_dir, _QWEN, 0, "t")["n_context_turns"] == 10   # not re-saved
    assert posts == []                                                 # skipped pre-probe


@pytest.mark.asyncio
async def test_throttle_fires_at_min_growth(mgr, kv_dir, make_httpx):
    """Incoming grew exactly +4 (== MIN_GROWTH) -> the throttle allows the re-save;
    the anchor grows to the incoming turn count and stays clean."""
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, _prefix_hash_chain(_msgs(10)),
                     prompt_len=40000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(14)}, admission_ctx_len=50000)

    # SPEC-V2 REWORK: probe defers disk to unload; this test pins the flush/deferred path
    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["n_context_turns"] == 14        # re-saved: anchor grew 10 -> 14
    assert after["clean_prefix"] is True         # still a valid anchor
    assert _n_save_posts(posts) == 1             # the re-save fired


@pytest.mark.asyncio
async def test_no_clean_bin_saves_unconditionally(mgr, kv_dir, make_httpx):
    """No clean anchor yet -> the throttle does not gate: the first clean save fires
    regardless of turn count (bootstrap the anchor)."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)

    # SPEC-V2 REWORK: probe defers disk to unload; this test pins the flush/deferred path
    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["clean_prefix"] is True
    assert after["n_context_turns"] == 6
    assert _n_save_posts(posts) == 1


# ================================================================================
# 3. GROW / MONOTONE + never-overwrite-with-smaller belt
# ================================================================================
@pytest.mark.asyncio
async def test_grow_monotone_across_resaves(mgr, kv_dir, make_httpx):
    """n_context_turns increases across re-saves and never regresses: 10 -> 16 -> 22,
    each jump >= MIN_GROWTH."""
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, _prefix_hash_chain(_msgs(10)),
                     prompt_len=40000)

    # SPEC-V2 REWORK: probe defers disk to unload; this test pins the flush/deferred path
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._probe_and_save_clean_kv(
        _handle(), Slot.new(_QWEN, thread_id="t", context=None,
                            client_meta={"messages": _msgs(16)}, admission_ctx_len=50000),
        save_to_disk=True)
    assert _read_meta(kv_dir, _QWEN, 0, "t")["n_context_turns"] == 16

    posts2 = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._probe_and_save_clean_kv(
        _handle(), Slot.new(_QWEN, thread_id="t", context=None,
                            client_meta={"messages": _msgs(22)}, admission_ctx_len=60000),
        save_to_disk=True)
    grown = _read_meta(kv_dir, _QWEN, 0, "t")
    assert grown["n_context_turns"] == 22        # strictly grew, never shrank
    assert grown["clean_prefix"] is True
    assert _n_save_posts(posts) == 1 and _n_save_posts(posts2) == 1


@pytest.mark.asyncio
async def test_never_overwrite_with_smaller_belt(mgr, kv_dir, make_httpx):
    """The prompt_len belt still holds: a saved clean bin whose prompt_len >= the
    incoming ctx len is NOT overwritten (even though the turn count grew a lot)."""
    big_chain = _prefix_hash_chain(_msgs(20))
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, big_chain, prompt_len=60000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    # admission_ctx_len (50000) < saved prompt_len (60000) -> belt returns early,
    # even though incoming turn count (40) would clear the throttle.
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(40)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["n_context_turns"] == 20        # unchanged (not overwritten-with-smaller)
    assert after["prompt_len"] == 60000
    assert posts == []


# ================================================================================
# 3b. Equal-length saves: equal length is not automatically the same
# content, and is not automatically different content either: the belt must
# tell the two apart via the content hash chain already computed for both
# sides, not via length alone.
# ================================================================================
@pytest.mark.asyncio
async def test_equal_length_identical_content_still_declines(mgr, kv_dir, make_httpx, caplog):
    """An equal-length incoming save whose content hash chain matches what's
    already saved is NOT freshness -- it's the exact same content, and
    overwriting it is pure wasted SSD I/O. The belt must still decline
    this via its OWN never-overwrite-with-smaller reason, same as a
    plain length compare would.

    Identical content necessarily means identical turn count too (the
    chain IS one hash per turn), so the SEPARATE zero-growth throttle a
    few lines below would ALSO decline this scenario if the belt let it
    through -- meaning "no save happened" alone does not actually pin the
    belt's own content-identical branch; a mutation that silently disabled
    it here would still pass on the throttle's coattails. Assert the
    specific reason_name via the real emitted log line instead, so the
    two decline paths can't be confused for each other."""
    chain = _prefix_hash_chain(_msgs(10))
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, chain, prompt_len=50000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                     client_meta={"messages": _msgs(10)}, admission_ctx_len=50000)

    with caplog.at_level(logging.INFO):
        await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["hash_chain"] == chain            # unchanged
    assert after["prompt_len"] == 50000
    assert posts == []
    decline_lines = [r.getMessage() for r in caplog.records
                      if r.getMessage().startswith("KV_SAVE_DECLINE")]
    assert len(decline_lines) == 1, decline_lines
    assert f"reason_name={KV_DECLINE_NEVER_OVERWRITE_SMALLER}" in decline_lines[0], (
        f"expected the belt's own never-overwrite reason, got: {decline_lines[0]}"
    )


@pytest.mark.asyncio
async def test_equal_length_different_content_is_allowed_through(mgr, kv_dir, make_httpx):
    """The positive case: an equal-length incoming save whose content hash
    chain DIFFERS from what's saved (e.g. an edited/regenerated turn, same
    total admission length) is genuinely fresher -- a plain `>=`
    length compare would discard it purely because length tied. This occurs
    on a long-running thread at a model swap, with equal lengths.
    Distinct from a smaller, compression-labelled incoming save
    -- this is the equal-length,
    role-agnostic case."""
    # save_to_disk=True (the unload-seam path, matching a model-
    # swap) hardcodes _min_growth=1 regardless of
    # LAGREDUCER_MIN_GROWTH_TURNS -- give the incoming one extra turn purely
    # to clear that SEPARATE, unrelated throttle gate; the property under
    # test is admission_ctx_len equality (the belt's own "length"), not
    # turn-count equality.
    old_chain = _prefix_hash_chain(_msgs(10))
    _write_clean_bin(kv_dir, _QWEN, _PORT, "t", 0, old_chain, prompt_len=50000)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    # same admission length, one more turn (clears the zero-growth throttle),
    # DIFFERENT content -> a genuinely different chain.
    new_messages = [{"role": ("user" if i % 2 == 0 else "assistant"),
                      "content": f"EDITED-turn-{i}-content"} for i in range(11)]
    new_chain = _prefix_hash_chain(new_messages)
    assert new_chain != old_chain  # sanity: this scenario really is a content diff
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                     client_meta={"messages": new_messages}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["hash_chain"] == new_chain, (
        "equal-length incoming with DIFFERENT content must overwrite the "
        f"stale bin -- got {after['hash_chain']}"
    )
    assert _n_save_posts(posts) == 1


# ================================================================================
# 4. FM-7 degenerate — 0-turn incoming never forces a save
# ================================================================================
@pytest.mark.asyncio
async def test_degenerate_empty_messages_no_save(mgr, kv_dir, make_httpx):
    """Empty admission messages -> no clean bin is written and no probe/save fires."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": []}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot)

    th = TurbohaulManager._thread_hash("t")
    assert not (kv_dir / kv_meta_fn(_QWEN, 0, th, _PORT)).exists()
    assert posts == []


@pytest.mark.asyncio
async def test_degenerate_zero_turn_chain_no_save(mgr, kv_dir, make_httpx, monkeypatch):
    """FM-7 guard: a truthy messages value that yields a 0-length chain must NOT force
    a save (the guard, not just the empty-list check, blocks it)."""
    monkeypatch.setattr(manager_mod, "_prefix_hash_chain", lambda *a, **k: [])
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(5)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot)

    th = TurbohaulManager._thread_hash("t")
    assert not (kv_dir / kv_meta_fn(_QWEN, 0, th, _PORT)).exists()
    assert posts == []
