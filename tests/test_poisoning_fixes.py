"""Two defects in the displacement-seam save path, shown by an executable
repro against the displacement-seam flagship test
(the displacement-seam clean-save test), verified against real
code before this fix.

First defect, tip_role staleness: the tip_role machinery (the write sites in
_probe_and_save_clean_kv, all gated on `_norm_tip_role`) lives entirely BELOW
the min-ctx-bar return. The displacement seam's recursive save-of-main call is hoisted
ABOVE that return. A short (<40000-char) curator/sub-agent visitor that
triggers the displacement seam gets its main-identity nested save stamped correctly
(tip_role="main", true at that instant), but then the OUTER call -- the
visitor's own turn -- returns at the min-ctx-bar before ever reaching its
own tip_role stamp. The anchor is left claiming tip_role="main" even after
the visitor's own turn overwrites the engine's real VRAM tip. A later
main-claiming caller's _vram_ours check (_maybe_force_clean_restore) then
reads a coherent-looking but FALSE "tip_role=main" and would natively reuse
VRAM that actually holds the visitor's state. The existing flagship test
(test_discriminator_curator_displacement_saves_outgoing_main) uses exactly
this short-curator shape and passes -- it asserts the save fired, never that
the anchor's bookkeeping is correct afterward, which is how the defect
went unnoticed inside code the flagship test called
"proven".

Second defect (prompt_len half): the `prompt_len` anchor field is overwritten by
EVERY stamp, main's own or a disposable visitor's own -- unlike
last_main_thread_id/last_main_client_meta, it is NOT carried forward. The displacement seam
originally read `_old_disp.get("prompt_len")` when re-saving main's own
identity; once a SECOND disposable visitor's own full stamp has landed since
main's last real turn, that value is the FIRST visitor's own prompt_len, not
main's -- silently mispairing main's real identity with the wrong length for
the recursive save's min-ctx-bar check.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import json
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
from turbohaul.manager import TurbohaulManager, _norm_tip_role
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_PORT = 59600

pytestmark = pytest.mark.asyncio


# --- fixtures (mirror the displacement-seam clean-save test's fixtures) --

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
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, idle_hot_load_seconds=60),
        pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    m.runtime.kv.covered_scaffold_strip = False
    return m


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    import turbohaul.subprocess_mgr as subprocess_mgr
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
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


def _n_probe_posts(posts, marker):
    n = 0
    for (url, j) in posts:
        if "action=save" in url or not j:
            continue
        msgs = j.get("messages") or []
        if any(marker in str(m.get("content", "")) for m in msgs):
            n += 1
    return n


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


def _fake_handle(model_tag=_QWEN, port=_PORT):
    proc = MagicMock()
    proc.pid = 77_777
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


# ================================================================================
# tip_role staleness after a SHORT (<40000-char) displacing visitor
# ================================================================================
class TestTipRoleStalenessFix:
    async def test_discriminator_short_curator_visitor_does_not_leave_stale_main_tip_role(
        self, mgr, kv_dir, make_httpx
    ):
        """MUST FAIL before the fix: exactly the flagship
        discriminator shape (test_discriminator_curator_displacement_saves_
        outgoing_main) -- a SHORT curator (admission_ctx_len defaults to 0,
        well under the 40000-char bar) displaces a long main thread. The displacement seam
        correctly fires and saves main's own KV (proven by the existing
        flagship test), but the curator's OWN call then short-circuits at
        the min-ctx-bar before reaching its own tip_role stamp. Without the
        fix, the anchor is left claiming tip_role="main" even though the
        engine's real occupant, after this call returns, is the curator --
        a later main-claiming caller's _vram_ours check would wrongly trust
        native VRAM reuse."""
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_slot = Slot.new(
            _QWEN, thread_id="main-thread-tiprole", admission_ctx_len=45000,
            client_meta={"is_main": True, "session_id": "s-main-tiprole",
                         "messages": _msgs(30, marker="MAINTIPROLE")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        curator_slot = Slot.new(
            _QWEN, thread_id="curator-thread-tiprole",
            client_meta={"is_curator": True, "session_id": "s-curator-tiprole",
                         "messages": _msgs(3, marker="CURATORTIPROLE")},
        )
        await mgr._probe_and_save_clean_kv(handle, curator_slot)

        # Sanity: the displacement seam did fire (flagship-proven behavior, unaffected by this fix).
        assert _n_probe_posts(posts, "MAINTIPROLE") >= 1

        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor is not None
        assert _norm_tip_role(anchor.get("tip_role")) != "main", (
            "anchor tip_role still reads 'main' after a short curator visitor's "
            "own turn -- stale claim from the displacement seam's nested main-save was never "
            "corrected, a later main caller's _vram_ours check would wrongly "
            "trust native VRAM reuse that actually holds the curator's state"
        )

    async def test_long_curator_visitor_still_gets_its_own_correct_tip_role(
        self, mgr, kv_dir, make_httpx
    ):
        """Good-shape-survives pairing: a curator LONG enough to clear the
        min-ctx-bar reaches its own full stamp regardless (unaffected by
        this fix either way) -- must still end up with its own tip_role,
        not "main"."""
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_slot = Slot.new(
            _QWEN, thread_id="main-thread-tiprole2", admission_ctx_len=45000,
            client_meta={"is_main": True, "session_id": "s-main-tiprole2",
                         "messages": _msgs(30, marker="MAINTIPROLE2")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        long_curator_slot = Slot.new(
            _QWEN, thread_id="curator-thread-tiprole2", admission_ctx_len=45000,
            client_meta={"is_curator": True, "session_id": "s-curator-tiprole2",
                         "messages": _msgs(30, marker="CURATORTIPROLE2")},
        )
        await mgr._probe_and_save_clean_kv(handle, long_curator_slot)

        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor is not None
        assert _norm_tip_role(anchor.get("tip_role")) == "curator"

    async def test_main_only_no_displacement_tip_role_still_main(
        self, mgr, kv_dir, make_httpx
    ):
        """Good-shape-survives: an ordinary main-only turn (no displacement seam in
        play at all) must still stamp tip_role="main" exactly as before
        shipped -- this fix must not touch the unrelated main-only path."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_slot = Slot.new(
            _QWEN, thread_id="main-thread-tiprole3", admission_ctx_len=45000,
            client_meta={"is_main": True, "session_id": "s-main-tiprole3",
                         "messages": _msgs(30, marker="MAINTIPROLE3")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor is not None
        assert _norm_tip_role(anchor.get("tip_role")) == "main"


# ================================================================================
# Displacement-seam prompt_len pairing -- last_main_prompt_len carried forward
# ================================================================================
class TestDisplacementSeamPromptLenPairingFix:
    async def test_discriminator_second_visitor_displacement_seam_uses_mains_own_prompt_len_not_first_visitors(
        self, mgr, kv_dir, make_httpx
    ):
        """MUST FAIL before the fix: main's real context is LONG (45000
        chars). A first SHORT curator visitor displaces it (the displacement seam fires,
        correctly saves main using main's own length). Before this fix, if
        that first visitor's own turn ever reaches its own full disposable
        stamp (it does here -- long enough itself), the anchor's plain
        "prompt_len" field is overwritten with the FIRST VISITOR's own
        (short) length. A second visitor then displaces again -- the displacement seam's
        recursive save for main's identity must still use MAIN's real
        45000-char length (via last_main_prompt_len), not the first
        visitor's short one, or main's own min-ctx-bar check could see a
        length too short to trust (silent save skip) or simply wrong
        bookkeeping."""
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_slot = Slot.new(
            _QWEN, thread_id="main-thread-plen", admission_ctx_len=45000,
            client_meta={"is_main": True, "session_id": "s-main-plen",
                         "messages": _msgs(30, marker="MAINPLEN")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        # First visitor: itself LONG enough to reach its own full disposable
        # stamp (so plain "prompt_len" gets overwritten with ITS OWN length,
        # much shorter than main's real 45000).
        first_visitor = Slot.new(
            _QWEN, thread_id="curator-thread-plen-1", admission_ctx_len=41000,
            client_meta={"is_curator": True, "session_id": "s-curator-plen-1",
                         "messages": _msgs(29, marker="CURATORPLEN1")},
        )
        await mgr._probe_and_save_clean_kv(handle, first_visitor)

        anchor_after_first = mgr._kv_vram_anchor.get(_PORT)
        assert anchor_after_first is not None
        assert anchor_after_first.get("last_main_prompt_len") == 45000, (
            "last_main_prompt_len must still read main's own true length "
            "after the first visitor's own full stamp"
        )
        # Sanity on the mechanism this test exists to distinguish: the
        # plain (non-carried) prompt_len field IS now the first visitor's
        # own, much shorter, length -- proving last_main_prompt_len and
        # prompt_len have genuinely diverged at this point.
        assert anchor_after_first.get("prompt_len") == 41000

        # Second visitor displaces again -- spy on the recursive save call
        # the displacement seam makes for the OUTGOING (main) identity to see exactly what
        # admission_ctx_len it was given. (A real re-probe would legitimately
        # no-op here via the pre-existing never-overwrite-with-smaller belt,
        # since main's own content hasn't grown since the first displacement
        # already saved it at 45000 -- that dedup is correct and unrelated to
        # this fix, so asserting on the recursive call's OWN argument is the
        # direct, unconfounded proof of the pairing fix.)
        calls = []
        _real_flush = mgr._flush_clean_kv_at_unload

        async def _spy_flush(handle, model_tag, thread_id, admission_ctx_len, client_meta, **kw):
            calls.append(admission_ctx_len)
            return await _real_flush(handle, model_tag, thread_id, admission_ctx_len, client_meta, **kw)

        mgr._flush_clean_kv_at_unload = _spy_flush

        second_visitor = Slot.new(
            _QWEN, thread_id="curator-thread-plen-2",
            client_meta={"is_curator": True, "session_id": "s-curator-plen-2",
                         "messages": _msgs(3, marker="CURATORPLEN2")},
        )
        await mgr._probe_and_save_clean_kv(handle, second_visitor)

        assert calls == [45000], (
            f"The displacement seam's recursive save for main's own identity was invoked with "
            f"admission_ctx_len={calls} -- expected [45000] (main's real "
            f"length, from last_main_prompt_len). A value of [41000] would "
            f"mean it read the first visitor's own overwritten prompt_len "
            f"instead (the pre-fix bug)."
        )


# ================================================================================
# Confirm last_main_prompt_len is None (never a stale carry) when no main
# turn has ever landed on this port.
# ================================================================================
class TestLastMainPromptLenAbsentByDefault:
    async def test_disposable_only_no_prior_main_last_main_prompt_len_is_none(
        self, mgr, kv_dir, make_httpx
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        curator_slot = Slot.new(
            _QWEN, thread_id="curator-thread-nomain", admission_ctx_len=45000,
            client_meta={"is_curator": True, "session_id": "s-curator-nomain",
                         "messages": _msgs(30, marker="CURATORNOMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle, curator_slot)

        anchor = mgr._kv_vram_anchor.get(_PORT)
        assert anchor is not None
        assert anchor.get("last_main_prompt_len") is None
        assert anchor.get("last_main_thread_id") is None
        assert anchor.get("last_main_client_meta") is None


# ================================================================================
# Subject-pairing bug:
# thread_id/prompt_len/client_meta all correctly travel as the OUTGOING
# (main) subject via last_main_*, but the displacement seam's recursive save read
# _old_disp.get("model_tag") for the FOURTH argument -- the CURRENT
# OCCUPANT's tag, not main's. kv_policy.kv_save_fn keys the bin filename on
# model_tag FIRST (f"{model_tag}.p{port}.{thread_hash}.slot{sid}.bin"), so a
# cross-model displacement -- the exact case this fix targets --
# saved main's transcript under a filename main's own restore would never
# look under: the save fires, logs FIRING, and is permanently unreachable.
# ================================================================================
class TestLastMainModelTagSubjectPairingFix:
    async def test_discriminator_displacement_seam_uses_mains_own_model_tag_not_first_visitors(
        self, mgr, kv_dir, make_httpx
    ):
        """MUST FAIL before the fix: main's real content is served under
        _QWEN. A first visitor -- itself long enough to reach its own full
        disposable stamp -- overwrites the anchor's plain (non-carried)
        "model_tag" field with ITS OWN model_tag (deliberately different
        from main's, isolating exactly which field the displacement seam's SECOND firing
        reads). A second visitor then displaces again: the displacement seam's recursive
        save for main's own identity must be invoked with main's REAL model
        tag (_QWEN, from last_main_model_tag), not the first visitor's
        (which the pre-fix code would have read from the anchor's plain,
        always-overwritten "model_tag" field)."""
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_1 = Slot.new(
            _QWEN, thread_id="main-thread-modeltag", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="MAINMODELTAG")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_1)
        assert mgr._kv_vram_anchor[_PORT]["last_main_model_tag"] == _QWEN

        # First visitor: long enough to reach its own full disposable stamp,
        # constructed with a DIFFERENT model_tag purely to isolate which
        # anchor field the displacement seam later reads from -- not meant to model a
        # realistic same-engine request (a live engine only ever serves one
        # model), just to prove the read source.
        _OTHER_MODEL = "curator-model-should-never-be-read-by-the-displacement-seam"
        first_visitor = Slot.new(
            _OTHER_MODEL, thread_id="curator-thread-modeltag-1", admission_ctx_len=41000,
            client_meta={"is_curator": True, "session_id": "s-curator-modeltag-1",
                         "messages": _msgs(29, marker="CURATORMODELTAG1")},
        )
        await mgr._probe_and_save_clean_kv(handle, first_visitor)
        anchor_after_first = mgr._kv_vram_anchor[_PORT]
        assert anchor_after_first["model_tag"] == _OTHER_MODEL, (
            "sanity: the first visitor's own stamp must have overwritten the "
            "anchor's plain model_tag field, or this test proves nothing"
        )
        assert anchor_after_first["last_main_model_tag"] == _QWEN, (
            "last_main_model_tag must still read main's own real tag after "
            "the first visitor's own full stamp"
        )

        calls = []
        _real_flush = mgr._flush_clean_kv_at_unload

        async def _spy_flush(handle, model_tag, thread_id, admission_ctx_len, client_meta, **kw):
            calls.append(model_tag)
            return await _real_flush(handle, model_tag, thread_id, admission_ctx_len, client_meta, **kw)

        mgr._flush_clean_kv_at_unload = _spy_flush

        second_visitor = Slot.new(
            _QWEN, thread_id="curator-thread-modeltag-2",
            client_meta={"is_curator": True, "session_id": "s-curator-modeltag-2",
                         "messages": _msgs(3, marker="CURATORMODELTAG2")},
        )
        await mgr._probe_and_save_clean_kv(handle, second_visitor)

        assert calls == [_QWEN], (
            f"The displacement seam's recursive save for main's own identity was invoked with "
            f"model_tag={calls} -- expected [{_QWEN!r}] (main's real tag, from "
            f"last_main_model_tag). A value of [{_OTHER_MODEL!r}] would mean it "
            f"read the first visitor's own overwritten model_tag instead (the "
            f"pre-fix subject-pairing bug) -- the save would land under the "
            f"WRONG filename and be permanently unreachable by main's own "
            f"future restore."
        )


# ================================================================================
# Stale-model guard: a last_main_* stamp can outlive the engine that
# produced it (_kv_vram_anchor is never popped on teardown). If a respawn on
# the same port now serves a DIFFERENT model than last_main_model_tag
# records, the displacement seam must NOT fire -- a silent skip here would be exactly the
# failure shape this fix exists to eliminate, so the decline must
# be named and logged distinguishably, not just silently absent.
# ================================================================================
class TestStaleModelGuard:
    async def test_discriminator_stale_last_main_model_tag_declines_not_fires(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        """MUST FAIL before the fix: main serves and is properly anchored
        under _QWEN. The engine is then respawned on the SAME port under a
        DIFFERENT model (simulated: a new SidecarHandle, same port, new
        model_tag) -- the anchor is never popped on teardown, so
        last_main_model_tag is now stale relative to the live handle. A
        curator visitor arrives on the respawned (new-model) handle: the displacement seam
        must decline (no probe posted for main's stashed content) rather
        than firing a save that would silently land under the WRONG
        (new, live) model's filename -- worse than not saving at all,
        since it would look like a successful save in the logs."""
        import logging

        handle_a = _fake_handle(model_tag=_QWEN, port=_PORT)
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_1 = Slot.new(
            _QWEN, thread_id="main-thread-matching-model", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="MAINSTALEGUARDCONTENT")},
        )
        await mgr._probe_and_save_clean_kv(handle_a, main_1)
        assert mgr._kv_vram_anchor[_PORT]["last_main_model_tag"] == _QWEN

        _RESPAWNED_MODEL = "respawned-different-model"
        handle_b = _fake_handle(model_tag=_RESPAWNED_MODEL, port=_PORT)

        curator_slot = Slot.new(
            _QWEN, thread_id="curator-thread-matching-model",
            client_meta={"is_curator": True, "session_id": "s-curator-matching-model",
                         "messages": _msgs(3, marker="CURATORSTALEGUARDCONTENT")},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle_b, curator_slot)

        assert _n_probe_posts(posts, "MAINSTALEGUARDCONTENT") == 0, (
            "The displacement seam fired a save for main's stashed content using a STALE "
            "last_main_model_tag against a respawned, different-model live "
            "handle -- this save would land under the WRONG filename, "
            "silently unreachable, and the FIRING log would falsely claim "
            "success"
        )
        skip_lines = [r.message for r in caplog.records if "last_main STALE" in r.message]
        assert skip_lines, (
            "no named 'last_main STALE' decline was logged -- a silent skip "
            "here is exactly the failure shape this fix exists to eliminate"
        )
        assert _QWEN in skip_lines[-1] and _RESPAWNED_MODEL in skip_lines[-1]

    async def test_good_shape_survives_matching_model_still_fires(
        self, mgr, kv_dir, make_httpx
    ):
        """Pairing assertion: when the live handle's model DOES match
        last_main_model_tag (the common, non-stale case), the displacement seam must still
        fire exactly as before -- this guard must not be overbroad."""
        handle = _fake_handle(model_tag=_QWEN, port=_PORT)
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        main_1 = Slot.new(
            _QWEN, thread_id="main-thread-matching-model-good", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="MAINMATCHINGMODELGOODCONTENT")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_1)

        curator_slot = Slot.new(
            _QWEN, thread_id="curator-thread-matching-model-good",
            client_meta={"is_curator": True, "session_id": "s-curator-matching-model-good",
                         "messages": _msgs(3, marker="CURATORMATCHINGMODELGOODCONTENT")},
        )
        await mgr._probe_and_save_clean_kv(handle, curator_slot)

        assert _n_probe_posts(posts, "MAINMATCHINGMODELGOODCONTENT") >= 1, (
            "The displacement seam did not fire even though last_main_model_tag matched the "
            "live handle's model -- the stale-model guard must not be overbroad"
        )
