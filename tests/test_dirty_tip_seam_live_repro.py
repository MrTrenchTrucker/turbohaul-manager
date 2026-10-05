"""A DETERMINISTIC, REPEATABLE reproduction of the dirty-tip
erase+reprefill seam (manager.py's `_probe_and_save_clean_kv`, `if _dirty:`
branch) -- the branch the restore-instead-of-reprefill fix targets, and which
never fired in smoke runs, which blocked verifying that fix.

DOES NOT FAKE THE PRECONDITION. The dirty-tip flag (`_kv_dirty_tail[port]`)
is set by the REAL admission-time code in `_probe_and_save_clean_kv` (the
`if not save_to_disk: if _tip_role not in (None, "main"): ...` branch,
manager.py) reacting to a REAL curator submission -- never written to
directly by this harness. The sequence that reaches it is driven through
`TurbohaulManager.submit()` + the REAL `worker_loop()` FSM (the SAME
dependency-injection pattern `tests/test_worker_loop.py`,
`tests/test_erase_order.py`, and the idle-park tip-freshness test
already use as this codebase's established way to drive FSM-level
behavior deterministically) -- only the outermost engine I/O (subprocess
spawn/health/sigterm/vram, and the sidecar's own `/slots` + `action=erase`
HTTP surface) is faked. Routing, grace-window matching (`pop_matched_thread`),
tip-dirtying, idle-park, and the unload-seam flush all run as REAL,
unmodified manager.py code.

★ THE SEQUENCING INSIGHT THIS REPRODUCTION NEEDED, found by reading (not guessing):
main's per-turn probe CLEARS the dirty-tip flag the moment main reprocesses
the tip (manager.py, "KV dirty-tip CLEARED ... main serve reprocessed
the tip") -- so a main -> curator -> main sequence (e.g.
the idle-park tip-freshness test's
`test_main_resumes_after_curator_interlude_...`) never reaches the erase branch: main's
OWN resumption clears the flag before any unload-seam flush ever sees it.
The flag survives to teardown ONLY when the disposable visit is the LAST
thing served on the port before the next unload-seam flush of MAIN's
identity -- i.e. main -> curator, with NOTHING main-class in between curator
and the idle-park/teardown that follows. That is also exactly the shape the
originally observed failure shows ("main continues on the same engine" after the contaminating
visit, not "main resumes and self-heals").

Also load-bearing: `_grace_tip` (the identity that actually gets parked and
flushed at teardown) only advances to a matched follow-up when that
follow-up is main-class (manager.py) -- so even though curator is the
LAST turn actually served, the identity flushed at idle-park is still MAIN's
own (the anchor), which is exactly why the flush discovers a dirty tip it
did not itself leave: main's identity is clean, the ENGINE's resident state
is not.

WHAT THIS PROVES, mechanically, on every run:
  1. `KV dirty-tip SET` fires at curator's own real admission (not injected).
  2. The dirty flag SURVIVES to the unload-seam flush (no intervening
     main-class turn clears it).
  3. `unload-seam DIRTY TIP` fires at that flush, with the REAL erase POST
     (`/slots/{id}?action=erase`) actually sent to the (faked) sidecar.
  4. `dirty-tip ERASED ... FULL-reprefill` fires, and the full render+
     prefill probe that follows is observable as a real POST too.

LIMITATION, stated plainly: this does not run
against a live engine or share its GPU/process, so it
cannot by itself prove the exact production timing/interleaving with real
client traffic -- it proves the BRANCH, mechanically, through the real code
path that reaches it, isolated (not merely surviving) live traffic by
construction, since nothing here touches a live system at all. If a live-system
reproduction is later required, this harness's `turns` sequence is the
concrete recipe for what payload shape/order to script against the real API
next time. Earlier attempts against a live engine showed that a model-swap
takes the wrong branch (KV dirty-tip DROPPED, not ERASED) because the engine
dies before the flush can discover the flag; the shared-thread main-curator-
main script self-cleared before ever reaching the erase branch (see the sequencing
insight above) even where it was not being preempted by live traffic.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this file> -v
"""
from __future__ import annotations

import asyncio
import logging
import os
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
_PORT = 59950


# --- harness (mirrors test_worker_loop.py's _boot_runtime/_seed_manifest and
# the idle-park tip-freshness test's kv_dir/make_httpx/_ProbeSaveClient
# -- same established DI pattern, not a new testing philosophy) -------------

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
    proc.pid = 88_886
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    p = tmp_path / "kvcache_persist"
    p.mkdir()
    import turbohaul.subprocess_mgr as subprocess_mgr
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(p))
    return d


class _SaveResp:
    # status_code is read directly (not via raise_for_status()) by
    # load_verify_log.verify_kv_restored -- the restore fix's own independent
    # verification call, a DIFFERENT convention than manager.py's own
    # raise_for_status()-style callers. Both must be satisfied by one fake.
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    """GET /slots (non-action=save) -> ONE populated slot -- this is what
    makes the erase POST actually fire (_classify_dirty_tip_probe needs
    populated_count == 1). Every POST is recorded, unfiltered, so the test
    can assert the REAL erase (`action=erase`) and REAL reprefill probe
    (`/v1/chat/completions`) calls the manager itself decided to make."""

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
        if "/v1/chat/completions" in url:
            # Plain-probe regime: the real sidecar's /v1/chat/completions
            # reply carries the tokenized prompt's usage; the manager's
            # strict clean stamp keys off usage.prompt_tokens,
            # so the fake reports the slot's engine-reported count.
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


class _ProbeSaveClientWithRealSaves(_ProbeSaveClient):
    """Same GET/erase behavior as _ProbeSaveClient, PLUS: an `action=save`
    POST actually materializes the bin on disk under SLOT_SAVE_DIR (mirrors
    the idle-park tip-freshness test's own established convention).
    This is what lets the displacement rescue-save -- which ALREADY fires in the
    plain harness above, but fails there with a FileNotFoundError
    because no file backs the tmp-rename -- actually SUCCEED and leave a
    real, restorable clean bin behind. Nothing else about the sequence
    changes; this is the ONLY difference from the other arms' fixture."""

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
            # Plain-probe regime: the real sidecar's /v1/chat/completions
            # reply carries the tokenized prompt's usage; the manager's
            # strict clean stamp keys off usage.prompt_tokens,
            # so the fake reports the slot's engine-reported count.
            _n = (self._slots_payload[0].get("n_prompt_tokens") or 1) if self._slots_payload else 1
            return _SaveResp({"status": "ok",
                               "usage": {"prompt_tokens": _n,
                                         "completion_tokens": 0,
                                         "total_tokens": _n}})
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx_with_real_saves(monkeypatch):
    import turbohaul.manager as manager_mod
    import turbohaul.load_verify_log as load_verify_log_mod

    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _ProbeSaveClientWithRealSaves(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        # load_verify_log.py imports httpx at ITS OWN module level (`import
        # httpx`, load_verify_log.py) -- a completely separate binding
        # from manager_mod.httpx. verify_kv_restored's own /slots GET
        # (the independent check the restore fix calls to confirm a restore
        # before trusting it) goes through THIS binding, not manager's --
        # without this patch the restore fires but comes back
        # "UNVERIFIED (actual=None)": the verification call hits a
        # real, non-existent 127.0.0.1:<port>/slots instead of this fixture.
        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        monkeypatch.setattr(load_verify_log_mod, "httpx", _FakeHttpx)
        return posts

    return _make


_MAIN_CM = {"is_main": True}
_CURATOR_CM = {"is_curator": True, "save_kv": False}


async def _run_main_then_curator_last_then_idle_park(
    tmp_path, kv_dir, make_httpx, caplog, *,
    grace_seconds=2, idle_hot_load_seconds=60,
):
    """main (anchor, establishes the resident+thread) -> curator (rides the
    SAME thread_id, is the LAST turn served -- nothing main-class follows it
    before teardown) -> let the grace window expire NATURALLY -> shutdown()
    (the real unload-seam flush of MAIN's identity). Returns (posts, records)
    for the caller to assert on."""
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
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        anchor = await mgr.submit(
            model_tag=_MODEL, prompt="main-anchor", thread_id="t-seam-repro",
            client_meta={**_MAIN_CM, "messages": _msgs(2, "MAIN-ANCHOR")},
            admission_ctx_len=60000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(anchor.completion_future, timeout=5.0)
        await asyncio.sleep(0.05)  # let worker enter the GRACE loop

        # THE CONTAMINATING VISIT -- rides the SAME thread_id, matches via
        # the REAL pop_matched_thread grace-window mechanism (no role
        # awareness there, exactly as production), and is the LAST submit
        # on this thread before teardown: nothing main-class follows it to
        # clear the dirty-tip flag the way it would in production if main
        # simply resumed first.
        curator = await mgr.submit(
            model_tag=_MODEL, prompt="curator-interlude", thread_id="t-seam-repro",
            client_meta={**_CURATOR_CM, "messages": _msgs(10, "CURATOR")},
            admission_ctx_len=90000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(curator.completion_future, timeout=5.0)

        # Let the grace window expire NATURALLY so _process_slot's own code
        # (not a cancellation-unwind finally) reaches idle-hot-enter with
        # curator's dirty stamp still on the port.
        await asyncio.sleep(grace_seconds + 0.5)

        # shutdown() finds the parked idle holder and performs the REAL
        # unload-seam flush: _teardown_idle_holder -> _flush_clean_kv_at_unload
        # -> _probe_and_save_clean_kv(save_to_disk=True) -> the erase branch.
        await mgr.shutdown()
    finally:
        if not mgr._stop_event.is_set():
            await mgr.shutdown()

    return posts, list(caplog.records)


def _erase_posts(posts):
    return [u for (u, _j) in posts if "action=erase" in u]


def _reprefill_posts(posts):
    return [u for (u, _j) in posts if "/v1/chat/completions" in u]


class TestSeamFiresOnDemand:
    async def test_dirty_tip_set_at_real_curator_admission(
            self, tmp_path, kv_dir, make_httpx, caplog):
        """Step 1 of the proof: the dirty flag is a SIDE EFFECT of a real
        curator submission reaching production code, not a value this
        harness ever assigns."""
        _posts, records = await _run_main_then_curator_last_then_idle_park(
            tmp_path, kv_dir, make_httpx, caplog)
        assert any("KV dirty-tip SET" in r.message for r in records), (
            "the dirty-tip flag was never set by the real admission-time "
            "code -- if this fails, the harness's curator submission is not "
            "reaching _probe_and_save_clean_kv's tip-dirtying branch at all"
        )

    async def test_dirty_tip_survives_to_the_unload_seam(
            self, tmp_path, kv_dir, make_httpx, caplog):
        """Step 2: with curator as the LAST turn (no main-class resumption
        to clear it), the flag must still be live when the unload-seam flush
        runs -- the CLEARED branch (main reprocessing) must NOT have fired."""
        _posts, records = await _run_main_then_curator_last_then_idle_park(
            tmp_path, kv_dir, make_httpx, caplog)
        assert not any("KV dirty-tip CLEARED" in r.message for r in records), (
            "the dirty flag was cleared before teardown -- something "
            "main-class reprocessed the tip in this sequence, which would "
            "defeat the whole point of curator being the LAST turn served"
        )

    async def test_erase_and_reprefill_branch_fires(
            self, tmp_path, kv_dir, make_httpx, caplog):
        """THE LOAD-BEARING ASSERTION for this file: the exact branch that
        never executed in smoke runs -- WARNING unload-seam DIRTY TIP, then
        INFO dirty-tip ERASED ... FULL-reprefill -- fires on demand, driven
        entirely through real code, with a REAL erase POST and a REAL
        reprefill POST observed on the (faked) sidecar transport."""
        posts, records = await _run_main_then_curator_last_then_idle_park(
            tmp_path, kv_dir, make_httpx, caplog)

        assert any("unload-seam DIRTY TIP" in r.message for r in records), (
            "the dirty-tip WARNING line never fired -- the unload-seam flush did not "
            "discover the dirty tip left by curator's real admission"
        )
        assert any(
            "dirty-tip ERASED" in r.message and "FULL-reprefill" in r.message
            for r in records
        ), "the erased log line never fired"
        assert len(_erase_posts(posts)) == 1, (
            f"expected exactly one real action=erase POST to the sidecar, "
            f"got {_erase_posts(posts)}"
        )
        assert len(_reprefill_posts(posts)) >= 1, (
            "the full render+prefill probe that follows a successful erase "
            "never POSTed to /v1/chat/completions"
        )

    async def test_repeatable_across_runs(
            self, tmp_path_factory, make_httpx, caplog, monkeypatch):
        """Not a one-off: the SAME harness, called 3 times with INDEPENDENT
        tmp_path/kv_dir per iteration (tmp_path_factory, not the shared
        tmp_path fixture -- reusing one tmp_path across iterations collides
        on the storage_root mkdir the second time round), fires the branch
        every time -- 'on demand and repeatably',
        not 'reproduced it once by luck'."""
        import turbohaul.subprocess_mgr as subprocess_mgr

        for i in range(3):
            iter_tmp = tmp_path_factory.mktemp(f"seam_repeat_{i}")
            kv_dir_i = iter_tmp / "kvcache"
            kv_dir_i.mkdir()
            persist_dir_i = iter_tmp / "kvcache_persist"
            persist_dir_i.mkdir()
            monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(kv_dir_i))
            monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(persist_dir_i))

            posts, records = await _run_main_then_curator_last_then_idle_park(
                iter_tmp, kv_dir_i, make_httpx, caplog)
            assert any(
                "dirty-tip ERASED" in r.message and "FULL-reprefill" in r.message
                for r in records
            ), f"iteration {i}: the erase success line did not fire"
            assert len(_erase_posts(posts)) == 1, (
                f"iteration {i}: expected exactly one erase POST, got {_erase_posts(posts)}"
            )
            caplog.clear()


async def _run_seeded_snapshot_then_curator_last_then_idle_park(
    tmp_path, make_httpx_with_real_saves, caplog, monkeypatch, *,
    grace_seconds=2, idle_hot_load_seconds=60,
):
    """Two REAL, separate phases on the SAME thread_id and SAME on-disk
    kvcache dirs (simulating a resident restart, not a single unbroken
    session):

    PHASE 1 -- seeds a genuine, smaller, on-disk clean bin the ONLY way
    this file will accept: a plain main-only session, served, let idle out
    NATURALLY, and flushed for real at shutdown (byte-for-byte the same
    shape as the idle-park tip-freshness test's own
    `_run_two_turn_thread_then_idle_park` control case). No curator
    involved in this phase at all.

    PHASE 2 -- a NEW manager instance (same kvcache dirs), main resumes the
    SAME thread with MORE turns than phase 1 saved (so there is real growth
    to restore-from-and-beyond, not just a length-for-length replay), then
    curator visits as the LAST turn, then the grace window expires and
    shutdown() runs the real unload-seam flush.

    WHY THE DISPLACEMENT-SAVE GATE IS PATCHED OFF FOR PHASE 2, AND WHY THAT IS NOT THE THING
    UNDER TEST: That gate (`_displacement_save_allowed`) is a SEPARATE,
    earlier-firing remediation for the exact same problem -- it already
    fires the instant curator is admitted (confirmed: with
    _ProbeSaveClientWithRealSaves alone, curator's own admission makes the
    gate succeed and immediately re-save main's CURRENT identity). But because
    the gate always captures main's identity at EXACTLY its state at that
    instant, whatever it saves is IDENTICAL to what the later unload-seam
    flush would try to save too -- so the flush's own zero-growth throttle
    (manager.py's lag-reducer, unconditionally upstream of the
    dirty-tip block) declines the redundant re-save and the erase branch is never reached
    at all. That is a REAL interaction, not a test artifact -- but it means
    isolating the erase branch's OWN restore-from-an-existing-snapshot behavior requires
    the gate to have declined, which happens for real and often in production
    (stale model on the port, no prior main identity, below the min-ctx
    bar -- all named, existing decline paths). This fixture pins ONE such
    decline deterministically rather than fighting to reproduce a specific
    one (e.g. a model respawn) by accident. It does NOT touch
    `_kv_dirty_tail`, the erase decision, or the restore fix's own restore/verify code
    -- only the SEPARATE, competing rescue-save gate.
    """
    import turbohaul.manager as manager_mod
    import turbohaul.subprocess_mgr as subprocess_mgr

    kv_dir = tmp_path / "kvcache"
    kv_dir.mkdir()
    persist_dir = tmp_path / "kvcache_persist"
    persist_dir.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(kv_dir))
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(persist_dir))

    # --- PHASE 1: real, ordinary main-only idle-park, produces a real
    # smaller bin on disk. Fresh manager instance; separate boot/runtime. ---
    boot1, runtime1 = _boot_runtime(
        tmp_path / "phase1", grace_seconds=0, idle_hot_load_seconds=60)
    _seed_manifest(boot1, _MODEL)
    make_httpx_with_real_saves([{"id": 0, "n_prompt_tokens": 100}])

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

    mgr1 = TurbohaulManager(
        boot1, runtime1,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )
    mgr1.runtime.kv.covered_scaffold_strip = False
    mgr1._worker_task = asyncio.create_task(mgr1.worker_loop())
    # Phase 2's grown conversation below must be a REAL prefix extension of
    # these exact 2 messages (same content, not just the same turn count) --
    # the restore path validates a genuine hash-chain prefix match, not just
    # "some bin exists". Sharing this literal list is what makes phase 2 a
    # true continuation instead of an unrelated same-length conversation.
    seed_messages = _msgs(2, "SEED-MAIN")
    seed = await mgr1.submit(
        model_tag=_MODEL, prompt="phase1-seed", thread_id="t-seam-restore",
        client_meta={**_MAIN_CM, "messages": seed_messages},
        admission_ctx_len=60000,
        wait_for_completion=True,
    )
    await asyncio.wait_for(seed.completion_future, timeout=5.0)
    await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)  # grace_seconds=0 -> expires almost immediately
    await mgr1.shutdown()

    seed_bins = _list_clean_bin_metas_local(kv_dir, _MODEL, _PORT)
    assert len(seed_bins) == 1, (
        f"phase 1 setup failed: expected exactly one seeded clean bin, got "
        f"{seed_bins} -- if this fails, the REST of this test proves "
        f"nothing, since there would be no real pre-existing snapshot"
    )
    assert seed_bins[0]["n_context_turns"] == 2

    # --- PHASE 2: a NEW manager (same on-disk dirs), main resumes the SAME
    # thread with MORE turns, then curator visits last, then real teardown. -
    monkeypatch.setattr(manager_mod, "_displacement_save_allowed",
                         lambda *a, **k: (False, "main", None))

    boot2, runtime2 = _boot_runtime(
        tmp_path / "phase2", grace_seconds, idle_hot_load_seconds)
    _seed_manifest(boot2, _MODEL)
    posts = make_httpx_with_real_saves([{"id": 0, "n_prompt_tokens": 100}])

    mgr2 = TurbohaulManager(
        boot2, runtime2,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
    )
    mgr2.runtime.kv.covered_scaffold_strip = False
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr2._worker_task = asyncio.create_task(mgr2.worker_loop())
    try:
        # Genuine prefix extension: the SAME 2 seed messages, plus 7 more --
        # not a fresh, unrelated 9-turn conversation. This is what lets the
        # restore path's own hash-chain prefix-validity check see the seeded
        # bin as a real, valid (if shorter) prefix of what's now being asked
        # for, instead of correctly refusing it as DIVERGED.
        grown_messages = seed_messages + _msgs(7, "GROWN-MAIN")
        anchor = await mgr2.submit(
            model_tag=_MODEL, prompt="phase2-main-grown", thread_id="t-seam-restore",
            client_meta={**_MAIN_CM, "messages": grown_messages},
            admission_ctx_len=90000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(anchor.completion_future, timeout=5.0)
        await asyncio.sleep(0.05)

        curator = await mgr2.submit(
            model_tag=_MODEL, prompt="phase2-curator", thread_id="t-seam-restore",
            client_meta={**_CURATOR_CM, "messages": _msgs(15, "CURATOR")},
            admission_ctx_len=95000,
            wait_for_completion=True,
        )
        await asyncio.wait_for(curator.completion_future, timeout=5.0)

        await asyncio.sleep(grace_seconds + 0.5)
        await mgr2.shutdown()
    finally:
        if not mgr2._stop_event.is_set():
            await mgr2.shutdown()

    return posts, list(caplog.records)


def _list_clean_bin_metas_local(kv_dir, model_tag, port):
    import json
    out = []
    for fn in os.listdir(kv_dir):
        if fn.startswith(f"{model_tag}.p{port}.") and fn.endswith(".json"):
            with open(os.path.join(kv_dir, fn)) as f:
                meta = json.load(f)
            if meta.get("clean_prefix"):
                out.append(meta)
    return out


class TestRestoreSuccessSkipsReprefill:
    """The restore-success arm. The classes
    above prove the SEAM is reached through real code; they never exercise
    the branch that actually delivers the restore fix's benefit, because the harness
    has no bins at all (resolved_from='restore-no-bins') -- `_restore_slot_kv`
    returns falsy and control falls through to the byte-identical-to-today
    fallback, which is exactly why unpatched main and a restore-patched tree
    both pass those arms identically.

    REQUIRES THE RESTORE FIX IN THE TREE. This is the one arm in this file
    that is NOT fix-agnostic: against unpatched main there is no restore-
    attempt code in manager.py for a snapshot to be used by at all,
    and this arm's own assertions would fail (no "RESTORED..." log line
    could ever fire). Run it against a tree with the restore fix applied -- the other
    classes in this file remain meaningful either way.

    THE SNAPSHOT IS PRODUCED BY THE REAL SAVE PATH, not hand-placed -- see
    `_run_seeded_snapshot_then_curator_last_then_idle_park`'s own docstring
    for the full account of why a second, real idle-park phase (not a
    hand-written bin file) was needed, and why the displacement-save gate's OWN competing
    rescue-save had to be isolated out (it fires first, catches main's
    identity at the same size the later flush would also try to save, and
    its own zero-growth throttle then blocks the erase branch from ever being reached --
    a real interaction, not assumed).
    """

    async def test_restore_success_fires_and_skips_full_reprefill(
            self, tmp_path, make_httpx_with_real_saves, caplog, monkeypatch):
        """THE DELIVERABLE ASSERTION. Not "a restore was attempted" (strictly
        weaker) -- the fix's whole claim is an AVOIDED cost, so the absence of
        the cold reprefill POST is what has to be proven, alongside the
        mitigation's own success log line actually firing."""
        posts, records = await _run_seeded_snapshot_then_curator_last_then_idle_park(
            tmp_path, make_httpx_with_real_saves, caplog, monkeypatch)

        assert any("unload-seam DIRTY TIP" in r.message for r in records), (
            "the erase branch was never reached at all -- the seeded-snapshot setup or "
            "the gate isolation broke the sequence before the seam"
        )
        assert any(
            "RESTORED existing verified clean snapshot" in r.message
            for r in records
        ), (
            "the restore mitigation's own success log line never fired even "
            "though a real, verified, on-disk snapshot existed"
        )
        assert _reprefill_posts(posts) == [], (
            f"THE DELIVERABLE: the cold reprefill fired anyway despite a "
            f"successful, verified restore -- the fix's claim is an avoided "
            f"cost, and the cost was not avoided here. reprefill posts: "
            f"{_reprefill_posts(posts)}"
        )
        assert len(_erase_posts(posts)) == 1, (
            "the erase itself must still have happened -- this arm proves "
            "restore-instead-of-reprefill, not skip-the-erase-too"
        )

    async def test_control_no_snapshot_still_falls_back_to_full_reprefill(
            self, tmp_path, kv_dir, make_httpx, caplog):
        """Control, using the ORIGINAL (single-phase, no seeded snapshot)
        sequence and the PLAIN fixture: without a real snapshot, the full
        reprefill must still fire. Proves the two arms actually discriminate
        (one skips, one doesn't) instead of both vacuously passing the same
        way regardless of what the fixture does."""
        posts, records = await _run_main_then_curator_last_then_idle_park(
            tmp_path, kv_dir, make_httpx, caplog)
        assert not any(
            "RESTORED existing verified clean snapshot" in r.message
            for r in records
        )
        assert len(_reprefill_posts(posts)) >= 1, (
            "control: with no snapshot available, the full reprefill must "
            "still fire -- the same assertion the other arms already "
            "proved, repeated here for direct contrast with the "
            "restore-success arm above"
        )
