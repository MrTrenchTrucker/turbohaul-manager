"""Log-surface instruments: idle-teardown gets its own INFO line, the
KV_SAVE_FIRING trigger= stops collapsing idle-expiry/model-swap/shutdown/
displacement/grace-expired/worker-uncaught-exception/loading-fail-health-
timeout into one generic "unload_seam" label, and tier= tells a RAM save
apart from an SSD persist. Zero behaviour change -- every assertion here is
about what gets logged, never about what gets saved/returned/decided.

Each test below proves a line ACTUALLY fires with the value it claims, and
proves the *other* branch produces a visibly different value on the same
line -- a log line nobody has seen fire under both conditions is not proof.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_kv_log_surface_and_tier_proof.py -v
"""
from __future__ import annotations

import logging
import time
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
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_PORT = 59900

pytestmark = pytest.mark.asyncio


# --- fixtures (same shape as the KV-save observability tests) ----------

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
def kv_dirs(tmp_path, monkeypatch):
    save_dir = tmp_path / "kvcache"
    save_dir.mkdir()
    persist_dir = tmp_path / "kvcache_persist"
    persist_dir.mkdir()
    import turbohaul.subprocess_mgr as subprocess_mgr
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(save_dir))
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(persist_dir))
    import turbohaul.manager as manager_mod
    monkeypatch.setattr(manager_mod, "SLOT_SAVE_DIR", str(save_dir), raising=False)
    monkeypatch.setattr(manager_mod, "SLOT_PERSIST_DIR", str(persist_dir), raising=False)
    return save_dir, persist_dir


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


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


def _fake_handle(model_tag=_QWEN, port=_PORT):
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _kv_lines(caplog, marker, **field_filters):
    """Pull KV_SAVE_FIRING/IDLE_TEARDOWN lines out of captured records,
    filtered by exact field=value substrings."""
    out = []
    for r in caplog.records:
        msg = r.message
        if not msg.startswith(marker):
            continue
        if all(f"{k}={v}" in msg for k, v in field_filters.items()):
            out.append(msg)
    return out


def _field(line, name):
    """Extract the value of `name=` out of a space-separated KV_* log line
    (all these lines are `MARKER key=value key=value ...`). Used to assert
    against what a line ACTUALLY claims, not against a value the test's own
    fixture happened to compute the same way the code under test does --
    that coincidence proves nothing."""
    for tok in line.split():
        if tok.startswith(f"{name}="):
            return tok[len(name) + 1:]
    return None


def _sync_idle_holder(mgr, *, alive, model_tag, thread_id, admission_ctx_len,
                       messages, expires_at):
    # A real SidecarHandle, not a bare MagicMock -- _probe_and_save_clean_kv
    # reads handle.parallel as an int (single-series gate math), which a
    # plain MagicMock auto-creates as a non-numeric Mock and breaks.
    held = _fake_handle(model_tag=model_tag, port=_PORT)
    held.proc.poll.return_value = None if alive else 0
    mgr._idle_handle = held
    mgr._idle_model_tag = model_tag
    mgr._idle_thread_id = thread_id
    mgr._idle_admission_ctx_len = admission_ctx_len
    mgr._idle_client_meta = {"messages": messages}
    mgr._idle_expires_at = expires_at
    return held


# ================================================================================
# IDLE_TEARDOWN — one line, fires for every reason, deadline_delta_s
# sign flips between a natural expiry and an early interruption.
# ================================================================================
class TestIdleTeardownLine:
    async def test_idle_expired_deadline_delta_is_positive_past_deadline(
        self, mgr, monkeypatch, caplog
    ):
        """Deadline already passed 5s ago when teardown runs -- the shape a
        natural idle_expired wakeup actually produces. Reproduces the REAL
        worker_loop calling convention, not just the function's own state:
        capture the deadline into a local, NULL self._idle_expires_at (the
        way worker_loop debounces a concurrent re-arm), THEN invoke the
        teardown with that local threaded in as expires_at -- entering
        through the caller's own sequencing, not just through state this
        test set directly on the manager. A version of this test that
        skipped the null step could not have caught the bug where
        deadline_delta_s read None on idle_expired
        because the field was already wiped by the time the function ran."""
        _sync_idle_holder(
            mgr, alive=False, model_tag=_QWEN, thread_id="t-idletd-expired",
            admission_ctx_len=0, messages=[],
            expires_at=time.monotonic() - 5.0,
        )
        expires = mgr._idle_expires_at  # worker_loop's own local, captured first
        mgr._set_idle_expires_at(None)  # worker_loop nulls it BEFORE dispatch

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        assert mgr._idle_expires_at is None, (
            "setup didn't reproduce the real precondition -- the field must "
            "already be null when the teardown coroutine runs, same as it "
            "is for the real idle_expired/idle_dead call sites"
        )
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_expired", expires_at=expires)
        lines = _kv_lines(caplog, "IDLE_TEARDOWN", reason="idle_expired",
                           model_tag=_QWEN)
        assert len(lines) == 1, caplog.records
        assert "deadline_delta_s=" in lines[0]
        delta_str = lines[0].split("deadline_delta_s=")[1].split()[0]
        assert delta_str != "None", (
            "deadline_delta_s is None on the one path (idle_expired) this "
            f"field exists to describe -- the caller's null-before-dispatch "
            f"sequencing beat the field read: {lines[0]}"
        )
        assert float(delta_str) > 0, (
            f"expected a positive (past-deadline) delta for a natural expiry, "
            f"got {delta_str}: {lines[0]}"
        )

    async def test_model_swap_deadline_delta_is_negative_time_remaining(
        self, mgr, monkeypatch, caplog
    ):
        """Deadline is still 120s away -- the shape an early model_swap
        interruption actually produces. Different reason AND different delta
        sign from the test above -- this is the "other branch" proof.
        Deliberately does NOT null self._idle_expires_at first and does NOT
        pass expires_at -- model_swap's real caller doesn't null it either,
        so the function's fallback (read the field directly) is the
        correct path to exercise here, not a gap."""
        _sync_idle_holder(
            mgr, alive=False, model_tag=_QWEN, thread_id="t-idletd-swap",
            admission_ctx_len=0, messages=[],
            expires_at=time.monotonic() + 120.0,
        )

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("model_swap")
        lines = _kv_lines(caplog, "IDLE_TEARDOWN", reason="model_swap",
                           model_tag=_QWEN)
        assert len(lines) == 1, caplog.records
        delta_str = lines[0].split("deadline_delta_s=")[1].split()[0]
        assert float(delta_str) < 0, (
            f"expected a negative (time-still-remaining) delta for an early "
            f"interruption, got {delta_str}: {lines[0]}"
        )

    async def test_idle_teardown_line_fires_even_on_dead_holder(
        self, mgr, monkeypatch, caplog
    ):
        """A line is required on ENTRY, not only on the happy alive path
        -- a dead holder (idle_dead) still gets one, even though it then
        declines the save via the pre-existing HANDLE_NOT_ALIVE path.
        idle_dead nulls self._idle_expires_at before dispatch too (same as
        idle_expired), so this also reproduces caller-order: capture, null,
        thread through."""
        _sync_idle_holder(
            mgr, alive=False, model_tag=_QWEN, thread_id="t-idletd-dead",
            admission_ctx_len=0, messages=[],
            expires_at=time.monotonic() - 1.0,
        )
        expires = mgr._idle_expires_at
        mgr._set_idle_expires_at(None)

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_dead", expires_at=expires)
        lines = _kv_lines(caplog, "IDLE_TEARDOWN", reason="idle_dead", alive="False")
        assert len(lines) == 1, caplog.records
        delta_str = lines[0].split("deadline_delta_s=")[1].split()[0]
        assert delta_str != "None", (
            f"idle_dead nulls the deadline before dispatch too -- must still "
            f"read a real number via the threaded expires_at: {lines[0]}"
        )

    async def test_idle_teardown_thread_hash_is_the_real_untruncated_join_key(
        self, mgr, kv_dirs, make_httpx, monkeypatch, caplog
    ):
        """IDLE_TEARDOWN's thread_hash is the ONLY field that ties it to the
        KV_SAVE_FIRING line the same teardown produces moments later.
        Blanking or truncating it differently than KV_SAVE_FIRING's own
        (untruncated) thread_hash would silently break that join for any
        thread_id past the cut -- so the test must assert the field's value (a
        test that asserted nothing about it would leave the suite
        green when it is blanked). Uses a thread_id longer than the old 60-char truncation to
        prove the join survives, and drives the ALIVE path through to a real
        save so both lines actually exist to compare."""
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        long_thread_id = "t-idletd-join-" + ("x" * 80)  # > the old [:60] cut
        assert len(long_thread_id) > 60

        _sync_idle_holder(
            mgr, alive=True, model_tag=_QWEN, thread_id=long_thread_id,
            admission_ctx_len=45000, messages=_msgs(30, marker="JOINKEY"),
            expires_at=time.monotonic() - 1.0,
        )

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_expired")

        teardown_lines = _kv_lines(caplog, "IDLE_TEARDOWN", reason="idle_expired")
        firing_lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                                  call_site="probe_and_save_clean_kv")
        assert len(teardown_lines) == 1, caplog.records
        assert len(firing_lines) == 1, caplog.records

        teardown_hash = _field(teardown_lines[0], "thread_hash")
        firing_hash = _field(firing_lines[0], "thread_hash")
        assert teardown_hash == long_thread_id, (
            f"IDLE_TEARDOWN must carry the real, full thread_id, not blank "
            f"or truncated: got {teardown_hash!r}: {teardown_lines[0]}"
        )
        assert teardown_hash == firing_hash, (
            f"the two lines describing the SAME teardown must agree on "
            f"thread_hash byte-for-byte, or a join between them silently "
            f"returns nothing: IDLE_TEARDOWN={teardown_hash!r} "
            f"KV_SAVE_FIRING={firing_hash!r}"
        )
        assert any("action=save" in url for (url, _j) in posts)


# ================================================================================
# trigger= threading through both save_to_disk=True teardown callers --
# the same authoritative call_site reports a different, real reason
# instead of always "unload_seam".
# ================================================================================
class TestTriggerThreading:
    async def test_idle_expired_vs_model_swap_produce_different_trigger(
        self, mgr, kv_dirs, make_httpx, monkeypatch, caplog
    ):
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)

        _sync_idle_holder(
            mgr, alive=True, model_tag=_QWEN, thread_id="t-trig-expired",
            admission_ctx_len=45000, messages=_msgs(30, marker="TRIGEXP"),
            expires_at=time.monotonic() - 1.0,
        )
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_expired")
        expired_lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                                   call_site="probe_and_save_clean_kv",
                                   tier="RAM")
        assert len(expired_lines) == 1, caplog.records
        assert "trigger=idle_expired" in expired_lines[0]
        assert any("action=save" in url for (url, _j) in posts), (
            "trigger=idle_expired logged but no real save POST was observed"
        )

        caplog.clear()
        posts.clear()
        _sync_idle_holder(
            mgr, alive=True, model_tag=_QWEN, thread_id="t-trig-swap",
            admission_ctx_len=45000, messages=_msgs(30, marker="TRIGSWAP"),
            expires_at=time.monotonic() + 900.0,
        )
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("model_swap")
        swap_lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                                call_site="probe_and_save_clean_kv", tier="RAM")
        assert len(swap_lines) == 1, caplog.records
        assert "trigger=model_swap" in swap_lines[0], (
            "same authoritative call_site, different reason -- must NOT "
            f"still read the old generic 'unload_seam': {swap_lines[0]}"
        )

    async def test_teardown_grace_expired_vs_worker_uncaught_exception(
        self, mgr, kv_dirs, make_httpx, monkeypatch, caplog
    ):
        """_teardown is the caller that is easy to miss -- the real origin
        of "failure-teardown". Two of its own reason strings, same
        call_site, must read back distinctly."""
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)

        # _teardown unpacks _vram_verify's real (bool, int|None)
        # contract (the return is not ignored). This
        # test is about KV-save trigger threading, not VRAM verification, so the
        # stub returns the same "cleared, reading unavailable" tuple
        # verify_vram_cleared itself returns on its own dev-tolerance path.
        async def fake_vram_verify(*a, **k):
            return True, None

        monkeypatch.setattr(mgr, "_vram_verify", fake_vram_verify, raising=False)
        monkeypatch.setattr(mgr, "_persist_clean_bin_to_ssd", lambda *a, **k: None)

        handle = _fake_handle()
        mgr._active_handle = handle
        slot = Slot.new(_QWEN, thread_id="t-teardown-grace", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30, marker="GRACEFN")})
        with caplog.at_level(logging.INFO):
            await mgr._teardown(slot, "grace-expired")
        grace_lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                                 call_site="probe_and_save_clean_kv", tier="RAM")
        assert len(grace_lines) == 1, caplog.records
        assert "trigger=grace-expired" in grace_lines[0]
        assert any("action=save" in url for (url, _j) in posts), (
            "trigger=grace-expired logged but no real save POST was observed"
        )

        caplog.clear()
        posts.clear()
        handle2 = _fake_handle()
        mgr._active_handle = handle2
        slot2 = Slot.new(_QWEN, thread_id="t-teardown-crash", admission_ctx_len=45000,
                          client_meta={"messages": _msgs(30, marker="CRASHFN")})
        with caplog.at_level(logging.INFO):
            await mgr._teardown(slot2, "worker-uncaught-exception")
        crash_lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                                 call_site="probe_and_save_clean_kv", tier="RAM")
        assert len(crash_lines) == 1, caplog.records
        assert "trigger=worker-uncaught-exception" in crash_lines[0], (
            "same call_site as grace-expired above, must read back a "
            f"different, real reason: {crash_lines[0]}"
        )

    async def test_teardown_ssd_persist_trigger_matches_reason_for_all_three_reasons(
        self, mgr, kv_dirs, make_httpx, monkeypatch, caplog
    ):
        """The two tests above prove _probe_and_save_clean_kv's
        own RAM-tier trigger reads back correctly for grace-expired and
        worker-uncaught-exception. This proves the SEPARATE SSD-persist call
        (_persist_clean_bin_to_ssd, the clean-bin
        RAM->SSD persist log line) also carries the
        matching trigger= for all three of _teardown's real reasons --
        including loading-fail-health-timeout, which nothing in this repo
        tested before, even though the manager reaches _teardown
        with it on every real health-timeout failure."""
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)

        # see the sibling test above -- _teardown unpacks
        # _vram_verify's real (bool, int|None) return; stub the same
        # "cleared, reading unavailable" tuple, unrelated to this test's actual
        # concern (SSD-persist trigger threading).
        async def fake_vram_verify(*a, **k):
            return True, None

        monkeypatch.setattr(mgr, "_vram_verify", fake_vram_verify, raising=False)

        persist_calls = []

        def spy_persist(*a, **k):
            persist_calls.append(k)

        monkeypatch.setattr(mgr, "_persist_clean_bin_to_ssd", spy_persist)

        for reason in (
            "grace-expired", "worker-uncaught-exception", "loading-fail-health-timeout",
        ):
            persist_calls.clear()
            handle = _fake_handle()
            mgr._active_handle = handle
            slot = Slot.new(
                _QWEN, thread_id=f"t-teardown-{reason}", admission_ctx_len=45000,
                client_meta={"messages": _msgs(30, marker=reason)},
            )
            await mgr._teardown(slot, reason)
            assert len(persist_calls) == 1, (reason, persist_calls)
            assert persist_calls[0].get("trigger") == reason, (
                f"reason={reason!r} but SSD persist got "
                f"trigger={persist_calls[0].get('trigger')!r}"
            )

    async def test_disposable_role_displacement_persists_old_identity_with_ownership_transfer_trigger(
        self, mgr, kv_dirs, make_httpx, caplog
    ):
        """A third real shape the trigger= field must cover, distinct from _teardown
        and _teardown_idle_holder: _probe_and_save_clean_kv's own
        disposable-role gate (in manager.py) fire-and-forgets an
        SSD persist of the OUTGOING identity before parking a disposable
        role's per-turn turn. Reuses the exact main-then-curator sequence
        TestDisplacementDoubleFireLegibility already proved produces one
        RAM-tier trigger=displacement firing -- that firing and this SSD
        persist are two separate instruments over the same real event, so
        driving the same sequence and spying on the SSD call proves the
        trigger= reaches this, the hardest-to-reach of the three
        ownership_transfer sites (a disposable role must first displace an
        already-anchored different identity)."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])

        persist_calls = []
        orig = mgr._persist_clean_bin_to_ssd_by_hash

        def spy_persist(*a, **k):
            persist_calls.append((a, k))
            return orig(*a, **k)

        mgr._persist_clean_bin_to_ssd_by_hash = spy_persist

        main_slot = Slot.new(
            _QWEN, thread_id="t-owntrans-main", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="OWNTRANSMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        # admission_ctx_len must clear _MIN_CTX_LEN (40000) -- unlike
        # TestDisplacementDoubleFireLegibility's short (3-message) curator,
        # which deliberately hits KV_DECLINE_BELOW_MIN_CTX and returns
        # BEFORE ever reaching this function's disposable-role-gate block
        # (in manager.py, the actual ownership_transfer persist
        # site). A long-enough curator turn passes that bar and reaches it.
        curator_slot = Slot.new(
            _QWEN, thread_id="t-owntrans-cur", admission_ctx_len=45000,
            client_meta={
                "is_curator": True, "session_id": "s-owntrans",
                "messages": _msgs(30, marker="OWNTRANSCUR")},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator_slot)

        # the persist is fire-and-forget (_spawn_bg + asyncio.to_thread) --
        # drain before asserting, the same mechanism the manager itself
        # uses at shutdown to guarantee these complete.
        await mgr._drain_bg_tasks(grace_s=5.0)

        assert len(persist_calls) == 1, persist_calls
        _, kwargs = persist_calls[0]
        assert kwargs.get("trigger") == "ownership_transfer", kwargs

        # cross-check against the sibling RAM-tier instrument proven in
        # TestDisplacementDoubleFireLegibility -- same real event, two
        # different log surfaces, both must agree it happened once.
        authoritative = _kv_lines(caplog, "KV_SAVE_FIRING",
                                   call_site="probe_and_save_clean_kv")
        assert len(authoritative) == 1, caplog.records
        assert _field(authoritative[0], "trigger") == "displacement"


# ================================================================================
# Double-fire legibility: the displacement seam's provisional pre-announcement
# and its authoritative record must not look identical, and a count
# filtered on the authoritative call_site must not double-count.
# ================================================================================
class TestDisplacementDoubleFireLegibility:
    async def test_provisional_and_authoritative_lines_are_distinguishable(
        self, mgr, kv_dirs, make_httpx, caplog
    ):
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        main_slot = Slot.new(
            _QWEN, thread_id="t-dblfire-main", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="DBLFIREMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        curator_slot = Slot.new(
            _QWEN, thread_id="t-dblfire-cur", client_meta={
                "is_curator": True, "session_id": "s-dblfire",
                "messages": _msgs(3)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator_slot)

        provisional = _kv_lines(caplog, "KV_SAVE_FIRING",
                                 call_site="displacement_seam")
        authoritative = _kv_lines(caplog, "KV_SAVE_FIRING",
                                   call_site="probe_and_save_clean_kv")
        assert len(provisional) == 1, caplog.records
        assert len(authoritative) == 1, caplog.records

        prov_trigger = _field(provisional[0], "trigger")
        auth_trigger = _field(authoritative[0], "trigger")
        assert prov_trigger == "seam_preflight", provisional[0]
        assert auth_trigger == "displacement", authoritative[0]
        # a specific defect guarded against here: "displacement" IS a
        # substring of "displacement_provisional", so a naive
        # `grep -c trigger=displacement` still matched both lines for one
        # save. Prove the actual chosen value has no such overlap in either
        # direction, not just that it LOOKS different by eye.
        assert "displacement" not in prov_trigger, prov_trigger
        assert prov_trigger not in auth_trigger and auth_trigger not in prov_trigger

        assert _field(provisional[0], "bin_id") == "pending-recursive-save"
        assert _field(authoritative[0], "bin_id") != "pending-recursive-save"

        # tier -- both lines are RAM-tier (the provisional line never
        # touches disk at all; the authoritative one fires only after the
        # RAM/tmpfs save confirms). A test that
        # checked trigger/bin_id but never tier would let a mutation flipping
        # JUST the provisional line's tier to "SSD" pass every assertion
        # here -- production would then over-count the SSD audit by one per
        # displacement. Check both explicitly.
        assert _field(provisional[0], "tier") == "RAM", provisional[0]
        assert _field(authoritative[0], "tier") == "RAM", authoritative[0]

        # Counting rule, proven here two ways: filtering on
        # the authoritative call_site alone yields exactly one save: a
        # substring grep for "trigger=displacement" ALSO yields exactly
        # one match (it would not if the
        # provisional line's own trigger value still matched that grep).
        all_firings_this_event = _kv_lines(caplog, "KV_SAVE_FIRING")
        by_call_site = [l for l in all_firings_this_event
                        if "call_site=probe_and_save_clean_kv" in l]
        by_trigger_substring = [l for l in all_firings_this_event
                                 if "trigger=displacement" in l]
        assert len(by_call_site) == 1, (
            "counting rule broken -- provisional line leaked into the "
            f"authoritative call_site filter: {by_call_site}"
        )
        assert len(by_trigger_substring) == 1, (
            "the provisional line's trigger value collides with a "
            f"substring grep for the authoritative one: {by_trigger_substring}"
        )


# ================================================================================
# tier= -- RAM sites always say RAM, the SSD-writer site is the only
# place tier=SSD can ever appear, and the two are simultaneously visible.
# ================================================================================
class TestTierField:
    async def test_ssd_persist_reports_the_bin_it_actually_copied(
        self, mgr, kv_dirs, caplog
    ):
        """The line's bin_id is read OUT of the emitted text and checked
        against the real filesystem, not compared to a name this test
        pre-computed with the same formula the code uses -- that coincidence
        would pass even if the code named a bin that was never copied. A test
        built on exactly
        that coincidence would pass with the field mutated to a fake
        "X-NEVER-WRITTEN" bin_id and a blanked thread_id."""
        save_dir, persist_dir = kv_dirs
        th = "deadbeefcafe0123"
        bin_fn = f"{_QWEN}.p{_PORT}.{th}.slot0.bin"
        meta_fn = f"{_QWEN}.p{_PORT}.{th}.slot0.json"
        bin_content = b"dummy kv cache data"
        (save_dir / bin_fn).write_bytes(bin_content)
        (save_dir / meta_fn).write_text("{}")

        with caplog.at_level(logging.INFO):
            mgr._persist_clean_bin_to_ssd_by_hash(_QWEN, th, _PORT, trigger="test_direct_call")

        ssd_lines = _kv_lines(caplog, "KV_SAVE_FIRING", tier="SSD",
                               call_site="persist_clean_bin_to_ssd_by_hash")
        assert len(ssd_lines) == 1, caplog.records
        assert "trigger=ssd_persist" in ssd_lines[0]

        emitted_bin_id = _field(ssd_lines[0], "bin_id")
        emitted_thread_hash = _field(ssd_lines[0], "thread_hash")
        assert emitted_bin_id, ssd_lines[0]
        assert emitted_thread_hash == th, (
            f"thread_hash on the firing must be the real identity, not "
            f"blank or a placeholder: {ssd_lines[0]}"
        )
        claimed_path = persist_dir / emitted_bin_id
        assert claimed_path.exists(), (
            f"KV_SAVE_FIRING claimed bin_id={emitted_bin_id!r} but no such "
            f"file exists in the SSD tier -- the line named a bin it never "
            f"verified was copied"
        )
        assert claimed_path.read_bytes() == bin_content, (
            "the claimed bin_id exists but its content doesn't match what "
            "was actually persisted -- wrong file entirely"
        )

        ram_lines_same_capture = _kv_lines(caplog, "KV_SAVE_FIRING", tier="RAM")
        assert len(ram_lines_same_capture) == 0, (
            "this call path is SSD-only -- no RAM-tier line should appear here"
        )

    async def test_ssd_persist_does_not_fire_when_no_bin_file_was_copied(
        self, mgr, kv_dirs, caplog
    ):
        """Reproduces the lone-sidecar gap: a lone sidecar (.json)
        present, plus a mid-write .bin.tmp that the .tmp guard correctly
        skips. copied>0 (the .json copies) but no .bin ever reaches the SSD
        tier -- the firing line must not claim one exists anywhere."""
        save_dir, persist_dir = kv_dirs
        th = "onlyjsoncafe0123"
        meta_fn = f"{_QWEN}.p{_PORT}.{th}.slot0.json"
        tmp_bin_fn = f"{_QWEN}.p{_PORT}.{th}.slot0.bin.tmp"
        real_bin_fn = f"{_QWEN}.p{_PORT}.{th}.slot0.bin"
        (save_dir / meta_fn).write_text("{}")
        (save_dir / tmp_bin_fn).write_bytes(b"mid-write, not a real .bin yet")

        with caplog.at_level(logging.INFO):
            mgr._persist_clean_bin_to_ssd_by_hash(_QWEN, th, _PORT, trigger="test_direct_call")

        ssd_lines = _kv_lines(caplog, "KV_SAVE_FIRING", tier="SSD")
        assert len(ssd_lines) == 0, (
            f"no .bin file was ever copied to the SSD tier -- a "
            f"KV_SAVE_FIRING line here would name a bin that exists in "
            f"neither tier: {ssd_lines}"
        )
        # setup sanity: the sidecar-count log (existing, untouched) still
        # fires and the json really was copied -- this test is about the
        # firing line's honesty specifically, not about that other log.
        assert (persist_dir / meta_fn).exists()
        assert not (persist_dir / real_bin_fn).exists()

    async def test_ram_site_never_reports_tier_ssd(
        self, mgr, kv_dirs, make_httpx, caplog
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot = Slot.new(_QWEN, thread_id="t-tierram", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30, marker="TIERRAM")})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _kv_lines(caplog, "KV_SAVE_FIRING",
                           call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, caplog.records
        assert "tier=RAM" in lines[0]
        assert "tier=SSD" not in lines[0]
