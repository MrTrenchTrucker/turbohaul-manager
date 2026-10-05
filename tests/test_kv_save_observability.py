"""Observability hardening -- every path that can save or displace
main-class KV emits exactly one of KV_SAVE_FIRING / KV_SAVE_DECLINE.

A save/kill opportunity could otherwise complete with neither line
emitted (see e.g. the cancellation-unwind path in _process_slot, or a
booting-pid reap) — the only way to know KV was thrown away would be to read the
raw engine log by eye. Each test below drives one specific opportunity and
asserts the line fires with the expected reason_name, exactly once.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest <this test file> -v
"""
from __future__ import annotations

import asyncio
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
from turbohaul.manager import (
    KV_DECLINE_BELOW_MIN_CTX,
    KV_DECLINE_CANCELLATION_UNWIND_NO_SAVE,
    KV_DECLINE_DIRTY_TIP_ERASE_FAILED,
    KV_DECLINE_DISPLACEMENT_COMPRESSION_EXCLUDED,
    KV_DECLINE_DISPLACEMENT_NO_MAIN_IDENTITY,
    KV_DECLINE_DISPLACEMENT_STALE_MODEL,
    KV_DECLINE_DISPOSABLE_ROLE_SAVE_OFF,
    KV_DECLINE_FORCE_COLD_NO_SAVE,
    KV_DECLINE_HANDLE_NOT_ALIVE,
    KV_DECLINE_NEVER_OVERWRITE_SMALLER,
    KV_DECLINE_NO_MESSAGES,
    KV_DECLINE_NO_SLOT_IDENTITY,
    KV_DECLINE_SAVE_DISABLED,
    KV_DECLINE_SAVE_NOT_CONFIRMED,
    KV_DECLINE_SINGLE_SERIES_GATE,
    KV_DECLINE_UNLOAD_SEAM_ONLY,
    KV_DECLINE_UNVERIFIED_EXIT,
    KV_DECLINE_ZERO_GROWTH_THROTTLE,
    KV_DECLINE_ZERO_TURN_CHAIN,
    TurbohaulManager,
)
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_PORT = 59800

pytestmark = pytest.mark.asyncio


# --- fixtures (same shape as the other KV save tests) --

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


def _verdicts(caplog, *, kind=None, reason=None, call_site=None):
    """Parse KV_SAVE_FIRING/KV_SAVE_DECLINE lines out of captured records,
    optionally filtered. Returns the matching log record messages."""
    out = []
    for r in caplog.records:
        msg = r.message
        if kind and not msg.startswith(kind):
            continue
        if not (msg.startswith("KV_SAVE_FIRING") or msg.startswith("KV_SAVE_DECLINE")):
            continue
        if reason is not None and f"reason_name={reason}" not in msg:
            continue
        if call_site is not None and f"call_site={call_site}" not in msg:
            continue
        out.append(msg)
    return out


# ================================================================================
# Every internal decline point in _probe_and_save_clean_kv, named and exactly-one.
# ================================================================================
class TestChokepointDeclines:
    async def test_discriminator_save_disabled_declines_exactly_once(self, mgr, kv_dir, caplog):
        mgr._clean_prefix_save_enabled = False
        handle = _fake_handle()
        slot = Slot.new(_QWEN, thread_id="t-disabled", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_SAVE_DISABLED}" in lines[0]

    async def test_discriminator_single_series_gate_declines_exactly_once(self, mgr, kv_dir, caplog):
        handle = _fake_handle()
        handle.parallel = 2  # single-series precondition violated
        slot = Slot.new(_QWEN, thread_id="t-series", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_SINGLE_SERIES_GATE}" in lines[0]

    async def test_discriminator_below_min_ctx_declines_exactly_once(self, mgr, kv_dir, caplog):
        handle = _fake_handle()
        slot = Slot.new(_QWEN, thread_id="t-short", admission_ctx_len=100,
                         client_meta={"messages": _msgs(2)})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_BELOW_MIN_CTX}" in lines[0]

    async def test_discriminator_no_messages_declines_exactly_once(self, mgr, kv_dir, caplog):
        handle = _fake_handle()
        slot = Slot.new(_QWEN, thread_id="t-nomsgs", admission_ctx_len=45000,
                         client_meta={"messages": []})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_NO_MESSAGES}" in lines[0]

    async def test_discriminator_disposable_role_save_off_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot = Slot.new(
            _QWEN, thread_id="t-disposable", admission_ctx_len=45000,
            client_meta={"is_curator": True, "save_kv": False,
                         "messages": _msgs(30)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DISPOSABLE_ROLE_SAVE_OFF}" in lines[0]

    async def test_discriminator_unload_seam_only_park_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        """Main's own per-turn save is OFF by design --
        save_to_disk=False on a main-class turn always parks, never probes."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot = Slot.new(
            _QWEN, thread_id="t-mainpark", admission_ctx_len=45000,
            client_meta={"is_main": True, "messages": _msgs(30)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=False)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_UNLOAD_SEAM_ONLY}" in lines[0]

    async def test_discriminator_zero_turn_chain_declines_exactly_once(
        self, mgr, kv_dir, monkeypatch, caplog
    ):
        handle = _fake_handle()
        slot = Slot.new(_QWEN, thread_id="t-zerochain", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})
        import turbohaul.manager as manager_mod
        monkeypatch.setattr(manager_mod, "_prefix_hash_chain", lambda messages: [])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_ZERO_TURN_CHAIN}" in lines[0]

    async def test_discriminator_never_overwrite_smaller_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        """The saved bin's prompt_len is derived from the REAL rendered
        content, not from admission_ctx_len (that field only gates the
        min-ctx-bar) -- 70 messages of filler renders to ~49.5k chars.
        The second call's admission_ctx_len must clear the 40000 min-ctx
        bar but still read as SMALLER than that real saved length."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot_big = Slot.new(_QWEN, thread_id="t-nover", admission_ctx_len=90000,
                             client_meta={"messages": _msgs(70, marker="BIGGER")})
        await mgr._probe_and_save_clean_kv(handle, slot_big, save_to_disk=True)

        slot_small = Slot.new(_QWEN, thread_id="t-nover", admission_ctx_len=41000,
                               client_meta={"messages": _msgs(3, marker="SMALLER")})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot_small, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_NEVER_OVERWRITE_SMALLER}" in lines[0]

    async def test_discriminator_zero_growth_throttle_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        """save_to_disk=True forces _min_growth=1 (any growth passes) --
        only EXACT zero turn-count growth throttles. admission_ctx_len on
        the second call is set well above the first save's real rendered
        length so never-overwrite-smaller does not fire first."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot_1 = Slot.new(_QWEN, thread_id="t-throttle", admission_ctx_len=90000,
                           client_meta={"messages": _msgs(60, marker="THROT")})
        await mgr._probe_and_save_clean_kv(handle, slot_1, save_to_disk=True)

        slot_2 = Slot.new(_QWEN, thread_id="t-throttle", admission_ctx_len=90000,
                           client_meta={"messages": _msgs(60, marker="THROT")})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot_2, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_ZERO_GROWTH_THROTTLE}" in lines[0]

    async def test_discriminator_dirty_tip_erase_failed_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, monkeypatch, caplog
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        mgr._kv_dirty_tail = {
            _PORT: {"role": "curator", "pid": handle.pid, "thread": "foreign", "ts": 0.0}
        }

        class _FailErase:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                raise RuntimeError("erase-probe network failure")

            async def post(self, url, **kw):
                raise RuntimeError("erase-probe network failure")

        import turbohaul.manager as manager_mod

        class _FakeHttpxFail:
            AsyncClient = staticmethod(lambda *a, **k: _FailErase())
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpxFail)

        slot = Slot.new(_QWEN, thread_id="t-dirtyerase", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DIRTY_TIP_ERASE_FAILED}" in lines[0]

    async def test_discriminator_save_not_confirmed_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, monkeypatch, caplog
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot = Slot.new(_QWEN, thread_id="t-notconfirmed", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})

        async def fake_save_slot_kv(*a, **k):
            return False

        monkeypatch.setattr(mgr, "_save_slot_kv", fake_save_slot_kv)
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_SAVE_NOT_CONFIRMED}" in lines[0]

    async def test_discriminator_confirmed_save_fires_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        slot = Slot.new(_QWEN, thread_id="t-fires", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30, marker="CONFIRMEDFIRE")})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _verdicts(caplog, kind="KV_SAVE_FIRING", call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        # non-vacuous: a real save POST actually happened, not just the log line
        assert any("action=save" in url for (url, _j) in posts), (
            "KV_SAVE_FIRING logged but no real save POST was observed"
        )
        all_verdicts = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(all_verdicts) == 1, (
            "expected exactly one primary verdict for this invocation, "
            f"got {all_verdicts}"
        )

    async def test_discriminator_unhandled_exception_backstop_fires_exactly_once(
        self, mgr, kv_dir, monkeypatch, caplog
    ):
        """Structural guarantee: an exception thrown before ANY explicit
        verdict still gets exactly one decline, via the finally-backstop --
        not two (no double-emission with the except block's own debug log),
        not zero."""
        handle = _fake_handle()
        slot = Slot.new(_QWEN, thread_id="t-crash", admission_ctx_len=45000,
                         client_meta={"messages": _msgs(30)})

        def _boom(*a, **k):
            raise RuntimeError("simulated crash before any verdict")

        monkeypatch.setattr(mgr, "_bin_role" if hasattr(mgr, "_bin_role") else "_thread_hash", _boom, raising=False)
        # _thread_hash is a bound method reached well after the early gates;
        # patch it directly to force a late-stage exception that still must
        # resolve to exactly one verdict via the backstop.
        monkeypatch.setattr(mgr, "_thread_hash", _boom)
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
        lines = _verdicts(caplog, call_site="probe_and_save_clean_kv")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_UNVERIFIED_EXIT}" in lines[0]


# ================================================================================
# The displacement seam — its own independent verdict, separate from the
# invocation's own primary verdict above.
# ================================================================================
class TestDisplacementSeamDeclines:
    async def test_discriminator_displacement_no_main_identity_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        """A disposable visitor arrives with no prior main turn on this port
        at all -- the displacement seam has nothing to displace."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        curator_slot = Slot.new(
            _QWEN, thread_id="t-nomain", client_meta={
                "is_curator": True, "session_id": "s-nomain",
                "messages": _msgs(3)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator_slot)
        lines = _verdicts(caplog, call_site="displacement_seam")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DISPLACEMENT_NO_MAIN_IDENTITY}" in lines[0]

    async def test_discriminator_displacement_compression_excluded_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        main_slot = Slot.new(
            _QWEN, thread_id="t-compexcl", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="COMPMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        compression_slot = Slot.new(
            _QWEN, thread_id="t-compexcl2", client_meta={
                "is_compression": True, "session_id": "s-compexcl",
                "messages": _msgs(3)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, compression_slot)
        lines = _verdicts(caplog, call_site="displacement_seam")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DISPLACEMENT_COMPRESSION_EXCLUDED}" in lines[0]

    async def test_discriminator_displacement_stale_model_declines_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        handle_a = _fake_handle(model_tag=_QWEN, port=_PORT)
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        main_slot = Slot.new(
            _QWEN, thread_id="t-stale", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="STALEMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle_a, main_slot)

        handle_b = _fake_handle(model_tag="a-different-model", port=_PORT)
        curator_slot = Slot.new(
            _QWEN, thread_id="t-stale2", client_meta={
                "is_curator": True, "session_id": "s-stale",
                "messages": _msgs(3)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle_b, curator_slot)
        lines = _verdicts(caplog, call_site="displacement_seam")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DISPLACEMENT_STALE_MODEL}" in lines[0]

    async def test_discriminator_displacement_fires_exactly_once(
        self, mgr, kv_dir, make_httpx, caplog
    ):
        handle = _fake_handle()
        posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        main_slot = Slot.new(
            _QWEN, thread_id="t-seam-fire", admission_ctx_len=45000,
            client_meta={"messages": _msgs(30, marker="SEAMFIREMAIN")},
        )
        await mgr._probe_and_save_clean_kv(handle, main_slot)

        curator_slot = Slot.new(
            _QWEN, thread_id="t-seam-fire2", client_meta={
                "is_curator": True, "session_id": "s-seam-fire",
                "messages": _msgs(3)},
        )
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, curator_slot)
        lines = _verdicts(caplog, kind="KV_SAVE_FIRING", call_site="displacement_seam")
        assert len(lines) == 1, lines
        assert any("SEAMFIREMAIN" in str(j) for (_u, j) in posts if j), (
            "displacement_seam FIRING logged but main's own content was "
            "never actually re-posted to the probe"
        )
        seam_lines = _verdicts(caplog, call_site="displacement_seam")
        assert len(seam_lines) == 1, seam_lines


# ================================================================================
# The four kill-path defs that never route through the chokepoint at all.
# ================================================================================
class TestExternalKillPathDeclines:
    async def test_discriminator_reap_booting_pid_declines_exactly_once(self, mgr, caplog):
        import os
        # A pid guaranteed not to be our child: waitpid raises ChildProcessError
        # immediately, _term_and_reap no-ops -- the decline must still fire.
        bogus_pid = os.getpid() + 999_000
        with caplog.at_level(logging.INFO):
            await mgr._reap_booting_pid(bogus_pid)
        lines = _verdicts(caplog, call_site="reap_booting_pid")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_NO_SLOT_IDENTITY}" in lines[0]

    async def test_discriminator_reap_resident_handle_declines_exactly_once(
        self, mgr, monkeypatch, caplog
    ):
        handle = _fake_handle()

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._reap_resident_handle(handle)
        lines = _verdicts(caplog, call_site="reap_resident_handle")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_NO_SLOT_IDENTITY}" in lines[0]

    async def test_discriminator_force_cold_defensive_teardown_declines_exactly_once(
        self, mgr, monkeypatch, caplog
    ):
        handle = _fake_handle()

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        mgr._active_handle = handle
        slot = Slot.new(_QWEN, thread_id="t-forcecold")
        slot.pid = handle.pid
        with caplog.at_level(logging.INFO):
            await mgr._force_cold(slot, "worker-uncaught-exception")
        lines = _verdicts(caplog, call_site="force_cold")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_FORCE_COLD_NO_SAVE}" in lines[0]


# ================================================================================
# _teardown / _teardown_idle_holder — the two sub-cases that could otherwise be silent even
# though the def itself is a real chokepoint caller on the happy path.
# ================================================================================
class TestTeardownDeadHandleDeclines:
    async def test_discriminator_teardown_dead_handle_declines_exactly_once(
        self, mgr, monkeypatch, caplog
    ):
        handle = MagicMock()
        handle.is_alive.return_value = False
        handle.port = _PORT
        handle.pid = 55555

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        mgr._active_handle = handle
        slot = Slot.new(_QWEN, thread_id="t-teardowndead")
        with caplog.at_level(logging.INFO):
            await mgr._teardown(slot, "worker-uncaught-exception")
        lines = _verdicts(caplog, call_site="teardown")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_HANDLE_NOT_ALIVE}" in lines[0]

    async def test_discriminator_teardown_idle_holder_dead_handle_declines_exactly_once(
        self, mgr, monkeypatch, caplog
    ):
        held = MagicMock()
        held.is_alive.return_value = False
        held.port = _PORT
        held.pid = 66666
        mgr._idle_handle = held
        mgr._idle_model_tag = _QWEN
        mgr._idle_thread_id = "t-idledead"
        mgr._idle_admission_ctx_len = 0
        mgr._idle_client_meta = {"messages": []}

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_expired")
        lines = _verdicts(caplog, call_site="teardown_idle_holder")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_HANDLE_NOT_ALIVE}" in lines[0]

    async def test_discriminator_teardown_idle_holder_disposable_declines_exactly_once(
        self, mgr, monkeypatch, caplog
    ):
        held = MagicMock()
        held.is_alive.return_value = True
        held.port = _PORT
        held.pid = 77777
        mgr._idle_handle = held
        mgr._idle_model_tag = _QWEN
        mgr._idle_thread_id = "hermes-sub-disposable"
        mgr._idle_admission_ctx_len = 0
        mgr._idle_client_meta = {"messages": []}

        async def fake_sigterm(*a, **k):
            return True, "ok"

        monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
        with caplog.at_level(logging.INFO):
            await mgr._teardown_idle_holder("idle_expired")
        lines = _verdicts(caplog, call_site="teardown_idle_holder")
        assert len(lines) == 1, lines
        assert f"reason_name={KV_DECLINE_DISPOSABLE_ROLE_SAVE_OFF}" in lines[0]
