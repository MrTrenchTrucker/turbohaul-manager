"""Observed-concurrency gate: the single-series KV-save gate keys on OBSERVED
concurrency, not the static max_parallel_sidecars cap.

Design rule: the KV save must be gated on whether
another sidecar is actually ACTIVE at save-time, NOT on the global
max_parallel_sidecars setting — so KV restore stays GREEN (a clean bin gets
saved) regardless of the parallel-sidecar setting, and only a genuinely
concurrent ACTIVE sidecar still declines the save.

Repro this closes: with
max_parallel_sidecars=2 every sub-agent turn logged
  "clean-prefix save SKIPPED: single-series gate (max_parallel_sidecars=2,
   handle.parallel=1 != 1) -> no clean bin saved"
plus KV_RESTORE final_status=unverified — the static config alone killed
every clean save on every request even when the host was effectively single-series.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/ -k observed_concurrency -v
"""
from __future__ import annotations

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
    KV_DECLINE_SINGLE_SERIES_GATE,
    Resident,
    ResidentState,
    TurbohaulManager,
)
from turbohaul.slot import Slot
from turbohaul.subprocess_mgr import SidecarHandle

_QWEN = "qwen3.6-27b"
_OTHER = "ornith-1.5-35b-mini"
_PORT = 59_801

pytestmark = pytest.mark.asyncio


# --- fixtures (mirror tests/test_kv_save_observability.py) -----------

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
    proc.pid = 88_889
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _verdicts(caplog, *, kind=None, reason=None):
    out = []
    for r in caplog.records:
        msg = r.message
        if kind and not msg.startswith(kind):
            continue
        if not (msg.startswith("KV_SAVE_FIRING") or msg.startswith("KV_SAVE_DECLINE")):
            continue
        if reason is not None and f"reason_name={reason}" not in msg:
            continue
        out.append(msg)
    return out


def _single_series_declines(caplog):
    return _verdicts(caplog, reason=KV_DECLINE_SINGLE_SERIES_GATE)


def _save_slot():
    return Slot.new(_QWEN, thread_id="t-obs", admission_ctx_len=45000,
                    client_meta={"messages": _msgs(30)})


# ============================================================================
# The gate sees the WORLD, not the config
# ============================================================================
class TestObservedConcurrencyGate:

    async def test_two_resident_cap_other_resident_idle_save_proceeds(self, mgr, kv_dir, make_httpx, caplog):
        """THE fix: cap=2 + the other resident IDLE_EVICTABLE == effectively
        single-series -> the clean save proceeds and the bin is written."""
        mgr.runtime.queue.max_parallel_sidecars = 2
        other = _fake_handle(model_tag=_OTHER, port=_PORT + 1)
        mgr._residents[_OTHER] = Resident(model_tag=_OTHER, state=ResidentState.IDLE_EVICTABLE, handle=other)
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot(), save_to_disk=True)
        assert not _single_series_declines(caplog), _single_series_declines(caplog)
        assert _verdicts(caplog, kind="KV_SAVE_FIRING"), "save did not fire — gate or later stage blocked it"

    async def test_two_resident_cap_other_resident_grace_save_proceeds(self, mgr, kv_dir, make_httpx, caplog):
        """GRACE (context finished, timer armed) is NOT an active series:
        eviction only targets IDLE_EVICTABLE at a later admission, so the
        probe->save window is uninterleaved and the save proceeds."""
        mgr.runtime.queue.max_parallel_sidecars = 2
        other = _fake_handle(model_tag=_OTHER, port=_PORT + 1)
        mgr._residents[_OTHER] = Resident(model_tag=_OTHER, state=ResidentState.GRACE, handle=other)
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot(), save_to_disk=True)
        assert not _single_series_declines(caplog)
        assert _verdicts(caplog, kind="KV_SAVE_FIRING")

    async def test_two_resident_cap_other_resident_active_still_declines(self, mgr, kv_dir, make_httpx, caplog):
        """A genuinely ACTIVE other sidecar breaks single-series for real:
        observed=1 -> exactly one SINGLE_SERIES_GATE decline (observed count
        in the line, config no longer the trigger)."""
        mgr.runtime.queue.max_parallel_sidecars = 2
        other = _fake_handle(model_tag=_OTHER, port=_PORT + 1)
        mgr._residents[_OTHER] = Resident(model_tag=_OTHER, state=ResidentState.ACTIVE, handle=other)
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot(), save_to_disk=True)
        lines = _single_series_declines(caplog)
        assert len(lines) == 1, lines
        # The observed count is reported in the SKIPPED warning (the line
        # operators read first); the structured DECLINE line carries reason+role.
        assert any(
            "OBSERVED 1 other ACTIVE sidecar(s)" in r.message for r in caplog.records
        ), [r.message for r in caplog.records if "single-series gate" in r.message]

    async def test_one_resident_cap_no_residents_behavior_unchanged(self, mgr, kv_dir, make_httpx, caplog):
        """Regression: the cap==1 singleton world (no model_tag residents) is
        unchanged by the gate — save proceeds as it always did."""
        handle = _fake_handle()
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot(), save_to_disk=True)
        assert not _single_series_declines(caplog)
        assert _verdicts(caplog, kind="KV_SAVE_FIRING")

    async def test_own_active_handle_is_not_counted(self, mgr, kv_dir, make_httpx, caplog):
        """The repro shape: the SAVING resident is itself ACTIVE (it is
        serving this turn) while the other resident is idle. Self-exclusion
        by handle identity means observed=0 -> the save proceeds."""
        mgr.runtime.queue.max_parallel_sidecars = 2
        handle = _fake_handle()
        mgr._residents[_QWEN] = Resident(model_tag=_QWEN, state=ResidentState.ACTIVE, handle=handle)
        other = _fake_handle(model_tag=_OTHER, port=_PORT + 1)
        mgr._residents[_OTHER] = Resident(model_tag=_OTHER, state=ResidentState.IDLE_EVICTABLE, handle=other)
        make_httpx([{"id": 0, "n_prompt_tokens": 100}])
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot(), save_to_disk=True)
        assert not _single_series_declines(caplog)
        assert _verdicts(caplog, kind="KV_SAVE_FIRING")

    async def test_handle_parallel_gt1_still_declines(self, mgr, kv_dir, caplog):
        """_hp stays a HARD condition: intra-engine parallel contexts displace
        regardless of observed sidecar count (config parallel=2, nobody else
        active -> still a single-series break)."""
        handle = _fake_handle()
        handle.parallel = 2
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, _save_slot())
        lines = _single_series_declines(caplog)
        assert len(lines) == 1, lines

    async def test_decline_line_carries_role_not_none(self, mgr, kv_dir, caplog):
        """Log completeness: the gate's KV_SAVE_DECLINE carries
        a role (normalized; main-class unlabeled traffic -> 'main'), not the
        role=None, which would make declines indistinguishable."""
        handle = _fake_handle()
        handle.parallel = 2
        slot = Slot.new(_QWEN, thread_id="t-role", admission_ctx_len=45000,
                        client_meta={"messages": _msgs(30)})
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _single_series_declines(caplog)
        assert len(lines) == 1, lines
        assert "role=None" not in lines[0]
        assert "role=main" in lines[0]

    async def test_decline_line_carries_sub_role(self, mgr, kv_dir, caplog):
        """A sub-agent turn's decline carries its own role (admission_role
        wins over client_meta, mirroring _tip_role derivation below the gate)."""
        handle = _fake_handle()
        handle.parallel = 2
        slot = Slot.new(_QWEN, thread_id="t-sub", admission_ctx_len=45000,
                        client_meta={"messages": _msgs(30), "is_curator": True})
        slot.admission_role = "curator"
        with caplog.at_level(logging.INFO):
            await mgr._probe_and_save_clean_kv(handle, slot)
        lines = _single_series_declines(caplog)
        assert len(lines) == 1, lines
        assert "role=curator" in lines[0]
