"""Manager-level wiring tests for the kv_ram_cap module.

These tests drive the real TurbohaulManager and pin what its live-protection gather, its save
gate and its accounting wrapper do. The fixtures and the save harness follow
tests/test_kv_hardcap_save_and_gc.py, which is not imported.
One thread, no processes.
"""

import logging
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
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "example-27b"


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


def _write_kv_bin(kv_dir, name: str, size: int, mtime: float, thread_hash: str) -> str:
    """A .bin plus a .json sidecar carrying thread_hash, so the GC can classify the bin."""
    import json

    p = kv_dir / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    meta = {"model_tag": _QWEN, "port": 59500, "slot_id": 0, "thread_hash": thread_hash}
    (kv_dir / (name[:-4] + ".json")).write_text(json.dumps(meta))
    return name


def _set_legacy_active(mgr, thread_id: str) -> None:
    """Make thread_id the legacy singleton active slot, with no identity beyond the thread id."""
    mgr._active_slot = Slot.new(_QWEN, thread_id=thread_id, admission_ctx_len=100, client_meta=None)


class _FakeResident:
    """A model resident as _model_residents() sees it: a live state and the per-resident fields."""

    def __init__(self, key, active_slot=None, inflight=(), idle_thread_id=None):
        self.resident_key = key
        self.state = ResidentState.ACTIVE
        self.active_slot = active_slot
        self.inflight = list(inflight)
        self.idle_thread_id = idle_thread_id
        self.idle_client_meta = None


class _ExplodesOnIteration:
    def __iter__(self):
        raise RuntimeError("inflight boom")


class _BrokenInflightResident:
    """A resident whose inflight list raises the moment it is iterated."""

    def __init__(self, key, active_slot):
        self.resident_key = key
        self.state = ResidentState.ACTIVE
        self.active_slot = active_slot
        self.idle_thread_id = None
        self.idle_client_meta = None

    @property
    def inflight(self):
        return _ExplodesOnIteration()


class _SaveHarness:
    """httpx fake: GET /slots returns the given slots, POST records the call and writes the bin."""

    def __init__(self, kv_dir, slots_payload, bin_bytes: int = 50):
        self.kv_dir = kv_dir
        self.slots_payload = slots_payload
        self.bin_bytes = bin_bytes
        self.posts: list = []
        self.gets: list = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _client(self):
        outer = self

        class _Client:
            async def __aenter__(self_):
                return self_

            async def __aexit__(self_, *a):
                return False

            async def get(self_, url):
                outer.gets.append(url)
                if "/slots" in url and "action=save" not in url:
                    return _SaveHarness._Resp(outer.slots_payload)
                return _SaveHarness._Resp({})

            async def post(self_, url, json=None, **kw):
                outer.posts.append((url, json))
                if "action=save" in url and json and "filename" in json:
                    target = os.path.join(str(outer.kv_dir), json["filename"])
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as f:
                        f.write(b"\x00" * outer.bin_bytes)
                return _SaveHarness._Resp({"n_saved": 1, "n_written": outer.bin_bytes})

        return _Client()


class _FakeHttpx:
    @staticmethod
    def AsyncClient(*a, **k):
        return _FakeHttpx._active._client()

    Timeout = staticmethod(lambda *a, **k: None)


def _install_httpx(monkeypatch, harness):
    _FakeHttpx._active = harness
    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


def _slots_payload(thread_id: str, n_prompt_tokens: int = 100):
    return [{"id": 0, "n_prompt_tokens": n_prompt_tokens, "thread_id": thread_id,
             "n_context": 4096, "n_remaining": 2048}]


def test_w1_cap2_resident_active_and_inflight_slots_are_hard_not_soft(mgr, monkeypatch):
    """Pins that the cap>=2 resident scan skips r.inflight and the cap>=2 HARD branch.

    A live resident's active slot and its in-flight slot both land in the HARD set, and neither
    lands in SOFT. The legacy singleton is empty here, so only the resident scan can supply them.
    """
    active = Slot.new(_QWEN, thread_id="w1-active", admission_ctx_len=100, client_meta=None)
    flying = Slot.new(_QWEN, thread_id="w1-inflight", admission_ctx_len=100, client_meta=None)
    monkeypatch.setitem(mgr._residents, "m-w1", _FakeResident("m-w1", active_slot=active, inflight=[flying]))
    hard, soft = mgr._live_kv_protection()
    expected_hard = {mgr._thread_hash("w1-active"), mgr._thread_hash("w1-inflight")}
    assert hard == expected_hard, (
        f"_live_kv_protection: the cap>=2 resident active and in-flight thread hashes must both be HARD; "
        f"expected {sorted(expected_hard)}, got {sorted(hard)}")
    assert soft == set(), f"_live_kv_protection: no SOFT thread was set, so SOFT must be empty; got {sorted(soft)}"


def test_w2_partial_gather_keeps_what_was_gathered_before_the_failure(mgr, monkeypatch, caplog):
    """Pins that split_protection runs inside the gather try, so a mid-gather failure gives empty sets.

    The first resident is gathered fully, the second gathers its active slot and then its inflight
    iteration raises. The HARD set must keep every triple gathered before the raise, and exactly
    one DEBUG record must carry the gather-failure message.
    """
    first_active = Slot.new(_QWEN, thread_id="w2-first-active", admission_ctx_len=100, client_meta=None)
    first_flying = Slot.new(_QWEN, thread_id="w2-first-inflight", admission_ctx_len=100, client_meta=None)
    second_active = Slot.new(_QWEN, thread_id="w2-second-active", admission_ctx_len=100, client_meta=None)
    monkeypatch.setitem(mgr._residents, "m-w2-a",
                        _FakeResident("m-w2-a", active_slot=first_active, inflight=[first_flying]))
    monkeypatch.setitem(mgr._residents, "m-w2-b", _BrokenInflightResident("m-w2-b", second_active))
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")
    hard, soft = mgr._live_kv_protection()
    expected_hard = {mgr._thread_hash("w2-first-active"), mgr._thread_hash("w2-first-inflight"),
                     mgr._thread_hash("w2-second-active")}
    assert hard == expected_hard, (
        f"_live_kv_protection: a mid-gather failure must keep the triples gathered before it; "
        f"expected {sorted(expected_hard)}, got {sorted(hard)}")
    assert soft == set(), f"_live_kv_protection: no SOFT thread was set, so SOFT must be empty; got {sorted(soft)}"
    records = [r for r in caplog.records
               if r.levelno == logging.DEBUG and "live KV protection-class scan failed" in r.getMessage()]
    assert len(records) == 1, (
        f"_live_kv_protection: a mid-gather failure must log exactly one DEBUG record; got {len(records)}")
    assert records[0].exc_info is not None and records[0].exc_info[0] is RuntimeError, (
        "_live_kv_protection: the gather-failure DEBUG record must carry the RuntimeError as exc_info")


def test_w3_idle_held_and_resident_idle_threads_are_soft_not_hard(mgr, monkeypatch):
    """Pins the SOFT gather of self._idle_thread_id and the SOFT gather of each
    resident's idle_thread_id.

    The idle-held thread and one resident's idle thread both land in SOFT. Neither lands in HARD,
    because no live slot is set.
    """
    mgr._idle_thread_id = "w3-idle-held"
    mgr._idle_client_meta = None
    monkeypatch.setitem(mgr._residents, "m-w3", _FakeResident("m-w3", idle_thread_id="w3-resident-idle"))
    hard, soft = mgr._live_kv_protection()
    expected_soft = {mgr._thread_hash("w3-idle-held"), mgr._thread_hash("w3-resident-idle")}
    assert soft == expected_soft, (
        f"_live_kv_protection: the idle-held and the resident idle threads must both be SOFT; "
        f"expected {sorted(expected_soft)}, got {sorted(soft)}")
    assert hard == set(), f"_live_kv_protection: idle threads must not be HARD; got {sorted(hard)}"


@pytest.mark.asyncio
async def test_w4_declined_save_logs_one_warning_naming_dir_and_cap(mgr, kv_dir, monkeypatch, caplog):
    """Pins the slot KV save SKIP warning.

    A HARD active bin holds the floor, so the save is declined. Exactly one WARNING must name the
    dir and cap byte numbers, and the save must return False without any engine POST.
    """
    active_tid = "w4-active"
    _set_legacy_active(mgr, active_tid)
    mgr.runtime.kv.ram_cache_max_bytes = 100
    _write_kv_bin(kv_dir, "m.w4.hard.bin", 200, mtime=1000.0, thread_hash=mgr._thread_hash(active_tid))
    harness = _SaveHarness(kv_dir, _slots_payload(active_tid), bin_bytes=50)
    _install_httpx(monkeypatch, harness)
    slot = Slot.new(_QWEN, thread_id=active_tid, admission_ctx_len=100, client_meta=None)
    caplog.set_level(logging.WARNING, logger="turbohaul.manager")
    result = await mgr._save_slot_kv(59500, _QWEN, slot)
    assert result is False, "_save_slot_kv: a save whose HARD floor holds the cap must be declined (False)"
    assert harness.posts == [], "_save_slot_kv: a declined save must not POST to the engine"
    skips = [r for r in caplog.records
             if r.levelno == logging.WARNING and "slot KV save SKIP (hard RAM cap)" in r.getMessage()]
    assert len(skips) == 1, f"_save_slot_kv: a declined save must log exactly one SKIP warning; got {len(skips)}"
    assert "dir=200 cap=100" in skips[0].getMessage(), (
        f"_save_slot_kv: the SKIP warning must name dir=200 and cap=100; got {skips[0].getMessage()!r}")


def test_w5_dir_accounting_listdir_failure_returns_zero_and_logs_debug(mgr, kv_dir, monkeypatch, caplog):
    """Pins the debug line of _kvcache_dir_accounting (the on_error the wrapper passes to dir_accounting).

    When listdir fails, the wrapper returns (0, []) even though a bin sits in the directory, and it
    logs exactly one DEBUG record, "slot KV dir accounting failed (best-effort)", with exc_info.
    """
    (kv_dir / "m.w5.bin").write_bytes(b"\x00" * 77)
    real_listdir = os.listdir
    target = str(kv_dir)

    def failing_listdir(path=".", *args, **kwargs):
        if os.fspath(path) == target:
            raise OSError("w5 listdir boom")
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", failing_listdir)
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")
    result = mgr._kvcache_dir_accounting()
    assert result == (0, []), f"_kvcache_dir_accounting: a listdir failure must return (0, []); got {result!r}"
    records = [r for r in caplog.records
               if r.levelno == logging.DEBUG and "slot KV dir accounting failed" in r.getMessage()]
    assert len(records) == 1, f"_kvcache_dir_accounting: a listdir failure must log one DEBUG; got {len(records)}"
    assert records[0].exc_info is not None and records[0].exc_info[0] is OSError, (
        "_kvcache_dir_accounting: the DEBUG record must carry the OSError as exc_info")


def test_w6_live_protected_thread_hashes_is_the_plain_union_of_the_two_tiers(mgr):
    """Pins the union view: _live_protected_thread_hashes returns exactly HARD | SOFT as a plain set.

    One live thread lands in HARD and one idle thread lands in SOFT. The union must hold both
    hashes, be a plain set, and be a new object (not the HARD or SOFT set itself).
    """
    _set_legacy_active(mgr, "w6-hard")
    mgr._idle_thread_id = "w6-idle"
    mgr._idle_client_meta = None
    hard, soft = mgr._live_kv_protection()
    assert hard == {mgr._thread_hash("w6-hard")}, f"_live_kv_protection: the live thread must be HARD; got {hard}"
    assert soft == {mgr._thread_hash("w6-idle")}, f"_live_kv_protection: the idle thread must be SOFT; got {soft}"
    protected = mgr._live_protected_thread_hashes()
    assert protected == hard | soft, (
        f"_live_protected_thread_hashes must return HARD | SOFT; got {sorted(protected)}")
    assert type(protected) is set, f"_live_protected_thread_hashes must return a plain set; got {type(protected)}"
    assert protected is not hard and protected is not soft, (
        "_live_protected_thread_hashes must return a new set, not the HARD or SOFT set")


def test_w7_split_protection_failure_logs_once_and_returns_two_empty_plain_sets(mgr, monkeypatch, caplog):
    """Pins that the on_error handler passed to split_protection logs a failed split.

    The manager's thread hash raises on its first call. One live HARD thread sits on the legacy
    singleton, so the failure happens inside the split, not in the gather. The call must not raise, both tiers must be plain empty sets (the first hash raised before
    anything was added), and exactly one DEBUG record must carry the scan-failure message with the
    raised class as exc_info.
    """

    class _ThreadHashBoom(RuntimeError):
        pass

    def _boom(tid):
        raise _ThreadHashBoom("w7 thread hash boom")

    _set_legacy_active(mgr, "w7-hard")
    monkeypatch.setattr(mgr, "_thread_hash", _boom)
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")
    hard, soft = mgr._live_kv_protection()
    assert type(hard) is set and type(soft) is set, (
        f"_live_kv_protection: both tiers must be plain sets; got {type(hard)} and {type(soft)}")
    assert hard == set(), f"_live_kv_protection: the first hash raised, so HARD must be empty; got {sorted(hard)}"
    assert soft == set(), f"_live_kv_protection: the first hash raised, so SOFT must be empty; got {sorted(soft)}"
    records = [r for r in caplog.records
               if r.levelno == logging.DEBUG and "live KV protection-class scan failed" in r.getMessage()]
    assert len(records) == 1, (
        f"_live_kv_protection: a failed split must log exactly one DEBUG record; got {len(records)}")
    assert records[0].exc_info is not None and records[0].exc_info[0] is _ThreadHashBoom, (
        "_live_kv_protection: the scan-failure DEBUG record must carry the raised class as exc_info")
