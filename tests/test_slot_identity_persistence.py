"""A slot row keeps the identity of its request, not the request body.

A slot row stores only the identity keys of its request, never the whole body (messages, tools, sampling
options). That keeps the state database small and the boot census cheap. The running request keeps its full
client metadata in memory.
"""
import asyncio
import json

import pytest

import turbohaul.manager as manager_mod
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot
from turbohaul.state import open_state_db, state_db_session

IDENTITY = {"ip", "session_id", "role", "is_main", "is_sub_agent", "is_curator", "is_compression", "save_kv"}


def _make_mgr(tmp_path):
    root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite"),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59700),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    m = TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(max_parallel_sidecars=1, safety_enabled=False),
                                             pull=PullConfig()))
    with state_db_session(m.boot.storage.state_db_path):
        pass
    return m


@pytest.fixture
def mgr(tmp_path):
    return _make_mgr(tmp_path)


def _full_meta(ip="192.0.2.9", **extra):
    return {
        "kind": "openai-chat-completion",
        "messages": [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "hello"}],
        "model": "m", "stream": False, "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "f"}}], "options": {"num_ctx": 4096},
        "req_id": "r-1", "context_size": 12, "turn0_meta": {"role": "user", "content_len": 5},
        "ip": ip, "session_id": "sess-1", "role": "main", "is_main": True, "is_sub_agent": False,
        "is_curator": None, "is_compression": False, "save_kv": True,
        **extra,
    }


def _stored(mgr, slot_id):
    conn = open_state_db(mgr.boot.storage.state_db_path)
    try:
        return json.loads(conn.execute(
            "SELECT client_meta_json FROM slots WHERE slot_id=?", (slot_id,)).fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def written(monkeypatch):
    """Every client_meta handed to upsert_slot by the manager, in order."""
    calls = []
    real = manager_mod.upsert_slot

    def spy(conn, slot):
        calls.append(slot.get("client_meta"))
        return real(conn, slot)

    monkeypatch.setattr(manager_mod, "upsert_slot", spy)
    return calls


@pytest.mark.asyncio
async def test_a_submitted_request_is_stored_as_its_identity_only(mgr, written):
    meta = _full_meta()
    slot = await mgr.submit(model_tag="m", prompt="hi", thread_id="t-1", client_meta=meta)
    stored = _stored(mgr, slot.slot_id)
    assert set(stored) == IDENTITY, f"the slot row must keep exactly the identity keys, got {sorted(stored)}"
    assert stored == {k: meta[k] for k in IDENTITY}
    assert written and all(set(c) <= IDENTITY for c in written), (
        "every write of a slot row from the manager must carry identity keys only")


@pytest.mark.asyncio
async def test_the_running_request_keeps_its_full_client_metadata_in_memory(mgr):
    meta = _full_meta()
    slot = await mgr.submit(model_tag="m", prompt="hi", thread_id="t-1", client_meta=meta)
    assert slot.client_meta["messages"] == meta["messages"]
    assert slot.client_meta["tools"] == meta["tools"]


def test_a_state_change_write_is_identity_only_too(mgr, written):
    slot = Slot.new("m", thread_id="t-2", admission_ctx_len=10, client_meta=_full_meta(ip="192.0.2.10"))
    mgr._audit(slot, "active")
    assert len(written) == 1
    assert set(written[0]) <= IDENTITY, f"the state-change write carried {sorted(set(written[0]) - IDENTITY)}"
    stored = _stored(mgr, slot.slot_id)
    assert set(stored) == IDENTITY and stored["ip"] == "192.0.2.10"
    assert slot.client_meta["messages"], "the in-memory metadata must be untouched by the write"


def test_a_request_with_no_identity_keys_is_stored_as_an_empty_object(mgr):
    slot = Slot.new("m", thread_id="t-3", admission_ctx_len=10, client_meta={"messages": ["x"], "model": "m"})
    mgr._audit(slot, "active")
    assert _stored(mgr, slot.slot_id) == {}


def test_a_request_with_no_client_metadata_at_all_is_stored_as_an_empty_object(mgr):
    slot = Slot.new("m", thread_id="t-4", admission_ctx_len=10, client_meta=None)
    mgr._audit(slot, "active")
    assert _stored(mgr, slot.slot_id) == {}


def test_the_census_built_from_identity_only_rows_is_complete(tmp_path):
    first = _make_mgr(tmp_path)
    cases = [
        ("192.0.2.1", {"is_main": True}, "main"),
        ("192.0.2.2", {"is_curator": True, "is_sub_agent": True}, "curator"),
        ("192.0.2.3", {"is_compression": True}, "compression"),
        ("192.0.2.4", {"is_sub_agent": True}, "sub-agent"),
        ("192.0.2.5", {"role": "user-message"}, "user-message"),
        ("192.0.2.6", {}, "unclassified"),
    ]

    async def submit_all():
        for i, (ip, labels, _expected) in enumerate(cases):
            meta = _full_meta(ip=ip)
            for key in ("role", "is_main", "is_sub_agent", "is_curator", "is_compression"):
                meta[key] = None
            meta.update(labels)
            await first.submit(model_tag="m", prompt="p", thread_id=f"t{i}", client_meta=meta)

    asyncio.run(submit_all())
    second = TurbohaulManager(first.boot, first.runtime)
    second.boot_reconcile()
    got = {ip: row["classes"] for ip, row in second._fastlane_census.items()}
    assert got == {ip: {expected: 1} for ip, _labels, expected in cases}
