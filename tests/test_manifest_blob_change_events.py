"""Manifest and blob mutations must announce themselves.

Background: pull.py has 12 event_bus publishers, import_.py 6,
live_monitor.py 2 -- and without the fix **api/manifests.py 0, api/config_put.py 0**. So a
manifest written through the API would be invisible to every other tab until
someone pressed Refresh. The per-model config flow only LOOKED two-way
because the editor re-GETs and re-seeds itself after its own save; no other
subscriber would ever be told.

These tests pin the six mutation sites that now publish, and -- more
importantly -- pin the REDACTION boundary on them. ws_state.py's rule: the state
channel "NEVER broadcasts: prompt text, response text, stderr lines, full
thread_ids, IPs." EventBus.REDACTED_KEYS strips prompt/response/context/
stderr/stdout/messages as defense-in-depth, but it does NOT know about
`description` -- an operator-written prose field sitting in scope at the
description route. Nothing but the publisher's own shape keeps it off the
wire, so that is asserted directly rather than assumed.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
from turbohaul.api.import_ import GGUF_MAGIC
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


SAMPLE_SHA = "abcdef01" + "0" * 56


@pytest.fixture
def app_test(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, storage_root


class Bus:
    """Subscribe a queue and drain it, matching tests/test_api_pull.py."""

    def __init__(self, app):
        self.q: asyncio.Queue = asyncio.Queue()
        app.state.manager.event_bus.subscribe(self.q)

    def drain(self) -> list[dict]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    def of(self, name: str) -> list[dict]:
        return [e for e in self.drain() if e.get("event") == name]


def _payload(tag="good-model", **over):
    p = {
        "model_tag": tag,
        "display_name": "Good Model",
        "description": "operator prose that must never reach the socket",
        "gguf_blob_sha256": SAMPLE_SHA,
        "gguf_size_bytes": 10_000_000_000,
        "context_size": 4096,
        "expected_vram_bytes": 11_000_000_000,
        "llama_server_flags": {"ctx_size": 4096, "n_gpu_layers": 999},
    }
    p.update(over)
    return p


def _seed(client, tag="good-model", **over):
    r = client.put(f"/api/manifests/{tag}", json=_payload(tag, **over))
    assert r.status_code == 200, r.text
    return r.json()["revision"], r.headers["ETag"]


def _import_blob(client, storage_root, body=b"payload") -> str:
    src = storage_root / "import-staging" / f"m-{body.hex()}.gguf"
    src.write_bytes(GGUF_MAGIC + body)
    r = client.post("/api/import", json={"path": str(src)})
    assert r.status_code == 200, r.text
    return r.json()["sha256"]


class TestManifestMutationsPublish:
    def test_put_publishes_manifest_changed(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)

        r = client.put("/api/manifests/good-model", json=_payload())

        assert r.status_code == 200
        events = bus.of("manifest_changed")
        assert len(events) == 1, f"expected exactly one event, got {events}"
        assert events[0]["model_tag"] == "good-model"
        assert events[0]["revision"] == r.json()["revision"]

    def test_patch_hidden_publishes_manifest_changed(self, app_test):
        app, client, _ = app_test
        _, etag = _seed(client)
        bus = Bus(app)

        r = client.patch(
            "/api/manifests/good-model", json={"hidden": True}, headers={"If-Match": etag}
        )

        assert r.status_code == 200, r.text
        events = bus.of("manifest_changed")
        assert len(events) == 1
        assert events[0]["model_tag"] == "good-model"
        assert events[0]["revision"] == r.json()["revision"]

    def test_restore_defaults_publishes_manifest_changed(self, app_test):
        app, client, _ = app_test
        _, etag = _seed(client)
        bus = Bus(app)

        r = client.post(
            "/api/manifests/good-model/restore-defaults", headers={"If-Match": etag}
        )

        assert r.status_code == 200, r.text
        events = bus.of("manifest_changed")
        assert len(events) == 1
        assert events[0]["model_tag"] == "good-model"

    def test_delete_publishes_manifest_changed_without_a_revision(self, app_test):
        app, client, _ = app_test
        _seed(client)
        bus = Bus(app)

        r = client.delete("/api/manifests/good-model")

        assert r.status_code == 200
        events = bus.of("manifest_changed")
        assert len(events) == 1
        assert events[0]["model_tag"] == "good-model"
        # The manifest is gone; there is no revision to name, and inventing one
        # would be a lie about a file that no longer exists.
        assert "revision" not in events[0]


class TestBlobMutationsPublish:
    def test_blob_delete_publishes_blob_changed(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        bus = Bus(app)

        r = client.request("DELETE", "/api/delete", json={"sha256": sha})

        assert r.status_code == 200, r.text
        events = bus.of("blob_changed")
        assert len(events) == 1
        assert events[0]["sha256"] == sha

    def test_description_put_publishes_blob_changed(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        bus = Bus(app)

        r = client.put(f"/api/blobs/{sha}/description", json={"description": "a note"})

        assert r.status_code == 200, r.text
        events = bus.of("blob_changed")
        assert len(events) == 1
        assert events[0]["sha256"] == sha


class TestRedactionBoundary:
    """ws_state.py's no-broadcast rule. The rule this file exists to keep honest."""

    def test_the_description_never_reaches_the_socket(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        bus = Bus(app)
        secret = "PRIVATE-OPERATOR-PROSE-marker"

        r = client.put(f"/api/blobs/{sha}/description", json={"description": secret})

        assert r.status_code == 200
        # The route echoes it back to the CALLER, which is correct...
        assert r.json()["description"] == secret
        # ...and it must appear nowhere on the broadcast channel. EventBus's
        # own REDACTED_KEYS denylist does NOT cover `description`, so the
        # publisher's shape is the only thing standing here.
        for event in bus.drain():
            assert secret not in repr(event), f"description leaked onto the bus: {event}"

    def test_manifest_events_carry_identifiers_and_nothing_else(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)

        client.put("/api/manifests/good-model", json=_payload())

        for event in bus.of("manifest_changed"):
            assert set(event) <= {"event", "model_tag", "revision"}, (
                f"a manifest event grew a field beyond its identifiers: {event}"
            )

    def test_no_manifest_body_reaches_the_socket(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)
        prose = "operator prose that must never reach the socket"

        client.put("/api/manifests/good-model", json=_payload())

        drained = repr(bus.drain())
        assert prose not in drained, "the manifest description leaked onto the bus"
        assert "Good Model" not in drained, "the display_name leaked onto the bus"
        assert "n_gpu_layers" not in drained, "llama_server_flags leaked onto the bus"
        assert SAMPLE_SHA not in drained, "the gguf digest leaked out of a manifest event"


class TestFailedMutationsPublishNothing:
    """An announcement is a claim that something changed. A rejected write
    changed nothing, so it must say nothing -- otherwise every subscriber
    re-fetches the whole registry on each failed attempt."""

    def test_a_412_concurrency_rejection_publishes_nothing(self, app_test):
        app, client, _ = app_test
        _seed(client)
        bus = Bus(app)

        r = client.patch(
            "/api/manifests/good-model",
            json={"hidden": True},
            headers={"If-Match": '"999"'},
        )

        assert r.status_code == 412
        assert bus.of("manifest_changed") == []

    def test_deleting_an_unknown_manifest_publishes_nothing(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)

        r = client.delete("/api/manifests/never-existed")

        assert r.status_code == 404
        assert bus.of("manifest_changed") == []

    def test_deleting_an_unknown_blob_publishes_nothing(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)

        r = client.request("DELETE", "/api/delete", json={"sha256": "f" * 64})

        assert r.status_code == 404
        assert bus.of("blob_changed") == []

    def test_a_rejected_description_publishes_nothing(self, app_test):
        app, client, _ = app_test
        bus = Bus(app)

        r = client.put(f"/api/blobs/{'f' * 64}/description", json={"description": "x"})

        assert r.status_code == 404
        assert bus.of("blob_changed") == []


class TestImportStillPublishesExactlyOnce:
    """POST /api/import already publishes an event. Pinned so a
    second publisher is never added on top of it."""

    def test_import_publishes_its_own_events_and_no_blob_changed(self, app_test):
        app, client, storage = app_test
        bus = Bus(app)

        _import_blob(client, storage)

        drained = bus.drain()
        names = [e.get("event") for e in drained]
        assert "import_started" in names
        assert "import_complete" in names
        assert "blob_changed" not in names, (
            "import must not have gained a second announcement on top of its own"
        )
