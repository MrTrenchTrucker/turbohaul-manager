"""API round trip for a manifest that still carries the retired `max_instances` key.

The per-model `max_instances` setting is gone. A stored file, a PUT body or a
saved browser tab that still carries it must keep working: reads succeed and show
no such key, writes succeed (never a 400 or 422) and store no such key, and the
revision/ETag behaviour is the normal one.
"""
from __future__ import annotations

import logging

import pytest
import yaml
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
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
from turbohaul.manifest import migrate_retired_keys

SAMPLE_SHA = "abcdef01" + "0" * 56
MANIFEST_LOGGER = "turbohaul.manifest"


@pytest.fixture
def app_blank(tmp_path):
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
        yield app, client, storage_root / "manifests"


def _payload(tag="my-model", **over):
    body = {
        "model_tag": tag,
        "display_name": "Test Model",
        "gguf_blob_sha256": SAMPLE_SHA,
        "gguf_size_bytes": 10_000_000_000,
        "context_size": 4096,
        "expected_vram_bytes": 11_000_000_000,
        "llama_server_flags": {"ctx_size": 4096, "n_gpu_layers": 999},
    }
    body.update(over)
    return body


def _write_old_style(manifests, tag="my-model", revision=1, max_instances=2):
    body = _payload(tag, revision=revision, max_instances=max_instances)
    (manifests / f"{tag}.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
    return body


def _contains_key(node, key):
    """True when `key` appears as a mapping key anywhere in a JSON-like tree."""
    if isinstance(node, dict):
        return key in node or any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(v, key) for v in node)
    return False


def _stored(manifests, tag="my-model"):
    return yaml.safe_load((manifests / f"{tag}.yaml").read_text())


def _retired_warnings(caplog):
    return [
        r for r in caplog.records
        if r.name == MANIFEST_LOGGER and r.levelno == logging.WARNING
        and "max_instances" in r.getMessage()
    ]


def test_get_of_an_old_style_file_succeeds_and_carries_no_key(app_blank):
    _, client, manifests = app_blank
    _write_old_style(manifests, revision=3)
    r = client.get("/api/manifests/my-model")
    assert r.status_code == 200
    assert r.headers["ETag"] == '"3"'
    assert not _contains_key(r.json(), "max_instances")
    assert r.json()["model_tag"] == "my-model"


def test_put_of_the_get_body_plus_the_key_is_accepted_and_stores_no_key(app_blank):
    _, client, manifests = app_blank
    _write_old_style(manifests, revision=3)
    got = client.get("/api/manifests/my-model")
    etag = got.headers["ETag"]
    body = got.json()
    body.pop("cache_reuse_inert_mmproj", None)   # derived, read-only marker
    body["max_instances"] = 1
    r = client.put("/api/manifests/my-model", json=body, headers={"If-Match": etag})
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 4
    assert not _contains_key(r.json(), "max_instances")
    stored = _stored(manifests)
    assert "max_instances" not in stored
    assert stored["revision"] == 4                  # exactly one more than before


def test_put_of_a_brand_new_tag_with_the_key_is_accepted_and_stores_no_key(app_blank):
    _, client, manifests = app_blank
    r = client.put("/api/manifests/fresh-one", json=_payload("fresh-one", max_instances=4))
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 1
    assert not _contains_key(r.json(), "max_instances")
    assert "max_instances" not in _stored(manifests, "fresh-one")


def test_put_with_an_old_range_breaking_value_is_still_accepted(app_blank):
    _, client, manifests = app_blank
    for tag, value in (("big-one", 99), ("zero-one", 0)):
        r = client.put(f"/api/manifests/{tag}", json=_payload(tag, max_instances=value))
        assert r.status_code == 200, (value, r.text)
        assert "max_instances" not in _stored(manifests, tag)


def test_put_with_another_unknown_key_is_still_rejected(app_blank):
    _, client, manifests = app_blank
    r = client.put(
        "/api/manifests/strict-one",
        json=_payload("strict-one", max_instances=2, not_a_real_key=1),
    )
    assert r.status_code == 400
    assert "not_a_real_key" in r.text
    assert not (manifests / "strict-one.yaml").exists()


def test_stale_tab_survives_the_sweep(app_blank):
    """A tab loaded an old-style manifest; the store is swept; the tab saves."""
    _, client, manifests = app_blank
    _write_old_style(manifests, revision=2)
    got = client.get("/api/manifests/my-model")
    etag = got.headers["ETag"]
    body = got.json()
    body.pop("cache_reuse_inert_mmproj", None)
    body["max_instances"] = 2                     # the stale tab still sends it

    assert migrate_retired_keys(manifests) == ["my-model"]
    assert "max_instances" not in _stored(manifests)

    r = client.put("/api/manifests/my-model", json=body, headers={"If-Match": etag})
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 3
    assert "max_instances" not in _stored(manifests)


def test_list_route_carries_no_key(app_blank):
    _, client, manifests = app_blank
    _write_old_style(manifests, "old-one", revision=1)
    client.put("/api/manifests/new-one", json=_payload("new-one"))
    r = client.get("/api/manifests")
    assert r.status_code == 200
    body = r.json()
    rows = {m["model_tag"]: m for m in body["manifests"]}
    assert set(rows) == {"old-one", "new-one"}
    # The old-style file is a normal, readable row (not an "unreadable" stub).
    assert rows["old-one"].get("error") is None
    assert rows["old-one"]["revision"] == 1
    assert rows["old-one"]["etag"] == '"1"'
    assert not _contains_key(body, "max_instances")
    assert "max_instances" not in r.text


def test_restore_defaults_route_loads_old_style_file_and_carries_no_key(app_blank):
    _, client, manifests = app_blank
    _write_old_style(manifests, revision=1)
    r = client.post("/api/manifests/my-model/restore-defaults", headers={"If-Match": '"1"'})
    assert r.status_code == 200, r.text
    assert r.json()["revision"] == 2
    assert not _contains_key(r.json(), "max_instances")
    assert "max_instances" not in _stored(manifests)


def test_patch_route_on_an_old_style_file_works_and_stores_no_key(app_blank):
    _, client, manifests = app_blank
    _write_old_style(manifests, revision=1)
    r = client.patch("/api/manifests/my-model", json={"hidden": True},
                     headers={"If-Match": '"1"'})
    assert r.status_code == 200, r.text
    assert not _contains_key(r.json(), "max_instances")
    assert "max_instances" not in _stored(manifests)


def test_put_carrying_the_key_logs_exactly_one_warning(app_blank, caplog):
    _, client, _ = app_blank
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    r = client.put("/api/manifests/warn-one", json=_payload("warn-one", max_instances=7))
    assert r.status_code == 200
    warnings = _retired_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "manifest 'warn-one': dropped retired key(s) max_instances (no longer supported)"
    )


def test_put_without_the_key_logs_no_retired_key_warning(app_blank, caplog):
    _, client, _ = app_blank
    caplog.set_level(logging.DEBUG, logger=MANIFEST_LOGGER)
    r = client.put("/api/manifests/quiet-one", json=_payload("quiet-one"))
    assert r.status_code == 200
    assert _retired_warnings(caplog) == []
    # Stricter than the key-name filter above: nothing at WARNING or above may
    # come from the manifest logger at all for a key-free body.
    assert [r.getMessage() for r in caplog.records
            if r.name == MANIFEST_LOGGER and r.levelno >= logging.WARNING] == []
