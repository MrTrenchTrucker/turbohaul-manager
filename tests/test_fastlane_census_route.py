"""Contract test of GET /api/fastlane/census.

The route reshapes each snapshot row for the front end (`classes` becomes
`tag_classes_seen`, `models` becomes `models_seen`) and passes every other
field through untouched. The container name the census shows rides on that
pass-through, so these tests pin both halves: the reshaped fields stay
intact, and a name set on the manager reaches the response.
"""
import pytest
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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


def _observe(mgr, ip, model_tag="model-a"):
    mgr._fastlane_census_observe(
        client_meta={"ip": ip}, thread_id="t1", model_tag=model_tag
    )


def _entry(body, address):
    found = [e for e in body["entries"] if e["address"] == address]
    assert len(found) == 1, f"expected one entry for {address}, got {body['entries']}"
    return found[0]


def test_named_row_reaches_the_response_with_its_container_name(app_test):
    app, client = app_test
    mgr = app.state.manager
    _observe(mgr, "192.0.2.10")
    mgr.fastlane_census_set_name("192.0.2.10", "app-one")

    resp = client.get("/api/fastlane/census")

    assert resp.status_code == 200
    entry = _entry(resp.json(), "192.0.2.10")
    assert entry["container_name"] == "app-one"


def test_unnamed_row_reports_container_name_null(app_test):
    app, client = app_test
    _observe(app.state.manager, "192.0.2.11")

    entry = _entry(client.get("/api/fastlane/census").json(), "192.0.2.11")

    assert "container_name" in entry
    assert entry["container_name"] is None


def test_names_stay_with_their_own_rows(app_test):
    app, client = app_test
    mgr = app.state.manager
    _observe(mgr, "192.0.2.10")
    _observe(mgr, "192.0.2.11")
    mgr.fastlane_census_set_name("192.0.2.11", "worker-two")

    body = client.get("/api/fastlane/census").json()

    assert _entry(body, "192.0.2.10")["container_name"] is None
    assert _entry(body, "192.0.2.11")["container_name"] == "worker-two"


def test_a_cleared_name_reads_null_again(app_test):
    app, client = app_test
    mgr = app.state.manager
    _observe(mgr, "192.0.2.10")
    mgr.fastlane_census_set_name("192.0.2.10", "app-one")
    mgr.fastlane_census_set_name("192.0.2.10", None)

    entry = _entry(client.get("/api/fastlane/census").json(), "192.0.2.10")

    assert entry["container_name"] is None


def test_reshaped_fields_are_intact_next_to_the_name(app_test):
    app, client = app_test
    mgr = app.state.manager
    _observe(mgr, "192.0.2.10", model_tag="model-a")
    _observe(mgr, "192.0.2.10", model_tag="model-b")
    mgr.fastlane_census_set_name("192.0.2.10", "app-one")

    entry = _entry(client.get("/api/fastlane/census").json(), "192.0.2.10")

    assert entry["tag_classes_seen"] == ["unclassified"]
    assert isinstance(entry["tag_classes_seen"], list)
    assert entry["models_seen"] == ["model-a", "model-b"]
    assert isinstance(entry["models_seen"], list)
    # The raw source fields are replaced by the reshaped ones, not duplicated.
    assert "classes" not in entry
    assert "models" not in entry
    # Other fields pass through as they are in the snapshot.
    assert entry["request_count"] == 2
    assert entry["normalized_address"] == "192.0.2.10"
    assert entry["assigned"] is False


def test_unknown_extra_snapshot_keys_pass_through(app_test, monkeypatch):
    app, client = app_test
    mgr = app.state.manager
    _observe(mgr, "192.0.2.10")
    mgr.fastlane_census_set_name("192.0.2.10", "app-one")
    real = mgr.fastlane_census_snapshot

    def with_extra_key():
        snap = real()
        for row in snap["rows"]:
            row["future_field"] = {"x": 1}
        return snap

    monkeypatch.setattr(mgr, "fastlane_census_snapshot", with_extra_key)

    entry = _entry(client.get("/api/fastlane/census").json(), "192.0.2.10")

    assert entry["future_field"] == {"x": 1}
    assert entry["container_name"] == "app-one"


def test_empty_census_returns_no_entries(app_test):
    _, client = app_test
    body = client.get("/api/fastlane/census").json()
    assert body["backend_pending"] is False
    assert body["entries"] == []
