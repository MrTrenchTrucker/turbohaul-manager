"""Tests for GET /api/config/schema.

Additive, read-only endpoint: per-field {type, default, minimum, maximum}
for every runtime-mutable section.

Background: a hand-typed list of sections (queue + pull + kv) would drift from
config_put.py's RUNTIME_SECTIONS, which has grown to 7 sections
(fastlane/persist/monitor/http were added later), so the endpoint is
derived from RUNTIME_SECTIONS directly. This file's own top-level
section-set assertion below tracks whatever RuntimeConfig actually declares,
not a second hand-typed copy of it.
"""
import pytest
from fastapi.testclient import TestClient

from turbohaul.api.config_put import RUNTIME_SECTIONS
from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig,
    HttpConfig,
    KVConfig,
    MonitorConfig,
    PersistConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)

QUEUE_FIELDS = set(QueueConfig.model_fields)
PULL_FIELDS = set(PullConfig.model_fields)
KV_FIELDS = set(KVConfig.model_fields)
PERSIST_FIELDS = set(PersistConfig.model_fields)
MONITOR_FIELDS = set(MonitorConfig.model_fields)
HTTP_FIELDS = set(HttpConfig.model_fields)
ALL_FIELDS = QUEUE_FIELDS | PULL_FIELDS | KV_FIELDS | PERSIST_FIELDS | MONITOR_FIELDS | HTTP_FIELDS
BOOT_SECTIONS = {"server", "storage", "runtime", "ui"}

HF_HOST_ALLOWLIST_DEFAULT = [
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
]


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
        yield app, client


class TestConfigSchema:
    def test_200_all_declared_fields_present(self, app_test):
        """Contract: /api/config/schema exposes EXACTLY the declared model
        fields of each runtime section, for EVERY runtime-mutable section --
        not a hand-picked subset (that's exactly how this endpoint
        could drift from RUNTIME_SECTIONS). Section-set assertion is
        against RUNTIME_SECTIONS directly, so it can't drift that way
        either. fastlane's own field set is asserted in its own test module
        (test_api_config_fastlane.py), per the existing convention below.

        The per-section set-equality assertions are self-maintaining. The
        final bound is deliberately `>=` and is NOT updated when a field is
        added: an exact total is a shared counter that two concurrent PRs
        cannot both satisfy, and resolving that conflict by picking one side
        ships a red test. `>=` still catches the dangerous direction -- a
        section silently losing fields. Each PR asserts its OWN new fields in
        its OWN test module."""
        _, client = app_test
        r = client.get("/api/config/schema")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == RUNTIME_SECTIONS
        assert set(body["queue"].keys()) == QUEUE_FIELDS
        assert set(body["pull"].keys()) == PULL_FIELDS
        assert set(body["kv"].keys()) == KV_FIELDS
        assert set(body["persist"].keys()) == PERSIST_FIELDS
        assert set(body["monitor"].keys()) == MONITOR_FIELDS
        assert set(body["http"].keys()) == HTTP_FIELDS
        assert len(QUEUE_FIELDS) + len(PULL_FIELDS) + len(KV_FIELDS) >= 27

    def test_hf_host_allowlist_default_is_real_5_host_list(self, app_test):
        """Guards the default_factory trap: model_json_schema() reports this
        field's default as null, but the real default_factory value is the
        5-host list. A/B: sourcing 'default' from the schema block instead
        of Model().model_dump() makes this fail (default becomes None)."""
        _, client = app_test
        r = client.get("/api/config/schema")
        body = r.json()
        assert body["pull"]["hf_host_allowlist"]["default"] == HF_HOST_ALLOWLIST_DEFAULT

    def test_no_boot_section_in_payload(self, app_test):
        _, client = app_test
        r = client.get("/api/config/schema")
        body = r.json()
        assert not (set(body.keys()) & BOOT_SECTIONS)
        for section_fields in body.values():
            assert not (set(section_fields.keys()) & BOOT_SECTIONS)

    def test_bounds_present_where_field_declares_ge_le(self, app_test):
        _, client = app_test
        r = client.get("/api/config/schema")
        body = r.json()

        # grace_seconds: Field(default=30, ge=0, le=3600)
        grace = body["queue"]["grace_seconds"]
        assert grace["minimum"] == 0
        assert grace["maximum"] == 3600

        # pull_concurrency: Field(default=2, ge=1, le=16)
        concurrency = body["pull"]["pull_concurrency"]
        assert concurrency["minimum"] == 1
        assert concurrency["maximum"] == 16

        # acceptance_buffer_max: Field(default=10000, ge=1) - no le declared
        acceptance = body["queue"]["acceptance_buffer_max"]
        assert acceptance["minimum"] == 1
        assert acceptance["maximum"] is None

        # safety_enabled: bool, no ge/le
        safety_enabled = body["queue"]["safety_enabled"]
        assert safety_enabled["minimum"] is None
        assert safety_enabled["maximum"] is None


class TestRelayFanInHardening:
    """Hardening against future field declarations.

    Both guard the endpoint against FUTURE field declarations rather than
    fixing a live defect: no queue/pull field uses Field(exclude=True) or
    gt/lt today. They are here so that adding one cannot silently 500 the
    endpoint or silently drop a bound.
    """

    def test_schema_present_dump_absent_field_degrades_instead_of_500(self, app_test, monkeypatch):
        """A field in the JSON schema but NOT in the dump must be omitted, not KeyError.

        Field(exclude=True) produces exactly this shape. Driving the comprehension
        off model_json_schema() would raise KeyError -> 500 the whole endpoint and
        take every other field down with it.
        """
        real_schema = QueueConfig.model_json_schema

        def schema_with_phantom_field(*args, **kwargs):
            out = real_schema()
            out["properties"] = dict(out["properties"])
            out["properties"]["phantom_excluded_field"] = {"type": "integer", "minimum": 0}
            return out

        monkeypatch.setattr(QueueConfig, "model_json_schema", schema_with_phantom_field)

        _, client = app_test
        r = client.get("/api/config/schema")

        assert r.status_code == 200, "a schema-present/dump-absent field must not 500 the endpoint"
        body = r.json()
        assert "phantom_excluded_field" not in body["queue"], "excluded field must be omitted"
        # the rest of the section must survive intact
        assert body["queue"]["grace_seconds"]["default"] == 30

    def test_exclusive_bound_is_not_served_as_unbounded(self, app_test, monkeypatch):
        """gt/lt land as exclusiveMinimum/Maximum; they must not read as 'no bound'.

        Serving None here would silently drop the FE's client-side range check for
        that field.
        """
        real_schema = PullConfig.model_json_schema

        def schema_with_exclusive_bounds(*args, **kwargs):
            out = real_schema()
            out["properties"] = {k: dict(v) for k, v in out["properties"].items()}
            spec = out["properties"]["pull_concurrency"]
            spec.pop("minimum", None)
            spec.pop("maximum", None)
            spec["exclusiveMinimum"] = 0
            spec["exclusiveMaximum"] = 17
            return out

        monkeypatch.setattr(PullConfig, "model_json_schema", schema_with_exclusive_bounds)

        _, client = app_test
        body = client.get("/api/config/schema").json()

        assert body["pull"]["pull_concurrency"]["minimum"] == 0
        assert body["pull"]["pull_concurrency"]["maximum"] == 17
