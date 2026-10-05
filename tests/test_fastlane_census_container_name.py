"""The Fast Lane census row's `container_name` field and the two manager
methods that feed it.

The name is display only. It is set by a background task after a forward
lookup confirmed it, read back through fastlane_census_snapshot(), and never
touched by the request path. These tests pin: the key exists (as None) on a
row from every creation site, the snapshot carries it, set/clear round-trips,
a row that is gone is never recreated, and the target list is capped.
"""
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
from turbohaul.manager import TurbohaulManager, _FASTLANE_CENSUS_MAX
from turbohaul.state import open_state_db, upsert_slot


@pytest.fixture
def mgr(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _observe(mgr, ip):
    mgr._fastlane_census_observe(client_meta={"ip": ip}, thread_id="t", model_tag="m")


def _row(mgr, ip):
    rows = [r for r in mgr.fastlane_census_snapshot()["rows"] if r["address"] == ip]
    assert len(rows) == 1, f"expected exactly one snapshot row for {ip}, got {rows}"
    return rows[0]


class TestRowCreationSites:
    def test_live_observe_creates_the_row_with_container_name_none(self, mgr):
        _observe(mgr, "192.0.2.10")
        stored = mgr._fastlane_census["192.0.2.10"]
        assert "container_name" in stored, "observe path must define the key"
        assert stored["container_name"] is None

    def test_boot_rebuild_creates_the_row_with_container_name_none(self, mgr):
        conn = open_state_db(mgr.boot.storage.state_db_path)
        upsert_slot(conn, {
            "slot_id": "s1", "model_tag": "m", "state": "COLD",
            "thread_id": "t1", "client_meta": {"ip": "192.0.2.20"},
        })
        conn.close()

        mgr.boot_reconcile()

        stored = mgr._fastlane_census["192.0.2.20"]
        assert "container_name" in stored, "boot rebuild must define the key"
        assert stored["container_name"] is None


class TestSnapshotRow:
    def test_snapshot_row_carries_container_name_none_by_default(self, mgr):
        _observe(mgr, "192.0.2.10")
        row = _row(mgr, "192.0.2.10")
        assert "container_name" in row
        assert row["container_name"] is None

    def test_snapshot_row_carries_the_name_once_set(self, mgr):
        _observe(mgr, "192.0.2.10")
        mgr.fastlane_census_set_name("192.0.2.10", "app-one")
        assert _row(mgr, "192.0.2.10")["container_name"] == "app-one"

    def test_a_name_on_one_row_does_not_leak_to_another(self, mgr):
        _observe(mgr, "192.0.2.10")
        _observe(mgr, "192.0.2.11")
        mgr.fastlane_census_set_name("192.0.2.10", "app-one")
        assert _row(mgr, "192.0.2.10")["container_name"] == "app-one"
        assert _row(mgr, "192.0.2.11")["container_name"] is None

    def test_snapshot_row_built_from_a_dict_without_the_key_reads_none(self, mgr):
        # A census dict written by hand (older fixtures, or a row from before
        # the key existed) has no container_name; the snapshot must not raise.
        _observe(mgr, "192.0.2.10")
        del mgr._fastlane_census["192.0.2.10"]["container_name"]
        row = _row(mgr, "192.0.2.10")
        assert row["container_name"] is None
        assert row["request_count"] == 1  # the rest of the row is intact


class TestSetName:
    def test_set_then_clear_with_none(self, mgr):
        _observe(mgr, "192.0.2.10")
        mgr.fastlane_census_set_name("192.0.2.10", "app-one")
        assert mgr._fastlane_census["192.0.2.10"]["container_name"] == "app-one"
        mgr.fastlane_census_set_name("192.0.2.10", None)
        assert mgr._fastlane_census["192.0.2.10"]["container_name"] is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_name_counts_as_none(self, mgr, blank):
        _observe(mgr, "192.0.2.10")
        mgr.fastlane_census_set_name("192.0.2.10", "app-one")
        mgr.fastlane_census_set_name("192.0.2.10", blank)
        assert mgr._fastlane_census["192.0.2.10"]["container_name"] is None

    def test_a_name_with_surrounding_whitespace_is_stored_trimmed(self, mgr):
        _observe(mgr, "192.0.2.10")
        mgr.fastlane_census_set_name("192.0.2.10", "  app-one \n")
        assert mgr._fastlane_census["192.0.2.10"]["container_name"] == "app-one"

    def test_unknown_key_is_a_no_op_and_creates_no_row(self, mgr):
        _observe(mgr, "192.0.2.10")
        before = dict(mgr._fastlane_census)
        mgr.fastlane_census_set_name("192.0.2.99", "ghost")
        assert "192.0.2.99" not in mgr._fastlane_census
        assert set(mgr._fastlane_census) == set(before)
        assert mgr.fastlane_census_snapshot()["rows"][0]["container_name"] is None

    def test_unknown_key_on_an_empty_census_creates_nothing(self, mgr):
        mgr.fastlane_census_set_name("192.0.2.99", "ghost")
        assert mgr._fastlane_census == {}

    def test_set_name_does_not_change_the_counters_or_timestamps(self, mgr):
        _observe(mgr, "192.0.2.10")
        before = dict(_row(mgr, "192.0.2.10"))
        mgr.fastlane_census_set_name("192.0.2.10", "app-one")
        after = dict(_row(mgr, "192.0.2.10"))
        before.pop("container_name")
        after.pop("container_name")
        assert after == before


class TestNameTargets:
    def test_empty_census_has_no_targets(self, mgr):
        assert mgr.fastlane_census_name_targets() == []

    def test_targets_are_key_and_current_name_pairs(self, mgr):
        _observe(mgr, "192.0.2.10")
        _observe(mgr, "192.0.2.11")
        mgr.fastlane_census_set_name("192.0.2.11", "worker-two")
        targets = mgr.fastlane_census_name_targets()
        assert sorted(targets, key=lambda t: t[0]) == [
            ("192.0.2.10", None),
            ("192.0.2.11", "worker-two"),
        ]

    def test_the_key_is_the_normalized_census_key_not_the_raw_form(self, mgr):
        _observe(mgr, "::ffff:192.0.2.10")
        (key, name), = mgr.fastlane_census_name_targets()
        assert key in mgr._fastlane_census
        assert key == "192.0.2.10"
        assert name is None

    def test_a_returned_key_can_be_fed_straight_back_to_set_name(self, mgr):
        _observe(mgr, "::ffff:192.0.2.10")
        (key, _), = mgr.fastlane_census_name_targets()
        mgr.fastlane_census_set_name(key, "app-one")
        assert mgr.fastlane_census_snapshot()["rows"][0]["container_name"] == "app-one"

    def test_targets_are_capped_at_the_census_maximum(self, mgr):
        # Fill the census past the cap by hand (observe refuses past the cap,
        # so this is the only way to hold more rows than the cap).
        _observe(mgr, "192.0.2.1")
        template = mgr._fastlane_census["192.0.2.1"]
        for i in range(_FASTLANE_CENSUS_MAX + 20):
            key = f"fd00::{i + 1:x}"
            mgr._fastlane_census[key] = dict(template, address=key, normalized_address=key)
        assert len(mgr._fastlane_census) > _FASTLANE_CENSUS_MAX
        targets = mgr.fastlane_census_name_targets()
        assert len(targets) == _FASTLANE_CENSUS_MAX
        assert len({k for k, _ in targets}) == _FASTLANE_CENSUS_MAX

    def test_a_census_at_the_cap_returns_every_row(self, mgr):
        _observe(mgr, "192.0.2.1")
        template = mgr._fastlane_census["192.0.2.1"]
        for i in range(_FASTLANE_CENSUS_MAX - 1):
            key = f"fd00::{i + 1:x}"
            mgr._fastlane_census[key] = dict(template, address=key, normalized_address=key)
        assert len(mgr._fastlane_census) == _FASTLANE_CENSUS_MAX
        assert len(mgr.fastlane_census_name_targets()) == _FASTLANE_CENSUS_MAX

    def test_a_row_without_the_key_reads_as_none_in_targets(self, mgr):
        _observe(mgr, "192.0.2.10")
        del mgr._fastlane_census["192.0.2.10"]["container_name"]
        assert mgr.fastlane_census_name_targets() == [("192.0.2.10", None)]
