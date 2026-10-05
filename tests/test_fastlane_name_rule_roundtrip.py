"""Round trip: a Fast Lane rule saved BY NAME, a census row for the address
that name resolves to, and the request matcher.

The name shown on a census row is display only. Whether a request matches a
name rule, and whether the census row reads `assigned`, is decided by the
name resolver's address set and nothing else. A resolver that answers with
the row's address makes the request match at the rule's rank and the row
`assigned`; an empty answer fails closed (unlisted, not assigned); and a
name set on a row for an address no rule covers changes neither.
"""
import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import compile_fastlane, match_fastlane, normalize_address
from turbohaul.manager import TurbohaulManager

CLIENT_IP = "192.0.2.10"
OTHER_IP = "192.0.2.77"
MAIN = {"is_main": True}


class _StubResolver:
    """Stands in for the forward resolver: a cache reader with a generation
    counter. `answers` maps a container name to the addresses it resolves to."""

    def __init__(self, answers):
        self.answers = answers
        self.generation = 1

    def addresses_for(self, name):
        return frozenset(normalize_address(a) for a in self.answers.get(name, ()))


def _make_manager(tmp_path, rules):
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
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=rules),
    )
    return TurbohaulManager(boot, runtime)


def _name_rule(name="app-one", rank=2):
    return FastLaneRule(container_name=name, label="by name", tag_ranks=FastLaneTagRanks(main=rank))


def _observe(mgr, ip):
    mgr._fastlane_census_observe(client_meta={"ip": ip}, thread_id="t", model_tag="m")


def _snapshot_row(mgr, ip):
    rows = [r for r in mgr.fastlane_census_snapshot()["rows"] if r["address"] == ip]
    assert len(rows) == 1
    return rows[0]


def _match(mgr, ip):
    return match_fastlane(mgr._fastlane_table(), ip, MAIN)


def test_a_by_name_rule_matches_the_resolved_address_at_its_rank(tmp_path):
    mgr = _make_manager(tmp_path, [_name_rule("app-one", rank=2)])
    mgr._fastlane_resolver = _StubResolver({"app-one": [CLIENT_IP]})
    _observe(mgr, CLIENT_IP)

    m = _match(mgr, CLIENT_IP)

    assert m is not None
    assert m.matched_by == "name"
    assert m.rule_index == 0
    assert m.rank == 2
    assert m.raw_address == "app-one"
    assert _snapshot_row(mgr, CLIENT_IP)["assigned"] is True


def test_the_roundtrip_with_the_name_set_on_the_row(tmp_path):
    # The shape the UI produces: the row shows a confirmed name, "Add" saves a
    # rule by that name, and the same row then reads assigned.
    mgr = _make_manager(tmp_path, [_name_rule("app-one", rank=2)])
    mgr._fastlane_resolver = _StubResolver({"app-one": [CLIENT_IP]})
    _observe(mgr, CLIENT_IP)
    mgr.fastlane_census_set_name(CLIENT_IP, "app-one")

    row = _snapshot_row(mgr, CLIENT_IP)
    assert row["container_name"] == "app-one"
    assert row["assigned"] is True
    m = _match(mgr, CLIENT_IP)
    assert m is not None and m.matched_by == "name" and m.rank == 2


def test_an_empty_resolution_fails_closed(tmp_path):
    mgr = _make_manager(tmp_path, [_name_rule("app-one")])
    mgr._fastlane_resolver = _StubResolver({"app-one": []})
    _observe(mgr, CLIENT_IP)

    assert _match(mgr, CLIENT_IP) is None
    assert _snapshot_row(mgr, CLIENT_IP)["assigned"] is False


def test_no_resolver_at_all_fails_closed(tmp_path):
    mgr = _make_manager(tmp_path, [_name_rule("app-one")])
    _observe(mgr, CLIENT_IP)

    assert _match(mgr, CLIENT_IP) is None
    assert _snapshot_row(mgr, CLIENT_IP)["assigned"] is False


def test_assigned_follows_the_resolver_when_the_address_changes(tmp_path):
    mgr = _make_manager(tmp_path, [_name_rule("app-one")])
    resolver = _StubResolver({"app-one": [CLIENT_IP]})
    mgr._fastlane_resolver = resolver
    _observe(mgr, CLIENT_IP)
    _observe(mgr, OTHER_IP)
    assert _snapshot_row(mgr, CLIENT_IP)["assigned"] is True
    assert _snapshot_row(mgr, OTHER_IP)["assigned"] is False

    # The container restarts and docker hands it a new address.
    resolver.answers = {"app-one": [OTHER_IP]}
    resolver.generation += 1

    assert _snapshot_row(mgr, CLIENT_IP)["assigned"] is False
    assert _snapshot_row(mgr, OTHER_IP)["assigned"] is True
    assert _match(mgr, CLIENT_IP) is None
    assert _match(mgr, OTHER_IP) is not None


def test_a_name_on_a_row_no_rule_covers_does_not_assign_or_match(tmp_path):
    # Rule covers CLIENT_IP only. The row for OTHER_IP shows the very same
    # name, but a name on a row is display, not identity.
    mgr = _make_manager(tmp_path, [_name_rule("app-one")])
    resolver = _StubResolver({"app-one": [CLIENT_IP]})
    mgr._fastlane_resolver = resolver
    _observe(mgr, OTHER_IP)
    mgr.fastlane_census_set_name(OTHER_IP, "app-one")
    # A resolver tick that moved nothing still recompiles the table once the
    # generation changes; recompile AFTER the name is set so the table is
    # built while the name is on the row.
    resolver.generation += 1

    assert _snapshot_row(mgr, OTHER_IP)["container_name"] == "app-one"
    assert _snapshot_row(mgr, OTHER_IP)["assigned"] is False
    assert _match(mgr, OTHER_IP) is None


def test_a_name_on_a_row_with_no_rules_at_all_does_not_match(tmp_path):
    mgr = _make_manager(tmp_path, [])
    _observe(mgr, OTHER_IP)
    mgr.fastlane_census_set_name(OTHER_IP, "app-one")

    assert _snapshot_row(mgr, OTHER_IP)["assigned"] is False
    assert _match(mgr, OTHER_IP) is None


def test_setting_or_clearing_a_name_leaves_the_compiled_table_alone(tmp_path):
    mgr = _make_manager(tmp_path, [_name_rule("app-one")])
    mgr._fastlane_resolver = _StubResolver({"app-one": [CLIENT_IP]})
    _observe(mgr, CLIENT_IP)
    table_before = mgr._fastlane_table()
    addresses_before = mgr._fastlane_rule_addresses()

    mgr.fastlane_census_set_name(CLIENT_IP, "worker-two")

    assert mgr._fastlane_table() is table_before, "a name must not recompile the table"
    assert mgr._fastlane_rule_addresses() == addresses_before
    # And the rule still matches its own resolved address, not the row's name.
    assert _match(mgr, CLIENT_IP).raw_address == "app-one"


def test_compile_with_a_stub_resolve_name_matches_the_same_way():
    # The same behaviour one layer down, without the manager: compile_fastlane
    # with a resolve_name stub, then match_fastlane on the resolved address.
    rules = [_name_rule("app-one", rank=3)]
    hit = compile_fastlane(rules, resolve_name=lambda n: frozenset([normalize_address(CLIENT_IP)]))
    miss = compile_fastlane(rules, resolve_name=lambda n: frozenset())

    m = match_fastlane(hit, CLIENT_IP, MAIN)
    assert m is not None and m.rank == 3 and m.matched_by == "name"
    assert match_fastlane(miss, CLIENT_IP, MAIN) is None
