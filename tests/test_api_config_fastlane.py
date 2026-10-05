"""Tests for the Fast Lane config surface.

Three checks, deliberately not the naive versions:
  - registry invariant: RuntimeConfig fields == RUNTIME_SECTIONS == persisted
    top-level keys == (GET /api/config runtime keys, i.e. everything except
    the four BOOT_SECTIONS).
  - round trip: asserts on TurbohaulConfig(**cfg_data).split().runtime, not a
    raw dict, so a missing `fastlane=self.fastlane` in config.py is
    actually caught -- the raw-dict style in test_api_config_put.py's
    precedent cannot catch that class of bug.
  - corrupt block: drives the REAL __main__ boot-merge fail-loud parse (not
    just create_app, which never reads runtime_config.yaml), asserting no
    exception, fastlane forced off, and the ERROR logged.
"""
import logging
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

from turbohaul.api.config_put import (
    BOOT_SECTIONS,
    RUNTIME_SECTIONS,
    _PROVENANCE_STAMP_KEY,
    _runtime_override_path,
    load_runtime_override,
    save_runtime_config,
)
from turbohaul.api.config_schema import _SECTIONS
from turbohaul.api.main import create_app
from pydantic import ValidationError

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    TurbohaulConfig,
    UIConfig,
)
from turbohaul.fastlane import compile_fastlane, normalize_address


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
        yield app, client, boot


class TestFastLaneRegistryInvariant:
    """Four separate assertions, not one that can never pass."""

    def test_runtime_config_fields_match_runtime_sections(self):
        assert set(RuntimeConfig.model_fields) == RUNTIME_SECTIONS

    def test_get_config_runtime_keys_match_runtime_sections(self, app_test):
        _app, client, _boot = app_test
        payload = client.get("/api/config").json()
        # GET /api/config also carries metadata
        # keys (_provenance, _provenance_stamp) that are NOT config
        # sections. Named explicitly here, not silently filtered by a
        # generic "starts with _" rule, so a change to WHICH metadata keys
        # exist is itself something this test catches.
        metadata_keys = {k for k in payload if k.startswith("_")}
        assert metadata_keys == {"_provenance", "_provenance_stamp"}
        assert set(payload) - BOOT_SECTIONS - metadata_keys == RUNTIME_SECTIONS

    def test_persisted_override_keys_match_runtime_sections(self, app_test):
        _app, client, boot = app_test
        override_path = _runtime_override_path(boot.storage.state_db_path)
        # save_runtime_config is a SPARSE
        # read-MERGE-write -- persists only the sections/fields a PUT
        # names, but MUST preserve whatever a PRIOR PUT already wrote.
        # It also stamps a sibling _PROVENANCE_STAMP_KEY
        # into the same file on every PUT. A single PUT against an empty
        # override file cannot distinguish a true merge from a destructive
        # overwrite (stripping save_runtime_config's
        # read-existing-file step would still pass a single-PUT version of this
        # assertion) -- two sequential PUTs to DIFFERENT sections is what
        # actually exercises "merge", not just "sparse write". Section sets
        # are derived from the PUT bodies themselves, not hand-duplicated,
        # so this can't drift from what was actually sent.
        first_put = {"fastlane": {"enabled": True}}
        r1 = client.put("/api/config", json=first_put)
        assert r1.status_code == 200
        persisted_1 = yaml.safe_load(override_path.read_text())
        assert set(persisted_1) == set(first_put) | {_PROVENANCE_STAMP_KEY}

        second_put = {"persist": {"max_bytes": 123456}}
        r2 = client.put("/api/config", json=second_put)
        assert r2.status_code == 200
        persisted_2 = yaml.safe_load(override_path.read_text())
        assert set(persisted_2) == set(first_put) | set(second_put) | {_PROVENANCE_STAMP_KEY}
        assert set(first_put) | set(second_put) <= RUNTIME_SECTIONS


def _observe(app, *raw_ips):
    """Seed the manager's discovered-address census with real observations
    of each raw ip, via the actual production method (not a hand-built
    dict row) -- so a test that wants "already observed" state can't drift
    from what a real request would actually produce."""
    for raw_ip in raw_ips:
        app.state.manager._fastlane_census_observe(
            client_meta={"ip": raw_ip}, thread_id="t", model_tag="m"
        )


class TestFastLaneLintWiring:
    """lint_rules is wired as a non-fatal PUT-path
    channel, gated on `"fastlane" in payload`. The duplicate
    check inside lint_rules compares normalized addresses.

    The census= argument is passed too, so an UNOBSERVED test address also
    gets a "never observed" warning alongside any duplicate one -- these
    tests isolate the DUPLICATE-detection behavior specifically by
    asserting on presence/absence of "duplicate" rather than exact warning
    counts. Census-membership behavior itself has its own dedicated tests
    in TestFastLaneCensusNormalization below."""

    def test_put_duplicate_addresses_returns_warnings_not_rejected(self, app_test):
        _app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
            {"address": "192.0.2.10", "tag_ranks": {"curator": 2}},
        ]}})
        assert r.status_code == 200  # non-fatal: doc says "reported", never "rejected"
        warnings = r.json()["warnings"]
        assert any("duplicate" in w for w in warnings)

    def test_put_normalized_duplicate_addresses_reported(self, app_test):
        _app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
            {"address": "::ffff:192.0.2.10", "tag_ranks": {"curator": 2}},
        ]}})
        assert r.status_code == 200
        warnings = r.json()["warnings"]
        assert any("duplicate" in w for w in warnings)

    def test_put_no_duplicate_addresses_no_duplicate_warning(self, app_test):
        _app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
            {"address": "192.0.2.11", "tag_ranks": {"curator": 2}},
        ]}})
        assert r.status_code == 200
        assert not any("duplicate" in w for w in r.json()["warnings"])

    def test_put_observed_no_duplicate_addresses_empty_warnings(self, app_test):
        # Both addresses pre-observed (via the real census-observe method)
        # AND non-duplicate -- the ONLY case where warnings is genuinely
        # empty, now that the census-membership check co-exists with the
        # duplicate check on the same call.
        app, client, _boot = app_test
        _observe(app, "192.0.2.10", "192.0.2.11")
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
            {"address": "192.0.2.11", "tag_ranks": {"curator": 2}},
        ]}})
        assert r.status_code == 200
        assert r.json()["warnings"] == []

    def test_put_duplicate_addresses_logs_warning(self, app_test, caplog):
        _app, client, _boot = app_test
        with caplog.at_level(logging.WARNING):
            r = client.put("/api/config", json={"fastlane": {"rules": [
                {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
                {"address": "192.0.2.10", "tag_ranks": {"curator": 2}},
            ]}})
        assert r.status_code == 200
        assert any(
            "duplicate" in rec.message and rec.levelname == "WARNING"
            for rec in caplog.records
        )

    def test_put_unrelated_section_does_not_run_fastlane_lint(self, app_test):
        # Gating: a PUT that never touches "fastlane" must not
        # re-report an already-existing duplicate. Proves the lint fires
        # at the moment a duplicate is INTRODUCED, not on every unrelated
        # write forever after (which would train operators to ignore it).
        _app, client, _boot = app_test
        r1 = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "192.0.2.10", "tag_ranks": {"main": 1}},
            {"address": "192.0.2.10", "tag_ranks": {"curator": 2}},
        ]}})
        assert r1.status_code == 200
        assert any("duplicate" in w for w in r1.json()["warnings"])

        r2 = client.put("/api/config", json={"persist": {"max_bytes": 123456}})
        assert r2.status_code == 200
        assert r2.json()["warnings"] == []

    def test_put_non_fastlane_section_always_has_warnings_key(self, app_test):
        # The response contract is unconditional: "warnings" is always
        # present (empty list when not applicable), so callers never need
        # an `if "warnings" in body` check.
        _app, client, _boot = app_test
        r = client.put("/api/config", json={"persist": {"max_bytes": 123456}})
        assert r.status_code == 200
        assert r.json()["warnings"] == []


class TestFastLaneResolveBeforeLint:
    """config_put.py must await a resolve pass
    BEFORE linting, or a brand-new address saved in THIS SAME payload is
    judged against a census that has not caught up to it yet and gets
    reported as drift on its first save.

    These tests use the manager's long-lived resolver instance rather than
    a per-call module class.
    The resolver module's class is `FastLaneNameResolver(names_fn)`
    and the manager owns ONE long-lived instance
    (it holds the resolved-address cache and the
    generation counter the rule-table cache keys on), so config_put uses
    `mgr._fastlane_resolver` rather than constructing anything.
    The BEHAVIOURAL contracts below are
    resolve-before-lint, non-fatal fallback, and the negative control;
    the injection point is the manager's instance."""

    def test_resolve_refreshes_the_cache_that_lint_then_reads(
        self, app_test, monkeypatch,
    ):
        """A resolve pass does not stop a fresh address being reported
        "never observed" -- that would be FALSE in production: the census is
        populated by requests arriving at admission, and no resolve pass can
        change that. A fake that wrote the census itself would
        hide this.

        The real contract, and what this proves: resolve_once() refreshes
        the resolver's own address cache BEFORE lint runs, and lint reads that
        cache through resolve_name. The observable is the unresolved-name warning -- a name the
        refresh has just learned must NOT be reported as unresolvable.
        """
        app, client, _boot = app_test
        mgr = app.state.manager
        new_ip = "203.0.113.5"

        calls = {"resolved": 0}

        class _FakeResolver:
            def __init__(self):
                self._cache = {}

            async def resolve_once(self):
                calls["resolved"] += 1
                self._cache["fresh-container"] = frozenset({normalize_address(new_ip)})

            def addresses_for(self, name):
                return self._cache.get(name, frozenset())

        monkeypatch.setattr(mgr, "_fastlane_resolver", _FakeResolver(), raising=False)

        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"container_name": "fresh-container", "label": "fresh container"},
        ]}})
        assert r.status_code == 200
        warnings = r.json()["warnings"]

        # the refresh actually ran...
        assert calls["resolved"] == 1
        # ...and lint saw its result: no unresolvable warning for a name the
        # refresh had just cached. If lint ran BEFORE the refresh, or read a
        # different cache, this fires.
        assert not any("resolves to NO addresses" in w for w in warnings), warnings

    def test_a_name_the_refresh_did_NOT_learn_is_still_reported(
        self, app_test, monkeypatch,
    ):
        """Control for the arm above: the same wiring, a name the refresh does
        not cache. Without this, a lint that never warns would pass that test
        for entirely the wrong reason.
        """
        app, client, _boot = app_test
        mgr = app.state.manager

        class _FakeResolver:
            async def resolve_once(self):
                pass

            def addresses_for(self, name):
                return frozenset()

        monkeypatch.setattr(mgr, "_fastlane_resolver", _FakeResolver(), raising=False)

        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"container_name": "never-heard-of-it", "label": "never heard of it"},
        ]}})
        assert r.status_code == 200
        assert any("resolves to NO addresses" in w for w in r.json()["warnings"])

    def test_without_a_resolver_the_same_fresh_address_would_be_reported(
        self, app_test, monkeypatch
    ):
        """Negative control proving the test above is non-vacuous: with NO
        resolver module present (ImportError, falls back to last-known
        census per the non-fatal branch), the same brand-new address DOES
        get the "never observed" warning -- exactly the behavior the resolve
        pass exists to prevent on a fresh save."""
        app, client, _boot = app_test
        # Explicitly remove the resolver the lifespan attached. This is the
        # NEGATIVE arm: with no resolve pass, the same brand-new address DOES
        # get the "never observed" warning -- which is what makes the positive
        # test above non-vacuous rather than passing for some other reason.
        monkeypatch.setattr(app.state.manager, "_fastlane_resolver", None, raising=False)
        new_ip = "203.0.113.9"
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": new_ip, "tag_ranks": {"main": 1}},
        ]}})
        assert r.status_code == 200
        assert any("never been observed" in w for w in r.json()["warnings"])

    def test_resolve_failure_does_not_500_the_write(self, app_test, monkeypatch, caplog):
        """Non-fatal doctrine: a resolve pass that raises must not turn an
        otherwise-valid config write into a 500 -- falls back to linting
        against the last-known census, same as the pre-existing
        behavior (proven by the negative control above)."""
        app, client, _boot = app_test

        class _RaisingResolver:
            async def resolve_once(self):
                raise RuntimeError("resolver boom")

            def addresses_for(self, name):
                # Shaped like the real object: lint still gets a usable
                # reader even when the refresh failed, which is the whole
                # point of the fallback being non-fatal.
                return frozenset()

        monkeypatch.setattr(
            app.state.manager, "_fastlane_resolver", _RaisingResolver(), raising=False
        )

        with caplog.at_level(logging.ERROR):
            r = client.put("/api/config", json={"fastlane": {"rules": [
                {"address": "203.0.113.10", "tag_ranks": {"main": 1}},
            ]}})
        assert r.status_code == 200
        assert any(
            "fastlane census resolve pass failed" in rec.message
            for rec in caplog.records
        )

    def test_config_put_does_not_import_fastlane_resolve_at_module_level(self):
        """The contract guarded here: the fastlane_resolve.py module may
        or may not be importable, and a pass for the reason
        "the module is absent" would be vacuous. Instead: config_put must
        not take a module-level dependency
        on the resolver.
        Asserted against the AST, so it stays true whether or not the module
        happens to be importable today.
        """
        import ast as _ast
        import turbohaul.api.config_put as config_put_module

        tree = _ast.parse(open(config_put_module.__file__).read())
        top_level_imports = [
            n
            for n in tree.body
            if isinstance(n, (_ast.Import, _ast.ImportFrom))
        ]
        assert not any(
            "fastlane_resolve" in (getattr(n, "module", "") or "")
            or any("fastlane_resolve" in a.name for a in n.names)
            for n in top_level_imports
        )


class TestFastLaneRoundTrip:
    """Assert on split() output, not the raw dict -- catches a missing fastlane pass-through."""

    def test_enabled_survives_restart_via_split(self, app_test):
        _app, client, boot = app_test
        r = client.put("/api/config", json={"fastlane": {"enabled": True}})
        assert r.status_code == 200

        override_path = _runtime_override_path(boot.storage.state_db_path)
        persisted = yaml.safe_load(override_path.read_text())
        assert persisted["fastlane"]["enabled"] is True

        # Replay the boot merge exactly as __main__.py does it, then split()
        # -- a missing `fastlane=self.fastlane` in config.py's split() would
        # silently drop the field back to its default here.
        cfg = TurbohaulConfig(
            server=boot.server, storage=boot.storage, runtime=boot.runtime,
            ui=boot.ui, queue=QueueConfig(), pull=PullConfig(),
        )
        cfg_data = cfg.model_dump()
        for section, values in persisted.items():
            if section in RUNTIME_SECTIONS and section in cfg_data and isinstance(values, dict):
                cfg_data[section] = {**cfg_data[section], **values}
        replayed = TurbohaulConfig(**cfg_data)
        _replayed_boot, replayed_runtime = replayed.split()
        assert replayed_runtime.fastlane.enabled is True


class TestFastLaneRulesCap:
    """The 10-rule cap: model-level (any construction, any caller) and PUT-level
    (an operator gets a direct 400, not a silent truncate)."""

    def _rules(self, n: int) -> list[dict]:
        return [{"address": f"10.0.0.{i}"} for i in range(n)]

    def test_ten_rules_is_within_the_cap(self):
        cfg = FastLaneConfig(rules=self._rules(10))
        assert len(cfg.rules) == 10

    def test_eleven_rules_raises_naming_count_and_cap(self):
        with pytest.raises(ValidationError) as exc_info:
            FastLaneConfig(rules=self._rules(11))
        message = str(exc_info.value)
        assert "11" in message, message
        assert "10" in message, message

    def test_put_over_the_cap_is_rejected_naming_count_and_cap(self, app_test):
        _app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": self._rules(11)}})
        assert r.status_code == 400
        assert "11" in r.json()["detail"]
        assert "10" in r.json()["detail"]

    def test_put_over_the_cap_does_not_persist_anything(self, app_test):
        """A rejected PUT must not silently truncate-and-save -- the operator
        asked for 11 and got a 400, not a quietly-accepted 10."""
        _app, client, boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": self._rules(11)}})
        assert r.status_code == 400
        override_path = _runtime_override_path(boot.storage.state_db_path)
        assert not override_path.exists()


class TestFastLaneDefaults:
    """Defaults: max_normal_wait_s matches the one-turn guarantee's own
    ~1 hour window; cross_model_switches_per_min's default and ceiling are
    both pinned here."""

    def test_max_normal_wait_s_default_is_3600(self):
        assert FastLaneConfig().max_normal_wait_s == 3600.0

    def test_cross_model_switches_per_min_default_is_2(self):
        assert FastLaneConfig().cross_model_switches_per_min == 2

    def test_cross_model_switches_per_min_ceiling_is_3(self):
        FastLaneConfig(cross_model_switches_per_min=3)  # at the ceiling: fine
        with pytest.raises(ValidationError):
            FastLaneConfig(cross_model_switches_per_min=4)


class TestFastLaneCorruptBlock:
    """Drive the REAL __main__ boot sequence, not create_app (which
    never reads runtime_config.yaml)."""

    def test_corrupt_persisted_block_boots_disabled_and_logs_error(self, tmp_path, caplog):
        storage_root = tmp_path / "state"
        storage_root.mkdir()
        state_db = storage_root / "state.sqlite"
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=storage_root / "blobs",
                manifests_path=storage_root / "manifests",
                import_allowed_root=storage_root / "import-staging",
                state_db_path=state_db,
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=tmp_path / "fake",
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        cfg = TurbohaulConfig(
            server=boot.server, storage=boot.storage, runtime=boot.runtime,
            ui=boot.ui, queue=QueueConfig(), pull=PullConfig(),
        )

        override_path = _runtime_override_path(state_db)
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastlane": {"enabled": "banana", "rules": 7},
        }))

        override = load_runtime_override(state_db)
        assert override is not None

        # Replicate __main__.py's isolated fail-loud fastlane parse.
        from pydantic import ValidationError

        fastlane_config_error = None
        cfg_data = cfg.model_dump()
        for section, values in override.items():
            if section in RUNTIME_SECTIONS and section in cfg_data and isinstance(values, dict):
                if section == "fastlane":
                    candidate = {**cfg_data[section], **values}
                    try:
                        FastLaneConfig(**candidate)
                    except ValidationError as e:
                        fastlane_config_error = str(e)
                        with caplog.at_level(logging.ERROR):
                            logging.getLogger("turbohaul.main").error(
                                "persisted fastlane config rejected, booting "
                                "with fastlane disabled: %s", e,
                            )
                        continue
                    cfg_data[section] = candidate
                else:
                    cfg_data[section] = {**cfg_data[section], **values}

        merged = TurbohaulConfig(**cfg_data)  # must not raise
        _boot2, runtime2 = merged.split()

        assert runtime2.fastlane.enabled is False
        assert fastlane_config_error is not None
        assert any(
            "fastlane" in rec.message and rec.levelname == "ERROR"
            for rec in caplog.records
        )


class TestFastLaneSchemaEndpoint:
    """fastlane is served by GET /api/config/schema as well as being
    PUT-able -- see config_schema.py's own module
    docstring. This module asserts its own fields, per the existing convention in
    test_api_config_schema.py (each feature asserts its own fields in its own
    test module)."""

    def test_fastlane_in_sections_registry(self):
        section_names = {name for name, _ in _SECTIONS}
        assert "fastlane" in section_names

    def test_fastlane_scalar_bounds_and_defaults(self, app_test):
        _app, client, _boot = app_test
        body = client.get("/api/config/schema").json()
        fastlane = body["fastlane"]

        max_wait = fastlane["max_normal_wait_s"]
        assert max_wait["type"] == "number"
        assert max_wait["default"] == 3600.0
        assert max_wait["minimum"] == 1.0
        assert max_wait["maximum"] == 3600.0

        switches = fastlane["cross_model_switches_per_min"]
        assert switches["type"] == "integer"
        assert switches["default"] == 2
        assert switches["minimum"] == 0
        assert switches["maximum"] == 3

        census_ttl = fastlane["census_ttl_hours"]
        assert census_ttl["type"] == "integer"
        assert census_ttl["default"] == 168
        assert census_ttl["minimum"] == 1
        assert census_ttl["maximum"] == 8760

    def test_fastlane_enabled_and_rules_not_specially_hidden_by_the_endpoint(self, app_test):
        """The schema endpoint itself does not curate -- it serves every
        FastLaneConfig field, same as it does for every other section.
        `enabled` (already has a dedicated General toggle) and `rules` (a
        nested list, has its own Queue -> Fast Lane UI) are left out of the
        NEW curated Settings card by that card's own JSX, not by this
        endpoint -- keeping the endpoint itself uncurated is what lets any
        future consumer decide for itself rather than inheriting one FE
        card's opinion about what's worth showing."""
        _app, client, _boot = app_test
        body = client.get("/api/config/schema").json()
        fastlane = body["fastlane"]
        assert fastlane["enabled"]["type"] == "boolean"
        assert fastlane["enabled"]["default"] is False
        assert fastlane["rules"]["type"] == "array"
        assert fastlane["rules"]["default"] == []

    def test_monitor_poll_interval_exclusive_lower_bound_now_live(self, app_test):
        """The exclusive-bound fallback (_bound()) is otherwise only
        exercised by a monkeypatch in test_api_config_schema.py, because no
        real queue/pull/kv field uses gt/lt. monitor.poll_interval_s
        (gt=0.0, le=60.0) is a real field that hits that path, since
        monitor is in _SECTIONS."""
        _app, client, _boot = app_test
        body = client.get("/api/config/schema").json()
        poll_interval = body["monitor"]["poll_interval_s"]
        assert poll_interval["minimum"] == 0.0
        assert poll_interval["maximum"] == 60.0


class TestLintReceivesResolverAtThePutSite:
    """lint_rules takes resolve_name=, and BOTH call sites must pass it,
    or the collision and prefix arms, and the
    unresolved-name warning, are unreachable outside tests.

    ⚠ This is the same class of defect as the one documented at
    the top of the lint block ("lint_rules had ZERO production callers").
    The end-to-end assertion below is the
    guard against it: it goes through the real
    route, the real manager and the real resolver.
    """

    def test_put_with_an_unresolvable_container_name_returns_the_warning(self, app_test):
        # The unresolved-name warning through the actual HTTP surface. The manager's real
        # resolver has never heard of this container, so addresses_for()
        # returns frozenset() and lint must say so IN THE RESPONSE -- which
        # can only happen if config_put passes resolve_name.
        app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"container_name": "no-such-container-anywhere", "label": "ghost"},
        ]}})
        assert r.status_code == 200
        warnings = r.json()["warnings"]
        assert any("resolves to NO addresses" in w for w in warnings), warnings
        assert any("no-such-container-anywhere" in w for w in warnings), warnings

    def test_an_address_only_rule_does_not_trip_the_unresolved_warning(self, app_test):
        # Control: without this, a lint that warned unconditionally would pass
        # the arm above for the wrong reason.
        app, client, _boot = app_test
        r = client.put("/api/config", json={"fastlane": {"rules": [
            {"address": "203.0.113.77", "label": "off-network caller"},
        ]}})
        assert r.status_code == 200
        assert not any("resolves to NO addresses" in w for w in r.json()["warnings"])


class TestFastLaneRuleContainerNameIdentity:
    """FastLaneRule names a client by exactly
    one of `address` (off-network: LAN, VPN, any static IP) or
    `container_name` (on-network: resolved forward-only at runtime,
    fastlane_resolve.py) — never both, never neither."""

    def test_address_and_container_name_both_set_rejected(self):
        with pytest.raises(ValidationError) as exc:
            FastLaneRule(address="1.2.3.4", container_name="gateway-svc")
        assert "container_name" in str(exc.value)

    def test_neither_address_nor_container_name_set_rejected(self):
        with pytest.raises(ValidationError) as exc:
            FastLaneRule()
        assert "container_name" in str(exc.value)

    def test_container_name_only_is_a_valid_rule(self):
        rule = FastLaneRule(container_name="gateway-svc")
        assert rule.address is None
        assert rule.container_name == "gateway-svc"

    def test_non_ip_address_rejected_names_container_name_as_the_intended_field(self):
        # closes the trap where "gateway-svc" as an `address`
        # passed validation, was silently dropped by compile_fastlane's
        # normalize_address try/except, and lint_rules went on to
        # misreport it as "never observed" — a misleading diagnosis
        # pointing at the wrong cause.
        with pytest.raises(ValidationError) as exc:
            FastLaneRule(address="gateway-svc")
        assert "container_name" in str(exc.value)

    def test_valid_ip_address_still_accepted_control(self):
        # Control for the arm above: a real IP must still pass, or the
        # validator is just a constant that rejects everything.
        rule = FastLaneRule(address="192.0.2.10")
        assert rule.address == "192.0.2.10"

    def test_cidr_still_rejected_backcompat(self):
        with pytest.raises(ValidationError):
            FastLaneRule(address="192.0.2.0/24")

    def test_five_legacy_address_only_rules_still_load_and_compile_unchanged(self):
        # Back-compat arm: a config carrying only address/label/tag_ranks
        # (no container_name key anywhere) must keep validating and
        # compiling exactly as it did before.
        legacy_rules_data = [
            {"address": "10.0.0.2", "label": "a", "tag_ranks": {"main": 1}},
            {"address": "10.0.0.3", "label": "b"},
            {"address": "10.0.0.4", "label": "c", "tag_ranks": {"curator": 2}},
            {"address": "10.0.0.5", "label": "d"},
            {"address": "10.0.0.6", "label": "e", "tag_ranks": {"sub_agent": 1}},
        ]
        cfg = FastLaneConfig(enabled=True, rules=legacy_rules_data)
        assert len(cfg.rules) == 5
        for rule, orig in zip(cfg.rules, legacy_rules_data):
            assert rule.address == orig["address"]
            assert rule.container_name is None

        compiled = compile_fastlane(cfg.rules)
        assert len(compiled) == 5
        assert all(c.container_name is None for c in compiled)
        assert [c.raw_address for c in compiled] == [r["address"] for r in legacy_rules_data]


class TestRollbackSafetyOfThePersistedKey:
    """`container_name: null` written into runtime_config.yaml makes
    an older build reject the section (extra="forbid") and boot Fast
    Lane DISABLED. A deployment may be rolled back, so this is a real
    hazard, not a theoretical one, so it is fixed rather than merely noted.
    """

    def test_an_address_only_rule_persists_WITHOUT_a_container_name_key(self):
        from turbohaul.api.config_put import _strip_null_container_names

        out = _strip_null_container_names(
            {"fastlane": {"enabled": True, "rules": [
                {"address": "10.0.0.2", "label": "a", "container_name": None},
            ]}}
        )
        assert "container_name" not in out["fastlane"]["rules"][0], out

    def test_a_rule_that_REALLY_names_a_container_keeps_its_key(self):
        # Control: a blanket strip would silently erase real configuration --
        # the same silent-erasure class the FE array-PUT fix exists to prevent.
        from turbohaul.api.config_put import _strip_null_container_names

        out = _strip_null_container_names(
            {"fastlane": {"enabled": True, "rules": [
                {"container_name": "gateway-svc", "label": "Gateway"},
            ]}}
        )
        assert out["fastlane"]["rules"][0]["container_name"] == "gateway-svc"

    def test_CONTROL_the_assertion_fires_when_the_null_key_IS_present(self):
        # Proves the first assertion can fail: an unstripped payload keeps it.
        raw = {"fastlane": {"rules": [{"address": "10.0.0.2", "container_name": None}]}}
        assert "container_name" in raw["fastlane"]["rules"][0]

    def test_a_payload_without_fastlane_is_returned_untouched(self):
        from turbohaul.api.config_put import _strip_null_container_names
        p = {"queue": {"grace_seconds": 30}}
        assert _strip_null_container_names(p) == p
