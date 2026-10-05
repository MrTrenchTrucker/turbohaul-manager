"""Tests for the PUT /api/config split (runtime-mutable vs boot-only sections)."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.config_put import (
    RUNTIME_SECTIONS,
    load_runtime_override,
    load_runtime_provenance_stamp,
    save_runtime_config,
)
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestPutConfigRuntime:
    def test_put_queue_grace_seconds(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"grace_seconds": 60}})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["current"]["queue"]["grace_seconds"] == 60
        # Verify manager's grace timer config was refreshed
        mgr = app.state.manager
        assert mgr.grace.grace_seconds == 60

    def test_put_queue_idle_hot(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"idle_hot_load_seconds": 240}})
        assert r.status_code == 200
        mgr = app.state.manager
        assert mgr.idle.idle_seconds == 240

    def test_put_pull_concurrency(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"pull": {"pull_concurrency": 4}})
        assert r.status_code == 200
        mgr = app.state.manager
        assert mgr.runtime.pull.pull_concurrency == 4

    def test_get_after_put_reflects_change(self, app_test):
        app, client = app_test
        client.put("/api/config", json={"queue": {"grace_seconds": 45}})
        r = client.get("/api/config")
        body = r.json()
        assert body["queue"]["grace_seconds"] == 45


class TestPutConfigBootForbidden:
    def test_put_server_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"server": {"port": 11500}})
        assert r.status_code == 403
        assert "BOOT-ONLY" in r.text or "boot-only" in r.text.lower()

    def test_put_storage_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"storage": {"blob_store_path": "/tmp/evil"}})
        assert r.status_code == 403

    def test_put_runtime_paths_403(self, app_test):
        """The killer attack: changing runtime.llama_server_binary -> RCE primitive (binary-swap attack)."""
        app, client = app_test
        r = client.put(
            "/api/config",
            json={"runtime": {"llama_server_binary": "/tmp/evil.sh"}},
        )
        assert r.status_code == 403

    def test_put_ui_403(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"ui": {"static_path": "/etc"}})
        assert r.status_code == 403


class TestPutConfigBadInput:
    def test_unknown_section_400(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"evil_section": {"x": 1}})
        assert r.status_code == 400

    def test_non_object_section_400(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"queue": "not-an-object"})
        assert r.status_code == 400

    def test_invalid_value_400(self, app_test):
        app, client = app_test
        # grace_seconds has bound 0..3600
        r = client.put("/api/config", json={"queue": {"grace_seconds": -1}})
        assert r.status_code == 400


class TestPutConfigUnknownFieldInSection:
    def test_unknown_field_in_queue_rejected(self, app_test):
        app, client = app_test
        # QueueConfig has extra="forbid" so unknown subfield -> 400
        r = client.put("/api/config", json={"queue": {"evil_field": 1}})
        assert r.status_code == 400


class TestMonitorSectionAccepted:
    """The monitor section was missing from valid_sections, causing
    FE edits to be silently rejected with HTTP 400 'unknown section'."""

    def test_put_monitor_enabled(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"monitor": {"enabled": False}})
        assert r.status_code == 200
        assert r.json()["current"]["monitor"]["enabled"] is False

    def test_put_monitor_poll_interval(self, app_test):
        app, client = app_test
        r = client.put("/api/config", json={"monitor": {"poll_interval_s": 2.5}})
        assert r.status_code == 200
        assert r.json()["current"]["monitor"]["poll_interval_s"] == 2.5

    def test_monitor_in_runtime_sections(self):
        assert "monitor" in RUNTIME_SECTIONS

    def test_get_config_includes_monitor(self, app_test):
        app, client = app_test
        r = client.get("/api/config")
        assert r.status_code == 200
        assert "monitor" in r.json()


class TestRuntimeConfigDurability:
    """PUT mutations must persist to disk and survive restart."""

    def test_put_persists_to_override_file(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"queue": {"grace_seconds": 99}})
        assert r.status_code == 200
        # Override file should exist next to state_db
        override_path = state_db.parent / "runtime_config.yaml"
        assert override_path.exists()
        import yaml
        data = yaml.safe_load(override_path.read_text())
        assert data["queue"]["grace_seconds"] == 99

    def test_put_monitor_persists(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"monitor": {"poll_interval_s": 3.0}})
        assert r.status_code == 200
        override_path = state_db.parent / "runtime_config.yaml"
        assert override_path.exists()
        import yaml
        data = yaml.safe_load(override_path.read_text())
        assert data["monitor"]["poll_interval_s"] == 3.0

    def test_load_runtime_override_returns_none_when_absent(self, tmp_path):
        result = load_runtime_override(tmp_path / "nonexistent.sqlite")
        assert result is None

    def test_load_runtime_override_returns_data_when_present(self, tmp_path):
        import yaml
        state_db = tmp_path / "state.sqlite"
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "queue": {"grace_seconds": 42},
            "monitor": {"enabled": False},
        }))
        result = load_runtime_override(state_db)
        assert result is not None
        assert result["queue"]["grace_seconds"] == 42
        assert result["monitor"]["enabled"] is False

    def test_save_and_reload_roundtrip(self, tmp_path):
        """save_runtime_config -> load_runtime_override roundtrip.

        save_runtime_config now takes the raw payload
        dict (what a PUT actually sent), not a full RuntimeConfig -- adapted
        here to the new sparse signature; the round-trip intent is unchanged.
        """
        state_db = tmp_path / "state.sqlite"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "queue": {"grace_seconds": 123},
            "monitor": {"poll_interval_s": 5.0},
        }
        save_runtime_config(state_db, payload)
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"]["grace_seconds"] == 123
        assert loaded["monitor"]["poll_interval_s"] == 5.0


class TestLegacyFastlineKeyStrippedOnSave:
    """_load_runtime_override_raw is
    deliberately UNfiltered (so _provenance_stamp survives the read-merge-
    write), so a legacy top-level "fastline" key from a pre-rename build
    would otherwise ride along untouched on every future save forever --
    shown by a no-op PUT repro (address count 2 -> 4,
    both blocks present). These prove the save path now cleans it up, without
    losing the rules it carries.
    """

    def _write_legacy_override(self, tmp_path):
        import yaml
        state_db = tmp_path / "state.sqlite"
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastline": {
                "enabled": True,
                "rules": [
                    {"address": "fdd0:0:0:1f::3", "label": "Example Agent A",
                     "tag_ranks": {"main": 4}},
                    {"address": "fdd0:0:0:1f::2", "label": "Example Agent B",
                     "tag_ranks": {"main": 4}},
                ],
            },
        }))
        return state_db

    def test_put_to_unrelated_section_still_strips_legacy_key(self, tmp_path):
        """A concrete repro: a save that doesn't even TOUCH Fast
        Lane (here, "queue") still cleans up the stale "fastline" block on
        disk -- proving the cleanup is not coupled to which section the PUT
        actually names.
        """
        state_db = self._write_legacy_override(tmp_path)
        save_runtime_config(state_db, {"queue": {"grace_seconds": 77}})

        import yaml
        raw = yaml.safe_load(
            (state_db.parent / "runtime_config.yaml").read_text()
        )
        assert "fastline" not in raw
        assert raw["fastlane"]["enabled"] is True
        assert raw["queue"]["grace_seconds"] == 77

    def test_rules_survive_byte_identical_after_the_strip(self, tmp_path):
        """A different, stronger claim than 'the key is gone': the rule DATA
        underneath -- both addresses, both labels, both tag_ranks -- must be
        byte-identical after the save, not just present-in-some-form.

        Reads the RAW file on disk (not load_runtime_override) deliberately:
        load_runtime_override already migrates on its way OUT (the read-side migration),
        so asserting against ITS return value would pass even with this change's
        save-side fix reverted -- it would just be re-proving the read-side
        migration, which was never in question here. The claim this change makes
        is about what's actually WRITTEN to disk.
        """
        import yaml
        state_db = self._write_legacy_override(tmp_path)
        save_runtime_config(state_db, {"monitor": {"poll_interval_s": 9.0}})

        raw = yaml.safe_load(
            (state_db.parent / "runtime_config.yaml").read_text()
        )
        assert "fastline" not in raw
        assert raw["fastlane"] == {
            "enabled": True,
            "rules": [
                {"address": "fdd0:0:0:1f::3", "label": "Example Agent A",
                 "tag_ranks": {"main": 4}},
                {"address": "fdd0:0:0:1f::2", "label": "Example Agent B",
                 "tag_ranks": {"main": 4}},
            ],
        }

    def test_no_legacy_key_present_is_a_true_no_op(self, tmp_path):
        """Negative control: a file with no legacy key at all is unaffected
        by the new migrate call -- proves the strip is conditional on the
        legacy key actually being present, not an unconditional rewrite."""
        state_db = tmp_path / "state.sqlite"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        save_runtime_config(state_db, {
            "fastlane": {"enabled": True, "rules": []},
            "queue": {"grace_seconds": 5},
        })
        loaded = load_runtime_override(state_db)
        assert loaded["fastlane"] == {"enabled": True, "rules": []}
        assert loaded["queue"]["grace_seconds"] == 5

    def test_provenance_stamp_survives_the_strip(self, tmp_path):
        """The whole reason _load_runtime_override_raw stays unfiltered:
        _provenance_stamp must still round-trip correctly through a save
        that also strips a legacy fastline key -- the two concerns (stamp
        preservation, legacy-key migration) must not interfere."""
        state_db = self._write_legacy_override(tmp_path)
        save_runtime_config(state_db, {"queue": {"grace_seconds": 10}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert "queue" in stamp
        assert "grace_seconds" in stamp["queue"]


class TestLegacyFastlineStampSectionMerged:
    """_provenance_stamp is ITSELF
    section-keyed, so the config-VALUE migration (which only rewrote
    the top-level key `existing` is stored under) left a stale
    stamp["fastline"] sibling sitting next to the correct stamp["fastlane"]
    forever (the symptom being that, once the config values were migrated,
    /api/config still served
    _provenance_stamp.sections == ['fastlane', 'fastline']). Inert (nothing
    reads the stamp per-field for a correctness decision -- confirmed by
    reading compute_config_provenance directly: it never calls
    load_runtime_provenance_stamp at all), but user-visible, and the last
    "fastline" string the API served.
    """

    def _write_two_stamp_sections(self, tmp_path, fastline_stamp, fastlane_stamp,
                                    fastlane_value=None):
        import yaml
        state_db = tmp_path / "state.sqlite"
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastlane": fastlane_value or {"enabled": True, "rules": []},
            "_provenance_stamp": {
                "fastline": fastline_stamp,
                "fastlane": fastlane_stamp,
            },
        }))
        return state_db

    def test_legacy_stamp_section_merged_and_removed(self, tmp_path):
        """A realistic shape: fastlane's own PUT is newer
        than the stale fastline entry for the field they share.
        After a save, only "fastlane" survives in the stamp, and the shared
        field keeps the NEWER (fastlane) timestamp."""
        state_db = self._write_two_stamp_sections(
            tmp_path,
            fastline_stamp={"enabled": "2026-08-16T22:55:01+00:00"},
            fastlane_stamp={"enabled": "2026-08-16T23:43:09+00:00"},
        )
        save_runtime_config(state_db, {"queue": {"grace_seconds": 1}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert "fastline" not in stamp
        assert stamp["fastlane"]["enabled"] == "2026-08-16T23:43:09+00:00"

    def test_legacy_only_field_carries_over(self, tmp_path):
        """A field the legacy entry recorded but the new one never touched
        must survive the merge under the new key, not be silently dropped --
        merging is the whole point, not just "prefer fastlane's own keys"."""
        state_db = self._write_two_stamp_sections(
            tmp_path,
            fastline_stamp={
                "enabled": "2026-08-16T22:55:01+00:00",
                "max_normal_wait_s": "2026-08-16T20:00:00+00:00",
            },
            fastlane_stamp={"enabled": "2026-08-16T23:43:09+00:00"},
        )
        save_runtime_config(state_db, {"queue": {"grace_seconds": 1}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert "fastline" not in stamp
        assert stamp["fastlane"]["enabled"] == "2026-08-16T23:43:09+00:00"
        assert stamp["fastlane"]["max_normal_wait_s"] == "2026-08-16T20:00:00+00:00"

    def test_inverted_timestamps_the_legacy_later_one_wins(self, tmp_path):
        """THE test that actually proves this is a real per-field
        comparison, not "fastlane always wins because that's the new name".

        In real data the legacy (fastline) entry is ALWAYS older
        than the new (fastlane) one -- the rename happened at one instant,
        so every fastline timestamp predates every fastlane timestamp by
        construction. An implementation that just does
        `current.setdefault(field, legacy_ts)` (silently preferring whichever
        section is already named "fastlane", never actually comparing) would
        pass every test built from that real chronology, INCLUDING
        test_legacy_stamp_section_merged_and_removed above -- the natural
        data distribution hides this whole bug class. Only a fixture where
        the legacy timestamp is deliberately LATER disagrees with that
        broken implementation, so it's the one that actually exercises the
        "newer wins" logic rather than the "new key wins" logic.
        """
        state_db = self._write_two_stamp_sections(
            tmp_path,
            fastline_stamp={"enabled": "2026-08-17T05:00:00+00:00"},  # later
            fastlane_stamp={"enabled": "2026-08-16T23:43:09+00:00"},  # earlier
        )
        save_runtime_config(state_db, {"queue": {"grace_seconds": 1}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert "fastline" not in stamp
        assert stamp["fastlane"]["enabled"] == "2026-08-17T05:00:00+00:00"

    def test_no_legacy_stamp_section_is_true_no_op(self, tmp_path):
        """Negative control: a stamp with no legacy "fastline" entry at all
        is unaffected by the merge -- proves it's conditional, not an
        unconditional rewrite of every stamp on every save."""
        import yaml
        state_db = tmp_path / "state.sqlite"
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastlane": {"enabled": True, "rules": []},
            "_provenance_stamp": {"fastlane": {"enabled": "2026-08-16T23:43:09+00:00"}},
        }))
        save_runtime_config(state_db, {"queue": {"grace_seconds": 1}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert stamp["fastlane"] == {"enabled": "2026-08-16T23:43:09+00:00"}

    def test_sibling_section_stamps_untouched(self, tmp_path):
        """A sibling section's own stamp entries (e.g. "monitor", already
        written before this save) must survive byte-identical -- the merge
        is scoped to fastline/fastlane only."""
        import yaml
        state_db = tmp_path / "state.sqlite"
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastlane": {"enabled": True, "rules": []},
            "monitor": {"poll_interval_s": 3.0},
            "_provenance_stamp": {
                "fastline": {"enabled": "2026-08-16T22:55:01+00:00"},
                "fastlane": {"enabled": "2026-08-16T23:43:09+00:00"},
                "monitor": {"poll_interval_s": "2026-08-16T18:00:00+00:00"},
            },
        }))
        save_runtime_config(state_db, {"queue": {"grace_seconds": 1}})
        stamp = load_runtime_provenance_stamp(state_db)
        assert stamp["monitor"] == {"poll_interval_s": "2026-08-16T18:00:00+00:00"}

    def test_get_api_config_no_longer_serves_the_legacy_section(self, app_test):
        """End-to-end: reproduces a manual check that found a stale
        _provenance_stamp section: it came back with
        sections: ['fastlane', 'fastline']. Writes the legacy-shaped stamp
        directly to the override file, issues one PUT (any section), then
        confirms GET /api/config's _provenance_stamp no longer has
        "fastline" -- and that _provenance (the OTHER, unrelated field) is
        untouched, since compute_config_provenance never reads the stamp at
        all.
        """
        import yaml
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(yaml.safe_dump({
            "fastlane": {"enabled": True, "rules": []},
            "_provenance_stamp": {
                "fastline": {"enabled": "2026-08-16T22:55:01+00:00"},
                "fastlane": {"enabled": "2026-08-16T23:43:09+00:00"},
            },
        }))
        r = client.put("/api/config", json={"queue": {"grace_seconds": 42}})
        assert r.status_code == 200
        body = client.get("/api/config").json()
        assert "fastline" not in body["_provenance_stamp"]
        assert body["_provenance_stamp"]["fastlane"]["enabled"] == "2026-08-16T23:43:09+00:00"
        assert body["_provenance"]["fastlane"]["enabled"] == "persisted"


class TestSparsePersistence:
    """A PUT persists ONLY the fields it actually
    touched, merged onto whatever is already on disk -- never a full
    snapshot of the in-memory RuntimeConfig. Causal both directions: a
    writer that persists nothing must fail
    test_sparse_write_persists_the_touched_key; one that persists
    everything (the pre-fix behaviour) must fail a DIFFERENT test,
    test_sparse_write_does_not_persist_untouched_sibling_keys.
    """

    def test_sparse_write_persists_the_touched_key(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"]["grace_seconds"] == 77

    def test_sparse_write_does_not_persist_untouched_sibling_keys(self, app_test):
        """The core of the sparse-write claim: QueueConfig has 22 fields. A PUT touching only
        grace_seconds must not freeze the other 21 -- pre-fix, this failed
        with all 22 present."""
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"] == {"grace_seconds": 77}, (
            f"expected ONLY the touched key, got {sorted(loaded['queue'].keys())}"
        )
        # And no OTHER section got dragged in at all.
        assert set(loaded.keys()) == {"queue"}

    def test_sparse_write_does_not_persist_untouched_sibling_sections(self, app_test):
        """Same claim, different axis: a PUT touching only `monitor` must not
        write `queue`/`pull`/`persist`/`kv` at all -- pre-fix, all five
        sections were dumped on every PUT regardless of which one the
        request named."""
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"monitor": {"poll_interval_s": 2.5}})
        assert r.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert set(loaded.keys()) == {"monitor"}

    def test_sparse_write_accumulates_across_separate_puts(self, app_test):
        """Read-merge-write, not overwrite: a second PUT to a different
        section must not erase the first PUT's already-persisted key."""
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r1 = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r1.status_code == 200
        r2 = client.put("/api/config", json={"monitor": {"poll_interval_s": 2.5}})
        assert r2.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"] == {"grace_seconds": 77}
        assert loaded["monitor"] == {"poll_interval_s": 2.5}
        assert set(loaded.keys()) == {"queue", "monitor"}

    def test_sparse_write_second_put_to_same_section_merges_not_replaces(self, app_test):
        """Two PUTs to different fields in the SAME section must both
        survive -- proves the read-merge-write is per-FIELD, not
        per-section (a section-level overwrite would silently drop the
        first PUT's field when the second PUT targets a different field in
        the same section)."""
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r1 = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r1.status_code == 200
        r2 = client.put("/api/config", json={"queue": {"max_grace_extensions": 9}})
        assert r2.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"] == {"grace_seconds": 77, "max_grace_extensions": 9}


class TestPutTimeProvenanceStamp:
    """A PUT stamps which (section, field) pairs it
    explicitly touched, and when -- forward-only, additive, does not change
    precedence or load_runtime_override's existing contract."""

    def test_put_stamps_the_touched_field(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        stamp = load_runtime_provenance_stamp(state_db)
        assert "grace_seconds" in stamp.get("queue", {}), stamp
        # a real, parseable UTC timestamp, not a placeholder
        ts = stamp["queue"]["grace_seconds"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_load_runtime_override_never_exposes_the_stamp_key(self, app_test):
        """The contract-preservation claim itself: load_runtime_override
        (every existing caller's function) must return EXACTLY the same
        key set as before this change, even after a PUT has written a stamp
        into the same file."""
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert set(loaded.keys()) == {"queue"}, (
            f"_provenance_stamp leaked into load_runtime_override's return: {loaded}"
        )
        assert loaded["queue"] == {"grace_seconds": 77}

    def test_old_file_with_no_stamp_key_still_loads_and_stamp_is_empty(self, tmp_path):
        """Backward compat, a REQUIRED constraint: a
        pre-existing runtime_config.yaml written before this code shipped
        (no _provenance_stamp key at all) must still load unchanged via
        load_runtime_override, and load_runtime_provenance_stamp must
        report {} for it -- not crash, not fabricate a stamp."""
        import yaml as _yaml

        state_db = tmp_path / "state.sqlite"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.write_text(_yaml.safe_dump({
            "queue": {"grace_seconds": 480, "max_grace_extensions": 50},
            "kv": {"ram_cache_max_bytes": 25769803776},
        }))

        loaded = load_runtime_override(state_db)
        assert loaded is not None
        assert loaded["queue"]["grace_seconds"] == 480
        assert loaded["kv"]["ram_cache_max_bytes"] == 25769803776

        stamp = load_runtime_provenance_stamp(state_db)
        assert stamp == {}, (
            "a field written before this code shipped must have NO stamp -- "
            f"there is nothing to infer it from, got {stamp}"
        )

    def test_a_field_never_put_has_no_stamp_entry_even_after_a_sibling_put(self, app_test):
        """Forward-only, precisely: PUTting queue.grace_seconds must not
        fabricate a stamp for queue.max_grace_extensions, which this
        request never touched -- absence must stay absence, not silently
        become 'stamped now' just because a sibling field in the same
        section was.

        Pre-seeds a field that predates this code (present on disk with no
        stamp, simulating a pre-stamp fragment) so an over-broad
        implementation that stamps every field IN THE MERGED SECTION
        (rather than just the ones this payload named) has something to
        wrongly touch -- a version of this test starting from a totally
        empty file can't tell "stamp only what's named" apart from "stamp
        the whole section" (checked directly: iterating existing[section]
        instead of sec_payload still passed against an empty-starting
        fixture, since the merged section then contains nothing else)."""
        import yaml as _yaml

        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        # pre-existing, unstamped sibling field -- written directly to the
        # raw file (NOT via save_runtime_config, which would itself stamp
        # it) to genuinely simulate data written before this code
        # shipped.
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(_yaml.safe_dump({"queue": {"safety_enabled": False}}))

        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        stamp = load_runtime_provenance_stamp(state_db)
        assert "grace_seconds" in stamp.get("queue", {}), stamp
        assert "safety_enabled" not in stamp.get("queue", {}), (
            f"a pre-existing sibling field got stamped by an unrelated PUT: {stamp}"
        )
        assert "max_grace_extensions" not in stamp.get("queue", {}), stamp

    def test_re_put_to_the_same_field_updates_the_timestamp(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        state_db = mgr.boot.storage.state_db_path
        r1 = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r1.status_code == 200
        first_ts = load_runtime_provenance_stamp(state_db)["queue"]["grace_seconds"]

        r2 = client.put("/api/config", json={"queue": {"grace_seconds": 88}})
        assert r2.status_code == 200
        second_ts = load_runtime_provenance_stamp(state_db)["queue"]["grace_seconds"]

        assert datetime.fromisoformat(second_ts) >= datetime.fromisoformat(first_ts)

    def test_get_config_surfaces_the_stamp_separately_from_provenance(self, app_test):
        """End-to-end through the real HTTP surface: GET /api/config's new
        _provenance_stamp key is a DISTINCT field from the existing
        _provenance key (which layer wins) -- both present, neither
        clobbers the other."""
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        body = client.get("/api/config").json()
        assert body["_provenance"]["queue"]["grace_seconds"] == "persisted"
        assert "grace_seconds" in body["_provenance_stamp"]["queue"]
        # untouched field: labelled by _provenance as usual, absent from the stamp
        assert body["_provenance"]["queue"]["max_grace_extensions"] == "default"
        assert "max_grace_extensions" not in body["_provenance_stamp"].get("queue", {})

    def test_precedence_and_effective_value_unaffected_by_the_stamp(self, app_test):
        """The hard constraint: precedence must be completely
        unchanged. A stamped field's EFFECTIVE value and _provenance label
        are identical to what an un-stamped (pre-stamp) persisted field
        already produced."""
        app, client = app_test
        r = client.put("/api/config", json={"queue": {"grace_seconds": 77}})
        assert r.status_code == 200
        body = client.get("/api/config").json()
        assert body["queue"]["grace_seconds"] == 77
        assert body["_provenance"]["queue"]["grace_seconds"] == "persisted"


class TestBootSectionTamperGuard:
    """Boot-section guard: a tampered runtime_config.yaml carrying boot sections
    (e.g. storage.llama_server_binary=/tmp/evil) must NOT override boot
    config at boot -- preserves binary-swap protection."""

    def test_tampered_boot_section_ignored_at_boot(self, tmp_path):
        """Simulate a tampered runtime_config.yaml with a boot section.
        The boot-merge in __main__ must skip it -- boot config unchanged."""
        import yaml
        from turbohaul.api.config_put import RUNTIME_SECTIONS

        state_db = tmp_path / "state.sqlite"
        state_db.parent.mkdir(parents=True, exist_ok=True)

        # Write a tampered override file with both a runtime section (valid)
        # and a boot section (should be ignored)
        override_path = state_db.parent / "runtime_config.yaml"
        override_path.write_text(yaml.safe_dump({
            "queue": {"grace_seconds": 99},
            "storage": {"blob_store_path": "/tmp/evil"},
        }))

        # load_runtime_override returns config sections unfiltered (it does
        # NOT guard against a tampered boot section here -- only strips the
        # _provenance_stamp bookkeeping key, which this file
        # doesn't have). The __main__.py boot merge's RUNTIME_SECTIONS check
        # is the actual guard, exercised below.
        override = load_runtime_override(state_db)
        assert override is not None
        assert override["queue"]["grace_seconds"] == 99
        assert override["storage"]["blob_store_path"] == "/tmp/evil"

        # The boot-section guard: only RUNTIME_SECTIONS are merged at boot
        # Simulate the __main__ boot-merge logic
        from turbohaul.config import BootConfig, RuntimeConfig, ServerConfig, StorageConfig, RuntimePathsConfig, UIConfig
        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=tmp_path / "blobs",
                manifests_path=tmp_path / "manifests",
                import_allowed_root=tmp_path / "import-staging",
                state_db_path=state_db,
            ),
            runtime=RuntimePathsConfig(
                llama_server_binary=tmp_path / "fake",
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
        from turbohaul.config import TurbohaulConfig
        cfg = TurbohaulConfig(
            server=boot.server,
            storage=boot.storage,
            runtime=boot.runtime,
            ui=boot.ui,
            queue=runtime.queue,
            pull=runtime.pull,
            persist=runtime.persist,
            monitor=runtime.monitor,
        )

        # Apply the boot-section guard merge (same logic as __main__.py)
        cfg_data = cfg.model_dump()
        for section, values in override.items():
            if section in RUNTIME_SECTIONS and section in cfg_data and isinstance(values, dict):
                cfg_data[section] = {**cfg_data[section], **values}

        # Boot sections must NOT be overridden
        assert cfg_data["storage"]["blob_store_path"] == tmp_path / "blobs"
        # Runtime sections ARE overridden
        assert cfg_data["queue"]["grace_seconds"] == 99
