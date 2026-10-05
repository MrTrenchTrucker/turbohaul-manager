"""Tests for the CLI entry point (src/turbohaul/__main__.py)."""
from __future__ import annotations

import logging
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from turbohaul.__main__ import build_parser, main


@pytest.fixture
def write_minimal_yaml(tmp_path: Path):
    """Yield a yaml path containing a valid TurbohaulConfig."""
    blob = tmp_path / "blobs"
    blob.mkdir()
    man = tmp_path / "manifests"
    man.mkdir()
    imp = tmp_path / "import-staging"
    imp.mkdir()
    ui = tmp_path / "ui_dist"
    ui.mkdir()
    binary = tmp_path / "fake_llama_server"
    binary.write_bytes(b"")

    yaml_text = f"""
server:
  host: 127.0.0.1
  port: 11401
  allow_public_bind: false
storage:
  blob_store_path: {blob}
  manifests_path: {man}
  import_allowed_root: {imp}
  state_db_path: {tmp_path / "state.sqlite"}
runtime:
  llama_server_binary: {binary}
  llama_server_binary_sha256: ""
  default_port_base: 11500
ui:
  enabled: true
  static_path: {ui}
queue: {{}}
pull: {{}}
"""
    path = tmp_path / "turbohaul.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


class TestParser:
    def test_default_config_path(self):
        args = build_parser().parse_args([])
        # Either the test env's TURBOHAUL_CONFIG_PATH OR the hardcoded default — both fine.
        assert args.config.suffix == ".yaml"

    def test_allow_public_bind_off_by_default(self, monkeypatch):
        monkeypatch.delenv("TURBOHAUL_ALLOW_PUBLIC_BIND", raising=False)
        args = build_parser().parse_args([])
        assert args.allow_public_bind is False

    def test_allow_public_bind_via_flag(self):
        args = build_parser().parse_args(["--allow-public-bind"])
        assert args.allow_public_bind is True

    def test_allow_public_bind_via_env(self, monkeypatch):
        monkeypatch.setenv("TURBOHAUL_ALLOW_PUBLIC_BIND", "1")
        args = build_parser().parse_args([])
        assert args.allow_public_bind is True


class TestMain:
    def test_main_missing_config_returns_2(self, tmp_path, capsys):
        rc = main(["--config", str(tmp_path / "nope.yaml")])
        assert rc == 2

    def test_main_loads_config_and_invokes_uvicorn(
        self, write_minimal_yaml, monkeypatch
    ):
        """main() must reach uvicorn.run with host from BootConfig (no public bind)."""
        called = {}

        def fake_run(app, **kwargs):
            called["app"] = app
            called["host"] = kwargs.get("host")
            called["port"] = kwargs.get("port")

        with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
            rc = main(["--config", str(write_minimal_yaml)])
        assert rc == 0
        assert called["host"] == "127.0.0.1"
        assert called["port"] == 11401

    def test_main_allow_public_bind_overrides_host(
        self, write_minimal_yaml
    ):
        """--allow-public-bind does not reach uvicorn.run at all: it binds its
        own dual-stack AF_INET6 socket and hands it to uvicorn.Server(...).run(
        sockets=...) instead. Patching only uvicorn.run here left that branch
        unmocked -- a real Server.run() would open the real event loop and
        serve forever, which would hang the whole suite."""
        fake_sock = MagicMock()
        fake_server = MagicMock()

        with patch("socket.socket", return_value=fake_sock) as mock_socket_cls, \
             patch("turbohaul.__main__.uvicorn.Server", return_value=fake_server) as mock_server_cls, \
             patch("turbohaul.__main__.uvicorn.run") as mock_run:
            rc = main([
                "--config", str(write_minimal_yaml),
                "--allow-public-bind",
            ])

        assert rc == 0
        mock_socket_cls.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM)
        fake_sock.setsockopt.assert_any_call(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        fake_sock.bind.assert_called_once_with(("::", 11401))
        mock_server_cls.assert_called_once()
        fake_server.run.assert_called_once_with(sockets=[fake_sock])
        mock_run.assert_not_called()

    def test_main_log_level_passed_to_uvicorn(self, write_minimal_yaml):
        called = {}

        def fake_run(app, **kwargs):
            called["log_level"] = kwargs.get("log_level")

        with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
            main([
                "--config", str(write_minimal_yaml),
                "--log-level", "debug",
            ])
        assert called["log_level"] == "debug"


def _boot_and_get_config(config_path: Path) -> dict:
    """Real boot path: run main() for real (uvicorn.run mocked, nothing
    else), capture the actual app object it builds, hit GET /api/config on
    it. This is what main() ACTUALLY builds, not a hand-constructed
    BootConfig/RuntimeConfig -- the construct-vs-derive distinction that
    matters for provenance checks, as a hand-built config can hide gaps."""
    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app

    with patch("turbohaul.__main__.uvicorn.run", side_effect=fake_run):
        rc = main(["--config", str(config_path)])
    assert rc == 0
    with TestClient(called["app"]) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200
    return resp.json()


class TestConfigProvenance:
    """GET /api/config's "_provenance" field, proven
    through the real boot path -- main() really loading a real
    runtime_config.yaml (the persisted-override mechanism), not a
    hand-built provenance dict passed straight to compute_config_provenance."""

    def test_env_layer_reported(self, write_minimal_yaml, monkeypatch):
        monkeypatch.setenv("TURBOHAUL_GRACE_S", "45")
        body = _boot_and_get_config(write_minimal_yaml)
        assert body["_provenance"]["queue"]["grace_seconds"] == "env"
        assert body["queue"]["grace_seconds"] == 45
        # Control, same request: a field neither in this minimal yaml nor
        # env-overridden must report "default", not "env" -- a provenance
        # function that always says "env" would fail this half.
        assert body["_provenance"]["queue"]["max_grace_extensions"] == "default"

    def test_persisted_layer_wins_over_env_for_the_same_key(
        self, write_minimal_yaml, monkeypatch
    ):
        """The load-bearing precedence claim itself: a persisted
        runtime_config.yaml value must win over a simultaneously-set env var
        for the SAME field, matching the real merge order in __main__.py.
        A provenance function that always says "persisted" would trivially
        pass this alone -- test_env_layer_reported above is the other half
        that rules that out (persisted absent there, "env" still wins)."""
        monkeypatch.setenv("TURBOHAUL_GRACE_S", "45")
        override_path = write_minimal_yaml.parent / "runtime_config.yaml"
        override_path.write_text("queue:\n grace_seconds: 99\n")

        body = _boot_and_get_config(write_minimal_yaml)
        assert body["_provenance"]["queue"]["grace_seconds"] == "persisted"
        assert body["queue"]["grace_seconds"] == 99  # persisted value, not env's 45

    def test_never_labels_a_boot_section_field_persisted(
        self, write_minimal_yaml
    ):
        """This is asserted in a test, not just in the code.
        A malformed/unexpected runtime_config.yaml containing a boot-section
        key (here: "server") must have ZERO effect on the real boot config
        (the boot-section guard in __main__.py already only merges
        RUNTIME_SECTIONS) AND must never be reported as "persisted" -- a
        fabricated provenance label would be worse than none, on an endpoint
        whose whole purpose is telling the truth about where a value came
        from."""
        override_path = write_minimal_yaml.parent / "runtime_config.yaml"
        override_path.write_text("server:\n port: 65000\n")

        body = _boot_and_get_config(write_minimal_yaml)
        assert body["server"]["port"] == 11401  # unchanged -- boot-section guard held
        assert body["_provenance"]["server"]["port"] == "yaml"  # never "persisted"

    def test_boot_and_runtime_share_the_word_runtime_without_collision(
        self, write_minimal_yaml
    ):
        """BOOT_SECTIONS' "runtime" (RuntimePathsConfig: binary path, sha256,
        port base) and the persisted-override mechanism's RuntimeConfig (the
        top-level container for queue/pull/persist/monitor/kv) share a name
        by pre-existing convention in this codebase -- confirm the boot
        "runtime" section's own fields are never treated as
        persisted-overridable because of that name collision."""
        override_path = write_minimal_yaml.parent / "runtime_config.yaml"
        override_path.write_text("runtime:\n default_port_base: 1\n")

        body = _boot_and_get_config(write_minimal_yaml)
        assert body["runtime"]["default_port_base"] == 11500  # unchanged
        assert body["_provenance"]["runtime"]["default_port_base"] == "yaml"


class TestConfigDivergenceLog:
    """The startup log line naming every key whose
    effective value differs from the shipped default."""

    def test_diverged_key_is_logged(self, write_minimal_yaml, monkeypatch, caplog):
        monkeypatch.setenv("TURBOHAUL_GRACE_S", "45")
        with patch("turbohaul.__main__.uvicorn.run"):
            with caplog.at_level("INFO", logger="turbohaul.main"):
                main(["--config", str(write_minimal_yaml)])
        diverge_lines = [
            r.message for r in caplog.records if "config diverges from shipped default" in r.message
        ]
        assert any("queue.grace_seconds" in m for m in diverge_lines)

    def test_undiverged_key_is_not_logged(self, write_minimal_yaml, caplog):
        # No env override, no persisted override -- write_minimal_yaml's
        # queue.grace_seconds is absent from its yaml too, so the effective
        # value is QueueConfig's own code default and must NOT be logged as
        # a divergence. A log line that fires unconditionally for every key
        # (not just diverged ones) would fail this half.
        with patch("turbohaul.__main__.uvicorn.run"):
            with caplog.at_level("INFO", logger="turbohaul.main"):
                main(["--config", str(write_minimal_yaml)])
        diverge_lines = [
            r.message for r in caplog.records if "config diverges from shipped default" in r.message
        ]
        assert not any("queue.grace_seconds" in m for m in diverge_lines)
        assert not any("queue.max_grace_extensions" in m for m in diverge_lines)


class TestSparsePersistenceSurvivesRealRestart:
    """The actual claim durability rests on -- a PUT
    survives restart -- proven through TWO independent, real main() boots
    against the SAME state_db_path, not by calling save_runtime_config
    directly. Boot 1 PUTs a change (writing the real runtime_config.yaml to
    disk); boot 2 is a completely fresh main() invocation, simulating a real
    process restart, and must pick the sparse override back up correctly
    while leaving every untouched field on whichever layer it was already on.
    """

    def test_sparse_override_survives_a_real_second_boot(self, write_minimal_yaml, monkeypatch):
        monkeypatch.setenv("TURBOHAUL_MAX_GRACE_EXT", "3")

        # Boot 1: PUT one field, let the TestClient context close cleanly
        # (shuts down boot 1's worker/lifespan) before boot 2 starts.
        called1 = {}
        with patch("turbohaul.__main__.uvicorn.run", side_effect=lambda app, **kw: called1.__setitem__("app", app)):
            rc1 = main(["--config", str(write_minimal_yaml)])
        assert rc1 == 0
        with TestClient(called1["app"]) as client1:
            r = client1.put("/api/config", json={"queue": {"grace_seconds": 77}})
            assert r.status_code == 200

        # Boot 2: a completely independent main() call, same --config path
        # (same state_db_path, so the same runtime_config.yaml on disk).
        body = _boot_and_get_config(write_minimal_yaml)

        # The PUT from boot 1 survived the restart, sparse: persisted.
        assert body["queue"]["grace_seconds"] == 77
        assert body["_provenance"]["queue"]["grace_seconds"] == "persisted"
        # A field this fix must NOT freeze just because grace_seconds
        # was touched: the env var set above must still win on boot 2 -- if
        # the old wholesale-dump behaviour were still in effect, boot 1's
        # save would have captured whatever max_grace_extensions resolved to
        # AT THAT MOMENT and pinned it, making this "persisted" instead.
        assert body["queue"]["max_grace_extensions"] == 3
        assert body["_provenance"]["queue"]["max_grace_extensions"] == "env"
        # And a field with no env var and absent from the yaml stays on
        # code default, also un-pinned by the unrelated PUT.
        assert body["_provenance"]["queue"]["staging_queue_depth"] == "default"


class TestFastLaneBootLint:
    """The boot-path lint runs on the FINAL
    merged runtime.fastlane.rules, right after cfg.split() -- not inside the
    persisted-override merge branch (an earlier placement anchored
    there), which only fires when a runtime_config.yaml override touches
    "fastlane". A duplicate baked into the SHIPPED yaml, with no runtime
    override at all, is exactly the case that placement would have missed and
    this placement catches."""

    def _write_yaml_with_fastlane(self, tmp_path: Path, fastlane_block: str) -> Path:
        blob = tmp_path / "blobs"
        blob.mkdir()
        man = tmp_path / "manifests"
        man.mkdir()
        imp = tmp_path / "import-staging"
        imp.mkdir()
        ui = tmp_path / "ui_dist"
        ui.mkdir()
        binary = tmp_path / "fake_llama_server"
        binary.write_bytes(b"")

        yaml_text = f"""
server:
  host: 127.0.0.1
  port: 11401
  allow_public_bind: false
storage:
  blob_store_path: {blob}
  manifests_path: {man}
  import_allowed_root: {imp}
  state_db_path: {tmp_path / "state.sqlite"}
runtime:
  llama_server_binary: {binary}
  llama_server_binary_sha256: ""
  default_port_base: 11500
ui:
  enabled: true
  static_path: {ui}
queue: {{}}
pull: {{}}
{fastlane_block}
"""
        path = tmp_path / "turbohaul.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        return path

    def test_boots_with_shipped_yaml_fastlane_duplicate_and_logs_warning(self, tmp_path, caplog):
        # No runtime_config.yaml override anywhere -- the duplicate exists
        # ONLY in the shipped yaml. The earlier placement is inside
        # `if runtime_override:` and would never even run here.
        path = self._write_yaml_with_fastlane(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "192.0.2.10"
      tag_ranks: {main: 1}
    - address: "192.0.2.10"
      tag_ranks: {curator: 2}
""")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True  # boot succeeded (rc == 0 asserted inside)
        assert any(
            "duplicate" in rec.message and rec.levelname == "WARNING"
            for rec in caplog.records
        )

    def test_boot_lint_raising_does_not_kill_boot(self, tmp_path, caplog):
        """the boot-path lint_rules call at
        __main__.py sits inside a bare try/except Exception, and stays
        that way (census is deliberately NOT threaded there: it
        is memory-only and empty at boot, so threading it would fire five
        false "never observed" warnings on every single boot). Force the
        call itself to raise and prove startup still completes."""
        path = self._write_yaml_with_fastlane(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "192.0.2.10"
      tag_ranks: {main: 1}
""")
        with patch(
            "turbohaul.__main__.lint_rules",
            side_effect=RuntimeError("boom"),
        ) as mock_lint, caplog.at_level(logging.ERROR):
            body = _boot_and_get_config(path)  # asserts rc == 0 internally
        assert body["fastlane"]["enabled"] is True  # boot completed despite the raise
        assert any(
            "fastlane config lint failed at boot" in rec.message
            and rec.levelname == "ERROR"
            for rec in caplog.records
        )
        # STRUCTURAL-ONLY, confirmed: called with the rule list alone, no
        # census kwarg -- threading a live census here is explicitly WRONG
        # by design (empty-at-boot census would false-positive).
        mock_lint.assert_called_once()
        assert mock_lint.call_args.kwargs == {}
        assert len(mock_lint.call_args.args) == 1

    def test_boots_with_no_shipped_yaml_duplicate_no_warning(self, tmp_path, caplog):
        path = self._write_yaml_with_fastlane(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "192.0.2.10"
      tag_ranks: {main: 1}
    - address: "192.0.2.11"
      tag_ranks: {curator: 2}
""")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert not any("duplicate" in rec.message for rec in caplog.records)

    def test_boots_with_no_fastlane_section_at_all_no_crash(self, tmp_path, caplog):
        # write_minimal_yaml-equivalent (no fastlane key) must still boot
        # clean -- lint_rules([]) on the default empty rule list.
        path = self._write_yaml_with_fastlane(tmp_path, "")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is False
        assert not any("duplicate" in rec.message for rec in caplog.records)

    # ========================================================================
    # Fast Lane rename: the legacy-key migration. TurbohaulConfig
    # has model_config=ConfigDict(extra="forbid"), so a shipped/boot yaml
    # still carrying the OLD top-level "fastline:" key would, with no
    # migration, reject the whole document and crash boot entirely -- not
    # just drop the feature. These prove that does NOT happen, at BOTH real
    # disk-read sites, with a realistic shape (2 rules,
    # 4 tag ranks each -- shrunk to 1 rank/rule here for test brevity, same
    # structural shape).
    # ========================================================================

    def test_boots_with_legacy_fastline_key_in_shipped_yaml_loads_rules(self, tmp_path):
        # The exact failure mode the migration exists to prevent: pre-migration,
        # this yaml (old key, new code) would hit extra="forbid" and main()
        # would never even reach rc == 0.
        path = self._write_yaml_with_fastlane(tmp_path, """
fastline:
  enabled: true
  rules:
    - address: "fdd0:0:0:1f::3"
      label: "Agent service"
      tag_ranks: {main: 1}
    - address: "fdd0:0:0:1f::2"
      label: "Gateway"
      tag_ranks: {main: 1}
""")
        body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert [r["address"] for r in body["fastlane"]["rules"]] == [
            "fdd0:0:0:1f::3", "fdd0:0:0:1f::2",
        ]
        # Old key never leaks into the effective response -- migrated, not duplicated.
        assert "fastline" not in body

    def test_boots_with_legacy_fastline_key_in_runtime_override_loads_rules(self, tmp_path):
        # The key scenario: the PERSISTED runtime_config.yaml
        # (written by PUT /api/config on a pre-rename build) still carries the
        # real rules under "fastline". No shipped-yaml duplicate
        # involved at all -- this is the __main__.py runtime-override merge
        # path, a DIFFERENT code path from the shipped-yaml test above.
        path = self._write_yaml_with_fastlane(tmp_path, "")  # no fastlane/fastline in shipped yaml
        override_path = tmp_path / "runtime_config.yaml"
        override_path.write_text(
            "fastline:\n"
            "  enabled: true\n"
            "  rules:\n"
            "    - address: \"fdd0:0:0:1f::3\"\n"
            "      label: \"Agent service\"\n"
            "      tag_ranks: {main: 1}\n"
            "    - address: \"fdd0:0:0:1f::2\"\n"
            "      label: \"Gateway\"\n"
            "      tag_ranks: {main: 1}\n",
            encoding="utf-8",
        )
        body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert [r["address"] for r in body["fastlane"]["rules"]] == [
            "fdd0:0:0:1f::3", "fdd0:0:0:1f::2",
        ]
        assert body["_provenance"]["fastlane"]["enabled"] == "persisted"
        assert "fastline" not in body

    def test_explicit_fastlane_key_wins_over_legacy_fastline_key(self, tmp_path):
        # Both keys present in the SAME shipped yaml (pathological, but the
        # migration's own contract -- new key wins, old key is a fallback
        # only -- needs a real assertion, not just "doesn't crash".
        path = self._write_yaml_with_fastlane(tmp_path, """
fastline:
  enabled: false
  rules: []
fastlane:
  enabled: true
  rules:
    - address: "192.0.2.50"
      tag_ranks: {main: 1}
""")
        body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert [r["address"] for r in body["fastlane"]["rules"]] == ["192.0.2.50"]


class TestFastLaneBootSalvage:
    """Gate: pydantic rejects the SECTION, not the rule, so one typo'd
    address would flip enabled:true to the code default False and take
    every good rule with it. Coerce-and-warn instead -- and NEVER by removing
    the rule, because rule_index is positional.
    """

    def _candidate(self):
        return {
            "enabled": True,
            "rules": [
                {"address": "gateway-svc", "label": "Gateway"},        # BAD: not an IP
                {"address": "10.0.0.3", "label": "Agent service"},
                {"address": "10.0.0.4", "label": "Advisor"},
                {"address": "10.0.0.5", "label": "Web UI"},
            ],
        }

    def test_one_bad_rule_does_not_disable_the_feature(self):
        from turbohaul.__main__ import _salvage_fastlane_rules

        salvaged, notes = _salvage_fastlane_rules(self._candidate())

        assert salvaged is not None, "a single bad rule disabled the whole section"
        assert salvaged["enabled"] is True
        assert len(salvaged["rules"]) == 4, "the rule list must not be shortened"
        assert any("not a single IP" in n for n in notes), notes

    def test_a_bad_rule_at_index_0_does_NOT_re_rank_the_rules_below_it(self):
        """⛔ The hazard: rule_index IS the position in the list
        handed to compile_fastlane. Dropping the bad rule at [0] would promote
        all three rules below it by one slot -- a silent PRIORITY change
        caused by a validation error, which is the exact defect class this
        mitigation exists to avoid.
        """
        from turbohaul.__main__ import _salvage_fastlane_rules
        from turbohaul.config import FastLaneConfig
        from turbohaul.fastlane import compile_fastlane

        salvaged, _ = _salvage_fastlane_rules(self._candidate())
        cfg = FastLaneConfig(**salvaged)
        compiled = compile_fastlane(cfg.rules)

        by_label = {c.label: c.index for c in compiled}
        # the three GOOD rules keep the slots they had in the original file
        assert by_label["Agent service"] == 1, by_label
        assert by_label["Advisor"] == 2, by_label
        assert by_label["Web UI"] == 3, by_label

    def test_CONTROL_removing_the_bad_rule_WOULD_re_rank_them(self):
        """The control for the arm above. Without it, that assertion could
        pass for a salvage that never had to preserve anything. This shows the
        promotion is real: drop the bad rule and every label moves up a slot.
        """
        from turbohaul.config import FastLaneConfig
        from turbohaul.fastlane import compile_fastlane

        shortened = {"enabled": True, "rules": self._candidate()["rules"][1:]}
        compiled = compile_fastlane(FastLaneConfig(**shortened).rules)
        by_label = {c.label: c.index for c in compiled}

        assert by_label["Agent service"] == 0, by_label   # was 1 -- PROMOTED
        assert by_label["Web UI"] == 2, by_label    # was 3 -- PROMOTED

    def test_a_blank_identity_block_is_salvaged_not_disabled(self):
        """A rule naming NEITHER field is exactly the blank-identity case
        the salvage fix handles -- even as the ONLY rule in the section, salvaging it
        (rather than bailing) preserves the operator's `enabled` flag and
        scheduling settings, which bailing the whole section would throw
        away too. An unsalvageable-block expectation (`salvaged is None`)
        does not apply here.
        See test_a_rule_setting_BOTH_fields_is_still_unsalvageable
        below for an example this fix deliberately leaves unsalvageable."""
        from turbohaul.__main__ import _salvage_fastlane_rules

        salvaged, notes = _salvage_fastlane_rules(
            {"enabled": True, "rules": [{"label": "neither field set"}]}
        )
        assert salvaged is not None
        assert salvaged["enabled"] is True
        assert len(salvaged["rules"]) == 1
        assert salvaged["rules"][0]["container_name"] == "__unsalvageable_fastlane_rule_0__"
        assert any("replaced with an inert placeholder" in n for n in notes), notes

    def test_a_rule_setting_BOTH_fields_is_still_unsalvageable(self):
        """The control for the rewrite above: a rule with a REAL, non-blank
        value in BOTH fields is ambiguous operator intent, not an empty one --
        a different failure class the blank-identity salvage does not touch. Still bails the
        whole section, unchanged."""
        from turbohaul.__main__ import _salvage_fastlane_rules

        salvaged, notes = _salvage_fastlane_rules(
            {"enabled": True, "rules": [
                {"address": "10.0.0.5", "container_name": "worker-x"},
            ]}
        )
        assert salvaged is None
        assert notes == []


class TestFastLaneRulesCapSalvage:
    """A persisted block that has grown past the rule cap must not disable
    the whole feature -- same "a scheduling preference must never take the
    manager down" precedent as TestFastLaneBootSalvage above, applied to a
    length problem instead of a per-rule content problem."""

    def _rules(self, n: int) -> list[dict]:
        return [
            {"address": f"10.0.0.{i}", "label": f"client-{i}"} for i in range(n)
        ]

    def test_at_the_cap_is_untouched(self):
        from turbohaul.__main__ import _cap_fastlane_rules

        candidate = {"enabled": True, "rules": self._rules(10)}
        assert _cap_fastlane_rules(candidate) is candidate

    def test_over_the_cap_keeps_the_first_ten_in_order(self):
        from turbohaul.__main__ import _cap_fastlane_rules

        candidate = {"enabled": True, "rules": self._rules(13)}
        capped = _cap_fastlane_rules(candidate)
        assert [r["label"] for r in capped["rules"]] == [
            f"client-{i}" for i in range(10)
        ]

    def test_over_the_cap_logs_a_warning_naming_count_and_cap(self, caplog):
        from turbohaul.__main__ import _cap_fastlane_rules

        with caplog.at_level(logging.WARNING):
            _cap_fastlane_rules({"enabled": True, "rules": self._rules(13)})
        assert any(
            "13" in rec.message and "10" in rec.message and rec.levelname == "WARNING"
            for rec in caplog.records
        ), caplog.records

    def test_real_boot_with_thirteen_persisted_rules_stays_enabled_with_ten(
        self, write_minimal_yaml, monkeypatch, caplog,
    ):
        """End to end through main(): the exact scenario the cap salvage
        exists for -- an operator-edited runtime_config.yaml with more rules
        than the cap allows must still boot with fastlane ON, not silently
        drop to disabled the way an un-truncated over-length list would
        (every individual rule here is well-formed; only the count is bad)."""
        import yaml as _yaml

        from turbohaul.api.config_put import _runtime_override_path

        state_db = write_minimal_yaml.parent / "state.sqlite"
        override_path = _runtime_override_path(state_db)
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(_yaml.safe_dump({
            "fastlane": {
                "enabled": True,
                "rules": self._rules(13),
            },
        }))

        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(write_minimal_yaml)

        assert body["fastlane"]["enabled"] is True
        assert len(body["fastlane"]["rules"]) == 10
        assert [r["label"] for r in body["fastlane"]["rules"]] == [
            f"client-{i}" for i in range(10)
        ]
        assert any(
            "13" in rec.message and "10" in rec.message and rec.levelname == "WARNING"
            for rec in caplog.records
        ), caplog.records


class TestFastLaneMaxWaitDefaultBootLog:
    """A boot log when max_normal_wait_s's provenance is exactly 'default' --
    the operator-visibility half of the default value moving to 3600s."""

    def test_default_provenance_logs_info(self, write_minimal_yaml, caplog):
        with caplog.at_level(logging.INFO):
            body = _boot_and_get_config(write_minimal_yaml)
        assert body["_provenance"]["fastlane"]["max_normal_wait_s"] == "default"
        assert any(
            "max_normal_wait_s" in rec.message and rec.levelname == "INFO"
            for rec in caplog.records
        ), caplog.records

    def test_yaml_override_suppresses_the_log(self, tmp_path, caplog):
        """CONTROL: an explicit shipped-yaml value is a different provenance
        ('yaml', not 'default') and must not trip the same-value coincidence
        (3600 written explicitly still means someone chose it)."""
        blob = tmp_path / "blobs"; blob.mkdir()
        man = tmp_path / "manifests"; man.mkdir()
        imp = tmp_path / "import-staging"; imp.mkdir()
        ui = tmp_path / "ui_dist"; ui.mkdir()
        binary = tmp_path / "fake_llama_server"
        binary.write_bytes(b"")
        yaml_text = f"""
server:
  host: 127.0.0.1
  port: 11401
  allow_public_bind: false
storage:
  blob_store_path: {blob}
  manifests_path: {man}
  import_allowed_root: {imp}
  state_db_path: {tmp_path / "state.sqlite"}
runtime:
  llama_server_binary: {binary}
  llama_server_binary_sha256: ""
  default_port_base: 11500
ui:
  enabled: true
  static_path: {ui}
queue: {{}}
pull: {{}}
fastlane:
  enabled: false
  rules: []
  max_normal_wait_s: 3600
"""
        path = tmp_path / "turbohaul.yaml"
        path.write_text(yaml_text, encoding="utf-8")

        with caplog.at_level(logging.INFO):
            body = _boot_and_get_config(path)
        assert body["_provenance"]["fastlane"]["max_normal_wait_s"] == "yaml"
        assert not any(
            "max_normal_wait_s" in rec.message and rec.levelname == "INFO"
            for rec in caplog.records
        ), caplog.records


class TestIsFastlaneOnlyErrorGuard:
    """`_is_fastlane_only_error` directly, on a REAL
    mixed pydantic ValidationError -- not through the full boot pipeline.

    The pre-existing end-to-end control,
    TestFastLaneInTheBootYamlCannotKillTheManager::
    test_CONTROL_a_NON_fastlane_error_still_fails_loudly, cannot fail no
    matter what this guard does: give it a MIXED error and the salvage body
    two frames below reconstructs TurbohaulConfig with the still-broken
    `port` field present either way (stripped-fastlane branch or
    salvaged-fastlane branch), so the SAME ValidationError resurfaces and the
    test's `pytest.raises(ValidationError)` passes regardless of whether the
    guard correctly says "not fastlane-only" or is completely broken.
    PROVEN, not asserted: with `_is_fastlane_only_error` mutated to
    unconditionally `return True`, that entire test class -- all 6 tests,
    including the control -- still passes. This class tests the guard
    directly so that such a regression cannot go unnoticed.
    """

    def test_a_MIXED_fastlane_and_non_fastlane_error_is_NOT_fastlane_only(
        self, write_minimal_yaml
    ):
        import yaml
        from turbohaul.__main__ import _is_fastlane_only_error
        from turbohaul.config import TurbohaulConfig

        data = yaml.safe_load(write_minimal_yaml.read_text())
        data["server"]["port"] = "not-a-port"
        data["fastlane"] = {"enabled": True, "rules": [{"label": "neither field set"}]}
        with pytest.raises(ValidationError) as exc_info:
            TurbohaulConfig(**data)
        assert _is_fastlane_only_error(exc_info.value) is False, (
            "a mixed error (server.port AND fastlane both broken) must NOT be "
            "treated as fastlane-only -- the whole point of this guard is to "
            "keep boot config (which has no salvage) failing loudly"
        )

    def test_CONTROL_a_fastlane_only_error_WITH_MULTIPLE_complaints_IS_fastlane_only(
        self, write_minimal_yaml
    ):
        """The positive control: several DIFFERENT fastlane complaints at
        once, nothing else wrong. Two complaints rather than one so a guard
        that only checks `errors()[0]` (instead of `all(...)`) cannot pass
        this by accident."""
        import yaml
        from turbohaul.__main__ import _is_fastlane_only_error
        from turbohaul.config import TurbohaulConfig

        data = yaml.safe_load(write_minimal_yaml.read_text())
        data["fastlane"] = {
            "enabled": True,
            "rules": [
                {"label": "neither field set"},
                {"address": "10.0.0.5", "container_name": "worker-x"},
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            TurbohaulConfig(**data)
        errs = exc_info.value.errors()
        assert len(errs) >= 2, "need a genuinely multi-complaint error for this control to mean anything"
        assert _is_fastlane_only_error(exc_info.value) is True

    def test_CONTROL_a_pure_non_fastlane_error_is_NOT_fastlane_only(self, write_minimal_yaml):
        """The other positive control: nothing fastlane-related is even
        present, so `errors()` has zero fastlane-location entries at all --
        `all()` over an empty-of-fastlane-locs-but-nonempty list must still
        read False, not vacuously True."""
        import yaml
        from turbohaul.__main__ import _is_fastlane_only_error
        from turbohaul.config import TurbohaulConfig

        data = yaml.safe_load(write_minimal_yaml.read_text())
        data["server"]["port"] = "not-a-port"
        with pytest.raises(ValidationError) as exc_info:
            TurbohaulConfig(**data)
        assert _is_fastlane_only_error(exc_info.value) is False


class TestFastLaneInTheBootYamlCannotKillTheManager:
    """The shipped-yaml half of the cap-and-salvage guarantee.

    TestFastLaneRulesCapSalvage covers the PERSISTED file (runtime_config.yaml,
    written by PUT). The operator's own bind-mounted turbohaul.yaml took a
    different route into the same validator and had no guard at all, so content
    the persisted path capped, coerced and served instead raised out of main()
    and exited the process. With compose's `restart: unless-stopped` and the
    config mounted read-only, that is a permanent crash loop nothing inside the
    container can fix -- over a scheduling preference.

    The boot yaml is also the likelier of the two to be wrong: it is typed by
    hand, and PUT /api/config rejects both of these shapes at the door, so no
    UI ever produces them.
    """

    def _write(self, tmp_path, block):
        # Reuse the sibling class's writer on purpose rather than copying the
        # template: if the minimal-boot yaml gains a required section, both
        # suites move together instead of one rotting quietly.
        return TestFastLaneBootLint()._write_yaml_with_fastlane(tmp_path, block)

    def test_a_bare_container_name_in_the_address_field_still_boots(self, tmp_path, caplog):
        """The natural operator mistake: container_name is new, `address` is
        the field that already existed, so the name gets typed into it."""
        path = self._write(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "gateway-svc"
      tag_ranks: {main: 1}
""")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert len(body["fastlane"]["rules"]) == 1
        assert body["fastlane"]["rules"][0]["container_name"] == "gateway-svc"
        assert any("coerced at boot" in rec.message for rec in caplog.records), caplog.records

    def test_an_eleventh_well_formed_rule_still_boots(self, tmp_path, caplog):
        """Every rule is valid; only the COUNT is over. The persisted path
        truncates and keeps serving, so this one must too."""
        rules = "\n".join(
            f'    - address: "192.0.2.{i}"\n      tag_ranks: {{main: 1}}'
            for i in range(1, 12)
        )
        path = self._write(tmp_path, f"fastlane:\n  enabled: true\n  rules:\n{rules}\n")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert len(body["fastlane"]["rules"]) == 10, "expected truncation to the rule cap"

    def test_fastlane_DISABLED_with_a_bad_rule_still_boots(self, tmp_path, caplog):
        """`enabled: false` did not save you: both validators run at model
        construction, so an operator who had switched the feature OFF and left
        a stale rule behind still got a dead manager."""
        path = self._write(tmp_path, """
fastlane:
  enabled: false
  rules:
    - address: "gateway-svc"
      tag_ranks: {main: 1}
""")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is False

    def test_a_blank_identity_rule_boots_WITH_fastlane_still_on_not_disabled(
        self, tmp_path, caplog
    ):
        """A rule naming NEITHER an address nor
        a container is salvaged rather than taking the WHOLE section down
        with it -- `enabled: true` is not flipped to the code default False, and any
        valid sibling rule (there are none in THIS yaml, see the sibling test
        below for that half) is kept too. That is precisely the "one blank
        identity costs every valid sibling rule" defect this guards against --
        an assertion of `enabled is False` here would encode that bug,
        not a requirement to preserve. `enabled` and
        the operator's other Fast Lane settings survive untouched; only
        this ONE rule is replaced in place with an inert placeholder that
        matches nothing, logged and shown in the Fast Lane tab (see the
        test class below for the sibling-preserving case and the
        genuinely-still-unsalvageable case, which still bails the section).
        """
        path = self._write(tmp_path, """
fastlane:
  enabled: true
  rules:
    - tag_ranks: {main: 1}
""")
        with caplog.at_level(logging.WARNING):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert len(body["fastlane"]["rules"]) == 1
        assert body["fastlane"]["rules"][0]["container_name"] == "__unsalvageable_fastlane_rule_0__"
        assert any(
            "replaced with an inert placeholder" in rec.message for rec in caplog.records
        ), caplog.records

    def test_a_rule_setting_BOTH_fields_is_STILL_unsalvageable_boots_off(
        self, tmp_path, caplog
    ):
        """The replacement above is scoped to BLANK identity only.
        A rule setting BOTH address and container_name to real, non-blank
        values is a different failure class -- ambiguous operator intent, not
        an empty one -- and still bails the whole section exactly as before
        the blank-identity salvage. This is what keeps
        `_load_boot_config_with_fastlane_salvage`'s "still genuinely
        unsalvageable -> boot with fastlane off rather than not at all"
        fallback path under live coverage now that the blank-identity example
        that used to exercise it is salvaged instead."""
        path = self._write(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "10.0.0.5"
      container_name: "worker-x"
      tag_ranks: {main: 1}
""")
        with caplog.at_level(logging.ERROR):
            body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is False
        assert body["fastlane"]["rules"] == []
        assert any(
            "could not be salvaged" in rec.message for rec in caplog.records
        ), caplog.records

    def test_CONTROL_a_well_formed_boot_yaml_is_untouched(self, tmp_path):
        """Proves the salvage path is not silently rewriting healthy configs --
        and that the assertions above can distinguish salvaged from normal."""
        path = self._write(tmp_path, """
fastlane:
  enabled: true
  rules:
    - address: "192.0.2.10"
      tag_ranks: {main: 1}
""")
        body = _boot_and_get_config(path)
        assert body["fastlane"]["enabled"] is True
        assert len(body["fastlane"]["rules"]) == 1
        assert body["fastlane"]["rules"][0]["address"] == "192.0.2.10"
        assert body["fastlane"]["rules"][0]["container_name"] is None

    def test_CONTROL_a_NON_fastlane_error_still_fails_loudly(self, tmp_path):
        """The salvage must stay all-or-nothing. Boot config has no safe
        default, so a broken `server` section must still abort -- otherwise
        this fix would have quietly become a general 'boot anyway' path."""
        path = self._write(tmp_path, "fastlane:\n  enabled: true\n  rules: []\n")
        path.write_text(
            path.read_text().replace("port: 11401", "port: not-a-port"), encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            _boot_and_get_config(path)
