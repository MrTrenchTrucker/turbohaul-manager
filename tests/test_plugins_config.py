"""Tests for PluginsConfig / PluginEndpoint / PluginRuntimeConfig.

PluginsConfig is the ONLY place a host/port for an external plugin container may
live -- it exists so a manifest never carries a URL. Every negative test here
asserts the SPECIFIC validation error (the exact field + the exact rule that
fired), not just "raises", so a wrong-reason false-green can't hide.
"""
import pytest
from pydantic import ValidationError

from turbohaul.config import (
    BootConfig,
    PluginEndpoint,
    PluginRuntimeConfig,
    PluginsConfig,
    RuntimeConfig,
)


def _errors_for(exc_info, loc):
    """All pydantic error dicts whose loc starts with `loc` (a tuple)."""
    return [e for e in exc_info.value.errors() if e["loc"][: len(loc)] == loc]


class TestPluginEndpointValid:
    def test_valid_entry_loads(self):
        ep = PluginEndpoint(host="whisperx.internal", port=8080)
        assert ep.host == "whisperx.internal"
        assert ep.port == 8080
        assert ep.health_path == "/health"  # default

    def test_valid_entry_with_explicit_health_path(self):
        ep = PluginEndpoint(host="10.0.0.5", port=9000, health_path="/healthz")
        assert ep.health_path == "/healthz"

    def test_frozen(self):
        ep = PluginEndpoint(host="whisperx.internal", port=8080)
        with pytest.raises(ValidationError):
            ep.port = 9999


class TestPluginEndpointHostRejectsUrls:
    def test_host_with_scheme_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="http://whisperx.internal", port=8080)
        errs = _errors_for(exc_info, ("host",))
        assert errs, f"expected a host-field error, got: {exc_info.value.errors()}"
        assert "URL" in errs[0]["msg"] or "://" in errs[0]["msg"]

    def test_host_with_slash_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal/api", port=8080)
        errs = _errors_for(exc_info, ("host",))
        assert errs, f"expected a host-field error, got: {exc_info.value.errors()}"
        assert "'/'" in errs[0]["msg"]

    def test_host_with_query_string_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal?x=1", port=8080)
        errs = _errors_for(exc_info, ("host",))
        assert errs and "'?'" in errs[0]["msg"]

    def test_host_with_userinfo_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="user@whisperx.internal", port=8080)
        errs = _errors_for(exc_info, ("host",))
        assert errs and "'@'" in errs[0]["msg"]


class TestPluginEndpointPortBounds:
    def test_port_zero_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal", port=0)
        errs = _errors_for(exc_info, ("port",))
        assert errs, f"expected a port-field error, got: {exc_info.value.errors()}"
        assert errs[0]["type"] == "greater_than_equal"

    def test_port_too_high_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal", port=70000)
        errs = _errors_for(exc_info, ("port",))
        assert errs, f"expected a port-field error, got: {exc_info.value.errors()}"
        assert errs[0]["type"] == "less_than_equal"

    def test_port_boundaries_accepted(self):
        assert PluginEndpoint(host="h", port=1).port == 1
        assert PluginEndpoint(host="h", port=65535).port == 65535


class TestPluginEndpointHealthPathShape:
    def test_health_path_not_starting_with_slash_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal", port=8080, health_path="health")
        errs = _errors_for(exc_info, ("health_path",))
        assert errs, f"expected a health_path-field error, got: {exc_info.value.errors()}"
        assert "must start with '/'" in errs[0]["msg"]

    def test_health_path_with_dotdot_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="whisperx.internal", port=8080, health_path="/../etc/passwd")
        errs = _errors_for(exc_info, ("health_path",))
        assert errs, f"expected a health_path-field error, got: {exc_info.value.errors()}"
        assert "must not contain '..'" in errs[0]["msg"]


class TestPluginEndpointExtraForbid:
    def test_endpoint_url_field_rejected(self):
        """The exact forbidden shape the whole feature exists to prevent: an
        attempt to smuggle a URL field onto the endpoint model directly."""
        with pytest.raises(ValidationError) as exc_info:
            PluginEndpoint(host="h", port=1, endpoint_url="http://evil.example/")
        errs = [e for e in exc_info.value.errors() if e["type"] == "extra_forbidden"]
        assert errs, f"expected an extra_forbidden error, got: {exc_info.value.errors()}"


class TestPluginsConfigRegistry:
    def test_valid_registry_entry_loads(self):
        cfg = PluginsConfig(registry={"whisperx": {"host": "whisperx.internal", "port": 8080}})
        assert "whisperx" in cfg.registry
        assert isinstance(cfg.registry["whisperx"], PluginEndpoint)
        assert cfg.registry["whisperx"].port == 8080

    def test_empty_registry_is_default(self):
        """An existing deployment with no plugins must still boot."""
        cfg = PluginsConfig()
        assert cfg.registry == {}

    def test_key_with_uppercase_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginsConfig(registry={"WhisperX": {"host": "h", "port": 1}})
        errs = _errors_for(exc_info, ("registry",))
        assert errs, f"expected a registry-field error, got: {exc_info.value.errors()}"
        assert "WhisperX" in errs[0]["msg"]

    def test_key_with_slash_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginsConfig(registry={"whisper/x": {"host": "h", "port": 1}})
        errs = _errors_for(exc_info, ("registry",))
        assert errs, f"expected a registry-field error, got: {exc_info.value.errors()}"
        assert "whisper/x" in errs[0]["msg"]

    def test_key_with_leading_dash_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginsConfig(registry={"-whisperx": {"host": "h", "port": 1}})
        errs = _errors_for(exc_info, ("registry",))
        assert errs, f"expected a registry-field error, got: {exc_info.value.errors()}"
        assert "-whisperx" in errs[0]["msg"]

    def test_frozen(self):
        cfg = PluginsConfig()
        with pytest.raises(ValidationError):
            cfg.registry = {"x": {"host": "h", "port": 1}}


class TestPluginRuntimeConfig:
    def test_defaults_instantiate_with_empty_enabled(self):
        cfg = PluginRuntimeConfig()
        assert cfg.enabled == {}
        assert cfg.max_concurrent == 4
        assert cfg.no_progress_timeout_s == 600.0

    def test_max_concurrent_bounds(self):
        with pytest.raises(ValidationError) as exc_info:
            PluginRuntimeConfig(max_concurrent=0)
        errs = _errors_for(exc_info, ("max_concurrent",))
        assert errs and errs[0]["type"] == "greater_than_equal"
        with pytest.raises(ValidationError) as exc_info2:
            PluginRuntimeConfig(max_concurrent=65)
        errs2 = _errors_for(exc_info2, ("max_concurrent",))
        assert errs2 and errs2[0]["type"] == "less_than_equal"

    def test_not_frozen_runtime_mutable(self):
        """Unlike PluginsConfig, this section is runtime-adjustable -- no
        frozen=True, matching every other RuntimeConfig section's shape."""
        cfg = PluginRuntimeConfig()
        cfg.max_concurrent = 8  # must NOT raise
        assert cfg.max_concurrent == 8


class TestTopLevelWiring:
    def test_boot_config_defaults_with_no_plugins_still_boots(self, tmp_path):
        """The baseline requirement: an existing deployment with no
        plugins configured must still construct BootConfig cleanly."""
        from turbohaul.config import ServerConfig, StorageConfig, RuntimePathsConfig, UIConfig

        boot = BootConfig(
            server=ServerConfig(),
            storage=StorageConfig(
                blob_store_path=tmp_path / "blobs",
                manifests_path=tmp_path / "manifests",
                import_allowed_root=tmp_path / "import",
                state_db_path=tmp_path / "state.sqlite",
            ),
            runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server"),
            ui=UIConfig(static_path=tmp_path / "ui_dist"),
        )
        assert boot.plugins.registry == {}

    def test_runtime_config_defaults_with_no_plugin_runtime(self):
        from turbohaul.config import QueueConfig, PullConfig

        runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
        assert runtime.plugin_runtime.enabled == {}
        assert runtime.plugin_runtime.max_concurrent == 4
