"""Contract test for the engine_launch module's public interface.

Asserts that engine_launch exports exactly its declared `public` list
(public = ["launch_env", "preset_device_mismatch"]),
each callable -- both names.
"""
import turbohaul.engine_launch as engine_launch
from turbohaul import manifest

EXPECTED_PUBLIC = {"launch_env", "preset_device_mismatch"}


def test_exports_exactly_the_registry_public_names():
    assert set(engine_launch.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_present_and_callable():
    for name in EXPECTED_PUBLIC:
        assert hasattr(engine_launch, name), f"engine_launch has no attribute {name!r}"
        assert callable(getattr(engine_launch, name)), f"engine_launch.{name} is not callable"


def test_launch_env_is_callable_with_argv_and_base_env():
    result = engine_launch.launch_env([], {"PATH": "/usr/bin"})
    assert result == {"PATH": "/usr/bin"}


def test_preset_device_mismatch_is_callable_with_argv_and_base_env():
    result = engine_launch.preset_device_mismatch([], {"PATH": "/usr/bin"})
    assert result is None


def test_no_device_or_rpc_flag_reaches_the_engine():
    """Follow-up check: nothing in engine_launch's own code enforces
    the module's "no --device, no --rpc" condition (a documented invariant) -- it holds
    only because manifest.SAFE_LLAMA_FLAGS has no 'device' key and manifest.DENIED_FLAGS
    contains 'rpc'. Pins the ACTUAL guard so a future allowlist change fails loudly
    here, at the module boundary that depends on it, rather than silently reopening
    the device-ordering problem (llama.cpp puts RPC devices first and --device reorders the list
    --main-gpu indexes)."""
    assert "device" not in manifest.SAFE_LLAMA_FLAGS
    assert "rpc" in manifest.DENIED_FLAGS
