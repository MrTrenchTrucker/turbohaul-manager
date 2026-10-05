"""RAM KV-cache AGE limit as a
real config field, backend and frontend.

Without a config field, `_gc_kv_cache`'s age ceiling is a hardcoded default
(`max_age_hours: float = 6.0`) whose sole production caller (manager.py's
background sweeper) passes no arguments -- the default would be the only value the
system could ever reach. The `config.py` field closes that gap. This
mirrors `KVConfig.ram_cache_max_bytes` almost exactly, and
this test file is deliberately structured the same way as
`tests/test_kvcache_ram_ceiling_config.py` -- same fixtures, same acceptance
shape -- for that sibling field.

Requirements:
  Entries in the RAM-disk tier are released after a configured age.
  The default is 6 hours.
  The age limit is a user-facing setting. It is exposed on
  the backend and the frontend, in the Settings tab, alongside
  the other settings. It is not an
  environment variable.

Covers:
  1. The age limit is a real KVConfig field (default 6.0, matching the
     pre-existing hardcoded default -- this promotion does not change the
     value, only makes it reachable), round-trips through the same
     `/api/config` surface as its byte-ceiling sibling.
  2. The GC actually COMPUTES a different age cutoff / evicts a different
     file set depending on the config value -- not merely proving the field
     exists. Reads `self.runtime.kv.ram_cache_max_age_hours` LIVE, no
     restart, same "replace mgr.runtime wholesale" mutation PUT /api/config
     performs.

Non-vacuity: test_age_limit_is_a_real_config_field and
test_gc_reads_age_limit_from_config_live_not_hardcoded_default both FAIL on
a build without the field (no such field on KVConfig -> AttributeError /
extra="forbid" ValidationError on the first; on the second, the config value
has ZERO effect on eviction since the hardcoded 6.0 default is the only value
the method ever computes with -- a bin aged 3h under a config age-limit of
1h would NOT be evicted, contradicting the assertion). PASSES with the field in place.
"""
import os

import pytest

from turbohaul.config import (
    BootConfig,
    KVConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
    TurbohaulConfig,
)
from turbohaul.manager import TurbohaulManager


@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
            default_port_base=59501,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    import turbohaul.subprocess_mgr as subprocess_mgr
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


def _write_bin(kv_dir, name: str, size: int, mtime: float) -> str:
    """Plain, un-pinned .bin, no .json sidecar -- never pin-elected, never
    live-protected on a fresh manager with no active state. A pure candidate
    for the age-based eviction scan."""
    p = kv_dir / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    return name


# === Acceptance: the age limit is a real config field ===

def test_age_limit_is_a_real_config_field_mirroring_ram_cache_max_bytes():
    """FAILS without the field (no such field on KVConfig -- either
    AttributeError reading it, or a ValidationError from extra='forbid' when
    constructing with it). PASSES with the field: same default (6.0 hours) as
    the pre-existing code default -- this promotion does not change the
    value."""
    kv = KVConfig()
    assert kv.ram_cache_max_age_hours == 6.0
    kv2 = KVConfig(ram_cache_max_age_hours=2.0)
    assert kv2.ram_cache_max_age_hours == 2.0


def test_config_surface_exposes_kv_section_including_new_age_field():
    """/api/config's live view is mgr.runtime.model_dump() -- confirm the new
    field round-trips through that exact call (what get_config() in
    api/main.py actually does), and appears in KVConfig's own schema dump
    (what config_schema.py's get_config_schema() iterates -- what
    Config.tsx's schema-driven editor renders in the Settings tab)."""
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    dumped = runtime.model_dump(mode="json")
    assert "ram_cache_max_age_hours" in dumped["kv"]
    assert dumped["kv"]["ram_cache_max_age_hours"] == 6.0
    schema_defaults = KVConfig().model_dump(mode="json")
    assert "ram_cache_max_age_hours" in schema_defaults


# === Acceptance: the GC actually COMPUTES with the config value, not a
#     hardcoded default -- a fixture that ONLY sets the config knob and never
#     touches age_cutoff/the deletion decision directly. ===

@pytest.mark.asyncio
async def test_gc_reads_age_limit_from_config_live_not_hardcoded_default(mgr, kv_dir):
    """One bin, 3 hours old. Under the pre-existing hardcoded default (6h) it
    would SURVIVE. Set `mgr.runtime.kv.ram_cache_max_age_hours = 1.0` (a
    config-only change; the assertion never sets `deleted` or touches the
    eviction decision) and call `_gc_kv_cache()` with NO explicit
    max_age_hours -- exactly how the sole production caller invokes it. If
    the config value is actually read, a 3h-old bin exceeds a 1h ceiling and
    must be evicted. This FAILS without the field: the hardcoded 6.0
    default survives regardless of what runtime.kv is set to, so the bin is
    never evicted and `deleted == 0`, contradicting the assertion below."""
    three_hours_ago = 10_000.0
    now_stub = three_hours_ago + 3 * 3600
    _write_bin(kv_dir, "m.a.slot0.bin", 1024, mtime=three_hours_ago)

    mgr.runtime.kv.ram_cache_max_age_hours = 1.0

    import time as _time
    real_time = _time.time
    try:
        _time.time = lambda: now_stub
        deleted = await mgr._gc_kv_cache()  # no explicit max_age_hours -- production shape
    finally:
        _time.time = real_time

    assert deleted == 1
    assert not (kv_dir / "m.a.slot0.bin").exists()


@pytest.mark.asyncio
async def test_age_limit_change_is_picked_up_live_no_restart(mgr, kv_dir):
    """Hazard: mutate `self.runtime.kv.ram_cache_max_age_hours` the same way
    PUT /api/config does (config_put.py's `mgr.runtime = new_runtime` --
    replacing the whole runtime object) BETWEEN two GC calls on the SAME
    manager instance, with no re-construction of TurbohaulManager -- proves
    the next GC pass honors the new age limit immediately."""
    two_hours_ago = 10_000.0
    now_stub = two_hours_ago + 2 * 3600
    _write_bin(kv_dir, "m.a.slot0.bin", 1024, mtime=two_hours_ago)

    import time as _time
    real_time = _time.time
    try:
        _time.time = lambda: now_stub

        # Generous limit (6h, the default) -> first pass evicts nothing.
        deleted_first = await mgr._gc_kv_cache()
        assert deleted_first == 0
        assert (kv_dir / "m.a.slot0.bin").exists()

        # Simulate a live PUT /api/config: REPLACE mgr.runtime wholesale
        # (exactly what config_put.py's `mgr.runtime = new_runtime` does),
        # tightening the age limit to 1h -- the 2h-old bin must now go.
        new_runtime = mgr.runtime.model_copy(deep=True)
        new_runtime.kv.ram_cache_max_age_hours = 1.0
        mgr.runtime = new_runtime

        deleted_second = await mgr._gc_kv_cache()
        assert deleted_second == 1
        assert not (kv_dir / "m.a.slot0.bin").exists()
    finally:
        _time.time = real_time


@pytest.mark.asyncio
async def test_explicit_max_age_hours_still_overrides_config_for_test_isolation(mgr, kv_dir):
    """Existing tests (test_classifier.py, test_kvcache_ram_ceiling_config.py)
    pass max_age_hours explicitly to disable the age axis while isolating an
    unrelated behavior (e.g. the byte ceiling). Config must NOT silently
    override an explicit call-site value -- explicit always wins, same
    relationship max_files already has (never promoted, always a plain
    override). Config is a fallback for the omitted case only."""
    _write_bin(kv_dir, "m.a.slot0.bin", 1024, mtime=10_000.0)
    mgr.runtime.kv.ram_cache_max_age_hours = 0.0  # config says "evict everything"

    import time as _time
    real_time = _time.time
    try:
        _time.time = lambda: 10_000.0 + 1.0  # ~1 second old
        deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000)
    finally:
        _time.time = real_time

    assert deleted == 0
    assert (kv_dir / "m.a.slot0.bin").exists()
