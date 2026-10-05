"""Boot runs a one-time sweep that removes the retired ``max_instances`` key from the
stored manifests.

The key no longer exists, but manifest files saved before that still carry it. The
manifest loader already ignores it (with one warning), so a stale file is harmless to
load; the sweep at boot rewrites each such file without the key so the warning does not
repeat on every read and the stored file matches what is served.

What is pinned here, against a real manager over a real manifests directory:

  * a stored manifest with the key is rewritten without it, every other field kept;
  * a manifest without the key and a backup file (``<tag>.yaml.bak-1``) are left
    byte-identical, so a backup is never rewritten;
  * the sweep runs before the rest of boot, and a second boot changes nothing;
  * if the sweep itself raises, boot still completes with its normal summary and the
    failure is logged: the sweep can never block the start.
"""
from __future__ import annotations

import logging

import pytest
import yaml

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.manager import TurbohaulManager

MIB = 1024 * 1024


def _body(tag, **extra):
    body = {
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 1000 * MIB,
        "context_size": 2048,
        "expected_vram_bytes": 1000 * MIB,
        "auto_place": True,
        "llama_server_flags": {"split_mode": "none"},
    }
    body.update(extra)
    return body


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A manager over a temp state dir, with the process scans stubbed out the way the
    existing boot_reconcile tests do (nothing in the managed port range)."""
    root = tmp_path / "state"
    root.mkdir()
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    monkeypatch.setattr("turbohaul.singleton.find_orphan_llama_servers",
                        lambda port_base, port_range_size=100, **kw: [])
    monkeypatch.setattr("turbohaul.singleton.port_listeners_in_range",
                        lambda port_base, port_range_size=100: [])
    monkeypatch.setattr("turbohaul.singleton.find_llama_servers_in_port_range",
                        lambda port_base, port_range_size=100, **kw: [])
    mgr = TurbohaulManager(boot, runtime)
    manifests = boot.storage.manifests_path
    old_style = _body("old-style", max_instances=2)
    (manifests / "old-style.yaml").write_text(yaml.safe_dump(old_style, sort_keys=False))
    (manifests / "key-free.yaml").write_text(yaml.safe_dump(_body("key-free"), sort_keys=False))
    (manifests / "old-style.yaml.bak-1").write_text(
        yaml.safe_dump(old_style, sort_keys=False))
    return mgr, manifests, old_style


def _snapshot(manifests):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(manifests.iterdir())}


def test_boot_rewrites_a_stored_manifest_without_the_retired_key(env):
    mgr, manifests, old_style = env
    mgr.boot_reconcile()
    rewritten = yaml.safe_load((manifests / "old-style.yaml").read_text())
    assert "max_instances" not in rewritten
    expected = {k: v for k, v in old_style.items() if k != "max_instances"}
    assert rewritten == expected, "every other field must be kept exactly"


def test_boot_leaves_a_key_free_manifest_and_a_backup_byte_identical(env):
    mgr, manifests, _ = env
    before = _snapshot(manifests)
    mgr.boot_reconcile()
    after = _snapshot(manifests)
    for name in ("key-free.yaml", "old-style.yaml.bak-1"):
        assert after[name] == before[name], f"{name} must not be touched (bytes and mtime)"
    assert b"max_instances" in after["old-style.yaml.bak-1"][0], (
        "precondition: the backup carries the key, and keeps it"
    )
    assert after["old-style.yaml"][0] != before["old-style.yaml"][0], (
        "precondition: the sweep did rewrite the stored manifest"
    )


def test_a_second_boot_changes_nothing(env, caplog):
    mgr, manifests, _ = env
    mgr.boot_reconcile()
    after_first = _snapshot(manifests)
    with caplog.at_level(logging.INFO, logger="turbohaul.manager"):
        mgr.boot_reconcile()
    assert _snapshot(manifests) == after_first
    assert not [r for r in caplog.records if "retired-key sweep" in r.getMessage()], (
        "an already-clean store must not log a sweep line"
    )


def test_boot_logs_how_many_files_were_rewritten(env, caplog):
    mgr, _, _ = env
    with caplog.at_level(logging.INFO, logger="turbohaul.manager"):
        mgr.boot_reconcile()
    lines = [r.getMessage() for r in caplog.records if "retired-key sweep" in r.getMessage()]
    assert lines == ["boot: retired-key sweep rewrote 1 manifest file(s)"]


def test_the_sweep_runs_before_the_rest_of_boot(env, monkeypatch):
    # The first thing boot does after the sweep is open the state database. Record the
    # stored manifest at that moment: it must already be rewritten.
    mgr, manifests, _ = env
    seen = []
    import turbohaul.manager as manager_module
    real = manager_module.state_db_session

    def spy(*a, **k):
        if not seen:
            seen.append((manifests / "old-style.yaml").read_text())
        return real(*a, **k)

    monkeypatch.setattr(manager_module, "state_db_session", spy)
    mgr.boot_reconcile()
    assert seen, "boot never opened the state database"
    assert "max_instances" not in seen[0], "the sweep must run before the rest of boot"


def test_a_failing_sweep_never_blocks_boot(env, monkeypatch, caplog):
    mgr, manifests, _ = env

    def boom(root):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr("turbohaul.manager.migrate_retired_keys", boom)
    before = _snapshot(manifests)
    with caplog.at_level(logging.ERROR, logger="turbohaul.manager"):
        result = mgr.boot_reconcile()
    assert set(result) == {
        "orphans_reaped", "orphans_failed", "foreign_gpu_apps",
        "slots_reconciled_to_cold", "stale_listeners", "kv_mount_issues",
    }
    assert result["orphans_reaped"] == 0 and result["stale_listeners"] == 0
    failures = [r for r in caplog.records
                if r.levelno == logging.ERROR and "retired-key sweep" in r.getMessage()]
    assert len(failures) == 1, "the sweep failure must be logged once, at error level"
    assert failures[0].exc_info is not None and failures[0].exc_info[0] is RuntimeError
    assert _snapshot(manifests) == before, "nothing is rewritten when the sweep fails"
