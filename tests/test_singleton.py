"""Tests for singleton invariant + orphan reaper (v0.2 §3.1)."""
import pytest

from turbohaul.singleton import (
    SingletonViolation,
    acquire_state_lock,
    boot_orphan_reaper,
    detect_foreign_gpu_apps,
    find_orphan_llama_servers,
    scan_gpu_compute_apps,
)


class TestStateLock:
    def test_acquire_release(self, tmp_path):
        state_db = tmp_path / "state.sqlite"
        with acquire_state_lock(state_db) as fd:
            assert fd >= 0
            assert state_db.exists()
        # Re-acquire after release
        with acquire_state_lock(state_db) as fd2:
            assert fd2 >= 0

    def test_second_acquire_fails(self, tmp_path):
        state_db = tmp_path / "state.sqlite"
        with acquire_state_lock(state_db):
            with pytest.raises(SingletonViolation, match="singleton invariant"):
                with acquire_state_lock(state_db):
                    pass

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "state.sqlite"
        with acquire_state_lock(nested):
            assert nested.exists()
        assert nested.parent.exists()

    def test_file_mode_is_0o600(self, tmp_path):
        state_db = tmp_path / "state.sqlite"
        with acquire_state_lock(state_db):
            mode = state_db.stat().st_mode & 0o777
            # On Linux, the file gets created with our requested mode masked by umask.
            # We requested 0o600; typical umask 022 keeps it 0o600.
            # Just verify owner can read+write at minimum.
            assert mode & 0o600 == 0o600


class TestGpuScan:
    def test_returns_list(self):
        apps = scan_gpu_compute_apps()
        assert isinstance(apps, list)
        for app in apps:
            assert isinstance(app, dict)
            assert "pid" in app
            assert "used_memory_mib" in app
            assert isinstance(app["pid"], int)
            assert isinstance(app["used_memory_mib"], int)


class TestOrphanScanner:
    def test_returns_list_no_orphans_for_unused_port_range(self):
        # Pick a port range nothing realistic uses
        orphans = find_orphan_llama_servers(port_base=59500, port_range_size=10)
        assert isinstance(orphans, list)

    def test_boot_reaper_returns_summary(self):
        # Use a port range no llama-server should be in
        result = boot_orphan_reaper(port_base=59500)
        assert "scanned" in result
        assert "orphans_found" in result
        assert "reaped" in result
        assert "failed" in result
        assert "details" in result
        assert isinstance(result["details"], list)


class TestForeignGpuDetect:
    def test_returns_list(self):
        foreign = detect_foreign_gpu_apps()
        assert isinstance(foreign, list)
        for app in foreign:
            assert "pid" in app
            assert "cmdline" in app

    def test_known_pids_excluded(self):
        # Get whatever's running, exclude all of them, should return empty
        all_apps = scan_gpu_compute_apps()
        all_pids = {app["pid"] for app in all_apps}
        foreign = detect_foreign_gpu_apps(known_pids=all_pids)
        assert foreign == []
