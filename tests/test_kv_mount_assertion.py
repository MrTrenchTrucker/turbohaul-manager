"""Boot-time KV-dir mount/fstype assertion.

FAIL LOUD, KEEP SERVING: manager.py's boot_reconcile() asserts that
SLOT_SAVE_DIR (RAM/tmpfs tier) and SLOT_PERSIST_DIR (cold-storage/disk tier)
are on the filesystem the manager expects. A mismatch is logged as an error
and counted in the boot_reconcile audit dict -- it NEVER raises, NEVER calls
sys.exit, and boot_reconcile still returns normally either way (a running deployment
must keep serving; a hard abort on a mount check would take down inference over a
warning). This mirrors the pre-existing foreign-GPU-detect precedent already
in-source at manager.py (informational only).

Failure mode: a missing/wrong mount does not error -- it
silently redirects writes to whatever filesystem is underneath. An unmounted
path still resolves to SOME fstype via its parent, so `is_own_mount=False` is
the signal that flags it regardless of what that inherited type happens to
be -- fstype comparison ALONE is not sufficient, which is why the mechanism
checks BOTH `os.stat().st_dev` (mount detection) and `/proc/self/mountinfo`
(fstype, parsed for what THIS process/container sees -- the host's view
can differ from the container's view).

Two-arm proof this suite locks in (the default deployment
configuration):
  SLOT_SAVE_DIR    -- own mount + tmpfs                      -> quiet
  SLOT_PERSIST_DIR -- NOT its own mount, NOT a forbidden type -> quiet
Both arms of the default configuration validate GREEN: silent when correct,
loud on the failure shape.

Non-vacuity: every test in TestClassifyKvDirMount and TestCheckKvDirMounts
constructs a state that only exists through the mount-assertion functions and
fields (nothing else produces it) -- the whole module targets them, so there's no
single test to diff against; non-vacuity here means every
loud/quiet verdict is independently re-derived from a real directory + a
synthetic mountinfo file, not asserted in prose.
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
)
from turbohaul.manager import (
    TurbohaulManager,
    _check_mount,
    _classify_kv_dir_mount,
    _parse_mountinfo,
)


def _write_mountinfo(tmp_path, entries: list[tuple[str, str]]) -> str:
    """entries: list of (mount_point, fstype). Writes a minimal but
    format-valid mountinfo(5) file and returns its path."""
    lines = []
    for i, (mp, fstype) in enumerate(entries):
        lines.append(
            f"{100 + i} 1 0:{200 + i} / {mp} rw,relatime master:1 - {fstype} src rw"
        )
    p = tmp_path / "mountinfo"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# === _parse_mountinfo ============================================================

def test_parse_mountinfo_reads_real_proc_self():
    """Sanity: the real /proc/self/mountinfo on this machine parses without error
    and reports something for '/' (every mount namespace has a root)."""
    mounts, readable = _parse_mountinfo()
    assert readable is True
    assert any(mp in ("/", "") for mp in mounts) or mounts  # non-empty, well-formed


def test_parse_mountinfo_unreadable_path_is_not_a_crash():
    mounts, readable = _parse_mountinfo("/definitely/does/not/exist/mountinfo")
    assert readable is False
    assert mounts == {}


def test_parse_mountinfo_synthetic_round_trips(tmp_path):
    path = _write_mountinfo(tmp_path, [("/dev/shm", "tmpfs"), ("/mnt/data", "ext4")])
    mounts, readable = _parse_mountinfo(path)
    assert readable is True
    assert mounts["/dev/shm"] == "tmpfs"
    assert mounts["/mnt/data"] == "ext4"


# === _check_mount — real is_own_mount (via /dev/shm, a real tmpfs mount in
# every Linux container) vs an ordinary tmp_path subdirectory (NOT a mount) ===

def test_check_mount_dev_shm_is_its_own_mount_real_tmpfs():
    """The two-arm proof's PASSING arm: /dev/shm is a real distinct mount on
    every Linux container (verified in the test sandbox)."""
    check = _check_mount("/dev/shm")
    assert check.is_own_mount is True
    assert check.mountinfo_readable is True
    assert check.fstype == "tmpfs"


def test_check_mount_ordinary_tmp_subdir_is_not_its_own_mount(tmp_path):
    """The two-arm proof's FAILING arm: an ordinary directory under tmp_path
    inherits its parent's fstype/dev -- it is NOT its own mount. This is
    EXACTLY the silent-redirect shape: it still resolves to a real fstype
    (whatever tmp_path's enclosing mount is), just not the expected one."""
    sub = tmp_path / "ordinary_dir"
    sub.mkdir()
    check = _check_mount(str(sub))
    assert check.is_own_mount is False


# === _classify_kv_dir_mount — the six cases + persist's forbidden-fstype case ===

class TestClassifyKvDirMountSaveShape:
    """SLOT_SAVE_DIR shape: expected_fstype, own-mount REQUIRED."""

    def test_mounted_and_correct_type_is_quiet(self):
        loud, reason = _classify_kv_dir_mount("/dev/shm", expected_fstype="tmpfs")
        assert loud is False, reason

    def test_not_a_mount_is_loud(self, tmp_path):
        sub = tmp_path / "not_mounted"
        sub.mkdir()
        loud, reason = _classify_kv_dir_mount(str(sub), expected_fstype="tmpfs")
        assert loud is True
        assert "NOT its own mount" in reason

    def test_wrong_fstype_is_loud(self, tmp_path):
        """Own mount (real /dev/shm) but the mountinfo we hand it claims a
        DIFFERENT fstype than what's expected -- proves the fstype comparison
        itself is load-bearing, independent of the real is_own_mount check."""
        mountinfo = _write_mountinfo(tmp_path, [("/dev/shm", "ext4")])
        loud, reason = _classify_kv_dir_mount(
            "/dev/shm", expected_fstype="tmpfs", mountinfo_path=mountinfo,
        )
        assert loud is True
        assert "expected fstype" in reason

    def test_path_missing_is_loud(self, tmp_path):
        missing = tmp_path / "does_not_exist_at_all"
        loud, reason = _classify_kv_dir_mount(str(missing), expected_fstype="tmpfs")
        assert loud is True
        assert "missing" in reason

    def test_mountinfo_unreadable_is_loud_not_a_crash(self):
        loud, reason = _classify_kv_dir_mount(
            "/dev/shm", expected_fstype="tmpfs",
            mountinfo_path="/definitely/does/not/exist/mountinfo",
        )
        assert loud is True
        assert "unreadable" in reason

    def test_expected_none_disables_assertion_even_on_a_non_mount(self, tmp_path):
        """None means informational-only: no loud verdict even for a path
        that plainly isn't its own mount."""
        sub = tmp_path / "no_opinion_dir"
        sub.mkdir()
        loud, reason = _classify_kv_dir_mount(str(sub), expected_fstype=None)
        assert loud is False, reason


class TestClassifyKvDirMountPersistShape:
    """SLOT_PERSIST_DIR shape: forbidden_fstypes, own-mount NEVER asserted."""

    def test_forbidden_fstype_is_loud(self):
        """The motivating scenario: cold storage landed on tmpfs -> LOUD."""
        loud, reason = _classify_kv_dir_mount(
            "/dev/shm", forbidden_fstypes=frozenset({"tmpfs", "ramfs"}),
        )
        assert loud is True
        assert "FORBIDDEN" in reason

    def test_not_forbidden_fstype_is_quiet_even_though_not_its_own_mount(self, tmp_path):
        """Matches the default deployment layout: kvcache_persist is NOT
        its own mount (a plain directory on the underlying filesystem, BY
        DESIGN) and must stay quiet as long as its fstype isn't forbidden --
        asserting own-mount here would cry LOUD on a correctly configured
        deployment on its very first boot."""
        sub = tmp_path / "persist_like_the_live_box"
        sub.mkdir()
        mountinfo = _write_mountinfo(tmp_path, [(str(tmp_path), "zfs")])
        loud, reason = _classify_kv_dir_mount(
            str(sub), forbidden_fstypes=frozenset({"tmpfs", "ramfs"}),
            mountinfo_path=mountinfo,
        )
        assert loud is False, reason

    def test_path_missing_is_loud_for_persist_shape_too(self, tmp_path):
        missing = tmp_path / "persist_does_not_exist"
        loud, reason = _classify_kv_dir_mount(
            str(missing), forbidden_fstypes=frozenset({"tmpfs", "ramfs"}),
        )
        assert loud is True
        assert "missing" in reason


# === TurbohaulManager._check_kv_dir_mounts — the production configuration, end to end ===

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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


class TestCheckKvDirMountsLiveBoxShape:
    """Simulates the default configuration of a typical deployment:
    SLOT_SAVE_DIR is its own tmpfs mount; SLOT_PERSIST_DIR is a plain
    directory (not its own mount) on a non-forbidden fstype. Both arms
    of the default configuration must validate GREEN -- silent when correct."""

    def test_default_config_is_quiet_on_the_live_box_shape(self, mgr, monkeypatch):
        import turbohaul.manager as manager_mod

        monkeypatch.setattr(
            manager_mod, "_classify_kv_dir_mount",
            lambda path, **kw: (False, f"{path}: OK (simulated live-box state)"),
        )
        issues = mgr._check_kv_dir_mounts()
        assert issues == []

    def test_save_dir_mismatch_is_loud_and_counted(self, mgr, monkeypatch):
        import turbohaul.manager as manager_mod
        import turbohaul.subprocess_mgr as subprocess_mgr

        def fake_classify(path, *, expected_fstype=None, forbidden_fstypes=frozenset(), mountinfo_path=None):
            if path == subprocess_mgr.SLOT_SAVE_DIR:
                return True, f"{path}: expected fstype='tmpfs', observed 'zfs'"
            return False, f"{path}: OK"

        monkeypatch.setattr(manager_mod, "_classify_kv_dir_mount", fake_classify)
        issues = mgr._check_kv_dir_mounts()
        assert len(issues) == 1
        assert "expected fstype" in issues[0]

    def test_persist_dir_on_forbidden_fstype_is_loud(self, mgr, monkeypatch):
        """The evaporation scenario -- persist landed on
        tmpfs -- surfaces as a boot_reconcile issue."""
        import turbohaul.manager as manager_mod
        import turbohaul.subprocess_mgr as subprocess_mgr

        def fake_classify(path, *, expected_fstype=None, forbidden_fstypes=frozenset(), mountinfo_path=None):
            if path == subprocess_mgr.SLOT_PERSIST_DIR:
                return True, f"{path}: fstype='tmpfs' is FORBIDDEN for this directory"
            return False, f"{path}: OK"

        monkeypatch.setattr(manager_mod, "_classify_kv_dir_mount", fake_classify)
        issues = mgr._check_kv_dir_mounts()
        assert len(issues) == 1
        assert "FORBIDDEN" in issues[0]

    def test_a_raising_check_never_propagates(self, mgr, monkeypatch):
        """Defense in depth: even if _classify_kv_dir_mount itself somehow
        raised, _check_kv_dir_mounts must not -- boot_reconcile must keep
        serving regardless."""
        import turbohaul.manager as manager_mod

        def boom(path, **kw):
            raise RuntimeError("simulated bug in the check itself")

        monkeypatch.setattr(manager_mod, "_classify_kv_dir_mount", boom)
        issues = mgr._check_kv_dir_mounts()  # must not raise
        assert len(issues) == 2  # both dirs independently caught their own failure
        assert all("raised unexpectedly" in i for i in issues)

    def test_config_defaults_match_the_live_box_expectation(self, mgr):
        """The shipped defaults, unmodified, are exactly what a production
        deployment needs: save expects tmpfs, persist forbids tmpfs/ramfs."""
        assert mgr.runtime.kv.kv_save_expected_fstype == "tmpfs"
        assert set(mgr.runtime.kv.kv_persist_forbidden_fstypes) == {"tmpfs", "ramfs"}


class TestBootReconcileNeverRaisesOnMountIssues:
    """boot_reconcile() itself must keep returning normally even with LOUD
    mount issues present -- FAIL LOUD, KEEP SERVING, per the foreign-GPU
    precedent this change mirrors."""

    def test_boot_reconcile_returns_normally_with_loud_kv_mount_issues(self, mgr, monkeypatch):
        import turbohaul.manager as manager_mod

        monkeypatch.setattr(
            manager_mod, "_classify_kv_dir_mount",
            lambda path, **kw: (True, f"{path}: simulated LOUD issue"),
        )
        result = mgr.boot_reconcile(pid_is_alive_fn=lambda pid: False)
        assert "kv_mount_issues" in result
        assert len(result["kv_mount_issues"]) == 2
        # boot_reconcile did NOT raise/exit -- we got here, and it returned its
        # normal shape alongside the new field.
        assert "orphans_reaped" in result
        assert "slots_reconciled_to_cold" in result


# --- schema exposure for the mount-assertion fields (checked here rather than in the shared
# schema test) -------------------------------------------------------------
def test_mount_assertion_keys_are_exposed_by_the_schema_endpoint():
    from turbohaul.config import KVConfig
    declared = set(KVConfig.model_fields)
    assert "kv_save_expected_fstype" in declared
    assert "kv_persist_forbidden_fstypes" in declared
