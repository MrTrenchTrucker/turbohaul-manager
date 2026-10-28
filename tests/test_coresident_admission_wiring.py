"""Wiring tests for the co-residence admission call sites in the manager.

The admission rule lives in the coresident_admission module. These tests drive
the real ``TurbohaulManager._vram_admits_locked`` and check what its call sites
return and how many times they read the GPU. Both reader names are patched with
scripted readers that count their calls:

* the primary read, made inside the safety module's budget helper;
* the manager's own retry read, used by the flaky-probe path and by the
  relocatable evict branch.

Each test sits on one branch and on one side of one inequality, so a change to
the reserve, the credit, the reclaim term or the number of reads shows up as a
wrong value or a wrong count. The tests pin behaviour, not code layout.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager


# === manager construction (same convention as the existing co-residence wiring test) ==========


def _boot_runtime(tmp_path, *, safety_min_free_vram_mib=4096):
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
        queue=QueueConfig(
            max_parallel_sidecars=2,
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            safety_min_free_vram_mib=safety_min_free_vram_mib,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _mk(boot, runtime):
    mgr = TurbohaulManager(boot, runtime)
    # The cross-resident gate is not gated by safety_enabled; these tests call
    # `_vram_admits_locked` directly.
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _resident(tag, *, state, main_gpu=0, reserved_need_mib=0, split_mode="none"):
    return Resident(
        model_tag=tag, state=state, main_gpu=main_gpu,
        reserved_need_mib=reserved_need_mib, split_mode=split_mode,
    )


def _manager_with(tmp_path, residents, pending=None):
    """A manager holding `residents`, a list of (key, state, main_gpu,
    reserved_need_mib), and the pending-reclaim map `pending` (card -> MiB)."""
    boot, runtime = _boot_runtime(tmp_path)
    mgr = _mk(boot, runtime)
    for key, state, main_gpu, reserved in residents:
        mgr._residents[key] = _resident(
            key, state=state, main_gpu=main_gpu, reserved_need_mib=reserved)
    for card, mib in (pending or {}).items():
        mgr._pending_reclaim_mib[card] = mib
    return mgr


# === scripted readers ========================================================


class _ScriptedReader:
    """Return the scripted answers in order (None or a list of MiB per card),
    then None for every later call. Every call is counted, so a test can assert
    the exact number of reads, including reads past the script."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._answers:
            value = self._answers.pop(0)
            return None if value is None else list(value)
        return None


@contextmanager
def _scripted_readers(safety_answers, manager_answers):
    """Patch both reader names: the safety module (primary read inside the
    budget helper) and the manager module (retry reads). Yields the two
    readers so a test can check its call count."""
    safety = _ScriptedReader(safety_answers)
    manager = _ScriptedReader(manager_answers)
    with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=safety):
        with patch("turbohaul.manager._read_free_vram_all_mib", side_effect=manager):
            yield safety, manager


# === flaky primary read: the manager retries before it decides =============

_ACTIVE_SIBLING = ("sib", ResidentState.ACTIVE, 0, 18000)


def test_flaky_probe_retry_recovers_and_admits_after_exactly_two_manager_reads(tmp_path):
    """The primary read flakes and the first retry flakes too; the second
    retry sees 80000 MiB free on card 0, which covers need 18000. The
    co-resident must be admitted, and the manager must have read exactly twice."""
    mgr = _manager_with(tmp_path, [_ACTIVE_SIBLING])
    with _scripted_readers([None], [None, [80000, 1000]]) as (safety, manager):
        got = mgr._vram_admits_locked(18000, 1, 0, "none")
    assert safety.calls == 1, (
        f"flaky path: the primary read must happen once per call, got {safety.calls}")
    assert manager.calls == 2, (
        "flaky path: after a flaky first read the manager must retry once, "
        f"expected exactly 2 manager reads, got {manager.calls}")
    assert got is True, (
        "flaky path: the retry recovered 80000 MiB free on card 0 for need 18000, "
        f"so the co-resident must be admitted, got {got}")


def test_flaky_probe_refuses_after_exactly_two_manager_reads_when_both_flake(tmp_path):
    """Every read flakes and credit is zero. The branch must stop after the
    bounded retry (two manager reads, no third) and refuse."""
    mgr = _manager_with(tmp_path, [_ACTIVE_SIBLING])
    with _scripted_readers([None], [None, None]) as (safety, manager):
        got = mgr._vram_admits_locked(18000, 1, 0, "none")
    assert manager.calls == 2, (
        "flaky path with both reads flaking: the retry must be bounded to exactly "
        f"2 manager reads, got {manager.calls}")
    assert got is False, (
        "flaky path with no usable reading and no credit must refuse, "
        f"got {got}")


def test_flaky_probe_healthy_retry_read_makes_exactly_one_manager_read(tmp_path):
    """The primary read flakes, and the manager's first read is healthy with
    room. The retry must return that reading at once: one manager read, no
    second read."""
    mgr = _manager_with(tmp_path, [_ACTIVE_SIBLING])
    with _scripted_readers([None], [[80000, 1000]]) as (safety, manager):
        got = mgr._vram_admits_locked(18000, 1, 0, "none")
    assert manager.calls == 1, (
        "flaky path: a healthy first retry read must be used as it is, with no "
        f"second read; expected exactly 1 manager read, got {manager.calls}")
    assert got is True, (
        f"flaky path: a healthy retry read with room must admit, got {got}")


# === normal path (healthy primary read): the booting-sibling reserve =========


@pytest.mark.parametrize(
    "need, expected",
    [
        pytest.param(4000, True, id="need-4000-fits-exactly"),
        pytest.param(4001, False, id="need-4001-one-over"),
    ],
)
def test_normal_path_subtracts_a_booting_sibling_reserve_at_the_boundary(
        tmp_path, need, expected):
    """A RESERVED_LOADING sibling on card 0 holds 6000 MiB. Card 0 has 10000
    MiB free, so the room after the reserve is 4000 and the boundary sits
    exactly at need 4000."""
    mgr = _manager_with(tmp_path, [("boot0", ResidentState.RESERVED_LOADING, 0, 6000)])
    with _scripted_readers([[10000, 1000]], []) as (_safety, manager):
        got = mgr._vram_admits_locked(need, 1, 0, "none")
    assert manager.calls == 0, (
        "healthy primary read: the manager's retry read must not run on the "
        f"probe-available path, got {manager.calls} manager reads")
    assert got is expected, (
        f"normal path: 10000 MiB free less a 6000 MiB booting reserve leaves 4000; "
        f"need {need} must be admitted={expected}, got {got}")


@pytest.mark.parametrize(
    "resident",
    [
        pytest.param(("loaded0", ResidentState.ACTIVE, 0, 6000),
                     id="loaded-sibling-on-card-0-is-not-subtracted"),
        pytest.param(("boot1", ResidentState.RESERVED_LOADING, 1, 6000),
                     id="booting-sibling-on-card-1-is-not-subtracted-from-card-0"),
    ],
)
def test_normal_path_reserve_counts_only_booting_siblings_on_this_card(tmp_path, resident):
    """Card 0 has 10000 MiB free and need is 10000. Only a booting sibling on
    card 0 may take room from card 0. A loaded sibling is already inside the
    live reading, and a booting sibling on card 1 is not on this card, so
    both must leave the full 10000 MiB for the request."""
    mgr = _manager_with(tmp_path, [resident])
    with _scripted_readers([[10000, 1000]], []) as (_safety, manager):
        got = mgr._vram_admits_locked(10000, 1, 0, "none")
    assert manager.calls == 0, (
        "healthy primary read: the manager's retry read must not run, "
        f"got {manager.calls} manager reads")
    assert got is True, (
        "normal path: a sibling that is loaded, or booting on another card, must "
        "not reduce card 0's room; 10000 MiB free must admit need 10000, "
        f"got {got}")


# === evict branch (relocatable, credit pending, per-card loop) ===============

_EVICT_PRIMARY = [[1000, 10000]]
_EVICT_RECOVERED = [1000, 10000]
# Card 0 holds a booting sibling of 5000 MiB; card 1 holds one of 6000 MiB.
# Card 0 is nearly full, card 1 has 10000 MiB free, and card 0 has 1 MiB of
# its own pending reclaim.
_EVICT_RESIDENTS = [
    ("boot0", ResidentState.RESERVED_LOADING, 0, 5000),
    ("boot1", ResidentState.RESERVED_LOADING, 1, 6000),
]


def _evict_mgr(tmp_path):
    return _manager_with(tmp_path, _EVICT_RESIDENTS, pending={0: 1})


@pytest.mark.parametrize(
    "need, expected",
    [
        pytest.param(4000, True, id="need-4000-fits-card-1-exactly"),
        pytest.param(4001, False, id="need-4001-one-over-card-1"),
    ],
)
def test_evict_branch_reserve_is_taken_per_card_at_the_boundary(tmp_path, need, expected):
    """Card 1 has 10000 MiB free and a booting sibling of 6000 MiB, so its room
    is 4000. Card 0's 5000 MiB booting reserve must not be taken off card 1: if
    it were, card 1 would have only 5000 MiB and need 4001 would be admitted."""
    mgr = _evict_mgr(tmp_path)
    with _scripted_readers(_EVICT_PRIMARY, [_EVICT_RECOVERED]) as (_safety, _manager):
        got = mgr._vram_admits_locked(
            need, 1, 0, "none", credit_pending=True, relocatable=True)
    assert got is expected, (
        "evict branch: card 1 has 10000 MiB free less its own 6000 MiB booting "
        f"reserve, so need {need} must be admitted={expected} (card 0's reserve "
        f"must not count against card 1), got {got}")


@pytest.mark.parametrize(
    "need, expected",
    [
        pytest.param(4500, True, id="need-4500-fits-with-reclaim-exactly"),
        pytest.param(4501, False, id="need-4501-one-over-with-reclaim"),
    ],
)
def test_evict_branch_reclaim_term_is_added_on_the_target_card(tmp_path, need, expected):
    """With 500 MiB of reclaim on card 1 the room is 10000 - 6000 + 500 = 4500.
    The reclaim term must be added to card 1's own room: 4500 admits, 4501
    does not."""
    mgr = _evict_mgr(tmp_path)
    with _scripted_readers(_EVICT_PRIMARY, [_EVICT_RECOVERED]) as (_safety, _manager):
        got = mgr._vram_admits_locked(
            need, 1, 0, "none", credit_pending=True, relocatable=True,
            reclaim_mib_by_card={1: 500})
    assert got is expected, (
        "evict branch: card 1 room is 10000 - 6000 booting reserve + 500 reclaim "
        f"= 4500, so need {need} must be admitted={expected}, got {got}")


# === evict branch: exact number of retry reads ===============================


def test_evict_branch_reads_once_when_the_first_read_is_healthy(tmp_path):
    """A healthy first retry read is used at once: one manager read."""
    mgr = _evict_mgr(tmp_path)
    with _scripted_readers(_EVICT_PRIMARY, [_EVICT_RECOVERED]) as (_safety, manager):
        got = mgr._vram_admits_locked(
            4000, 1, 0, "none", credit_pending=True, relocatable=True)
    assert manager.calls == 1, (
        "evict branch: a healthy first retry read must be used as it is; "
        f"expected exactly 1 manager read, got {manager.calls}")
    assert got is True, (
        f"evict branch: card 1 admits need 4000 on the recovered reading, got {got}")


def test_evict_branch_reads_twice_when_the_first_read_flakes_and_the_second_recovers(tmp_path):
    """The first retry read flakes, the second recovers. Exactly two manager
    reads, and the recovered reading decides the answer."""
    mgr = _evict_mgr(tmp_path)
    with _scripted_readers(_EVICT_PRIMARY, [None, _EVICT_RECOVERED]) as (_safety, manager):
        got = mgr._vram_admits_locked(
            4000, 1, 0, "none", credit_pending=True, relocatable=True)
    assert manager.calls == 2, (
        "evict branch: after a flaky first read the manager must retry once; "
        f"expected exactly 2 manager reads, got {manager.calls}")
    assert got is True, (
        "evict branch: the second read recovered card 1 with room for need 4000, "
        f"so the answer must be True, got {got}")


def test_evict_branch_reads_exactly_twice_and_refuses_when_both_flake(tmp_path):
    """Both retry reads flake. The retry is bounded to two reads, and with no
    reading the branch must refuse."""
    mgr = _evict_mgr(tmp_path)
    with _scripted_readers(_EVICT_PRIMARY, [None, None]) as (_safety, manager):
        got = mgr._vram_admits_locked(
            4000, 1, 0, "none", credit_pending=True, relocatable=True)
    assert manager.calls == 2, (
        "evict branch with both reads flaking: the retry must be bounded to "
        f"exactly 2 manager reads, got {manager.calls}")
    assert got is False, (
        f"evict branch with no usable reading must refuse, got {got}")
