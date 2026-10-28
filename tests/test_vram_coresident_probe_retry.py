"""The co-residence branch of the VRAM gate retries a flaky free-VRAM probe once, instead of
refusing on one bad read.

The relocatable evict branch of ``_vram_admits_locked`` already retries
``_read_free_vram_all_mib() or _read_free_vram_all_mib() or []``. The co-residence branch
(the ``free_fit is None`` + ``siblings`` case in ``_vram_admits_locked``, manager.py) uses the same
two-read retry. Under GPU saturation the probe reads ``gpu0_vram_used_mib=unavailable`` (util
98-100%), and a single flaky read must not refuse a routable 2nd co-resident that has physical VRAM
room and drop it into the evict-pending defer (1s backoff).

This file's CONTROL test (``test_probe_flake_then_recover_ADMITS_2nd_co_resident``) checks that a
flaky first read followed by a good read admits. Three guards: an all-reads-flake probe must STILL
refuse (the retry never becomes admit-blind), the healthy-probe path admits, and the
``_scoped_credit() >= need`` fast path admits with no usable probe.

Behaviour tests, exercised through the real `_vram_admits_locked` gate with
``_read_free_vram_all_mib`` monkeypatched to flake-then-recover (nvidia-smi is
absent in the test env, so the probe is always stubbed).
"""
from __future__ import annotations

from contextlib import ExitStack
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


# === shared manager-construction fixture (same convention as
# tests/test_occupied_vram.py / tests/test_multislot_concurrency.py) ==

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
    # The cross-resident gate is not gated by safety_enabled (VRAM tests re-enable
    # only the per-spawn host gate); here we call `_vram_admits_locked` directly.
    mgr.runtime.queue.safety_enabled = False
    return mgr


def _resident(tag, *, state, main_gpu=0, reserved_need_mib=0, split_mode="none"):
    return Resident(
        model_tag=tag, state=state, main_gpu=main_gpu,
        reserved_need_mib=reserved_need_mib, split_mode=split_mode,
    )


def _flake_once_then_recover(recover_vals):
    """Return a probe callable whose FIRST invocation flakes to None (the
    primary read inside ``_vram_budget``) and every later invocation recovers
    with ``recover_vals``. ONE shared counter is patched into BOTH the safety
    namespace (``_vram_budget`` reads through it) and the manager namespace (the
    retry reads through it), so the call order is: #1 -> None, #2+ -> room."""
    state = {"n": 0}

    def _flaky():
        state["n"] += 1
        return None if state["n"] == 1 else list(recover_vals)

    return _flaky


def _always_flake():
    return None


class TestCoResidenceProbeRetry:
    """The co-residence ``free_fit is None`` branch must retry the probe
    before refusing — exactly like the relocatable evict branch and its
    `vals = _read_free_vram_all_mib() or ...` line."""

    def _sibling_only_mgr(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = _mk(boot, runtime)
        # A tensor-isolated (split=none) ACTIVE sibling on gpu0 — co-residence is
        # only ever offered to split=none models on a per-card budget, and an
        # ACTIVE sibling's VRAM is already reflected in the live probe (not
        # subtracted again). Its mere presence makes `siblings` non-empty.
        mgr._residents["sib"] = _resident(
            "sib", state=ResidentState.ACTIVE, main_gpu=0, reserved_need_mib=18000)
        return mgr

    def test_probe_flake_then_recover_ADMITS_2nd_co_resident(self, tmp_path):
        """CONTROL: a single flaky read must not refuse.

        A sibling is loaded, credit alone is short (``credit_pending=False`` ->
        ``_scoped_credit() == 0 < need``), and the primary probe flakes — but
        the retry recovers with ample room, so the routable 2nd co-resident is
        ADMITTED. Without the retry the branch would refuse on the single flaky read
        (refuse-blind) and this assertion would fail; with the retry it sees
        room and admits."""
        mgr = self._sibling_only_mgr(tmp_path)
        need = 18000
        flaky = _flake_once_then_recover([80000, 80000])
        with patch_safety_and_manager(flaky):
            admitted = mgr._vram_admits_locked(need, 1, 0, "none")
        assert admitted, (
            "the co-residence probe control: a flaky-then-recovered probe that shows room must "
            "admit the 2nd co-resident, not refuse-blind on the first flaky read"
        )

    def test_all_reads_flake_still_REFUSES(self, tmp_path):
        """Regression guard: when EVERY read flakes and
        credit alone is short, the branch must still refuse — the retry must
        never become admit-blind."""
        mgr = self._sibling_only_mgr(tmp_path)
        with patch_safety_and_manager(_always_flake):
            assert not mgr._vram_admits_locked(18000, 1, 0, "none"), (
                "all-reads-flake with short credit must still refuse-blind"
            )

    def test_healthy_probe_admits_unchanged(self, tmp_path):
        """Regression guard: a healthy probe with room
        admits on the normal probe-available path."""
        mgr = self._sibling_only_mgr(tmp_path)
        with patch_safety_and_manager([80000, 80000], static=True):
            assert mgr._vram_admits_locked(18000, 1, 0, "none"), (
                "healthy probe with room must admit"
            )

    def test_scoped_credit_fast_path_admits_without_probe(self, tmp_path):
        """Regression guard: the ``_scoped_credit() >= need`` fast path
        admits with NO usable probe at all — credit alone covers ``need``."""
        mgr = self._sibling_only_mgr(tmp_path)
        mgr._pending_reclaim_mib[0] = 20000  # >= need on gpu0
        with patch_safety_and_manager(_always_flake):
            assert mgr._vram_admits_locked(18000, 1, 0, "none", credit_pending=True), (
                "scoped-credit fast-path must admit without any probe"
            )


# === patch helper (both namespaces: `_vram_budget` reads through the safety
# name; the manager-module retry reads through the manager name) ===============


def patch_safety_and_manager(probe, *, static=False):
    """Return a context manager patching ``_read_free_vram_all_mib`` in BOTH the
    safety module (where ``_vram_budget`` calls it) and the manager module (where
    the retry calls it). ``static=True`` makes ``probe`` a fixed ``return_value``
    (healthy-probe path); otherwise it is a callable used as ``side_effect``
    (per-call flake/recover semantics)."""
    if static:
        p1 = patch("turbohaul.safety._read_free_vram_all_mib", return_value=list(probe))
        p2 = patch("turbohaul.manager._read_free_vram_all_mib", return_value=list(probe))
    else:
        p1 = patch("turbohaul.safety._read_free_vram_all_mib", side_effect=probe)
        p2 = patch("turbohaul.manager._read_free_vram_all_mib", side_effect=probe)
    es = ExitStack()
    es.enter_context(p1)
    es.enter_context(p2)
    return es


# === coverage of the recovered-reading arithmetic ======
#
# The retry in the `free_fit is None` co-residence branch ends in
#     vals[idx] - reserve_g + _scoped_credit() >= need
# The tests above pin the retry itself (recover -> admit, all-flake -> refuse,
# healthy probe, credit fast-path). The tests below pin the three terms of that
# line: the `reserve_g` subtraction, the `idx` fallback, and the scoped credit.
# Every case uses the same shape as the control test: the first probe read flakes
# to None (inside `_vram_budget`), the retry recovers with RECOVERED (free MiB per
# card), split 'none', need 18000, parallel 1.

_NEED = 18000


def _coverage_mgr(tmp_path, residents):
    """A manager holding `residents`, a list of
    (key, state, main_gpu, reserved_need_mib), inserted in the given order."""
    boot, runtime = _boot_runtime(tmp_path)
    mgr = _mk(boot, runtime)
    for key, state, main_gpu, reserved in residents:
        mgr._residents[key] = _resident(
            key, state=state, main_gpu=main_gpu, reserved_need_mib=reserved)
    return mgr


_ACTIVE_SIB = ("sib", ResidentState.ACTIVE, 0, 18000)


def _loading(key, main_gpu, reserved):
    return (key, ResidentState.RESERVED_LOADING, main_gpu, reserved)


def _admits_after_flake(mgr, recovered, *, main_gpu=0, credit_pending=False):
    """`_vram_admits_locked(_NEED, ...)` with the primary probe read flaking to
    None and the retry recovering with `recovered`."""
    with patch_safety_and_manager(_flake_once_then_recover(recovered)):
        return mgr._vram_admits_locked(
            _NEED, 1, main_gpu, "none", credit_pending=credit_pending)


class TestCoResidenceProbeRetryCoverage:
    """the co-residence probe coverage: reserve_g, the out-of-range idx fallback, and the scoped
    credit, all on the RECOVERED reading of the co-residence retry."""

    # -- reserve_g -----------------------------------------------------------

    @pytest.mark.parametrize(
        "recovered, loadings, expected",
        [
            pytest.param([40000], [_loading("load0", 0, 30000)], False,
                         id="40000-free-minus-30000-reserved-refuses"),
            pytest.param([60000], [_loading("load0", 0, 30000)], True,
                         id="60000-free-minus-30000-reserved-admits"),
            pytest.param([40000],
                         [_loading("load0", 0, 15000), _loading("load1", 0, 15000)], False,
                         id="two-loading-siblings-are-both-subtracted-refuses"),
            pytest.param([70000],
                         [_loading("load0", 0, 15000), _loading("load1", 0, 15000)], True,
                         id="two-loading-siblings-subtracted-once-each-admits"),
        ],
    )
    def test_loading_sibling_reserve_is_subtracted_from_recovered_reading(
            self, tmp_path, recovered, loadings, expected):
        """Pins `reserve_g`: a RESERVED_LOADING sibling on the target card has
        weights the probe cannot show yet, so its `reserved_need_mib` is
        subtracted from the recovered reading, once per sibling. Fails if the
        subtraction is dropped, added instead, or counts only one sibling."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB, *loadings])
        got = _admits_after_flake(mgr, recovered)
        assert got == expected, (
            f"the co-residence probe reserve_g: recovered={recovered} loadings="
            f"{[ld[3] for ld in loadings]} need={_NEED} expected admit={expected}, got {got}"
        )

    @pytest.mark.parametrize(
        "state",
        [ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE],
    )
    def test_non_loading_sibling_reserve_is_not_subtracted_again(self, tmp_path, state):
        """Pins the RESERVED_LOADING filter of `reserve_g`: an ACTIVE, GRACE or
        IDLE_EVICTABLE sibling is already inside the live reading, so its
        `reserved_need_mib` (18000) must NOT be subtracted a second time.
        30000 free admits; subtracting 18000 would leave 12000 and refuse."""
        mgr = _coverage_mgr(tmp_path, [("other", state, 0, 18000)])
        got = _admits_after_flake(mgr, [30000])
        assert got is True, (
            f"the co-residence probe reserve_g: a {state.name} sibling must not be subtracted "
            "again (double count); 30000 free must admit need=18000"
        )

    def test_loading_sibling_on_another_card_is_not_subtracted(self, tmp_path):
        """Pins the `r.main_gpu == main_gpu` filter of `reserve_g`: a sibling
        still loading on card 1 does not take room from card 0. Card 0 has
        40000 free, so need 18000 admits; subtracting card 1's 30000 would refuse."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB, _loading("load1", 1, 30000)])
        got = _admits_after_flake(mgr, [40000, 40000], main_gpu=0)
        assert got is True, (
            "the co-residence probe reserve_g: a RESERVED_LOADING sibling on ANOTHER card must "
            "not be subtracted from the target card's recovered reading"
        )

    # -- idx out of range ----------------------------------------------------

    @pytest.mark.parametrize(
        "main_gpu, recovered, expected",
        [
            pytest.param(7, [80000, 1000], True, id="main_gpu-7-of-2-cards-reads-card-0-room"),
            pytest.param(7, [1000, 80000], False, id="main_gpu-7-of-2-cards-reads-card-0-no-room"),
            pytest.param(-1, [80000, 1000], True, id="main_gpu-minus-1-reads-card-0-room"),
            pytest.param(-1, [1000, 80000], False, id="main_gpu-minus-1-reads-card-0-no-room"),
            pytest.param(2, [80000, 1000], True, id="main_gpu-equal-to-card-count-reads-card-0"),
        ],
    )
    def test_out_of_range_main_gpu_reads_card_zero(
            self, tmp_path, main_gpu, recovered, expected):
        """Pins the `idx` fallback: a main_gpu outside 0..len(vals)-1 (too high,
        negative, or equal to the card count) reads card 0, like `_vram_budget`
        does, instead of raising IndexError or reading the last card. The
        paired room/no-room cases tell card 0 apart from the last card.
        (reserve_g only counts siblings whose main_gpu equals this same
        out-of-range number, so it is empty here; the cases use no loading sibling.)"""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB])
        got = _admits_after_flake(mgr, recovered, main_gpu=main_gpu)
        assert got == expected, (
            f"the co-residence probe idx fallback: main_gpu={main_gpu} recovered={recovered} "
            f"must read card 0 (expected admit={expected}), got {got}"
        )

    @pytest.mark.parametrize(
        "main_gpu, recovered, expected",
        [
            pytest.param(1, [1000, 80000], True, id="card-1-has-room-admits"),
            pytest.param(0, [80000, 1000], True, id="card-0-has-room-admits"),
            pytest.param(1, [80000, 1000], False, id="card-1-has-no-room-refuses"),
        ],
    )
    def test_in_range_main_gpu_reads_its_own_card(
            self, tmp_path, main_gpu, recovered, expected):
        """Pins that the fallback is only for out-of-range: an in-range
        main_gpu reads its own card, so the answer follows that card's room
        and not card 0's."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB])
        got = _admits_after_flake(mgr, recovered, main_gpu=main_gpu)
        assert got == expected, (
            f"the co-residence probe idx: main_gpu={main_gpu} recovered={recovered} must read "
            f"its own card (expected admit={expected}), got {got}"
        )

    # -- credit_pending combine ---------------------------------------------

    @pytest.mark.parametrize(
        "recovered, loadings, pending, expected",
        [
            pytest.param([14000], [], 6000, True, id="14000-plus-6000-credit-admits"),
            pytest.param([14000], [], 3000, False, id="14000-plus-3000-credit-is-short"),
            pytest.param([40000], [_loading("load0", 0, 30000)], 9000, True,
                         id="40000-minus-30000-reserved-plus-9000-credit-admits"),
            pytest.param([30000], [_loading("load0", 0, 15000)], 3000, True,
                         id="exactly-need-admits"),
            pytest.param([30000], [_loading("load0", 0, 15000)], 2999, False,
                         id="one-short-of-need-refuses"),
        ],
    )
    def test_recovered_reading_plus_scoped_credit_admits(
            self, tmp_path, recovered, loadings, pending, expected):
        """Pins the combine: with credit_pending the target card's pending
        reclaim is added ONCE to (recovered reading - reserve_g). Credit that
        is too small (below need on its own, so the fast path is not taken)
        still has to be added: not lost, not doubled, not subtracted, and the
        boundary is `>=`."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB, *loadings])
        mgr._pending_reclaim_mib[0] = pending
        got = _admits_after_flake(mgr, recovered, credit_pending=True)
        assert got == expected, (
            f"the co-residence probe credit combine: recovered={recovered} reserve="
            f"{[ld[3] for ld in loadings]} pending={pending} need={_NEED} "
            f"expected admit={expected}, got {got}"
        )

    @pytest.mark.parametrize(
        "recovered, loadings, pending",
        [
            pytest.param([14000], [], 6000, id="credit-would-have-made-room"),
            pytest.param([40000], [_loading("load0", 0, 30000)], 9000,
                         id="credit-would-have-covered-the-reserve"),
        ],
    )
    def test_credit_not_requested_is_not_added(self, tmp_path, recovered, loadings, pending):
        """Pins the credit_pending flag: pending reclaim exists on the card but
        the caller did not ask for credit (credit_pending=False, the real
        reserve path), so nothing is added and the branch refuses. Never spawn
        onto VRAM a teardown has not yet released."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB, *loadings])
        mgr._pending_reclaim_mib[0] = pending
        got = _admits_after_flake(mgr, recovered, credit_pending=False)
        assert got is False, (
            "the co-residence probe credit combine: pending reclaim must not be credited when "
            "credit_pending is False"
        )

    def test_credit_is_scoped_to_the_target_card(self, tmp_path):
        """Pins the scope of the credit: 6000 pending on card 1 is not credit
        for a split 'none' request on card 0 (14000 free), so the branch
        refuses. Summing every card's pending reclaim would admit."""
        mgr = _coverage_mgr(tmp_path, [_ACTIVE_SIB])
        mgr._pending_reclaim_mib[1] = 6000
        got = _admits_after_flake(mgr, [14000, 14000], main_gpu=0, credit_pending=True)
        assert got is False, (
            "the co-residence probe credit combine: pending reclaim on ANOTHER card must not be "
            "credited to a split 'none' request on card 0"
        )
