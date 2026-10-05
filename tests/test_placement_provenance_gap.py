"""The placement PROVENANCE gap: a THIRD producer moves the card
and sets no provenance bit, so the VRAM gate and the spawned process disagree.

Placement carries PROVENANCE, not just a value, and
that `manifest.auto_place is no longer a complete test for that`. Its bit is
produced in `_reserve_and_start_locked`:

    placement_overridden = (main_gpu_override is not None
                            or split_mode_override is not None)

and consumed by BOTH `_run_spawn_safety_gate` and the argv build in
`_spawn_for_resident`, which must keep the EXACT same disjunction:

    if (m.auto_place and msm == "none") or r.placement_overridden:  -> use r.*

⛔ THE GAP: `_resolve_placement_locked` is a THIRD way placement changes.

    if (auto_place or self._fastlane_enabled()) and split_mode == "none":
        auto_gpu, auto_split = self._auto_pick_gpu(need)
        if auto_gpu is not None:
            main_gpu, split_mode = auto_gpu, auto_split

It assigns the LOCALS and returns them, and returns the MANIFEST's `auto_place`
VERBATIM -- so no caller can distinguish "the manifest asked for auto-placement"
from "Fast Lane forced it". The value is used; the PROVENANCE was never
produced. `placement_overridden` stays False.

AFFECTED POPULATION: a manifest with `auto_place: false` AND `split_mode: none`,
with Fast Lane ENABLED. The resolver forces the auto-pick, so `r.main_gpu` is
the chosen card -- but `m.auto_place` is False and `placement_overridden` is
False, so BOTH disjuncts are False at BOTH consumers and each keeps the
MANIFEST's card. The same hazard, for a new producer.

⚠ LATENT, NOT FIRING TODAY, and this file does not pretend otherwise: in
the usual configuration every manifest reads auto_place=True, which
satisfies the first disjunct. `auto_place: false` is a supported setting and is
exactly the population the Fast Lane forcing rule exists to override.

★ WHY THIS WAS MISSED: the existing test asserts
only on `_resolve_placement_locked`'s RETURN VALUE (every assertion in
the Fast Lane auto-place forcing test is on `main_gpu`). It never crosses
the spawn seam, so the mismatch it created was invisible to it. These arms cross
that seam: they drive the REAL `_route_or_reserve` -> REAL
`_reserve_and_start_locked` -> REAL `_spawn_for_resident` and read the argv the
manager actually built.

THE PATH: `_route_or_reserve`'s "under cap + fits" admission calls
`_reserve_and_start_locked(slot, main_gpu_override=slot.fastlane_relocated_main_gpu,
split_mode_override=slot.fastlane_relocated_split_mode)` -- both None unless a
PRIOR pass relocated the slot. On a plain first admission both are None, so the
provenance bit is False while the resolver has already moved the card. That is
the ordinary path, not an exotic one.

Harness helpers are imported from the relocated-placement spawn-seam test module
rather than copied: that file's spawn-seam capture is PROVEN to observe the argv
the manager really built (it is what made that file's own arms go red), and a
second hand-rolled copy of the one seam this clause turns on would be a
divergence risk for no gain.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import yaml

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import TurbohaulManager

from test_relocated_placement_reaches_spawn import (  # proven seam
    _argv_value,
    _mocks_capturing_argv,
    _spawn_argv_for,
)

pytestmark = pytest.mark.asyncio

_TAG = "ref-pinned"
_NEED_MIB = 5000
# card 0 is the manifest's pin and CANNOT fit _NEED_MIB; card 1 can. If the
# auto-picker fires, the resolved card is 1. This is the sole observable the
# whole file turns on.
_VRAM_FREE = [1200, 20000]


def _boot_runtime(tmp_path, *, fastlane_enabled: bool):
    root = tmp_path / "state"
    root.mkdir()
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
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
            safety_min_free_vram_mib=100,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=fastlane_enabled),
    )
    return boot, runtime


def _seed(boot, *, auto_place: bool, main_gpu=0, split_mode="none"):
    p = boot.storage.manifests_path / f"{_TAG}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": _TAG,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": _NEED_MIB * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": _NEED_MIB * 1024 * 1024,
        "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


@contextmanager
def _vram(vals):
    with patch("turbohaul.safety._read_free_vram_all_mib", return_value=vals), \
         patch("turbohaul.manager._read_free_vram_all_mib", return_value=vals):
        yield


async def _drive(tmp_path, *, auto_place, fastlane_enabled, safety_enabled=False,
                 vram=None):
    """REAL _route_or_reserve -> _reserve_and_start_locked -> _spawn_for_resident."""
    from turbohaul.slot import Slot

    boot, runtime = _boot_runtime(tmp_path, fastlane_enabled=fastlane_enabled)
    _seed(boot, auto_place=auto_place)
    spawn_calls = []
    mgr = TurbohaulManager(boot, runtime, **_mocks_capturing_argv(spawn_calls))
    mgr.runtime.queue.safety_enabled = safety_enabled

    slot = Slot.new(_TAG, prompt="hi", thread_id="t-placement-gap")
    with _vram(vram or _VRAM_FREE):
        await mgr._route_or_reserve(slot)
        argv = await _spawn_argv_for(spawn_calls, _TAG)
    return mgr, slot, argv


class TestPlacementProvenanceGap:

    async def test_ARM1_resolved_card_reaches_spawn_argv(self, tmp_path):
        """RED on today's code. The resolver forces the auto-pick (Fast Lane ON
        overrides auto_place:false), the Resident records the RESOLVED card, but
        argv still carries the manifest's --main-gpu 0 because no provenance bit
        was produced. The gate budgeted one card; the process binds another."""
        mgr, slot, argv = await _drive(
            tmp_path, auto_place=False, fastlane_enabled=True)
        try:
            r = mgr._residents.get(_TAG)
            # PRECONDITIONS -- non-vacuity. Without these the arm could pass on
            # a run where the auto-picker never fired at all, which would prove
            # nothing about provenance.
            assert r is not None, "the model never reached a Resident"
            assert r.main_gpu == 1, (
                f"precondition: the resolver must have moved the card 0 -> 1 "
                f"(card 0 has {_VRAM_FREE[0]} MiB free, need {_NEED_MIB}); "
                f"got main_gpu={r.main_gpu}"
            )
            # CLAIM, not a precondition. (It must not be a precondition:
            # a precondition asserting the DEFECT can
            # only hold pre-fix, so the arm could never go green. What is
            # genuinely a precondition is that the card MOVED, above.)
            assert r.placement_overridden is True, (
                "the resolver applied the auto-pick, so the placement DID come "
                "from an override and the provenance bit must say so -- this is "
                "the bit both consumers read"
            )
            assert argv is not None, "the model never reached _spawn_for_resident"
            assert _argv_value(argv, "main_gpu") == "1", (
                f"argv must bind the RESOLVED card (1) -- the one the "
                f"cross-resident VRAM gate admitted against -- not the "
                f"manifest's pin. argv main_gpu={_argv_value(argv, 'main_gpu')!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_ARM2_resolved_card_reaches_spawn_safety_gate(self, tmp_path):
        """RED on today's code, SECOND CONSUMER. The gate mirrors the argv
        disjunction deliberately, so it budgets the manifest card too. Fixing
        only one site would not fix the mismatch, it would INVERT it -- which is
        why both consumers are covered here."""
        gate_kwargs = {}

        def fake_gates(**kw):
            gate_kwargs.update(kw)
            return []

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates):
            mgr, slot, argv = await _drive(
                tmp_path, auto_place=False, fastlane_enabled=True,
                safety_enabled=True)
            try:
                r = mgr._residents.get(_TAG)
                assert r is not None and r.main_gpu == 1, (
                    f"precondition: resolver must have moved the card, got {r}"
                )
                assert gate_kwargs, "the spawn safety gate never ran"
                assert gate_kwargs.get("main_gpu") == 1, (
                    f"the spawn safety gate must budget the RESOLVED card, got "
                    f"main_gpu={gate_kwargs.get('main_gpu')!r}"
                )
            finally:
                await mgr.shutdown()

    async def test_ARM3_CONTROL_fastlane_disabled_keeps_the_manifest_card(
        self, tmp_path
    ):
        """★ CONTROL, must PASS on both arms. With Fast Lane OFF and
        auto_place:false the resolver's guard is False, so nothing moves the
        card and argv legitimately carries the manifest pin. This shares no code
        path with the provenance bit -- it is the arm that proves the fix is not
        simply forcing r.main_gpu into argv unconditionally."""
        # ⚠ card 0 must FIT here. With the killing arms' VRAM ([1200, 20000])
        # and Fast Lane OFF nothing can move the model off its too-small pin, so
        # it is never admitted and never becomes a Resident -- the control would
        # die on its own setup and prove nothing. Both cards roomy: the resolver
        # still does not fire (auto_place:false + Fast Lane OFF), so the card
        # stays 0 because nothing moved it, which is the claim under test.
        mgr, slot, argv = await _drive(
            tmp_path, auto_place=False, fastlane_enabled=False,
            vram=[20000, 20000])
        try:
            r = mgr._residents.get(_TAG)
            assert r is not None and r.main_gpu == 0, (
                f"with Fast Lane OFF and auto_place:false nothing may move the "
                f"card; got main_gpu={r.main_gpu if r else None}"
            )
            # ⚠ THE ASSERTION THAT GIVES THIS CONTROL TEETH: the test is weak
            # without it. Asserting only on argv
            # CANNOT catch an over-broad bit here: if `placement_overridden`
            # were raised when nothing overrode anything, the consumers would
            # switch to `r.main_gpu` -- which in this fixture EQUALS the
            # manifest card, so argv reads "0" either way and the two branches
            # are observationally identical. That file's own ARM4 records the
            # same trap ("r.main_gpu is also 1, so an over-broad fix passes it
            # unchanged"). The BIT is the thing this fix produces, so the bit is
            # what the no-override direction must assert.
            assert r.placement_overridden is False, (
                "nothing overrode placement here (Fast Lane OFF, auto_place "
                "false, the resolver's guard is False), so the provenance bit "
                "must stay False -- a bit that is always raised is not "
                "provenance, it is a constant"
            )
            assert argv is not None, "the model never reached _spawn_for_resident"
            assert _argv_value(argv, "main_gpu") == "0", (
                f"argv must keep the manifest pin when nothing overrode it; got "
                f"{_argv_value(argv, 'main_gpu')!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_ARM4_CONTROL_auto_place_true_already_agreed(self, tmp_path):
        """★ CONTROL, must PASS on both arms. auto_place:true satisfies the
        FIRST disjunct (`m.auto_place and msm == "none"`), so this population
        already agreed before this fix and must still agree after. It bounds the
        fix: whatever provenance is added must not disturb the arm that was
        never broken."""
        mgr, slot, argv = await _drive(
            tmp_path, auto_place=True, fastlane_enabled=True)
        try:
            r = mgr._residents.get(_TAG)
            assert r is not None and r.main_gpu == 1, (
                f"precondition: auto_place:true must still auto-pick card 1, "
                f"got {r.main_gpu if r else None}"
            )
            assert _argv_value(argv, "main_gpu") == "1", (
                f"auto_place:true already reached argv via the first disjunct "
                f"and must continue to; got {_argv_value(argv, 'main_gpu')!r}"
            )
        finally:
            await mgr.shutdown()
