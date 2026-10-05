"""Fast Lane must FORCE
placement auto-selection ON regardless of a manifest's `auto_place` flag.
Fast Lane design rule:
"Fast Lane overrides that selection. The manifest expresses a
preference; IT DOES NOT CONSTRAIN PREEMPTION."

THE DEFECT: `_resolve_placement_locked` (manager.py) gated its call into
`_auto_pick_gpu` on the MANIFEST's own `auto_place` flag
(`if auto_place and split_mode == "none":`, manager.py) -- so a manifest
with `auto_place: false` could veto Fast Lane's own placement override. Consequence:
two manifests that were both auto_place:false and
pinned to main_gpu:1 could never
co-reside on a card too small for both, so card 0 stayed
idle while card 1 starved both.

THE FIX, one line (manager.py):
    if (auto_place or self._fastlane_enabled()) and split_mode == "none":
fastlane enabled -> OR term always True -> forced on regardless of manifest.
fastlane disabled -> `auto_place or False` == auto_place, byte-identical to
today. `split_mode == "none"` stays a real precondition, unchanged.

Fixtures mirror tests/test_auto_placer.py's own conventions
(_boot_runtime_multislot / _seed_manifest / _vram) and
the staleness-gate test's RuntimeConfig(fastlane=...)
construction -- both already-established patterns in this suite, not
invented here. `_resolve_placement_locked` is called directly and for real
(no mock of the method under test); `_auto_pick_gpu` runs its real body
against a mocked VRAM probe, same as test_auto_placer.py.

Behaviour changed. The decisive arm is
`auto_place=False, fastlane=enabled` -- the key requirement: this
must fail on the pre-fix code, because it reproduces the real production
failure, not a hypothetical.

⛔ Do NOT judge this fix by whether two real models co-reside on a real
host -- a SECOND, separate
defect (no CUDA_VISIBLE_DEVICES card isolation at split_mode:none -- a
stray context lands on the idle card, leaving it too small for
a second model) that also gates that outcome and is explicitly not this change's
scope. These tests check placement RESOLUTION only, the thing this change
actually changes -- not end-to-end VRAM admission on a real host.
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

MODEL_TAG = "ref-victim"
NEED_MIB = 5000


def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2,
                             fastlane=None):
    """Mirrors tests/test_auto_placer.py's `_boot_runtime_multislot` +
    the staleness-gate test's `_manager`
    fastlane-kwarg convention (fastlane=None omits the section entirely --
    the "never configured" off-case; both off-cases are exercised below)."""
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
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    queue_kwargs = dict(max_parallel_sidecars=max_parallel_sidecars,
                         safety_enabled=False)
    runtime_kwargs = dict(queue=QueueConfig(**queue_kwargs), pull=PullConfig())
    if fastlane is not None:
        runtime_kwargs["fastlane"] = fastlane
    return boot, RuntimeConfig(**runtime_kwargs)


def _seed_manifest(boot, model_tag, *, expected_vram_mib, auto_place,
                    main_gpu=0, split_mode="none"):
    """Verbatim convention from tests/test_auto_placer.py's `_seed_manifest`."""
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": expected_vram_mib * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": expected_vram_mib * 1024 * 1024,
        "auto_place": auto_place,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


@contextmanager
def _vram(vals):
    """Verbatim convention from tests/test_auto_placer.py's `_vram`."""
    with patch("turbohaul.safety._read_free_vram_all_mib", return_value=vals), \
         patch("turbohaul.manager._read_free_vram_all_mib", return_value=vals):
        yield


def _resolved_main_gpu(mgr):
    """`_resolve_placement_locked` returns (need, parallel, main_gpu,
    split_mode, sleep_idle_s, auto_place) -- main_gpu is index 2."""
    return mgr._resolve_placement_locked(MODEL_TAG)[2]


# card0 (index 0) is pinned in the manifest but too small for NEED_MIB; card1
# (index 1) fits easily. If auto-place fires, main_gpu must move 0 -> 1. If it
# does not, main_gpu stays the manifest's pin, 0 -- this is the sole
# observable this whole suite turns on.
_VRAM_FREE = [1000, 20000]


class TestFastlaneForcesAutoPlaceOverManifest:

    def test_KILLING_ARM_auto_place_false_fastlane_enabled_forces_override(
        self, tmp_path,
    ):
        """THE production defect, reproduced as a unit test. On pre-fix code
        this asserts main_gpu == 1 (forced) and OBSERVES main_gpu == 0 (the
        manifest pin, unmoved) -- a genuine WRONG-VALUE failure, not an
        AttributeError or a vacuous pass."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, fastlane=FastLaneConfig(enabled=True),
        )
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=False, main_gpu=0)
        mgr = TurbohaulManager(boot, runtime)
        with _vram(_VRAM_FREE):
            main_gpu = _resolved_main_gpu(mgr)
        assert main_gpu == 1, (
            f"Fast Lane design rule: 'the manifest expresses a preference; it does "
            f"not constrain preemption' -- with Fast Lane ENABLED, "
            f"auto_place=False must NOT veto the auto-placer. Card 0 (the "
            f"manifest's pin) does not fit {NEED_MIB} MiB with only "
            f"{_VRAM_FREE[0]} MiB free; card 1 has {_VRAM_FREE[1]} MiB free "
            f"and fits. Got main_gpu={main_gpu} -- the manifest pin was "
            f"honoured instead of overridden."
        )

    @pytest.mark.parametrize("label,fastlane", [
        ("never_configured", None),
        ("explicitly_disabled", FastLaneConfig(enabled=False)),
    ])
    def test_NO_REGRESSION_ARM_auto_place_false_fastlane_disabled_unchanged(
        self, tmp_path, label, fastlane,
    ):
        """Fast Lane OFF (both ways a real deployment is "off" -- never
        configured, and explicitly disabled) must be BYTE-IDENTICAL to
        today: the manifest governs, main_gpu stays the pin."""
        boot, runtime = _boot_runtime_multislot(tmp_path, fastlane=fastlane)
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=False, main_gpu=0)
        mgr = TurbohaulManager(boot, runtime)
        with _vram(_VRAM_FREE):
            main_gpu = _resolved_main_gpu(mgr)
        assert main_gpu == 0, (
            f"Fast Lane disabled ({label}) must leave auto_place=False's "
            f"veto exactly as it was before this change -- manifest pin (0) "
            f"stands even though card 0 does not fit. Got main_gpu={main_gpu}."
        )

    def test_GREEN_CONTROL_auto_place_true_fastlane_disabled_shares_no_path(
        self, tmp_path,
    ):
        """The pre-existing, wholly untouched auto_place=True path -- this
        change's OR term short-circuits on `auto_place` before `_fastlane_enabled`
        is even called (verified below), so this arm exercises ZERO new
        code and is a genuine control, not a second copy of the fix."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, fastlane=FastLaneConfig(enabled=False),
        )
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=True, main_gpu=0)
        mgr = TurbohaulManager(boot, runtime)
        calls = {"n": 0}
        real = mgr._fastlane_enabled
        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)
        mgr._fastlane_enabled = counting
        with _vram(_VRAM_FREE):
            main_gpu = _resolved_main_gpu(mgr)
        assert main_gpu == 1, "pre-existing auto_place=True behaviour must be unchanged"
        assert calls["n"] == 0, (
            "`or` must short-circuit on a True auto_place -- _fastlane_enabled "
            f"was called {calls['n']} time(s), so this arm shares code with "
            "the fix and is not a clean control"
        )

    def test_auto_place_true_fastlane_enabled_unchanged(self, tmp_path):
        """Both conditions true -- redundant OR, same outcome as today."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, fastlane=FastLaneConfig(enabled=True),
        )
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=True, main_gpu=0)
        mgr = TurbohaulManager(boot, runtime)
        with _vram(_VRAM_FREE):
            main_gpu = _resolved_main_gpu(mgr)
        assert main_gpu == 1

    def test_split_mode_precondition_still_real_not_removed(self, tmp_path):
        """Constraint: split_mode == "none" must stay a real
        gate. A layer-split manifest must NEVER be routed through
        _auto_pick_gpu, fastlane or not -- it spans all cards already."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, fastlane=FastLaneConfig(enabled=True),
        )
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=False, main_gpu=0, split_mode="layer")
        mgr = TurbohaulManager(boot, runtime)
        with _vram(_VRAM_FREE):
            main_gpu = _resolved_main_gpu(mgr)
        assert main_gpu == 0, (
            "split_mode='layer' must bypass the auto-placer entirely even "
            "with Fast Lane enabled -- got main_gpu={main_gpu}, meaning the "
            "split_mode=='none' precondition was weakened or removed"
        )

    def test_auto_place_field_and_manifest_plumbing_are_unchanged(self, tmp_path):
        """The field must not become dead code: verified as behaviour, not
        just 'the field is still in the file': _read_model_footprint must
        still READ auto_place off the manifest and _resolve_placement_locked
        must still RETURN it verbatim, unaffected by Fast Lane's state."""
        boot, runtime = _boot_runtime_multislot(
            tmp_path, fastlane=FastLaneConfig(enabled=True),
        )
        _seed_manifest(boot, MODEL_TAG, expected_vram_mib=NEED_MIB,
                        auto_place=False, main_gpu=0)
        mgr = TurbohaulManager(boot, runtime)
        with _vram(_VRAM_FREE):
            returned_auto_place = mgr._resolve_placement_locked(MODEL_TAG)[5]
        assert returned_auto_place is False, (
            "the manifest's own auto_place value must still be returned "
            "verbatim (index 5) regardless of what the gate DOES with it"
        )
