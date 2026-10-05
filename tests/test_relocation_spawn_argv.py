"""Relocation spawn argv:
cross-card relocation must reach the SPAWN ARGV (and the gate-side
mirror), and the relocation stamp must carry the CLAIMANT's own
split_mode, not the victim's.

Bug:
  1. _spawn_for_resident's argv-override condition
     (``manifest.auto_place and manifest_split == "none"``) never fires
     for a PINNED manifest (auto_place=False) -- so a relocated claimant
     whose reservation was admitted on the relocated (freed) card via
     main_gpu_override/split_mode_override (cross-card relocation) still spawns
     on the manifest's ORIGINAL card: the VRAM gate checks card B, the
     process binds card A.
  2. The relocation stamp site (the ``slot.fastlane_relocated_split_mode =
     victim.split_mode`` line in _route_or_reserve's pinned arm) stamps
     the VICTIM's split_mode onto the claimant -- split mode is a
     property of the claimant's model, not of whoever it evicted.

Fix shape: a
placement-PROVENANCE bit (Resident.placement_overridden), set iff the
reservation actually PASSED a placement override, consumed as an ADDITIVE
DISJUNCT at the spawn-argv site AND the gate-side mirror. A value
comparison (r.main_gpu != manifest main_gpu) would be unsafe -- it
fires spuriously exactly when an operator edits
main_gpu in the real reserve->spawn window (the driver runs as a separate
asyncio task, so that window is real).

Tests (1, 2 and 5 fail without the change, 3 and 4 pass either way):
  1.  pinned manifest + relocation in force -> argv carries the
      relocated card (the discriminating test).
  2.  the gate-side mirror checks the relocated card for a relocated
      resident (gate and spawn keep agreeing).
  3.  pinned manifest, NO relocation, manifest edited between reserve and
      spawn -> argv follows the FRESH read (the mid-window contract the
      value-comparison approach would break; GREEN both sides).
  4.  auto_place override path unchanged (GREEN both sides).
  5.  relocation with victim split_mode != claimant split_mode -> the
      relocation stamp carries the CLAIMANT's split_mode (RED pre-fix).
"""
import asyncio
import ipaddress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import turbohaul.manager as manager_mod
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
from turbohaul.fastlane import CompiledRule, FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot

NEED = 100


def _match(rule_index=0, rank=1):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag="main", rank=rank,
    )


def _rule(index, raw_address, tag_ranks=None):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index,
        raw_address=raw_address,
        address=addr,
        container_name=None,
        match_addresses=frozenset({addr}),
        label=f"rule{index}",
        tag_ranks=tag_ranks or {},
    )


# rule 0 = HIGH priority ("the outranking claimant"); rule 9 = LOW ("the
# outranked victim"). Same convention as the cross-card relocation
# test in this suite.
TABLE = [
    _rule(0, "10.0.0.1", tag_ranks={"unclassified": 1}),
    _rule(9, "10.0.0.9", tag_ranks={"unclassified": 1}),
]


class _FakeVram:
    """Per-card free-MiB ledger (same shape as the cross-card relocation test's -- this
    file is about the spawn-argv/gate stamp, not VRAM sizing)."""

    def __init__(self, free):
        self.free = dict(free)

    def admits(self, need, parallel, main_gpu, split_mode, **_kw):
        return self.free.get(main_gpu, 0) >= need

    def free_card(self, card):
        self.free[card] = 10 ** 9


class _FakeManifest:
    """Duck-typed manifest for the fresh-read seam: only the attributes
    _spawn_for_resident / _run_spawn_safety_gate actually touch."""

    def __init__(self, auto_place, main_gpu, split_mode="none"):
        self.auto_place = auto_place
        self.llama_server_flags = {"main_gpu": main_gpu, "split_mode": split_mode}
        self.gguf_blob_sha256 = "0" * 64
        self.mmproj_blob_sha256 = None
        self.spec_draft_gguf_blob_sha256 = None
        self.expected_vram_bytes = 0
        self.gguf_size_bytes = 0
        self.context_size = 0
        self.hybrid_kv_ratio = 0.0


class _SpawnStop(Exception):
    """Marker exception raised from the _spawn seam: capture the argv and stop the
    lifecycle there -- no real engine, no health-wait, no registry
    mutation beyond what the real method already did pre-spawn."""


def _flag(argv, name):
    """Value of a `--flag value` pair in a flags_to_argv output."""
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


@pytest.fixture
def mgr(tmp_path):
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
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=8),
        pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    m._fastlane_table = lambda: TABLE
    return m


def _slot(tag):
    s = Slot.new(tag)
    s.completion_future = asyncio.get_event_loop().create_future()
    return s


def _pinned_claimant(tag, rule_index=None):
    s = _slot(tag)
    if rule_index is not None:
        s.fastlane = _match(rule_index=rule_index)
    return s


def _idle_resident(tag, gpu, holder_ip, split_mode="none", last_active=1.0):
    return Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=last_active,
        main_gpu=gpu, split_mode=split_mode,
        idle_client_meta={"ip": holder_ip},
    )


def _relocated_resident(tag, main_gpu, split_mode):
    """A Resident as _reserve_and_start_locked builds one for a RELOCATED
    reservation: the relocation arm passed main_gpu_override/split_mode_override
    into the reservation, so the placement provenance bit is True. Pre-fix
    the attribute does not exist yet -- plain-dataclass assignment still
    lands it and the pre-fix code never reads it (the argv assertion is
    the discriminator); post-fix it is exactly the value the real
    reservation path stamps for a relocation."""
    r = Resident(
        model_tag=tag, resident_key=tag,
        state=ResidentState.RESERVED_LOADING,
        main_gpu=main_gpu, split_mode=split_mode,
    )
    r.placement_overridden = True
    return r


@pytest.mark.asyncio
class TestFixARelocationSpawnArgv:
    async def test_pinned_manifest_relocated_resident_spawns_on_relocated_card(self, mgr, monkeypatch):
        """THE discriminating test. Pinned manifest (auto_place=False,
        manifest main_gpu=0) + a resident whose reservation was relocated
        onto card 1 (provenance bit set by the real reservation path): the
        spawn argv must carry --main-gpu 1, not the manifest's original 0.
        RED pre-fix (argv carries 0), GREEN post-fix."""
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: _FakeManifest(
                auto_place=False, main_gpu=0, split_mode="none"),
        )
        captured = {}

        def _spawn(binary, gguf, port, tag, argv, binary_fd=None):
            captured["argv"] = argv
            raise _SpawnStop

        mgr._spawn = _spawn

        r = _relocated_resident("m1", main_gpu=1, split_mode="none")
        slot = _slot("m1")
        with pytest.raises(_SpawnStop):
            await mgr._spawn_for_resident(r, slot)
        argv = captured["argv"]
        assert _flag(argv, "--main-gpu") == "1", (
            f"the relocated claimant must spawn on the relocated card (1), "
            f"not the manifest's original card (0); "
            f"argv main_gpu={_flag(argv, '--main-gpu')!r}"
        )
        assert _flag(argv, "--split-mode") == "none"

    async def test_gate_mirror_checks_relocated_card(self, mgr, monkeypatch):
        """The gate-side mirror (_run_spawn_safety_gate) must check the
        SAME card the spawn argv binds for a relocated resident -- its own
        docstring contract: 'Mirrors _spawn_for_resident's argv-override
        condition exactly, so the gate and the actual spawn command always
        agree with each other'. RED pre-fix (gate checks the manifest's
        card 0), GREEN post-fix (card 1)."""
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: _FakeManifest(
                auto_place=False, main_gpu=0, split_mode="none"),
        )
        seen = {}

        def _gates(**kw):
            seen.update(kw)
            return [SimpleNamespace(ok=True, name="fake")]

        monkeypatch.setattr(manager_mod, "all_safety_gates", _gates)
        mgr._attn_kv_dims_for = lambda m: None
        mgr._expert_offloaded_mib_for = lambda m: 0

        r = _relocated_resident("m1", main_gpu=1, split_mode="none")
        slot = _slot("m1")
        # _run_spawn_safety_gate returns a list[GateResult], empty when
        # every gate passes; this test asserts placement-card mirroring,
        # not the return
        # type itself.
        # An empty list means pass.
        detail = await mgr._run_spawn_safety_gate(r, slot)
        assert detail == []
        assert seen["main_gpu"] == 1, (
            f"the gate must check the relocated card (1) -- the same card "
            f"the spawn argv binds -- got {seen['main_gpu']!r}"
        )
        assert seen["split_mode"] == "none"

    async def test_pinned_no_relocation_fresh_read_wins_mid_window(self, mgr, monkeypatch):
        """REGRESSION CONTROL for the rejected value-comparison approach.
        Pinned manifest, NO relocation in force (provenance bit False):
        the reservation read main_gpu=0, and an operator edited the
        manifest to main_gpu=2 in the real reserve->spawn window. The
        FRESH read must win: argv carries 2. GREEN before AND after the
        fix -- a value-comparison condition (r.main_gpu != manifest
        main_gpu) would spuriously override the fresh read with 0 and
        fails exactly here. This pins the mid-window contract the
        rejected approach would break (see _run_spawn_safety_gate's docstring)."""
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: _FakeManifest(
                auto_place=False, main_gpu=2, split_mode="none"),  # edited mid-window
        )
        captured = {}

        def _spawn(binary, gguf, port, tag, argv, binary_fd=None):
            captured["argv"] = argv
            raise _SpawnStop

        mgr._spawn = _spawn

        # Reserved against the PRE-edit manifest (main_gpu 0); no
        # relocation happened, so the provenance bit stays False (the
        # dataclass default -- the test must NOT set it).
        r = Resident(
            model_tag="m1", resident_key="m1",
            state=ResidentState.RESERVED_LOADING,
            main_gpu=0, split_mode="none",
        )
        slot = _slot("m1")
        with pytest.raises(_SpawnStop):
            await mgr._spawn_for_resident(r, slot)
        argv = captured["argv"]
        assert _flag(argv, "--main-gpu") == "2", (
            f"no relocation in force: the fresh manifest read must win "
            f"(mid-window contract); argv main_gpu={_flag(argv, '--main-gpu')!r}"
        )

    async def test_auto_place_override_path_unchanged(self, mgr, monkeypatch):
        """The EXISTING auto_place override must stay exactly
        as it is: auto_place=True + manifest split_mode=='none' -> argv
        carries r.main_gpu/r.split_mode regardless of the manifest's own
        main_gpu. GREEN before AND after the fix (the fix's disjunct adds
        nothing for an auto_place resident)."""
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: _FakeManifest(
                auto_place=True, main_gpu=3, split_mode="none"),  # manifest says 3
        )
        captured = {}

        def _spawn(binary, gguf, port, tag, argv, binary_fd=None):
            captured["argv"] = argv
            raise _SpawnStop

        mgr._spawn = _spawn

        # auto_place resident: the auto-placer's card (1) already wins via
        # the pre-existing condition -- no provenance bit involved.
        r = Resident(
            model_tag="m1", resident_key="m1",
            state=ResidentState.RESERVED_LOADING,
            main_gpu=1, split_mode="none",
        )
        slot = _slot("m1")
        with pytest.raises(_SpawnStop):
            await mgr._spawn_for_resident(r, slot)
        argv = captured["argv"]
        assert _flag(argv, "--main-gpu") == "1", (
            "auto_place=True + split 'none' must keep overriding the "
            "manifest's main_gpu with the auto-placer's card, unchanged"
        )

    async def test_relocation_stamps_claimant_split_mode_not_victim(self, mgr):
        """The relocation stamp, end-to-end through the REAL _route_or_reserve
        pinned arm: a pinned claimant (own resolved split_mode 'none')
        strictly outranks an off-card idle victim whose split_mode is
        'layer'. The relocation must stamp the CLAIMANT's own resolved
        split_mode, not victim.split_mode -- split mode is a property of
        the claimant's model, not of whoever it evicted. RED pre-fix
        (stamps 'layer'), GREEN post-fix (stamps 'none')."""
        vram = _FakeVram({0: 0, 1: 0})
        mgr._vram_admits_locked = vram.admits
        # Claimant's own resolved placement: pinned card 0, split_mode 'none'.
        mgr._resolve_placement_locked = lambda tag: (NEED, 1, 0, "none", 0, False, False)
        # Off-card victim on card 1 with a DIFFERENT split_mode:
        mgr._residents["vic-1"] = _idle_resident(
            "vic-1", gpu=1, holder_ip="10.0.0.9", split_mode="layer")
        evicted = []
        mgr._begin_unload_locked = lambda r: (
            evicted.append(r.model_tag), vram.free_card(r.main_gpu))
        mgr._reserve_and_start_locked = AsyncMock(return_value=None)

        claimant = _pinned_claimant("claimant", rule_index=0)  # HIGH -- outranks rule 9

        await mgr._route_or_reserve(claimant)

        assert evicted == ["vic-1"], f"the off-card outranked victim must be evicted, got {evicted}"
        assert claimant.fastlane_relocated_main_gpu == 1
        assert claimant.fastlane_relocated_split_mode == "none", (
            f"the relocation stamp must carry the CLAIMANT's own resolved "
            f"split_mode ('none'), not the victim's ('layer'); got "
            f"{claimant.fastlane_relocated_split_mode!r}"
        )
        # And the retry's reserve call must thread the same claimant-side
        # values through (the retry retargets from these two slot fields).
        vram.free[0] = 10 ** 9
        await mgr._route_or_reserve(claimant)
        mgr._reserve_and_start_locked.assert_called_once_with(
            claimant, main_gpu_override=1, split_mode_override="none",
        )
