"""The cross-card relocation must reach the
ENGINE, not just the ledger.

Defect: `_route_or_reserve` records a Fast Lane claimant's relocation onto the
card an off-card eviction just freed (in manager.py, the PINNED arm — i.e.
`manifest.auto_place is False`), threads it through `main_gpu_override` into the
Resident, and gates cross-resident VRAM against it. But `_spawn_for_resident`
re-derives argv from the raw manifest and only substitutes the Resident's
admitted placement `if manifest.auto_place and manifest_split == "none"`
(in manager.py) — a predicate the relocation arm can never satisfy, because it
is the complement of the arm that sets the relocation. The two guards are
mutually exclusive on the SAME field, so the relocated card cannot reach argv by
construction. The manager gates one card while the process binds another —
verbatim the hazard the comment in manager.py exists to prevent for
the auto_place case.

`_run_spawn_safety_gate` (in manager.py) carries the IDENTICAL condition, and
its docstring makes mirroring it a stated invariant:
"Mirrors _spawn_for_resident's argv-override condition exactly, so the gate and
the actual spawn command always agree with each other." Fixing only the argv
site would break that invariant and merely INVERT the mismatch, so both sites
are pinned here.

Arms:
  ARM1  RED — argv must carry the relocated card (end-to-end through the real
        _route_or_reserve -> _reserve_and_start_locked -> _spawn_for_resident).
  ARM2  RED — the spawn safety gate must budget the relocated card too (site 2).
  ARM3  RED — the relocation moves the claimant onto a CARD; it must NOT hand the
        claimant the victim's split topology. DIVERGING fixture (claimant
        'none', victim 'layer') — the existing suite only ever agrees by
        coincidence (the existing cross-card relocation tests have both ends 'none'), which is not
        evidence.
  ARM4  CONTROL — a pinned, NON-relocated resident's argv stays byte-identical.
        This is the over-broadness guard: an unconditional
        "always stamp r.main_gpu/r.split_mode" fix passes ARM1-3 and fails here.
  ARM5  CONTROL — the auto_place path is unchanged.
"""
from __future__ import annotations

import asyncio
import ipaddress
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import yaml

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
from turbohaul.fsm import transition as fsm_transition
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

_CLAIM_TAG = "ref-claimant"
_VICTIM_TAG = "ref-victim"
_CLAIM_IP = "10.0.0.1"
_VICTIM_IP = "10.0.0.9"

# Small model (3000 MiB expected -> ~3050 MiB need after the ctx=2048/f16 KV
# estimate), same generous-margin convention as tests/test_auto_placer.py, so
# KV rounding never flips a pass/fail boundary.
_MIB = 3000


# rule 0 = HIGH priority (the claimant); rule 9 = LOW (the outranked victim).
# tag_ranks pins the resolved rank for an unclassified {"ip": ...} meta, the
# convention the existing cross-card relocation tests use.
def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index,
        raw_address=raw_address,
        address=addr,
        container_name=None,
        match_addresses=frozenset({addr}),
        label=f"rule{index}",
        tag_ranks={"unclassified": 1},
    )


_TABLE = [_rule(0, _CLAIM_IP), _rule(9, _VICTIM_IP)]


def _boot_runtime(tmp_path, *, max_parallel_sidecars=2, safety_enabled=False):
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            safety_min_free_vram_mib=1000,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, main_gpu, split_mode="none",
                   auto_place=False, expected_vram_mib=_MIB):
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


def _fake_handle(model_tag: str, port: int, pid: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks_capturing_argv(spawn_calls):
    """tests/test_auto_placer.py's helper, verbatim in shape: the fake
    spawn seam records the argv the manager actually built."""
    pid = [96000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port, "argv": list(argv)})
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete)


def _mk(boot, runtime, *, safety_enabled=False, **mocks):
    mgr = TurbohaulManager(boot, runtime, **mocks)
    mgr.runtime.queue.safety_enabled = safety_enabled
    mgr._fastlane_table = lambda: _TABLE
    return mgr


def _argv_value(argv, flag):
    """tests/test_auto_placer.py's helper, verbatim. None => the flag is ABSENT from
    argv (flags_to_argv only emits keys present in the dict)."""
    key = "--" + flag.replace("_", "-")
    if key not in argv:
        return None
    return argv[argv.index(key) + 1]


def _idle_victim(gpu, *, split_mode="none"):
    return Resident(
        model_tag=_VICTIM_TAG, resident_key=_VICTIM_TAG,
        state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        main_gpu=gpu, split_mode=split_mode,
        idle_client_meta={"ip": _VICTIM_IP},
    )


def _claimant(tag=_CLAIM_TAG, *, rule_index=0):
    s = Slot.new(tag)
    s.client_meta = {"ip": _CLAIM_IP}
    s.fastlane = FastLaneMatch(
        rule_index=rule_index, raw_address=_CLAIM_IP, label="",
        effective_tag="main", rank=1,
    )
    s.completion_future = asyncio.get_running_loop().create_future()
    # RECEIVED -> STAGED, establishing the SAME precondition
    # queue.py's own enqueue/pop_next path ALWAYS establishes before a slot is
    # visible to _route_or_reserve in production (fsm_transition(slot,
    # SlotState.STAGED), in queue.py). _route_or_reserve has exactly
    # one caller in manager.py (fed exclusively by
    # self.queue.pop_next(), which stages first), never branches on
    # slot.state anywhere in its own body (only reads/writes the
    # fastlane_relocated_* fields), and its own sibling helper documents the
    # invariant explicitly: "every real inbox-drained slot IS already STAGED
    # ... _route_or_reserve hands it to r.inbox with no transition in
    # between" (in manager.py). So this is a hard precondition
    # established BEFORE the call, not something that becomes true on its
    # own the longer you wait -- nothing in _route_or_reserve's own path
    # ever calls fsm_transition(..., STAGED).
    #
    # This test calls mgr._route_or_reserve(claimant) DIRECTLY, bypassing
    # queue.py's staging pipeline entirely, which no real request path can
    # do. Without this line the claimant stays in RECEIVED, which is
    # invisible under most timing (the driver task's own continuation past
    # _spawn_for_resident usually loses the race against this test's
    # immediate mgr.shutdown()) but order/rig-dependently exposes a real,
    # separate production inconsistency the moment the driver wins
    # that race: _serve_on_resident (in manager.py) only
    # conditionally stages STAGED->LOADING but then UNCONDITIONALLY
    # transitions to ACTIVE, raising InvalidTransition(RECEIVED -> ACTIVE)
    # on a slot this test itself left unstaged. That inconsistency is
    # a separate, known issue (NOT fixed here;
    # whether it is reachable via some OTHER real path is a separate
    # question, not this test's). ARM5/ARM4/etc. were never testing that
    # transition -- staging here only restores the precondition every real
    # caller already guarantees, so the auto_place/pinned-argv assertions
    # these arms actually make are reached deterministically regardless of
    # scheduling, on every rig and every pytest-asyncio version tried.
    fsm_transition(s, SlotState.STAGED)
    return s


@contextmanager
def _probe(fn):
    """Dual-patch both binding points (turbohaul.manager re-imports the name),
    the convention documented in tests/test_multislot_concurrency.py."""
    with patch("turbohaul.safety._read_free_vram_all_mib", side_effect=fn), \
         patch("turbohaul.manager._read_free_vram_all_mib", side_effect=fn):
        yield


async def _spawn_argv_for(spawn_calls, tag, *, timeout=5.0):
    """Wait for the driver task to actually reach _spawn_for_resident and
    return the argv it built. Polls rather than sleeping a fixed span."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        hits = [c for c in spawn_calls if c["model_tag"] == tag]
        if hits:
            return hits[-1]["argv"]
        await asyncio.sleep(0.01)
    return None


class TestRelocatedPlacementReachesSpawn:

    async def _drive_relocation(self, tmp_path, *, victim_split="none",
                                claim_split="none", safety_enabled=False,
                                gate_calls=None):
        """Shared driver: a PINNED (auto_place=False) claimant on card 0 that
        strictly outranks an idle resident on card 1, with card 0 too full for
        it. Pass 1 evicts the off-card victim and records the relocation; pass 2
        (standing in for the real evict-pending retry) admits onto the freed
        card through the REAL _reserve_and_start_locked, whose driver reaches
        the REAL _spawn_for_resident. Returns (mgr, claimant, spawn_calls)."""
        boot, runtime = _boot_runtime(tmp_path)
        _seed_manifest(boot, _CLAIM_TAG, main_gpu=0, split_mode=claim_split)
        _seed_manifest(boot, _VICTIM_TAG, main_gpu=1, split_mode=victim_split)
        spawn_calls = []
        mgr = _mk(boot, runtime, safety_enabled=safety_enabled,
                  **_mocks_capturing_argv(spawn_calls))
        mgr._residents[_VICTIM_TAG] = _idle_victim(1, split_mode=victim_split)

        def live_probe():
            # card0 is permanently too full for the claimant, so it can only be
            # admitted by the cross-card widening. card1 frees the moment the
            # victim is gone.
            free1 = 500 if mgr._residents.get(_VICTIM_TAG) is not None else 30000
            return [500, free1]

        claimant = _claimant()
        with _probe(live_probe):
            await mgr._route_or_reserve(claimant)
            # let the real teardown remove the victim
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while (mgr._residents.get(_VICTIM_TAG) is not None
                   and loop.time() < deadline):
                await asyncio.sleep(0.01)
            await mgr._route_or_reserve(claimant)
            argv = await _spawn_argv_for(spawn_calls, _CLAIM_TAG)
        return mgr, claimant, spawn_calls, argv

    async def test_ARM1_relocated_card_reaches_spawn_argv(self, tmp_path):
        """RED on today's code: the claimant is admitted onto card 1 (the card
        the eviction freed) and the Resident records main_gpu=1, but argv still
        carries the manifest's --main-gpu 0. The expectation that the claimant is placed on
        that card must be true of the PROCESS, not only of the ledger."""
        mgr, claimant, spawn_calls, argv = await self._drive_relocation(tmp_path)
        try:
            assert claimant.fastlane_relocated_main_gpu == 1, (
                "precondition: the claim must have been relocated onto the freed card"
            )
            r = mgr._residents.get(_CLAIM_TAG)
            assert r is not None and r.main_gpu == 1, (
                f"precondition: the Resident must record the relocated card, got {r}"
            )
            assert argv is not None, "the claimant never reached _spawn_for_resident"
            assert _argv_value(argv, "main_gpu") == "1", (
                f"argv must bind the RELOCATED card, not the manifest pin. argv={argv}"
            )
        finally:
            await mgr.shutdown()

    async def test_ARM2_relocated_card_reaches_spawn_safety_gate(self, tmp_path):
        """RED on today's code, SECOND SITE. _run_spawn_safety_gate mirrors the
        argv-override condition on purpose (per its docstring: "so the gate
        and the actual spawn command always agree with each other"), so it too
        budgets the manifest card while the reservation admitted the relocated
        one. Fixing only the argv site would keep this red and merely invert the
        mismatch -- which is why both sites are in this change."""
        gate_kwargs = {}

        def fake_gates(**kw):
            gate_kwargs.update(kw)
            return []          # no gate objects -> `failed` empty -> passes

        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates):
            mgr, claimant, spawn_calls, argv = await self._drive_relocation(
                tmp_path, safety_enabled=True)
            try:
                assert claimant.fastlane_relocated_main_gpu == 1, (
                    "precondition: the claim must have been relocated"
                )
                assert gate_kwargs, "the spawn safety gate never ran"
                assert gate_kwargs.get("main_gpu") == 1, (
                    "the spawn safety gate must budget the RELOCATED card -- it is "
                    "the card the cross-resident gate admitted against. got "
                    f"main_gpu={gate_kwargs.get('main_gpu')!r}"
                )
            finally:
                await mgr.shutdown()

    async def test_ARM3_relocation_keeps_claimant_own_split_mode(self, tmp_path):
        """RED on today's code. DIVERGING fixture, required here:
        the claimant is split_mode 'none', the victim is 'layer'. manager.py
        stamps `victim.split_mode` onto the claimant, and nothing constrains the
        two to agree -- _fastlane_outrank_unload_target_locked calls _lru_idle_unloadable
        with main_gpu=None and that function's card/split
        filter is gated on `main_gpu is not None`, so on the
        global widening lookup it is skipped entirely.

        The relocation moves the claimant onto a CARD. It says nothing about the
        claimant adopting the victim's split topology, which is a property of
        the claimant's own model. Today the wrongness is contained only because
        argv never reads it; the ARM1 fix makes it live, so it is fixed here.

        NOTE the existing suite cannot catch this: the cross-card relocation tests assert
        'none' where the victim is ALSO 'none' -- coincidental agreement, not
        evidence."""
        mgr, claimant, spawn_calls, argv = await self._drive_relocation(
            tmp_path, victim_split="layer", claim_split="none")
        try:
            assert claimant.fastlane_relocated_main_gpu == 1, (
                "precondition: the claim must have been relocated onto the freed card"
            )
            assert claimant.fastlane_relocated_split_mode == "none", (
                "the relocation must carry the CLAIMANT's own split_mode, never "
                "the victim's -- got "
                f"{claimant.fastlane_relocated_split_mode!r} (the victim's)"
            )
            r = mgr._residents.get(_CLAIM_TAG)
            assert r is not None and r.split_mode == "none", (
                f"the Resident must record the claimant's own split_mode, got {r}"
            )
            assert argv is not None, "the claimant never reached _spawn_for_resident"
            assert _argv_value(argv, "split_mode") == "none", (
                f"argv must keep the claimant's own split topology. argv={argv}"
            )
        finally:
            await mgr.shutdown()

    async def test_ARM4_CONTROL_pinned_not_relocated_argv_byte_identical(self, tmp_path):
        """OVER-BROADNESS CONTROL. A pinned resident that nothing relocated must
        get argv byte-identical to its manifest. This is the arm that discriminates
        a correct fix from `always stamp r.main_gpu/r.split_mode`: the manifest
        below declares NEITHER flag, and flags_to_argv only emits keys present in
        the dict, so an unconditional stamp would INJECT `--main-gpu 0
        --split-mode layer` where today there are none.

        Named explicitly because test_auto_placer.py
        (test_auto_place_false_stays_pinned_explicit_main_gpu) LOOKS like this
        guard rail but cannot discriminate: its manifest main_gpu is 1 and
        r.main_gpu is also 1, so an over-broad fix passes it unchanged."""
        boot, runtime = _boot_runtime(tmp_path)
        p = boot.storage.manifests_path / f"{_CLAIM_TAG}.yaml"
        p.write_text(yaml.safe_dump({
            "model_tag": _CLAIM_TAG,
            "gguf_blob_sha256": "a" * 64,
            "gguf_size_bytes": _MIB * 1024 * 1024,
            "context_size": 2048,
            "expected_vram_bytes": _MIB * 1024 * 1024,
            "auto_place": False,
            "llama_server_flags": {},          # declares NO placement at all
        }))
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        slot = _claimant()
        with _probe(lambda: [30000, 30000]):
            await mgr._route_or_reserve(slot)
            argv = await _spawn_argv_for(spawn_calls, _CLAIM_TAG)
        try:
            r = mgr._residents.get(_CLAIM_TAG)
            assert r is not None, "the claimant should have been admitted outright"
            assert slot.fastlane_relocated_main_gpu is None, (
                "precondition: nothing may have relocated this slot"
            )
            assert argv is not None, "the claimant never reached _spawn_for_resident"
            assert _argv_value(argv, "main_gpu") is None, (
                "an un-overridden pinned spawn must not gain a --main-gpu it never "
                f"had. argv={argv}"
            )
            assert _argv_value(argv, "split_mode") is None, (
                "an un-overridden pinned spawn must not gain a --split-mode it never "
                f"had. argv={argv}"
            )
        finally:
            await mgr.shutdown()

    async def test_ARM5_CONTROL_auto_place_path_unchanged(self, tmp_path):
        """CONTROL: the auto_place arm keeps overriding argv from the Resident
        exactly as it does today. The fix is additive -- it must not disturb the
        path that already works."""
        boot, runtime = _boot_runtime(tmp_path)
        _seed_manifest(boot, _CLAIM_TAG, main_gpu=0, split_mode="none",
                       auto_place=True)
        spawn_calls = []
        mgr = _mk(boot, runtime, **_mocks_capturing_argv(spawn_calls))
        slot = _claimant()
        # card0 tight, card1 ample -> the auto-placer picks card1 over the
        # manifest's main_gpu 0.
        with _probe(lambda: [500, 30000]):
            await mgr._route_or_reserve(slot)
            argv = await _spawn_argv_for(spawn_calls, _CLAIM_TAG)
        try:
            r = mgr._residents.get(_CLAIM_TAG)
            assert r is not None and r.main_gpu == 1, (
                f"precondition: the auto-placer should have picked card1, got {r}"
            )
            assert slot.fastlane_relocated_main_gpu is None, (
                "the auto_place arm must set NO relocation override "
                "(manager.py:5535-5539)"
            )
            assert argv is not None, "the claimant never reached _spawn_for_resident"
            assert _argv_value(argv, "main_gpu") == "1", (
                f"auto_place argv must still follow the auto-picked card. argv={argv}"
            )
            assert _argv_value(argv, "split_mode") == "none", f"argv={argv}"
        finally:
            await mgr.shutdown()
