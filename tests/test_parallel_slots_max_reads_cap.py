"""parallel_slots.max must read the RUNTIME CAP, unconditionally.

DEFECT: ``{'used': 2, 'max': 1}`` --
``used`` EXCEEDS ``max``. The resident's OWN ``--parallel`` width is 1
(``residents[].parallel == 1``) while ``TURBOHAUL_MAX_PARALLEL`` is 2, and ``max``
read ``handle.parallel`` with a fallback to the cap that fires only when NOTHING
runs. The fraction was therefore wrong precisely while something was serving --
the only time anyone looks at it.

INTENT: show the amount of maximum sidecars allowed (here 2)
versus how many are actually active. ``max`` = the maximum
sidecars ALLOWED (the cap).

EXPECTED BEHAVIOUR:
* cap=2 is set EXPLICITLY in the fixture (the container env lies);
* ONE turn is driven to ACTIVE; ``max == 2`` is asserted WHILE a handle resolves;
* the IDLE-BOX-ONLY test is the VACUOUS one (``max`` already reads 2 when nothing
  runs, so an idle-only assertion passes on the broken tree) -- the assertion here
  holds while a resident is serving, and the serving state arises from real
  traffic, never from a hand-set field;
* Verified to fail on the unpatched base: the active-resident assertion must
  FAIL with the value printed (expected ``max == 1``), observed, not inferred.
"""
from __future__ import annotations

import asyncio
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
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def _boot(tmp_path, *, max_parallel_sidecars):
    """Boot with the cap set EXPLICITLY in the fixture (the container env lies)."""
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
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=0,
            idle_hot_load_seconds=120,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _seed(boot, model_tag):
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


def _spawn_width1():
    """Spawn a handle shaped like the DEPLOYED resident: its OWN --parallel
    width is 1 (`residents[].parallel == 1`) while the cap is 2. That is the
    exact shape in which the defect was observed."""
    pid = [91000]

    def spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None  # is_alive() -> True
        return SidecarHandle(
            proc=proc, port=port, model_tag=model_tag, parallel=1
        )

    return spawn


def _gated_complete(gate):
    async def fake_complete(slot, handle):
        await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    return fake_complete


def _two_vram_cards():
    return patch(
        "turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000]
    )


class TestParallelSlotsMaxReadsCap:
    """max = the maximum sidecars ALLOWED (the runtime cap), read unconditionally --
    not the resident's own --parallel width, which only masked the cap whenever a
    handle resolved."""

    async def test_max_reads_cap_while_handle_resolves(self, tmp_path):
        """Fails on the unpatched base with max == 1
        (handle.parallel shadows the cap=2). On the fixed tree max == 2 while the
        handle resolves. The IDLE case is the vacuous one -- it already reads 2.

        BINDING (a constant-2 mutant): this test ALONE does NOT
        bind `max` to the config -- a hardcoded `2` passes it too (at cap=2 both
        produce 2). It binds the cap only TOGETHER with
        test_one_resident_cap_parallel_slots_output_unchanged: no single constant satisfies
        cap=2 -> 2 AND cap=1 -> 1. Keep both; do not drop either for coverage."""
        boot, runtime = _boot(tmp_path, max_parallel_sidecars=2)
        _seed(boot, "m1")
        gate = asyncio.Event()  # never set -> the turn parks in complete, ACTIVE stays
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=_spawn_width1(), health_fn=_health,
            sigterm_fn=_sigterm, vram_fn=_vram_ok,
            complete_fn=_gated_complete(gate),
        )
        mgr.runtime.queue.safety_enabled = False
        with _two_vram_cards():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            try:
                # PRECONDITION: the serving state must ARISE from real
                # traffic, never be hand-set -- the assertion below must hold while
                # a resident is genuinely ACTIVE.
                active = False
                for _ in range(250):
                    await asyncio.sleep(0.02)
                    active = any(
                        getattr(getattr(r.active_slot, "state", None), "value", None)
                        in ("ACTIVE", "ACTIVE_MATCH")
                        for r in mgr._model_residents()
                    )
                    if active:
                        break
                assert active, (
                    "precondition unmet: no resident ACTIVE after 5s -- the "
                    "assertion below would be vacuous"
                )
                # The defect trigger: a HANDLE RESOLVES. Without it the broken
                # expression falls through to the cap and passes anyway.
                handle = mgr._resolve_top_level_active_handle()
                assert handle is not None, (
                    "precondition unmet: _resolve_top_level_active_handle() "
                    "returned None while a resident is ACTIVE -- the assertion "
                    "below would be vacuous"
                )

                snap = mgr.status_snapshot()
                maxv = snap["parallel_slots"]["max"]
                assert maxv == 2, (
                    f"parallel_slots.max reported {maxv} (cap=2, resident width=1) "
                    f"WHILE a handle resolves ({handle.model_tag} ACTIVE). Pre-fix "
                    f"the handle's OWN --parallel width shadows the cap, so the "
                    f"surface reads {maxv} exactly while something is serving -- "
                    f"the observed value was used=2, max=1 (used exceeds max). "
                    f"max = the maximum sidecars ALLOWED; it must read the cap "
                    f"unconditionally."
                )
            finally:
                gate.set()
                f1.cancel()
                await asyncio.gather(f1, return_exceptions=True)
                await mgr.shutdown()

    async def test_one_resident_cap_parallel_slots_output_unchanged(self, tmp_path):
        """CAP<=1 PIN: at cap 1 the cap IS 1, so dropping the handle preference is
        a no-op -- asserted here, not argued. This passes on the UNPATCHED base
        (watched green) and must still pass after the fix: cap<=1 output unchanged.
        `used` is a separate matter -- untouched by this change.

        BINDING (against a constant-2 implementation): this is the ONLY test in
        the file that discriminates "reads the cap" from "returns a hardcoded
        constant" -- an implementation returning `max: 2` passes
        test_max_reads_cap_while_handle_resolves (cap=2 -> 2 either way) and fails
        HERE (it reports 2; this asserts 1). Together the pair binds `max` to the
        runtime config because no single constant satisfies cap=2 -> 2 AND
        cap=1 -> 1. DO NOT remove this test as a redundant regression pin."""
        boot, runtime = _boot(tmp_path, max_parallel_sidecars=1)
        _seed(boot, "m1")
        gate = asyncio.Event()
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=_spawn_width1(), health_fn=_health,
            sigterm_fn=_sigterm, vram_fn=_vram_ok,
            complete_fn=_gated_complete(gate),
        )
        mgr.runtime.queue.safety_enabled = False
        with _two_vram_cards():
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            f1 = asyncio.create_task(mgr.submit_and_wait("m1", "a", thread_id="t1"))
            try:
                # Cap-1 precondition: the top-level ACTIVE HANDLE resolves (the
                # legacy singleton path; _model_residents() is empty at cap<=1).
                resolving = False
                for _ in range(250):
                    await asyncio.sleep(0.02)
                    resolving = mgr._resolve_top_level_active_handle() is not None
                    if resolving:
                        break
                assert resolving, (
                    "precondition unmet: no active handle resolving at cap=1 after "
                    "5s -- the pin below would be vacuous"
                )

                snap = mgr.status_snapshot()
                used = snap["parallel_slots"]["used"]
                maxv = snap["parallel_slots"]["max"]
                assert used == 1, (
                    f"cap<=1 pin: parallel_slots.used reported {used} with the "
                    f"singleton serving -- `used` is a separate matter; "
                    f"this change must not touch it."
                )
                assert maxv == 1, (
                    f"cap<=1 pin: parallel_slots.max reported {maxv} at cap=1 "
                    f"while a handle resolves. Pre-fix: handle.width(1) or cap(1) "
                    f"= 1. Post-fix: cap = 1. The output at cap<=1 is unchanged "
                    f"by the fix -- this test is that assertion."
                )
            finally:
                gate.set()
                f1.cancel()
                await asyncio.gather(f1, return_exceptions=True)
                await mgr.shutdown()


async def _health(*a, **k):
    return True


async def _sigterm(*a, **k):
    return True, "sigterm-clean"


async def _vram_ok(**k):
    return True, 100
