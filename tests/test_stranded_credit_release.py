"""RELEASE A STRANDED PENDING-RECLAIM CREDIT WHEN
THE CARD WAS RE-USED.

Scenario modelled below (the numbers used throughout the tests):
card 1 carries a 22716 MiB pending-reclaim credit
while ALSO holding a live resident (a text-only quantised model,
admitted after the credit was raised).
_reconcile_orphaned_vram_credit's only release test is an
ABSOLUTE free-VRAM-vs-credit comparison -- on a re-used card the new
resident occupies the freed memory forever, so free_now never reaches
0.9*credit and the credit is immortal: the "keeping credit" log line
repeats and no MAKE_ROOM_* attempt is ever made.
This is the pattern the tests below pin.
The numbers used below are illustrative.

THE FIX: two new write-once-per-event pieces of state --
  Resident.reserved_at_monotonic  -- stamped ONCE at reservation, never
                                      touched again (unlike last_active_
                                      monotonic, which drifts on every later
                                      ACTIVE transition and so is NOT used for
                                      this purpose, for exactly that reason).
  TurbohaulManager._pending_reclaim_raised_at[card] -- the LATEST instant
                                      this card's credit was raised
                                      (overwritten, not accumulated).
If a live resident on a credited card was admitted (reserved_at_monotonic)
AFTER that card's latest raise, this card is no longer over-committed for
admission purposes -- release from registry evidence alone, no VRAM witness
needed. This is NOT proof the full credited amount returned (a smaller
resident can be admitted into a genuinely-free remainder without the full
amount having come back); see the
observation-only release-log wording in _reconcile_orphaned_vram_credit.
Purely additive: the existing absolute-VRAM branch is untouched and is the
only path a card with no qualifying resident ever takes.

Both negative controls: a credited card with NO
resident on it at all must still KEEP the credit (the original correct
behaviour, unregressed); a credited card WITH a resident whose
reserved_at_monotonic is NOT after the raise (an old co-resident, not the
"came back" one) must also fall through to the unchanged absolute path.
"""
import time

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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _resident(tag, resident_key, main_gpu, reserved_at, state=ResidentState.ACTIVE):
    return Resident(
        model_tag=tag, resident_key=resident_key, state=state,
        main_gpu=main_gpu, reserved_at_monotonic=reserved_at,
        last_active_monotonic=reserved_at,
    )


def _raise_probe():
    def _fail(*, device_index=0):
        raise AssertionError(
            "_gpu_used_mib must not be called -- registry evidence alone "
            "must resolve this card"
        )
    return _fail


# ---------------------------------------------------------------------------
# Group 1: POSITIVE -- registry evidence releases without a VRAM witness.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPositiveRegistryEvidenceRelease:
    async def test_releases_from_registry_evidence_alone_real_incident_shape(self, mgr):
        """Reproduces the stranded-credit shape: card 1,
        22716 MiB credit, a resident on main_gpu=1 admitted after the raise.
        _gpu_used_mib is wired to raise if called at all -- the release
        must happen before the absolute-VRAM branch is ever reached."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[1] = 22716
        mgr._pending_reclaim_raised_at[1] = T0
        mgr._residents["qwen3.8-27b-q4-textonly"] = _resident(
            "qwen3.8-27b-q4-textonly", "qwen3.8-27b-q4-textonly", main_gpu=1,
            reserved_at=T0 + 1.0,  # admitted AFTER the raise
        )
        mgr._vram_total_mib = [24467, 24467]
        mgr._gpu_used_mib = _raise_probe()

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(1, 0) == 0, (
            "a resident admitted after the raise must release the credit "
            "from registry evidence alone"
        )

    async def test_releases_even_when_absolute_vram_would_say_keep(self, mgr):
        """Same shape, but _gpu_used_mib is wired to a realistic
        reading (low free VRAM on both cards) instead of
        raising -- proving the release does not merely avoid the VRAM
        witness, it wins against it. Under the unmodified absolute check
        alone this exact reading would KEEP the credit (2374 free MiB is far
        below 0.9 * 22716 ~= 20444)."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[1] = 22716
        mgr._pending_reclaim_raised_at[1] = T0
        mgr._residents["qwen3.8-27b-q4-textonly"] = _resident(
            "qwen3.8-27b-q4-textonly", "qwen3.8-27b-q4-textonly", main_gpu=1,
            reserved_at=T0 + 1.0,
        )
        mgr._vram_total_mib = [24467, 24467]
        mgr._gpu_used_mib = lambda *, device_index=0: [24467 - 1020, 24467 - 2374][device_index]

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(1, 0) == 0


# ---------------------------------------------------------------------------
# Group 2: negative controls.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNegativeControls:
    async def test_credit_with_no_resident_on_card_still_kept(self, mgr):
        """The original correct behaviour, unregressed: a genuinely-not-
        yet-returned credit with NO resident on the card at all must still
        be KEPT -- falls through entirely to the unchanged absolute check,
        which correctly keeps it on insufficient free VRAM."""
        mgr._pending_reclaim_mib[0] = 10000
        mgr._pending_reclaim_raised_at[0] = 1000.0
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = lambda *, device_index=0: 24467 - 500  # only 500 MiB free

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000, (
            "a credit with no resident on the card must stay kept, unchanged"
        )

    async def test_resident_admitted_before_the_raise_does_not_release(self, mgr):
        """A resident IS on the card, but it is an OLD co-resident -- its
        reserved_at_monotonic is BEFORE the credit's raise, not the resident
        that 'came back'. Must fall through to the unchanged absolute path,
        not spuriously release (an additional arm)."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 10000
        mgr._pending_reclaim_raised_at[0] = T0
        mgr._residents["old-co-resident"] = _resident(
            "other-model", "old-co-resident", main_gpu=0, reserved_at=T0 - 5.0,
        )
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = lambda *, device_index=0: 24467 - 500  # only 500 MiB free

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000, (
            "a resident admitted BEFORE the raise must not trigger release"
        )

    async def test_resident_admitted_at_exactly_the_raise_instant_does_not_release(self, mgr):
        """Strict inequality only -- an exact tie is not 'after'."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 10000
        mgr._pending_reclaim_raised_at[0] = T0
        mgr._residents["tied"] = _resident(
            "other-model", "tied", main_gpu=0, reserved_at=T0,
        )
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = lambda *, device_index=0: 24467 - 500

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000

    async def test_resident_on_a_different_card_does_not_release(self, mgr):
        """A newly-admitted resident on a DIFFERENT card must not release
        THIS card's credit -- main_gpu must match."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 10000
        mgr._pending_reclaim_raised_at[0] = T0
        mgr._residents["elsewhere"] = _resident(
            "other-model", "elsewhere", main_gpu=1, reserved_at=T0 + 5.0,
        )
        mgr._vram_total_mib = [24467, 24467]
        mgr._gpu_used_mib = lambda *, device_index=0: 24467 - 500

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000

    async def test_live_release_task_still_defers_the_card_entirely(self, mgr):
        """A card with a LIVE release task is excluded from `candidates`
        before the registry-evidence check ever runs (inherited from the
        existing filter) -- even with a qualifying newer resident present,
        the card must be left alone for whatever the live task will do."""
        import asyncio
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 10000
        mgr._pending_reclaim_raised_at[0] = T0
        mgr._residents["newcomer"] = _resident(
            "other-model", "newcomer", main_gpu=0, reserved_at=T0 + 5.0,
        )
        live_task = asyncio.ensure_future(asyncio.sleep(3600))
        mgr._card_release_tasks[0] = {live_task}
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = _raise_probe()

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000, (
            "a card with a live release task must be left alone entirely"
        )
        live_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await live_task


# ---------------------------------------------------------------------------
# Group 3: LATEST-raise semantics (the conservative choice).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLatestRaiseSemantics:
    async def test_resident_admitted_between_two_accumulating_raises_does_not_release(self, mgr):
        """The credit accumulates from TWO evictions on the same card. A
        resident is admitted between the first and second raise. Because
        _pending_reclaim_raised_at is OVERWRITTEN (latest, not earliest),
        that resident is NOT after the card's recorded raise -- must not
        release.

        This proves the COMPARISON only: given a latest-stamp, a resident
        admitted before it must not release. It does NOT prove the stamp
        actually IS the latest one -- this fixture hand-stamps
        _pending_reclaim_raised_at directly (T0, then T0 + 2.0) and never
        calls _begin_unload_locked, so the production site that overwrites
        the stamp on a real second eviction is never exercised here. The
        latest-not-earliest overwrite behaviour itself has no behavioural
        guard in this suite; closing that gap would mean exercising
        _begin_unload_locked twice on the same card and asserting the stamp
        moves forward, not backward."""
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 5000
        mgr._pending_reclaim_raised_at[0] = T0  # first raise
        mgr._residents["between"] = _resident(
            "other-model", "between", main_gpu=0, reserved_at=T0 + 1.0,
        )
        # second contribution arrives, overwriting the raised-at to LATER
        # than the resident's own admission
        mgr._pending_reclaim_mib[0] += 5000
        mgr._pending_reclaim_raised_at[0] = T0 + 2.0
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = lambda *, device_index=0: 24467 - 500

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 10000, (
            "a resident admitted before the LATEST raise must not release "
            "the accumulated credit"
        )

    async def test_resident_admitted_after_both_raises_releases(self, mgr):
        T0 = 1000.0
        mgr._pending_reclaim_mib[0] = 5000
        mgr._pending_reclaim_raised_at[0] = T0
        mgr._pending_reclaim_mib[0] += 5000
        mgr._pending_reclaim_raised_at[0] = T0 + 2.0  # latest raise
        mgr._residents["after-both"] = _resident(
            "other-model", "after-both", main_gpu=0, reserved_at=T0 + 3.0,
        )
        mgr._vram_total_mib = [24467]
        mgr._gpu_used_mib = _raise_probe()

        await mgr._reconcile_orphaned_vram_credit()

        assert mgr._pending_reclaim_mib.get(0, 0) == 0


# ---------------------------------------------------------------------------
# Group 4: Resident.reserved_at_monotonic wiring at the real construction site.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReservedAtWiredAtConstruction:
    async def test_reserve_and_start_locked_stamps_reserved_at_monotonic(self, mgr, monkeypatch):
        """The real admission path (_reserve_and_start_locked) must stamp
        the new field -- not just the dataclass default. Guards against the
        fix living only in a test-only Resident(...) call and never
        actually reaching production admission.

        ⚠ THIS TEST IS THE SOLE LINK between the stamp and the release logic
        (by construction): every OTHER test in
        this file hand-sets ``reserved_at_monotonic`` directly in its own
        fixture (legitimate unit isolation for testing the release logic on
        its own), so none of them can see a broken PRODUCTION stamp -- a
        change that neuters the fix into a silent no-op by zeroing the
        timestamp at its one real construction site leaves the positive
        release-logic tests green, and is caught by ONLY
        this one. Deleting this test as "redundant" with the release-logic
        tests would silently disconnect every other test arm in this file
        from the code path that actually feeds them real data -- do not
        remove or weaken it without replacing what it uniquely proves.
        """
        from turbohaul.slot import Slot

        mgr._vram_admits_locked = lambda *a, **k: True

        mgr._resolve_placement_locked = lambda *a, **k: (1024, 1, 0, "none", 0, False, False)

        before = time.monotonic()
        slot = Slot.new("some-model")
        async with mgr._registry_lock:
            r = await mgr._reserve_and_start_locked(slot)
        after = time.monotonic()

        assert r is not None
        assert before <= r.reserved_at_monotonic <= after
        assert r.reserved_at_monotonic == r.last_active_monotonic, (
            "both stamps must come from the SAME clock read at reservation"
        )
