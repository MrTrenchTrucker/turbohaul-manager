"""The listed_waiter clause this
file's two "worlds" were built on was removed; the two
world-based tests (byte-identical old totals; shield-blocker vs
shield-redundant discrimination) are retired with it. The retained tests
below cover the five surviving clauses.

`_starvation_reason` emitted six ANONYMOUS totals.
The six clauses are evaluated independently, not short-circuited, so one
resident can be counted under several of them -- at `residents=2`, two
completely different worlds can produce BYTE-IDENTICAL totals, making "was
it the shield or the state?" unfalsifiable on the old instrument.

Both worlds below reproduce the exact same totals:
`state_not_idle=1 active_slot=1 listed_waiter=1` either way.

  World A: resident X is ACTIVE with an active_slot (both flags on X);
           resident Y is parked, excluded ONLY by the listed-waiter shield
           -- the shield is the real, sole blocker for Y.
  World B: resident X is not-IDLE but holds no active_slot, AND is (also)
           listed -- the shield "lands on" X, but X was already excluded
           by state, so the shield is redundant there; resident Y is
           parked but holds the active_slot instead.

`why` is untouched by this change (frozen byte-for-byte for
an external regex-based consumer) -- these tests do not assert
on it beyond confirming it did not move.
"""
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
from turbohaul.manager import TurbohaulManager, Resident, ResidentState
from turbohaul.slot import Slot


@pytest.fixture
def boot_and_runtime(tmp_path):
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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent, unused here
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


class TestStarvationReasonPerResident:
# The two world-based tests
# above (test_old_totals_really_are_byte_identical_across_the_two_worlds and
# test_per_resident_discriminates_shield_is_the_blocker_from_shield_is_
# redundant) were retired here -- both premises are built on the
# listed_waiter clause that was removed.

    def test_per_resident_shows_eligible_when_no_clause_excludes_it(self, boot_and_runtime):
        """The other half of the field's own contract: a resident excluded by
        NOTHING reads "<tag>:eligible", not an empty clause string."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["free"] = Resident(
            model_tag="free", resident_key="free",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        result = mgr._starvation_reason()
        assert result["per_resident"] == "free:eligible"
        assert result["eligible"] == 1

    def test_per_resident_names_card_split_on_the_excluded_resident(self, boot_and_runtime):
        """card_split is a SECOND-stage filter (only stage-1 survivors, only
        under a main_gpu pin) -- the per-resident field must still name it
        on the specific resident it excludes, not just bump the total."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["pinned"] = Resident(
            model_tag="pinned", resident_key="pinned",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
            main_gpu=1, split_mode="none",
        )
        result = mgr._starvation_reason(main_gpu=0, split_mode="none")
        assert result["per_resident"] == "pinned:card_split"
        assert result["card_split"] == 1
        assert result["eligible"] == 0

    def test_card_split_lands_on_the_right_resident_with_a_prior_exclusion(
        self, boot_and_runtime,
    ):
        """Alignment discriminator: the card_split second pass must update
        the RIGHT resident's clause list via `survivor_indices`, not
        whichever resident happens to sit at that POSITION within
        `stage1_survivors`. A single-resident fixture (the previous test)
        cannot catch a survivor-index/resident-index misalignment, since the
        two coincide trivially at n=1 -- this needs an excluded resident
        BEFORE the card_split survivor so the two indices genuinely
        diverge (survivor index 0, resident index 1).

        RATIONALE: this arm exists because the
        single-resident card_split test above could not
        see the misalignment defect this arm exists to
        catch.

        ⛔ THE MULTI-RESIDENT SHAPE IS LOAD-BEARING, NOT COSMETIC:
        collapsing this to a single resident SILENTLY DESTROYS THE
        DISCRIMINATION WITHOUT FAILING ANYTHING -- that is exactly how the
        original card_split test came to be unable to catch the
        misalignment defect in the first place (one resident, so
        survivor-index and resident-index coincided). Do not "simplify"
        this fixture down to one resident."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        # resident 0: excluded at stage 1 (not IDLE_EVICTABLE) -- NOT a survivor.
        mgr._residents["busy"] = Resident(
            model_tag="busy", resident_key="busy",
            state=ResidentState.ACTIVE, last_active_monotonic=1.0,
            main_gpu=0, split_mode="none",
        )
        # resident 1: survives stage 1 (survivor index 0), then excluded by card_split.
        mgr._residents["pinned"] = Resident(
            model_tag="pinned", resident_key="pinned",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=2.0,
            main_gpu=1, split_mode="none",
        )
        result = mgr._starvation_reason(main_gpu=0, split_mode="none")
        by_tag = dict(entry.split(":", 1) for entry in result["per_resident"].split(","))
        assert by_tag["pinned"] == "card_split", by_tag
        assert by_tag["busy"] == "state_not_idle", by_tag

    def test_single_resident_output_unchanged_in_meaning(self, boot_and_runtime):
        """Negative control: a single ACTIVE resident, no listed waiters --
        the pre-existing why/totals must read exactly as they always did;
        per_resident is the only new information, additive on top."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["solo"] = Resident(
            model_tag="solo", resident_key="solo",
            state=ResidentState.ACTIVE, last_active_monotonic=1.0,
        )
        result = mgr._starvation_reason()

        assert result["why"] == "reason_unknown"
        assert result["residents"] == 1
        assert result["eligible"] == 0
        assert result["state_not_idle"] == 1
        assert result["active_slot"] == 0
        assert result["inflight"] == 0
        assert result["staleness_protected"] == 0
        assert result["card_split"] == 0
        assert result["per_resident"] == "solo:state_not_idle"
