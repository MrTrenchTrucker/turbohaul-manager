"""The loaded_priority_keys PRODUCER must resolve resident identity like its sibling.

``TurbohaulManager._fastlane_pop_kwarg`` builds ``FastLanePopPolicy.loaded_priority_keys``
-- the set a cold-jump candidate's own key is compared against to decide the swap-budget
exemption in ``queue.py``. It resolves each resident's Fast Lane identity from
``r.idle_client_meta`` ALONE, with no state filter and no ``rank_client_meta`` fallback.

Its sibling ``_resident_priority_key`` (manager.py) resolves
the identical question with the rank-first rule::

    meta = r.rank_client_meta
    if meta is None and r.state is ResidentState.IDLE_EVICTABLE:
        meta = r.idle_client_meta

and its docstring calls ``idle_client_meta`` "a WRONG identity (not merely a missing one) for a
resident now mid-turn for a different client". That docstring enumerates the ONE excused
``idle_client_meta``-only consumer (``_unload_priority_key_for_meta``) and gives the excuse --
"only ever invoked on already-PARKED residents". ``_fastlane_pop_kwarg`` is a SECOND such
consumer that the docstring does not name and whose excuse does not apply: it iterates
``_model_residents()`` with no park filter.

The divergence runs in BOTH directions, which is why a consumer-side change ("``all(())`` is
vacuously true, make it fail closed") addresses neither half:

  * MISSING identity -- a resident with a perfectly resolvable ``rank_client_meta`` contributes
    nothing, manufacturing an empty tuple that reads to the consumer as "nothing is loaded".
  * WRONG identity -- a resident mid-turn for a DIFFERENT, unlisted client contributes the
    PREVIOUS client's key.

Reachability of each fixture below is derived from source, not assumed:

  * ``rank_client_meta`` is stamped at reserve (manager.py) and at every one of the four
    ``r.state = ResidentState.ACTIVE`` assignments, all UNGATED,
    and is never cleared -- so it persists past a park.
  * ``idle_client_meta`` is written on a model-keyed resident at only two sites, both at
    park and both gated on ``parallel == 1``. (A third site writes the singleton, which
    ``_model_residents()`` excludes.)

Consequently an ACTIVE resident ALWAYS carries ``rank_client_meta``, and a ``parallel >= 2``
resident NEVER carries ``idle_client_meta`` -- see ARM 3, which is the case Fast Lane exists to
arbitrate and which the current producer can never see.
"""

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import Resident, ResidentState, TurbohaulManager

# Rule 0 -> key (0, 1); rule 1 -> key (1, 2). 203.0.113.77 is deliberately UNLISTED.
_LISTED_A = {"ip": "192.0.2.5", "is_main": True}
_LISTED_B = {"ip": "192.0.2.9", "is_main": True}
_UNLISTED = {"ip": "203.0.113.77", "is_main": True}


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
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


class TestLoadedPriorityKeysProducerIdentity:
    """The producer must answer the identity question the same way its sibling does."""

    def _mgr(self, boot_and_runtime):
        boot, runtime = boot_and_runtime
        runtime = RuntimeConfig(
            queue=runtime.queue,
            pull=runtime.pull,
            fastlane=FastLaneConfig(
                enabled=True,
                rules=[
                    FastLaneRule(address="192.0.2.5", tag_ranks=FastLaneTagRanks(main=1)),
                    FastLaneRule(address="192.0.2.9", tag_ranks=FastLaneTagRanks(main=2)),
                ],
            ),
        )
        return TurbohaulManager(boot, runtime)

    def _keys(self, mgr):
        policy = mgr._fastlane_pop_kwarg()
        assert policy is not None, "fastlane is enabled; the policy must not be None"
        return policy.loaded_priority_keys

    # ---------------------------------------------------------------- ARM 1

    def test_ARM1_active_first_turn_resident_contributes_its_rank_identity(
        self, boot_and_runtime
    ):
        """MISSING-identity direction. A resident on its FIRST turn has
        rank_client_meta stamped at reserve (manager.py, whose comment says
        outright "idle_client_meta stays None until [it parks]"). Reading
        idle_client_meta alone therefore sees nothing and manufactures an empty
        tuple, which the consumer reads as "no loaded resident resolves" and grants
        a vacuous exemption to every candidate.

        Regression guard: a producer that reads only `meta = r.idle_client_meta`
        would return `()` here instead of `((0, 1),)`.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.ACTIVE,
            rank_client_meta=dict(_LISTED_A),
            idle_client_meta=None,
        )
        assert self._keys(mgr) == ((0, 1),)

    # ---------------------------------------------------------------- ARM 2

    def test_ARM2_active_resident_mid_turn_for_another_client_contributes_no_stale_key(
        self, boot_and_runtime
    ):
        """WRONG-identity direction -- the half no consumer-side change can reach.

        The resident is ACTIVE for an UNLISTED client right now, but still carries the
        LISTED meta of whoever it served before it last parked. Reading idle_client_meta
        alone attributes the previous client's priority to a model currently serving
        someone else -- residents are keyed by model_tag, not by client. Under the rank-first rule the
        answer is UNRESOLVABLE, never a stale guess (the decision must reflect
        reality, not a stale snapshot).

        Regression guard: a producer that reads only `meta = r.idle_client_meta`
        would return `((0, 1),)` here instead of `()`.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.ACTIVE,
            rank_client_meta=dict(_UNLISTED),
            idle_client_meta=dict(_LISTED_A),
        )
        assert self._keys(mgr) == ()

    # ---------------------------------------------------------------- ARM 3

    def test_ARM3_parked_multislot_resident_is_not_invisible(self, boot_and_runtime):
        """A parallel>=2 resident can NEVER be seen by the current producer, in any
        state. Both idle_client_meta writers (in manager.py) are gated on
        `parallel == 1`, so the field is never stamped for a multi-slot engine, while
        rank_client_meta is stamped ungated and never cleared. The swap-budget
        exemption is therefore permanently vacuous for exactly the multi-slot
        concurrency case Fast Lane exists to arbitrate.

        Regression guard: a producer that reads only `meta = r.idle_client_meta`
        would return `()` here instead of `((0, 1),)`.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.IDLE_EVICTABLE,
            parallel=2,
            rank_client_meta=dict(_LISTED_A),
            idle_client_meta=None,
        )
        assert self._keys(mgr) == ((0, 1),)

    # ---------------------------------------------------------------- ARM 4

    def test_ARM4_aggregates_rank_resolved_and_idle_resolved_residents_together(
        self, boot_and_runtime
    ):
        """Composition arm: a mid-turn resident resolved via rank_client_meta and a
        parked one resolved via the idle fallback must BOTH appear. Guards a fix that
        swaps one source for the other instead of layering the fallback beneath it.

        Regression guard: dropping the `if meta is None and state is IDLE_EVICTABLE`
        fallback entirely (rank-only) would lose r2, giving
        `{(0, 1)}` instead of `{(0, 1), (1, 2)}`.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="model-a",
            state=ResidentState.ACTIVE,
            rank_client_meta=dict(_LISTED_A),
            idle_client_meta=None,
        )
        mgr._residents["r2"] = Resident(
            model_tag="model-b",
            state=ResidentState.IDLE_EVICTABLE,
            rank_client_meta=None,
            idle_client_meta=dict(_LISTED_B),
        )
        assert set(self._keys(mgr)) == {(0, 1), (1, 2)}

    # ------------------------------------------------------- ARM 5 (CONTROL)

    def test_ARM5_CONTROL_parked_resident_resolved_via_idle_fallback_unchanged(
        self, boot_and_runtime
    ):
        """OVER-BROADNESS CONTROL. The legitimate parked-resident path -- the one shape
        where the current producer and the rank-first sibling already agree -- must keep
        working. GREEN BEFORE AND AFTER the fix by design: it is here to fail a
        "fix" that simply replaces idle_client_meta with rank_client_meta and
        silently drops every parked resident on the floor.

        Regression guard: a rank-only producer with no IDLE_EVICTABLE fallback.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.IDLE_EVICTABLE,
            rank_client_meta=None,
            idle_client_meta=dict(_LISTED_A),
        )
        assert self._keys(mgr) == ((0, 1),)

    # ---------------------------------------------------------------- ARM 7

    def test_ARM7_anonymous_current_turn_does_not_fall_back_to_a_stale_identity(
        self, boot_and_runtime
    ):
        """PRESENT-BUT-EMPTY rank_client_meta must still SUPPRESS the idle fallback.

        This arm exists because the narrower alternative fix -- ``meta =
        r.rank_client_meta or r.idle_client_meta``, i.e. rank-first with an
        any-state fallback -- passes every other arm in this file. The two
        differ on exactly one input: a resident whose CURRENT turn has a
        present-but-empty ``client_meta``. ``or`` treats ``{}`` as absent and
        reaches past it to the stale parked identity; the rank-first rule's ``is None``
        test treats it as present-and-unresolvable and stops.

        That input is not exotic -- it is the DEFAULT. ``Slot.client_meta`` is
        ``dataclasses.field(default_factory=dict)`` (slot.py), ``Slot``'s
        constructor coerces ``None`` to ``{}`` (slot.py), and the API
        boundary produces ``{}`` for any request that omits the key
        (api/chat_completion.py). Reserve then stamps that ``{}`` straight
        onto ``rank_client_meta`` (manager.py). So ANY anonymous request
        served by a model that previously served a LISTED client lands in this
        shape, and under ``or`` it would inherit that previous client's Fast
        Lane priority -- the exact "WRONG identity, not merely a missing one"
        that docstring forbids.

        Regression guard: `meta = r.rank_client_meta or r.idle_client_meta`
        would return `((0, 1),)` here instead of `()`.
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.ACTIVE,
            rank_client_meta={},              # anonymous CURRENT turn
            idle_client_meta=dict(_LISTED_A),  # listed, but a PREVIOUS turn's client
        )
        assert self._keys(mgr) == ()

    # ------------------------------------------------------- ARM 6 (CONTROL)

    def test_ARM6_CONTROL_genuinely_unresolvable_resident_contributes_nothing(
        self, boot_and_runtime
    ):
        """NO-PHANTOM-SENTINEL CONTROL. A resident with neither meta is genuinely
        UNRESOLVABLE and must contribute nothing -- not a (0, 0)-shaped sentinel that
        would make an ordinary candidate look exempt. GREEN BEFORE AND AFTER by
        design; it pins the one empty-tuple case that is legitimately empty, so the
        fix cannot be mistaken for "make loaded_priority_keys never empty".
        """
        mgr = self._mgr(boot_and_runtime)
        mgr._residents["r1"] = Resident(
            model_tag="loaded-model",
            state=ResidentState.ACTIVE,
            rank_client_meta=None,
            idle_client_meta=None,
        )
        assert self._keys(mgr) == ()
