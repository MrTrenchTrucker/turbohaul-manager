"""Which _vram_verify call sites take the new VRAM
settle floor, and which are deliberately exempted.

Design question: do ALL FOUR call
sites want the settle delay, or only the eviction-teardown ones? The
callers were traced in the current tree.

THREE OF FOUR CALL SITES ARE EXEMPT, NOT TWO OF FOUR.
The exemptions are ``_teardown`` and
``_teardown_idle_holder``, plus a third:
``_late_vram_reconcile``, which
only ever runs AFTER ``_unload_teardown``'s own floor+timeout_s (5.0s+30s)
has ALREADY timed out, plus its own additional ``grace_seconds`` sleep on
top -- by the time its verify call fires, real-elapsed since the original
SIGTERM is comfortably past any physical settle window. There are no
guaranteed-miss samples left to skip; a floor here is pure added
latency on the SAME notify path the settle floor exists to protect,
with nothing to show for it.
This exemption is deliberate and
is pinned
below.

    _unload_teardown (manager.py)        -- takes the floor (default)
    _late_vram_reconcile (manager.py)   -- settle_floor_s=0.0, explicit (see above)
    _teardown (manager.py)             -- settle_floor_s=0.0, explicit
    _teardown_idle_holder (manager.py) -- settle_floor_s=0.0, explicit

``_teardown``/``_teardown_idle_holder`` live ONLY inside the cap<=1
single-slot ``worker_loop`` body (the early fork in it returns
into cap>=2's ``_dispatch_loop`` before reaching either) -- structurally
unreachable from ``_drive_resident``/``_begin_unload_locked``, so
``_make_room_signal`` never has a waiter parked behind either call.
``_teardown``'s own verify is LOG-ONLY by its own comment (no
credit to reconcile on that path); ``_teardown_idle_holder`` is awaited
in-line by the very coroutine about to spawn the next model. A floor on
either is pure added wall-clock with no faster-notice benefit -- exactly
the "call site where the delay actively hurts" that this scoping targets.

Same "named set, both directions" idiom as
the turn-boundary handoff test's ``TestNotifyCallSitesAreTheNamedSet``
(that suite's own structural guardrail) -- a count or a single membership
check would miss a SWAP (e.g. the exemption moving from ``_teardown`` onto
``_unload_teardown`` by mistake, silently reintroducing the admission-
latency regression while leaving some assertion still green). Source-scans
each of the four methods for the literal ``settle_floor_s=0.0`` override and
asserts it is present in exactly the three exempt methods and absent from the
notify-gating method.
"""
import inspect

from turbohaul.manager import TurbohaulManager

#: Methods that MUST explicitly pass settle_floor_s=0.0 -- no waiting
#: claimant benefits from the floor on these paths (see module docstring).
EXPECTED_ZERO_FLOOR_SITES = {
    "_teardown",
    "_teardown_idle_holder",
    "_late_vram_reconcile",  # the third exemption -- see module docstring
}

#: Methods that MUST NOT override settle_floor_s -- they gate the
#: moment-3 notify and rely on verify_vram_cleared's own default (5.0s,
#: named/defaulted so the number can move without the call site
#: moving).
EXPECTED_DEFAULT_FLOOR_SITES = {
    "_unload_teardown",
}


class TestSettleFloorScopingIsTheNamedSplit:
    def test_zero_floor_sites_pass_settle_floor_s_explicitly(self):
        for name in EXPECTED_ZERO_FLOOR_SITES:
            src = inspect.getsource(getattr(TurbohaulManager, name))
            assert "settle_floor_s=0.0" in src, (
                f"{name} must explicitly pass settle_floor_s=0.0 to "
                f"_vram_verify -- it has no waiting claimant to benefit "
                f"from the floor (cap<=1 only; see module docstring)"
            )

    def test_default_floor_sites_do_not_override_settle_floor_s(self):
        for name in EXPECTED_DEFAULT_FLOOR_SITES:
            src = inspect.getsource(getattr(TurbohaulManager, name))
            # "settle_floor_s=" (the kwarg-pass syntax), not the bare word --
            # both methods carry an explanatory comment that NAMES
            # settle_floor_s without passing it (see manager.py), which a
            # bare substring check would misread as an override.
            assert "settle_floor_s=" not in src, (
                f"{name} must rely on verify_vram_cleared's own default "
                f"settle_floor_s -- it gates the moment-3 notify, "
                f"and a waiting claimant genuinely benefits from the floor. "
                f"An explicit override here (even one that happens to match "
                f"the default) defeats the whole point: the number moves "
                f"without the call site moving."
            )

    def test_the_two_sets_are_disjoint_and_are_exactly_the_four_known_call_sites(self):
        # A swap (a name moving from one set into the other, or dropping out
        # of both) fails here even if the two tests above each still pass
        # individually against whatever the swap left behind.
        assert EXPECTED_ZERO_FLOOR_SITES.isdisjoint(EXPECTED_DEFAULT_FLOOR_SITES)
        all_sites = EXPECTED_ZERO_FLOOR_SITES | EXPECTED_DEFAULT_FLOOR_SITES
        assert all_sites == {
            "_teardown", "_teardown_idle_holder",
            "_unload_teardown", "_late_vram_reconcile",
        }
        assert len(EXPECTED_DEFAULT_FLOOR_SITES) == 1, (
            "only _unload_teardown -- the FIRST verify right after a fresh "
            "SIGTERM -- has a genuine guaranteed-miss window to skip"
        )
