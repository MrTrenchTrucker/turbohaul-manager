"""verify_vram_cleared computed an UNSATISFIABLE target on
split-card models.

A worked example on a two-GPU host (forced ``_unload_teardown`` on a
large split-card model)
showed:

``_unload_teardown`` passes ``expected_drop_mib=reclaim_mib`` (the model's
WHOLE credit, 32424 MiB in that example) with ``device_index=reclaim_card`` (ONE
card, which held only 17464 MiB of it). Pre-fix:
``target = max(0, 17464 - int(32424 * 0.9)) = max(0, 17464-29181) = 0``.
Neither card ever reads exactly 0 in that example -- a live CUDA context
floors at a few MiB on each card -- so the predicate was
UNSATISFIABLE: it polled the full ``timeout_s`` on every split-card eviction
and always reported not-cleared, even though both cards had fully released.
This also made the post-unload notify inert on that path
(``dec = min(measured_delta, reclaim_mib)`` computed 0 -> the ``if dec:``
guard, which is itself correct, blocked on a FALSE zero).

Fix (subprocess_mgr.py, ``verify_vram_cleared``): clamp the expected
drop to ``initial`` (this card's own actual pre-poll reading) before
computing target -- "we expect at least a 90% drop of whatever THIS card
actually held," the correct, achievable claim, instead of an impossible
cross-card total. Rejected alternatives (three shapes were considered):
  (b) per-card share -- would need a genuine per-GPU footprint breakdown;
      grepped the manifest/Resident fields (split_mode, main_gpu) and found
      no such tracking exists (the card-capacity guard is a
      SAFETY CAP against a stuck credit exceeding physical VRAM, not a true
      per-card allocation split) -- would require NEW instrumentation this
      change was not scoped to add.
  (c) epsilon floor -- an arbitrary hardcoded "N MiB counts as cleared"
      loosens the check for EVERY card (split or not) and is hardware/
      driver-fragile (a live CUDA context leaves a small residual that varies
      by card and driver); the clamp is self-calibrating off the actual initial
      reading on whatever hardware is running, with no new constant to tune.

This is kept SEPARATE from the settle-floor changes, since
two distinct defects and a fix to one must not mask a regression in the
other -- this file's tests exercise ONLY the clamp; the settle-floor
tests live in test_subprocess_mgr.py::TestVramVerify and
the settle-floor scoping test.
"""
import pytest

from turbohaul.subprocess_mgr import verify_vram_cleared

# Modeled directly on the worked example above (card 0's row).
_INITIAL_ON_THIS_CARD = 17464   # what THIS card actually held pre-teardown
_FULL_MODEL_CREDIT = 32424      # the split model's WHOLE credit (both cards)
_GENUINE_CLEAR_FLOOR = 2        # live-CUDA-context floor, card 0


@pytest.mark.asyncio
class TestSplitCardTargetIsSatisfiable:
    async def test_split_card_that_genuinely_cleared_is_confirmed(self):
        """The killing case: expected_drop_mib (32424) is far GREATER than
        what this card ever held (17464), and the card bottoms at a
        nonzero floor (2 MiB) -- never exactly 0. Pre-fix this must time
        out; post-fix it must confirm quickly."""
        curve = [
            _INITIAL_ON_THIS_CARD,  # initial sample
            11510,                  # mid-drop (a measured curve)
            452,
            _GENUINE_CLEAR_FLOOR,   # settled -- never reaches 0
        ]
        calls = {"n": 0}

        def runner():
            # Repeats the settled floor forever once the curve is exhausted
            # -- pre-fix, this card can be polled far past the curve's
            # length (unsatisfiable target -> polls the full timeout_s), and
            # a StopIteration there would fail this test for an unrelated
            # fixture reason instead of the diagnosed one.
            i = min(calls["n"], len(curve) - 1)
            calls["n"] += 1
            return f"{curve[i]}\n"

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=_FULL_MODEL_CREDIT,
            nvidia_smi_runner=runner,
            timeout_s=1.0, poll_interval_s=0.02, settle_floor_s=0.0,
            device_index=0,
        )
        assert cleared is True, (
            "a card that genuinely dropped from 17464 to a 2 MiB floor "
            "must confirm cleared -- an unsatisfiable target=0 would time "
            "out here instead (the exact defect measured live)"
        )
        # Clamped target: max(0, initial - int(min(expected, initial)*0.9)).
        # min(32424, 17464)=17464 -> max(0, 17464-15717)=1747. Whichever
        # sample in the curve first crosses that threshold is the one
        # returned -- assert the THRESHOLD, not a specific reading, so this
        # doesn't depend on exactly which poll tick catches it.
        expected_target = max(
            0, _INITIAL_ON_THIS_CARD
            - int(min(_FULL_MODEL_CREDIT, _INITIAL_ON_THIS_CARD) * 0.9)
        )
        assert current is not None and current <= expected_target

    async def test_split_card_that_did_not_release_still_fails(self):
        """CONTROL: same magnitude mismatch (expected_drop_mib far exceeds
        what this card holds), but the card genuinely does NOT release --
        stuck well above 90% of its OWN initial reading. The fix must not
        make the check trivially true; this must still report not-cleared."""
        stuck_at = int(_INITIAL_ON_THIS_CARD * 0.95)  # released only ~5%

        def runner():
            return f"{stuck_at}\n"

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=_FULL_MODEL_CREDIT,
            nvidia_smi_runner=runner,
            timeout_s=0.1, poll_interval_s=0.02, settle_floor_s=0.0,
            device_index=0,
        )
        assert cleared is False, (
            "a card stuck at 95% of its own initial reading released far "
            "less than the required 90% drop -- must still fail, proving "
            "the clamp is a real check, not a trivially-true one"
        )
        assert current == stuck_at
