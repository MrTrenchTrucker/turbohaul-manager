"""Drift guard for the swap-budget exemption's
comparator routing in ``_pick_fastlane_locked`` (queue.py, the swap-budget
exemption block).

BEFORE THIS CHANGE: the exemption computed ``cand_key < loaded_key`` inline --
the same expression as ``_fastlane_strictly_higher``'s own
return statement, minus its two ``None`` guards. The signed comparator
carries a documented rule in its contract: "listed beats unlisted
absolutely; between two listed candidates, earlier ``rule_index`` wins, then
lower ``rank``; an EQUAL ``(rule_index, rank)`` is a real tie and is
deliberately NOT strictly higher." A second inline copy of that rule means
the rule has to be re-applied by hand in two places, and the second place
does not know the first exists -- exactly the "mirror that agrees today,
with nothing asserting it will keep agreeing" shape an earlier drift guard
existed to catch, applied here to a different pair of call sites.

THIS CHANGE: routes the exemption through ``self._fastlane_strictly_higher``
directly (``cand_key = self._fastlane_priority_key(cand); exempt =
all(self._fastlane_strictly_higher(cand_key, k) for k in
policy.loaded_priority_keys)``).

WHY THE RED CANNOT BE "ASSERT THE TWO AGREE" (the corrected shape):
once the exemption calls the comparator, "the exemption
agrees with the comparator" collapses into "the comparator agrees with
itself" -- mutating the comparator's tie rule then moves BOTH sides
together and the mutual-agreement assertion stays green. That is a green
that cannot go red, in the very test written to prevent drift.

SO THE TESTS BELOW PIN THE COMPARATOR'S VERDICT TO A FIXED TABLE SOURCED
FROM THE DOCUMENTED CONTRACT ITSELF (``CONTRACT_TABLE``), not from whatever the comparator
currently computes -- the literal values in that table ARE the contract,
written down, which is the one case a hardcoded expected side is not the
usual "test that cannot fail" trap. Read ``_fastlane_strictly_higher``'s
own docstring before touching ``CONTRACT_TABLE``; it is meant
to go stale (and fail loudly) the moment that docstring's contract changes
out from under it.

NOT VERIFIED -- MEASURED REACHABILITY LIMIT (the argument is cited here because it
belongs where
the next editor meets it, not asserted quietly): on today's reachable
path, ``cand_key`` (queue.py, the swap-budget exemption's own
``_fastlane_priority_key(cand)`` call) and every member of
``policy.loaded_priority_keys`` (built in manager.py, appending
``(match.rule_index, match.rank)`` only when ``match is not None``) can
never actually be ``None`` -- the candidate list feeding ``cand`` is itself
filtered to ``slot.fastlane is not None`` a few lines above,
and its own sort key unpack would raise on a ``None`` key
first. So ``CONTRACT_TABLE``'s two None-position rows and the None/None row
exercise the comparator directly (``TestComparatorMatchesContract``), proving
its documented contract, but are NOT exercised through the exemption
line/``pop_next`` integration path below -- that gap is disclosed, not
assumed closed. The fix is still strictly safer off that unreachable path
either way: a ``None`` candidate now degrades to "not exempt" and a
``None`` loaded key now degrades to "exempt", instead of the inline
``cand_key < loaded_key`` raising ``TypeError``.
"""
import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


# ---------------------------------------------------------------------------
# The contract, written down as data. Each row: (candidate_key, holder_key,
# expected `_fastlane_strictly_higher` verdict). Source: `_fastlane_strictly_
# higher`'s own docstring, quoted above.
# ---------------------------------------------------------------------------
CONTRACT_TABLE = [
    pytest.param(None, (0, 1), False, id="unlisted_candidate_never_beats_a_listed_holder"),
    pytest.param((0, 1), None, True, id="listed_candidate_always_beats_an_unresolved_holder"),
    pytest.param(None, None, False, id="unlisted_candidate_still_loses_even_if_holder_also_unresolved"),
    pytest.param((0, 1), (0, 1), False, id="equal_rule_index_and_rank_is_a_real_tie_not_strictly_higher"),
    pytest.param((0, 1), (1, 1), True, id="earlier_rule_index_wins"),
    pytest.param((1, 1), (0, 1), False, id="later_rule_index_loses"),
    pytest.param((0, 1), (0, 2), True, id="same_rule_index_lower_rank_wins"),
    pytest.param((0, 2), (0, 1), False, id="same_rule_index_higher_rank_loses"),
]


class TestComparatorMatchesContract:
    """Direct unit coverage of `_fastlane_strictly_higher` itself against the
    documented contract, written down as `CONTRACT_TABLE` -- including both
    `None` positions, which are not reachable through the exemption
    integration path below (see module docstring)."""

    @pytest.mark.parametrize("cand_key,holder_key,expected", CONTRACT_TABLE)
    def test_comparator_verdict_matches_the_contract(self, cand_key, holder_key, expected):
        q = TurbohaulQueue(staging_max=10)
        assert q._fastlane_strictly_higher(cand_key, holder_key) is expected


@pytest.mark.asyncio
class TestSwapExemptionFollowsTheContractOnEqualKeys:
    """Integration-level drift guard: the swap-budget exemption in
    `_pick_fastlane_locked` (queue.py) must treat an EQUAL (rule_index,
    rank) tie as NOT exempt -- this is the case the original ("assert
    the two agree") shape could not have caught post-fix (see module
    docstring), and it is the case where a hand-applied second copy of the
    tie rule is most likely to silently diverge from the comparator if the
    rule ever changes.

    This test is deliberately built to exercise the SAME reachable
    production path `TestSwapBudget.test_cold_jump_refused_when_budget_
    exhausted_falls_to_next_candidate` (test_queue_fastlane.py) already
    does for the same EQUAL-tie shape -- an independent, purpose-built
    confirmation of that existing test's own drift-detection value, not a
    replacement for it.
    """

    async def test_equal_priority_candidate_is_not_exempt_from_an_exhausted_budget(self):
        """RE-POINT: a later change lets Fast Lane override the swap budget
        entirely, via a blanket
        grant in `_pick_fastlane_locked` that fires for EVERY Fast-Lane-
        matched candidate reaching the loop, exempt-via-tie-rule or not --
        so `cold_best` now wins this scenario EITHER WAY, regardless of
        whether the tie-rule under test here is right or wrong. `got.slot_id`
        can therefore no longer distinguish "tie correctly NOT exempt" from
        "tie wrongly treated as exempt" -- asserting on it the old way would
        make this file's own stated purpose (a comparator drift guard) a
        green that cannot go red, the exact trap its own module docstring
        warns against for a different pair of call sites. The drift-guard
        value now lives in the SWAP REASON instead: a correct (not-exempt)
        tie must fall through to the blanket grant, reason
        "budget_exempt", never the rank-based "exempt" reason a wrongly-
        exempt tie would produce."""
        q = TurbohaulQueue(staging_max=10)
        cold_best = Slot.new(model_tag="m2", prompt="cold")
        cold_best.fastlane = _match(rule_index=0, rank=1)
        warm_fallback = Slot.new(model_tag="m1", prompt="warm")
        warm_fallback.fastlane = _match(rule_index=1, rank=1)  # strictly worse, real fallback
        for s in (cold_best, warm_fallback):
            await q.enqueue(s)

        # A loaded resident TYING cold_best's own (rule_index, rank) exactly
        # -- CONTRACT_TABLE's "equal_rule_index_and_rank_is_a_real_tie" row.
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=((0, 1),),
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        # cold_best wins regardless of the tie-exemption verdict
        # (the blanket grant covers a wrongly-exempt tie too) -- this is
        # expected and is NOT what this test guards against.
        assert got.slot_id == cold_best.slot_id
        assert q._fastlane_refusal_counts.get("budget", 0) == 0
        # THE REAL ASSERTION: reason must be "budget_exempt" (fell through
        # the tie-rule correctly), never "exempt" (would mean the tie was
        # wrongly treated as strictly-higher-priority).
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}
