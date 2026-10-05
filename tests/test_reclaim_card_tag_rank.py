"""The tag rank inside one client, in the choice of the card.

When a request needs idle models unloaded to fit on one card, the card is chosen by how protected
the idle models are that the card would have to unload. The client comes first (the listed order;
a client in no rule comes before every listed one). Between two cards whose most protected model to
unload belongs to the SAME client, the tag decides: the card whose most protected model to unload
holds the worse tag (main, curator, compression, sub_agent, unclassified, best to worst) is chosen,
so the better-tagged model stays loaded. A model whose class cannot be resolved counts as the
unclassified tag of its client. A tag never lifts one client above another. Recency stays out of
the card choice; it only orders the models inside the chosen card. A card that unloads nobody is
still the best of all.

Free VRAM: every test here reaches the card choice through the fixtures of the cold-start tests,
which pin BOTH `turbohaul.safety._read_free_vram_all_mib` and `turbohaul.manager._read_free_vram_all_mib`
(the direct tests through `_bare_manager`, the real-path tests through `world`).
"""
import pytest

from test_reclaim_first_at_cold_start import (
    AUTO_IDS,
    IP_1,
    IP_2,
    IP_3,
    IP_UNLISTED,
    TAG_IDLE,
    TAG_IDLE2,
    _assert_card_chosen,
    _assert_served_after_only_this_unload,
    _bare_manager,
    _choose,
    _rank_real_models,
    _rank_real_scene,
    _resident,
    submit_claimant,
    world,
)
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState

pytestmark = pytest.mark.asyncio

# main 1, curator 2, compression 3, sub_agent 4; unclassified is left unset, which ranks it after
# every ranked tag (the worst), the same for every listed client
TAG_RANKS = FastLaneTagRanks(main=1, curator=2, compression=3, sub_agent=4)
TAG_RULES = [FastLaneRule(address=ip, tag_ranks=TAG_RANKS) for ip in (IP_1, IP_2, IP_3)]
NEED = 15000                 # every direct scene: need 15000, floor 1000 (FLOOR_MIB)

TAG_LABELS = {
    "main": {"is_main": True},
    "curator": {"is_curator": True},
    "compression": {"is_compression": True},
    "sub_agent": {"is_sub_agent": True},
    "no_labels": {},                       # no class label: the class cannot be resolved
    "unknown_role": {"role": "zzz"},       # a role no class carries: the class cannot be resolved
    "user_message_role": {"role": "user-message"},   # a class with no tag of its own: folds to unclassified
}


def meta_of(ip, tag):
    return {"ip": ip, **TAG_LABELS[tag]}


def _tag_manager(tmp_path, monkeypatch, free, residents, rules=None):
    """A bare manager (scripted free-VRAM probe, both bindings pinned) that carries the tag-ranked
    table of three clients (or `rules`), with the given idle residents registered."""
    mgr, _probe = _bare_manager(tmp_path, monkeypatch, free, rules=rules or TAG_RULES)
    for r in residents:
        mgr._residents[r.resident_key] = r
    return mgr


def _unload_tuple(choice):
    return tuple(choice)


# ------------------------------------------------------------------------------------------------
# (a) the worse tag of the same client is unloaded first, so its card is chosen
# ------------------------------------------------------------------------------------------------
# Card BIG reads 6000 free and holds a 17000 idle model of the worse tag: 10000 still to unload.
# Card SMALL reads 14000 free and holds a 3000 idle model of the better tag: 2000 still to unload.
# Both are usable. The fewer MiB would pick SMALL; the tag picks BIG. Both orders of the cards.
@pytest.mark.parametrize("better,worse,big_card,want", [
    ("main", "curator", 0, 0),
    ("main", "curator", 1, 1),
    ("curator", "compression", 0, 0),
    ("curator", "compression", 1, 1),
    ("compression", "sub_agent", 0, 0),
    ("compression", "sub_agent", 1, 1),
], ids=[
    "main_over_curator_worse_on_card_0", "main_over_curator_worse_on_card_1",
    "curator_over_compression_worse_on_card_0", "curator_over_compression_worse_on_card_1",
    "compression_over_sub_agent_worse_on_card_0", "compression_over_sub_agent_worse_on_card_1",
])
async def test_between_cards_of_one_client_the_card_whose_model_holds_the_worse_tag_is_chosen(
        tmp_path, monkeypatch, better, worse, big_card, want):
    """Both models belong to the first-listed client; only the tag differs. The worse-tagged model
    goes first, so its card wins although it needs far more unloading; the card index plays no part
    (the worse-tagged card is card 0 in one case and card 1 in the other)."""
    small_card = 1 - big_card
    free = [0, 0]
    free[big_card], free[small_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-big", big_card, 17000, meta=meta_of(IP_1, worse)),
        _resident("idle-small", small_card, 3000, meta=meta_of(IP_1, better)),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, (
        f"card {big_card}: {worse} model of 17000 MiB (10000 to unload); card {small_card}: {better} model "
        f"of 3000 MiB (2000 to unload); same client: the worse tag goes first"))
    assert _unload_tuple(choice) == (want, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("auto_place", [True, False], ids=AUTO_IDS)
@pytest.mark.parametrize("a_card,want", [(0, 1), (1, 0)], ids=["better_tag_on_card_0", "better_tag_on_card_1"])
async def test_the_worse_tagged_model_of_the_same_client_is_the_one_unloaded_on_the_real_path(
        tmp_path, monkeypatch, a_card, want, auto_place):
    """Need 15000. Card A reads 14000 free with a 3000 idle main model of the first-listed client; card
    B reads 6000 free with a 17000 idle sub_agent model of the same client. Card A needs 2000 MiB
    unloaded, card B 10000, but the sub_agent model goes first: only it is unloaded, the main model
    keeps its card and the claimant is served on card B (a literal: card 1 when A is card 0)."""
    async with world(tmp_path, monkeypatch, models=_rank_real_models(a_card, auto_place), rules=TAG_RULES) as b:
        await _rank_real_scene(b, a_card, meta_of(IP_1, "main"), meta_of(IP_1, "sub_agent"))
        submit_claimant(b)
        await _assert_served_after_only_this_unload(b, want, TAG_IDLE2, TAG_IDLE)


# ------------------------------------------------------------------------------------------------
# (b) a model whose class cannot be resolved counts as the worst tag of its client
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["no_labels", "unknown_role"])
@pytest.mark.parametrize("big_card,want", [(0, 0), (1, 1)], ids=["unresolved_on_card_0", "unresolved_on_card_1"])
async def test_a_model_whose_class_cannot_be_resolved_sorts_as_the_worst_tag_of_its_client(
        tmp_path, monkeypatch, kind, big_card, want):
    """Same client on both cards. The unresolved model (17000 MiB, 10000 to unload) sits against a
    sub_agent model (3000 MiB, 2000 to unload; the worst tag the table ranks). The unresolved model
    is unloaded first, so its card wins."""
    small_card = 1 - big_card
    free = [0, 0]
    free[big_card], free[small_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-unresolved", big_card, 17000, meta=meta_of(IP_1, kind)),
        _resident("idle-sub-agent", small_card, 3000, meta=meta_of(IP_1, "sub_agent")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, f"[{kind}] a model with no resolvable class is the worst tag of its client")
    assert _unload_tuple(choice) == (want, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("big_card,want", [(0, 1), (1, 0)], ids=["unresolved_on_card_0", "unresolved_on_card_1"])
async def test_an_unresolved_model_is_still_unloaded_after_a_model_of_a_worse_listed_client(
        tmp_path, monkeypatch, big_card, want):
    """The unresolved class only falls to the bottom of its OWN client: the model of the third-listed
    client, even holding the best tag, is unloaded before the unresolved model of the first-listed
    client. The 17000 MiB unresolved model on the big card loses to the 3000 MiB main model of the
    worse client."""
    small_card = 1 - big_card
    free = [0, 0]
    free[big_card], free[small_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-unresolved", big_card, 17000, meta=meta_of(IP_1, "no_labels")),
        _resident("idle-third", small_card, 3000, meta=meta_of(IP_3, "main")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "the client rank decides first, whatever the tags")
    assert _unload_tuple(choice) == (want, 3000, 0, 2000), tuple(choice)


# ------------------------------------------------------------------------------------------------
# (c) a tag never overrides the client
# ------------------------------------------------------------------------------------------------
# Better client = first-listed, worse client = third-listed. Card BIG reads 6000 free (17000 model,
# 10000 to unload), card SMALL reads 14000 free (3000 model, 2000 to unload).
CLIENT_CASES = [
    # (id, tag of the first-listed client's model, tag of the third-listed client's model,
    #  card of the first-listed client's model, card that must win)
    ("better_client_holds_worse_tag_on_card_0", "no_labels", "main", 0, 1),
    ("better_client_holds_worse_tag_on_card_1", "no_labels", "main", 1, 0),
    ("better_client_holds_better_tag_on_card_0", "main", "no_labels", 0, 1),
    ("better_client_holds_better_tag_on_card_1", "main", "no_labels", 1, 0),
]


@pytest.mark.parametrize("first_tag,third_tag,first_card,want", [c[1:] for c in CLIENT_CASES],
                         ids=[c[0] for c in CLIENT_CASES])
async def test_a_better_tag_never_lifts_a_worse_clients_card_above_a_better_clients(
        tmp_path, monkeypatch, first_tag, third_tag, first_card, want):
    """The first-listed client's model is the big one (17000 MiB, 10000 to unload) on the card where
    it sits; the third-listed client's model is the small one (3000 MiB, 2000 to unload) on the
    other. The third-listed client is unloaded first whatever the tags, so its card wins; the
    cases with the first-listed client on the worst tag (and the third-listed on the best) are the
    ones a tag that outranked the client would get wrong."""
    third_card = 1 - first_card
    free = [0, 0]
    free[first_card], free[third_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-first", first_card, 17000, meta=meta_of(IP_1, first_tag)),
        _resident("idle-third", third_card, 3000, meta=meta_of(IP_3, third_tag)),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "the client rank decides first; the tag is only read inside one client")
    assert _unload_tuple(choice) == (want, 3000, 0, 2000), tuple(choice)


@pytest.mark.parametrize("a_card,want", [(0, 1), (1, 0)], ids=["first_listed_on_card_0", "first_listed_on_card_1"])
async def test_a_better_tag_does_not_lift_a_worse_client_on_the_real_path(tmp_path, monkeypatch, a_card, want):
    """Need 15000. Card A reads 14000 free with a 3000 idle model of the first-listed client holding the
    worst tag (no class label); card B reads 6000 free with a 17000 idle main model of the third-listed
    client. The third-listed client's model goes first: only it is unloaded and the claimant is served
    on card B."""
    async with world(tmp_path, monkeypatch, models=_rank_real_models(a_card, True), rules=TAG_RULES) as b:
        await _rank_real_scene(b, a_card, meta_of(IP_1, "no_labels"), meta_of(IP_3, "main"))
        submit_claimant(b)
        await _assert_served_after_only_this_unload(b, want, TAG_IDLE2, TAG_IDLE)


# ------------------------------------------------------------------------------------------------
# (d) equal tags: the existing tie-breaks decide
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("small_card,want", [(0, 0), (1, 1)], ids=["fewer_mib_on_card_0", "fewer_mib_on_card_1"])
async def test_between_equal_tags_of_one_client_the_fewer_mib_to_unload_decides(
        tmp_path, monkeypatch, small_card, want):
    """Control for the tie-breaks. Both models are sub_agent models of the first-listed client. The
    small card reads 14000 free with a 3000 model (2000 to unload, headroom 1000); the big card reads
    6000 free with a 17000 model (10000 to unload, headroom 7000). The fewer MiB wins."""
    big_card = 1 - small_card
    free = [0, 0]
    free[small_card], free[big_card] = 14000, 6000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-small", small_card, 3000, meta=meta_of(IP_1, "sub_agent")),
        _resident("idle-big", big_card, 17000, meta=meta_of(IP_1, "sub_agent")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "equal tag: 2000 still to unload beats 10000 although the other card keeps more headroom")
    assert _unload_tuple(choice) == (want, 3000, 0, 2000), tuple(choice)


async def test_between_equal_tags_with_the_same_mib_to_unload_the_larger_headroom_decides(tmp_path, monkeypatch):
    """Both cards read 12000 free (4000 to unload on each) and hold a curator model of the same client.
    Card 0's model is 5000 (headroom 1000), card 1's is 9000 (headroom 5000). Card 1 wins on headroom."""
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident("idle-a", 0, 5000, meta=meta_of(IP_1, "curator")),
        _resident("idle-b", 1, 9000, meta=meta_of(IP_1, "curator")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, 1, "equal tag, equal 4000 still to unload: the larger headroom wins")
    assert _unload_tuple(choice) == (1, 9000, 0, 4000), tuple(choice)


async def test_between_equal_tags_that_are_equal_in_everything_the_lowest_card_decides(tmp_path, monkeypatch):
    """Both cards read 12000 free and hold a 6000 compression model of the same client: the lowest
    card index decides."""
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident("idle-a", 0, 6000, meta=meta_of(IP_1, "compression")),
        _resident("idle-b", 1, 6000, meta=meta_of(IP_1, "compression")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, 0, "everything equal: the lowest card index decides")
    assert _unload_tuple(choice) == (0, 6000, 0, 4000), tuple(choice)


# ------------------------------------------------------------------------------------------------
# (e) recency stays out of the card choice and orders the models inside the chosen card
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("worse_card,want", [(0, 0), (1, 1)], ids=["worse_tag_on_card_0", "worse_tag_on_card_1"])
async def test_how_recently_a_model_was_used_does_not_outweigh_its_tag_in_the_card_choice(
        tmp_path, monkeypatch, worse_card, want):
    """Need 15000. The worse-tagged (sub_agent) model on the big card is the MORE recently active one
    (last_active 100.0); the better-tagged (main) model on the small card is the less recent (1.0),
    which is the one the victim order would reach first if recency decided. Same client. The card of
    the worse tag wins."""
    better_card = 1 - worse_card
    free = [0, 0]
    free[worse_card], free[better_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-worse", worse_card, 17000, meta=meta_of(IP_1, "sub_agent"), last_active=100.0),
        _resident("idle-better", better_card, 3000, meta=meta_of(IP_1, "main"), last_active=1.0),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "recency must not decide the card; the tag does")
    assert _unload_tuple(choice) == (want, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("recent_card", [0, 1], ids=["recent_model_on_card_0", "recent_model_on_card_1"])
async def test_between_equal_tags_how_recently_a_model_was_used_does_not_decide_the_card(
        tmp_path, monkeypatch, recent_card):
    """Both cards read 12000 free and hold a 6000 sub_agent model of the same client (everything equal).
    One model was used more recently (100.0 against 1.0). The card index decides: card 0, whichever
    card holds the recent model."""
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 12000], [
        _resident(f"idle-{card}", card, 6000, meta=meta_of(IP_1, "sub_agent"),
                  last_active=100.0 if card == recent_card else 1.0)
        for card in (0, 1)
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, 0, f"everything equal but recency (card {recent_card} is recent): card 0")
    assert _unload_tuple(choice) == (0, 6000, 0, 4000), tuple(choice)


@pytest.mark.parametrize("older,first_victim", [("a", "idle-x-a"), ("b", "idle-x-b")],
                         ids=["older_is_a", "older_is_b"])
async def test_inside_the_chosen_card_the_less_recently_active_model_of_equal_tag_is_unloaded_first(
        tmp_path, monkeypatch, older, first_victim):
    """Card X reads 12000 free and holds two 3000 sub_agent models of the first-listed client (4000 to
    unload: both are needed). Card Y reads 14000 free and holds a 3000 main model of the same client
    (2000 to unload). The worse-tagged card X is chosen although Y needs fewer MiB; then, inside X,
    the victim order unloads the less recently active of its two models first."""
    last = {"a": 100.0, "b": 100.0}
    last[older] = 1.0
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 14000], [
        _resident("idle-x-a", 0, 3000, meta=meta_of(IP_1, "sub_agent"), last_active=last["a"]),
        _resident("idle-x-b", 0, 3000, meta=meta_of(IP_1, "sub_agent"), last_active=last["b"]),
        _resident("idle-y", 1, 3000, meta=meta_of(IP_1, "main"), last_active=50.0),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, 0, "the worse-tagged card is chosen whatever the recency")
    assert _unload_tuple(choice) == (0, 6000, 0, 4000), tuple(choice)
    victim = mgr._lru_idle_unloadable(main_gpu=0, split_mode="none")
    assert victim is not None and victim.model_tag == first_victim, (
        f"first victim inside the chosen card: {victim and victim.model_tag}, wanted {first_victim}")


# ------------------------------------------------------------------------------------------------
# (f) the tag is read from the one identity reader: the stamp wins over the idle stash
# ------------------------------------------------------------------------------------------------
def _stamped(tag, card, mib, *, stamp, stash, ip=IP_1):
    r = _resident(tag, card, mib, meta=meta_of(ip, stash))
    r.rank_client_meta = meta_of(ip, stamp)
    return r


@pytest.mark.parametrize("stamp,stash,x_card,want", [
    ("sub_agent", "main", 0, 0),       # stamped worse than the stash and than the other card: X wins
    ("sub_agent", "main", 1, 1),
    ("main", "sub_agent", 0, 1),       # stamped better than the other card (the stash says worse): Y wins
    ("main", "sub_agent", 1, 0),
], ids=[
    "stamp_worse_than_stash_on_card_0", "stamp_worse_than_stash_on_card_1",
    "stamp_better_than_stash_on_card_0", "stamp_better_than_stash_on_card_1",
])
async def test_the_tag_of_a_model_is_read_from_its_stamped_identity_and_not_from_its_idle_stash(
        tmp_path, monkeypatch, stamp, stash, x_card, want):
    """Card X (17000 MiB, 10000 to unload) holds a model whose stamped identity (latest turn) says
    `stamp` while its idle stash says `stash`. Card Y (3000 MiB, 2000 to unload) holds a curator model
    with no stamp (its stash counts). Same client. The stamp decides: a sub_agent stamp is worse than
    curator, so X wins; a main stamp is better than curator, so Y wins (the stash would say the
    opposite in both cases)."""
    y_card = 1 - x_card
    free = [0, 0]
    free[x_card], free[y_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _stamped("idle-x", x_card, 17000, stamp=stamp, stash=stash),
        _resident("idle-y", y_card, 3000, meta=meta_of(IP_1, "curator")),
    ])
    assert mgr._residents["idle-x"].state is ResidentState.IDLE_EVICTABLE
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, f"stamp={stamp} stash={stash}: the stamped identity is the one read")
    want_tuple = (want, 17000, 0, 10000) if want == x_card else (want, 3000, 0, 2000)
    assert _unload_tuple(choice) == want_tuple, tuple(choice)


# ------------------------------------------------------------------------------------------------
# a client in no rule stays first; a card that unloads nobody stays best of all
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("unlisted_card,want", [(0, 0), (1, 1)], ids=["unlisted_on_card_0", "unlisted_on_card_1"])
async def test_a_client_in_no_rule_is_still_unloaded_before_the_worst_tag_of_a_listed_client(
        tmp_path, monkeypatch, unlisted_card, want):
    """The unlisted client's model is the big one (17000 MiB, 10000 to unload); the listed client's
    model holds the worst tag (no class label) and is the small one (3000 MiB, 2000 to unload). An
    unlisted client goes first, even before the worst tag of the lowest-ranked listed client."""
    listed_card = 1 - unlisted_card
    free = [0, 0]
    free[unlisted_card], free[listed_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-unlisted", unlisted_card, 17000, meta={"ip": IP_UNLISTED, "is_main": True}),
        _resident("idle-listed", listed_card, 3000, meta=meta_of(IP_3, "no_labels")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "a client in no rule goes before every listed client, whatever the tags")
    assert _unload_tuple(choice) == (want, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("x_card,want", [(0, 0), (1, 1)], ids=["zero_unload_on_card_0", "zero_unload_on_card_1"])
async def test_a_card_that_unloads_nobody_wins_whatever_the_tag_of_the_other_cards_model(
        tmp_path, monkeypatch, x_card, want):
    """Card X reads 16000 free (fits as it stands, 0 MiB to unload) and holds a 3000 main model of the
    first-listed client. Card Y reads 6000 free and holds a 17000 model of the same client with the
    worst tag (10000 to unload). Card X wins: it unloads nobody."""
    y_card = 1 - x_card
    free = [0, 0]
    free[x_card], free[y_card] = 16000, 6000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-x", x_card, 3000, meta=meta_of(IP_1, "main")),
        _resident("idle-y", y_card, 17000, meta=meta_of(IP_1, "no_labels")),
    ])
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "the card that unloads nobody beats a card that has to unload a model")
    assert _unload_tuple(choice) == (want, 3000, 0, 0), tuple(choice)


# ------------------------------------------------------------------------------------------------
# an operator who ranks `unclassified` above another tag: a model whose class cannot be resolved
# carries the rank `unclassified` carries, in the card choice and in the unload order alike
# ------------------------------------------------------------------------------------------------
# unclassified 1, main 2, curator 3, compression 4, sub_agent 5: an unresolved model is the MOST
# protected tag of its client, not the last one
UNCL_FIRST = FastLaneTagRanks(unclassified=1, main=2, curator=3, compression=4, sub_agent=5)
UNCL_FIRST_RULES = [FastLaneRule(address=ip, tag_ranks=UNCL_FIRST) for ip in (IP_1, IP_2, IP_3)]
# main 1, curator 2, unclassified 3, compression 4, sub_agent 5: an unresolved model sits in the middle
UNCL_MIDDLE = FastLaneTagRanks(main=1, curator=2, unclassified=3, compression=4, sub_agent=5)
UNCL_MIDDLE_RULES = [FastLaneRule(address=ip, tag_ranks=UNCL_MIDDLE) for ip in (IP_1, IP_2, IP_3)]
UNRESOLVED_KINDS = ["no_labels", "unknown_role", "user_message_role"]


@pytest.mark.parametrize("kind", UNRESOLVED_KINDS)
@pytest.mark.parametrize("big_card,want", [(0, 1), (1, 0)], ids=["unresolved_on_card_0", "unresolved_on_card_1"])
async def test_an_unresolved_model_ranked_above_another_tag_is_not_the_last_in_the_card_choice(
        tmp_path, monkeypatch, kind, big_card, want):
    """Same client on both cards, `unclassified` ranked 1 and main ranked 2. The unresolved model is
    the big one (17000 MiB, 10000 to unload); the main model is the small one (3000 MiB, 2000 to
    unload). The unresolved model carries rank 1, so the main model (rank 2) is unloaded first and the
    small card wins, not the card of the unresolved model."""
    small_card = 1 - big_card
    free = [0, 0]
    free[big_card], free[small_card] = 6000, 14000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-unresolved", big_card, 17000, meta=meta_of(IP_1, kind)),
        _resident("idle-main", small_card, 3000, meta=meta_of(IP_1, "main")),
    ], rules=UNCL_FIRST_RULES)
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, f"[{kind}] unclassified ranked 1: the main model (rank 2) is unloaded first")
    assert _unload_tuple(choice) == (want, 3000, 0, 2000), tuple(choice)


@pytest.mark.parametrize("small_card,want", [(0, 1), (1, 0)], ids=["unresolved_on_card_0", "unresolved_on_card_1"])
async def test_an_unresolved_model_ranked_above_another_tag_wins_against_fewer_mib_to_unload_in_the_card_choice(
        tmp_path, monkeypatch, small_card, want):
    """The same table, the other way round on size: the unresolved model is the small one (3000 MiB,
    2000 to unload) and the main model the big one (17000 MiB, 10000 to unload). The main model
    (rank 2) is unloaded first, so its card wins although it needs far more unloading."""
    big_card = 1 - small_card
    free = [0, 0]
    free[small_card], free[big_card] = 14000, 6000
    mgr = _tag_manager(tmp_path, monkeypatch, free, [
        _resident("idle-unresolved", small_card, 3000, meta=meta_of(IP_1, "no_labels")),
        _resident("idle-main", big_card, 17000, meta=meta_of(IP_1, "main")),
    ], rules=UNCL_FIRST_RULES)
    choice = await _choose(mgr, NEED)
    _assert_card_chosen(choice, want, "unclassified ranked 1: the main model (rank 2) is unloaded first, whatever its size")
    assert _unload_tuple(choice) == (want, 17000, 0, 10000), tuple(choice)


@pytest.mark.parametrize("kind", UNRESOLVED_KINDS)
async def test_an_unresolved_model_ranked_above_another_tag_is_not_the_first_victim_inside_the_card(
        tmp_path, monkeypatch, kind):
    """One card, `unclassified` ranked 1 and main ranked 2, same client. The main model is the worse
    tag, so it is the first victim; the unresolved model is not. The unresolved model is the less
    recently active one (1.0 against 100.0), so a victim order that fell back to recency, or that put
    the unresolved model last-ranked, would name it."""
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 14000], [
        _resident("idle-unresolved", 0, 3000, meta=meta_of(IP_1, kind), last_active=1.0),
        _resident("idle-main", 0, 3000, meta=meta_of(IP_1, "main"), last_active=100.0),
    ], rules=UNCL_FIRST_RULES)
    victim = mgr._lru_idle_unloadable(main_gpu=0, split_mode="none")
    assert victim is not None and victim.model_tag == "idle-main", (
        f"[{kind}] first victim: {victim and victim.model_tag}, wanted idle-main")


@pytest.mark.parametrize("kind", UNRESOLVED_KINDS)
async def test_an_unresolved_model_sorts_in_the_unload_order_where_unclassified_is_ranked(
        tmp_path, monkeypatch, kind):
    """One client, `unclassified` ranked 3 between curator (2) and compression (4). Five idle models of
    the client, one per tag (the unresolved one counts as unclassified), all equally recent. The unload
    order, least protected first, is sub_agent, compression, the unresolved model, curator, main: the
    unresolved model is neither first nor last."""
    mgr = _tag_manager(tmp_path, monkeypatch, [12000, 14000], [
        _resident("idle-main", 0, 1000, meta=meta_of(IP_1, "main")),
        _resident("idle-unresolved", 0, 1000, meta=meta_of(IP_1, kind)),
        _resident("idle-curator", 0, 1000, meta=meta_of(IP_1, "curator")),
        _resident("idle-compression", 0, 1000, meta=meta_of(IP_1, "compression")),
        _resident("idle-sub-agent", 0, 1000, meta=meta_of(IP_1, "sub_agent")),
    ], rules=UNCL_MIDDLE_RULES)
    table = mgr._fastlane_table()
    mine = [r for r in mgr._residents.values() if (r.model_tag or "").startswith("idle-")]
    assert len(mine) == 5, [r.model_tag for r in mgr._residents.values()]   # the manager's own placeholder is left out
    order = [r.model_tag for r in sorted(mine, key=lambda r: mgr._resident_unload_priority_key(r, table))]
    assert order == ["idle-sub-agent", "idle-compression", "idle-unresolved", "idle-curator", "idle-main"], (
        f"[{kind}] unload order, least protected first: {order}")
    victim = mgr._lru_idle_unloadable(main_gpu=0, split_mode="none")
    assert victim is not None and victim.model_tag == "idle-sub-agent", victim and victim.model_tag
