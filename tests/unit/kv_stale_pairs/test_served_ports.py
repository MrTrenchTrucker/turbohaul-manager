"""kv_stale_pairs: a port a live turn serves for this thread is never touched; an unknown port stops the drop."""

import pytest

from kv_stale_pairs.helpers import CARD, OTHER_TH, TH, call, listing, write_pair


def _two_stale(tmp_path):
    write_pair(tmp_path, 11500)
    write_pair(tmp_path, 11502)
    write_pair(tmp_path, 11501)


def _ports(res):
    return [p.port for p in res.dropped]


def test_a_served_port_is_spared_and_the_other_stale_port_goes(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [{TH}])])
    assert _ports(res) == [11502], f"the served port 11500 must stay, 11502 must go ({CARD})"
    assert any(n.startswith("qwen3.6-27b.p11500.") for n in listing(tmp_path))


@pytest.mark.parametrize("make", [set, frozenset, tuple, list])
def test_the_hash_collection_may_be_any_container(tmp_path, make):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [make([TH])])])
    assert _ports(res) == [11502], f"a {make.__name__} of hashes must spare the served port ({CARD})"


def test_the_thread_in_a_second_hash_collection_of_the_same_entry_is_served(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [{OTHER_TH}, {"cccccccccccc", TH}])])
    assert _ports(res) == [11502], f"an in-flight style second collection must spare the port ({CARD})"


def test_two_serving_entries_spare_both_ports(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [{TH}]), (11502, [{TH}])])
    assert res.skipped is None and res.dropped == (), f"both served ports must stay ({CARD})"
    assert len(listing(tmp_path)) == 9


def test_an_entry_serving_another_thread_does_not_spare_the_port(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [{OTHER_TH}])])
    assert _ports(res) == [11500, 11502], f"a non-matching entry spares nothing ({CARD})"


def test_an_entry_with_no_hash_collections_does_not_spare_the_port(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [])])
    assert _ports(res) == [11500, 11502], f"an entry with no slots serves nobody ({CARD})"


def test_a_non_serving_entry_with_port_none_changes_nothing(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[(None, [{OTHER_TH}])])
    assert res.skipped is None and _ports(res) == [11500, 11502], f"port None matters only when serving ({CARD})"


def test_a_serving_entry_with_port_none_skips_as_served_unknown(tmp_path):
    _two_stale(tmp_path)
    before = listing(tmp_path)
    res = call(tmp_path, live_turns=[(None, [{TH}])])
    assert res.skipped == "served_unknown" and res.dropped == (), f"unknown port must stop the drop ({CARD})"
    assert listing(tmp_path) == before


def test_a_serving_entry_with_port_none_wins_over_a_known_serving_entry(tmp_path):
    _two_stale(tmp_path)
    before = listing(tmp_path)
    res = call(tmp_path, live_turns=[(11500, [{TH}]), (None, [{TH}])])
    assert res.skipped == "served_unknown", f"any unknown serving port stops the drop ({CARD})"
    assert listing(tmp_path) == before


def test_a_serving_entry_whose_port_is_not_an_int_skips_as_served_unknown(tmp_path):
    _two_stale(tmp_path)
    before = listing(tmp_path)
    res = call(tmp_path, live_turns=[("11500", [{TH}])])
    assert res.skipped == "served_unknown", f"a port that is not an int cannot be trusted ({CARD})"
    assert listing(tmp_path) == before


def test_live_turns_none_skips_as_served_unknown(tmp_path):
    _two_stale(tmp_path)
    before = listing(tmp_path)
    res = call(tmp_path, live_turns=None)
    assert res.skipped == "served_unknown" and res.dropped == (), f"a failed live scan must keep every pair ({CARD})"
    assert listing(tmp_path) == before


def test_empty_live_turns_means_no_live_turn_and_the_drop_runs(tmp_path):
    _two_stale(tmp_path)
    res = call(tmp_path, live_turns=[])
    assert res.skipped is None and _ports(res) == [11500, 11502], f"no live turn spares nothing ({CARD})"


def test_malformed_live_turns_never_raise_and_keep_every_pair(tmp_path):
    _two_stale(tmp_path)
    before = listing(tmp_path)
    res = call(tmp_path, live_turns=[42])
    assert res.skipped == "scan_failed" and res.dropped == (), f"a bad shape must not raise or delete ({CARD})"
    assert listing(tmp_path) == before
