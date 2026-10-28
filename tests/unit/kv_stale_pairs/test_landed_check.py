"""kv_stale_pairs: nothing is deleted unless the new pair's size is proven (landed bytes == engine n_written)."""

import pytest

from kv_stale_pairs.helpers import CARD, LANDED, call, listing, write_pair


def _seed(tmp_path):
    write_pair(tmp_path, 11500)
    write_pair(tmp_path, 11501)
    return listing(tmp_path)


def _assert_kept(tmp_path, res, before, landed, n_written):
    assert res.skipped == "landed_size_mismatch", f"expected landed_size_mismatch, got {res.skipped!r} ({CARD})"
    assert res.dropped == (), f"nothing may be dropped on a landed mismatch ({CARD})"
    assert listing(tmp_path) == before, f"the old pair must stay on disk ({CARD})"
    assert (res.landed_bytes, res.n_written) == (landed, n_written), f"the arguments are echoed back ({CARD})"


def test_equal_positive_ints_run_the_drop(tmp_path):
    before = _seed(tmp_path)
    res = call(tmp_path, landed=LANDED, n_written=LANDED)
    assert res.skipped is None and [p.port for p in res.dropped] == [11500], f"clean landed check must drop ({CARD})"
    assert listing(tmp_path) != before


def test_n_written_none_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=LANDED, n_written=None), before, LANDED, None)


def test_landed_none_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=None, n_written=LANDED), before, None, LANDED)


def test_a_string_n_written_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=LANDED, n_written=str(LANDED)), before, LANDED, str(LANDED))


def test_a_string_landed_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=str(LANDED), n_written=LANDED), before, str(LANDED), LANDED)


def test_a_float_equal_in_value_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=float(LANDED), n_written=float(LANDED)), before,
                 float(LANDED), float(LANDED))


def test_a_float_n_written_against_an_int_landed_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=LANDED, n_written=float(LANDED)), before, LANDED, float(LANDED))


def test_bool_values_are_not_ints_here(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=True, n_written=True), before, True, True)


def test_bool_n_written_against_int_one_keeps_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=1, n_written=True), before, 1, True)


@pytest.mark.parametrize("landed,n_written", [(1000, 999), (999, 1000), (1000, 2000)])
def test_unequal_sizes_keep_the_old_pair(tmp_path, landed, n_written):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=landed, n_written=n_written), before, landed, n_written)


def test_equal_negative_sizes_keep_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=-5, n_written=-5), before, -5, -5)


def test_equal_zero_sizes_keep_the_old_pair(tmp_path):
    before = _seed(tmp_path)
    _assert_kept(tmp_path, call(tmp_path, landed=0, n_written=0), before, 0, 0)


def test_the_landed_check_runs_before_the_served_check(tmp_path):
    before = _seed(tmp_path)
    res = call(tmp_path, live_turns=None, landed=None, n_written=None)
    assert res.skipped == "landed_size_mismatch", f"landed check is step 1, before served_unknown ({CARD})"
    assert listing(tmp_path) == before


def test_the_landed_check_runs_before_the_directory_check(tmp_path):
    res = call(tmp_path / "absent", landed=0, n_written=0)
    assert res.skipped == "landed_size_mismatch", f"landed check is step 1, before dir_missing ({CARD})"
