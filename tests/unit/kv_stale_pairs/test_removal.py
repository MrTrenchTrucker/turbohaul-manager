"""kv_stale_pairs: how a stale pair is removed (bin, .ckpt, .json) and when it counts as dropped."""

import os

from kv_stale_pairs.helpers import CARD, TH, bin_name, call, listing, write_pair


class _Remover:
    """Records every path asked for, then deletes it for real unless a rule says otherwise."""

    def __init__(self, fail=None):
        self.fail = fail or {}
        self.calls = []

    def __call__(self, path):
        self.calls.append(os.path.basename(path))
        err = self.fail.get(os.path.basename(path))
        if err is not None:
            raise err
        os.remove(path)


def test_the_bin_the_ckpt_and_the_json_all_go_in_that_order(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover()
    res = call(tmp_path, remove=rm)
    assert rm.calls == [name, name + ".ckpt", name[:-4] + ".json"], f"bin, then .ckpt, then .json ({CARD})"
    assert listing(tmp_path) == [] and [p.name for p in res.dropped] == [name]


def test_the_default_remover_deletes_real_files(tmp_path):
    write_pair(tmp_path, 11500)
    res = call(tmp_path)
    assert len(res.dropped) == 1 and listing(tmp_path) == [], f"the default remove is os.remove ({CARD})"


def test_a_missing_ckpt_is_fine(tmp_path):
    name = write_pair(tmp_path, 11500, ckpt=False)
    res = call(tmp_path)
    assert [p.name for p in res.dropped] == [name], f"a pair without a .ckpt is still dropped ({CARD})"
    assert listing(tmp_path) == []


def test_a_missing_json_is_fine(tmp_path):
    name = write_pair(tmp_path, 11500, meta=False)
    res = call(tmp_path)
    assert [p.name for p in res.dropped] == [name], f"a pair without a .json is still dropped ({CARD})"
    assert listing(tmp_path) == []


def test_a_remove_that_raises_on_the_ckpt_does_not_stop_the_json(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name + ".ckpt": PermissionError("busy")})
    res = call(tmp_path, remove=rm)
    assert rm.calls[-1] == name[:-4] + ".json", f"the loop goes on after a failed target ({CARD})"
    assert listing(tmp_path) == [name + ".ckpt"] and [p.name for p in res.dropped] == [name]


def test_a_remove_that_raises_on_the_json_still_counts_the_pair_dropped(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name[:-4] + ".json": OSError("io")})
    res = call(tmp_path, remove=rm)
    assert [p.name for p in res.dropped] == [name], f"a pair counts once its bin is gone ({CARD})"
    assert listing(tmp_path) == [name[:-4] + ".json"]


def test_a_remove_that_raises_on_the_bin_is_not_counted_as_dropped(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name: PermissionError("busy")})
    res = call(tmp_path, remove=rm)
    assert res.skipped is None and res.dropped == (), f"a bin that did not go is not dropped ({CARD})"
    assert (tmp_path / name).exists()


def test_a_bin_that_raises_still_lets_its_ckpt_and_json_be_tried(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name: PermissionError("busy")})
    call(tmp_path, remove=rm)
    assert rm.calls == [name, name + ".ckpt", name[:-4] + ".json"], f"each target is tried ({CARD})"


def test_a_bin_already_gone_is_not_counted_as_dropped(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name: FileNotFoundError("gone")})
    res = call(tmp_path, remove=rm)
    assert res.dropped == (), f"this call did not remove the bin, so it did not drop it ({CARD})"
    assert rm.calls[1:] == [name + ".ckpt", name[:-4] + ".json"]


def test_a_failing_pair_does_not_stop_the_next_pair(tmp_path):
    first = write_pair(tmp_path, 11498)
    second = write_pair(tmp_path, 11500)
    rm = _Remover(fail={first: PermissionError("busy")})
    res = call(tmp_path, remove=rm)
    assert [p.name for p in res.dropped] == [second], f"the second pair is still dropped ({CARD})"


def test_a_remove_raising_something_other_than_oserror_never_escapes(tmp_path):
    name = write_pair(tmp_path, 11500)
    rm = _Remover(fail={name: RuntimeError("boom")})
    res = call(tmp_path, remove=rm)
    assert res.dropped == (), f"the function never raises ({CARD})"


def test_pairs_are_removed_in_name_order(tmp_path):
    for port in (11502, 11499, 11500):
        write_pair(tmp_path, port)
    rm = _Remover()
    call(tmp_path, remove=rm)
    bins = [c for c in rm.calls if c.endswith(".bin")]
    assert bins == sorted(bins) == [bin_name(11499), bin_name(11500), bin_name(11502)], f"name order ({CARD})"


def test_other_threads_files_are_never_passed_to_remove(tmp_path):
    write_pair(tmp_path, 11500, thread_hash="ffffffffffff")
    write_pair(tmp_path, 11500)
    rm = _Remover()
    call(tmp_path, remove=rm)
    assert all(TH in c for c in rm.calls) and len(rm.calls) == 3, f"only this thread's three files ({CARD})"
