"""kv_stale_pairs: which saved bins count as the thread's stale pairs (real files, through drop_stale_pairs)."""

from kv_stale_pairs.helpers import CARD, NEW_PORT, OTHER_TH, TAG, TH, bin_name, call, listing, write_pair
from turbohaul.kv_stale_pairs import StalePair


def test_drops_the_threads_pair_on_another_port_with_its_bytes(tmp_path):
    write_pair(tmp_path, 11500, bin_bytes=700, ckpt_bytes=50)
    write_pair(tmp_path, NEW_PORT, bin_bytes=900)
    res = call(tmp_path)
    assert res.dropped == (StalePair(bin_name(11500), 11500, 0, 750),), f"one stale pair, 700+50 bytes ({CARD})"
    assert listing(tmp_path) == sorted([bin_name(NEW_PORT), bin_name(NEW_PORT) + ".ckpt",
                                        bin_name(NEW_PORT)[:-4] + ".json"])


def test_never_drops_a_pair_on_the_new_port_whatever_its_slot(tmp_path):
    write_pair(tmp_path, NEW_PORT, slot=0)
    write_pair(tmp_path, NEW_PORT, slot=3)
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.skipped is None and res.dropped == (), f"pairs on the new port are never stale ({CARD})"
    assert listing(tmp_path) == before


def test_never_drops_a_pair_on_a_served_port(tmp_path):
    write_pair(tmp_path, 11500)
    write_pair(tmp_path, 11502)
    res = call(tmp_path, live_turns=[(11500, [{TH}])])
    assert [p.port for p in res.dropped] == [11502], f"11500 is served, only 11502 is stale ({CARD})"


def test_never_drops_another_threads_pair(tmp_path):
    write_pair(tmp_path, 11500, thread_hash=OTHER_TH)
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.dropped == () and listing(tmp_path) == before, f"another thread's pair must stay ({CARD})"


def test_never_drops_another_models_pair(tmp_path):
    write_pair(tmp_path, 11500, tag="llama-8b")
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.dropped == () and listing(tmp_path) == before, f"another model's pair must stay ({CARD})"


def test_a_model_tag_that_only_starts_with_ours_does_not_match(tmp_path):
    write_pair(tmp_path, 11500, tag=TAG + "-q4")
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.dropped == () and listing(tmp_path) == before, f"a longer tag is a different model ({CARD})"


def test_a_dotted_model_tag_matches_on_the_whole_name(tmp_path):
    name = write_pair(tmp_path, 11500, tag="m.v2", slot=1, bin_bytes=3, ckpt_bytes=2)
    res = call(tmp_path, tag="m.v2")
    assert res.dropped == (StalePair(name, 11500, 1, 5),), f"dots in the tag are literal ({CARD})"


def test_a_tag_with_regex_characters_is_matched_literally(tmp_path):
    write_pair(tmp_path, 11500, tag="mXv2")
    before = listing(tmp_path)
    res = call(tmp_path, tag="m.v2")
    assert res.dropped == () and listing(tmp_path) == before, f"the tag must be escaped, not a pattern ({CARD})"


def test_a_thread_hash_with_regex_characters_is_matched_literally(tmp_path):
    write_pair(tmp_path, 11500, thread_hash="abc")
    before = listing(tmp_path)
    res = call(tmp_path, thread_hash="a.c")
    assert res.dropped == () and listing(tmp_path) == before, f"the thread hash must be escaped ({CARD})"


def test_files_that_are_not_a_bin_name_are_ignored(tmp_path):
    base = bin_name(11500)
    for junk in (base + ".ckpt", base[:-4] + ".json", base + ".tmp", base[:-4] + ".shadow.bin", "junk.bin"):
        (tmp_path / junk).write_bytes(b"x")
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.skipped is None and res.dropped == () and listing(tmp_path) == before, f"only exact bins match ({CARD})"


def test_every_slot_on_a_stale_port_goes_and_the_result_is_sorted_by_name(tmp_path):
    write_pair(tmp_path, 11500, slot=2)
    write_pair(tmp_path, 11500, slot=0)
    write_pair(tmp_path, 11499, slot=1)
    names = [p.name for p in call(tmp_path).dropped]
    assert names == sorted(names) and len(names) == 3, f"three stale pairs, sorted by name ({CARD})"
    assert listing(tmp_path) == []


def test_the_slot_id_and_port_are_read_from_the_name(tmp_path):
    write_pair(tmp_path, 11499, slot=7)
    (pair,) = call(tmp_path).dropped
    assert (pair.port, pair.slot_id) == (11499, 7), f"port and slot come from the file name ({CARD})"


def test_pair_bytes_count_the_bin_and_the_ckpt_but_not_the_json(tmp_path):
    write_pair(tmp_path, 11500, bin_bytes=100, ckpt_bytes=30)
    (pair,) = call(tmp_path).dropped
    assert pair.pair_bytes == 130, f"pair_bytes is bin plus .ckpt ({CARD})"


def test_a_missing_ckpt_counts_zero_bytes(tmp_path):
    write_pair(tmp_path, 11500, bin_bytes=100, ckpt=False)
    (pair,) = call(tmp_path).dropped
    assert pair.pair_bytes == 100, f"no .ckpt means the pair is the bin alone ({CARD})"


def test_a_directory_named_like_a_bin_is_ignored(tmp_path):
    (tmp_path / bin_name(11500)).mkdir()
    asked = []
    res = call(tmp_path, remove=asked.append)
    assert res.skipped is None and res.dropped == (), f"only regular files are bins ({CARD})"
    assert asked == [], f"remove must not be asked to delete a directory ({CARD})"
    assert (tmp_path / bin_name(11500)).is_dir()


def test_a_dangling_link_named_like_a_bin_is_ignored(tmp_path):
    (tmp_path / bin_name(11500)).symlink_to(tmp_path / "nowhere")
    res = call(tmp_path)
    assert res.skipped is None and res.dropped == (), f"a link to nothing is not a file ({CARD})"
    assert (tmp_path / bin_name(11500)).is_symlink()


def test_a_model_tag_differing_only_at_its_dot_is_another_model_beside_a_real_stale_pair(tmp_path):
    real = write_pair(tmp_path, 11500)
    lookalike = write_pair(tmp_path, 11502, tag=TAG.replace(".", "x"))
    assert lookalike != real and TAG.replace(".", "x") != TAG
    res = call(tmp_path)
    assert [p.name for p in res.dropped] == [real], f"only the real tag's pair goes, the dot is literal ({CARD})"
    assert lookalike in listing(tmp_path), f"a tag that differs at the dot is another model ({CARD})"
