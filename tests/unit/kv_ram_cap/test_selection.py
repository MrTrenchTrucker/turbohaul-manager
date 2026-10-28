"""Module-interface tests for kv_ram_cap.soft_age_targets and soft_ceiling_order (GC bin selection).

Entries are (fn, fpath, mtime, size) tuples. The hash map and tier sets are the caller's
state, passed in, so each test sets them explicitly.
"""

import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_ram_cap import soft_age_targets, soft_ceiling_order  # noqa: E402

CARD = "see src/turbohaul/kv_ram_cap/AGENTS.md"

A = ("a.bin", "/d/a.bin", 1.0, 10)
B = ("b.bin", "/d/b.bin", 2.0, 20)
C = ("c.bin", "/d/c.bin", 3.0, 30)
D = ("d.bin", "/d/d.bin", 0.5, 4)
E = ("e.bin", "/d/e.bin", 4.0, 5)
HASH_BY_PATH = {"/d/a.bin": "HA", "/d/b.bin": "HB", "/d/c.bin": "HC", "/d/d.bin": "HD", "/d/e.bin": "HE"}


def _names(entries) -> list[str]:
    return [e[0] for e in entries]


# ---------------------------------------------------------------- soft_age_targets


def test_candidates_come_first_then_soft_entries_in_entries_order():
    out = soft_age_targets([], [C, B, A], HASH_BY_PATH, {"HA", "HC"})
    assert out == [C, A], f"soft_age_targets: SOFT entries must follow entries order, not name or mtime ({CARD})"
    out = soft_age_targets([B], [C, B, A], HASH_BY_PATH, {"HA", "HC"})
    assert out == [B, C, A], f"soft_age_targets: candidates must come first, then SOFT entries ({CARD})"


def test_no_soft_hashes_returns_the_candidates_only():
    out = soft_age_targets([B], [A, B, C], HASH_BY_PATH, set())
    assert out == [B], f"soft_age_targets: with no SOFT hashes only the candidates remain ({CARD})"


def test_an_entry_that_is_a_candidate_and_soft_appears_twice():
    out = soft_age_targets([B], [B], HASH_BY_PATH, {"HB"})
    assert out == [B, B], f"soft_age_targets: the list is not de-duplicated, as the GC has always built it ({CARD})"


def test_result_is_a_new_list_and_the_inputs_are_not_changed():
    candidates = [C]
    entries = [A, B, C]
    out = soft_age_targets(candidates, entries, HASH_BY_PATH, {"HB"})
    assert out == [C, B], f"soft_age_targets: wrong list for the mixed case ({CARD})"
    assert out is not candidates, f"soft_age_targets must return a new list, not the candidates object ({CARD})"
    assert candidates == [C], f"soft_age_targets must not change the candidates it is given ({CARD})"
    assert entries == [A, B, C], f"soft_age_targets must not change the entries it is given ({CARD})"


# ---------------------------------------------------------------- soft_ceiling_order
# With hard {HA} and soft {HB, HD, HE}: A is HARD, B, D, E are SOFT, C is neither.
# Oldest first by mtime gives D (0.5), B (2.0), E (4.0).

_SURVIVING = [A, B, C, D, E]


def test_ceiling_order_takes_soft_bins_oldest_first():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, {"HB", "HD", "HE"}, set(), HASH_BY_PATH)
    assert _names(out) == ["d.bin", "b.bin", "e.bin"], f"soft_ceiling_order: SOFT bins must be oldest first ({CARD})"


def test_ceiling_order_excludes_deleted_paths():
    out = soft_ceiling_order(_SURVIVING, {"/d/b.bin"}, {"HA"}, {"HB", "HD", "HE"}, set(), HASH_BY_PATH)
    assert _names(out) == ["d.bin", "e.bin"], f"soft_ceiling_order: a bin already deleted must be excluded ({CARD})"


def test_ceiling_order_never_takes_a_hard_bin_even_when_pinned():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, {"HB", "HD", "HE"}, {"/d/a.bin"}, HASH_BY_PATH)
    assert _names(out) == ["d.bin", "b.bin", "e.bin"], (
        f"soft_ceiling_order: a HARD bin must be excluded even when it is pinned ({CARD})")


def test_ceiling_order_never_takes_a_bin_that_is_both_hard_and_soft():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, {"HA", "HB", "HD", "HE"}, set(), HASH_BY_PATH)
    assert _names(out) == ["d.bin", "b.bin", "e.bin"], (
        f"soft_ceiling_order: HARD wins over SOFT, so a bin in both tiers is excluded ({CARD})")


def test_ceiling_order_includes_a_pinned_non_hard_bin():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, set(), {"/d/c.bin"}, HASH_BY_PATH)
    assert _names(out) == ["c.bin"], f"soft_ceiling_order: a pinned non-HARD bin is takeable ({CARD})"


def test_ceiling_order_includes_a_soft_bin_that_is_not_pinned():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, {"HD"}, set(), HASH_BY_PATH)
    assert _names(out) == ["d.bin"], f"soft_ceiling_order: an unpinned SOFT bin is takeable ({CARD})"


def test_ceiling_order_excludes_a_bin_that_is_neither_soft_nor_pinned():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, set(), set(), HASH_BY_PATH)
    assert out == [], f"soft_ceiling_order: a bin that is neither SOFT nor pinned must be excluded ({CARD})"


def test_ceiling_order_excludes_a_bin_with_no_known_hash_unless_pinned():
    hashes = {k: v for k, v in HASH_BY_PATH.items() if k != "/d/e.bin"}
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, {"HB", "HD", "HE"}, set(), hashes)
    assert _names(out) == ["d.bin", "b.bin"], f"soft_ceiling_order: a bin with no hash is not SOFT ({CARD})"


def test_ceiling_order_is_stable_on_mtime_ties():
    x = ("x.bin", "/d/x.bin", 5.0, 1)
    y = ("y.bin", "/d/y.bin", 5.0, 1)
    hashes = {"/d/x.bin": "HX", "/d/y.bin": "HY"}
    out = soft_ceiling_order([y, x], set(), set(), {"HX", "HY"}, set(), hashes)
    assert _names(out) == ["y.bin", "x.bin"], f"soft_ceiling_order: ties keep the input order (stable) ({CARD})"


def test_ceiling_order_returns_a_new_list_and_changes_no_input():
    surviving = list(_SURVIVING)
    deleted = {"/d/x.bin"}
    hard = {"HA"}
    soft = {"HB", "HD", "HE"}
    pinned = {"/d/c.bin"}
    hashes = dict(HASH_BY_PATH)
    out = soft_ceiling_order(surviving, deleted, hard, soft, pinned, hashes)
    assert out is not surviving, f"soft_ceiling_order must return a new list ({CARD})"
    assert surviving == _SURVIVING, f"soft_ceiling_order must not reorder its surviving input ({CARD})"
    assert deleted == {"/d/x.bin"}, f"soft_ceiling_order must not change deleted_paths ({CARD})"
    assert hard == {"HA"} and soft == {"HB", "HD", "HE"}, f"soft_ceiling_order must not change the hash sets ({CARD})"
    assert pinned == {"/d/c.bin"} and hashes == HASH_BY_PATH, (
        f"soft_ceiling_order must not change pins or hashes ({CARD})")


def test_ceiling_order_with_nothing_eligible_returns_an_empty_list():
    out = soft_ceiling_order(_SURVIVING, set(), {"HA"}, set(), set(), HASH_BY_PATH)
    assert out == [], f"soft_ceiling_order: no SOFT and no pinned bins gives an empty list ({CARD})"
