"""Module-interface tests for kv_ram_cap.split_protection and ProtectionSplit (HARD/SOFT tiers).

The hash and identity functions are fakes that record their calls, so each test can pin both
the sets and the order in which the module asked for them.
"""

import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_ram_cap import ProtectionSplit, split_protection  # noqa: E402

CARD = "see src/turbohaul/kv_ram_cap/AGENTS.md"


class _Fakes:
    """thread_hash wraps the text as H(text); identity reads 'ident' from client_meta."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.hash_calls: list[str] = []
        self.identity_calls: list[str] = []

    def thread_hash(self, text: str) -> str:
        self.hash_calls.append(text)
        if text == self.fail_on:
            raise ValueError("hash boom")
        return f"H({text})"

    def identity(self, tid, client_meta, chain) -> str:
        self.identity_calls.append(tid)
        return (client_meta or {}).get("ident", "")


def test_thread_id_hash_goes_into_the_hard_tier():
    f = _Fakes()
    split = split_protection([("t1", None, None)], [], f.thread_hash, f.identity)
    assert split.hard == {"H(t1)"}, f"split_protection: the thread-id hash must land in HARD ({CARD})"
    assert split.soft == set(), f"split_protection: no SOFT triples means an empty SOFT tier ({CARD})"


def test_identity_hash_is_added_next_to_the_thread_hash_when_it_differs():
    f = _Fakes()
    split = split_protection([("t1", {"ident": "S1/R"}, None)], [], f.thread_hash, f.identity)
    assert split.hard == {"H(t1)", "H(S1/R)"}, (
        f"split_protection: a differing identity hash must be added too ({CARD})")


def test_identity_equal_to_the_thread_id_adds_no_second_hash():
    f = _Fakes()
    split = split_protection([("t1", {"ident": "t1"}, None)], [], f.thread_hash, f.identity)
    assert split.hard == {"H(t1)"}, f"split_protection: identity equal to the tid adds nothing ({CARD})"
    assert f.hash_calls == ["t1"], f"split_protection: the tid must be hashed once only ({CARD})"


def test_empty_identity_is_not_hashed():
    f = _Fakes()
    split = split_protection([("t1", {"ident": ""}, None)], [], f.thread_hash, f.identity)
    assert split.hard == {"H(t1)"}, f"split_protection: an empty identity must not be added ({CARD})"
    assert f.hash_calls == ["t1"], f"split_protection: an empty identity must not reach thread_hash ({CARD})"


def test_empty_or_none_thread_id_is_skipped_but_identity_is_computed_from_empty_text():
    for tid in (None, ""):
        f = _Fakes()
        split = split_protection([(tid, {"ident": "S9/R"}, None)], [], f.thread_hash, f.identity)
        assert split.hard == {"H(S9/R)"}, (
            f"split_protection: tid={tid!r} must add only the identity hash ({CARD})")
        assert f.hash_calls == ["S9/R"], f"split_protection: tid={tid!r} must not be hashed ({CARD})"
        assert f.identity_calls == [""], f"split_protection: identity must be asked with '' for tid={tid!r} ({CARD})"


def test_empty_tid_with_no_identity_adds_nothing():
    f = _Fakes()
    split = split_protection([(None, None, None)], [], f.thread_hash, f.identity)
    assert split == ProtectionSplit(hard=set(), soft=set()), (
        f"split_protection: a triple with no tid and no identity must add nothing ({CARD})")


def test_hard_triples_are_added_before_soft_and_the_tiers_stay_apart():
    f = _Fakes()
    split = split_protection([("h1", None, None)], [("s1", None, None)], f.thread_hash, f.identity)
    assert f.hash_calls == ["h1", "s1"], f"split_protection: HARD triples must be processed first ({CARD})"
    assert split.hard == {"H(h1)"}, f"split_protection: the SOFT thread must not leak into HARD ({CARD})"
    assert split.soft == {"H(s1)"}, f"split_protection: the HARD thread must not leak into SOFT ({CARD})"


def test_protected_is_the_union_and_a_new_set_on_every_call():
    split = ProtectionSplit(hard={"a"}, soft={"a", "b"})
    p = split.protected
    assert p == {"a", "b"}, f"ProtectionSplit.protected must be hard | soft ({CARD})"
    assert p is not split.hard and p is not split.soft, f"ProtectionSplit.protected must be a new set ({CARD})"
    p.add("zzz")
    assert split.protected == {"a", "b"}, f"ProtectionSplit.protected must not share state across calls ({CARD})"
    assert split.hard == {"a"}, f"ProtectionSplit.hard must not change when protected is edited ({CARD})"


def test_tiers_are_plain_mutable_sets():
    f = _Fakes()
    split = split_protection([("t1", None, None)], [("t2", None, None)], f.thread_hash, f.identity)
    assert type(split.hard) is set and type(split.soft) is set, f"split tiers must be plain set values ({CARD})"


def test_callable_failure_calls_on_error_once_and_returns_the_partial_split():
    f = _Fakes(fail_on="bad")
    errors = []
    split = split_protection(
        [("ok", None, None), ("bad", None, None), ("later", None, None)],
        [("s1", None, None)],
        f.thread_hash, f.identity,
        on_error=lambda: errors.append("protection"),
    )
    assert len(errors) == 1, f"split_protection: on_error must run exactly once on a callable failure ({CARD})"
    assert split.hard == {"H(ok)"}, f"split_protection: the partial HARD tier must keep what was added ({CARD})"
    assert split.soft == set(), f"split_protection: nothing after the failure may be added to SOFT ({CARD})"
    assert f.hash_calls == ["ok", "bad"], f"split_protection: the scan must stop at the failing triple ({CARD})"


def _hard_then_iteration_fails():
    yield ("a", None, None)
    raise RuntimeError("iteration boom")


def test_iteration_failure_keeps_the_triples_added_before_it():
    f = _Fakes()
    errors = []
    split = split_protection(_hard_then_iteration_fails(), [("s", None, None)], f.thread_hash, f.identity,
                             on_error=lambda: errors.append("iter"))
    assert len(errors) == 1, f"split_protection: on_error must run once when iteration fails ({CARD})"
    assert split.hard == {"H(a)"}, f"split_protection: a failed iteration keeps the earlier HARD triples ({CARD})"
    assert split.soft == set(), f"split_protection: a failed iteration must not add SOFT triples ({CARD})"


def test_failure_with_no_on_error_still_returns_a_split_and_does_not_raise():
    f = _Fakes(fail_on="bad")
    split = split_protection([("bad", None, None)], [], f.thread_hash, f.identity)
    assert split == ProtectionSplit(hard=set(), soft=set()), (
        f"split_protection: with on_error=None a failure must return the empty partial split ({CARD})")
