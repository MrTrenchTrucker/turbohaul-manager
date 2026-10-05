"""name_candidates: the whole reverse answer first, then trailing labels removed one at a time."""
import pytest

from turbohaul.fastlane_client_names import name_candidates

SAMPLE_NAMES = [
    "app",
    "app-one.net_default",
    "worker-two.net_default.example",
    "a.b.c.d.e",
    "App-One.Net_Default",
]


def test_single_label_gives_itself_only():
    assert name_candidates("app-one") == ("app-one",)


def test_two_labels_whole_name_then_bare_name():
    assert name_candidates("app-one.net_default") == ("app-one.net_default", "app-one")


def test_five_labels_drop_one_trailing_label_at_a_time():
    assert name_candidates("a.b.c.d.e") == ("a.b.c.d.e", "a.b.c.d", "a.b.c", "a.b", "a")


def test_three_labels_order_is_longest_first():
    assert name_candidates("a.b.c") == ("a.b.c", "a.b", "a")


def test_result_is_a_tuple():
    assert type(name_candidates("a.b")) is tuple


def test_case_is_preserved():
    assert name_candidates("App-One.Net_Default") == ("App-One.Net_Default", "App-One")


def test_one_trailing_dot_is_dropped():
    assert name_candidates("a.b.") == ("a.b", "a")
    assert name_candidates("app-one.") == ("app-one",)


def test_two_trailing_dots_are_malformed():
    assert name_candidates("a.b..") == ()


def test_leading_dot_is_malformed():
    assert name_candidates(".a") == ()
    assert name_candidates(".a.b") == ()


def test_double_dot_is_malformed():
    assert name_candidates("a..b") == ()


def test_only_dots_is_malformed():
    assert name_candidates(".") == ()
    assert name_candidates("..") == ()
    assert name_candidates("...") == ()


def test_empty_string_is_malformed():
    assert name_candidates("") == ()


@pytest.mark.parametrize("answer", ["a b", " a", "a ", "a.b c", "a\tb", "a\nb", "a.b\n", "a b"])
def test_whitespace_anywhere_is_malformed(answer):
    assert name_candidates(answer) == ()


@pytest.mark.parametrize("answer", [None, 5, b"a.b", ["a.b"], ("a", "b"), object()])
def test_non_string_is_malformed(answer):
    assert name_candidates(answer) == ()


@pytest.mark.parametrize("answer", ["192.0.2.5", "0.0.0.0", "192.0.2.5."])
def test_ipv4_literal_is_not_a_name(answer):
    assert name_candidates(answer) == ()


@pytest.mark.parametrize("answer", ["2001:db8::1", "::1", "fd00::5", "::ffff:192.0.2.5"])
def test_ipv6_literal_is_not_a_name(answer):
    assert name_candidates(answer) == ()


def test_a_name_that_only_starts_with_digits_is_still_a_name():
    assert name_candidates("192.example") == ("192.example", "192")
    assert name_candidates("1.2.3.example") == ("1.2.3.example", "1.2.3", "1.2", "1")


def test_no_candidate_is_longer_than_the_answer():
    for answer in SAMPLE_NAMES + [name + "." for name in SAMPLE_NAMES]:
        candidates = name_candidates(answer)
        assert candidates, answer
        for candidate in candidates:
            assert len(candidate) <= len(answer), (answer, candidate)


def test_candidate_count_is_the_label_count():
    for answer in SAMPLE_NAMES:
        assert len(name_candidates(answer)) == len(answer.split(".")), answer
        assert len(name_candidates(answer + ".")) == len(answer.split(".")), answer


def test_first_candidate_is_the_whole_name_and_each_is_a_label_prefix_of_it():
    for answer in SAMPLE_NAMES:
        candidates = name_candidates(answer)
        assert candidates[0] == answer
        for candidate in candidates:
            assert answer == candidate or answer.startswith(candidate + ".")


def test_candidates_strictly_shrink_and_have_no_duplicates():
    for answer in SAMPLE_NAMES:
        candidates = name_candidates(answer)
        lengths = [len(c) for c in candidates]
        assert lengths == sorted(lengths, reverse=True)
        assert len(set(candidates)) == len(candidates)


def test_same_input_gives_same_output():
    assert name_candidates("a.b.c") == name_candidates("a.b.c")
