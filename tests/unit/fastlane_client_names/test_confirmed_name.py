"""confirmed_name: a candidate counts only when its forward lookup holds the address; shortest wins."""
import ipaddress
import itertools

from turbohaul.fastlane_client_names import confirmed_name

ADDR = ipaddress.ip_address("192.0.2.5")
OTHER = ipaddress.ip_address("192.0.2.6")
ADDR6 = ipaddress.ip_address("2001:db8::5")


def test_single_confirmed_candidate_is_chosen():
    assert confirmed_name(ADDR, {"app-one": frozenset({ADDR})}) == "app-one"


def test_candidate_whose_addresses_lack_the_address_is_not_chosen_even_if_shorter():
    forward = {"app": frozenset({OTHER}), "app.net_default": frozenset({ADDR})}
    assert confirmed_name(ADDR, forward) == "app.net_default"


def test_shortest_confirmed_wins():
    forward = {
        "app-one.net_default": {ADDR},
        "app-one": {ADDR},
        "app-one.net_default.example": {ADDR},
    }
    assert confirmed_name(ADDR, forward) == "app-one"


def test_shortest_is_by_character_length_not_label_count():
    forward = {"a.b": {ADDR}, "abcdef": {ADDR}}
    assert confirmed_name(ADDR, forward) == "a.b"


def test_tie_is_broken_alphabetically():
    assert confirmed_name(ADDR, {"bb": {ADDR}, "aa": {ADDR}}) == "aa"


def test_tie_result_does_not_depend_on_dict_order():
    items = [("cc", {ADDR}), ("aa", {ADDR}), ("bb", {ADDR})]
    results = {confirmed_name(ADDR, dict(order)) for order in itertools.permutations(items)}
    assert results == {"aa"}


def test_tie_with_unconfirmed_shorter_neighbours_still_alphabetical():
    forward = {"zz": {ADDR}, "yy": {ADDR}, "a": {OTHER}}
    assert confirmed_name(ADDR, forward) == "yy"


def test_tie_is_ordinary_string_order():
    assert confirmed_name(ADDR, {"b1": {ADDR}, "C2": {ADDR}}) == "C2"


def test_none_confirmed_gives_none():
    forward = {"app": {OTHER}, "app.net_default": {OTHER}}
    assert confirmed_name(ADDR, forward) is None


def test_empty_forward_gives_none():
    assert confirmed_name(ADDR, {}) is None


def test_name_mapped_to_none_is_not_confirmed():
    assert confirmed_name(ADDR, {"app": None}) is None
    assert confirmed_name(ADDR, {"app": None, "app.net_default": {ADDR}}) == "app.net_default"


def test_name_mapped_to_empty_iterable_is_not_confirmed():
    for empty in ([], (), set(), frozenset(), iter(())):
        assert confirmed_name(ADDR, {"app": empty}) is None
    assert confirmed_name(ADDR, {"app": [], "app.net_default": [ADDR]}) == "app.net_default"


def test_iterable_kinds_all_work():
    for make in (frozenset, set, list, tuple, iter, lambda items: (a for a in items)):
        assert confirmed_name(ADDR, {"app": make([OTHER, ADDR])}) == "app", make
        assert confirmed_name(ADDR, {"app": make([OTHER])}) is None, make


def test_address_among_several_is_found():
    assert confirmed_name(ADDR, {"app": [OTHER, ADDR6, ADDR]}) == "app"


def test_ipv6_address():
    assert confirmed_name(ADDR6, {"app": {ADDR6}, "app.net_default": {ADDR6}}) == "app"
    assert confirmed_name(ADDR6, {"app": {ADDR}}) is None


def test_address_of_a_different_type_than_the_members_never_confirms():
    forward = {"app": {ADDR}, "app.net_default": {ADDR}}
    assert confirmed_name("192.0.2.5", forward) is None
    assert confirmed_name(int(ADDR), forward) is None
    assert confirmed_name(None, forward) is None


def test_members_of_a_different_type_than_the_address_never_confirm():
    assert confirmed_name(ADDR, {"app": ["192.0.2.5"]}) is None


def test_same_inputs_same_answer():
    forward = {"bb": {ADDR}, "aa": {ADDR}, "c": {OTHER}}
    assert confirmed_name(ADDR, forward) == confirmed_name(ADDR, forward) == "aa"


def test_forward_mapping_is_not_modified():
    forward = {"app": [ADDR], "app.net_default": [ADDR]}
    snapshot = {k: list(v) for k, v in forward.items()}
    confirmed_name(ADDR, forward)
    assert forward == snapshot


def test_wrong_address_chooses_nothing_even_though_names_exist():
    assert confirmed_name(OTHER, {"app": {ADDR}}) is None
