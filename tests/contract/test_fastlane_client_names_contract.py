"""Contract test for the fastlane_client_names module's public interface.

Asserts that fastlane_client_names exports exactly the registry's `public` list
(modules.toml [modules.fastlane_client_names]), that each name is usable, and that
the shapes the module card promises hold.
"""
import inspect
import ipaddress

import turbohaul.fastlane_client_names as names

EXPECTED_PUBLIC = {"name_candidates", "confirmed_name"}


def test_exports_exactly_the_registry_public_names():
    assert set(names.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_present_and_callable():
    for name in EXPECTED_PUBLIC:
        assert hasattr(names, name), f"fastlane_client_names has no attribute {name!r}"
        assert callable(getattr(names, name)), f"fastlane_client_names.{name} is not callable"


def test_signatures_match_the_card():
    assert list(inspect.signature(names.name_candidates).parameters) == ["reverse_answer"]
    assert list(inspect.signature(names.confirmed_name).parameters) == ["address", "forward"]


def test_name_candidates_returns_a_tuple_of_strings_whole_name_first():
    result = names.name_candidates("app-one.net_default")
    assert result == ("app-one.net_default", "app-one")
    assert type(result) is tuple
    assert all(type(item) is str for item in result)


def test_name_candidates_malformed_answer_is_an_empty_tuple():
    for bad in ("", ".a", "a..b", ".", "a b", "192.0.2.5", None, 5):
        assert names.name_candidates(bad) == (), f"{bad!r} must give no candidates"


def test_confirmed_name_is_a_str_or_none():
    address = ipaddress.ip_address("192.0.2.5")
    forward = {"app-one": [address], "app-one.net_default": [address]}
    assert names.confirmed_name(address, forward) == "app-one"
    assert names.confirmed_name(address, {"app-one": []}) is None
    assert names.confirmed_name(address, {}) is None


def test_confirmed_name_is_independent_of_input_order():
    address = ipaddress.ip_address("192.0.2.5")
    forward = {"bb": [address], "aa": [address]}
    reordered = dict(reversed(list(forward.items())))
    assert names.confirmed_name(address, forward) == "aa"
    assert names.confirmed_name(address, reordered) == "aa"


def test_module_imports_nothing_from_turbohaul():
    source = inspect.getsource(names)
    assert "import turbohaul" not in source
    assert "from turbohaul" not in source
