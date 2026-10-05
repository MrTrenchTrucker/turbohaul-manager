"""Fast Lane lint: every warning names rules by the position the Rules
table shows (rows numbered from 1), not by the 0-based list index.

Every lookup is a stub; nothing touches DNS or a socket.
"""
import dataclasses

from turbohaul.fastlane import lint_rules, normalize_address

PREFIX_TEXT = "docker-lease-sized prefix"


@dataclasses.dataclass
class _Rule:
    address: str | None = None
    container_name: str | None = None
    label: str = ""
    tag_ranks: dict = dataclasses.field(default_factory=dict)


def _resolver(table):
    def resolve_name(name):
        return frozenset(normalize_address(a) for a in table.get(name, ()))

    return resolve_name


def _prefix_warnings(rules, table):
    warnings = lint_rules(rules, resolve_name=_resolver(table))
    return warnings, [w for w in warnings if PREFIX_TEXT in w]


def _padding(count, start=1):
    # Harmless address rules: distinct documentation addresses, none in the
    # docker-style space used below, so they add no warning of their own.
    return [_Rule(address=f"192.0.2.{n}", label=f"pad-{n}") for n in range(start, start + count)]


class TestPrefixWarningUsesOneBasedRuleNumbers:
    def test_ipv4_address_rule_at_row_9_names_row_9_and_row_1(self):
        rules = [_Rule(container_name="app-one", label="App one")]
        rules += _padding(7)
        rules.append(_Rule(address="203.0.113.50", label="stale-lease"))
        assert len(rules) == 9

        warnings, prefix = _prefix_warnings(rules, {"app-one": ["203.0.113.7"]})

        assert len(prefix) == 1, warnings  # the branch was reached, once
        msg = prefix[0]
        assert "rule 9 (address '203.0.113.50')" in msg
        assert "rule 1's resolved container_name 'app-one'" in msg
        assert "rule index 8" not in msg
        assert "rule index 0" not in msg
        assert len(warnings) == 1  # nothing else fired for this list

    def test_ipv6_address_rule_at_row_9_names_row_9_and_row_1(self):
        rules = [_Rule(container_name="worker-two", label="Worker two")]
        rules += _padding(7)
        rules.append(_Rule(address="fd00::1:50", label="stale-lease"))

        warnings, prefix = _prefix_warnings(rules, {"worker-two": ["fd00::1:7"]})

        assert len(prefix) == 1, warnings
        msg = prefix[0]
        assert "rule 9 (address 'fd00::1:50')" in msg
        assert "rule 1's resolved container_name 'worker-two'" in msg
        assert "rule index 8" not in msg
        assert "rule index 0" not in msg

    def test_address_rule_first_and_named_rule_second_checks_both_numbers(self):
        # Row 1 is the address rule (index 0), row 2 the named rule (index 1):
        # a swap of the two numbers, or a 0-based index, shows up here.
        rules = [
            _Rule(address="203.0.113.50", label="stale-lease"),
            _Rule(container_name="app-one", label="App one"),
        ]

        warnings, prefix = _prefix_warnings(rules, {"app-one": ["203.0.113.7"]})

        assert len(prefix) == 1, warnings
        msg = prefix[0]
        assert "rule 1 (address '203.0.113.50')" in msg
        assert "rule 2's resolved container_name 'app-one'" in msg
        assert "rule index 0" not in msg
        assert "rule index 1" not in msg

    def test_wording_after_the_numbers_is_unchanged(self):
        rules = [
            _Rule(container_name="app-one"),
            _Rule(address="203.0.113.50"),
        ]
        _, prefix = _prefix_warnings(rules, {"app-one": ["203.0.113.7"]})
        assert len(prefix) == 1
        assert prefix[0] == (
            "rule 2 (address '203.0.113.50') shares a docker-lease-sized "
            "prefix with rule 1's resolved container_name 'app-one' -- this "
            "looks like a docker-assigned address that can roll on restart; "
            "prefer container_name for this rule. If this is a deliberate "
            "off-network address that happens to share a prefix by "
            "coincidence, that intent is unverifiable by this check."
        )

    def test_off_network_address_stays_silent(self):
        rules = [_Rule(container_name="app-one")]
        rules += _padding(7)
        rules.append(_Rule(address="198.51.100.50", label="static-fallback"))

        warnings, prefix = _prefix_warnings(rules, {"app-one": ["203.0.113.7"]})

        assert prefix == []
        assert warnings == []

    def test_one_warning_per_address_rule(self):
        rules = [
            _Rule(container_name="app-one"),
            _Rule(address="203.0.113.50"),
            _Rule(address="203.0.113.60"),
            _Rule(address="198.51.100.50"),
        ]
        warnings, prefix = _prefix_warnings(rules, {"app-one": ["203.0.113.7"]})

        assert len(prefix) == 2, warnings
        assert "rule 2 (address '203.0.113.50')" in prefix[0]
        assert "rule 3 (address '203.0.113.60')" in prefix[1]
        assert len(warnings) == 2


class TestEveryLintMessageNamesRulesByTablePosition:
    """Each message names rules by the 1-based row the Rules table shows. In
    every case the offending rule sits beyond row 1, so a 0-based number
    would read differently; each test asserts the whole message text."""

    def test_duplicate_container_name_names_both_rows(self):
        rules = _padding(2)
        rules.append(_Rule(container_name="app-one", label="first"))  # row 3
        rules += _padding(5, start=3)
        rules.append(_Rule(container_name="app-one", label="second"))  # row 9
        assert len(rules) == 9

        warnings = lint_rules(rules)

        assert warnings == [
            "duplicate container_name: rule 3 and rule 9 both name "
            "'app-one' -- the second can never match, since the first "
            "already claims every address that name resolves to. Config "
            "error, not auto-resolved."
        ]
        assert not any("rule index" in w for w in warnings)

    def test_duplicate_address_names_both_rows(self):
        rules = [
            _Rule(address="192.0.2.51", label="other"),  # row 1
            _Rule(address="192.0.2.50", label="first"),  # row 2
        ]
        rules += [_Rule(address=f"198.51.100.{n}") for n in range(1, 7)]  # rows 3-8
        rules.append(_Rule(address="::ffff:192.0.2.50", label="second"))  # row 9
        assert len(rules) == 9

        warnings = lint_rules(rules)

        assert warnings == [
            "duplicate address: rule 2 ('192.0.2.50') and rule 9 "
            "('::ffff:192.0.2.50') both target the same host -- config "
            "error, not auto-resolved."
        ]
        assert not any("rule index" in w for w in warnings)

    def test_never_observed_address_names_its_row(self):
        rules = _padding(8)
        rules.append(_Rule(address="198.51.100.9", label="never-seen"))  # row 9
        census = {f"192.0.2.{n}": {} for n in range(1, 8 + 1)}

        warnings = lint_rules(rules, census=census)

        assert warnings == [
            "rule 9 (address '198.51.100.9') has never been observed in "
            "the discovered-address census."
        ]
        assert not any("rule index" in w for w in warnings)

    def test_name_resolving_to_nothing_names_its_row(self):
        rules = _padding(8)
        rules.append(_Rule(container_name="worker-two", label="typo"))  # row 9

        warnings = lint_rules(rules, resolve_name=_resolver({}))

        assert warnings == [
            "rule 9 names container 'worker-two', which currently "
            "resolves to NO addresses -- that rule matches nothing and the "
            "client has no Fast Lane priority. Check the container name "
            "and whether it is running."
        ]
        assert not any("rule index" in w for w in warnings)

    def test_address_colliding_with_a_resolved_name_names_both_rows(self):
        rules = _padding(2)
        rules.append(_Rule(container_name="app-one", label="named"))  # row 3
        rules += _padding(5, start=3)
        rules.append(_Rule(address="203.0.113.7", label="by-address"))  # row 9
        assert len(rules) == 9

        warnings = lint_rules(
            rules, resolve_name=_resolver({"app-one": ["203.0.113.7"]})
        )

        assert warnings == [
            "rule 9 (address '203.0.113.7') collides with rule 3's "
            "container_name 'app-one' -- the same client may be listed "
            "twice, once by address and once by name."
        ]
        assert not any("rule index" in w for w in warnings)
