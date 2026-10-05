"""Fast Lane matcher tests — normalize_address, resolve_rank,
compile_fastlane, match_fastlane, lint_rules.

Every check here has an explicit failing arm baked into the test itself (a
wrong implementation choice is asserted against directly), not just a
"does it run" smoke check.
"""
import dataclasses
import itertools

import pytest

from turbohaul import fastlane
from turbohaul.fastlane import (
    TAG_UNCLASSIFIED,
    UNRANKED,
    CompiledRule,
    compile_fastlane,
    lint_rules,
    match_fastlane,
    normalize_address,
    resolve_rank,
)
from turbohaul.kv_classify import _class_from_label


@dataclasses.dataclass
class _Rule:
    """Minimal duck-typed stand-in for the config-layer FastLaneRule model —
    compile_fastlane only reads .address/.container_name/.label/.tag_ranks."""

    address: str | None = None
    container_name: str | None = None
    label: str = ""
    tag_ranks: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------

class TestNormalizeAddress:
    def test_ipv4_mapped_ipv6_matches_plain_dotted_quad(self):
        # This is the exact form the manager observes for anything arriving
        # through the published port. A naive string-compare
        # implementation FAILS this (the strings are textually different).
        assert normalize_address("::ffff:192.0.2.10") == normalize_address("192.0.2.10")

    def test_plain_dotted_quad_matches_ipv4_mapped_ipv6_reverse(self):
        assert normalize_address("192.0.2.10") == normalize_address("::ffff:192.0.2.10")

    def test_zone_suffix_stripped_before_compare(self):
        # ipaddress preserves scope_id in equality by default — a naive
        # ip_address(raw) WITHOUT stripping %zone first FAILS this (verified
        # empirically: `ip_address('fe80::1%eth0') == ip_address('fe80::1')`
        # is False in the stdlib).
        assert normalize_address("fe80::1%eth0") == normalize_address("fe80::1")

    def test_cidr_rejected(self):
        with pytest.raises(ValueError):
            normalize_address("192.0.2.0/24")

    def test_prefix_rejected(self):
        with pytest.raises(ValueError):
            normalize_address("10.0.0.0/8")

    def test_unparseable_rejected(self):
        with pytest.raises(ValueError):
            normalize_address("not-an-address")


# ---------------------------------------------------------------------------
# resolve_rank — 16-boolean table + POLICIES roles + junk role
# ---------------------------------------------------------------------------

_ALL_RANKED_TAGS = {"main": 1, "curator": 2, "compression": 3, "sub_agent": 4}
BOOL_FIELDS = ("is_curator", "is_compression", "is_sub_agent", "is_main")


def _expected_tag(client_meta: dict) -> str:
    cls = _class_from_label(client_meta)
    return fastlane._TAG_BY_CLASS.get(cls, TAG_UNCLASSIFIED)


class TestResolveRankBooleanTable:
    @pytest.mark.parametrize(
        "combo", list(itertools.product([True, False], repeat=4))
    )
    def test_all_16_boolean_combinations(self, combo):
        client_meta = dict(zip(BOOL_FIELDS, combo))
        expected_tag = _expected_tag(client_meta)
        tag, rank = resolve_rank(client_meta, dict(_ALL_RANKED_TAGS))
        assert tag == expected_tag
        if expected_tag in _ALL_RANKED_TAGS:
            assert rank == _ALL_RANKED_TAGS[expected_tag]
        else:
            assert rank == UNRANKED

    def test_none_client_meta(self):
        tag, rank = resolve_rank(None, dict(_ALL_RANKED_TAGS))
        assert tag == TAG_UNCLASSIFIED
        assert rank == UNRANKED

    def test_curator_carrying_sub_agent_resolves_curator_not_sub_agent(self):
        # THE canonical trap: a curator carries BOTH is_curator
        # AND is_sub_agent. A raw-boolean / wrong-priority implementation
        # hands this traffic to the sub_agent rank instead — hardcoded
        # expectation here, independent of _expected_tag's own use of
        # _class_from_label, so a priority-order regression in EITHER place
        # cannot silently agree with itself.
        client_meta = {"is_curator": True, "is_sub_agent": True}
        tag, rank = resolve_rank(client_meta, {"curator": 1, "sub_agent": 2})
        assert tag == "curator"
        assert rank == 1

    @pytest.mark.parametrize("rank_arrangement", [
        {"main": 1, "curator": 2, "compression": 3, "sub_agent": 4},
        {"main": 5, "curator": 4, "compression": 3, "sub_agent": 1},
        {"main": 3, "curator": 1, "compression": 5, "sub_agent": 2},
    ])
    def test_curator_wins_under_every_rank_arrangement(self, rank_arrangement):
        client_meta = {"is_curator": True, "is_sub_agent": True}
        tag, rank = resolve_rank(client_meta, rank_arrangement)
        assert tag == "curator"
        assert rank == rank_arrangement["curator"]

    @pytest.mark.parametrize("role,expected_tag", [
        ("main", "main"),
        ("user-message", TAG_UNCLASSIFIED),
        ("sub-agent", "sub_agent"),
        ("curator", "curator"),
        ("compression", "compression"),
    ])
    def test_policies_role_fallback(self, role, expected_tag):
        client_meta = {"role": role}
        tag, rank = resolve_rank(client_meta, dict(_ALL_RANKED_TAGS))
        assert tag == expected_tag

    def test_junk_role_never_raises_folds_to_unclassified(self):
        # role is caller-controlled and unvalidated. A .get()-free
        # implementation (a raw dict index) raises KeyError here instead.
        client_meta = {"role": "whatever-a-client-feels-like-sending"}
        tag, rank = resolve_rank(client_meta, dict(_ALL_RANKED_TAGS))
        assert tag == TAG_UNCLASSIFIED
        assert rank == UNRANKED

    def test_all_unset_tag_table_yields_identical_rank_for_all_classes(self):
        for role in ("main", "curator", "compression", "sub-agent", "user-message"):
            tag, rank = resolve_rank({"role": role}, {})
            assert rank == UNRANKED  # first-come-first-served parity


# ---------------------------------------------------------------------------
# compile_fastlane / match_fastlane
# ---------------------------------------------------------------------------

class TestCompileAndMatch:
    def test_no_rules_no_match(self):
        assert match_fastlane([], "1.2.3.4", {}) is None

    def test_no_ip_no_match(self):
        compiled = compile_fastlane([_Rule(address="1.2.3.4")])
        assert match_fastlane(compiled, None, {}) is None
        assert match_fastlane(compiled, "", {}) is None

    def test_exact_match_returns_rule_index_and_tag(self):
        rules = [
            _Rule(address="1.2.3.4", label="ops-box", tag_ranks={"main": 1}),
        ]
        compiled = compile_fastlane(rules)
        result = match_fastlane(compiled, "1.2.3.4", {"is_main": True})
        assert result is not None
        assert result.rule_index == 0
        assert result.raw_address == "1.2.3.4"
        assert result.label == "ops-box"
        assert result.effective_tag == "main"
        assert result.rank == 1

    def test_unmatched_address_returns_none(self):
        compiled = compile_fastlane([_Rule(address="1.2.3.4")])
        assert match_fastlane(compiled, "9.9.9.9", {}) is None

    def test_mapped_and_plain_forms_both_match_same_rule(self):
        compiled = compile_fastlane([_Rule(address="192.0.2.10", tag_ranks={"main": 1})])
        a = match_fastlane(compiled, "::ffff:192.0.2.10", {"is_main": True})
        b = match_fastlane(compiled, "192.0.2.10", {"is_main": True})
        assert a is not None and b is not None
        assert a.rule_index == b.rule_index == 0

    def test_effective_tag_never_written_into_client_meta(self):
        # The fallback is matcher-local, display/log only. Assert the
        # INPUT dict is byte-identical after matching (pure-matcher level;
        # the manager-stamp-site level lives in test_queue_fastlane.py per
        # a separate manager-stamp-site level check).
        import json
        client_meta = {"is_main": True, "role": "main"}
        before = json.dumps(client_meta, sort_keys=True, default=str)
        rules = [_Rule(address="1.2.3.4", tag_ranks={"main": 1})]
        compiled = compile_fastlane(rules)
        match_fastlane(compiled, "1.2.3.4", client_meta)
        after = json.dumps(client_meta, sort_keys=True, default=str)
        assert before == after

    def test_invalid_address_in_rule_list_skipped_not_raised(self):
        # Bad input reaching the hot admission path degrades, it doesn't crash.
        rules = [
            _Rule(address="10.0.0.0/8"),  # CIDR — invalid, skipped
            _Rule(address="1.2.3.4", tag_ranks={"main": 1}),
        ]
        compiled = compile_fastlane(rules)
        assert len(compiled) == 1
        assert compiled[0].address == normalize_address("1.2.3.4")

    def test_priority_is_list_index_not_reordered(self):
        rules = [_Rule(address="1.1.1.1"), _Rule(address="2.2.2.2"), _Rule(address="3.3.3.3")]
        compiled = compile_fastlane(rules)
        assert [c.index for c in compiled] == [0, 1, 2]
        assert [c.raw_address for c in compiled] == ["1.1.1.1", "2.2.2.2", "3.3.3.3"]


# ---------------------------------------------------------------------------
# container_name identity
# ---------------------------------------------------------------------------

class TestContainerNameIdentity:
    """A rule may now identify its client by container_name (on-network),
    resolved forward-only through a `resolve_name` callable passed into
    compile_fastlane — fastlane_resolve.FastLaneNameResolver.addresses_for
    is the real thing this stands in for; these tests use a hand-rolled
    stub so the matcher-level behavior is provable without any real I/O."""

    def test_container_name_rule_survives_an_address_change(self):
        # RED FIRST against today's address-only matcher: pre-change,
        # compile_fastlane(rules, resolve_name=...) TypeErrors on the
        # unexpected keyword, and neither CompiledRule nor _Rule has any
        # container_name/match_addresses concept — there is no way for a
        # rule's identity to survive an address change today.
        calls = {"n": 0}
        first_addr = normalize_address("192.0.2.5")
        second_addr = normalize_address("192.0.2.9")

        def resolve_name(name):
            calls["n"] += 1
            return frozenset([first_addr if calls["n"] == 1 else second_addr])

        rules = [_Rule(container_name="gateway-svc", label="gateway", tag_ranks={"main": 1})]

        compiled_1 = compile_fastlane(rules, resolve_name=resolve_name)
        m1 = match_fastlane(compiled_1, "192.0.2.5", {"is_main": True})
        assert m1 is not None
        assert m1.rule_index == 0
        assert m1.label == "gateway"
        assert m1.matched_by == "name"

        # The container restarted; docker handed it a new address on the
        # (unchanged) name — a real recompile would happen on the resolver's
        # generation bump, simulated here by just calling compile_fastlane
        # again against the same resolve_name stub, which now answers with
        # the new address.
        compiled_2 = compile_fastlane(rules, resolve_name=resolve_name)
        assert match_fastlane(compiled_2, "192.0.2.5", {"is_main": True}) is None
        m2 = match_fastlane(compiled_2, "192.0.2.9", {"is_main": True})
        assert m2 is not None
        assert m2.rule_index == 0
        assert m2.label == "gateway"
        assert m2.matched_by == "name"

    def test_off_network_address_rule_matches_without_invoking_the_resolver(self):
        calls = {"n": 0}

        def resolve_name(name):
            calls["n"] += 1
            return frozenset()

        rules = [_Rule(address="203.0.113.9", label="vpn-box", tag_ranks={"main": 1})]
        compiled = compile_fastlane(rules, resolve_name=resolve_name)
        result = match_fastlane(compiled, "203.0.113.9", {"is_main": True})
        assert result is not None
        assert result.matched_by == "address"
        assert calls["n"] == 0  # the resolver is never invoked for a pure address rule

    def test_unresolved_name_yields_unlisted_and_does_not_leak_into_the_next_rule(self):
        # ⛔ MANDATORY NEGATIVE ARM: a name that resolves to nothing must
        # degrade to unlisted, never promote a request into a DIFFERENT
        # rule's lane. Fail-closed, proven here, not merely asserted by the
        # module docstring.
        def resolve_name(name):
            return frozenset()  # "ghost" — never resolves

        rules = [
            _Rule(container_name="ghost", label="ghost-rule", tag_ranks={"main": 1}),
            _Rule(address="9.9.9.9", label="real-rule", tag_ranks={"curator": 1}),
        ]
        compiled = compile_fastlane(rules, resolve_name=resolve_name)
        assert compiled[0].match_addresses == frozenset()

        result = match_fastlane(compiled, "9.9.9.9", {"is_curator": True})
        assert result is not None
        assert result.rule_index == 1  # falls through past ghost to the real rule
        assert result.label == "real-rule"

        assert match_fastlane(compiled, "1.2.3.4", {}) is None  # matches neither rule


# ---------------------------------------------------------------------------
# lint_rules — report-only, never auto-fixes
# ---------------------------------------------------------------------------

class TestLintRules:
    def test_duplicate_address_reported(self):
        rules = [_Rule(address="1.2.3.4"), _Rule(address="1.2.3.4")]
        warnings = lint_rules(rules)
        assert len(warnings) == 1
        assert "duplicate" in warnings[0]

    def test_no_duplicates_no_warning(self):
        rules = [_Rule(address="1.2.3.4"), _Rule(address="5.6.7.8")]
        assert lint_rules(rules) == []

    def test_address_never_seen_in_census_reported(self):
        rules = [_Rule(address="1.2.3.4")]
        warnings = lint_rules(rules, census={"9.9.9.9": {}})
        assert any("never been observed" in w for w in warnings)

    def test_address_seen_in_census_no_warning(self):
        rules = [_Rule(address="1.2.3.4")]
        warnings = lint_rules(rules, census={"1.2.3.4": {}})
        assert not any("never been observed" in w for w in warnings)

    def test_lint_never_mutates_input(self):
        rules = [_Rule(address="1.2.3.4"), _Rule(address="1.2.3.4")]
        original_order = [r.address for r in rules]
        lint_rules(rules)
        assert [r.address for r in rules] == original_order  # no reorder, no auto-fix

    def test_normalized_duplicate_reported(self):
        # match_fastlane normalizes both sides
        # (this module's normalize_address) before comparing, so two
        # textually different rules that resolve to the same host are the
        # exact duplicate class lint_rules exists to catch. A raw string
        # compare passes right over this (this is what a naive
        # implementation misses -- pinned directly, not just implied by the
        # matcher-level test above).
        rules = [_Rule(address="192.0.2.10"), _Rule(address="::ffff:192.0.2.10")]
        warnings = lint_rules(rules)
        assert len(warnings) == 1
        assert "duplicate" in warnings[0]

    def test_normalized_duplicate_message_names_both_raw_forms(self):
        # The two raw strings are textually different -- the warning must
        # show BOTH exactly as the operator typed them, or a human reading
        # the log/response has no way to find the second one.
        rules = [_Rule(address="192.0.2.10"), _Rule(address="::ffff:192.0.2.10")]
        warnings = lint_rules(rules)
        assert "192.0.2.10" in warnings[0]
        assert "::ffff:192.0.2.10" in warnings[0]

    def test_unparseable_address_falls_back_to_raw_compare_no_crash(self):
        # normalize_address raises ValueError on anything that isn't a
        # single exact host (CIDR, unparseable). lint_rules must degrade,
        # not crash, on bad input that slipped past compile_fastlane's own
        # skip-on-invalid handling -- mirrors that function's own posture.
        rules = [_Rule(address="10.0.0.0/8"), _Rule(address="10.0.0.0/8")]
        warnings = lint_rules(rules)
        assert len(warnings) == 1
        assert "duplicate" in warnings[0]


# ---------------------------------------------------------------------------
# lint_rules -- mislabel + collision checks
# ---------------------------------------------------------------------------

class TestLintMislabelAndCollision:
    """Two checks that cost no I/O: (b) label-vs-container_name
    disagreement, and (c) an address-only rule colliding with a DIFFERENT
    rule's resolved container_name (which reuses the existing forward-only
    cache and adds no new lookups). Both deliberately leave a stated blind
    spot -- see lint_rules' own docstring -- rather than reintroduce a
    reverse-DNS lookup on the admission path."""

    # (b) label-vs-container_name is GONE, and so are its three tests.
    # In practice it warned only falsely on realistic rule tables: every
    # warning was a false positive, none was true. `label` is free text ("operator's own note"
    # per config.py) and the schema asserts no relationship to the container
    # name, so the check had nothing real to be right about. What replaces it
    # is the control below -- the property that actually matters is that lint
    # stays QUIET on a correct table, because a channel that cries wolf is a
    # channel an operator stops reading, and Contract 6's genuine drift
    # warning shares it.

    def test_two_rules_naming_the_SAME_container_warn(self):
        """Coverage regression: duplicate container_name rules must warn.

        Originally there was one way to name a host and the duplicate
        detector covered it. container_name was added later without teaching the
        detector about it, so a pasted-twice name rule was DEAD and silent:
        the first rule claims every address that name resolves to, the
        ordered scan never reaches the second, and the operator sees it in
        the table with ranks configured and believes it applies.

        Same failure shape as a duplicate address, so it gets the same
        message shape."""
        rules = [
            _Rule(container_name="gateway-svc", label="Gateway"),
            _Rule(container_name="gateway-svc", label="Gateway duplicate"),
        ]
        warnings = lint_rules(rules)
        assert any("duplicate container_name" in w for w in warnings), warnings
        joined = " ".join(warnings)
        assert "rule 1 and rule 2" in joined, (
            "the warning must name BOTH rules -- an operator cannot act on "
            f"'there is a duplicate somewhere': {warnings}"
        )

    def test_two_rules_naming_DIFFERENT_containers_stay_silent(self):
        """CONTROL. Without it the check above is satisfied by a warning that
        fires on every name rule regardless of whether anything is duplicated
        -- which is precisely how arm (b) died."""
        rules = [
            _Rule(container_name="gateway-svc", label="Gateway"),
            _Rule(container_name="agent-svc", label="Agent service"),
        ]
        assert not any("duplicate container_name" in w for w in lint_rules(rules)), \
            lint_rules(rules)

    def test_a_correct_five_rule_table_produces_NO_warnings(self):
        """THE CONTROL. lint must be SILENT on a table that is correct.

        An illustrative five-rule table, not any particular deployment's:
        the ORDER HERE IS ARBITRARY and carries no meaning. (Order is the
        priority in a real table, since rule_index is the primary sort key,
        so nothing about a real one belongs in a test fixture.) What matters
        for this arm is only that the rules are well-formed and distinct --
        one entry per container, no duplicate names, no duplicate addresses.

        Every rule here tripped the removed (b) check. If a future lint arm
        fires on a table shaped like this, that arm is wrong."""
        rules = [
            _Rule(container_name="agent-svc", label="Agent service"),
            _Rule(container_name="gateway-svc", label="Gateway"),
            _Rule(container_name="web-ui", label="Web UI"),
            _Rule(container_name="advisor", label="Advisor"),
            _Rule(container_name="advisor-large", label="Advisor (large)"),
        ]
        assert lint_rules(rules) == [], lint_rules(rules)

    def test_the_control_is_not_vacuous_lint_CAN_still_warn(self):
        """A silence test proves nothing unless the same call can produce
        noise on the same path, with no census and no resolver.

        ⭐ THIS ARM EARNED ITS KEEP. It was first written with two rules
        naming the same CONTAINER and it FAILED -- lint was silent on that,
        because the duplicate detector predated container_name and was never
        extended it. The noise arm of a silence test found a real coverage
        regression. It is pointed back at the duplicate-NAME case now that
        the case is covered, so the control follows the code forward instead
        of being pinned to the weaker input that was available before."""
        rules = [
            _Rule(container_name="gateway-svc", label="Gateway"),
            _Rule(container_name="gateway-svc", label="Gateway duplicate"),
        ]
        assert lint_rules(rules) != [], (
            "lint_rules went silent on a duplicate container_name -- the "
            "locked-table control above is therefore unfalsifiable"
        )

    # --- Contract 6: the container_name acceptance test -----------------
    #
    # "a rule naming a container that resolves to nothing must WARN, not go
    # silent." A typo'd or not-yet-started container silently loses its Fast
    # Lane priority otherwise -- the exact silent-drift class this check is
    # for. This was absent, and a sibling test pinned the silence.

    def test_name_resolving_to_nothing_WARNS(self):
        rules = [_Rule(container_name="gateway-svc", label="gateway service")]
        warnings = lint_rules(rules, census={}, resolve_name=lambda n: frozenset())
        assert any("resolve" in w.lower() for w in warnings), (
            f"a container_name that resolves to NOTHING must warn; got {warnings!r}"
        )
        # the warning has to name the container, or an operator cannot act on it
        assert any("gateway-svc" in w for w in warnings)

    def test_name_that_DOES_resolve_stays_silent(self):
        # Control: without this, a checker that always fired would pass the
        # arm above for the wrong reason.
        rules = [_Rule(container_name="gateway-svc", label="gateway service")]
        warnings = lint_rules(
            rules, census={},
            resolve_name=lambda n: frozenset({normalize_address("192.0.2.5")}),
        )
        assert not any("resolve" in w.lower() for w in warnings), warnings

    def test_no_resolver_supplied_cannot_and_must_not_judge_resolution(self):
        # Second control, and it guards a REAL false-positive: with no
        # resolver we have no evidence either way, so claiming the name is
        # unresolvable would be an unfounded accusation on every boot.
        rules = [_Rule(container_name="gateway-svc", label="gateway service")]
        warnings = lint_rules(rules, census={})
        assert not any("resolve" in w.lower() for w in warnings), warnings

    def test_address_rule_colliding_with_a_name_rules_resolved_set_warns(self):
        collision_addr = normalize_address("192.0.2.7")

        def resolve_name(name):
            return frozenset({collision_addr})

        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)", tag_ranks={"main": 1}),
            _Rule(address="192.0.2.7", label="stale-entry", tag_ranks={"curator": 1}),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert any("collides with rule 1's container_name" in w for w in warnings)

    def test_no_collision_when_addresses_are_genuinely_disjoint(self):
        # Control for the arm above: a genuinely unrelated address must not
        # warn, or (c) is a constant too.
        def resolve_name(name):
            return frozenset({normalize_address("192.0.2.7")})

        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)"),
            _Rule(address="9.9.9.9", label="unrelated"),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert not any("collides with" in w for w in warnings)

    def test_collision_check_is_silent_without_a_resolver_supplied(self):
        # Back-compat / boot-time structural-only call site: omitting
        # resolve_name must not crash and must not attempt check (c) at all.
        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)"),
            _Rule(address="192.0.2.7", label="stale-entry"),
        ]
        warnings = lint_rules(rules)  # no resolve_name
        assert not any("collides with" in w for w in warnings)

    def test_address_sharing_a_docker_lease_prefix_warns_d_prime(self):
        # Same /24 as a resolved container address, but NOT the exact same
        # address (that would be (c)'s stronger exact-collision case).
        def resolve_name(name):
            return frozenset({normalize_address("192.0.2.7")})

        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)"),
            _Rule(address="192.0.2.42", label="probably-a-docker-lease"),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert any("docker-lease-sized prefix" in w for w in warnings)

    def test_exact_collision_does_not_also_fire_d_prime(self):
        # (c) already gives the stronger exact-address warning; (d') must
        # not pile a second, weaker warning on the exact same rule.
        def resolve_name(name):
            return frozenset({normalize_address("192.0.2.7")})

        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)"),
            _Rule(address="192.0.2.7", label="stale-entry"),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert any("collides with" in w for w in warnings)
        assert not any("docker-lease-sized prefix" in w for w in warnings)

    def test_off_network_address_stays_silent_under_d_prime(self):
        # ⭐ REQUIREMENT, not just a control (operator's own off-network
        # fallback): a LAN/VPN address numerically shares no
        # prefix with anything docker resolves, so it must stay silent.
        def resolve_name(name):
            return frozenset({normalize_address("192.0.2.7")})

        rules = [
            _Rule(container_name="advisor-large", label="Advisor (large)"),
            _Rule(address="203.0.113.9", label="vpn-box"),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert warnings == []

    def test_d_prime_is_inert_on_a_fully_address_keyed_table(self):
        # Disclosed blind spot #2, proven directly: with no container_name
        # rule at all, there is nothing resolved to compare against, so
        # (d') can only ever be silent -- even for two address rules that
        # are numerically close to each other.
        def resolve_name(name):
            return frozenset()

        rules = [
            _Rule(address="192.0.2.7", label="a"),
            _Rule(address="192.0.2.42", label="b"),
        ]
        warnings = lint_rules(rules, resolve_name=resolve_name)
        assert warnings == []
