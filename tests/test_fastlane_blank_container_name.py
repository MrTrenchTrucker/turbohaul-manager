"""A BLANK `container_name` must be refused by the PUT, not persisted.

The defect this pins is an asymmetry between two readers of the same field:

  - config.py `_exactly_one_of_address_or_container_name` tested `is not None`,
    so `container_name: ""` counted as SET and the rule VALIDATED.
  - config_put.py `_strip_null_container_names` tests truthiness, so the same
    `""` counted as UNSET and the key was DELETED on the way to disk.

The persisted rule therefore carried NEITHER identity field. On the next boot
`FastLaneConfig(**persisted)` raises, `_salvage_fastlane_rules` refuses it (the
salvage arm needs a bad ADDRESS to coerce, and there is none), and the caller's
`continue` skips the whole section assignment -- so every VALID sibling rule is
discarded too and Fast Lane boots disabled.

`""` is `is not None` (SET at validation) and falsy (UNSET at persistence). That
one asymmetry is the entire bug.

Fixed on the validation side rather than the persistence side on purpose. The
strip exists for rollback safety -- it removes the `container_name: null`
placeholder so a rolled-back older build, which rejects unknown keys under
extra="forbid", can still load the section. Narrowing the strip to `is None`
would let `""` reach disk as a real key and defeat exactly that guarantee. The
right place is the door: refuse the blank identity in the PUT. That also makes
config.py agree with this field's other two readers -- compile_fastlane and
lint_rules (fastlane.py) both already test truthiness.

Whitespace-only is refused for the same reason: it is equally not an identity,
it produces the same silently-inert rule, and the frontend adder already
`.trim()`s before it will emit at all.
"""
import pytest
from pydantic import ValidationError

from turbohaul.config import FastLaneConfig, FastLaneRule


class TestABlankContainerNameIsRefusedAtTheDoor:
    def test_an_empty_string_container_name_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(container_name="", label="typed-nothing")
        assert "blank" in str(ei.value).lower(), str(ei.value)

    def test_a_whitespace_only_container_name_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(container_name="   ", label="typed-spaces")
        assert "blank" in str(ei.value).lower(), str(ei.value)

    def test_the_rejection_message_names_the_actual_problem(self):
        """A rule that sets container_name to "" has not 'set NEITHER field' from
        the operator's point of view -- they set one and left it empty. The
        message has to say which, or the operator reads it as a bug in us."""
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(container_name="", label="typed-nothing")
        msg = str(ei.value)
        assert "container_name" in msg, msg
        assert "typed-nothing" in msg, msg


class TestTheControlsThatKeepTheAboveHonest:
    """Each of these passes BEFORE and AFTER the fix. If one of them ever fails,
    the fix has over-reached and started rejecting legitimate configuration."""

    def test_CONTROL_a_real_container_name_is_still_accepted(self):
        r = FastLaneRule(container_name="openwebui", label="OWUI")
        assert r.container_name == "openwebui"
        assert r.address is None

    def test_CONTROL_an_address_only_rule_is_still_accepted(self):
        r = FastLaneRule(address="10.0.0.5", label="gw")
        assert r.address == "10.0.0.5"
        assert r.container_name is None

    def test_CONTROL_setting_BOTH_is_still_rejected(self):
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(address="10.0.0.5", container_name="openwebui")
        assert "BOTH" in str(ei.value), str(ei.value)

    def test_CONTROL_setting_NEITHER_is_still_rejected_with_its_own_message(self):
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(label="empty")
        assert "NEITHER" in str(ei.value), str(ei.value)

    def test_CONTROL_a_blank_container_name_ALONGSIDE_an_address_is_still_BOTH(self):
        """Deliberately NOT loosened. A naive version of this fix made the
        both-set check blank-aware too, which would have started ACCEPTING this
        shape -- a loosening the defect does not require. The fix touches the
        NEITHER arm only, so this stays rejected exactly as it is today.

        Safe to leave strict: the frontend sends `container_name: null` on
        address rules, never `""`, and validation runs before the strip, so no
        real caller reaches this branch."""
        with pytest.raises(ValidationError) as ei:
            FastLaneRule(address="10.0.0.5", container_name="", label="gw")
        assert "BOTH" in str(ei.value), str(ei.value)


class TestTheBootBrickingChainIsClosedEndToEnd:
    def test_a_config_carrying_a_blank_rule_cannot_be_built_at_all(self):
        """The whole point: this never reaches _strip_null_container_names, so it
        never reaches disk, so there is nothing for the next boot to choke on."""
        with pytest.raises(ValidationError):
            FastLaneConfig(enabled=True, rules=[
                {"container_name": "", "label": "typed-nothing"},
                {"address": "10.0.0.5", "label": "good-sibling-A"},
            ])

    def test_CONTROL_the_same_config_with_a_REAL_name_builds_and_keeps_both_rules(self):
        """Proves the assertion above fails for the blank field specifically, and
        not because this two-rule shape is rejected for some unrelated reason."""
        cfg = FastLaneConfig(enabled=True, rules=[
            {"container_name": "openwebui", "label": "typed-something"},
            {"address": "10.0.0.5", "label": "good-sibling-A"},
        ])
        assert len(cfg.rules) == 2
        assert cfg.rules[0].container_name == "openwebui"
        assert cfg.rules[1].address == "10.0.0.5"

    def test_the_persisted_shape_that_used_to_brick_boot_still_fails_loudly(self):
        """Pins the OTHER half of the chain independently of the fix: if a rule
        with neither field is already sitting in someone's runtime_config.yaml
        from before this fix, loading it must still raise rather than silently
        produce a ruleless config. This is what makes the boot log honest."""
        with pytest.raises(ValidationError) as ei:
            FastLaneConfig(enabled=True, rules=[{"label": "typed-nothing"}])
        assert "NEITHER" in str(ei.value), str(ei.value)


class TestTheRollbackStripIsUnharmed:
    """Rollback regression pin. The fix above deliberately did NOT touch
    _strip_null_container_names; these assert that its documented purpose still
    holds, so a reviewer can see the rollback guarantee was preserved rather
    than assumed."""

    def test_a_null_container_name_is_still_stripped_for_rollback_safety(self):
        from turbohaul.api.config_put import _strip_null_container_names

        out = _strip_null_container_names(
            {"fastlane": {"enabled": True, "rules": [
                {"address": "10.0.0.2", "label": "a", "container_name": None},
            ]}}
        )
        assert "container_name" not in out["fastlane"]["rules"][0], out

    def test_a_real_container_name_still_survives_the_strip(self):
        from turbohaul.api.config_put import _strip_null_container_names

        out = _strip_null_container_names(
            {"fastlane": {"enabled": True, "rules": [
                {"container_name": "gateway-svc", "label": "Gateway"},
            ]}}
        )
        assert out["fastlane"]["rules"][0]["container_name"] == "gateway-svc"
