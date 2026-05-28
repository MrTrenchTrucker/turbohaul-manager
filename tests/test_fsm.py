"""Tests for state machine transitions (v0.2 §6)."""
import pytest

from turbohaul.fsm import (
    LEGAL_TRANSITIONS,
    InvalidTransition,
    can_transition,
    is_request_in_flight,
    is_terminal,
    is_warm_state,
    legal_targets,
    transition,
)
from turbohaul.slot import Slot, SlotState


class TestLegalTransitions:
    def test_received_to_staged(self):
        s = Slot.new("m")
        assert s.state == SlotState.RECEIVED
        transition(s, SlotState.STAGED)
        assert s.state == SlotState.STAGED

    def test_received_to_accept_buffer(self):
        s = Slot.new("m")
        transition(s, SlotState.ACCEPT_BUFFER)
        assert s.state == SlotState.ACCEPT_BUFFER

    def test_full_happy_path(self):
        """RECEIVED → STAGED → LOADING → ACTIVE → GRACE → POPPED → IDLE_HOT → COLD."""
        s = Slot.new("m")
        for target in [
            SlotState.STAGED,
            SlotState.LOADING,
            SlotState.ACTIVE,
            SlotState.GRACE,
            SlotState.POPPED,
            SlotState.IDLE_HOT,
            SlotState.COLD,
        ]:
            transition(s, target)
            assert s.state == target

    def test_loading_fail_retry(self):
        """LOADING → LOADING_FAIL → STAGED (retry)."""
        s = Slot.new("m")
        transition(s, SlotState.STAGED)
        transition(s, SlotState.LOADING)
        transition(s, SlotState.LOADING_FAIL)
        transition(s, SlotState.STAGED)
        assert s.state == SlotState.STAGED

    def test_loading_fail_exhaust_to_popped(self):
        """LOADING_FAIL → POPPED on retry exhaust."""
        s = Slot.new("m")
        transition(s, SlotState.STAGED)
        transition(s, SlotState.LOADING)
        transition(s, SlotState.LOADING_FAIL)
        transition(s, SlotState.POPPED)
        assert s.state == SlotState.POPPED

    def test_grace_busy_followup(self):
        """ACTIVE → GRACE → GRACE_BUSY → GRACE → POPPED."""
        s = Slot.new("m")
        for target in [
            SlotState.STAGED,
            SlotState.LOADING,
            SlotState.ACTIVE,
            SlotState.GRACE,
            SlotState.GRACE_BUSY,
            SlotState.GRACE,
            SlotState.POPPED,
        ]:
            transition(s, target)

    def test_active_match_mid_stream(self):
        """ACTIVE → ACTIVE_MATCH → ACTIVE."""
        s = Slot.new("m")
        for target in [SlotState.STAGED, SlotState.LOADING, SlotState.ACTIVE]:
            transition(s, target)
        transition(s, SlotState.ACTIVE_MATCH)
        assert s.state == SlotState.ACTIVE_MATCH
        transition(s, SlotState.ACTIVE)
        assert s.state == SlotState.ACTIVE

    def test_idle_hot_swap_to_staged(self):
        """IDLE_HOT → POPPED → STAGED (different model swap)."""
        s = Slot.new("m")
        for target in [
            SlotState.STAGED, SlotState.LOADING, SlotState.ACTIVE,
            SlotState.GRACE, SlotState.POPPED, SlotState.IDLE_HOT,
            SlotState.POPPED, SlotState.STAGED,
        ]:
            transition(s, target)

    def test_idle_hot_to_active_same_model(self):
        """IDLE_HOT → ACTIVE (fresh request for same model_tag)."""
        s = Slot.new("m")
        for target in [
            SlotState.STAGED, SlotState.LOADING, SlotState.ACTIVE,
            SlotState.GRACE, SlotState.POPPED, SlotState.IDLE_HOT,
            SlotState.ACTIVE,
        ]:
            transition(s, target)


class TestIllegalTransitions:
    def test_received_to_active_illegal(self):
        s = Slot.new("m")
        with pytest.raises(InvalidTransition, match="RECEIVED.*ACTIVE"):
            transition(s, SlotState.ACTIVE)

    def test_active_to_loading_illegal(self):
        s = Slot.new("m")
        s.state = SlotState.ACTIVE
        with pytest.raises(InvalidTransition):
            transition(s, SlotState.LOADING)

    def test_cold_is_terminal(self):
        s = Slot.new("m")
        s.state = SlotState.COLD
        for target in [SlotState.ACTIVE, SlotState.STAGED, SlotState.GRACE]:
            with pytest.raises(InvalidTransition):
                transition(s, target)

    def test_grace_busy_to_active_match_illegal(self):
        # GRACE_BUSY only goes to GRACE or POPPED
        s = Slot.new("m")
        s.state = SlotState.GRACE_BUSY
        with pytest.raises(InvalidTransition):
            transition(s, SlotState.ACTIVE_MATCH)


class TestCanTransition:
    def test_can_transition_true(self):
        s = Slot.new("m")
        assert can_transition(s, SlotState.STAGED) is True

    def test_can_transition_false(self):
        s = Slot.new("m")
        assert can_transition(s, SlotState.COLD) is False


class TestLegalTargets:
    def test_received_targets(self):
        targets = legal_targets(SlotState.RECEIVED)
        assert SlotState.STAGED in targets
        assert SlotState.ACCEPT_BUFFER in targets

    def test_cold_no_targets(self):
        assert legal_targets(SlotState.COLD) == set()

    def test_returns_copy_not_alias(self):
        targets = legal_targets(SlotState.STAGED)
        targets.clear()
        # Original unmodified
        assert SlotState.LOADING in legal_targets(SlotState.STAGED)


class TestIsTerminal:
    def test_cold_is_terminal(self):
        assert is_terminal(SlotState.COLD) is True

    def test_other_states_not_terminal(self):
        for st in [SlotState.STAGED, SlotState.LOADING, SlotState.ACTIVE, SlotState.GRACE]:
            assert is_terminal(st) is False


class TestIsWarmState:
    def test_warm_states(self):
        for st in [
            SlotState.LOADING, SlotState.ACTIVE, SlotState.GRACE,
            SlotState.GRACE_BUSY, SlotState.ACTIVE_MATCH, SlotState.IDLE_HOT,
        ]:
            assert is_warm_state(st) is True

    def test_cold_states(self):
        for st in [SlotState.RECEIVED, SlotState.STAGED, SlotState.POPPED, SlotState.COLD]:
            assert is_warm_state(st) is False


class TestIsRequestInFlight:
    def test_in_flight(self):
        assert is_request_in_flight(SlotState.ACTIVE) is True
        assert is_request_in_flight(SlotState.GRACE_BUSY) is True
        assert is_request_in_flight(SlotState.ACTIVE_MATCH) is True

    def test_not_in_flight(self):
        for st in [SlotState.STAGED, SlotState.LOADING, SlotState.GRACE,
                   SlotState.IDLE_HOT, SlotState.COLD]:
            assert is_request_in_flight(st) is False


class TestTransitionTableInvariants:
    def test_all_enum_states_present(self):
        """Every SlotState must be a key in LEGAL_TRANSITIONS (even if empty)."""
        for st in SlotState:
            assert st in LEGAL_TRANSITIONS

    def test_all_target_states_are_valid_enum(self):
        for from_state, targets in LEGAL_TRANSITIONS.items():
            for t in targets:
                assert isinstance(t, SlotState)
