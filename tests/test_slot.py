"""Tests for Slot + thread_id prefix-hash derivation (v0.2 §4, §6, §9)."""
from turbohaul.slot import Slot, SlotState, derive_thread_id_prefix_hash


class TestSlotState:
    def test_state_values(self):
        assert SlotState.RECEIVED.value == "RECEIVED"
        assert SlotState.STAGED.value == "STAGED"
        assert SlotState.LOADING.value == "LOADING"
        assert SlotState.ACTIVE.value == "ACTIVE"
        assert SlotState.GRACE.value == "GRACE"
        assert SlotState.GRACE_BUSY.value == "GRACE_BUSY"
        assert SlotState.ACTIVE_MATCH.value == "ACTIVE_MATCH"
        assert SlotState.POPPED.value == "POPPED"
        assert SlotState.IDLE_HOT.value == "IDLE_HOT"
        assert SlotState.COLD.value == "COLD"

    def test_state_is_string(self):
        assert SlotState.STAGED == "STAGED"


class TestSlot:
    def test_new_generates_slot_id(self):
        s = Slot.new("qwen3.6-35b-moe")
        assert s.slot_id.startswith("slot-")
        assert len(s.slot_id) > len("slot-")
        assert s.state == SlotState.RECEIVED
        assert s.model_tag == "qwen3.6-35b-moe"

    def test_new_with_thread_id(self):
        s = Slot.new("m", thread_id="thr-abc")
        assert s.thread_id == "thr-abc"

    def test_new_with_client_meta(self):
        s = Slot.new("m", client_meta={"requester": "agent-a", "audit_id": "x"})
        assert s.client_meta["requester"] == "agent-a"
        assert s.client_meta["audit_id"] == "x"

    def test_new_unique_slot_ids(self):
        s1 = Slot.new("m")
        s2 = Slot.new("m")
        assert s1.slot_id != s2.slot_id

    def test_default_extension_count_zero(self):
        s = Slot.new("m")
        assert s.extension_count == 0

    def test_created_at_set(self):
        s = Slot.new("m")
        assert s.created_at > 0.0


class TestThreadIdDerivation:
    def test_same_prefix_same_id(self):
        prompt1 = "Translate this English text into French: The quick brown fox jumps over the lazy dog"
        prompt2 = "Translate this English text into French: The quick brown fox jumps over the lazy dog. Additional context here."
        t1 = derive_thread_id_prefix_hash(prompt1, "m", prefix_tokens=10)
        t2 = derive_thread_id_prefix_hash(prompt2, "m", prefix_tokens=10)
        assert t1 == t2

    def test_different_model_tag_different_id(self):
        prompt = "hello world"
        t1 = derive_thread_id_prefix_hash(prompt, "model-a")
        t2 = derive_thread_id_prefix_hash(prompt, "model-b")
        assert t1 != t2

    def test_different_prompt_different_id(self):
        t1 = derive_thread_id_prefix_hash("foo bar baz", "m")
        t2 = derive_thread_id_prefix_hash("qux quux corge", "m")
        assert t1 != t2

    def test_starts_with_auto_prefix(self):
        t = derive_thread_id_prefix_hash("hello", "m")
        assert t.startswith("auto-")
        assert len(t) > len("auto-")

    def test_deterministic(self):
        t1 = derive_thread_id_prefix_hash("hello world", "m", prefix_tokens=10)
        t2 = derive_thread_id_prefix_hash("hello world", "m", prefix_tokens=10)
        assert t1 == t2
