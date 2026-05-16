"""Slot dataclass + state enum + thread_id derivation.

Per v0.2 ARCHITECTURE.md §4 + §6 + §9 thread_id prefix-hash fallback.
"""
import dataclasses
import enum
import hashlib
import time
import uuid
from typing import Any


class SlotState(str, enum.Enum):
    """States per v0.2 §6 state machine (10 states total)."""

    RECEIVED = "RECEIVED"
    ACCEPT_BUFFER = "ACCEPT_BUFFER"
    STAGED = "STAGED"
    LOADING = "LOADING"
    LOADING_FAIL = "LOADING_FAIL"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"
    GRACE_BUSY = "GRACE_BUSY"
    ACTIVE_MATCH = "ACTIVE_MATCH"
    POPPED = "POPPED"
    IDLE_HOT = "IDLE_HOT"
    COLD = "COLD"


@dataclasses.dataclass
class Slot:
    """A single queued or active request slot."""

    slot_id: str
    model_tag: str
    state: SlotState
    prompt: str = ""
    context: list[dict] | None = None
    thread_id: str = ""
    port: int | None = None
    pid: int | None = None
    extension_count: int = 0
    client_meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at: float = 0.0  # monotonic time at creation
    started_active_at: float = 0.0  # monotonic when entered ACTIVE
    grace_started_at: float = 0.0  # monotonic when entered GRACE

    @classmethod
    def new(
        cls,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict[str, Any] | None = None,
    ) -> "Slot":
        return cls(
            slot_id=f"slot-{uuid.uuid4().hex[:12]}",
            model_tag=model_tag,
            state=SlotState.RECEIVED,
            prompt=prompt,
            context=context,
            thread_id=thread_id,
            client_meta=client_meta or {},
            created_at=time.monotonic(),
        )


def derive_thread_id_prefix_hash(
    prompt: str, model_tag: str, prefix_tokens: int = 64
) -> str:
    """Auto-derive thread_id for naive Ollama clients (v0.2 §9 - Devil F7 fix).

    Same-prefix follow-ups in the same model get the same thread_id, enabling
    warm-slot reuse without the client knowing about thread_id semantics.

    Token counting is word-based (whitespace split) - approximate but good for routing.
    """
    tokens = prompt.split()[:prefix_tokens]
    payload = (model_tag + "\0" + " ".join(tokens)).encode("utf-8")
    return "auto-" + hashlib.sha256(payload).hexdigest()[:24]
