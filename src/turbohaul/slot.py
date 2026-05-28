"""Slot dataclass + state enum + thread_id derivation.

Per v0.2 ARCHITECTURE.md §4 + §6 + §9 thread_id prefix-hash fallback.
"""
import asyncio
import dataclasses
import enum
import hashlib
import time
import uuid
from typing import Any, Optional


class SlotEvictedError(Exception):
    """: raised when a slot's completion_future is failed due
    to client-disconnect eviction in the queue.

    DEDICATED Exception subclass — NOT a reuse of asyncio.CancelledError —
    because:
    - CancelledError inherits from BaseException (Py 3.8+), which slips past
      ``except Exception:`` handlers and ``raise from None`` chains. Routes
      catching ``Exception`` would silently drop CancelledError-shaped
      eviction signals.
    - Semantic clarity: 'we evicted because the client disconnected' is a
      different fault domain from 'event loop cancelled this task'. Mixing
      them costs us metric / 4xx-vs-5xx clarity.

    Routes catch this in their non-streaming await chain and surface HTTP 499
    (client_closed_request) per Bonus-1.
    """


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
    # Optional asyncio.Future, set by submit(wait_for_completion=True) so caller can
    # await the slot's completion response. worker_loop sets the result after
    # completion_fn returns.
    completion_future: Any = None
    # SSE streaming pass-through:
    # When client sends stream:true, the route uses submit_for_streaming() instead
    # of submit_and_wait(). The slot stays ACTIVE for the full stream lifetime —
    # worker_loop sets stream_ready_event after llama-server health-200 (handle
    # stored on stream_handle), the route opens its own httpx.stream() to the
    # sidecar, yields SSE chunks, and signals stream_done_event when the gen
    # exhausts (or on client disconnect / error). Only then does worker_loop
    # advance ACTIVE → GRACE.
    stream_ready_event: Any = None  # asyncio.Event, set by worker_loop on ACTIVE
    stream_done_event: Any = None   # asyncio.Event, set by route on stream close
    stream_handle: Any = None       # SidecarHandle assigned when ACTIVE
    # client-disconnect eviction signal.
    # MOD-A / FAIL-A + FAIL-B: lazy-init `None` (NOT default_factory=asyncio.Event).
    # default_factory binds the Event to whatever loop is current at dataclass
    # construction time — wrong-loop fragility in tests + BootInventory replay.
    # Routes attach an Event constructed FROM their own request handler scope
    # (correct loop). Non-HTTP callers (BootInventory, internal probes) pass None.
    disconnect_event: Optional[asyncio.Event] = None
    is_evicted: bool = False  # set by pop_*_non_evicted_from when caller-disconnected

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
    """Auto-derive thread_id for naive Ollama clients (v0.2 §9 - F7 fix).

    Same-prefix follow-ups in the same model get the same thread_id, enabling
    warm-slot reuse without the client knowing about thread_id semantics.

    Token counting is word-based (whitespace split) - approximate but good for routing.
    """
    tokens = prompt.split()[:prefix_tokens]
    payload = (model_tag + "\0" + " ".join(tokens)).encode("utf-8")
    return "auto-" + hashlib.sha256(payload).hexdigest()[:24]
