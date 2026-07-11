"""Slot dataclass + state enum + thread_id derivation.

Per ARCHITECTURE.md — thread_id prefix-hash fallback.
"""
import asyncio
import dataclasses
import enum
import hashlib
import time
import uuid
from typing import Any, Optional


class SlotEvictedError(Exception):
    """Raised when a slot's completion_future is failed due
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
    (client_closed_request).
    """


class SlotState(str, enum.Enum):
    """States per the state machine (10 states total)."""

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
    admission_ctx_len: int = 0  # incoming context size recorded at admission
    # Classifier: incoming turn-hash chain recorded at admission
    # (parallel to admission_ctx_len). _prefix_hash_chain(messages) computed at the
    # chat_completion admission site so the manager restore path can classify the
    # request by prefix-VALIDITY vs the pinned clean bin (not by char length alone).
    # Default [] = no chain threaded -> restore gate SKIPS (fail-safe, never blind).
    admission_hash_chain: list[str] = dataclasses.field(default_factory=list)
    # Named engine operation for the dashboard pill.
    # Tracks the high-level engine phase: "kv_restore" | "kv_save" | "prefill" | "decode" | "idle" | "stream" | "unload".
    engine_op: str = "idle"
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
    # The SSE route accumulates the streamed generated
    # assistant text (content + reasoning merged as <think>...</think>{content},
    # mirroring _merge_reasoning_into_content) and stashes it here BEFORE setting
    # stream_done_event, so the manager can reconstruct the engine-view warm_chain
    # for the STREAMING forced clean-restore (streaming agent workloads). None
    # when unknown (non-thinking/tool-call/parse-miss) -> gate safe-degrades.
    streamed_assistant_text: str | None = None
    # Client-disconnect eviction signal.
    # Lazy-init `None` (NOT default_factory=asyncio.Event).
    # default_factory binds the Event to whatever loop is current at dataclass
    # construction time — wrong-loop fragility in tests + BootInventory replay.
    # Routes attach an Event constructed FROM their own request handler scope
    # (correct loop). Non-HTTP callers (BootInventory, internal probes) pass None.
    disconnect_event: Optional[asyncio.Event] = None
    is_evicted: bool = False  # set by pop_*_non_evicted_from when caller-disconnected
    # Completion-cache single-flight carrier. When a
    # non-streaming request becomes the cache LEADER, submit_and_wait stashes the
    # completion-key here BEFORE enqueue so the _process_slot WRITE site can cache
    # the result + resolve the leader's single-flight future keyed by it. None for
    # riders / streaming / cache-disabled requests -> the WRITE site is a no-op.
    completion_cache_key: str | None = None

    @classmethod
    def new(
        cls,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict[str, Any] | None = None,
        admission_ctx_len: int = 0,
        admission_hash_chain: list[str] | None = None,
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
            admission_ctx_len=admission_ctx_len,
            admission_hash_chain=admission_hash_chain or [],
        )


def derive_thread_id_prefix_hash(
    prompt: str, model_tag: str, prefix_tokens: int | None = None
) -> str:
    """Auto-derive thread_id for clients that send no explicit thread_id.

    Prefix-token keying: hash only the first N tokens of the
    normalized prompt rather than the full prompt. This ensures that conversation
    extensions — same prefix, more tokens appended — produce the SAME thread_id,
    allowing the grace window to match and the KV cache restore to fire.

    ``prefix_tokens`` controls the cutoff (default 256). When None, the default
    is used. When supplied (e.g. via config), it overrides the default.

    ``prefix_tokens`` defaults to 256 which captures a typical system prompt +
    initial context. Adjust via QueueConfig.prefix_token_count /
    TURBOHAUL_PREFIX_TOKEN_COUNT if needed.

    Normalization is word-based (whitespace split) so incidental whitespace
    differences still map a semantically-identical prompt to one thread_id.
    """
    n = prefix_tokens if prefix_tokens is not None else 256
    normalized = " ".join(prompt.split())
    words = normalized.split()
    prefix = " ".join(words[:n]) if len(words) > n else normalized
    payload = (model_tag + "\0" + prefix).encode("utf-8")
    return "auto-" + hashlib.sha256(payload).hexdigest()[:24]
