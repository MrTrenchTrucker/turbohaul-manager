"""The host contract: exactly what the re-check asks of the object that runs it.

The module imports only ``turbohaul.queue_liveness`` (the liveness predicate) from the rest of
the project. It is handed one ``host`` (the manager) and may touch only the members declared
here. The contract test records every member the module reads on a recording host and fails
when one is not declared, and when a declared member is never used. All of these are typed
structurally, so a test can pass a small fake with the same members.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol

from .state import RecheckState


class _StateView(Protocol):
    """A resident's lifecycle state; the module reads only its name."""

    @property
    def name(self) -> str: ...


class _SlotView(Protocol):
    """A waiting request: the attributes the module reads."""

    slot_id: str
    model_tag: str
    thread_id: str | None
    created_at: float
    is_evicted: bool
    disconnect_event: asyncio.Event | None
    fastlane_relocated_main_gpu: int | None
    fastlane_relocated_split_mode: str | None


class _InboxView(Protocol):
    """A resident's inbox: take everything out and put the rest back, without waiting."""

    def get_nowait(self) -> _SlotView: ...

    def put_nowait(self, item: _SlotView) -> None: ...


class _ResidentView(Protocol):
    """One engine's registry entry: the attributes the module reads."""

    state: _StateView
    model_tag: str
    inbox: _InboxView | None
    idle_thread_id: str | None
    resident_key: str


class _StagingQueueView(Protocol):
    """The request queue: a snapshot of staged requests and removal of one by id."""

    def staging_snapshot(self) -> Sequence[_SlotView]: ...

    async def remove(self, slot_id: str) -> _SlotView | None: ...


class _RuntimeQueueSettings(Protocol):
    """The box-wide budget. Only the arrival verdict reads it; the re-check never does."""

    max_parallel_sidecars: int


class _RuntimeView(Protocol):
    """The runtime configuration, reduced to the queue settings."""

    queue: _RuntimeQueueSettings


class _AsyncLockView(Protocol):
    """An asyncio lock: usable with ``async with``."""

    async def __aenter__(self) -> None: ...

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None: ...


class CoResidenceHost(Protocol):
    """Everything the module touches on the manager. Nothing else may be used."""

    queue: _StagingQueueView
    runtime: _RuntimeView
    _registry_lock: _AsyncLockView

    def _co_residence_now(self) -> float: ...

    def _co_residence_state(self) -> RecheckState: ...

    def _model_residents(self) -> Sequence[_ResidentView]: ...

    def _instances_for(self, model_tag: str | None) -> Sequence[_ResidentView]: ...

    async def _grace_active_exclusions(self) -> frozenset[tuple[str, str]]: ...

    def _route_cap_and_placement(self, model_tag: str) -> tuple[int, bool]: ...

    def _resolve_placement_locked(self, model_tag: str) -> tuple[int, int, int, str, int, bool, bool]: ...

    def _card_avail_mib(self) -> Sequence[int]: ...

    def _pick_card_for_new_instance(self, model_tag: str, need_mib: int) -> tuple[int | None, str]: ...

    def _vram_admits_locked(self, need: int, parallel: int, main_gpu: int, split_mode: str) -> bool: ...

    async def _reserve_and_start_locked(
        self, slot: _SlotView, main_gpu_override: int | None = None, split_mode_override: str | None = None
    ) -> _ResidentView | None: ...

    def _remember_affinity(self, model_tag: str, session_key: str | None, resident_key: str) -> None: ...

    def _session_key_for(self, slot: _SlotView) -> str | None: ...

    def _reserving(self, model_tag: str) -> AbstractContextManager[None]: ...

    def _unpark_fastlane_claim(self, slot: _SlotView) -> None: ...

    def _handle_unloaded_slot(self, slot: _SlotView) -> None: ...

    def _fail_completion_future(self, slot: _SlotView, exc: BaseException) -> None: ...
