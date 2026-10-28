"""The host contract: exactly what the module asks of the object that runs it.

The module imports nothing from the rest of the project. It is handed one ``host`` (the
manager) and may touch only the members declared here, on the host and on the slot and
resident objects it is given. The contract test records every member the module reads and
fails when one is not declared, and when a declared one is never used. All of these are typed
structurally, so a test can pass a small fake with the same members.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol


class _Named(Protocol):
    """A state enum member: only its ``name`` is read, so the module needs no enum import."""

    name: str


class _Event(Protocol):
    """The client's disconnect event (an ``asyncio.Event``)."""

    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class _Future(Protocol):
    """The slot's completion future (an ``asyncio.Future``): only ``done`` is read."""

    def done(self) -> bool: ...


class SlotLike(Protocol):
    """One waiting request: the attributes the module reads and writes."""

    slot_id: str
    state: _Named
    disconnect_event: _Event | None
    is_evicted: bool
    completion_future: _Future | None


class ResidentLike(Protocol):
    """One engine's registry entry: the attributes the module reads.

    ``inbox`` is an ``asyncio.Queue`` (or None); the module uses only its ``get_nowait`` and
    ``put_nowait``, never a private attribute.
    """

    state: _Named
    active_slot: SlotLike | None
    inbox: object | None


class _QueueView(Protocol):
    """The request queue, reduced to the two members the module uses."""

    def waiting_snapshot(self) -> Sequence[SlotLike]: ...

    async def remove(self, slot_id: str) -> SlotLike | None: ...


class QueueLivenessHost(Protocol):
    """The manager, reduced to the six members the module uses."""

    queue: _QueueView
    _registry_lock: AbstractAsyncContextManager

    def _model_residents(self) -> Sequence[ResidentLike]: ...

    def _release_fastlane_claim_locked(self, slot: SlotLike, reason: str) -> None: ...

    def _handle_unloaded_slot(self, slot: SlotLike) -> None: ...

    def _spawn_bg(self, coro: object) -> object: ...
