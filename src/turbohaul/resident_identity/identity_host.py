"""The host contract: exactly what the module asks of the object that runs it.

The module imports nothing from the rest of the project. It is handed one ``host`` (the
manager) and may touch only the members declared here, on the host and on the resident and
slot objects it is given. The contract test records every member the module reads and fails
when one is not declared, and when a declared one is never used. All of these are typed
structurally, so a test can pass a small fake with the same members.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ResidentLike(Protocol):
    """One engine's registry entry: the attributes the module reads."""

    resident_key: str
    model_tag: str
    handle: object | None
    spawn_seq: int


class SlotLike(Protocol):
    """The request just bound to a resident: the attributes the identity is built from."""

    client_meta: dict | None
    model_tag: str
    thread_id: str
    fastlane: object | None


class ResidentIdentityHost(Protocol):
    """The manager, reduced to the four members the module uses."""

    _last_request_identity_by_model: dict[str, dict]

    def _identity_dict(
        self,
        *,
        ip: str | None,
        model_tag: str,
        client_meta: dict | None,
        thread_id: str,
        fastlane: object | None = None,
    ) -> dict: ...

    def _model_residents(self) -> Sequence[ResidentLike]: ...

    def _spawn_seq_for_model(self, model_tag: str | None) -> int: ...
