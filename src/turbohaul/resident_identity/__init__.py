"""Which request each resident is serving: the last request identity per resident, and the handle lookup.

* ``stash_identity``, ``identity_for`` and ``forget_identity`` keep the identity of the last
  request bound to each resident, under the resident's own key.
* ``serving_resident`` and ``spawn_seq_for_handle`` find the resident that serves a request
  handle and its spawn counter.
* ``pending_load`` is the load of one resident (riders in flight, a running turn, the inbox depth), the
  number the manager uses to pick a resident for a new request. A resident in its grace loop keeps the
  slot of its last request; while that slot is in its finished state it is not load. ``LoadLike`` is its
  structural type.
* ``ResidentIdentityHost``, ``ResidentLike`` and ``SlotLike`` are the structural types of the
  manager and of the resident and request objects it hands in. The module imports nothing
  from the rest of the project.
"""

from __future__ import annotations

from .engine_load import LoadLike, pending_load
from .identity import forget_identity, identity_for, stash_identity
from .identity_host import ResidentIdentityHost, ResidentLike, SlotLike
from .serving import serving_resident, spawn_seq_for_handle

__all__ = [
    "ResidentIdentityHost",
    "ResidentLike",
    "SlotLike",
    "stash_identity",
    "identity_for",
    "forget_identity",
    "serving_resident",
    "spawn_seq_for_handle",
    "LoadLike",
    "pending_load",
]
