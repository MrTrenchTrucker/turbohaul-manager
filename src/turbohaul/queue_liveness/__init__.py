"""Does a waiting request still have a client? One shared answer, and the eviction of the ones that do not.

* ``client_gone`` reads the answer, ``stamp_if_disconnected`` flags the slot, and ``drop_if_gone``
  also fails it once. All three are await-free.
* ``evict_unstarted`` removes one waiting request from wherever it waits.
* ``notify_client_gone`` is the synchronous call a route makes when it sees a hang-up.
* ``QueueLivenessHost``, ``SlotLike`` and ``ResidentLike`` are the structural types of the manager
  and of the request and resident objects it hands in. The module imports nothing from the rest
  of the project.
"""

from __future__ import annotations

from .eviction import evict_unstarted, notify_client_gone
from .liveness import client_gone, drop_if_gone, stamp_if_disconnected
from .liveness_host import QueueLivenessHost, ResidentLike, SlotLike

__all__ = [
    "QueueLivenessHost",
    "ResidentLike",
    "SlotLike",
    "client_gone",
    "stamp_if_disconnected",
    "drop_if_gone",
    "evict_unstarted",
    "notify_client_gone",
]
