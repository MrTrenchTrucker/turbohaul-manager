"""Why a resident is leaving, named once so every exit line says the same thing.

Each member names one way a resident reaches DEAD on the live path. The caller picks
the member for the path it is on; this module only names it and decides whether that
exit needs its own log line (see exit_line.needs_exit_line).
"""

from __future__ import annotations

import enum


class ExitReason(enum.StrEnum):
    """The cause recorded on a resident's exit line.

    The value is the text that appears in the log, so a changed value changes the log
    contract and needs a matching change in the card and the README.
    """

    IMMEDIATE_EVICT = "immediate_evict_keep_alive_0"
    """keep_alive is 0 or less and nothing outranks the resident, so it is torn down at
    once. Kept apart from DESIGNATED_VICTIM so a reader can tell a client that asked for
    no warm window from a make-room pick."""

    DESIGNATED_VICTIM = "designated_victim_idle_break"
    """A pending higher-ranked request has designated this resident as its victim (the
    worst-ranked loaded resident), so it unloads at its turn boundary. Kept apart from
    IMMEDIATE_EVICT: the cause is the designation, not a keep_alive of 0."""

    SELF_IDLE_TIMEOUT = "self_idle_timeout"
    """The resident's own idle wait ran out with nothing queued for it, so it unloads
    itself. A timer death, not a decision by another request."""

    DRIVER_EXIT = "driver_exit"
    """The driver's own finally block deregistered a resident that nobody unloaded. Without this
    reason that death would leave no line at all."""

    DRIVER_REAP = "driver_reap"
    """The supervisor reaper (_reap_dead_resident), scheduled when the driver task ends or
    is cancelled, unloaded a resident that was still registered. Kept apart from
    DRIVER_EXIT so the line says the reaper did the unload, not the driver's own finally."""

    IDLE_ENGINE_DEAD = "idle_engine_dead"
    """The idle liveness sweep found an idle resident whose engine had died. The sweep is
    not the reaper, so it carries its own cause."""

    MAKE_ROOM = "make_room"
    """A make-room path evicted the resident to free capacity for another request. The
    caller already writes this death as MAKE_ROOM_EVICTION, so it must not get a second
    exit line."""

    UNSPECIFIED = "unspecified"
    """A caller passed no reason. It is loud on purpose: the line reads
    reason=unspecified instead of the death going unlogged."""
