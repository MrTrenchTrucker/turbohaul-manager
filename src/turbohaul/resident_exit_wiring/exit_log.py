"""The manager-facing half of the resident exit line: write it, and pick its reason when nobody stamped one.

Best-effort and stateless. The caller passes the line formatter and the state names in, so
this file does not reach back into the manager.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from turbohaul.resident_exit import ExitReason, needs_exit_line

# The logger name is the manager's, so existing log handlers and filters keep matching.
_log = logging.getLogger("turbohaul.manager")


def log_resident_exit(
    r: Any, reason: Any, state_before: object, port: object = None, *,
    format_line: Callable[..., str], state_after: object,
) -> None:
    """One RESIDENT_EXIT line per resident death. Best-effort: never raises into the death path."""
    if not needs_exit_line(reason):
        return
    try:
        slot = getattr(r, "active_slot", None)
        tip = getattr(r, "grace_tip", None)
        thread = (getattr(tip, "thread_id", None) or getattr(slot, "thread_id", None)
                  or getattr(r, "idle_thread_id", None))
        _log.info("%s", format_line(
            reason=reason, resident_key=getattr(r, "resident_key", None),
            model_tag=getattr(r, "model_tag", None), thread=thread,
            port=port if port is not None else getattr(getattr(r, "handle", None), "port", None),
            state_before=state_before, state_after=state_after))
    except Exception:  # noqa: BLE001 — best-effort, never raises into the death path
        _log.debug("resident exit line failed (best-effort)", exc_info=True)


def resolve_exit_reason(r: Any, idle: bool) -> Any:
    """The reason a resident's death is logged under: the one the caller stamped, else a default.

    A resident the caller did not stamp resolves to SELF_IDLE_TIMEOUT when ``idle`` is true
    and to UNSPECIFIED otherwise.
    """
    reason = getattr(r, "exit_reason", None)
    if reason is None:
        # The pinned self idle-timeout caller and any unstamped future caller land here; an
        # unstamped caller on a non-idle resident is logged as reason=unspecified.
        reason = (
            ExitReason.SELF_IDLE_TIMEOUT if idle
            else ExitReason.UNSPECIFIED
        )
    return reason
