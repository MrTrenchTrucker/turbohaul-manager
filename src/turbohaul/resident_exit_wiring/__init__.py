"""resident_exit_wiring: the manager-facing side of the resident exit feature.

Measures the live inputs for the keep-warm decision and asks resident_exit, writes the
exit log line, and picks the reason when the caller did not stamp one. It holds no state
and starts no tasks; the manager keeps the registry lock, the stamps and the call order.
"""

from __future__ import annotations

from .exit_log import log_resident_exit, resolve_exit_reason
from .hold import hold_warm_window

__all__ = [
    "hold_warm_window",
    "log_resident_exit",
    "resolve_exit_reason",
]
