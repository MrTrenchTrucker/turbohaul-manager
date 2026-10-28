"""resident_exit: keep a finishing co-resident warm, or let it go, and name every resident death.

Pure policy. The caller measures the live state, decides, and logs; this package only
answers two questions. Should a finishing same-model co-resident stay warm for a bounded
window (retention.decide_retention)? Which exit-class line does a resident death produce,
and does it need one at all (exit_line.format_resident_exit, needs_exit_line)?

Stdlib only. No I/O, no logging, no locks, no globals, no live state. The caller holds
the registry lock and supplies the already-measured inputs.
"""

from __future__ import annotations

from .exit_line import format_resident_exit, needs_exit_line
from .reasons import ExitReason
from .retention import RetentionDecision, RetentionInputs, decide_retention

__all__ = [
    "ExitReason",
    "needs_exit_line",
    "format_resident_exit",
    "RetentionInputs",
    "RetentionDecision",
    "decide_retention",
]
