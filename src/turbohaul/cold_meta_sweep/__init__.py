"""The manager-facing side of the cold slot metadata retention pass, as one module.

An ended slot keeps only its identity keys, but rows written before that rule still hold a full request
body. A background pass reduces them in small runs. This module holds the run itself (the database access
for one bounded batch) and the decision whether a pass is due, with its log line. It is given everything it
touches and keeps no state of its own. Which keys are kept, and the strip of one row, live elsewhere.
"""

from __future__ import annotations

from .sweep import maybe_sweep, run_once

__all__ = [
    "run_once",
    "maybe_sweep",
]
