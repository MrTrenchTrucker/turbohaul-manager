"""Tuning numbers for the co-residence re-check.

They live in one file so the manager, the tests and the card all quote the same
values. Every figure is seconds on the monotonic clock unless the name says otherwise.
"""

from __future__ import annotations

# At most one re-check ATTEMPT per interval for the whole box: this is what bounds the
# number of card probes the re-check can cause, however many tags are waiting.
RECHECK_MIN_INTERVAL_S = 10.0

# After a no-room verdict for a tag the re-check does not probe that tag again for this
# long, so a full or unreadable card is not asked about on every tick.
NO_ROOM_MEMO_S = 60.0

# A waiting request younger than this is ordinary arrival traffic that the normal
# dispatch path is still handling, not starved demand.
DEMAND_MIN_AGE_S = 5.0

# A warm follow-up from a thread the resident's grace loop still holds is claimed by that
# loop within seconds, so it only counts as starved after this much longer wait.
PROTECTED_DEMAND_MIN_AGE_S = 120.0

# Hard bound on how many tags a memo remembers, so a stream of distinct tag names cannot
# grow it without limit.
MEMO_MAX_ENTRIES = 64
