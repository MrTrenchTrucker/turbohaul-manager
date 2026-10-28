"""A bounded memo of "no room" verdicts that expire on their own.

The re-check records here that a tag found no room, so it does not ask the card probe
about that tag again until the entry expires. The class never reads a clock: the caller
passes ``now`` (monotonic seconds) on every call, which keeps it pure and lets tests drive
time directly.
"""

from __future__ import annotations

from .constants import MEMO_MAX_ENTRIES, NO_ROOM_MEMO_S


class NoRoomMemo:
    """Key -> expiry time, with a fixed time-to-live and a fixed size bound."""

    def __init__(self, ttl_s: float = NO_ROOM_MEMO_S, max_entries: int = MEMO_MAX_ENTRIES) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._expiry: dict[str, float] = {}

    def note(self, key: str, now: float) -> None:
        """Remember ``key`` until ``now + ttl_s``; trim the earliest-expiring others if over the bound."""
        self._expiry[key] = now + self._ttl_s
        excess = len(self._expiry) - self._max_entries
        if excess > 0:
            # Earliest expiry goes first (the key closest to forgetting anyway); ties break by
            # key so the result does not depend on insertion order. The fresh key is excluded
            # so the verdict just recorded always survives, even at a bound below one.
            victims = sorted((exp, k) for k, exp in self._expiry.items() if k != key)[:excess]
            for _exp, k in victims:
                del self._expiry[k]

    def blocked(self, key: str, now: float) -> bool:
        """True while ``key`` is remembered and not yet expired; an expired key is forgotten here."""
        exp = self._expiry.get(key)
        if exp is None:
            return False
        if now < exp:
            return True
        del self._expiry[key]
        return False

    def clear(self, key: str) -> None:
        """Forget ``key``; nothing happens if it is not stored."""
        self._expiry.pop(key, None)

    def __len__(self) -> int:
        return len(self._expiry)

    def keys(self) -> list[str]:
        """The stored keys, sorted, for tests and diagnostics."""
        return sorted(self._expiry)
