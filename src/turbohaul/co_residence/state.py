"""The mutable state the re-check keeps between dispatch-loop passes.

One object per manager. It holds no clock: every time value is passed in by the caller.
"""

from __future__ import annotations

from .memo import NoRoomMemo


class RecheckState:
    """When the last attempt ran, and the two bounded memos the re-check uses.

    ``norom`` remembers, per model tag, that the fit check found no room, so the tag is not
    probed again for a while. ``declog`` throttles the decline line the arrival path writes
    (key ``"<tag>|<reason>"``), so the same verdict is logged once per window. Every state
    builds its own memos: two states never share one.
    """

    def __init__(self) -> None:
        self.last_recheck_at: float | None = None
        self.norom: NoRoomMemo = NoRoomMemo()
        self.declog: NoRoomMemo = NoRoomMemo()
