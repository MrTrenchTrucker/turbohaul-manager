"""The two log lines of the co-residence work, and the reason a second engine was declined.

The caller decides to log and logs; this file only builds the text. Each line is a single
line with a fixed field order so a log search and a test can both rely on its shape.
"""

from __future__ import annotations

import enum
from itertools import groupby
from typing import Any


class DeclineReason(enum.StrEnum):
    """Why the step-2 spawn path did not add a second engine for a tag."""

    CAP = "cap"
    """The tag already runs as many engines as its effective cap allows."""

    BOX_FULL = "box_full"
    """The box-wide sidecar budget is spent, so no tag may add an engine."""

    NO_CARD = "no_card"
    """The card picker found no single card that fits, or the free-VRAM probe was unreadable."""

    VRAM_REFUSED = "vram_refused"
    """The cross-resident VRAM gate refused the card that was picked."""


def _clean(value: Any) -> str:
    """Render a value as one log token: no spaces, no line breaks, never empty.

    Whitespace would split a field in two and a newline would split the line, so each run of
    whitespace becomes one underscore. An absent or blank value becomes a dash so the field
    is still present and the line keeps its shape.
    """
    if value is None:
        return "-"
    text = str(value)
    if not text.strip():
        return "-"
    return "".join("_" if is_space else "".join(chars) for is_space, chars in groupby(text, str.isspace))


def format_step2_decline(
    *,
    via: Any,
    tag: Any,
    reason: Any,
    eff_cap: Any,
    instances: Any,
    residents: Any,
    max_sidecars: Any,
    need_mib: Any,
    card: Any,
    split: Any,
) -> str:
    """The single INFO line for a declined second engine; no trailing newline."""
    return (
        f"CO_RESIDENCE_DECLINED via={_clean(via)} tag={_clean(tag)} reason={_clean(reason)} "
        f"eff_cap={_clean(eff_cap)} instances={_clean(instances)} residents={_clean(residents)} "
        f"max_sidecars={_clean(max_sidecars)} need_mib={_clean(need_mib)} card={_clean(card)} "
        f"split={_clean(split)}"
    )


def format_recheck_action(
    *, tag: Any, source: Any, slot_id: Any, waited_s: Any, instances: Any, residents: Any
) -> str:
    """The single INFO line for a re-check that is about to take a waiting slot; no trailing newline."""
    try:
        waited = f"{float(waited_s):.1f}"
    except Exception:  # noqa: BLE001 - any value that is not a number is shown as absent
        # Any value that cannot be a float is shown as absent rather than failing the log.
        waited = "-"
    return (
        f"CO_RESIDENCE_RECHECK tag={_clean(tag)} source={_clean(source)} slot={_clean(slot_id)} "
        f"waited_s={waited} instances={_clean(instances)} residents={_clean(residents)}"
    )
