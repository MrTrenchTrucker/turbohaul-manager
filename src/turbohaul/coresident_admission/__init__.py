"""Admission for a model that shares a card with other single-card models.

A tensor-isolated model (split_mode 'none') may load onto a card that already holds
other such models, as long as the card has room for it. This module holds the one
rule that answers that question, and the helpers the admission gate builds on.

The rule is one inequality:

    free - reserve + credit >= need

* ``free`` is the free VRAM on the card, from the live reading the caller takes.
* ``reserve`` is the footprint of sibling models that are still booting on that
  card. Their weights are not in the live reading yet, so they are subtracted here.
  A sibling that is already loaded is inside the reading and is never subtracted.
* ``credit`` is VRAM that in-flight unloads have promised to free. The caller works
  it out, because only the caller knows which unloads are pending.
* ``need`` is what the incoming model needs.

:func:`retry_read_free_vram` gets a usable per-card reading when a probe can flake.
:func:`reserved_loading_mib` sums the booting siblings on one card. :func:`fits` is
the only place the comparison is written. :func:`admit` composes them for a per-card
reading that is already in hand.

The module never reads the GPU. The caller passes its own reader in as a callable,
looked up at the moment of the call, so the probe that runs nvidia-smi stays where
it is and a test that replaces that probe still replaces the one the caller uses.

The module is pure: no I/O, no logging, no locks, no globals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "SiblingLoad",
    "retry_read_free_vram",
    "reserved_loading_mib",
    "fits",
    "admit",
]


@dataclass(frozen=True)
class SiblingLoad:
    """One model already resident on the box, as the caller sees it.

    Only what the rule needs: the card the sibling is pinned to, the VRAM it is
    reserved to take, and whether it is still booting.
    """

    main_gpu: int
    """Card the sibling is pinned to."""

    reserved_need_mib: int
    """VRAM the sibling is reserved to take once it has loaded, in MiB."""

    loading: bool
    """True only while the sibling is still booting: its weights are not yet in the live reading."""


def retry_read_free_vram(
    read: Callable[[], list[int] | None],
    attempts: int = 2,
) -> list[int]:
    """Return the first non-empty per-card free-MiB reading, or an empty list.

    A probe can flake and return None or an empty list. Re-reading a bounded number
    of times recovers from a single hiccup. The loop stops at the first usable
    reading, so a healthy probe costs exactly one call. When every read flakes the
    answer is an empty list, which callers must treat as "unknown", never as room.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    for _ in range(attempts):
        vals = read()
        if vals:
            return vals
    return []


def reserved_loading_mib(siblings: Iterable[SiblingLoad], gpu: int) -> int:
    """Return the VRAM still to be taken by booting siblings on card ``gpu``.

    Only siblings that are still loading count, and only those on the card asked
    about. A loaded sibling is already in the live reading, so counting it here
    would subtract its VRAM twice.
    """
    return sum(s.reserved_need_mib for s in siblings if s.loading and s.main_gpu == gpu)


def fits(free_mib: int, reserve_mib: int, credit_mib: int, need_mib: int) -> bool:
    """Return True when the card has room for ``need_mib`` after reserve and credit.

    Every admission path asks this one function, so the rule cannot drift between
    copies. The comparison is ``>=``: a card with exactly enough room admits.
    """
    return free_mib - reserve_mib + credit_mib >= need_mib


def admit(
    vals: list[int] | None,
    main_gpu: int,
    siblings: Sequence[SiblingLoad],
    credit: int,
    need: int,
) -> bool:
    """Decide admission from a per-card free-MiB reading.

    An empty or missing reading refuses, because with no evidence about the cards
    the module must never admit blind. The card read is ``main_gpu`` when it is in
    range and card 0 otherwise, the same fallback the primary read uses. The reserve
    is always taken for ``main_gpu`` itself, not for the card that was picked. That
    is deliberate: with an out-of-range ``main_gpu`` the reading comes from card 0,
    but siblings pinned to the out-of-range number still count.
    """
    if not vals:
        return False
    idx = main_gpu if 0 <= main_gpu < len(vals) else 0
    return fits(vals[idx], reserved_loading_mib(siblings, main_gpu), credit, need)
