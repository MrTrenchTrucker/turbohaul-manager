"""May a tag start a further engine, and on which card? Two answers to two questions.

``fit_verdict`` is the periodic re-check's answer. It looks only at whether the engine
fits: a card of its own (the placement and the card picker) and the cross-resident VRAM
gate. It never reads the box-wide engine budget and ignores the per-tag engine count, so the
only ceilings are ``no_card`` and ``vram_refused``.

``arrival_verdict`` is the answer for a request that has just arrived. It keeps the old
limits (the tag's own engine ceiling and the box-wide budget) before it asks the same fit
questions, and it writes the decline line.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import NamedTuple

from .host import CoResidenceHost, _ResidentView, _SlotView
from .lines import DeclineReason, format_step2_decline
from .state import RecheckState

_log = logging.getLogger(__name__)


class Verdict(NamedTuple):
    """Admit a further engine on ``card`` / ``split`` (need in MiB), or decline with ``reason``.

    ``eff_cap`` is the per-tag ceiling the arrival path used; the re-check has none (None).
    """

    admit: bool
    card: int | None
    split: str | None
    reason: DeclineReason | None
    need: int | None
    eff_cap: int | None


def _log_decline(
    host: CoResidenceHost,
    *,
    via: str,
    tag: str,
    reason: DeclineReason,
    eff_cap: int | None,
    insts: Sequence[_ResidentView],
    max_sidecars: int | None,
    need: int | None,
    card: int | None,
    split: str | None,
) -> None:
    """Write the one CO_RESIDENCE_DECLINED line. Raises whatever a host call raises."""
    _log.info(
        "%s",
        format_step2_decline(
            via=via, tag=tag, reason=reason, eff_cap=eff_cap, instances=len(insts),
            residents=len(host._model_residents()), max_sidecars=max_sidecars,
            need_mib=need, card=card, split=split,
        ),
    )


async def fit_verdict(
    host: CoResidenceHost, state: RecheckState, tag: str, insts: Sequence[_ResidentView], now: float
) -> Verdict:
    """The re-check verdict: does one more engine of ``tag`` fit? Caller holds the registry lock.

    Declines: ``no_card`` when the tag cannot be placed as a second engine or no card is
    visible at all; ``vram_refused`` when cards are visible but none has room for one more
    engine, or the cross-resident gate refused the card that was picked.

    ``state`` and ``now`` are accepted so both verdicts have the same shape; the re-check
    needs neither, because its decline line is already bounded by the no-room memo.
    """
    # The first number is the cap-derived engine count. It is deliberately dropped.
    _cap_derived, wants_one_card = await asyncio.to_thread(host._route_cap_and_placement, tag)
    need = card = split = None
    reason = DeclineReason.NO_CARD
    if wants_one_card:
        need, parallel, *_ = await asyncio.to_thread(host._resolve_placement_locked, tag)
        card, split = await asyncio.to_thread(host._pick_card_for_new_instance, tag, need)
        if card is not None:
            if await asyncio.to_thread(host._vram_admits_locked, need, parallel, card, split):
                return Verdict(True, card, split, None, need, None)
            reason = DeclineReason.VRAM_REFUSED
        elif await asyncio.to_thread(host._card_avail_mib):
            # The picker keeps a safety floor free that the VRAM gate does not, so a plain
            # shortage of VRAM reads as "no card". Cards that are visible but have no room
            # are a VRAM refusal; only an empty reading (no card visible) is "no card".
            reason = DeclineReason.VRAM_REFUSED
    try:
        _log_decline(host, via="recheck", tag=tag, reason=reason, eff_cap=None, insts=insts,
                     max_sidecars=None, need=need, card=card, split=split)
    except Exception:  # noqa: BLE001 - a logging failure must never change the verdict
        _log.debug("co-residence decline line failed (ignored)", exc_info=True)
    return Verdict(False, card, split, reason, need, None)


async def arrival_verdict(
    host: CoResidenceHost, tag: str, insts: Sequence[_ResidentView], eff_cap: int, slot: _SlotView | None
) -> Verdict:
    """The arrival verdict: the tag's own ceiling and the box budget first, then the fit.

    Caller holds the registry lock. A Fast Lane relocation chosen for ``slot`` decides the
    card of a cold start; a tag that already runs an engine keeps the picker's card, so a
    second copy never lands on a card that already hosts one.
    """
    max_sidecars = host.runtime.queue.max_parallel_sidecars
    need = card = split = None
    if len(insts) < eff_cap and len(host._model_residents()) < max_sidecars:
        need, parallel, *_ = await asyncio.to_thread(host._resolve_placement_locked, tag)
        card, split = await asyncio.to_thread(host._pick_card_for_new_instance, tag, need)
        if slot is not None and slot.fastlane_relocated_main_gpu is not None and not insts:
            card = slot.fastlane_relocated_main_gpu
            split = slot.fastlane_relocated_split_mode
        if card is not None and await asyncio.to_thread(host._vram_admits_locked, need, parallel, card, split):
            return Verdict(True, card, split, None, need, eff_cap)
        reason = DeclineReason.NO_CARD if card is None else DeclineReason.VRAM_REFUSED
    else:
        reason = DeclineReason.CAP if len(insts) >= eff_cap else DeclineReason.BOX_FULL
    if insts:
        # A cold tag with no engine is not a co-residence question: no line.
        try:
            state = host._co_residence_state()
            now = host._co_residence_now()
            key = f"{tag}|{reason}"
            # Arrivals repeat the same verdict on every request: log it once per window.
            if not state.declog.blocked(key, now):
                state.declog.note(key, now)
                _log_decline(host, via="arrival", tag=tag, reason=reason, eff_cap=eff_cap, insts=insts,
                             max_sidecars=max_sidecars, need=need, card=card, split=split)
        except Exception:  # noqa: BLE001 - a logging failure must never change the verdict
            _log.debug("co-residence decline line failed (ignored)", exc_info=True)
    return Verdict(False, card, split, reason, need, eff_cap)
