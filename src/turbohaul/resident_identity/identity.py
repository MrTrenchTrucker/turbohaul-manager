"""The last request identity of each resident: store it, read it, forget it.

The identity is kept under the resident's own key, never under the model tag alone, so two
engines of one model each show the request they serve. The dict lives on the host and the
host decides when a resident is gone; this module only reads and writes it, and looks the
dict up on the host at every call because the host may replace it.
"""

from __future__ import annotations

import logging

from .identity_host import ResidentIdentityHost, ResidentLike, SlotLike

_log = logging.getLogger(__name__)


def stash_identity(host: ResidentIdentityHost, r: ResidentLike, slot: SlotLike, singleton_key: str) -> None:
    """Store the identity of the request just bound to resident ``r`` under ``r.resident_key``.

    An empty key and the legacy singleton key (handed in by the caller) are never stored and
    the dict is not touched. The identity is built by the host from the slot's own fields.
    This is display-only bookkeeping, so any failure is logged at debug level and swallowed:
    it must never fail a request, and a failed stash leaves the dict as it was.
    """
    try:
        key = r.resident_key
        if not key or key == singleton_key:
            return
        host._last_request_identity_by_model[key] = host._identity_dict(
            ip=(slot.client_meta or {}).get("ip"),
            model_tag=slot.model_tag,
            client_meta=slot.client_meta,
            thread_id=slot.thread_id,
            fastlane=slot.fastlane,
        )
    except Exception:  # noqa: BLE001 -- display-only, never fail a request
        _log.debug("resident identity stash failed (ignored)", exc_info=True)


def identity_for(host: ResidentIdentityHost, r: ResidentLike) -> dict | None:
    """The stored identity of resident ``r``, or None. A resident with no key reads the entry of its model tag."""
    return host._last_request_identity_by_model.get(r.resident_key or r.model_tag)


def forget_identity(host: ResidentIdentityHost, r: ResidentLike) -> None:
    """Drop the identity stored under ``r.resident_key``, always exactly that key: a same-model sibling has its own."""
    host._last_request_identity_by_model.pop(r.resident_key, None)
