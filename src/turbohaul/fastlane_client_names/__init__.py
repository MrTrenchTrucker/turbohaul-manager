"""Which container name the Fast Lane Discovered list may show for a client address.

A reverse lookup turns a client address into a name, usually with the container
network as a suffix (``app-one.net_default``). The name an operator would type is
often the bare one, so this module offers the whole answer and the answer with its
trailing labels removed one at a time as candidates. The caller looks each candidate
up forward and passes the results back; a candidate counts only when its forward
lookup returned the client's own address. The shortest confirmed candidate is the
name to show, ties broken alphabetically. With none confirmed there is no name.

The shown name is for display and for building a rule only. It never reaches the
request matcher.

The module is pure: no I/O, no lookups, no logging, no clock and no module-level
state. The caller does the lookups and keeps the cache.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable, Mapping

__all__ = ["name_candidates", "confirmed_name"]


def name_candidates(reverse_answer: str) -> tuple[str, ...]:
    """Candidate names for one reverse lookup answer, whole name first.

    ``a.b.c`` gives ``("a.b.c", "a.b", "a")``. Case is kept as given. One trailing
    dot (the absolute-name form) is dropped first. A malformed answer gives ``()``:
    not a string, empty, any whitespace, an empty label other than that one trailing
    dot (leading dot, double dot, only dots), or an IP address literal, which is not
    a name.
    """
    if not isinstance(reverse_answer, str) or not reverse_answer:
        return ()
    if any(ch.isspace() for ch in reverse_answer):
        return ()
    name = reverse_answer[:-1] if reverse_answer.endswith(".") else reverse_answer
    if not name:
        return ()
    labels = name.split(".")
    if any(not label for label in labels):
        return ()
    try:
        ipaddress.ip_address(name)
    except ValueError:
        pass
    else:
        return ()
    return tuple(".".join(labels[:count]) for count in range(len(labels), 0, -1))


def confirmed_name(address, forward: Mapping[str, Iterable]) -> str | None:
    """The name to show for ``address``, or ``None`` when no candidate confirms.

    ``forward`` maps each candidate name to the addresses its forward lookup
    returned (any iterable, or ``None`` for a failed lookup). A candidate is
    confirmed only when ``address`` is among its addresses. The shortest confirmed
    name wins; equal lengths are ordered alphabetically, never by input order.
    """
    confirmed = []
    for name, addresses in forward.items():
        if addresses is None:
            continue
        if address in set(addresses):
            confirmed.append(name)
    if not confirmed:
        return None
    return min(confirmed, key=lambda name: (len(name), name))
