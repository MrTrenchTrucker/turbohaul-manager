"""Which resident serves a request handle, and the spawn counter that goes with it."""

from __future__ import annotations

from .identity_host import ResidentIdentityHost, ResidentLike


def serving_resident(host: ResidentIdentityHost, handle: object | None) -> ResidentLike | None:
    """The first resident of the host's snapshot whose ``handle`` is ``handle``, or None.

    The match is by identity, not equality. A None handle matches nothing, not even a resident
    that has no handle yet. The snapshot already leaves out the legacy singleton and dead
    residents, so a request on one of those finds no owner. Read-only.
    """
    if handle is None:
        return None
    return next((r for r in host._model_residents() if r.handle is handle), None)


def spawn_seq_for_handle(host: ResidentIdentityHost, handle: object | None, model_tag: str | None) -> int:
    """The spawn counter of the resident serving ``handle``.

    When no resident owns the handle (the single-engine path, or a resident already gone) the
    answer is the host's counter for ``model_tag``, the value the single-engine path uses.
    """
    serving = serving_resident(host, handle)
    if serving is not None:
        return serving.spawn_seq
    return host._spawn_seq_for_model(model_tag)
