"""Split the live KV-protection threads into a HARD tier and a SOFT tier.

HARD bins belong to a live turn and are never evicted. SOFT bins are idle-held and may be
reclaimed under pressure. The caller gathers the live thread triples and supplies the hash
and identity functions. This module only decides which tier each triple goes into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class ProtectionSplit:
    """The two protection tiers, as sets of thread hashes."""

    hard: set[str]
    soft: set[str]

    @property
    def protected(self) -> set[str]:
        """Every protected hash, HARD and SOFT together, as a new set."""
        return self.hard | self.soft


def _add_one(tier: set[str], tid: str | None, client_meta, chain,
             thread_hash: Callable[[str], str], bin_identity: Callable[[str, object, object], str]) -> None:
    # A bin can be named by the raw thread id (legacy, before the one-time migration) or by
    # the (session, role) identity. Both hashes go in, so either name keeps the bin protected.
    if tid:
        tier.add(thread_hash(tid))
    ident = bin_identity(tid or "", client_meta, chain)
    if ident and ident != (tid or ""):
        tier.add(thread_hash(ident))


def split_protection(hard_threads: Iterable, soft_threads: Iterable,
                     thread_hash: Callable[[str], str], bin_identity: Callable[[str, object, object], str],
                     on_error: Callable[[], None] | None = None) -> ProtectionSplit:
    """Build the HARD and SOFT hash sets from ``(thread_id, client_meta, chain)`` triples.

    HARD triples are added first, then SOFT, in the order given. Best-effort and never
    raises. If a callable or the iteration fails, ``on_error`` runs inside the except block
    and the split built so far is returned. A partial split is kept on purpose: the protection
    scan is best-effort, and a failed scan must not stop the cap from working.
    """
    hard: set[str] = set()
    soft: set[str] = set()
    try:
        for tid, client_meta, chain in hard_threads:
            _add_one(hard, tid, client_meta, chain, thread_hash, bin_identity)
        for tid, client_meta, chain in soft_threads:
            _add_one(soft, tid, client_meta, chain, thread_hash, bin_identity)
    except Exception:
        if on_error is not None:
            on_error()
    return ProtectionSplit(hard, soft)
