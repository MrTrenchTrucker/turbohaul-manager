"""Which bins the GC may take, as plain lists.

The GC's own state (its candidate list, the entries it scanned, the hash of each path, the
hash sets of each tier, the pinned paths) is passed in. The unlink loop stays with the GC
in the caller, so the caller keeps the order and the logging. These builders only decide
the list.
"""

from __future__ import annotations


def soft_age_targets(candidates, entries, hash_by_path, soft_hashes) -> list:
    """Bins the age prune may reclaim: the unprotected candidates, then the SOFT-thread bins.

    Candidates come first, then SOFT entries in ``entries`` order. An entry that is both a
    candidate and SOFT appears twice. The list is not de-duplicated, so the order the
    caller walks stays the same.
    """
    return candidates + [e for e in entries if hash_by_path.get(e[1]) in soft_hashes]


def soft_ceiling_order(surviving, deleted_paths, hard_hashes, soft_hashes, pinned_paths,
                       hash_by_path) -> list:
    """SOFT bins the ceiling prune may take once every unprotected bin has gone, oldest first.

    An entry qualifies when it is still on disk, its thread is not HARD, and either it is a
    pinned path or its thread is SOFT. Oldest first means the bin idle the longest goes first.
    The inputs are not changed.
    """
    picked = [e for e in surviving
              if e[1] not in deleted_paths
              and hash_by_path.get(e[1]) not in hard_hashes
              and (e[1] in pinned_paths or hash_by_path.get(e[1]) in soft_hashes)]
    picked.sort(key=lambda x: x[2])
    return picked
