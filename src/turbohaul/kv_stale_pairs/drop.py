"""Find and remove the pairs a thread left on other ports, once its new pair has landed."""

from __future__ import annotations

import contextlib
import os
import re

from .model import DropResult, StalePair

_LANDED_MISMATCH = "landed_size_mismatch"
_SERVED_UNKNOWN = "served_unknown"
_DIR_MISSING = "dir_missing"
_SCAN_FAILED = "scan_failed"


def _is_int(value) -> bool:
    """True for a real int. A bool is not an int here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _landed_ok(landed_bytes, n_written) -> bool:
    """The new pair is proven when the file size and the engine's byte count are the same positive int."""
    return _is_int(landed_bytes) and _is_int(n_written) and landed_bytes == n_written and landed_bytes > 0


def _served_ports(live_turns, thread_hash):
    """The ports a live turn serves this thread on, or None when that cannot be known.

    ``live_turns`` is None (the caller's scan failed) or a sequence of ``(port, slots)``, where
    ``slots`` holds one collection of thread hashes per live turn of that engine process. An entry
    serves the thread when any of its collections holds ``thread_hash``. A serving entry without a
    usable port makes every port unknown.
    """
    if live_turns is None:
        return None
    ports: set[int] = set()
    for port, slots in live_turns:
        if not any(thread_hash in hashes for hashes in slots):
            continue
        if not _is_int(port):
            return None
        ports.add(port)
    return ports


def _scan_bins(save_dir: str) -> list[tuple[str, int]]:
    """``(bin name, bin bytes + .ckpt bytes)`` for each regular ``.bin`` file, from one directory scan."""
    sizes: dict[str, int] = {}
    with os.scandir(save_dir) as entries:
        for entry in entries:
            try:
                if entry.is_file():
                    sizes[entry.name] = entry.stat().st_size
            except OSError:
                continue
    return [(name, size + sizes.get(name + ".ckpt", 0)) for name, size in sizes.items() if name.endswith(".bin")]


def _stale_pairs(bins, model_tag: str, thread_hash: str, new_port: int, served: set[int]) -> list[StalePair]:
    """The thread's bins on a port that is neither the new one nor a served one, sorted by name.

    Only an exact match on both the model tag and the thread hash counts, so another thread, another
    model, or a tag that merely starts with this one is never returned.
    """
    pattern = re.compile(
        r"^" + re.escape(model_tag) + r"\.p(\d+)\." + re.escape(thread_hash) + r"\.slot(\d+)\.bin$")
    found = []
    for name, pair_bytes in bins:
        match = pattern.match(name)
        if match is None:
            continue
        port = int(match.group(1))
        if port == new_port or port in served:
            continue
        found.append(StalePair(name, port, int(match.group(2)), pair_bytes))
    return sorted(found, key=lambda pair: pair.name)


def _remove_pair(save_dir: str, name: str, remove) -> bool:
    """Remove the bin, then its checkpoint, then its metadata. True when the bin removal did not raise.

    A failure on one target is ignored for that target and the next one is still tried.
    """
    base = os.path.join(save_dir, name)
    bin_gone = False
    for target in (base, base + ".ckpt", base[:-4] + ".json"):
        with contextlib.suppress(Exception):
            remove(target)
            bin_gone = bin_gone or target == base
    return bin_gone


def drop_stale_pairs(save_dir, model_tag, thread_hash, new_port, live_turns, landed_bytes, n_written,
                     remove=os.remove) -> DropResult:
    """Remove the pairs ``thread_hash`` left on ports other than ``new_port``, and report what went.

    Nothing is scanned or removed unless the new pair is proven (``landed_bytes`` and ``n_written`` are
    the same positive int) and the ports a live turn serves for this thread are known. A pair on a
    served port stays. Never raises: any unexpected failure becomes a ``scan_failed`` result.
    """
    def _result(dropped=(), skipped=None):
        return DropResult(tuple(dropped), skipped, landed_bytes, n_written)

    try:
        if not _landed_ok(landed_bytes, n_written):
            return _result(skipped=_LANDED_MISMATCH)
        served = _served_ports(live_turns, thread_hash)
        if served is None:
            return _result(skipped=_SERVED_UNKNOWN)
        if not os.path.isdir(save_dir):
            return _result(skipped=_DIR_MISSING)
        stale = _stale_pairs(_scan_bins(save_dir), model_tag, thread_hash, new_port, served)
        return _result(dropped=[pair for pair in stale if _remove_pair(save_dir, pair.name, remove)])
    except Exception:  # noqa: BLE001 - the contract is "never raises"; the safe answer is keep
        return _result(skipped=_SCAN_FAILED)
