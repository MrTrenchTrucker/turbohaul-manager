"""Hard RAM cap for the slot KV save directory, as one module.

The save path declines a snapshot when the directory is still at or over
``ram_cache_max_bytes`` after the GC has had a chance to make room. The GC evicts by tier:
bins of a live turn (HARD) are never evicted, idle-held bins (SOFT) can be reclaimed.

The module is pure policy. The caller measures the directory, gathers the live thread
state, drives the GC and does all the logging. This module decides what the numbers mean.
It imports nothing from turbohaul, uses only the standard library and logs nothing.
"""

from __future__ import annotations

from .accounting import dir_accounting
from .gate import SaveGate, cap_exceeded, enforce_save_cap
from .protection import ProtectionSplit, split_protection
from .selection import soft_age_targets, soft_ceiling_order

__all__ = [
    "dir_accounting",
    "ProtectionSplit",
    "split_protection",
    "cap_exceeded",
    "SaveGate",
    "enforce_save_cap",
    "soft_age_targets",
    "soft_ceiling_order",
]
