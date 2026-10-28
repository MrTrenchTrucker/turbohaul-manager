"""The manager-facing side of the slot KV save directory cap and the stale-pair drop, as one module.

``kv_ram_cap`` and ``kv_stale_pairs`` decide; they log nothing and read no live state. This module
is the glue that the manager calls: it reads the live manager through small protocol views, logs
under the manager's logger name, awaits the GC the cap gate asks for, and hands the decision
modules their inputs. It holds no state and starts no tasks. The manager keeps thin methods that
call these functions and pass in anything it looks up in its own module at call time.
"""

from __future__ import annotations

from .live_state import live_kv_protection, live_turn_groups
from .save_path import (
    dir_accounting_logged,
    drop_stale_pairs_after_save,
    evict_soft_floor,
    save_declined_by_cap,
)

__all__ = [
    "dir_accounting_logged",
    "live_turn_groups",
    "live_kv_protection",
    "save_declined_by_cap",
    "drop_stale_pairs_after_save",
    "evict_soft_floor",
]
