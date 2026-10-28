"""One-time cleanup of old request bodies in state.sqlite, as one module.

Older versions stored the whole request in ``slots.client_meta_json``. Ended (COLD) slots no longer need
it: only the identity keys matter. This module reduces every COLD row's stored value to those keys, then
rebuilds the database file so the freed space goes back to the filesystem. It is run by hand, once, with the
manager stopped, through ``tools/state_db_cleanup.py``; the manager never imports it.
"""

from __future__ import annotations

from .cli import main, parse_args
from .config import Config, Stop
from .sequence import run
from .world import Env

__all__ = ["Config", "Env", "Stop", "main", "parse_args", "run"]
