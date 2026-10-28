#!/usr/bin/env python3
"""One-time cleanup of old request bodies in state.sqlite (a thin launcher for src/turbohaul/state_db_cleanup).

    python3 tools/state_db_cleanup.py --db PATH --backup-dir DIR            # dry run (the default)
    python3 tools/state_db_cleanup.py --db PATH --backup-dir DIR --execute  # the real run

The manager MUST be stopped for ``--execute``. What the tool does, step by step, and what stops it, is in
src/turbohaul/state_db_cleanup/README.md and in docs/STATE_DATABASE_MEMORY.md.

Exit status: 0 done (or dry run complete), 2 stopped by a gate (nothing further was attempted),
1 unexpected error.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from turbohaul.state_db_cleanup import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
