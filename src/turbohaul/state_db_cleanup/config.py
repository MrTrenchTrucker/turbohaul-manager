"""Constants, the stop signal and the run configuration of the state database cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_GIB = 1 << 30
_BODY_COLUMN = "client_meta_json"
_DEFAULT_BATCH = 500
_DEFAULT_SLEEP_MS = 50
_DEFAULT_CHECKPOINT_EVERY = 20
_DEFAULT_MARGIN_BYTES = 5 * _GIB
_MIN_SQLITE = (3, 27, 0)  # VACUUM INTO
_SAMPLE_SIZE = 20
_BODY_CHUNK = 5  # stored values fetched together; they can be hundreds of KB each
_TMP_PREFIX = ".state_cleanup_tmp_"


class Stop(Exception):
    """A gate failed. The run ends here; ``step`` names the gate."""

    def __init__(self, step, message):
        super().__init__(message)
        self.step = step
        self.message = message


@dataclass
class Config:
    db: Path
    backup_dir: Path
    work_dir: Path = None
    execute: bool = False
    swap: bool = True
    min_age_hours: float = 0.0
    batch: int = _DEFAULT_BATCH
    sleep_ms: float = _DEFAULT_SLEEP_MS
    checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY
    margin_bytes: int = _DEFAULT_MARGIN_BYTES
    report_path: Path = None
    lock_timeout_s: float = 1.0
    cutoff_iso: str = None  # fixed once at the start of a run so every pass chooses the same rows
