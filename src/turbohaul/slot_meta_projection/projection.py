"""The identity projection for writes and the census projection for the boot read."""

from __future__ import annotations

import json
from typing import Any

# The keys worth keeping in the state database for a slot. Everything else in the client
# metadata (messages, tools, sampling options) is request payload, needed only while the
# request runs, and lives in memory.
IDENTITY_KEYS: tuple[str, ...] = (
    "ip",
    "session_id",
    "role",
    "is_main",
    "is_sub_agent",
    "is_curator",
    "is_compression",
    "save_kv",
)

# The keys the boot census reads from a stored body: the address, and the labels that decide
# a request's class. The order is the order of the values in the projected array.
CENSUS_KEYS: tuple[str, ...] = ("ip", "is_curator", "is_compression", "is_sub_agent", "is_main", "role")

_PATHS = ", ".join(f"'$.{k}'" for k in CENSUS_KEYS)

# One typed JSON array per row, built by SQLite from the stored body, so the body itself never
# reaches the caller. A body that is not valid JSON gives NULL: without the guard, one such row
# would fail the whole statement ("malformed JSON").
CENSUS_SLOTS_SQL = (
    "SELECT slot_id, thread_id, model_tag, created_at, "
    f"CASE WHEN json_valid(client_meta_json) THEN json_extract(client_meta_json, {_PATHS}) END "
    "FROM slots"
)


def identity_client_meta(client_meta: Any) -> dict:
    """Return the identity keys of ``client_meta`` that are present, and nothing else.

    A value that is not a dict gives an empty dict. The input is not changed.
    """
    if not isinstance(client_meta, dict):
        return {}
    return {k: client_meta[k] for k in IDENTITY_KEYS if k in client_meta}


def parse_census_meta(projected: str | None) -> dict:
    """Rebuild the small metadata dict the census needs from one projected array.

    ``projected`` is the text SQLite returned for the census projection. A NULL or empty value,
    text that is not JSON, or an array of the wrong length gives an empty dict, which the
    census treats as a row with no address. A null inside the array is a key that was absent
    or null in the body, and is left out, so a later ``.get`` sees the same result as on the full body.
    """
    if not projected:
        return {}
    try:
        values = json.loads(projected)
    except (TypeError, ValueError):
        return {}
    if not isinstance(values, list) or len(values) != len(CENSUS_KEYS):
        return {}
    return {k: v for k, v in zip(CENSUS_KEYS, values) if v is not None}


def strip_stored_meta(stored: str | None) -> str | None:
    """Return the identity-only JSON text for a stored slot body, or None when it cannot be reduced.

    ``stored`` is the text of the ``client_meta_json`` column. It must parse as a JSON object. A NULL
    value, text that is not JSON, or JSON that is not an object gives None, so the caller leaves the
    row as it is. A body that is already identity-only comes back byte-identical, so the caller can
    compare the result with what it read and skip the write.
    """
    if not stored:
        return None
    try:
        body = json.loads(stored)
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return json.dumps(identity_client_meta(body))
