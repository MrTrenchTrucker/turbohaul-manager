"""The frozen wire keys must survive any identifier rename.

WHY THIS TEST EXISTS
--------------------
A substring rename across a wire boundary breaks consumers silently.
The existing suite stays GREEN through exactly that mistake, because these keys are
STRING LITERALS on the wire, not Python identifiers - nothing imports them, nothing
type-checks them, and the frontend that consumes them is not in this test suite.

TWO OF THESE ARE PROVEN FRONTEND-CRITICAL (verified by inspection):
  "likely_victim" -> src/frontend/src/components/Queue.tsx, Queue.test.tsx, api.ts
  "slot_evicted"  -> src/frontend/src/components/Logs.tsx, inside the regex
                     /^safety_gate|slot_evicted/ ; emitted in manager.py
Rename either and the frontend breaks SILENTLY. No backend test would catch it.
That is the failure this file exists to make loud.
"""
import pathlib
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "turbohaul"
QUEUE = SRC / "queue.py"
MANAGER = SRC / "manager.py"

# key -> exact occurrence count across queue.py + manager.py, measured on the current tree.
# A count change is as much a violation as a disappearance: it means the wire moved.
FROZEN_WIRE_KEYS = {
    "likely_victim": 2,                 # FRONTEND-CRITICAL: Queue.tsx, Queue.test.tsx, api.ts
    "slot_evicted": 2,                  # FRONTEND-CRITICAL: Logs.tsx regex
    "evictions": 6,
    "evict_teardown": 4,
    "evicted_pre_fanout": 2,
    "background_sweeper_evicted": 1,
    "grace_designated_victim_skip": 2,
    "last_evicted_at": 1,
    "shadow_evicted": 2,
}

FRONTEND_CRITICAL = ("likely_victim", "slot_evicted")


def _sources() -> str:
    return QUEUE.read_text() + MANAGER.read_text()


@pytest.mark.parametrize("key,expected", sorted(FROZEN_WIRE_KEYS.items()))
def test_frozen_wire_key_unchanged(key, expected):
    """Each frozen key appears, as a quoted literal, exactly as many times as before."""
    actual = _sources().count('"' + key + '"')
    msg = (
        "FROZEN WIRE KEY " + key + " moved: expected " + str(expected)
        + " occurrence(s), found " + str(actual)
        + ". A rename crossed a wire boundary. This key is consumed OUTSIDE this repo's"
        + " test suite; changing it breaks consumers silently."
    )
    if key in FRONTEND_CRITICAL:
        msg += "  *** " + key + " IS FRONTEND-CRITICAL - the UI reads it directly. ***"
    assert actual == expected, msg


def test_frontend_critical_keys_are_still_emitted():
    """The two keys the UI reads must still be emitted by the backend at all."""
    src = _sources()
    for key in FRONTEND_CRITICAL:
        assert '"' + key + '"' in src, (
            "FRONTEND-CRITICAL wire key " + key + " is GONE from queue.py/manager.py. "
            "The UI still reads it. This breaks the frontend with no backend error."
        )
