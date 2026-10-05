"""Isolating arm for the _bin_role literal-'main' forgery.

WHAT THIS PINS
--------------
_bin_role's docstring states an invariant:

    "A literal client_meta['role'] string is honored for NON-main roles only —
     'main' is reachable ONLY via an explicit is_main label, NEVER by default or
     by a bare literal."

On the base commit that invariant is FALSE. _bin_role delegates to
_class_from_label FIRST, and _class_from_label's own back-compat tail resolves a
literal ``role`` that names a known class:

    if labels.get("is_main"):   -> False for {"is_main": False}, falls through
    role = labels.get("role")
    if role in POLICIES:        -> "main" IS in POLICIES
        return role             -> "main"

_bin_role then returns that at ``if role is not None: return str(role)``, so its
own anti-literal guard (``if r and str(r) != "main"``) is UNREACHABLE for exactly
the input it was written to stop. The guard is dead code and the docstring is a
comment rather than an enforced rule.

WHY IT MATTERS (the destination, not just the predicate)
--------------------------------------------------------
_bin_role feeds _bin_identity, which is the bin key: one KV copy per
(session_id, role). A request that can forge role="main" therefore lands on the
SAME bin as the session's real main context — it can restore from it, overwrite
it, or reset it. The harm is a bin collision, so this file asserts the collision
directly (test_forged_main_does_not_collide_with_real_main_bin), not only the
string _bin_role returns.

NON-VACUITY
-----------
Every assertion here is expected to FAIL on the base commit. If this file passes
before the fix is applied, the arm is inert and proves nothing — re-derive it.
The fail-on-base run is recorded alongside the change that fixes it.
"""

import pytest

from turbohaul.manager import _bin_identity, _bin_role

SESSION = "sess-forgery-arm-01"
THREAD = "thread-forgery-arm-01"


# --------------------------------------------------------------------------
# The forgery: a literal role="main" must never resolve to main.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "client_meta, why",
    [
        ({"role": "main"}, "bare literal, no booleans at all"),
        ({"role": "main", "is_main": False}, "literal PLUS an explicit denial"),
        ({"role": "main", "is_main": None}, "literal plus a null label"),
        ({"role": "main", "is_main": 0}, "literal plus a falsy-but-present label"),
        ({"role": "MAIN"}, "case variant of the literal"),
    ],
)
def test_literal_role_main_never_resolves_to_main(client_meta, why):
    """A caller that merely SAYS main does not get main. Unlabeled -> None,
    which means raw thread_id keying and therefore no access to the main bin."""
    assert _bin_role(client_meta) != "main", (
        f"forged main accepted ({why}): {client_meta!r}"
    )


def test_forged_main_does_not_collide_with_real_main_bin():
    """The harm, asserted at the destination. A forged-main request and the
    session's genuine main must not key the same bin, or the forger can read,
    overwrite, or reset the real main KV copy."""
    real_main = _bin_identity(THREAD, {"session_id": SESSION, "is_main": True})
    forged = _bin_identity(THREAD, {"session_id": SESSION, "role": "main"})
    assert forged != real_main, (
        f"bin collision: forged request keys the real main bin {real_main!r}"
    )


# --------------------------------------------------------------------------
# The other side of the fix: it must not over-correct.
# --------------------------------------------------------------------------

def test_explicit_is_main_still_resolves_to_main():
    """The one legitimate road to main stays open."""
    assert _bin_role({"is_main": True}) == "main"
    assert _bin_role({"role": "main", "is_main": True}) == "main"


@pytest.mark.parametrize(
    "client_meta, expected",
    [
        ({"is_curator": True}, "curator"),
        # NOTE the hyphen: CLASS_SUB_AGENT == "sub-agent", not "sub_agent".
        ({"is_sub_agent": True}, "sub-agent"),
        ({"is_compression": True}, "compression"),
        # A boolean label outranks a literal 'role' in BOTH directions: the
        # documented priority runs over the is_* booleans, and the literal is
        # only _class_from_label's back-compat tail.
        ({"role": "main", "is_curator": True}, "curator"),
        ({"role": "curator", "is_main": True}, "main"),
    ],
)
def test_label_priority_is_unchanged(client_meta, expected):
    """Fixing the main-literal must not disturb label-derived resolution."""
    assert _bin_role(client_meta) == expected


def test_non_main_literals_are_still_honored():
    """Back-compat: only the 'main' literal is untrusted; the others still work.

    Uses the CANONICAL spellings — the ones that are actually keys in POLICIES.
    """
    assert _bin_role({"role": "curator"}) == "curator"
    assert _bin_role({"role": "sub-agent"}) == "sub-agent"


def test_unrecognized_literal_role_is_passed_through_verbatim():
    """CHARACTERIZATION, not an endorsement — pins today's behavior so a change
    to it is visible rather than silent.

    A literal ``role`` that is NOT a POLICIES key falls past _class_from_label
    (which returns None) into _bin_role's tail, which returns the raw string.
    That string then becomes a bin key. Consequence: the underscore spelling
    "sub_agent" and the canonical "sub-agent" are DIFFERENT bins for the same
    session, and the former has no POLICIES entry behind it. Whether the harness
    can actually emit such a spelling is an open question for the maintainers; this
    test only records that the pass-through exists.
    """
    assert _bin_role({"role": "sub_agent"}) == "sub_agent"
    assert _bin_role({"role": "sub_agent"}) != _bin_role({"is_sub_agent": True})


@pytest.mark.parametrize("client_meta", [None, {}, {"session_id": SESSION}])
def test_unlabeled_stays_none(client_meta):
    """No labels -> None -> raw thread_id keying, today's behavior, unchanged."""
    assert _bin_role(client_meta) is None
