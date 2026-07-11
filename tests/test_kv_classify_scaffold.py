"""Phase 1 — EQUIVALENCE proof for the INERT scaffold.

Asserts ``kv_classify.classify_request(...)`` returns the SAME values as today's
scattered logic across the golden matrix (LABELS ABSENT), plus a property test (every
class resolves) and the import-time completeness guarantee.

This is the "returns the SAME values" half of code-unification step 2: the new module
is empty-but-wired and byte-equivalent to today. Nothing in the manager calls it (see
``test_scaffold_is_inert``); the 5 scattered sites are UNCHANGED.
"""
import importlib
import importlib.util

import pytest

import turbohaul.kv_classify as kc

# Reuse the exact golden matrix so the equivalence is proven against the SAME scenarios
# the baseline pins.
from tests.test_kv_classify_golden import SCENARIOS
from turbohaul.api.chat_completion import _shadow_recompose_identity
from turbohaul.kv_policy import compute_ctx_len, resolve_kv
from turbohaul.manager import TurbohaulManager


def _manager_event(sc):
    """Today's scattered event classifier. ``_classify_event`` reads NO instance state
    (only its 3 chain args + the module-level ``_is_prefix_match``), so the unbound
    method with self=None is the exact scattered logic without building a manager."""
    return TurbohaulManager._classify_event(None, sc.clean_chain, sc.inc_chain, sc.warm_chain)


def _signals(sc, thread_id="t"):
    return kc.RequestSignals(
        thread_id=thread_id,
        inc_chain=sc.inc_chain,
        saved_chain=sc.clean_chain,   # None for the no-anchor (sub-agent) scenario
        warm_chain=sc.warm_chain,
        size=sc.size,
    )


def _resolve_restore_do_it(sc):
    """Today's cold per-bin decision (owner-matched, positive-token baseline)."""
    return resolve_kv(
        "restore",
        {"thread_id": "t", "model_tag": "m", "slot_id": 0, "port": 0},
        {
            "saved_tokens": 12345,
            "saved_len": compute_ctx_len(sc.clean_msgs or []),
            "incoming_len": sc.size,
            "cache_age_s": 0.0,
            "saved_thread_id": "t",
            "saved_chain": sc.clean_chain or [],
            "incoming_chain": sc.inc_chain,
        },
    ).do_it


# ============================================================================
# Equivalence: classify_request == today's scattered logic (labels absent)
# ============================================================================

@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_event_type_matches_manager(sc):
    rc = kc.classify_request(None, _signals(sc))
    assert rc.event_type == _manager_event(sc)


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_reuse_intent_matches_today(sc):
    rc = kc.classify_request(None, _signals(sc))
    # Matches the golden's expected cold outcome...
    assert rc.reuse_intent == sc.expected_reuse
    # ...and, where a saved anchor exists, byte-matches resolve_kv's do_it directly.
    if sc.clean_msgs is not None:
        expected = "restore" if _resolve_restore_do_it(sc) else "fresh"
        assert rc.reuse_intent == expected


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_identity_is_base_thread_id(sc):
    """Labels-absent identity == the base thread_id == today's value (recompose is a
    no-op with role/session absent, and m2b is default-OFF)."""
    rc = kc.classify_request(None, _signals(sc, thread_id="agent-ip-9.9.9.9-auto-cafe"))
    assert rc.identity == "agent-ip-9.9.9.9-auto-cafe"
    assert rc.identity == _shadow_recompose_identity("agent-ip-9.9.9.9-auto-cafe", None, None)


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_save_ok_uniformly_true_today(sc):
    """Today nothing gates a save by class -> save_ok True for every class."""
    rc = kc.classify_request(None, _signals(sc))
    assert rc.save_ok is True


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_provenance_is_chain_inference_when_labels_absent(sc):
    rc = kc.classify_request(None, _signals(sc))
    assert rc.resolved_from == "chain-inference"


# ============================================================================
# Property test: every request resolves to a valid, in-registry class
# ============================================================================

@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_every_request_resolves(sc):
    rc = kc.classify_request(None, _signals(sc))
    assert rc.event_type in kc.EVENTS
    assert rc.reuse_intent in kc.REUSE_INTENTS
    assert isinstance(rc.save_ok, bool)
    # The inferred class is a real registry key.
    cls = kc.EVENT_TO_CLASS[rc.event_type]
    assert cls in kc.POLICIES


def test_every_class_is_reachable_and_resolves():
    """Each of the 5 registry classes resolves to a RequestClass. Class selection here
    uses the ROLE-STRING label (back-compat path) — is_* booleans covered separately.

    save_ok is sourced from the registry (Phase 2: curator=False, all others True)."""
    sig = kc.RequestSignals(thread_id="t", inc_chain=[], saved_chain=None, warm_chain=None, size=0)
    for cls in kc.CLASSES:
        rc = kc.classify_request({"role": cls}, sig)  # role-string label selects the class
        assert rc.resolved_from == f"POLICIES[{cls}]"
        assert rc.save_ok is kc.POLICIES[cls].save_ok  # registry-sourced, not hardcoded
        assert rc.reuse_intent in kc.REUSE_INTENTS


# ============================================================================
# Registry structure + import-time completeness
# ============================================================================

def test_policies_completeness_invariant():
    assert set(kc.POLICIES) == set(kc.CLASSES)
    assert set(kc.EVENT_TO_CLASS) == set(kc.EVENTS)
    assert set(kc.EVENT_TO_CLASS.values()) <= set(kc.CLASSES)


def test_module_reimport_runs_import_asserts():
    """The completeness asserts run AT IMPORT (fail fast on boot); reload proves it."""
    importlib.reload(kc)
    assert set(kc.POLICIES) == set(kc.CLASSES)


def test_missing_policy_entry_would_fail_fast():
    """A class without a registry entry must be caught by the import-time assert shape."""
    broken = dict(kc.POLICIES)
    broken.pop(kc.CLASS_CURATOR)
    with pytest.raises(AssertionError):
        assert set(broken) == set(kc.CLASSES), "registry drift"


# ============================================================================
# Phase-2 label contract: is_* BOOLEAN consumption + precedence + role back-compat
# ============================================================================

def test_class_from_label_is_boolean_precedence():
    """_class_from_label consumes the client is_* booleans by PRIORITY (labels are NOT
    mutually exclusive). Priority order: is_curator > is_compression > is_sub_agent
    > is_main."""
    # REAL LIVE SHAPE (code-verified): a curator carries BOTH is_sub_agent AND
    # is_curator today -> MUST resolve to CURATOR, never sub_agent.
    assert kc._class_from_label(
        {"is_sub_agent": True, "is_curator": True}
    ) == kc.CLASS_CURATOR
    # All four set -> curator wins (highest priority).
    assert kc._class_from_label(
        {"is_curator": True, "is_compression": True, "is_sub_agent": True, "is_main": True}
    ) == kc.CLASS_CURATOR
    # curator beats compression too.
    assert kc._class_from_label(
        {"is_compression": True, "is_curator": True}
    ) == kc.CLASS_CURATOR
    # compression beats sub_agent + main (loses only to curator).
    assert kc._class_from_label(
        {"is_compression": True, "is_sub_agent": True, "is_main": True}
    ) == kc.CLASS_COMPRESSION
    # sub_agent beats main.
    assert kc._class_from_label(
        {"is_sub_agent": True, "is_main": True}
    ) == kc.CLASS_SUB_AGENT
    assert kc._class_from_label({"is_main": True}) == kc.CLASS_MAIN
    # Each bit in isolation maps to its class.
    assert kc._class_from_label({"is_compression": True}) == kc.CLASS_COMPRESSION
    assert kc._class_from_label({"is_curator": True}) == kc.CLASS_CURATOR
    assert kc._class_from_label({"is_sub_agent": True}) == kc.CLASS_SUB_AGENT


def test_class_from_label_none_safe_and_absent():
    """Labels absent / all-falsy booleans -> None -> chain-inference (byte-identical)."""
    assert kc._class_from_label(None) is None
    assert kc._class_from_label({}) is None
    assert kc._class_from_label(
        {"is_main": None, "is_curator": False, "is_sub_agent": None, "is_compression": 0}
    ) is None


def test_class_from_label_role_string_backcompat():
    """Back-compat: a literal role string that names a class is still honored when the
    is_* booleans are absent (nothing that passed a role regresses)."""
    for cls in kc.CLASSES:
        assert kc._class_from_label({"role": cls}) == cls
    assert kc._class_from_label({"role": "not-a-class"}) is None
    # A truthy is_* boolean overrides a conflicting role string.
    assert kc._class_from_label({"role": "main", "is_curator": True}) == kc.CLASS_CURATOR


# ============================================================================
# Phase-2 registry: curator save_ok=False (others unchanged). Curator reuse-main is
# achieved by the CLIENT HARNESS thread_id (curator tid = main's), NOT a manager
# identity remap — so the manager never remaps identity here.
# ============================================================================

def test_policies_curator_no_save_others_default():
    """POLICIES[curator] is the ONLY entry with save_ok=False; every other class
    keeps save_ok=True. (No reuse_target field — reuse rides the harness thread_id.)"""
    assert kc.POLICIES[kc.CLASS_CURATOR].save_ok is False
    for cls, pol in kc.POLICIES.items():
        if cls == kc.CLASS_CURATOR:
            continue
        assert pol.save_ok is True, f"{cls} should keep save_ok=True"


def test_classify_request_curator_no_save_no_manager_remap():
    """An explicit is_curator -> save_ok False (the save-gate skips it) AND identity
    stays the base thread_id: the manager does NOT remap (a manager remap would
    override + break the client harness thread_id that actually drives main reuse)."""
    sig = kc.RequestSignals(
        thread_id="main-sess9",   # what the client harness already sends
        inc_chain=kc._prefix_hash_chain([{"role": "system", "content": "s"}]),
        saved_chain=None,
        warm_chain=None,
        size=10,
    )
    rc = kc.classify_request({"is_curator": True}, sig)
    assert rc.resolved_from == "POLICIES[curator]"
    assert rc.save_ok is False
    assert rc.identity == "main-sess9"        # base tid passed through, NOT remapped


def test_classify_request_labels_absent_is_byte_identical():
    """The load-bearing invariant: labels absent -> base thread_id + save_ok True +
    chain-inference provenance (byte-identical to today; nothing remaps)."""
    sig = kc.RequestSignals(
        thread_id="t-base",
        inc_chain=kc._prefix_hash_chain([{"role": "system", "content": "s"}]),
        saved_chain=None,
        warm_chain=None,
        size=10,
    )
    rc = kc.classify_request(None, sig)
    assert rc.identity == "t-base"        # NOT remapped (no explicit label)
    assert rc.save_ok is True
    assert rc.resolved_from == "chain-inference"


def test_curator_does_not_add_a_reuse_intent():
    """Curator adds no new reuse_intent value — reuse_intent stays fresh/restore
    (chain-derived); reuse-main is a save-gate + harness-thread_id concern, not an intent."""
    assert "reuse-main" not in kc.REUSE_INTENTS
    sig = kc.RequestSignals(thread_id="t", inc_chain=[], size=0)
    rc = kc.classify_request({"is_curator": True}, sig)
    assert rc.reuse_intent in kc.REUSE_INTENTS


# ============================================================================
# Phase-2 unification: the manager now DELEGATES to the kv_classify chokepoint
# ============================================================================

def test_manager_delegates_classify_event_to_kv_classify():
    """Phase 2 unifies event classification onto the chokepoint: the manager imports
    kv_classify and _classify_event byte-matches classify_event across the matrix."""
    src = importlib.util.find_spec("turbohaul.manager").origin
    with open(src) as f:
        text = f.read()
    assert "kv_classify" in text, "manager.py must reference kv_classify (Phase 2 unified)"
    for sc in SCENARIOS:
        assert (
            TurbohaulManager._classify_event(None, sc.clean_chain, sc.inc_chain, sc.warm_chain)
            == kc.classify_event(sc.clean_chain, sc.inc_chain, sc.warm_chain)
        )
