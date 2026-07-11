"""Unit tests for kv_policy.py — admission-size based restore (NO hash gates)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from turbohaul.kv_policy import (
    resolve_kv, KVDecision,
    kv_save_fn, kv_meta_fn,
    compute_ctx_len,
    _prefix_hash_chain, _is_prefix_match,
)


def test_compute_ctx_len_basic():
    """Sum of content chars across str-content messages."""
    assert compute_ctx_len([{'role': 'user', 'content': 'hello'},
                            {'role': 'assistant', 'content': 'hi!'}]) == 8


def test_compute_ctx_len_none_safe():
    """Edge case: content=null (tool-call turns) must NOT crash."""
    msgs = [{'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x'}]},
            {'role': 'user', 'content': 'abc'}]
    assert compute_ctx_len(msgs) == 3  # None contributes 0, no TypeError


def test_compute_ctx_len_list_content_safe():
    """Multimodal/list content coerced via str(), never raises."""
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]
    assert compute_ctx_len(msgs) == len(str([{'type': 'text', 'text': 'hi'}]))


def test_compute_ctx_len_empty_and_none():
    """Empty / None message list -> 0, no crash."""
    assert compute_ctx_len([]) == 0
    assert compute_ctx_len(None) == 0


def test_compute_ctx_len_non_dict_message():
    """Non-dict message coerced via str()."""
    assert compute_ctx_len(['raw']) == len("raw")


def test_save_with_identity():
    d = resolve_kv('save', {'thread_id': 'agent-ip-1.2.3.4', 'model_tag': 'm'}, {'saved_tokens': 100})
    assert d.do_it and d.action == 'save'


def test_save_without_identity():
    d = resolve_kv('save', {'thread_id': '', 'model_tag': 'm'}, {'saved_tokens': 100})
    assert not d.do_it


def test_save_zero_tokens():
    d = resolve_kv('save', {'thread_id': 't', 'model_tag': 'm'}, {'saved_tokens': 0})
    assert not d.do_it


# --- PREFIX CLASSIFIER: the restore gate is now prefix-VALIDITY, not
# length. These four tests REPLACE the earlier length-gate tests (restore-extension
# / restore-equal / restore-compaction / admission-size-drives) which exercised the
# `incoming_len < saved_len -> discard` rule the classifier removed at the
# :190 seam. saved_len/incoming_len are still supplied (the inc_len==0 fail-safe reads
# them) but no longer GATE the decision. -------------------------------------------
_SYS = {'role': 'system', 'content': 'you are a helpful assistant with a long prompt'}
_U1 = {'role': 'user', 'content': 'first user turn'}
_A1 = {'role': 'assistant', 'content': 'first answer'}
_U2 = {'role': 'user', 'content': 'second user turn extends the context'}


def test_restore_prefix_valid_extension():
    """WS2 (was restore-extension): saved_chain is a strict prefix of incoming
    (incoming EXTENDS it) -> RESTORE, resolved_from='restore-prefix-valid'."""
    saved = _prefix_hash_chain([_SYS, _U1])
    incoming = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 1500,
        'saved_thread_id': 't', 'saved_chain': saved, 'incoming_chain': incoming,
    })
    assert d.do_it and d.resolved_from == 'restore-prefix-valid'


def test_restore_prefix_valid_equal():
    """WS2 (was restore-equal): saved_chain == incoming_chain (a valid prefix covers
    the equal case) -> RESTORE, resolved_from='restore-prefix-valid'."""
    chain = _prefix_hash_chain([_SYS, _U1])
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 1000,
        'saved_thread_id': 't', 'saved_chain': chain, 'incoming_chain': list(chain),
    })
    assert d.do_it and d.resolved_from == 'restore-prefix-valid'


def test_restore_diverged_fresh():
    """WS2 (replaces restore-compaction): incoming DIVERGES from saved before the
    saved chain ends (e.g. the harness rewrote/summarized an early turn) -> NOT a
    prefix -> FRESH, resolved_from='restore-diverged-fresh'. Same number of turns,
    different content at turn 1, so the physics belt does not fire first."""
    saved = _prefix_hash_chain([_SYS, _U1])
    incoming = _prefix_hash_chain([_SYS, {'role': 'user', 'content': 'REWRITTEN turn'}])
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 990,
        'saved_thread_id': 't', 'saved_chain': saved, 'incoming_chain': incoming,
    })
    assert not d.do_it and d.resolved_from == 'restore-diverged-fresh'


def test_restore_physics_belt_saved_longer():
    """WS2 PHYSICS BELT: saved has MORE turns than incoming (a bin longer than the
    incoming would CLEAR + reprefill) -> FRESH, never restored. This is the length-
    shrink ('compaction') case reframed as turns: the extra saved tail cannot be a
    prefix of the shorter incoming."""
    saved = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    incoming = _prefix_hash_chain([_SYS, _U1])
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 500,
        'saved_thread_id': 't', 'saved_chain': saved, 'incoming_chain': incoming,
    })
    assert not d.do_it and d.resolved_from == 'restore-physics-belt-saved-longer'


def test_restore_empty_incoming_chain_skips():
    """WS2 fail-safe: a restore request with a matching thread_id + size but NO
    admission chain (older submit path / monolithic client) cannot be validated as a
    prefix -> SKIP (never restore blindly). Preserves the fail-safe posture."""
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 1500,
        'saved_thread_id': 't', 'saved_chain': _prefix_hash_chain([_SYS, _U1]),
        'incoming_chain': [],
    })
    assert not d.do_it and d.resolved_from == 'restore-no-incoming-chain'


def test_restore_owner_mismatch():
    """Owner mismatch: thread_id differs from saved → REJECT."""
    d = resolve_kv('restore', {'thread_id': 'agent-ip-1.2.3.4', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 1500,
        'saved_thread_id': 'agent-ip-5.6.7.8'
    })
    assert not d.do_it and d.resolved_from == 'restore-owner-mismatch'


def test_restore_no_incoming_size_skips():
    """Hardening: inc_len=0 (no admission size recorded, e.g.
    a wiring miss) -> SKIP, never optimistically restore. Fails safe regardless of
    cache freshness."""
    fresh = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 0,
        'saved_thread_id': 't', 'cache_age_s': 60
    })
    assert not fresh.do_it and fresh.resolved_from == 'restore-no-incoming-size'
    stale = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 0,
        'saved_thread_id': 't', 'cache_age_s': 999999
    })
    assert not stale.do_it and stale.resolved_from == 'restore-no-incoming-size'


def test_restore_no_data():
    """saved_tokens=0 → REJECT."""
    d = resolve_kv('restore', {'thread_id': 't', 'model_tag': 'm'}, {
        'saved_tokens': 0, 'saved_len': 1000, 'incoming_len': 1500,
        'saved_thread_id': 't'
    })
    assert not d.do_it and d.resolved_from == 'restore-no-data'


def test_restore_no_identity():
    """Empty thread_id → REJECT."""
    d = resolve_kv('restore', {'thread_id': '', 'model_tag': 'm'}, {
        'saved_tokens': 100, 'saved_len': 1000, 'incoming_len': 1500,
        'saved_thread_id': 't'
    })
    assert not d.do_it and d.resolved_from == 'restore-no-identity'


def test_filename_round_trip():
    """CRITICAL: meta path derived from bin path must match kv_meta_fn output.
    Round-trip safe regardless of port/hash layout."""
    bin_fn = kv_save_fn('model-27b', 0, 'c0c7d0e9a892e790', port=11500)
    meta_from_bin = bin_fn[:-4] + '.json'
    meta_from_save = kv_meta_fn('model-27b', 0, 'c0c7d0e9a892e790', port=11500)
    assert meta_from_bin == meta_from_save, (
        f"ROUND-TRIP BREAK: restore looks for '{meta_from_bin}' "
        f"but save wrote '{meta_from_save}'"
    )


def test_filename_round_trip_no_port():
    """Round-trip must also work with default port=0."""
    bin_fn = kv_save_fn('model', 1, 'abc123')
    meta_from_bin = bin_fn[:-4] + '.json'
    meta_from_save = kv_meta_fn('model', 1, 'abc123')
    assert meta_from_bin == meta_from_save


def test_restore_prefix_validity_drives_decision():
    """WS2 (replaces admission-size-drives): prefix VALIDITY (not char length) drives
    RESTORE-vs-FRESH. A genuine swap-back extension whose saved chain is a valid
    prefix of incoming RESTOREs even though this is a large context; a request whose
    early turns diverge is FRESH even if it is 'bigger'."""
    tid = 'agent-ip-192.168.1.10'
    base = [_SYS, _U1, _A1]
    # Real extension: incoming extends the saved chain -> RESTORE (prefix-valid)
    ext = resolve_kv('restore', {'thread_id': tid, 'model_tag': 'm'}, {
        'saved_tokens': 153953, 'saved_len': 434593, 'incoming_len': 436000,
        'saved_thread_id': tid,
        'saved_chain': _prefix_hash_chain(base),
        'incoming_chain': _prefix_hash_chain(base + [_U2]),
    })
    assert ext.do_it and ext.resolved_from == 'restore-prefix-valid'
    # Diverged early turn: NOT a prefix -> FRESH regardless of size
    div = resolve_kv('restore', {'thread_id': tid, 'model_tag': 'm'}, {
        'saved_tokens': 153953, 'saved_len': 434593, 'incoming_len': 500000,
        'saved_thread_id': tid,
        'saved_chain': _prefix_hash_chain(base),
        'incoming_chain': _prefix_hash_chain(
            [_SYS, {'role': 'user', 'content': 'summarized/rewritten'}, _A1, _U2]),
    })
    assert not div.do_it and div.resolved_from == 'restore-diverged-fresh'


def test_prefix_hash_chain_empty():
    """Empty/None context -> [], no crash."""
    assert _prefix_hash_chain([]) == []
    assert _prefix_hash_chain(None) == []


def test_prefix_hash_chain_determinism():
    """Determinism: same input -> same chain, every time."""
    ctx = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi there'}]
    assert _prefix_hash_chain(ctx) == _prefix_hash_chain(ctx)


def test_prefix_hash_chain_content_type_stability():
    """str / list(multimodal) / tool-call(content=None) / non-dict turn
    must all hash without raising."""
    str_ctx = [{'role': 'user', 'content': 'plain string'}]
    assert len(_prefix_hash_chain(str_ctx)) == 1

    list_ctx = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]
    assert len(_prefix_hash_chain(list_ctx)) == 1

    tool_ctx = [{'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x'}]}]
    assert len(_prefix_hash_chain(tool_ctx)) == 1

    # Non-dict turn must NOT raise (manager's pre-refactor copy would
    # AttributeError here on turn.get(); the canonical copy keeps kv_policy's safety).
    non_dict_ctx = ['raw string turn', {'role': 'user', 'content': 'ok'}]
    chain = _prefix_hash_chain(non_dict_ctx)
    assert len(chain) == 2


def test_prefix_hash_chain_known_vector_pin():
    """Pin a fixed expected SHA for a small known context, so a future
    delimiter/format change is caught by this test."""
    pin_ctx = [{'role': 'user', 'content': 'pin'}]
    assert _prefix_hash_chain(pin_ctx) == [
        '68dfb0701a9d1e696318cdbb8066889fd78e208185765055ea15f4516a27d8cb',
    ]


def test_prefix_hash_chain_parity_with_pre_refactor_manager():
    """The unified _prefix_hash_chain must reproduce manager.py's
    PRE-REFACTOR output bit-for-bit for a representative context (str + list/
    multimodal content), proving zero meta-format regression for the live save
    path. Expected chain hardcoded from the pre-edit manager._prefix_hash_chain
    (\\x00 delimiter + json.dumps(sort_keys=True, separators=(',', ':')) for list
    content)."""
    ctx = [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'hi there'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'multimodal bit'}]},
    ]
    assert _prefix_hash_chain(ctx) == [
        '8d8900f2502929edcec0d67942dfb597d91bcf536e86a103029d932267d706d3',
        'a163094adce482cb90c2d8e675be2aa3aefe43b88e6cc73b9ad0cb758c65f1ca',
        '4f510d12b3ea4e47d92f96d7791cc6d491192f26e2fabb9308bf457e994877ee',
    ]


def test_is_prefix_match_saved_equals_incoming():
    chain = _prefix_hash_chain([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}])
    assert _is_prefix_match(chain, chain) is True


def test_is_prefix_match_incoming_extends_saved():
    """Strict extension: saved=2 turns, incoming=3, first 2 match -> True."""
    saved = _prefix_hash_chain([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}])
    incoming = _prefix_hash_chain([
        {'role': 'user', 'content': 'a'},
        {'role': 'assistant', 'content': 'b'},
        {'role': 'user', 'content': 'c'},
    ])
    assert _is_prefix_match(saved, incoming) is True


def test_is_prefix_match_mismatch():
    saved = _prefix_hash_chain([{'role': 'user', 'content': 'a'}])
    incoming = _prefix_hash_chain([{'role': 'user', 'content': 'different'}])
    assert _is_prefix_match(saved, incoming) is False


def test_is_prefix_match_saved_longer_than_incoming():
    saved = _prefix_hash_chain([
        {'role': 'user', 'content': 'a'},
        {'role': 'assistant', 'content': 'b'},
    ])
    incoming = _prefix_hash_chain([{'role': 'user', 'content': 'a'}])
    assert _is_prefix_match(saved, incoming) is False


def test_is_prefix_match_empty_saved():
    incoming = _prefix_hash_chain([{'role': 'user', 'content': 'a'}])
    assert _is_prefix_match([], incoming) is False


if __name__ == '__main__':
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            print(f'  PASS: {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL: {t.__name__}: {e}')
            traceback.print_exc()
            failed.append(t.__name__)
    print(f'\n{passed}/{len(tests)} passed')
    if failed:
        print(f'Failed: {", ".join(failed)}')
    sys.exit(0 if passed == len(tests) else 1)