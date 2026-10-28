"""resident_identity.identity: store, read and forget the last request identity per resident key."""

import logging

import _resident_identity_fakes as fakes

from turbohaul.resident_identity import forget_identity, identity_for, stash_identity

SINGLETON = "__legacy_singleton__"
LOGGER = "turbohaul.resident_identity.identity"


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"resident_identity: {text} (the contract is in src/turbohaul/resident_identity/AGENTS.md, Invariants)"


def stash_quietly(host, r, slot):
    """Call stash_identity and turn an escaping Exception into a plain assertion failure."""
    try:
        return stash_identity(host, r, slot, SINGLETON)
    except Exception as exc:  # noqa: BLE001 -- the point is to name the escape
        raise AssertionError(why(f"stash_identity let {type(exc).__name__} escape, it must never raise")) from exc


def host_with(*residents, store=None):
    host = fakes.FakeHost(residents=residents)
    if store is not None:
        host._last_request_identity_by_model = store
    return host


# -- stash_identity: what is stored ---------------------------------------------------------------

def test_stash_stores_the_built_identity_under_the_resident_key():
    r = fakes.FakeResident(resident_key="m#2", model_tag="m")
    slot = fakes.FakeSlot(client_meta={"ip": "10.1.2.3"}, model_tag="m", thread_id="th", fastlane="fl")
    host = host_with(r)
    assert stash_identity(host, r, slot, SINGLETON) is None, why("stash returns None")
    assert list(host._last_request_identity_by_model) == ["m#2"], why("stored under the resident's own key only")
    assert host._last_request_identity_by_model["m#2"] == {
        "ip": "10.1.2.3", "model_tag": "m", "client_meta": {"ip": "10.1.2.3"}, "thread_id": "th", "fastlane": "fl"}, \
        why("the stored value is what the host built from the slot")


def test_stash_asks_the_host_to_build_the_identity_exactly_once_from_the_slot_fields():
    r = fakes.FakeResident(resident_key="k")
    meta = {"ip": "1.1.1.1", "x": 1}
    slot = fakes.FakeSlot(client_meta=meta, model_tag="other", thread_id="T", fastlane=None)
    host = host_with(r)
    stash_identity(host, r, slot, SINGLETON)
    assert host.called("_identity_dict") == [("_identity_dict", "1.1.1.1", "other", meta, "T", None)], \
        why("one build call carrying ip, model_tag, client_meta, thread_id and fastlane of the slot")


def test_stash_stores_the_object_the_host_built_not_a_copy():
    r = fakes.FakeResident(resident_key="k")
    built = {"marker": 1}
    host = host_with(r)
    host._identity_dict = lambda **kw: built
    stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    assert host._last_request_identity_by_model["k"] is built, why("the built dict itself is stored")


def test_stash_without_client_meta_passes_a_none_ip_and_the_none_meta():
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(client_meta=None), SINGLETON)
    assert host.called("_identity_dict") == [("_identity_dict", None, "m", None, "t1", None)], \
        why("client_meta None gives ip None and is passed on as None")


def test_stash_with_client_meta_lacking_ip_passes_none_ip():
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(client_meta={"agent": "a"}), SINGLETON)
    assert host.called("_identity_dict")[0][1] is None, why("no ip key gives ip None")


def test_stash_overwrites_the_previous_entry_of_the_same_key_only():
    a, b = fakes.FakeResident(resident_key="a"), fakes.FakeResident(resident_key="b")
    host = host_with(a, b)
    stash_identity(host, a, fakes.FakeSlot(thread_id="first"), SINGLETON)
    stash_identity(host, b, fakes.FakeSlot(thread_id="other"), SINGLETON)
    stash_identity(host, a, fakes.FakeSlot(thread_id="second"), SINGLETON)
    store = host._last_request_identity_by_model
    assert store["a"]["thread_id"] == "second", why("a later request replaces the earlier entry of the same key")
    assert store["b"]["thread_id"] == "other", why("a sibling's entry is untouched")


def test_stash_writes_under_the_key_and_not_under_the_model_tag():
    r = fakes.FakeResident(resident_key="m#3", model_tag="m")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(model_tag="m"), SINGLETON)
    assert "m" not in host._last_request_identity_by_model, why("never keyed by the model tag alone")


# -- stash_identity: what is never stored ---------------------------------------------------------

def test_singleton_key_is_never_stored_and_the_store_is_not_touched():
    store = fakes.RecordingDict()
    r = fakes.FakeResident(resident_key=SINGLETON)
    host = host_with(r, store=store)
    stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    assert store.ops == [] and dict(store) == {}, why(f"singleton key left an operation {store.ops}")
    assert host.called("_identity_dict") == [], why("no identity is built for the singleton key")


def test_empty_and_missing_keys_are_never_stored_and_the_store_is_not_touched():
    for key in ("", None):
        store = fakes.RecordingDict()
        r = fakes.FakeResident(resident_key=key)
        host = host_with(r, store=store)
        stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
        assert store.ops == [] and dict(store) == {}, why(f"key {key!r} left an operation {store.ops}")
        assert host.called("_identity_dict") == [], why(f"no identity is built for key {key!r}")


def test_the_singleton_key_is_the_argument_not_a_constant_of_the_module():
    r = fakes.FakeResident(resident_key="alpha")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(), "alpha")
    assert dict(host._last_request_identity_by_model) == {}, why("a key equal to the passed singleton is skipped")
    host2 = host_with(fakes.FakeResident(resident_key=SINGLETON))
    stash_identity(host2, host2.residents[0], fakes.FakeSlot(), "other-singleton")
    assert list(host2._last_request_identity_by_model) == [SINGLETON], \
        why("a key that is not the passed singleton is stored, whatever its text")


def test_a_key_that_merely_contains_the_singleton_text_is_stored():
    r = fakes.FakeResident(resident_key=SINGLETON + "x")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    assert list(host._last_request_identity_by_model) == [SINGLETON + "x"], why("equality, not containment")


# -- stash_identity: best effort -----------------------------------------------------------------

def test_a_failing_identity_build_is_swallowed_and_leaves_the_store_unchanged():
    store = fakes.RecordingDict({"keep": {"v": 1}})
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r, store=store)

    def boom(**kw):
        raise ValueError("cannot build")

    host._identity_dict = boom
    assert stash_quietly(host, r, fakes.FakeSlot()) is None, why("returns None on failure")
    assert dict(store) == {"keep": {"v": 1}}, why("a failed stash leaves the dict unchanged")


def test_a_failing_store_write_is_swallowed():
    store = fakes.RefusingDict({"keep": 1})
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r, store=store)
    assert stash_quietly(host, r, fakes.FakeSlot()) is None, why("a refused write does not raise")
    assert dict(store) == {"keep": 1}, why("the dict is unchanged after a refused write")


def test_a_failing_key_read_is_inside_the_guard():
    class BadResident:
        @property
        def resident_key(self):
            raise RuntimeError("no key")

    host = host_with()
    assert stash_quietly(host, BadResident(), fakes.FakeSlot()) is None, \
        why("the key read is inside the best-effort guard")
    assert dict(host._last_request_identity_by_model) == {}, why("nothing stored")


def test_a_failing_slot_read_is_swallowed():
    class BadSlot:
        client_meta = None
        model_tag = "m"
        thread_id = "t"

        @property
        def fastlane(self):
            raise RuntimeError("no fastlane")

    r = fakes.FakeResident(resident_key="k")
    host = host_with(r)
    assert stash_quietly(host, r, BadSlot()) is None, why("slot attribute failure swallowed")
    assert dict(host._last_request_identity_by_model) == {}, why("nothing stored")


def test_a_swallowed_failure_is_logged_at_debug_with_the_traceback(caplog):
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r)

    def boom(**kw):
        raise ValueError("cannot build")

    host._identity_dict = boom
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        stash_quietly(host, r, fakes.FakeSlot())
    recs = [x for x in caplog.records if x.name == LOGGER]
    assert len(recs) == 1, why(f"exactly one log record on the module logger, got {len(recs)}")
    assert recs[0].levelno == logging.DEBUG, why("logged at DEBUG, not higher")
    assert recs[0].exc_info and recs[0].exc_info[0] is ValueError, why("the traceback is attached")


def test_a_successful_or_skipped_stash_logs_nothing(caplog):
    r = fakes.FakeResident(resident_key="k")
    s = fakes.FakeResident(resident_key=SINGLETON)
    host = host_with(r, s)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
        stash_identity(host, s, fakes.FakeSlot(), SINGLETON)
    assert [x for x in caplog.records if x.name == LOGGER] == [], why("only a failure is logged")


def test_a_non_exception_base_error_is_not_swallowed():
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r)

    def stop(**kw):
        raise KeyboardInterrupt

    host._identity_dict = stop
    try:
        stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    except KeyboardInterrupt:
        return
    raise AssertionError(why("only Exception is swallowed, not KeyboardInterrupt"))


# -- the store is read at call time ----------------------------------------------------------------

def test_a_store_swapped_in_after_construction_is_honoured_by_all_three_functions():
    r = fakes.FakeResident(resident_key="k", model_tag="m")
    host = host_with(r)
    stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    old = host._last_request_identity_by_model
    new = fakes.RecordingDict({"k": {"marker": "new"}})
    host._last_request_identity_by_model = new
    assert identity_for(host, r) == {"marker": "new"}, why("identity_for reads the replaced store")
    forget_identity(host, r)
    assert new.ops == [("get", "k"), ("pop", "k")], why(f"operations went to the new store, got {new.ops}")
    assert "k" in old, why("the old store was not touched after the swap")
    stash_identity(host, r, fakes.FakeSlot(), SINGLETON)
    assert [op for op, _ in new.ops] == ["get", "pop", "setitem"], why("stash wrote into the replaced store")
    assert list(old) == ["k"] and old["k"]["thread_id"] == "t1", why("the old store keeps its single old entry")


# -- identity_for ----------------------------------------------------------------------------------

def test_identity_for_returns_the_entry_of_the_resident_key():
    r = fakes.FakeResident(resident_key="m#2", model_tag="m")
    entry = {"who": "a"}
    host = host_with(r, store={"m#2": entry, "m": {"who": "model"}})
    assert identity_for(host, r) is entry, why("the stored dict itself, found by the resident key")


def test_identity_for_with_a_key_and_no_entry_does_not_fall_back_to_the_model_tag():
    r = fakes.FakeResident(resident_key="m#2", model_tag="m")
    host = host_with(r, store={"m": {"who": "model"}})
    assert identity_for(host, r) is None, why("a keyed resident never reads another entry")


def test_identity_for_with_an_empty_key_falls_back_to_the_model_tag():
    for key in ("", None):
        r = fakes.FakeResident(resident_key=key, model_tag="m")
        entry = {"who": "model"}
        host = host_with(r, store={"m": entry})
        assert identity_for(host, r) is entry, why(f"key {key!r} reads the model tag entry")


def test_identity_for_with_an_empty_key_and_no_model_entry_is_none():
    r = fakes.FakeResident(resident_key="", model_tag="m")
    assert identity_for(host_with(r, store={"other": {}}), r) is None, why("nothing stored gives None")


def test_identity_for_is_read_only_and_does_one_lookup():
    store = fakes.RecordingDict({"k": {"v": 1}})
    r = fakes.FakeResident(resident_key="k")
    identity_for(host_with(r, store=store), r)
    assert store.ops == [("get", "k")], why(f"one get and nothing else, got {store.ops}")


# -- forget_identity -------------------------------------------------------------------------------

def test_forget_removes_the_entry_of_the_resident_key():
    r = fakes.FakeResident(resident_key="k")
    host = host_with(r, store={"k": {"v": 1}})
    assert forget_identity(host, r) is None, why("forget returns None")
    assert dict(host._last_request_identity_by_model) == {}, why("the entry is gone")


def test_forgetting_one_resident_keeps_a_same_model_sibling_entry():
    a = fakes.FakeResident(resident_key="m#1", model_tag="m")
    b = fakes.FakeResident(resident_key="m#2", model_tag="m")
    host = host_with(a, b, store={"m#1": {"v": 1}, "m#2": {"v": 2}, "m": {"v": 0}})
    forget_identity(host, a)
    assert host._last_request_identity_by_model == {"m#2": {"v": 2}, "m": {"v": 0}}, \
        why("only the forgotten resident's own key was removed")
    assert host.called("_model_residents") == [], why("forget never looks at the other residents")


def test_forget_of_a_missing_key_is_quiet_and_changes_nothing():
    r = fakes.FakeResident(resident_key="absent")
    host = host_with(r, store={"x": {"v": 1}})
    forget_identity(host, r)
    assert host._last_request_identity_by_model == {"x": {"v": 1}}, why("nothing else removed")


def test_forget_is_one_unconditional_pop_of_the_key():
    store = fakes.RecordingDict({"k": {"v": 1}})
    r = fakes.FakeResident(resident_key="k")
    forget_identity(host_with(r, store=store), r)
    assert store.ops == [("pop", "k")], why(f"exactly one pop of the key, got {store.ops}")


def test_forget_with_an_empty_key_pops_that_key_and_not_the_model_tag():
    r = fakes.FakeResident(resident_key="", model_tag="m")
    host = host_with(r, store={"m": {"v": 1}})
    forget_identity(host, r)
    assert host._last_request_identity_by_model == {"m": {"v": 1}}, why("the model tag entry is not forgotten")
