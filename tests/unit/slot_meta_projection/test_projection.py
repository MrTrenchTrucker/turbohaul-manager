"""slot_meta_projection: the identity projection, the census projection and its parse (pure)."""
import json
import re
import sqlite3

import pytest

from turbohaul.slot_meta_projection import (
    CENSUS_KEYS,
    CENSUS_SLOTS_SQL,
    IDENTITY_KEYS,
    identity_client_meta,
    parse_census_meta,
    strip_stored_meta,
)


class TestKeyLists:
    def test_the_identity_keys_are_exactly_the_persisted_set(self):
        assert set(IDENTITY_KEYS) == {
            "ip", "session_id", "role", "is_main", "is_sub_agent", "is_curator", "is_compression", "save_kv"}
        assert len(IDENTITY_KEYS) == len(set(IDENTITY_KEYS))

    def test_the_census_keys_are_the_address_and_the_class_labels(self):
        assert list(CENSUS_KEYS) == ["ip", "is_curator", "is_compression", "is_sub_agent", "is_main", "role"]

    def test_every_census_key_is_persisted(self):
        assert set(CENSUS_KEYS) <= set(IDENTITY_KEYS), "a key the census reads must be one the slot row keeps"


class TestIdentityClientMeta:
    def test_keeps_the_identity_keys_and_drops_everything_else(self):
        meta = {"ip": "192.0.2.1", "messages": [{"role": "user", "content": "hi"}], "tools": [{"x": 1}],
                "temperature": 0.2, "session_id": "s", "role": "main", "is_main": True, "save_kv": False,
                "kind": "openai-chat-completion", "req_id": "r", "requester": "q"}
        assert identity_client_meta(meta) == {
            "ip": "192.0.2.1", "session_id": "s", "role": "main", "is_main": True, "save_kv": False}

    def test_a_present_key_with_a_null_value_is_kept(self):
        assert identity_client_meta({"ip": None, "is_curator": None}) == {"ip": None, "is_curator": None}

    def test_an_absent_key_stays_absent(self):
        assert identity_client_meta({"ip": "192.0.2.1"}) == {"ip": "192.0.2.1"}

    @pytest.mark.parametrize("bad", [None, "text", 7, ["ip"], ("ip",), 1.5, True])
    def test_a_value_that_is_not_a_dict_gives_an_empty_dict(self, bad):
        assert identity_client_meta(bad) == {}

    def test_the_input_is_not_changed(self):
        meta = {"ip": "192.0.2.1", "messages": ["a"]}
        identity_client_meta(meta)
        assert meta == {"ip": "192.0.2.1", "messages": ["a"]}

    def test_all_eight_keys_survive_together(self):
        meta = {k: i for i, k in enumerate(IDENTITY_KEYS)}
        assert identity_client_meta({**meta, "messages": "x"}) == meta


class TestParseCensusMeta:
    def test_rebuilds_the_non_null_keys_in_the_projection_order(self):
        text = '["1.2.3.4",null,null,null,true,"main"]'
        assert parse_census_meta(text) == {"ip": "1.2.3.4", "is_main": True, "role": "main"}

    def test_a_falsy_value_is_kept_not_dropped(self):
        text = '["1.2.3.4",false,0,[],"",null]'
        got = parse_census_meta(text)
        assert got == {"ip": "1.2.3.4", "is_curator": False, "is_compression": 0, "is_sub_agent": [], "is_main": ""}

    def test_every_position_maps_to_its_own_key(self):
        text = json.dumps([f"v{i}" for i in range(len(CENSUS_KEYS))])
        assert parse_census_meta(text) == {k: f"v{i}" for i, k in enumerate(CENSUS_KEYS)}

    @pytest.mark.parametrize("bad", [None, "", "not json", "{", '{"ip": "1.2.3.4"}', "[]", '["a","b"]',
                                     '["a",null,null,null,null,null,null]', "null", "7", '"text"'])
    def test_anything_unusable_gives_an_empty_dict_and_never_raises(self, bad):
        error = None
        try:
            got = parse_census_meta(bad)
        except Exception as exc:  # the property under test is that nothing is raised
            error, got = exc, None
        assert error is None, f"parse_census_meta must never raise, but raised {error!r} for {bad!r}"
        assert got == {}

    def test_an_all_null_array_gives_an_empty_dict(self):
        assert parse_census_meta("[null,null,null,null,null,null]") == {}


def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE slots (slot_id TEXT, thread_id TEXT, model_tag TEXT, created_at TEXT, client_meta_json TEXT)")
    conn.executemany("INSERT INTO slots VALUES (?,?,?,?,?)", rows)
    return conn


class TestCensusSql:
    def test_the_body_is_never_a_select_item(self):
        select_list = CENSUS_SLOTS_SQL.split(" FROM ")[0]
        stripped = re.sub(r"json_valid\(client_meta_json\)", "", select_list)
        stripped = re.sub(r"json_extract\(client_meta_json,[^)]*\)", "", stripped)
        assert "client_meta_json" not in stripped

    def test_returns_one_projected_array_per_row_and_the_plain_columns(self):
        body = {"ip": "192.0.2.1", "is_main": True, "role": "main", "messages": ["x" * 50] * 4, "tools": [1]}
        conn = _db([("s1", "t1", "m", "2026-01-01T00:00:00+00:00", json.dumps(body))])
        (slot_id, thread_id, model_tag, created_at, projected), = conn.execute(CENSUS_SLOTS_SQL).fetchall()
        assert (slot_id, thread_id, model_tag, created_at) == ("s1", "t1", "m", "2026-01-01T00:00:00+00:00")
        assert parse_census_meta(projected) == {"ip": "192.0.2.1", "is_main": True, "role": "main"}
        assert len(projected) < 100, "the projected value must stay tiny whatever the body holds"

    def test_a_body_that_is_not_valid_json_gives_null_instead_of_failing_the_statement(self):
        conn = _db([("s1", "t", "m", "c", "{not json"), ("s2", "t", "m", "c", json.dumps({"ip": "192.0.2.2"}))])
        rows = dict((r[0], r[4]) for r in conn.execute(CENSUS_SLOTS_SQL).fetchall())
        assert rows["s1"] is None
        assert parse_census_meta(rows["s2"]) == {"ip": "192.0.2.2"}

    @pytest.mark.parametrize("body", [None, ""])
    def test_a_null_or_empty_body_gives_null(self, body):
        conn = _db([("s1", "t", "m", "c", body)])
        assert conn.execute(CENSUS_SLOTS_SQL).fetchall()[0][4] is None

    def test_a_body_that_is_valid_json_but_not_an_object_projects_to_no_keys(self):
        conn = _db([("s1", "t", "m", "c", "[1, 2]")])
        assert parse_census_meta(conn.execute(CENSUS_SLOTS_SQL).fetchall()[0][4]) == {}

    def test_nan_in_a_body_is_not_valid_json_for_sqlite(self):
        conn = _db([("s1", "t", "m", "c", '{"ip": "192.0.2.1", "x": NaN}')])
        assert conn.execute(CENSUS_SLOTS_SQL).fetchall()[0][4] is None

    def test_booleans_and_empty_containers_keep_their_python_truthiness(self):
        body = {"ip": "192.0.2.1", "is_curator": False, "is_sub_agent": [], "is_main": {}, "role": ""}
        conn = _db([("s1", "t", "m", "c", json.dumps(body))])
        got = parse_census_meta(conn.execute(CENSUS_SLOTS_SQL).fetchall()[0][4])
        assert got == {"ip": "192.0.2.1", "is_curator": False, "is_sub_agent": [], "is_main": {}, "role": ""}
        assert [bool(got[k]) for k in ("is_curator", "is_sub_agent", "is_main", "role")] == [False] * 4


class TestStripStoredMeta:
    def test_reduces_a_full_body_to_the_identity_keys(self):
        body = {"ip": "192.0.2.1", "messages": ["a"] * 5, "tools": [1], "role": "main", "is_main": True, "save_kv": False}
        got = strip_stored_meta(json.dumps(body))
        assert json.loads(got) == {"ip": "192.0.2.1", "role": "main", "is_main": True, "save_kv": False}

    def test_the_result_is_what_a_new_slot_would_have_been_written_with(self):
        body = {"messages": ["a"], "is_curator": None, "ip": "192.0.2.1", "session_id": "s"}
        assert strip_stored_meta(json.dumps(body)) == json.dumps(identity_client_meta(body))

    def test_an_already_identity_only_body_comes_back_byte_identical(self):
        once = strip_stored_meta(json.dumps({"ip": "192.0.2.1", "messages": ["a"], "is_main": True}))
        assert strip_stored_meta(once) == once

    def test_an_object_with_no_identity_keys_becomes_an_empty_object(self):
        assert strip_stored_meta('{"messages": [1], "model": "m"}') == "{}"

    @pytest.mark.parametrize("bad", [None, "", "not json", "{broken", "[1, 2]", '"text"', "7", "null", "true"])
    def test_anything_that_is_not_a_json_object_gives_none_and_never_raises(self, bad):
        error = None
        try:
            got = strip_stored_meta(bad)
        except Exception as exc:
            error, got = exc, "raised"
        assert error is None, f"strip_stored_meta must never raise, but raised {error!r} for {bad!r}"
        assert got is None

    def test_a_present_null_identity_value_is_kept(self):
        assert json.loads(strip_stored_meta('{"ip": null, "messages": []}')) == {"ip": None}
