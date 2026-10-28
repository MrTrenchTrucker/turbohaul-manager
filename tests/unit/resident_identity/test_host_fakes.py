"""The shared fakes carry exactly the members of the host contract, so a test host cannot drift from it."""

import inspect

import _resident_identity_fakes as fakes

from turbohaul.resident_identity import ResidentIdentityHost, ResidentLike, SlotLike


def members(proto) -> set[str]:
    names = set(getattr(proto, "__annotations__", {}))
    names |= {n for n, v in vars(proto).items() if inspect.isfunction(v) and not n.startswith("__")}
    return names


def test_the_fake_host_has_every_declared_member_and_no_other_contract_member():
    host = fakes.FakeHost()
    declared = {"_last_request_identity_by_model", "_identity_dict", "_model_residents", "_spawn_seq_for_model"}
    assert members(ResidentIdentityHost) == declared, "resident_identity: host contract differs from the pinned members"
    for name in declared:
        assert hasattr(host, name), f"resident_identity: the fake host lacks {name}; see its AGENTS.md card"


def test_the_fake_resident_and_slot_carry_the_declared_attributes():
    assert set(ResidentLike.__annotations__) == {"resident_key", "model_tag", "handle", "spawn_seq"}
    assert set(SlotLike.__annotations__) == {"client_meta", "model_tag", "thread_id", "fastlane"}
    r, s = fakes.FakeResident(), fakes.FakeSlot()
    for name in ResidentLike.__annotations__:
        assert hasattr(r, name), f"resident_identity: the fake resident lacks {name}"
    for name in SlotLike.__annotations__:
        assert hasattr(s, name), f"resident_identity: the fake slot lacks {name}"


def test_the_recording_dict_sees_every_kind_of_access():
    d = fakes.RecordingDict()
    d["a"] = 1
    d.get("a")
    d["a"]
    _ = "a" in d
    d.setdefault("b", 2)
    d.update({"c": 3})
    d.pop("a")
    del d["b"]
    d.clear()
    assert [op for op, _ in d.ops] == [
        "setitem", "get", "getitem", "contains", "setdefault", "update", "pop", "delitem", "clear"], \
        "resident_identity: the recording dict must see every operation, or a no-touch test proves nothing"
