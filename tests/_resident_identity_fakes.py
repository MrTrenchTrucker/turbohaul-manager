"""Small fakes shared by the resident_identity tests.

The fake host has exactly the members the host contract declares and nothing else, and it
records every method call, so a test can see what the module asked of it.
"""

from dataclasses import dataclass, field


class AlwaysEqual:
    """A handle that equals everything: only an identity check can tell two of these apart."""

    def __eq__(self, other) -> bool:
        return True

    def __hash__(self) -> int:
        return id(type(self))


class RecordingDict(dict):
    """A dict that remembers every operation done to it, as (name, key) pairs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ops: list[tuple[str, object]] = []

    def __getitem__(self, key):
        self.ops.append(("getitem", key))
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.ops.append(("setitem", key))
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self.ops.append(("delitem", key))
        super().__delitem__(key)

    def __contains__(self, key) -> bool:
        self.ops.append(("contains", key))
        return super().__contains__(key)

    def get(self, key, default=None):
        self.ops.append(("get", key))
        return super().get(key, default)

    def pop(self, key, *default):
        self.ops.append(("pop", key))
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self.ops.append(("setdefault", key))
        return super().setdefault(key, default)

    def update(self, *args, **kwargs) -> None:
        self.ops.append(("update", None))
        super().update(*args, **kwargs)

    def clear(self) -> None:
        self.ops.append(("clear", None))
        super().clear()


class RefusingDict(dict):
    """A dict whose writes fail, to prove a failed stash leaves the store alone."""

    def __setitem__(self, key, value) -> None:
        raise RuntimeError("write refused")


@dataclass
class FakeResident:
    resident_key: str = "m"
    model_tag: str = "m"
    handle: object | None = None
    spawn_seq: int = 0


@dataclass
class FakeSlot:
    client_meta: dict | None = field(default_factory=lambda: {"ip": "10.0.0.1"})
    model_tag: str = "m"
    thread_id: str = "t1"
    fastlane: object | None = None


class FakeHost:
    """The manager's side of the contract, reduced to the four members the module may use."""

    def __init__(self, residents=(), model_spawn_seq: int = 99) -> None:
        self._last_request_identity_by_model: dict[str, dict] = {}
        self.residents = list(residents)
        self.model_spawn_seq = model_spawn_seq
        self.calls: list[tuple] = []

    def _identity_dict(self, *, ip, model_tag, client_meta, thread_id, fastlane=None) -> dict:
        self.calls.append(("_identity_dict", ip, model_tag, client_meta, thread_id, fastlane))
        return {"ip": ip, "model_tag": model_tag, "client_meta": client_meta,
                "thread_id": thread_id, "fastlane": fastlane}

    def _model_residents(self):
        self.calls.append(("_model_residents",))
        return tuple(self.residents)

    def _spawn_seq_for_model(self, model_tag) -> int:
        self.calls.append(("_spawn_seq_for_model", model_tag))
        return self.model_spawn_seq

    def called(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]
