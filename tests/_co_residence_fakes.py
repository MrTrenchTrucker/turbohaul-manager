"""Small fakes shared by the co_residence unit tests: a host that records every call in order.

The fake host has exactly the members the host contract declares and nothing else. In
particular it has no routing method at all, so a re-check that tried to route a slot would
fail with an AttributeError instead of passing quietly.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field

from turbohaul.co_residence import RecheckState


@dataclass
class FakeSlot:
    slot_id: str = "s1"
    model_tag: str = "m"
    thread_id: str | None = None
    created_at: float = 0.0
    is_evicted: bool = False
    disconnect_event: asyncio.Event | None = None
    fastlane_relocated_main_gpu: int | None = None
    fastlane_relocated_split_mode: str | None = None


@dataclass
class FakeState:
    name: str = "ACTIVE"


@dataclass
class FakeResident:
    model_tag: str = "m"
    resident_key: str = "m"
    state: FakeState = field(default_factory=FakeState)
    inbox: asyncio.Queue | None = field(default_factory=asyncio.Queue)
    idle_thread_id: str | None = None

    def put(self, *slots: FakeSlot) -> "FakeResident":
        for s in slots:
            self.inbox.put_nowait(s)
        return self


class FakeLock:
    """An async context manager that tells the host when it is held."""

    def __init__(self, host: "FakeHost") -> None:
        self._host = host

    async def __aenter__(self) -> None:
        self._host.held = True
        self._host.calls.append("lock_enter")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._host.held = False
        self._host.calls.append("lock_exit")


class FakeStagingQueue:
    def __init__(self, host: "FakeHost") -> None:
        self._host = host

    def staging_snapshot(self) -> list[FakeSlot]:
        self._host.record("queue.staging_snapshot")
        return list(self._host.staged)

    async def remove(self, slot_id: str) -> FakeSlot | None:
        self._host.record("queue.remove")
        for s in list(self._host.staged):
            if s.slot_id == slot_id:
                self._host.staged.remove(s)
                return s
        return None


class _Settings:
    def __init__(self, host: "FakeHost") -> None:
        self._host = host

    @property
    def max_parallel_sidecars(self) -> int:
        if self._host.cap_raises:
            raise AssertionError("the box budget must not be read here")
        self._host.record("runtime.queue.max_parallel_sidecars")
        return self._host.cap


class _Runtime:
    def __init__(self, host: "FakeHost") -> None:
        self.queue = _Settings(host)


class FakeHost:
    """Config in, calls out. ``calls`` is the ordered list of member names the code used."""

    def __init__(self, *, residents=(), staged=(), now: float = 1000.0) -> None:
        self.residents = list(residents)
        self.staged = list(staged)
        self.now = now
        self.state = RecheckState()
        self.calls: list[str] = []
        self.held = False
        self.held_at: dict[str, list[bool]] = {}
        self.cap = 3
        self.cap_raises = False
        self.grace_pairs: frozenset = frozenset()
        self.eff_cap = 1
        self.wants_one_card = True
        self.placement = (4000, 1, 0, "none", 0, False, False)
        self.pick_result: tuple = (1, "none")
        self.vram_ok = True
        self.card_avail: list[int] = [16000, 16000]
        self.reserve_result: FakeResident | None = FakeResident(resident_key="new-key")
        self.reserve_exc: BaseException | None = None
        self.args: dict[str, list] = {}
        self.queue = FakeStagingQueue(self)
        self.runtime = _Runtime(self)
        self._registry_lock = FakeLock(self)

    def record(self, name: str, *args) -> None:
        self.calls.append(name)
        self.held_at.setdefault(name, []).append(self.held)
        self.args.setdefault(name, []).append(args)

    # -- the declared host methods ------------------------------------------------
    def _co_residence_now(self) -> float:
        self.record("_co_residence_now")
        return self.now

    def _co_residence_state(self) -> RecheckState:
        self.record("_co_residence_state")
        return self.state

    def _model_residents(self) -> list[FakeResident]:
        self.record("_model_residents")
        return list(self.residents)

    def _instances_for(self, model_tag) -> list[FakeResident]:
        self.record("_instances_for", model_tag)
        return [r for r in self.residents if r.model_tag == model_tag]

    async def _grace_active_exclusions(self) -> frozenset:
        self.record("_grace_active_exclusions")
        return self.grace_pairs

    def _route_cap_and_placement(self, model_tag):
        self.record("_route_cap_and_placement", model_tag)
        return self.eff_cap, self.wants_one_card

    def _resolve_placement_locked(self, model_tag):
        self.record("_resolve_placement_locked", model_tag)
        return self.placement

    def _pick_card_for_new_instance(self, model_tag, need_mib):
        self.record("_pick_card_for_new_instance", model_tag, need_mib)
        return self.pick_result

    def _card_avail_mib(self) -> list[int]:
        self.record("_card_avail_mib")
        return list(self.card_avail)

    def _vram_admits_locked(self, need, parallel, main_gpu, split_mode) -> bool:
        self.record("_vram_admits_locked", need, parallel, main_gpu, split_mode)
        return self.vram_ok

    async def _reserve_and_start_locked(self, slot, main_gpu_override=None, split_mode_override=None):
        self.record("_reserve_and_start_locked", slot, main_gpu_override, split_mode_override)
        if self.reserve_exc is not None:
            raise self.reserve_exc
        return self.reserve_result

    def _remember_affinity(self, model_tag, session_key, resident_key) -> None:
        self.record("_remember_affinity", model_tag, session_key, resident_key)

    def _session_key_for(self, slot):
        self.record("_session_key_for", slot)
        return f"tid:{slot.thread_id}" if slot.thread_id else None

    @contextlib.contextmanager
    def _reserving(self, model_tag):
        """The in-flight reservation marker: enter and exit are in the call log."""
        self.record("reserving_enter", model_tag)
        try:
            yield
        finally:
            self.calls.append("reserving_exit")

    def _unpark_fastlane_claim(self, slot) -> None:
        self.record("_unpark_fastlane_claim", slot)

    def _handle_unloaded_slot(self, slot) -> None:
        self.record("_handle_unloaded_slot", slot)

    def _fail_completion_future(self, slot, exc) -> None:
        self.record("_fail_completion_future", slot, exc)
