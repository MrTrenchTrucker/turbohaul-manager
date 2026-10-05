"""Tests for fastlane_resolve.FastLaneClientNamer: the background task that
puts a CONFIRMED container name on each Discovered-list row.

Every lookup is stubbed (`reverse_fn`, `forward_fn`, or `socket.getnameinfo`
/ `socket.getaddrinfo` for the executor tests): no real DNS, no sockets.
Time is a fake clock handed to the namer, so due-ness and backoff are tested
without sleeping.
"""
import asyncio
import inspect
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from turbohaul import fastlane_resolve
from turbohaul.fastlane import normalize_address
from turbohaul.fastlane_resolve import FastLaneClientNamer
from turbohaul.manager import TurbohaulManager

_A = "192.0.2.5"
_B = "192.0.2.9"
_C = "192.0.2.12"
_D = "192.0.2.13"
_E = "192.0.2.14"
_F = "192.0.2.15"


def _addrs(*raw):
    return frozenset(normalize_address(a) for a in raw)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Rig:
    """A namer wired to in-memory stand-ins for the manager's census."""

    def __init__(self, keys, *, reverse=None, forward=None):
        self.names: dict = {k: None for k in keys}
        self.set_calls: list = []
        self.reverse_calls: list = []
        self.forward_calls: list = []
        self.reverse_answers = dict(reverse or {})  # str address -> answer or Exception
        self.forward_answers = dict(forward or {})  # name -> frozenset or Exception
        self.set_raises: set = set()
        self.clock = _Clock()
        self.namer = FastLaneClientNamer(
            self._targets,
            self._set_name,
            reverse_fn=self._reverse,
            forward_fn=self._forward,
            clock=self.clock,
        )

    def _targets(self):
        return list(self.names.items())

    def _set_name(self, key, name):
        if key in self.set_raises:
            raise RuntimeError("cannot record")
        self.set_calls.append((key, name))
        self.names[key] = name

    async def _reverse(self, address):
        self.reverse_calls.append(address)
        answer = self.reverse_answers[address]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def _forward(self, name):
        self.forward_calls.append(name)
        answer = self.forward_answers.get(name, OSError("no such name"))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def close(self):
        self.namer.close()


@pytest.fixture
def make_rig():
    rigs = []

    def _make(*a, **kw):
        rig = _Rig(*a, **kw)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.close()


class TestConfirmation:
    @pytest.mark.asyncio
    async def test_name_appears_when_forward_contains_the_rows_own_address(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A)},
        )
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one")]
        assert rig.forward_calls == ["app-one"]

    @pytest.mark.asyncio
    async def test_no_name_when_forward_holds_a_different_address(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_B)},
        )
        await rig.namer.tick_once()
        assert rig.forward_calls == ["app-one"]  # it WAS checked, and refused
        assert rig.set_calls == []

    @pytest.mark.asyncio
    async def test_no_name_when_forward_holds_only_other_addresses(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_B, _C)},
        )
        await rig.namer.tick_once()
        assert rig.set_calls == []

    @pytest.mark.asyncio
    async def test_no_name_when_reverse_raises(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("no ptr")},
                       forward={"app-one": _addrs(_A)})
        await rig.namer.tick_once()
        assert rig.set_calls == []
        assert rig.forward_calls == []  # nothing to confirm without an answer

    @pytest.mark.asyncio
    async def test_no_name_when_reverse_times_out(self, make_rig, monkeypatch):
        monkeypatch.setattr(fastlane_resolve, "_NAMER_LOOKUP_TIMEOUT_S", 0.05)
        rig = make_rig([_A], forward={"app-one": _addrs(_A)})

        async def slow_reverse(address):
            await asyncio.sleep(5)
            return "app-one"

        rig.namer._reverse_fn = slow_reverse
        started = time.monotonic()
        await rig.namer.tick_once()
        assert time.monotonic() - started < 2.0
        assert rig.set_calls == []
        assert rig.namer._state[_A].misses == 1

    @pytest.mark.asyncio
    async def test_a_forward_lookup_that_times_out_confirms_nothing(self, make_rig, monkeypatch):
        monkeypatch.setattr(fastlane_resolve, "_NAMER_LOOKUP_TIMEOUT_S", 0.05)
        rig = make_rig([_A], reverse={_A: "app-one"})

        async def slow_forward(name):
            await asyncio.sleep(5)
            return _addrs(_A)

        rig.namer._forward_fn = slow_forward
        await rig.namer.tick_once()
        assert rig.set_calls == []

    @pytest.mark.asyncio
    async def test_no_name_when_the_answer_has_no_candidates(self, make_rig):
        # An address literal is not a name, so there is nothing to look up.
        rig = make_rig([_A], reverse={_A: _A},
                       forward={_A: _addrs(_A)})
        await rig.namer.tick_once()
        assert rig.forward_calls == []
        assert rig.set_calls == []

    @pytest.mark.asyncio
    async def test_the_network_suffix_is_stripped_when_only_the_bare_name_confirms(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one.net_default"},
            forward={"app-one": _addrs(_A)},  # the full name fails forward
        )
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one")]
        assert sorted(rig.forward_calls) == ["app-one", "app-one.net_default"]

    @pytest.mark.asyncio
    async def test_the_shortest_confirmed_name_wins_when_both_confirm(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one.net_default"},
            forward={
                "app-one.net_default": _addrs(_A),
                "app-one": _addrs(_A),
            },
        )
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one")]

    @pytest.mark.asyncio
    async def test_the_full_name_is_kept_when_only_it_confirms(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one.net_default"},
            forward={"app-one.net_default": _addrs(_A),
                     "app-one": _addrs(_B)},  # the bare name is someone else
        )
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one.net_default")]


class TestAddressLookedUp:
    @pytest.mark.asyncio
    async def test_an_ipv4_mapped_key_is_looked_up_as_the_plain_address(self, make_rig):
        key = "::ffff:192.0.2.5"
        rig = make_rig(
            [key],
            reverse={"192.0.2.5": "app-one"},
            forward={"app-one": _addrs("192.0.2.5")},
        )
        await rig.namer.tick_once()
        assert rig.reverse_calls == ["192.0.2.5"]
        # The name lands on the census KEY, not on the normalized form.
        assert rig.set_calls == [(key, "app-one")]

    @pytest.mark.asyncio
    async def test_an_ipv6_key_is_looked_up_as_given(self, make_rig):
        rig = make_rig(
            ["2001:db8::5"],
            reverse={"2001:db8::5": "app-six"},
            forward={"app-six": _addrs("2001:db8::5")},
        )
        await rig.namer.tick_once()
        assert rig.reverse_calls == ["2001:db8::5"]
        assert rig.set_calls == [("2001:db8::5", "app-six")]

    @pytest.mark.asyncio
    async def test_a_key_that_is_not_an_address_is_skipped_and_counted_as_a_miss(self, make_rig):
        rig = make_rig(["not-an-address", _A],
                       reverse={_A: "app-one"},
                       forward={"app-one": _addrs(_A)})
        await rig.namer.tick_once()
        assert rig.reverse_calls == [_A]  # the bad key was never looked up
        assert rig.namer._state["not-an-address"].misses == 1
        assert rig.set_calls == [(_A, "app-one")]  # and the good row went through


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_a_named_row_whose_next_confirmation_fails_is_dropped_the_same_tick(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A)},
        )
        await rig.namer.tick_once()
        assert rig.names[_A] == "app-one"

        rig.forward_answers["app-one"] = OSError("name no longer resolves")
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one"), (_A, None)]
        assert rig.names[_A] is None

    @pytest.mark.asyncio
    async def test_a_named_row_whose_name_now_points_elsewhere_is_dropped(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A)},
        )
        await rig.namer.tick_once()
        rig.forward_answers["app-one"] = _addrs(_B)  # the name moved to another container
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert rig.names[_A] is None

    @pytest.mark.asyncio
    async def test_a_named_row_whose_reverse_now_fails_is_dropped(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A)},
        )
        await rig.namer.tick_once()
        rig.reverse_answers[_A] = OSError("no ptr any more")
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert rig.names[_A] is None

    @pytest.mark.asyncio
    async def test_a_changed_confirmed_name_replaces_the_old_one(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A), "worker-two": _addrs(_A)},
        )
        await rig.namer.tick_once()
        rig.reverse_answers[_A] = "worker-two"
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert rig.set_calls == [(_A, "app-one"), (_A, "worker-two")]

    @pytest.mark.asyncio
    async def test_an_unchanged_name_is_not_written_again(self, make_rig):
        rig = make_rig(
            [_A],
            reverse={_A: "app-one"},
            forward={"app-one": _addrs(_A)},
        )
        await rig.namer.tick_once()
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 2  # it WAS re-confirmed
        assert rig.set_calls == [(_A, "app-one")]  # but nothing changed

    @pytest.mark.asyncio
    async def test_an_unnamed_row_that_stays_unnamed_is_not_written(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("no ptr")})
        await rig.namer.tick_once()
        assert rig.set_calls == []  # no needless None write for a row with no name


class TestScheduling:
    @pytest.mark.asyncio
    async def test_at_most_the_per_tick_cap_of_rows_is_looked_up(self, make_rig):
        keys = [_A, _B, _C, _D, _E, _F]
        rig = make_rig(keys, reverse={k: OSError("no ptr") for k in keys})
        await rig.namer.tick_once()
        assert fastlane_resolve._NAMER_ROWS_PER_TICK == 4
        assert rig.reverse_calls == [_A, _B, _C, _D]
        # The rows left over are taken on the next tick, before any rechecks.
        rig.clock.t += 1.0
        await rig.namer.tick_once()
        assert rig.reverse_calls[4:] == [_E, _F]

    @pytest.mark.asyncio
    async def test_the_oldest_checked_rows_go_first(self, make_rig):
        keys = [_A, _B, _C, _D, _E, _F]
        rig = make_rig(keys, reverse={k: OSError("no ptr") for k in keys})
        await rig.namer.tick_once()      # A-D checked at t0
        rig.clock.t += 1.0
        await rig.namer.tick_once()      # E, F checked at t0+1
        rig.reverse_calls.clear()
        rig.clock.t += fastlane_resolve._NAMER_MISS_MAX_S  # everything is due now
        await rig.namer.tick_once()
        assert rig.reverse_calls == [_A, _B, _C, _D]  # not E, F

    @pytest.mark.asyncio
    async def test_a_row_listed_last_but_checked_longest_ago_goes_first(self, make_rig, monkeypatch):
        keys = [_A, _B]
        rig = make_rig(keys, reverse={k: OSError("no ptr") for k in keys})
        await rig.namer.tick_once()
        # Row B is refreshed on its own later; A has waited longer.
        rig.clock.t += 5.0
        rig.namer._state[_B].last_checked = rig.clock.t
        rig.reverse_calls.clear()
        rig.clock.t += fastlane_resolve._NAMER_MISS_MAX_S
        monkeypatch.setattr(fastlane_resolve, "_NAMER_ROWS_PER_TICK", 1)
        await rig.namer.tick_once()
        assert rig.reverse_calls == [_A]

    @pytest.mark.asyncio
    async def test_overlapping_ticks_do_not_look_a_row_up_twice(self, make_rig):
        rig = make_rig([_A, _B], forward={"app-one": _addrs(_A)})
        release = asyncio.Event()

        async def gated_reverse(address):
            rig.reverse_calls.append(address)
            await release.wait()
            return "app-one"

        rig.namer._reverse_fn = gated_reverse
        first = asyncio.create_task(rig.namer.tick_once())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = asyncio.create_task(rig.namer.tick_once())
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(first, second)
        assert sorted(rig.reverse_calls) == [_A, _B]  # once each, not four calls

    @pytest.mark.asyncio
    async def test_a_cancelled_tick_releases_its_rows(self, make_rig):
        rig = make_rig([_A], forward={"app-one": _addrs(_A)})
        never = asyncio.Event()

        async def stuck_reverse(address):
            rig.reverse_calls.append(address)
            await never.wait()

        rig.namer._reverse_fn = stuck_reverse
        task = asyncio.create_task(rig.namer.tick_once())
        await asyncio.sleep(0.01)
        assert rig.namer._in_flight == {_A}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rig.namer._in_flight == set()

        rig.namer._reverse_fn = rig._reverse
        rig.reverse_answers[_A] = "app-one"
        assert _A not in rig.namer._state  # a cancelled tick records nothing
        await rig.namer.tick_once()
        assert rig.names[_A] == "app-one"

    @pytest.mark.asyncio
    async def test_a_miss_backs_off_and_the_wait_doubles(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("no ptr")})
        base = fastlane_resolve._NAMER_MISS_BASE_S
        await rig.namer.tick_once()                      # miss 1
        assert len(rig.reverse_calls) == 1

        rig.clock.t += base - 1
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 1               # still waiting
        rig.clock.t += 1
        await rig.namer.tick_once()                      # due after BASE: miss 2
        assert len(rig.reverse_calls) == 2

        rig.clock.t += 2 * base - 1
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 2               # the wait is now 2 x BASE
        rig.clock.t += 1
        await rig.namer.tick_once()                      # miss 3
        assert len(rig.reverse_calls) == 3

    @pytest.mark.asyncio
    async def test_the_miss_wait_is_capped(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("no ptr")})
        await rig.namer.tick_once()
        rig.namer._state[_A].misses = 40  # 30 * 2**39 without a cap
        rig.namer._state[_A].last_checked = rig.clock.t
        rig.reverse_calls.clear()
        rig.clock.t += fastlane_resolve._NAMER_MISS_MAX_S - 1
        await rig.namer.tick_once()
        assert rig.reverse_calls == []
        rig.clock.t += 1
        await rig.namer.tick_once()
        assert rig.reverse_calls == [_A]

    @pytest.mark.asyncio
    async def test_a_confirmed_name_is_rechecked_only_after_the_confirm_interval(self, make_rig):
        rig = make_rig([_A], reverse={_A: "app-one"},
                       forward={"app-one": _addrs(_A)})
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 1
        # Well past the miss backoff, still inside the confirm interval.
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S - 1
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 1
        rig.clock.t += 1
        await rig.namer.tick_once()
        assert len(rig.reverse_calls) == 2

    @pytest.mark.asyncio
    async def test_a_successful_lookup_resets_the_miss_count(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("no ptr")},
                       forward={"app-one": _addrs(_A)})
        await rig.namer.tick_once()
        rig.clock.t += fastlane_resolve._NAMER_MISS_BASE_S
        await rig.namer.tick_once()
        assert rig.namer._state[_A].misses == 2
        rig.reverse_answers[_A] = "app-one"
        rig.clock.t += 2 * fastlane_resolve._NAMER_MISS_BASE_S
        await rig.namer.tick_once()
        assert rig.namer._state[_A].misses == 0
        assert rig.names[_A] == "app-one"

    @pytest.mark.asyncio
    async def test_state_for_a_vanished_row_is_forgotten(self, make_rig):
        rig = make_rig([_A, _B], reverse={_A: OSError("x"), _B: OSError("x")})
        await rig.namer.tick_once()
        assert set(rig.namer._state) == {_A, _B}
        del rig.names[_B]
        await rig.namer.tick_once()
        assert set(rig.namer._state) == {_A}

    @pytest.mark.asyncio
    async def test_a_row_that_comes_back_is_looked_up_afresh(self, make_rig):
        rig = make_rig([_A], reverse={_A: OSError("x")})
        await rig.namer.tick_once()
        del rig.names[_A]
        await rig.namer.tick_once()
        rig.names[_A] = None
        await rig.namer.tick_once()  # no clock advance: forgotten state means due now
        assert len(rig.reverse_calls) == 2

    @pytest.mark.asyncio
    async def test_one_row_whose_set_name_raises_does_not_stop_the_others(self, make_rig):
        rig = make_rig(
            [_A, _B],
            reverse={_A: "app-one", _B: "worker-two"},
            forward={"app-one": _addrs(_A), "worker-two": _addrs(_B)},
        )
        rig.set_raises = {_A}
        await rig.namer.tick_once()
        assert rig.set_calls == [(_B, "worker-two")]
        assert rig.namer._state[_A].misses == 1  # counted as a miss
        assert rig.namer._state[_B].misses == 0

    @pytest.mark.asyncio
    async def test_a_row_whose_name_could_not_be_dropped_is_retried_on_the_miss_backoff(self, make_rig):
        rig = make_rig([_A], reverse={_A: "app-one"},
                       forward={"app-one": _addrs(_A)})
        await rig.namer.tick_once()
        rig.forward_answers["app-one"] = OSError("gone")
        rig.set_raises = {_A}
        rig.clock.t += fastlane_resolve._NAMER_CONFIRM_INTERVAL_S
        await rig.namer.tick_once()
        assert rig.names[_A] == "app-one"  # the drop failed
        rig.set_raises = set()
        rig.clock.t += fastlane_resolve._NAMER_MISS_BASE_S  # far sooner than a confirm interval
        await rig.namer.tick_once()
        assert rig.names[_A] is None

    @pytest.mark.asyncio
    async def test_a_lookup_for_one_row_that_raises_does_not_stop_the_others(self, make_rig):
        rig = make_rig(
            [_A, _B],
            reverse={_A: RuntimeError("resolver blew up"), _B: "worker-two"},
            forward={"worker-two": _addrs(_B)},
        )
        await rig.namer.tick_once()
        assert rig.set_calls == [(_B, "worker-two")]


class TestExecutor:
    @pytest.fixture
    def socket_stubs(self, monkeypatch):
        seen = {"threads": [], "nameinfo_args": [], "addrinfo_args": []}

        def fake_getnameinfo(sockaddr, flags):
            seen["threads"].append(threading.current_thread().name)
            seen["nameinfo_args"].append((sockaddr, flags))
            return ("app-one.net_default", "0")

        def fake_getaddrinfo(host, port, *args, **kwargs):
            seen["threads"].append(threading.current_thread().name)
            seen["addrinfo_args"].append((host, port))
            if host == "app-one":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_A, 0))]
            raise socket.gaierror("unknown name")

        monkeypatch.setattr(socket, "getnameinfo", fake_getnameinfo)
        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        return seen

    @pytest.mark.asyncio
    async def test_default_lookups_run_in_the_namers_own_threads_not_the_loops_default_executor(
        self, socket_stubs
    ):
        loop = asyncio.get_running_loop()
        default_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="loop-default")
        loop.set_default_executor(default_exec)
        used = []
        real = loop.run_in_executor

        def spy(executor, func, *args):
            used.append(executor)
            return real(executor, func, *args)

        loop.run_in_executor = spy
        names = {_A: None}
        sets = []
        namer = FastLaneClientNamer(
            lambda: list(names.items()), lambda k, n: sets.append((k, n))
        )
        try:
            await namer.tick_once()
        finally:
            del loop.run_in_executor
            namer.close()
            default_exec.shutdown(wait=True)

        assert sets == [(_A, "app-one")]
        assert len(socket_stubs["threads"]) == 3  # one reverse + two forward
        assert all(t.startswith("fastlane-names") for t in socket_stubs["threads"])
        assert not any(t.startswith("loop-default") for t in socket_stubs["threads"])
        assert used and all(e is namer._executor for e in used)
        # The loop's default executor never ran anything (it starts a thread
        # only when it is handed work).
        assert default_exec._threads == set()

    @pytest.mark.asyncio
    async def test_the_default_reverse_lookup_demands_a_name(self, socket_stubs):
        namer = FastLaneClientNamer(lambda: [], lambda k, n: None)
        try:
            answer = await namer._default_reverse(_A)
        finally:
            namer.close()
        assert answer == "app-one.net_default"
        (sockaddr, flags), = socket_stubs["nameinfo_args"]
        assert sockaddr == (_A, 0)
        # Without NI_NAMEREQD an address with no name comes back as its own
        # numeric text, which would then be "confirmed" as a name.
        assert flags & socket.NI_NAMEREQD

    @pytest.mark.asyncio
    async def test_the_default_forward_lookup_returns_normalized_addresses(self, socket_stubs):
        namer = FastLaneClientNamer(lambda: [], lambda k, n: None)
        try:
            result = await namer._default_forward("app-one")
        finally:
            namer.close()
        assert result == _addrs(_A)
        assert socket_stubs["addrinfo_args"] == [("app-one", None)]

    def test_the_namer_source_uses_its_own_executor_only(self):
        src = inspect.getsource(FastLaneClientNamer)
        # Positive: both lookups are submitted to the namer's own executor.
        assert len(re.findall(r"run_in_executor\(\s*self\._executor", src)) == 2
        # Negative: nothing goes through the loop's default executor.
        assert _default_executor_uses(src) == []

    def test_the_default_executor_scan_flags_what_it_should(self):
        # Control: the scan reports each banned form, so a clean result on
        # the namer's source means something.
        for bad in (
            "await asyncio.to_thread(socket.getnameinfo, a)",
            "await loop.getaddrinfo(name, None)",
            "await loop.getnameinfo((a, 0))",
            "await loop.run_in_executor(None, fn)",
            "await loop.run_in_executor(\n    None, fn)",
        ):
            assert _default_executor_uses(bad), bad
        assert _default_executor_uses(
            "await loop.run_in_executor(self._executor, fn)") == []

    def test_the_namers_executor_is_named_and_sized_from_the_constants(self):
        namer = FastLaneClientNamer(lambda: [], lambda k, n: None)
        try:
            assert isinstance(namer._executor, ThreadPoolExecutor)
            assert namer._executor._thread_name_prefix == "fastlane-names"
            assert namer._executor._max_workers == fastlane_resolve._NAMER_WORKERS
        finally:
            namer.close()

    def test_close_is_idempotent_and_stops_the_executor(self):
        namer = FastLaneClientNamer(lambda: [], lambda k, n: None)
        namer.close()
        namer.close()
        assert namer._executor._shutdown is True


def _default_executor_uses(src):
    """The ways a piece of source could reach the loop's default executor."""
    patterns = (
        r"to_thread", r"loop\.getaddrinfo", r"loop\.getnameinfo",
        r"run_in_executor\(\s*None",
    )
    return [p for p in patterns if re.search(p, src)]


def _namer_threads():
    return [t for t in threading.enumerate() if t.name.startswith("fastlane-names")]


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_cancelling_run_periodic_leaves_no_lookup_thread_behind(self, monkeypatch):
        ran = []

        def fake_getnameinfo(sockaddr, flags):
            ran.append(threading.current_thread().name)
            time.sleep(0.01)
            return ("app-one", "0")

        def fake_getaddrinfo(host, port, *args, **kwargs):
            time.sleep(0.01)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_A, 0))]

        monkeypatch.setattr(socket, "getnameinfo", fake_getnameinfo)
        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        sets = []
        namer = FastLaneClientNamer(lambda: [(_A, None)], lambda k, n: sets.append((k, n)))
        task = asyncio.create_task(namer.run_periodic(tick_s=0.01))
        deadline = time.monotonic() + 3.0
        while not sets and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        # Real thread work happened, on a namer thread, before the cancel.
        assert sets == [(_A, "app-one")]
        assert ran and ran[0].startswith("fastlane-names")
        assert _namer_threads()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 2.0
        while _namer_threads() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert _namer_threads() == []

    @pytest.mark.asyncio
    async def test_run_periodic_survives_a_failing_tick(self, make_rig):
        rig = make_rig([_A], reverse={_A: "app-one"},
                       forward={"app-one": _addrs(_A)})
        calls = {"n": 0}
        real_targets = rig.namer._targets_fn

        def flaky_targets():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("census unavailable")
            return real_targets()

        rig.namer._targets_fn = flaky_targets
        task = asyncio.create_task(rig.namer.run_periodic(tick_s=0.01))
        try:
            deadline = time.monotonic() + 3.0
            while not rig.set_calls and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert calls["n"] >= 2
        assert rig.set_calls == [(_A, "app-one")]

    @pytest.mark.asyncio
    async def test_run_periodic_closes_the_executor_when_it_ends(self, make_rig):
        rig = make_rig([])
        task = asyncio.create_task(rig.namer.run_periodic(tick_s=0.01))
        await asyncio.sleep(0.03)
        assert rig.namer._executor._shutdown is False
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rig.namer._executor._shutdown is True


class TestNotOnTheRequestPath:
    _TOKENS = ("namer", "getnameinfo", "gethostbyaddr", "reverse")

    def _sources(self):
        from turbohaul import fastlane
        return {
            "TurbohaulManager._fastlane_census_observe":
                inspect.getsource(TurbohaulManager._fastlane_census_observe),
            "TurbohaulManager._fastlane_table":
                inspect.getsource(TurbohaulManager._fastlane_table),
            "fastlane.match_fastlane": inspect.getsource(fastlane.match_fastlane),
        }

    def test_the_request_path_never_mentions_the_namer_or_a_reverse_lookup(self):
        for where, src in self._sources().items():
            lowered = src.lower()
            for token in self._TOKENS:
                assert token not in lowered, f"{where} mentions {token!r}"

    def test_the_scan_is_not_vacuous(self):
        # Control: the table accessor does reach the forward resolver, and
        # the same scan finds that name, so it can see what it is looking for.
        table_src = self._sources()["TurbohaulManager._fastlane_table"]
        assert "addresses_for" in table_src
        # And the token scan does flag a mention when one is present.
        sample = "mgr._fastlane_namer.names()".lower()
        assert any(token in sample for token in self._TOKENS)

    def test_the_manager_module_does_not_import_the_namer(self):
        import turbohaul.manager as manager_mod
        assert not hasattr(manager_mod, "FastLaneClientNamer")
        assert "FastLaneClientNamer" not in inspect.getsource(manager_mod)


class TestConstants:
    def test_each_constant_has_its_documented_value(self):
        assert fastlane_resolve._NAMER_TICK_S == 5.0
        assert fastlane_resolve._NAMER_ROWS_PER_TICK == 4
        assert fastlane_resolve._NAMER_LOOKUP_TIMEOUT_S == 2.0
        assert fastlane_resolve._NAMER_CONFIRM_INTERVAL_S == 300.0
        assert fastlane_resolve._NAMER_MISS_BASE_S == 30.0
        assert fastlane_resolve._NAMER_MISS_MAX_S == 600.0
        assert fastlane_resolve._NAMER_WORKERS == 2
