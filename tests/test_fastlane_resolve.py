"""Tests for fastlane_resolve.FastLaneNameResolver.

The 5 named arms in test_fastlane_matcher.py / test_api_config_fastlane.py
exercise compile_fastlane/match_fastlane against a HAND-ROLLED stub
`resolve_name` callable — none of them exercise this module's own internals
(generation-bumps-only-on-real-change, backoff progression, cache-only
reads never triggering a lookup, concurrent-not-serial resolution, and the
fail-closed-clears-a-stale-cache-entry design). Genuinely new code needs
its own real coverage, not borrowed coverage from a stub standing in for it
elsewhere.


`FastLaneNameResolver._resolve_name` is swapped per-instance in these tests
(a plain function assigned to the instance attribute, not the class) rather
than mocking `asyncio.get_running_loop().getaddrinfo` — it is the one seam
the module itself was written to make replaceable for exactly this reason.
"""
import asyncio
import time

import pytest

from turbohaul.fastlane import normalize_address
from turbohaul.fastlane_resolve import (
    _BASE_INTERVAL_S,
    _MAX_INTERVAL_S,
    FastLaneNameResolver,
    _NameState,
)

_ADDR_A = normalize_address("192.0.2.5")
_ADDR_B = normalize_address("192.0.2.9")


def _force_due(resolver: FastLaneNameResolver, name: str) -> None:
    """Test helper: makes `name` due on the NEXT resolve_once() regardless
    of backoff, by rewinding its last-checked timestamp — avoids sleeping
    real wall-clock seconds in a unit test."""
    resolver._state[name].last_checked_monotonic = 0.0


class TestGenerationBumpsOnlyOnRealChange:
    @pytest.mark.asyncio
    async def test_first_successful_resolution_bumps_generation(self):
        async def fake_resolve(name):
            return frozenset({_ADDR_A})

        resolver = FastLaneNameResolver(names_fn=lambda: {"gateway-svc"})
        resolver._resolve_name = fake_resolve

        assert resolver.generation == 0
        await resolver.resolve_once()
        assert resolver.generation == 1
        assert resolver.addresses_for("gateway-svc") == frozenset({_ADDR_A})

    @pytest.mark.asyncio
    async def test_a_re_resolve_to_the_same_set_does_not_bump_generation(self):
        calls = {"n": 0}

        async def fake_resolve(name):
            calls["n"] += 1
            return frozenset({_ADDR_A})

        resolver = FastLaneNameResolver(names_fn=lambda: {"gateway-svc"})
        resolver._resolve_name = fake_resolve

        await resolver.resolve_once()
        assert resolver.generation == 1

        _force_due(resolver, "gateway-svc")
        await resolver.resolve_once()
        assert calls["n"] == 2  # resolution WAS attempted again
        assert resolver.generation == 1  # but the set didn't change, so no bump

    @pytest.mark.asyncio
    async def test_a_genuinely_different_resolved_set_bumps_again(self):
        responses = [frozenset({_ADDR_A}), frozenset({_ADDR_B})]

        async def fake_resolve(name):
            return responses.pop(0)

        resolver = FastLaneNameResolver(names_fn=lambda: {"gateway-svc"})
        resolver._resolve_name = fake_resolve

        await resolver.resolve_once()
        assert resolver.generation == 1

        _force_due(resolver, "gateway-svc")
        await resolver.resolve_once()
        assert resolver.generation == 2
        assert resolver.addresses_for("gateway-svc") == frozenset({_ADDR_B})


class TestCacheOnlyRead:
    def test_addresses_for_an_unknown_name_is_the_empty_set_and_triggers_nothing(self):
        names_fn_calls = {"n": 0}

        def names_fn():
            names_fn_calls["n"] += 1
            return {"gateway-svc"}

        resolver = FastLaneNameResolver(names_fn=names_fn)
        assert resolver.addresses_for("gateway-svc") == frozenset()
        assert names_fn_calls["n"] == 0  # addresses_for never even calls names_fn
        assert resolver.generation == 0


class TestFailClosedOnFailure:
    @pytest.mark.asyncio
    async def test_a_failure_after_success_clears_the_cache_rather_than_keeping_it_stale(self):
        # This is the load-bearing design decision: keeping a stale address
        # during an outage would be the exact bug fixes (docker
        # hands the freed address to whatever container starts next).
        call_n = {"n": 0}

        async def fake_resolve(name):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return frozenset({_ADDR_A})
            raise OSError("simulated: name stopped resolving")

        resolver = FastLaneNameResolver(names_fn=lambda: {"gateway-svc"})
        resolver._resolve_name = fake_resolve

        await resolver.resolve_once()
        assert resolver.addresses_for("gateway-svc") == frozenset({_ADDR_A})
        gen_after_success = resolver.generation

        _force_due(resolver, "gateway-svc")
        await resolver.resolve_once()
        assert resolver.addresses_for("gateway-svc") == frozenset()  # cleared, not kept
        assert resolver.generation == gen_after_success + 1  # clearing IS a real change
        assert resolver._state["gateway-svc"].consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_a_failure_on_a_never_resolved_name_does_not_bump_generation(self):
        # Control for the arm above: going from "never resolved" (already
        # empty) to "still failing" (still empty) is NOT an observable
        # change — only a transition that actually changes addresses_for's
        # return value should ever bump generation.
        async def fake_resolve(name):
            raise OSError("simulated: never resolves")

        resolver = FastLaneNameResolver(names_fn=lambda: {"ghost"})
        resolver._resolve_name = fake_resolve

        await resolver.resolve_once()
        assert resolver.addresses_for("ghost") == frozenset()
        assert resolver.generation == 0
        assert resolver._state["ghost"].consecutive_failures == 1


class TestBackoff:
    def test_interval_is_base_with_zero_failures(self):
        resolver = FastLaneNameResolver(names_fn=lambda: set())
        assert resolver._interval_for(_NameState()) == _BASE_INTERVAL_S

    def test_interval_grows_exponentially_with_consecutive_failures(self):
        resolver = FastLaneNameResolver(names_fn=lambda: set())
        entry = _NameState(consecutive_failures=1)
        assert resolver._interval_for(entry) == _BASE_INTERVAL_S
        entry.consecutive_failures = 2
        assert resolver._interval_for(entry) == _BASE_INTERVAL_S * 2
        entry.consecutive_failures = 3
        assert resolver._interval_for(entry) == _BASE_INTERVAL_S * 4

    def test_interval_is_capped_at_the_ceiling(self):
        resolver = FastLaneNameResolver(names_fn=lambda: set())
        entry = _NameState(consecutive_failures=20)
        assert resolver._interval_for(entry) == _MAX_INTERVAL_S

    @pytest.mark.asyncio
    async def test_a_success_resets_the_backoff_to_zero_failures(self):
        call_n = {"n": 0}

        async def fake_resolve(name):
            call_n["n"] += 1
            if call_n["n"] <= 2:
                raise OSError("failing twice")
            return frozenset({_ADDR_A})

        resolver = FastLaneNameResolver(names_fn=lambda: {"gateway-svc"})
        resolver._resolve_name = fake_resolve

        await resolver.resolve_once()
        _force_due(resolver, "gateway-svc")
        await resolver.resolve_once()
        assert resolver._state["gateway-svc"].consecutive_failures == 2

        _force_due(resolver, "gateway-svc")
        await resolver.resolve_once()
        assert resolver._state["gateway-svc"].consecutive_failures == 0


class TestConcurrentResolution:
    @pytest.mark.asyncio
    async def test_due_names_are_resolved_concurrently_not_serially(self):
        # An addition to the confirmed design: N due names resolved
        # one-at-a-time would cost N * DELAY; concurrently (asyncio.gather)
        # it costs about one DELAY regardless of N.
        import asyncio

        delay_s = 0.05
        names = {"a", "b", "c", "d", "e"}

        async def fake_resolve(name):
            await asyncio.sleep(delay_s)
            raise OSError("simulated negative lookup")

        resolver = FastLaneNameResolver(names_fn=lambda: names)
        resolver._resolve_name = fake_resolve

        start = time.monotonic()
        await resolver.resolve_once()
        elapsed = time.monotonic() - start

        serial_cost = delay_s * len(names)
        assert elapsed < serial_cost * 0.6, (
            f"resolve_once took {elapsed:.3f}s resolving {len(names)} names at "
            f"{delay_s}s each — expected close to one delay ({delay_s}s) via "
            f"asyncio.gather, not the serial sum ({serial_cost}s); this many "
            "consecutive_failures increments prove all 5 really were attempted:"
            f" {[resolver._state[n].consecutive_failures for n in sorted(names)]}"
        )
        assert all(resolver._state[n].consecutive_failures == 1 for n in names)


# --- The REAL _resolve_name body --------------------------------------------
#
# Every other test in this file does `resolver._resolve_name = fake`, and every
# matcher test hands compile_fastlane a stub, so NOTHING executed the real
# body. Without these arms a regression there goes unnoticed, including a swap of
# the non-blocking `await loop.getaddrinfo` for BLOCKING `socket.getaddrinfo`
# -- exactly the Contract 4 violation this module's docstring promises
# against. These arms pin the real body; they are the acceptance
# criteria, not coverage for its own sake.

class TestRealResolveNameBody:

    async def test_resolves_a_real_name_through_the_actual_body(self):
        # 'localhost' resolves on any host this can run on, needs no network,
        # and is deterministic. Fails if the body is not implemented.
        r = FastLaneNameResolver(lambda: ["localhost"])
        got = await r._resolve_name("localhost")
        assert got, "the real body returned nothing for localhost"
        assert normalize_address("127.0.0.1") in got, got

    async def test_addresses_are_NORMALIZED_not_raw_strings(self):
        # Fails if normalize_address is dropped. The cached value must be
        # the same TYPE a request's ip normalizes to, or membership testing in
        # match_fastlane silently never matches.
        r = FastLaneNameResolver(lambda: ["localhost"])
        got = await r._resolve_name("localhost")
        assert all(not isinstance(a, str) for a in got), (
            f"resolved addresses must be normalized objects, not raw strings: {got!r}"
        )

    async def test_a_name_that_cannot_resolve_RAISES(self):
        # The docstring's own contract: raises on failure, never a partial or
        # empty success. resolve_once relies on this to drive backoff.
        r = FastLaneNameResolver(lambda: ["x"])
        with pytest.raises(Exception):
            await r._resolve_name("no-such-host.invalid")

    async def test_resolution_does_NOT_BLOCK_THE_EVENT_LOOP(self):
        """⛔ CONTRACT 4, and the arm that catches the blocking-getaddrinfo
        call. A ticker runs alongside the lookup: if the resolve is
        non-blocking the loop keeps scheduling it; if `await
        loop.getaddrinfo` is swapped for the blocking `socket.getaddrinfo`,
        the loop is pinned and the ticker cannot advance.

        Asserted on a FAILING name deliberately -- a negative lookup is the
        slow case (~82ms measured vs ~1.8ms positive), so it is the one where
        blocking is both most damaging and most detectable.
        """
        ticks = {"n": 0}

        async def ticker():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.001)

        r = FastLaneNameResolver(lambda: ["x"])
        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.005)          # let it start
        before = ticks["n"]
        try:
            await r._resolve_name("no-such-host.invalid")
        except Exception:
            pass
        advanced = ticks["n"] - before
        t.cancel()

        assert advanced > 0, (
            "the event loop did not advance during resolution -- the lookup is "
            f"BLOCKING (ticker advanced {advanced} times)"
        )
