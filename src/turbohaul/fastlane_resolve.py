"""Fast Lane container-name resolution (forward) and client naming (reverse).

Two classes live here, and they are deliberately different.

`FastLaneNameResolver` does FORWARD resolution only: name -> address set, for
the operator-typed `container_name` of a rule. The rule MATCHER never uses
reverse DNS: a matcher built on it would be unreliable, because the
anonymous bridge gateway sits inside the on-link subnet (an in-subnet miss
goes forward upstream -- slow, and slower still when upstream DNS is dead -- exactly
where a rolled container address lands), and `socket.gethostbyaddr` blocks
inside `async def submit`. The resolver only ever resolves an operator-typed
`container_name` forward, on a schedule, into a cache the admission path
reads.

`FastLaneClientNamer` looks up the container name behind each address on the
Discovered list, so the list can offer "add by name". It does a reverse
lookup ONLY for that display, off the request path, in an executor of its
own, and a name is shown only after a forward lookup confirms that the name
resolves back to the row's own address. Nothing it finds reaches the matcher
and nothing on the request path waits on it.

*** SECURITY PROPERTY -- FAIL CLOSED, STRUCTURALLY, NOT BY GUARD ***
`addresses_for(name)` returns `frozenset()` for any name this resolver has
never successfully resolved, OR whose most recent resolution attempt
failed (see `_apply_result`: a failure CLEARS a previously-cached address
set rather than keeping it -- keeping a stale address here would be the
exact bug this design exists to avoid, since docker hands a now-freed
address to the next container that starts, lowest-free-first). An empty set can never
satisfy a `req_addr in match_addresses` membership test, so an unresolved
or currently-failing name degrades to UNLISTED and `fastlane.match_fastlane`'s
loop falls through to the next rule, which owns a disjoint address set. A
resolver failure -- DNS down, container not yet up, name typo'd -- can
therefore NEVER promote a request into another client's Fast Lane. There
is no wildcard, no shared fallback, and no last-known-good value kept
anywhere outside this one cache: once a name's entry is empty, it is
unlisted, full stop.

Modeled on plugin_health.py's refresh/backoff/restate shape
(`PluginHealthMonitor`): a bounded periodic refresh into a cache,
exponential backoff on repeated failure, `await loop.getaddrinfo(...)` as
this codebase's established non-blocking resolution primitive (mirrors
plugin_health.py's own `_resolve`) -- never the blocking
`socket.gethostbyname`/`gethostbyaddr` on the event loop. The admission path (via
`fastlane.compile_fastlane`'s `resolve_name=` callable) reads
`addresses_for` ONLY -- it never triggers a lookup itself and never blocks.

Interval/backoff constants below assume lookups are cheap (a positive
docker-embedded-DNS lookup and a negative one are both fast -- unlike the
slow worst case of a reverse lookup of an in-subnet miss under a dead
upstream, which this resolver never performs). They are
deliberately much shorter than plugin_health.py's 60s/600s: staleness here
IS the bug this design exists to avoid (a rolled container address briefly
not matching, or a stale one briefly matching the wrong client), and a name
stuck unresolved means a client silently has NO Fast Lane priority at all
-- more operationally significant than one degraded plugin endpoint, so
even the worst-case backoff should still recover within about a minute of
the container coming back.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import socket
import time
from dataclasses import dataclass

from .fastlane import normalize_address
from .fastlane_client_names import confirmed_name, name_candidates

log = logging.getLogger(__name__)

_BASE_INTERVAL_S = 5.0
_BACKOFF_FACTOR = 2.0
_MAX_INTERVAL_S = 60.0
# Outer scheduling granularity -- roughly the same tick:base ratio
# plugin_health.py uses (~1:4), scaled down with the smaller base interval.
_TICK_S = 2.0


def _normalized_addresses(infos) -> frozenset:
    """The set of normalized addresses in a `getaddrinfo` result. An entry
    that does not parse as a single host is skipped, not fatal."""
    addresses = set()
    for info in infos:
        sockaddr = info[4]
        try:
            addresses.add(normalize_address(sockaddr[0]))
        except ValueError:
            continue
    return frozenset(addresses)


@dataclass
class _NameState:
    addresses: frozenset = frozenset()
    consecutive_failures: int = 0
    last_checked_monotonic: float = 0.0


class FastLaneNameResolver:
    """Owns per-name resolved-address cache state for one TurbohaulManager.
    `names_fn` is a zero-arg callable returning the CURRENT collection of
    container_name values that need resolution -- asked fresh every tick
    (mirrors plugin_health.py's `registry_fn`), not captured once, since
    the configured rule set can change via PUT /api/config. Duplicates in
    `names_fn`'s return are wasted work, not a correctness problem; pass a
    deduplicated collection (e.g. a set) for a clean tick."""

    def __init__(self, names_fn) -> None:
        self._names_fn = names_fn
        self._state: dict[str, _NameState] = {}
        self._generation = 0
        # Names with a lookup in the air right now. Due-ness is READ from
        # last_checked_monotonic before the gather and only WRITTEN after it,
        # so without this set two overlapping callers -- the periodic tick and
        # a PUT-triggered pass -- both see the same unmarked entry and both
        # issue a lookup for it. See _due_names and resolve_once.
        self._in_flight: set[str] = set()

    @property
    def generation(self) -> int:
        """Bumps ONLY when some name's resolved address set actually
        CHANGES value -- not on every tick, not on a no-op re-resolve of an
        unchanged set. A consumer that recompiles the Fast Lane rule table
        keyed on this value only ever recompiles when resolved membership
        genuinely moved."""
        return self._generation

    def addresses_for(self, name: str) -> frozenset:
        """CACHE ONLY -- never triggers a resolution. Returns frozenset()
        for any name never successfully resolved, or currently failing.
        See the module docstring's fail-closed property."""
        entry = self._state.get(name)
        if entry is None:
            return frozenset()
        return entry.addresses

    def _interval_for(self, entry: _NameState) -> float:
        if entry.consecutive_failures == 0:
            return _BASE_INTERVAL_S
        interval = _BASE_INTERVAL_S * (_BACKOFF_FACTOR ** (entry.consecutive_failures - 1))
        return min(interval, _MAX_INTERVAL_S)

    def _due_names(self, now: float) -> list[str]:
        due = []
        for name in self._names_fn():
            # A name already being looked up is NOT due again. The mark that
            # makes a name un-due (last_checked_monotonic) is written only
            # after the gather completes, so a second caller entering during
            # that window would otherwise re-issue the very same lookup.
            if name in self._in_flight:
                continue
            entry = self._state.get(name)
            if entry is None or now - entry.last_checked_monotonic >= self._interval_for(entry):
                due.append(name)
        return due

    async def resolve_once(self) -> None:
        """Resolves every name whose own backoff says it is due, THIS
        tick, CONCURRENTLY -- not one at a time. A negative lookup is
        slower than a positive one; resolving N
        unresolvable names serially would cost N negative lookups, all inside
        one _TICK_S=2s tick, growing with rule count. Concurrently (asyncio.gather) it
        costs one slowest-lookup's worth regardless of N -- a for-loop
        with an await inside would make the refresher get SLOWER exactly
        as the config gets MORE broken, which is precisely when it needs
        to stay responsive.

        NOT re-entrant against itself without the in-flight claim below.
        Two callers share one instance -- the periodic tick and the
        PUT /api/config path -- and neither serialises the other. Without the
        claim, both resolve the same due name in the same interval and the
        results land in COMPLETION order: a slow FAILING lookup applied after
        a fast SUCCEEDING one clears addresses that had just been filled, and
        bumps the generation, forcing consumers to recompile away from an
        address that is in fact live. Claiming the names before the await and
        releasing them in a `finally` makes the second caller skip them
        instead, so each name is resolved at most once per interval no matter
        how the two callers overlap."""
        now = time.monotonic()
        due = self._due_names(now)
        if not due:
            return
        self._in_flight.update(due)
        try:
            results = await asyncio.gather(
                *(self._resolve_name(name) for name in due),
                return_exceptions=True,
            )
            for name, result in zip(due, results):
                self._apply_result(name, now, result)
        finally:
            # `finally`, not a trailing statement: a cancelled tick (shutdown)
            # or a raising _apply_result must not strand a name as permanently
            # in-flight, which would silently stop refreshing it forever.
            self._in_flight.difference_update(due)

    async def _resolve_name(self, name: str) -> frozenset:
        """Raises on failure (caught by resolve_once via
        return_exceptions=True); a caller sees either a real resolved set
        or an exception, never a partial one."""
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(name, None)
        return _normalized_addresses(infos)

    def _apply_result(self, name: str, now: float, result) -> None:
        entry = self._state.setdefault(name, _NameState())
        entry.last_checked_monotonic = now
        if isinstance(result, BaseException):
            entry.consecutive_failures += 1
            if entry.addresses:
                # fail-closed design: a name that WAS resolving
                # and just stopped is cleared immediately, not kept as
                # last-known-good -- see the module docstring. Clearing IS
                # an addresses_for() change, so it counts as a generation
                # bump too (a consumer must recompile away from the now-
                # stale address, same as any other real change).
                entry.addresses = frozenset()
                self._generation += 1
                log.warning(
                    "fastlane name resolver: %s stopped resolving (attempt "
                    "%d) -- clearing its cached addresses; the rule fails "
                    "closed until it resolves again: %s",
                    name, entry.consecutive_failures, result,
                )
            else:
                log.warning(
                    "fastlane name resolver: %s did not resolve (attempt %d): %s",
                    name, entry.consecutive_failures, result,
                )
            return
        entry.consecutive_failures = 0
        if result != entry.addresses:
            entry.addresses = result
            self._generation += 1
            log.info(
                "fastlane name resolver: %s -> %s",
                name, sorted(str(a) for a in result),
            )

    async def run_periodic(self, tick_s: float = _TICK_S) -> None:
        """Runs until cancelled. Never raises out of the loop -- one
        resolve_once exception must not cost every future tick too
        (mirrors plugin_health.py's run_periodic). Caller is expected to
        `await resolve_once()` once directly at boot (same two-step
        pattern plugin_health.py's PluginHealthMonitor uses: a direct
        best-effort `probe_once()` at boot, then `run_periodic` handed to
        `asyncio.create_task` for the lifespan) before starting this loop,
        so the cache is populated before the first admission-time read
        rather than starting empty for up to one tick."""
        while True:
            await asyncio.sleep(tick_s)
            try:
                await self.resolve_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("fastlane name resolver tick failed (best-effort)")


# --- Discovered-list client names (reverse lookup, confirmed forward) -------
# Every value below is an UNVERIFIED starting default (not tuned to any real network) and is meant to be
# tuned once real lookup times are known.
_NAMER_TICK_S = 5.0  # UNVERIFIED: scheduling granularity of the naming loop; tune.
_NAMER_ROWS_PER_TICK = 4  # UNVERIFIED: most rows looked up in one tick; tune.
_NAMER_LOOKUP_TIMEOUT_S = 2.0  # UNVERIFIED: cap on any one lookup; tune.
_NAMER_CONFIRM_INTERVAL_S = 300.0  # UNVERIFIED: how often a shown name is re-confirmed; tune.
_NAMER_MISS_BASE_S = 30.0  # UNVERIFIED: first wait before retrying an unnamed row; tune.
_NAMER_MISS_MAX_S = 600.0  # UNVERIFIED: longest wait before retrying an unnamed row; tune.
_NAMER_WORKERS = 2  # UNVERIFIED: lookup threads owned by the namer; tune.


@dataclass
class _NamerState:
    last_checked: float = 0.0
    misses: int = 0


class FastLaneClientNamer:
    """Finds the container name behind each Discovered address, for display.

    `targets_fn()` returns `(census_key, current_name_or_None)` pairs, asked
    fresh every tick; `set_name_fn(census_key, name_or_None)` records a
    result. Both are plain synchronous callables.

    A name is recorded only when a forward lookup of exactly that name
    returned the row's own address (`confirmed_name`), and it is dropped the
    same tick a later confirmation fails: docker hands a freed address to the
    next container, so a name that stops confirming is never kept as
    last-known-good. A row with no name is retried on an exponential
    backoff, since most unnamed rows stay unnamed.

    The lookups are blocking stdlib calls, so they run in an executor that
    this object owns. They must never use the event loop's default executor
    (which the request path shares): a reverse lookup of an address nothing
    answers for can take seconds. The reverse and forward lookups are
    injectable (`reverse_fn`, `forward_fn`) so tests need no network.

    Nothing here is read by the request path; it only writes display names
    through `set_name_fn`."""

    def __init__(self, targets_fn, set_name_fn, *, reverse_fn=None,
                 forward_fn=None, clock=time.monotonic) -> None:
        self._targets_fn = targets_fn
        self._set_name_fn = set_name_fn
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_NAMER_WORKERS, thread_name_prefix="fastlane-names",
        )
        self._reverse_fn = reverse_fn if reverse_fn is not None else self._default_reverse
        self._forward_fn = forward_fn if forward_fn is not None else self._default_forward
        self._clock = clock
        self._state: dict[str, _NamerState] = {}
        # Rows with a lookup in the air right now; claimed before the await
        # and released in a `finally`, so two overlapping ticks never look the
        # same row up twice (same discipline as FastLaneNameResolver).
        self._in_flight: set[str] = set()

    async def _default_reverse(self, address: str) -> str:
        loop = asyncio.get_running_loop()
        host, _service = await loop.run_in_executor(
            self._executor, socket.getnameinfo, (address, 0), socket.NI_NAMEREQD,
        )
        return host

    async def _default_forward(self, name: str) -> frozenset:
        loop = asyncio.get_running_loop()
        infos = await loop.run_in_executor(
            self._executor, socket.getaddrinfo, name, None,
        )
        return _normalized_addresses(infos)

    def _interval_for(self, state: _NamerState, has_name: bool) -> float:
        if has_name and state.misses == 0:
            return _NAMER_CONFIRM_INTERVAL_S
        if state.misses == 0:
            return _NAMER_MISS_BASE_S
        return min(_NAMER_MISS_BASE_S * 2 ** (state.misses - 1), _NAMER_MISS_MAX_S)

    def _due_rows(self, targets: dict, now: float) -> list[str]:
        due = []
        for key, current in targets.items():
            if key in self._in_flight:
                continue
            state = self._state.get(key)
            if state is None:
                due.append((float("-inf"), key))
            elif now - state.last_checked >= self._interval_for(state, bool(current)):
                due.append((state.last_checked, key))
        # Oldest-checked first (never-checked rows before any checked one);
        # the sort is stable, so ties keep the order the targets came in.
        due.sort(key=lambda item: item[0])
        return [key for _checked, key in due[:_NAMER_ROWS_PER_TICK]]

    async def _lookup(self, call, arg):
        return await asyncio.wait_for(call(arg), _NAMER_LOOKUP_TIMEOUT_S)

    async def _name_for(self, key: str):
        """The confirmed name for one census key, or None. Raises when the
        key is not an address or the reverse lookup fails or times out."""
        address = normalize_address(key)
        # The NORMALIZED address is looked up, so an IPv4-mapped row is asked
        # about as the plain IPv4 address the name server knows.
        answer = await self._lookup(self._reverse_fn, str(address))
        candidates = name_candidates(answer)
        if not candidates:
            return None
        results = await asyncio.gather(
            *(self._lookup(self._forward_fn, name) for name in candidates),
            return_exceptions=True,
        )
        # A candidate whose forward lookup failed simply has no addresses.
        forward = {
            name: result
            for name, result in zip(candidates, results)
            if not isinstance(result, BaseException)
        }
        return confirmed_name(address, forward)

    def _apply_result(self, key: str, current, now: float, result) -> None:
        state = self._state.setdefault(key, _NamerState())
        state.last_checked = now
        name = None if isinstance(result, BaseException) else result
        try:
            if name:
                state.misses = 0
                if name != current:
                    self._set_name_fn(key, name)
                    log.info("fastlane client namer: %s is %s", key, name)
                return
            state.misses += 1
            log.debug(
                "fastlane client namer: no confirmed name for %s (miss %d): %s",
                key, state.misses, result,
            )
            if current:
                # Fail closed: a name that stopped confirming is dropped now.
                self._set_name_fn(key, None)
                log.info(
                    "fastlane client namer: %s no longer confirms as %s; name dropped",
                    key, current,
                )
        except Exception:
            # One row's failure to record must not touch the others. It counts
            # as a miss (once: a lookup that already missed has been counted).
            state.misses = max(state.misses, 1)
            log.exception("fastlane client namer: could not record the name for %s", key)

    async def tick_once(self) -> None:
        """Looks up the rows that are due, concurrently, at most
        `_NAMER_ROWS_PER_TICK` of them, oldest-checked first."""
        now = self._clock()
        targets = dict(self._targets_fn())
        for key in [k for k in self._state if k not in targets]:
            del self._state[key]
        due = self._due_rows(targets, now)
        if not due:
            return
        self._in_flight.update(due)
        try:
            results = await asyncio.gather(
                *(self._name_for(key) for key in due), return_exceptions=True,
            )
            for key, result in zip(due, results):
                self._apply_result(key, targets[key], now, result)
        finally:
            # `finally`: a cancelled tick must not strand a row as in flight.
            self._in_flight.difference_update(due)

    def close(self) -> None:
        """Stops the lookup threads without waiting for a stuck lookup.
        Safe to call more than once."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def run_periodic(self, tick_s: float = _NAMER_TICK_S) -> None:
        """Runs until cancelled, then closes the executor. One failing tick
        must not cost every later tick."""
        try:
            while True:
                await asyncio.sleep(tick_s)
                try:
                    await self.tick_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("fastlane client namer tick failed (best-effort)")
        finally:
            self.close()
