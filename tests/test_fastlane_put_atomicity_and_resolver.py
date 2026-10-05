"""Two independent concurrency hazards that share one trigger.

Both come from the Fast Lane container_name resolve pass, and both are only
reachable when at least one rule identifies a client by container_name -- an
address-only rule table never resolves anything, so `resolve_once` returns
before its `await` and neither hazard can fire.

HAZARD 1 -- PUT /api/config must stay atomic.
  Read -> merge -> apply -> persist must complete inside a single event loop
  turn, with no `await` in between: atomic BY CONSTRUCTION, a property that costs
  nothing to keep until a suspension point appears in that span. An
  `await resolver.resolve_once()` between the in-memory apply and the on-disk
  persist would be such a suspension point, so two overlapping PUTs could apply
  in one order and persist in the other -- both returning ok, with the live
  manager and runtime_config.yaml then disagreeing until a restart silently
  adopts the value the operator watched LOSE.
  (The structural test below pins that no await sits in that span.)
  Only ONE of the two requests has to suspend: the other runs to completion
  inside the first one's suspension.

HAZARD 2 -- FastLaneNameResolver.resolve_once must be re-entrant.
  Due-ness is READ from `last_checked_monotonic` before the gather and only
  WRITTEN after it. Two callers share one resolver instance and neither
  serialises the other -- the periodic tick (every _TICK_S) and the PUT path.
  Both would therefore see the same unmarked entry and issue the same lookup, and
  the results would apply in COMPLETION order, so a slow FAILING lookup could
  clear addresses that a fast SUCCEEDING one had just filled.

Fixing one does NOT fix the other: an in-flight guard makes the second caller
return without awaiting, but the FIRST caller is still suspended inside the
gather it owns, which is all Hazard 1 needs.
"""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from turbohaul.api.config_put import load_runtime_override, put_config
from turbohaul.config import RuntimeConfig
from turbohaul.fastlane_resolve import FastLaneNameResolver

NAME_RULES = [{"container_name": "svc-a", "label": "on-network"}]
ADDR_RULES = [{"address": "192.168.1.9", "label": "off-network"}]


def _install_getaddrinfo(delays, fail_indices=()):
    """Deterministic, slow, per-call-ordered lookups. Returns the call log."""
    calls = []

    async def fake(host, port, *a, **kw):
        i = len(calls)
        calls.append(host)
        await asyncio.sleep(delays[min(i, len(delays) - 1)])
        if i in fail_indices:
            raise OSError(f"synthetic lookup failure #{i}")
        return [(2, 1, 6, "", (f"10.0.0.{i + 1}", 0))]

    asyncio.get_running_loop().getaddrinfo = fake
    return calls


def _make_mgr(tmp_path, rules):
    runtime = RuntimeConfig(
        queue={"grace_seconds": 10},
        pull={},
        fastlane={"enabled": True, "rules": rules},
    )
    mgr = SimpleNamespace(
        runtime=runtime,
        boot=SimpleNamespace(storage=SimpleNamespace(state_db_path=tmp_path / "state.sqlite")),
        grace=SimpleNamespace(grace_seconds=10, max_extensions=1),
        idle=SimpleNamespace(idle_seconds=1),
        _fastlane_census={},
    )
    mgr.fastlane_rule_names = lambda: {
        r["container_name"]
        for r in mgr.runtime.fastlane.model_dump()["rules"]
        if r.get("container_name")
    }
    mgr._fastlane_resolver = FastLaneNameResolver(mgr.fastlane_rule_names)
    return mgr


def _request(mgr):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(manager=mgr)))


async def _two_overlapping_puts(tmp_path, rules):
    """Both PUTs write the SAME field to DIFFERENT values and both carry a
    "fastlane" key, so both reach the gated resolve pass."""
    mgr = _make_mgr(tmp_path, rules)
    calls = _install_getaddrinfo([0.30, 0.05])

    t1 = asyncio.create_task(
        put_config({"queue": {"grace_seconds": 111}, "fastlane": {"enabled": True}}, _request(mgr))
    )
    await asyncio.sleep(0)  # let PUT-1 reach its first suspension point
    t2 = asyncio.create_task(
        put_config({"queue": {"grace_seconds": 222}, "fastlane": {"enabled": True}}, _request(mgr))
    )
    r1, r2 = await asyncio.gather(t1, t2)

    memory = mgr.runtime.queue.grace_seconds
    disk = (load_runtime_override(mgr.boot.storage.state_db_path) or {}) \
        .get("queue", {}).get("grace_seconds", "<absent>")
    return memory, disk, calls, (r1, r2)


class TestThePutStaysAtomicAcrossTheResolvePass:
    async def test_two_overlapping_puts_leave_memory_and_disk_agreeing(self, tmp_path):
        memory, disk, calls, (r1, r2) = await _two_overlapping_puts(tmp_path, NAME_RULES)
        assert calls, "the resolve pass never ran -- this arm did not exercise the defect"
        assert r1["status"] == "ok" and r2["status"] == "ok"
        assert memory == disk, (
            f"live runtime says {memory} but runtime_config.yaml says {disk} -- "
            "a restart would silently adopt the value the operator watched lose"
        )

    async def test_CONTROL_address_only_rules_never_reach_the_await(self, tmp_path):
        """Scoping control. With no container_name anywhere, `_due_names` is
        empty and `resolve_once` returns before its gather, so this arm CANNOT
        diverge. It passing tells you nothing on its own -- which is exactly
        why the arm above has to assert that a lookup really happened."""
        memory, disk, calls, _ = await _two_overlapping_puts(tmp_path, ADDR_RULES)
        assert calls == [], f"expected no lookups for an address-only table, got {calls}"
        assert memory == disk

    def test_no_await_may_sit_between_the_apply_and_the_persist(self):
        """The rot-proof half, and the one that will actually catch a
        regression. The behavioural test above only fails if someone recreates
        this exact race; this fails the moment ANY suspension point is
        introduced into the span, which is the most likely way for this race to
        return."""
        import turbohaul.api.config_put as mod

        tree = ast.parse(Path(mod.__file__).read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "put_config"
        )
        apply_ln = persist_ln = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == "runtime":
                        apply_ln = node.lineno
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "save_runtime_config":
                persist_ln = node.lineno

        # Anti-vacuity: if either anchor is renamed away this test must fail
        # loudly rather than silently pass by finding nothing to check.
        assert apply_ln is not None, "could not find `mgr.runtime = ...` -- anchor moved"
        assert persist_ln is not None, "could not find save_runtime_config(...) -- anchor moved"
        assert apply_ln < persist_ln, "the persist no longer follows the apply"

        awaits = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Await)]
        assert awaits, "put_config has no await at all -- the instrument would pass vacuously"
        between = [ln for ln in awaits if apply_ln < ln < persist_ln]
        assert between == [], (
            f"await(s) at line(s) {between} sit between the apply at {apply_ln} and the "
            f"persist at {persist_ln}; that window is what two concurrent PUTs interleave in"
        )


class TestTheResolverToleratesOverlappingPasses:
    async def test_overlapping_passes_issue_ONE_lookup_per_name(self, tmp_path):
        mgr = _make_mgr(tmp_path, NAME_RULES)
        r = mgr._fastlane_resolver
        calls = _install_getaddrinfo([0.20, 0.05])

        t1 = asyncio.create_task(r.resolve_once())
        await asyncio.sleep(0)
        t2 = asyncio.create_task(r.resolve_once())
        await asyncio.gather(t1, t2)

        assert calls == ["svc-a"], f"expected exactly one lookup, got {calls}"

    async def test_a_late_FAILING_lookup_cannot_clear_a_freshly_resolved_address(self, tmp_path):
        """The harm, not just the duplicate work. First pass succeeds quickly;
        the overlapping second pass would fail slowly and, applying last, clear
        an address that is in fact live -- bumping the generation and forcing
        every consumer to recompile away from a working rule."""
        mgr = _make_mgr(tmp_path, NAME_RULES)
        r = mgr._fastlane_resolver
        _install_getaddrinfo([0.05, 0.30], fail_indices={1})

        t1 = asyncio.create_task(r.resolve_once())
        await asyncio.sleep(0)
        t2 = asyncio.create_task(r.resolve_once())
        await asyncio.gather(t1, t2)

        # str() because the cache stores normalize_address()'s ipaddress
        # objects, not the raw strings the stub handed back.
        assert {str(a) for a in r.addresses_for("svc-a")} == {"10.0.0.1"}, (
            "a live name was left with no cached address -- the rule now fails closed "
            "against a container that resolves perfectly well"
        )

    async def test_CONTROL_a_single_pass_still_resolves_normally(self, tmp_path):
        """Proves the in-flight claim does not simply suppress resolution. If
        the guard were over-broad -- never releasing, or claiming names it
        never resolves -- this is what would catch it."""
        mgr = _make_mgr(tmp_path, NAME_RULES)
        r = mgr._fastlane_resolver
        calls = _install_getaddrinfo([0.0])

        await r.resolve_once()
        assert calls == ["svc-a"]
        assert {str(a) for a in r.addresses_for("svc-a")} == {"10.0.0.1"}

    async def test_a_CANCELLED_pass_does_not_strand_the_name_in_flight(self, tmp_path):
        """The claim is released in a `finally` on purpose. A tick cancelled at
        shutdown must not leave a name permanently claimed -- that would stop it
        ever refreshing again, which is worse and quieter than the duplicate-lookup
        hazard this guard prevents."""
        mgr = _make_mgr(tmp_path, NAME_RULES)
        r = mgr._fastlane_resolver
        _install_getaddrinfo([5.0])

        t = asyncio.create_task(r.resolve_once())
        await asyncio.sleep(0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

        calls = _install_getaddrinfo([0.0])
        await r.resolve_once()
        assert calls == ["svc-a"], (
            "the name was still claimed after a cancelled pass, so it stopped "
            f"being refreshed entirely (lookups this pass: {calls})"
        )
