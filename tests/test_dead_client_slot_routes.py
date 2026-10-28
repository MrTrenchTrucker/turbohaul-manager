"""A request that is still waiting must not outlive its client, seen through the real routes.

Shape of every test: a manager with one engine slot, a first (live) request mid-turn on that
engine, and a second request from another client that the REAL chat route accepted and that
now waits behind the first. The second client then hangs up (a raw socket close) before its
request was admitted. The routes run behind an in-process loopback server; nothing in these
tests sets a slot flag, evicts a slot or releases a claim.

Contract pinned for a request whose client is gone and whose engine turn has not started:
  * the slot is flagged (``disconnect_event`` set, ``is_evicted`` True) within a bounded time;
  * it is removed from wherever it waits (staging queue, accept buffer, a resident's inbox);
  * its claim is released (nobody else of that client waits, so there is nothing to move to);
  * when the busy engine's turn ends, no engine turn runs for it and no engine is started for it.
Each test has a live-client control with the same shape where the client stays connected: that
request is NOT evicted and IS served when the busy turn ends.

A client that hangs up after its engine turn started is a different case and is not covered.
"""

import asyncio
import time

from _dead_client_routes_support import (
    MODEL,
    OBSERVE_WINDOW_S,
    SERVED_WINDOW_S,
    describe,
    end_holder_turn_and_watch,
    hang_up_and_observe,
    route_world,
)

CHAT_PATH = "/v1/chat/completions"
OLLAMA_PATH = "/api/chat"
HOLD_LIVE_CLIENT_S = 3.0


def _payload(stream):
    return {"model": MODEL, "stream": stream, "messages": [{"role": "user", "content": "hello"}]}


async def _dead_client_scenario(tmp_path, path, payload):
    """Park a request behind a busy engine, hang its client up, observe, end the busy turn."""
    async with route_world(tmp_path) as world:
        await world.start_holder()
        spawns_before = len(world.spawns)
        client = await world.open_waiting_client(path, payload)
        first, at_end = await hang_up_and_observe(world, client)
        slot = client.slot
        turn_ran = await end_holder_turn_and_watch(world, slot)
        return {
            "first": first,
            "turn_ran": turn_ran,
            "extra_spawns": len(world.spawns) - spawns_before,
            "places_at_window_end": at_end["places"],
            "claims_at_window_end": at_end["claims"],
        }


def _verdict(label, seen):
    """Collect every violated part of the contract; print what was measured."""
    first = seen["first"]
    print(f"[{label}] after the hang-up: {describe(first)}; "
          f"turn_for_dead_slot={seen['turn_ran']}; extra_engine_starts={seen['extra_spawns']}; "
          f"waiting_in_at_window_end={seen['places_at_window_end']}; "
          f"claims_at_window_end={seen['claims_at_window_end']}")
    bad = []
    if "flag_set" not in first:
        bad.append(f"the slot's disconnect flag was never set within {OBSERVE_WINDOW_S}s of the hang-up")
    if "marked_evicted" not in first:
        bad.append(f"the slot was never marked evicted within {OBSERVE_WINDOW_S}s of the hang-up")
    if "left_queue" not in first:
        bad.append(
            f"the slot was still waiting (in {seen['places_at_window_end']}) {OBSERVE_WINDOW_S}s after the hang-up"
        )
    if "claim_released" not in first:
        bad.append(f"its claim was still registered {OBSERVE_WINDOW_S}s after the hang-up")
    if seen["turn_ran"]:
        bad.append("an engine turn ran for the slot of a client that was gone")
    if seen["extra_spawns"]:
        bad.append(f"{seen['extra_spawns']} engine(s) were started while the dead client's request waited")
    return bad


async def test_streaming_route_hangup_while_waiting_removes_the_request(tmp_path):
    seen = await _dead_client_scenario(tmp_path, CHAT_PATH, _payload(stream=True))
    bad = _verdict("streaming", seen)
    assert not bad, "streaming route, client hung up while waiting: " + " | ".join(bad)


async def test_non_streaming_route_hangup_while_waiting_removes_the_request(tmp_path):
    seen = await _dead_client_scenario(tmp_path, CHAT_PATH, _payload(stream=False))
    bad = _verdict("non-streaming", seen)
    assert not bad, "non-streaming route, client hung up while waiting: " + " | ".join(bad)


async def test_ollama_chat_route_hangup_while_waiting_removes_the_request(tmp_path):
    seen = await _dead_client_scenario(tmp_path, OLLAMA_PATH, _payload(stream=False))
    bad = _verdict("ollama-chat", seen)
    assert not bad, "ollama-style chat route, client hung up while waiting: " + " | ".join(bad)


async def _live_client_scenario(tmp_path, path, payload, *, expect_ok_status):
    """Same shape, but the second client stays connected."""
    async with route_world(tmp_path) as world:
        await world.start_holder()
        client = await world.open_waiting_client(path, payload)
        slot = client.slot
        # Long enough for the route's disconnect watcher to poll at least once more.
        deadline = time.monotonic() + HOLD_LIVE_CLIENT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        waiting_while_busy = world.places(slot)
        claim_while_busy = world.claim_rows(slot)
        flag_while_busy = slot.disconnect_event.is_set()
        evicted_while_busy = slot.is_evicted
        served = await end_holder_turn_and_watch(world, slot, window_s=SERVED_WINDOW_S)
        status = None
        if expect_ok_status:
            status = await client.status_line(timeout=5.0)
        return {
            "waiting_while_busy": waiting_while_busy,
            "claim_while_busy": claim_while_busy,
            "flag_while_busy": flag_while_busy,
            "evicted_while_busy": evicted_while_busy,
            "served": served,
            "evicted_at_end": slot.is_evicted,
            "flag_at_end": slot.disconnect_event.is_set(),
            "status": status,
        }


def _live_verdict(label, seen, *, expect_ok_status):
    print(f"[{label} control] {seen}")
    bad = []
    if not seen["waiting_while_busy"]:
        bad.append("the connected client's request left the queue while the engine was busy")
    if not seen["claim_while_busy"]:
        bad.append("the connected client's claim was released while it waited")
    if seen["flag_while_busy"] or seen["flag_at_end"]:
        bad.append("the disconnect flag of a connected client was set")
    if seen["evicted_while_busy"] or seen["evicted_at_end"]:
        bad.append("a connected client's request was marked evicted")
    if not seen["served"]:
        bad.append(
            f"the connected client's request got no engine turn within {SERVED_WINDOW_S}s of the busy turn ending"
        )
    if expect_ok_status and not (seen["status"] or "").startswith("HTTP/1.1 200"):
        bad.append(f"the connected client did not get a 200 response, got {seen['status']!r}")
    return bad


async def test_streaming_route_live_client_keeps_waiting_and_is_served(tmp_path):
    seen = await _live_client_scenario(tmp_path, CHAT_PATH, _payload(stream=True), expect_ok_status=False)
    bad = _live_verdict("streaming", seen, expect_ok_status=False)
    assert not bad, "streaming route, client stayed connected: " + " | ".join(bad)


async def test_non_streaming_route_live_client_keeps_waiting_and_is_served(tmp_path):
    seen = await _live_client_scenario(tmp_path, CHAT_PATH, _payload(stream=False), expect_ok_status=True)
    bad = _live_verdict("non-streaming", seen, expect_ok_status=True)
    assert not bad, "non-streaming route, client stayed connected: " + " | ".join(bad)
