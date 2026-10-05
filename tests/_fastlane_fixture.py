"""Shared fixture for the grace-follow-up guarantee and its designated-victim override.

Imported by BOTH the manager-side (designated-victim) and queue-side (grace
exclusion) test suites, so the two halves of that seam are exercised against
one construction of the world rather than two that merely resemble each other.

WHY A SHARED MODULE AT ALL
--------------------------
The follow-up guarantee and the victim override meet at a seam neither side can test alone:

  Follow-up guarantee: a same-thread follow-up arriving inside a resident's grace window "is
      served warm via the existing ACTIVE_MATCH cascade", and that
      window is "what a same-thread follow-up needs to be served warm".
  Victim override: the designated victim "gets NO grace timer. Not 30 seconds, not a
      warm-hold, nothing" -- so the exclusion that protects everyone
      else must NOT protect the victim.

The queue side asserts a NON-victim's follow-up is protected. The manager side
asserts a DESIGNATED VICTIM's is not. Neither assertion passes without the
other side's code, which is the point: a seam tested from one side only is a
seam that ships broken.

TWO CAPABILITIES THAT DID NOT PREVIOUSLY EXIST TOGETHER
-------------------------------------------------------
1. Ranked-rule injection WITH tag_ranks at manager level. This DOES exist in
   the tree (see the two-rule table in the grace-breakout suite) but the
   multi-resident boot helper builds rules with no ranks, so no single helper
   offered both. `boot_ranked_runtime` does.
2. A resident that is MID-TURN and whose identity RESOLVES. Before the
   current-turn identity stamp landed, priority resolution read a field only
   written at park, so a mid-turn resident resolved to None -- and starved
   residents are typically all mid-turn. `drive_to_active`
   builds that resident; `assert_resolves` is the guard that it really does.

TRAPS MEASURED WHILE BUILDING THIS, SO YOU DO NOT PAY FOR THEM AGAIN
--------------------------------------------------------------------
* `max_parallel_sidecars` IS A HARD CEILING ON *ACTIVE* RESIDENTS. With the
  default of 2, a third `drive_to_active` NEVER SUCCEEDS -- and it fails as a
  five-second timeout in `wait_until`, which reads exactly like a scheduling
  bug in the code under test. It is not: it is the fixture being asked for a
  state the runtime cannot hold. Raise the cap or release a gate first.
* A resident driven with a `client_meta` that matches NO rule resolves to None.
  That is correct behaviour, not a broken fixture -- `assert_resolves` refuses
  it deliberately, and that refusal is the control proving the guard can fail.
* `seed_manifest` needs a DISTINCT `main_gpu` per model for co-residence. Two
  models on one card silently give you one resident and a queue.

⛔ TREE REQUIREMENT — READ BEFORE IMPORTING
-------------------------------------------
`assert_resolves` calls `TurbohaulManager._resident_priority_key`, which does NOT exist on every
tree. It arrived with the shared priority-resolution change. If your checkout predates that, the helper
raises a NAMED error telling you so rather than an AttributeError from the depths -- but the fix is
to PULL, not to work around it. Everything else in this module works on an older tree.

⛔ `drive_to_active` REQUIRES THE DISPATCHER ALREADY RUNNING.
`submit_and_wait` makes no progress until `worker_loop()` is a live background task, so a bare call
HANGS to `wait_until`'s five-second timeout and reads like a scheduling bug. Start it first:

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())

and cancel it when you are done. This bit the fixture's own first version.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import yaml

from turbohaul.config import (
    BootConfig, FastLaneConfig, FastLaneRule, FastLaneTagRanks, PullConfig,
    QueueConfig, RuntimeConfig, RuntimePathsConfig, ServerConfig,
    StorageConfig, UIConfig,
)
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def ranked_rules(*specs):
    """specs: (address, main_rank) pairs, in PRIORITY ORDER.

    LIST INDEX IS THE PRIORITY -- index 0 outranks index 1. Passing them in the
    wrong order silently inverts every priority assertion built on them, so the
    order here is the thing to get right, not the ranks.
    """
    return [FastLaneRule(address=a, tag_ranks=FastLaneTagRanks(main=r)) for a, r in specs]


def boot_ranked_runtime(tmp_path, *, rules=(), max_parallel_sidecars=2,
                        grace_seconds=0, max_grace_extensions=0,
                        idle_hot_load_seconds=0, max_normal_wait_s=3600.0):
    """A real BootConfig/RuntimeConfig with RANKED Fast Lane rules.

    grace_seconds defaults to 0 so a test opts IN to a grace window; a test
    asserting anything about grace must set it explicitly, which keeps an
    accidental zero from silently making a grace assertion vacuous.
    """
    root = tmp_path / "state"
    root.mkdir(exist_ok=True)
    for d in ("blobs", "manifests", "import-staging"):
        (root / d).mkdir(exist_ok=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=grace_seconds,
            max_grace_extensions=max_grace_extensions,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(
            enabled=bool(rules), rules=list(rules),
            max_normal_wait_s=max_normal_wait_s,
        ),
    )
    return boot, runtime


def seed_manifest(boot, model_tag, *, split_mode="none", main_gpu=0):
    """Co-residence needs split_mode='none' AND a DISTINCT main_gpu per model.

    Two models seeded onto the same card cannot co-reside, so a test intending
    two live residents gets one and a queue -- which looks like a scheduling
    result and is actually a fixture error.
    """
    (boot.storage.manifests_path / f"{model_tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": split_mode, "main_gpu": main_gpu},
    }))


def make_fakes(gates, *, pid_start=90000):
    """gates: model_tag -> asyncio.Event the fake completion blocks on.

    A tag NOT in `gates` completes instantly. A tag WITH an unset event stays
    mid-turn indefinitely -- that is how a resident is held ACTIVE while
    something else is asserted about it.
    """
    pid = [pid_start]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        gate = gates.get(handle.model_tag)
        if gate is not None:
            await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete


def high_vram():
    return patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000])


async def wait_until(predicate, *, timeout=5.0, interval=0.005):
    t0 = time.monotonic()
    while True:
        if predicate():
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"predicate never became true within {timeout}s")
        await asyncio.sleep(interval)


async def drive_to_active(mgr, model_tag, *, thread_id, client_meta, timeout=5.0):
    """Submit and hold a resident MID-TURN with an admission-time identity.

    Returns the in-flight task; the caller keeps it and releases the gate when
    done. The resident is ACTIVE with `active_slot` set, which is the state
    a starved resident is typically in.
    """
    task = asyncio.create_task(
        mgr.submit_and_wait(model_tag, "p", thread_id=thread_id, client_meta=client_meta)
    )
    await wait_until(
        lambda: any(r.model_tag == model_tag and r.state is ResidentState.ACTIVE
                    for r in mgr._model_residents()),
        timeout=timeout,
    )
    return task


def resident_for(mgr, model_tag):
    for r in mgr._model_residents():
        if r.model_tag == model_tag:
            return r
    raise AssertionError(f"no resident for {model_tag!r}")


def assert_resolves(mgr, model_tag):
    """Guard: this resident's identity RESOLVES right now.

    Call it before asserting anything about relative priority. Without it a
    designation test can pass because BOTH sides came back None -- agreement
    between two blind answers, which is the failure this whole seam exists to
    prevent.
    """
    r = resident_for(mgr, model_tag)
    if not hasattr(mgr, "_resident_priority_key"):
        raise AssertionError(
            "_resident_priority_key is ABSENT from this tree, so identity resolution "
            "cannot be asserted here. This is a TREE problem, not a test failure: the "
            "method arrived with the shared priority-resolution change. Pull, then re-run. Do not "
            "route around this helper -- a designation test on a tree without it would be "
            "asserting over an unresolvable."
        )
    key = mgr._resident_priority_key(r, mgr._fastlane_table())
    assert key is not None, (
        f"{model_tag}: identity did NOT resolve (state={r.state.name}). A test "
        f"built on this resident would be asserting over an unresolvable, not "
        f"over a ranked client."
    )
    return key


def install_victim_predicate(mgr, victim_tags):
    """Stand in for the manager-side designated-victim predicate.

    The real predicate does not exist in the tree yet -- it is the manager-side
    half of this seam. The queue side needs SOMETHING to call so its exclusion
    can be exercised today, and a stand-in whose behaviour is stated is honest
    where a silent default is not.

    ⛔ THE NAME IS THE CONTRACT: `_is_designated_unload_target_locked`. The queue side
    probes for it by that name; the manager side makes it exist under that name.
    If either half renames it, the exclusion silently protects everyone and the
    victim-override arm passes for the wrong reason.
    """
    tags = set(victim_tags)

    def _is_designated_unload_target_locked(r):
        return r.model_tag in tags

    mgr._is_designated_unload_target_locked = _is_designated_unload_target_locked
    return _is_designated_unload_target_locked


def victim_predicate_absent(mgr):
    """Shadow the class method with an instance-level None, so a test can
    assert the ABSENT-predicate path.

    That path must protect everyone -- degrading toward the behaviour from before the follow-up guarantee --
    never toward failing to protect. It is the state the tree is actually in
    today, so it is the one most likely to ship untested.

    NOTE: this helper replaces an earlier hasattr-then-delattr form, which
    worked
    only while the predicate could exclusively be an instance-level
    stand-in. It breaks the instant the real predicate lands as a permanent
    CLASS method -- delattr on an instance cannot remove a class attribute,
    so this raised AttributeError rather than representing "absent" at all.
    None is what "absent" means to the consumer's own
    getattr(self, "_is_designated_unload_target_locked", None) probe -- an
    instance attribute shadows the class method in normal attribute lookup,
    so getattr sees None either way.
    """
    mgr._is_designated_unload_target_locked = None
