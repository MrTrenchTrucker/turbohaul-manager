"""Follow-up cover for the durable engine-identity lifecycle work.

The identity record proves OWNERSHIP. It does not prove STALENESS — a healthy
co-resident engine is recorded at spawn exactly like the dying one, so the
identity gate alone cannot tell them apart. Three defects follow from that,
and each one is pinned here:

  1. the ordinary cold spawn never recorded an identity, so on a default
     single-sidecar deployment the table stayed empty and the reaper (which
     refuses to kill what it cannot prove it owns) reaped nothing at all;
  2. the non-boot reapers passed an empty preserve-set, so a LIVE sibling
     matched the ownership gate and was killed during another engine's
     teardown;
  3. the listener diagnostic compared a Path against a string prefix, raised
     AttributeError on the first socket fd, and was swallowed by the caller's
     broad except — so it permanently reported zero.

(1) MASKS (2): recording without preserving turns an inert reaper into one
that kills live engines, so those two must never be fixed apart.
"""
import ast
from pathlib import Path

import pytest

MANAGER_SRC = Path(__file__).resolve().parents[1] / "src" / "turbohaul" / "manager.py"


def _manager_tree() -> ast.Module:
    return ast.parse(MANAGER_SRC.read_text())


def _functions_calling(tree: ast.Module, attr: str) -> dict[str, ast.AST]:
    """Map {function name -> node} for every function calling self.<attr>()."""
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == attr
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"
            ):
                out[node.name] = node
                break
    return out


class TestEverySpawnRecordsItsIdentity:
    """Defect 1: an engine that is never recorded can never be reaped."""

    def test_every_spawn_site_records_its_identity(self):
        """Checked per BLOCK, not per function. A single function can hold
        several engine paths — _process_slot holds both the cold spawn and the
        idle-handle inherit — so a function-level check passes while one of its
        branches records nothing.
        """
        tree = _manager_tree()

        def self_calls(stmt, attr):
            """Calls to self.<attr>() anywhere inside one statement."""
            return [
                s for s in ast.walk(stmt)
                if isinstance(s, ast.Call)
                and isinstance(s.func, ast.Attribute)
                and s.func.attr == attr
                and isinstance(s.func.value, ast.Name)
                and s.func.value.id == "self"
            ]

        # Every spawn is judged against the branch that DIRECTLY holds it.
        # Judged against an enclosing block instead, an unrelated record in a
        # sibling branch satisfies the check and the gap stays invisible.
        parent: dict[int, ast.AST] = {}
        owner: dict[int, list] = {}
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    for stmt in block:
                        owner[id(stmt)] = block
            for child in ast.iter_child_nodes(node):
                parent[id(child)] = node

        spawn_blocks = 0
        unrecorded: list[int] = []
        seen: set[int] = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_spawn"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                continue
            # climb to the innermost statement that holds this call
            cur: ast.AST = node
            while id(cur) in parent and id(cur) not in owner:
                cur = parent[id(cur)]
            block = owner.get(id(cur))
            if block is None or id(block) in seen:
                continue
            seen.add(id(block))
            spawn_blocks += 1
            if not any(self_calls(st, "_record_engine_identity") for st in block):
                unrecorded.append(node.lineno)

        assert spawn_blocks, "no engine-spawn blocks found — test is vacuous"
        assert not unrecorded, (
            f"engine spawned at manager.py line(s) {unrecorded} without "
            "recording its durable identity. An unrecorded engine cannot be "
            "proven Turbohaul-owned, so the reaper leaves it running forever "
            "— and on a default single-sidecar deployment that is EVERY "
            "engine. Record at the spawn site; a pid assignment that reuses "
            "an already-recorded handle must NOT record again."
        )


class TestLiveSiblingIsPreserved:
    """Defect 2: ownership is not staleness."""

    def test_non_boot_reapers_preserve_the_live_sidecar_set(self):
        """boot_reconcile deliberately passes an EMPTY preserve-set (flock
        proves any other manager is dead). Every OTHER caller runs while this
        manager owns live engines and must pass them, or it kills its own.
        """
        tree = _manager_tree()
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "boot_reconcile":
                continue  # boot: empty preserve-set is correct by design
            for sub in ast.walk(node):
                if not (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "boot_orphan_reaper"
                ):
                    continue
                kw = {k.arg: k.value for k in sub.keywords}
                known = kw.get("known_pids")
                # an empty literal set()/{} is the defect
                if known is None or (
                    isinstance(known, ast.Call)
                    and isinstance(known.func, ast.Name)
                    and known.func.id == "set"
                    and not known.args
                ):
                    offenders.append(node.name)
        assert not offenders, (
            f"{offenders} call boot_orphan_reaper with no live-sidecar "
            "preserve-set while this manager owns running engines. A healthy "
            "co-resident engine is recorded at spawn just like the dying one, "
            "so the durable-identity gate matches it and SIGTERMs it. Pass "
            "self._live_handle_pids()."
        )

    def test_a_recorded_live_sibling_survives_a_teardown_sweep(self, monkeypatch):
        """Behavioural proof of the same thing, at the reaper itself."""
        from turbohaul import singleton

        stale = (1001, 11401, 555001)   # the engine being torn down
        live = (1002, 11402, 555002)    # a healthy sibling, actively serving
        cmdlines = {
            stale[0]: f"/opt/llama-server --port {stale[1]} --model a.gguf",
            live[0]: f"/opt/llama-server --port {live[1]} --model b.gguf",
        }
        starttimes = {stale[0]: stale[2], live[0]: live[2]}

        monkeypatch.setattr(singleton, "_SUBREAPER_PID", None)
        monkeypatch.setattr(singleton, "_list_proc_pids", lambda: list(cmdlines))
        monkeypatch.setattr(singleton, "_read_proc_cmdline", lambda p: cmdlines.get(p, ""))
        monkeypatch.setattr(singleton, "_read_proc_ppid", lambda p: 9999)
        monkeypatch.setattr(singleton, "_read_proc_starttime", lambda p: starttimes.get(p))
        monkeypatch.setattr(singleton, "port_listeners_in_range", lambda *a, **k: [])

        reaped: list[int] = []

        def fake_reap(pid, **kw):
            reaped.append(pid)
            return True, "sigterm-clean"

        singleton.boot_orphan_reaper(
            port_base=11400,
            known_pids={live[0]},          # what the fixed call sites pass
            reap_fn=fake_reap,
            known_engine_identities={stale, live},
        )

        assert live[0] not in reaped, (
            "a LIVE co-resident engine was reaped during another engine's "
            "teardown — it is recorded, so the ownership gate matches it; only "
            "the preserve-set can save it"
        )
        assert stale[0] in reaped, "the genuinely stale engine must still be reaped"


class TestListenerDiagnosticActuallyRuns:
    """Defect 3: a diagnostic that always throws always reports zero."""

    def test_scan_reaches_the_fd_walk_and_finds_a_real_listener(self):
        """Path.readlink() returns a Path. Comparing it to a string prefix
        raises AttributeError on the first socket fd; the caller's broad
        except then hides it, so the count silently stays 0 forever.

        A listener MUST actually exist in the scanned range: the scan returns
        early when the inode map is empty, so scanning an idle range never
        reaches the defective fd walk at all.
        """
        import socket

        from turbohaul.singleton import port_listeners_in_range

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            try:
                result = port_listeners_in_range(port, port_range_size=1)
            except AttributeError as e:  # pragma: no cover - the defect itself
                pytest.fail(
                    f"listener scan raised {e!r} — the caller swallows this, "
                    "leaving stale_listeners permanently 0"
                )
            assert any(r["port"] == port for r in result), (
                "the scan reached the fd walk but did not report a socket we "
                "are certain is listening — the diagnostic is not working"
            )
        finally:
            srv.close()
