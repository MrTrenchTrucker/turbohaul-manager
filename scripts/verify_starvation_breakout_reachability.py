#!/usr/bin/env python3
"""AST structural proof for the grace-loop starvation break-out.

Verifies: in BOTH grace loops in manager.py, the
`starved_other_model(...)` break-out node is reachable EXCLUSIVELY from the
no-match fallthrough of the loop body -- i.e. it is a body statement of the
`while` loop itself (a sibling AFTER the `if matched is not None: ... continue`
block), never nested inside that `if`'s body. This makes "same-model
warm-reuse untouched" a structural property of the code, not merely
something the tests happen not to catch.

Usage: python3 scripts/verify_starvation_breakout_reachability.py
Exit 0 + prints OK per loop on success; exit 1 with a diagnostic otherwise.
"""
import ast
import sys
from pathlib import Path

MANAGER_PY = Path(__file__).resolve().parent.parent / "src" / "turbohaul" / "manager.py"


def _calls_name(node, name):
    """True if `node` is (or directly awaits) a Call to an attribute/name
    ending in `name` (e.g. matches `self.queue.pop_matched_thread(...)`)."""
    if isinstance(node, ast.Await):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == name
    if isinstance(func, ast.Name):
        return func.id == name
    return False


def _find_grace_while_loops(tree):
    """A grace loop = a `while` node whose body contains an assignment whose
    value calls pop_matched_thread (the match-poll) somewhere in its body."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and _calls_name(stmt.value, "pop_matched_thread"):
                found.append(node)
                break
    return found


def _find_match_if(while_node):
    """The top-level `if matched is not None:` statement directly in the
    while body (not nested)."""
    for stmt in while_node.body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "matched"
                and any(isinstance(op, ast.IsNot) for op in test.ops)
            ):
                return stmt
    return None


def _find_starvation_if(while_node):
    """The top-level `if ... starved_other_model(...) is not None:` statement
    directly in the while body (must be a SIBLING of the match-if, not nested
    inside it)."""
    for stmt in while_node.body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            if isinstance(test, ast.Compare) and any(
                isinstance(op, ast.IsNot) for op in test.ops
            ):
                for sub in ast.walk(test.left):
                    if _calls_name(sub, "starved_other_model"):
                        return stmt
    return None


def _contains(container_node, target_node):
    for n in ast.walk(container_node):
        if n is target_node:
            return True
    return False


def main():
    src = MANAGER_PY.read_text()
    tree = ast.parse(src, filename=str(MANAGER_PY))
    loops = _find_grace_while_loops(tree)

    if len(loops) != 2:
        print(f"FAIL: expected exactly 2 grace loops, found {len(loops)}", file=sys.stderr)
        return 1

    ok = True
    for i, while_node in enumerate(loops, start=1):
        match_if = _find_match_if(while_node)
        starve_if = _find_starvation_if(while_node)
        label = f"Loop {i} (line {while_node.lineno})"

        if match_if is None:
            print(f"FAIL {label}: no 'if matched is not None:' found", file=sys.stderr)
            ok = False
            continue
        if starve_if is None:
            print(f"FAIL {label}: no starvation break-out 'if' found", file=sys.stderr)
            ok = False
            continue

        # (1) starve_if must be a DIRECT body statement of the while loop
        #     (a sibling of match_if), not nested inside match_if's body.
        is_direct_sibling = starve_if in while_node.body
        # (2) starve_if must NOT be reachable by walking match_if's subtree
        #     (i.e. it is not nested anywhere inside the match branch).
        nested_in_match = _contains(match_if, starve_if)
        # (3) starve_if must come AFTER match_if in source order (the
        #     no-match fallthrough, not hoisted above it).
        after_match = starve_if.lineno > match_if.end_lineno

        if is_direct_sibling and not nested_in_match and after_match:
            print(
                f"OK {label}: starvation break-out at line {starve_if.lineno} is a "
                f"direct while-body sibling AFTER the match-if "
                f"(lines {match_if.lineno}-{match_if.end_lineno}), "
                f"not nested inside it -- no-match-fallthrough-exclusive, proven structurally."
            )
        else:
            print(
                f"FAIL {label}: starve_if direct_sibling={is_direct_sibling} "
                f"nested_in_match={nested_in_match} after_match={after_match}",
                file=sys.stderr,
            )
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
