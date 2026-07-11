"""Unit test for `_shadow_recompose_identity` (DORMANT shadow).

Imports the REAL symbol from the worktree (PYTHONPATH=<wt>/src). Proves the
safe-by-construction properties the corpus analysis will rely on:
  (a) no role/session          -> new_key == base           (append-only identity)
  (b) new_key.startswith(base) ALWAYS                        (never replaces base)
  (c) different session OR role -> different new_key         (does-SPLIT)
  (d) same inputs               -> same new_key              (per-convo stable)
  (e) raw session_id / role NOT present literally in new_key (hashed suffix)
"""
import hashlib

from turbohaul.api.chat_completion import _shadow_recompose_identity as recompose

BASE = "agent-ip-10.0.0.5-abc123def456"  # a realistic IP-fallback base


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    print("test_shadow_recompose_identity:")

    # (a) no role/session -> identity (new_key == base)
    _check("a: no fields -> new_key == base", recompose(BASE, None, None) == BASE)
    _check("a: empty-string fields -> new_key == base (falsy)",
           recompose(BASE, "", "") == BASE)

    # (b) startswith(base) ALWAYS — across every field combination
    for role, sess in [(None, None), ("main", None), (None, "sess-1"),
                       ("compression", "sess-9"), ("sub", "sess-1")]:
        _check(f"b: startswith(base) role={role!r} sess={sess!r}",
               recompose(BASE, role, sess).startswith(BASE))

    # (c) different session OR role -> different new_key (SPLIT)
    k_main_s1 = recompose(BASE, "main", "sess-1")
    k_sub_s1 = recompose(BASE, "sub", "sess-1")      # role differs
    k_main_s2 = recompose(BASE, "main", "sess-2")     # session differs
    _check("c: different role -> different key", k_main_s1 != k_sub_s1)
    _check("c: different session -> different key", k_main_s1 != k_main_s2)
    _check("c: main/sub/compression all split from base",
           len({recompose(BASE, r, "sess-1") for r in ("main", "sub", "compression")}) == 3)

    # (d) same inputs -> same new_key (deterministic / per-convo stable)
    _check("d: deterministic", recompose(BASE, "main", "sess-1") == recompose(BASE, "main", "sess-1"))

    # (e) hashed suffix — raw session_id / role NOT present literally in new_key.
    #     Uses delimiter-injection-style values to prove they cannot forge a field.
    raw_role = "sub-r=forged"
    raw_sess = "sess=x-r=evil-injection"
    k = recompose(BASE, raw_role, raw_sess)
    _check("e: raw role absent from new_key", raw_role not in k)
    _check("e: raw session_id absent from new_key", raw_sess not in k)
    # positive: the expected HASH prefixes ARE present (proves hashing occurred)
    exp_s = "s=" + hashlib.sha256(raw_sess.encode()).hexdigest()[:12]
    exp_r = "r=" + hashlib.sha256(raw_role.encode()).hexdigest()[:8]
    _check("e: hashed session prefix present", exp_s in k)
    _check("e: hashed role prefix present", exp_r in k)
    # exact shape: base + '-s=<12hex>' + '-r=<8hex>' (session before role)
    _check("e: exact append shape", k == f"{BASE}-{exp_s}-{exp_r}")

    # NO-MERGE corollary: two distinct bases keep distinct keys (base is preserved)
    other = "agent-ip-10.0.0.9-zzz"
    _check("no-merge: distinct bases stay distinct",
           recompose(BASE, "main", "s") != recompose(other, "main", "s"))

    print("ALL PASS")


if __name__ == "__main__":
    main()
