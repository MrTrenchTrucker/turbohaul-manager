"""M2b flag + flip semantics. Run: PYTHONPATH=src python3 tests/test_m2b_activate.py"""
import os, sys
os.environ.pop("TURBOHAUL_M2B_ACTIVE", None)
from turbohaul.api.chat_completion import _m2b_active, _shadow_recompose_identity as R
F=[]
def ck(n,c): F.append((n,c)); print(("  ok " if c else "  FAIL ")+n)
ck("default OFF", _m2b_active() is False)
for v in ("1","true","yes","on","TRUE","On"," 1 "):
    os.environ["TURBOHAUL_M2B_ACTIVE"]=v; ck("on="+repr(v), _m2b_active() is True)
for v in ("0","false","","no","off","garbage"):
    os.environ["TURBOHAUL_M2B_ACTIVE"]=v; ck("off="+repr(v), _m2b_active() is False)
os.environ.pop("TURBOHAUL_M2B_ACTIVE", None)
base="agent-ip-1.2.3.4-auto-abcdef"
nk=R(base,"main","sess1")
ck("split: new_key startswith base + != base", nk.startswith(base) and nk!=base)
ck("no role/session -> new_key==base (flip is no-op)", R(base,None,None)==base)
ck("distinct role -> distinct key", R(base,"main","s1")!=R(base,"sub","s1"))
ck("distinct session -> distinct key", R(base,"main","s1")!=R(base,"main","s2"))
ck("deterministic (stable within convo)", R(base,"main","s1")==R(base,"main","s1"))
bad=[n for n,c in F if not c]
print(("ALL PASS" if not bad else "FAILURES: "+str(bad))); sys.exit(1 if bad else 0)
