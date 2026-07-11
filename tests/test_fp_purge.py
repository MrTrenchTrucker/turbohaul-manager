"""Standalone behavioral test for WIN 4 fingerprint purge. Run: PYTHONPATH=src python3 scratchpad/test_fp_purge.py"""
import os, json, sys, types, tempfile, shutil
import turbohaul.manager as M
import turbohaul.subprocess_mgr as SM
from turbohaul.manager import TurbohaulManager as T

FP_A = {"gguf_sha256": "a"*64, "engine_build_id": "b"*64, "n_ctx": 4096, "n_rs_seq": 1}
FP_B = {"gguf_sha256": "c"*64, "engine_build_id": "b"*64, "n_ctx": 8192, "n_rs_seq": 2}
CUR = {"modelA": FP_A, "modelB": FP_B}  # modelZ absent -> unknown current identity

def meta(model_tag, fp=None, extra=None):
    d = {"thread_id": "t", "thread_hash": "h", "prompt_tokens": 10, "prompt_len": 5,
         "hash_chain": ["x"], "model_tag": model_tag, "slot_id": 0, "port": 1, "clean_prefix": False}
    if fp: d.update(fp)
    if extra: d.update(extra)
    return d

def write(d, base, model_tag, fp=None, extra=None):
    (open(os.path.join(d, base + ".bin"), "w")).write("KVDATA")
    json.dump(meta(model_tag, fp, extra), open(os.path.join(d, base + ".json"), "w"))

def make_stub(cur_map, idle=None):
    s = types.SimpleNamespace()
    s._idle_handle = idle[0] if idle else None
    s._idle_model_tag = idle[1] if idle else None
    s._idle_thread_id = idle[2] if idle else None
    s._engine_fingerprint = lambda mt: cur_map.get(mt, {"gguf_sha256": None, "engine_build_id": None, "n_ctx": None, "n_rs_seq": None})
    s._fingerprint_matches = T._fingerprint_matches
    s._fp_summary = T._fp_summary
    s._thread_hash = T._thread_hash  # staticmethod: ""->"nothread"
    s._find_clean_bin = types.MethodType(T._find_clean_bin, s)  # real, reads SLOT_SAVE_DIR
    s._purge_protected_basenames = types.MethodType(T._purge_protected_basenames, s)
    return s

def run():
    d = tempfile.mkdtemp(prefix="fp_purge_")
    SM.SLOT_SAVE_DIR = d
    # matching current-build bin (RED-HAT c: must NOT be purged)
    write(d, "modelA.p1.h.slot0", "modelA", FP_A)
    # mismatched gguf (re-quantized same tag) -> purge
    write(d, "modelA.p1.h.slot1", "modelA", {**FP_A, "gguf_sha256": "d"*64})
    # mismatched n_ctx (ctx bumped) -> purge
    write(d, "modelA.p1.h.slot2", "modelA", {**FP_A, "n_ctx": 999})
    # mismatched n_rs_seq (parallel changed) -> purge
    write(d, "modelA.p1.h.slot3", "modelA", {**FP_A, "n_rs_seq": 9})
    # unstamped legacy bin (no fp fields) -> purge
    write(d, "modelA.p1.h.slot4", "modelA")
    # different model, matching ITS manifest -> KEEP
    write(d, "modelB.p1.h.slot0", "modelB", FP_B)
    # unknown model (no current manifest / gguf None) -> KEEP (never purge blind)
    write(d, "modelZ.p1.h.slot0", "modelZ", {**FP_A})
    # metaless orphan .bin (no .json) -> untouched by json-driven sweep
    open(os.path.join(d, "orphan.p1.h.slot0.bin"), "w").write("X")

    def bins():
        return sorted(f for f in os.listdir(d) if f.endswith(".bin"))

    # 1) FLAG OFF -> no-op
    os.environ.pop("TURBOHAUL_FINGERPRINT_PURGE", None)
    n = T._purge_mismatched_bins(make_stub(CUR), reason="test")
    assert n == 0 and len(bins()) == 8, ("flag-off must no-op", n, bins())

    # 2) FLAG ON -> purge the 4 bad bins (3 mismatched + 1 unstamped)
    os.environ["TURBOHAUL_FINGERPRINT_PURGE"] = "1"
    n = T._purge_mismatched_bins(make_stub(CUR), reason="test")
    left = bins()
    assert n == 4, ("expected 4 purged", n, left)
    assert "modelA.p1.h.slot0.bin" in left, ("valid current-build bin WRONGLY purged!", left)
    assert "modelB.p1.h.slot0.bin" in left, ("other-model valid bin purged!", left)
    assert "modelZ.p1.h.slot0.bin" in left, ("unknown-current-model bin purged blind!", left)
    assert "orphan.p1.h.slot0.bin" in left, ("metaless orphan touched", left)
    for gone in ("slot1", "slot2", "slot3", "slot4"):
        assert not any(gone in f for f in left), (gone + " should be purged", left)
    # meta sidecars for purged bins also gone
    assert not os.path.exists(os.path.join(d, "modelA.p1.h.slot1.json"))

    # 3) PROTECTION: a MISMATCHED bin that is the live idle clean anchor is KEPT.
    #    Re-seed a mismatched-but-clean bin for modelA/thread h, mark clean_prefix,
    #    set idle holder -> _find_clean_bin elects it -> protected from purge.
    shutil.rmtree(d); os.makedirs(d); SM.SLOT_SAVE_DIR = d
    # idle thread_id "" -> thread_hash "nothread"; filename must match .p7.nothread.
    write(d, "modelA.p7.nothread.slot0", "modelA", {**FP_A, "gguf_sha256": "d"*64},
          extra={"clean_prefix": True, "hash_chain": ["x", "y"], "thread_hash": "nothread", "slot_id": 0})
    idle_handle = types.SimpleNamespace(port=7)
    n = T._purge_mismatched_bins(make_stub(CUR, idle=(idle_handle, "modelA", "")), reason="test")
    assert n == 0 and bins() == ["modelA.p7.nothread.slot0.bin"], ("live idle anchor must be protected", n, bins())

    # 4) MOD (a): a CURRENT engine identity with NO engine_build_id (unpinned/dev
    #    binary) must SKIP the model entirely -> KEEP all its bins. Without the mod,
    #    _fingerprint_matches returns False for EVERY bin -> all purged = the footgun.
    shutil.rmtree(d); os.makedirs(d); SM.SLOT_SAVE_DIR = d
    write(d, "modelBL.p1.h.slot0", "modelBL", FP_A)
    buildless_cur = {"modelBL": {"gguf_sha256": "a"*64, "engine_build_id": None,
                                 "n_ctx": 4096, "n_rs_seq": 1}}
    os.environ["TURBOHAUL_FINGERPRINT_PURGE"] = "1"
    n = T._purge_mismatched_bins(make_stub(buildless_cur), reason="test")
    assert n == 0 and bins() == ["modelBL.p1.h.slot0.bin"], (
        "MOD a: buildless current identity must KEEP all bins (not nuke them)", n, bins())

    shutil.rmtree(d)
    print("ALL FP-PURGE BEHAVIORAL ASSERTS PASSED")

if __name__ == "__main__":
    run()
