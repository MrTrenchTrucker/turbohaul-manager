import os, re, shutil, subprocess, tempfile
import pytest

def _cuobjdump():
    return shutil.which("cuobjdump")

def test_host_vendor_binary_is_sm86():
    if not _cuobjdump(): pytest.skip("cuobjdump not installed")
    lib = "/home/sahil/ai/turbohaul-manager/vendor/turboquant-bin/libggml-cuda.so.0.15.1"
    if not os.path.exists(lib): pytest.skip("vendor binary not present")
    elf = subprocess.run(["cuobjdump","--list-elf",lib],capture_output=True,text=True)
    ptx = subprocess.run(["cuobjdump","--list-ptx",lib],capture_output=True,text=True)
    archs = sorted(set(re.findall(r"sm_(\d+)", elf.stdout+ptx.stdout)))
    archs = ["sm_"+a for a in archs]
    assert "sm_86" in archs
    non_86 = [a for a in archs if a!="sm_86"]
    assert not non_86, "vendor binary has non-sm_86: "+str(non_86)
