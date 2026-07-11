"""VRAM gate must count cache_type_v, not just cache_type_k."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turbohaul.safety import estimate_kv_cache_mib, _KV_QUANT_SCALE


def test_v_quant_counted_separately():
    # K=f16 (1.0), V=turbo3 (0.1875) -> effective scale (1.0+0.1875)/2 = 0.59375
    full = estimate_kv_cache_mib(275000, 20365809408, "f16")            # legacy: both f16
    mixed = estimate_kv_cache_mib(275000, 20365809408, "f16", "turbo3")  # mixed K/V quant
    assert mixed < full, "V=turbo3 must shrink the estimate vs full f16"
    # effective ~0.59x of full-f16
    assert abs(mixed - int(full * 0.59375)) <= full * 0.02


def test_v_none_is_legacy_k_only():
    # kv_cache_quant_v=None must reproduce the old single-quant behavior.
    assert estimate_kv_cache_mib(65536, 20365809408, "turbo3") == \
           estimate_kv_cache_mib(65536, 20365809408, "turbo3", None)


def test_symmetric_matches_single():
    # K==V should equal the old single-quant path.
    assert estimate_kv_cache_mib(65536, 16810713312, "turbo3", "turbo3") == \
           estimate_kv_cache_mib(65536, 16810713312, "turbo3")


if __name__ == "__main__":
    test_v_quant_counted_separately(); test_v_none_is_legacy_k_only(); test_symmetric_matches_single()
    print("ALL PASS")
