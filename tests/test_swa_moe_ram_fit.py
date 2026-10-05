"""Two-resource fit math -- SWA-aware KV sizing, exact MoE
expert-byte placement via tensor-info offsets, and the additive MoE
RAM-fit gate.

A separate, non-pytest script cross-checks this against real per-model
header facts for real models (too large a fixture for a
hermetic unit test); its results are not part of this suite.
"""
import struct

import pytest

from turbohaul._gguf_meta import KVDims
from turbohaul._gguf_tensor_offsets import read_expert_layout, split_expert_bytes
from turbohaul.safety import check_moe_ram_fit, estimate_kv_cache_mib


# --- SWA-aware KV sizing -------------------------------------------------

class TestSwaAwareKv:
    def test_laguna_shaped_swa_much_smaller_than_all_layers_grow(self):
        """The documented worked example: 12 global + 36 sliding(512) at
        ctx=250000 should be roughly 4x smaller than treating every layer
        as growing (block_count=48, interval=1 -- the pre-SWA-aware model
        of "every layer grows")."""
        laguna = KVDims(arch="laguna", block_count=48, full_attention_interval=4,
                         n_head_kv=8, key_length=128, value_length=128,
                         sliding_window=512)
        assert laguna.n_attn_layers == 12
        assert laguna.n_swa_layers == 36
        kv_swa = estimate_kv_cache_mib(250_000, 73_395_171_776, "turbo3", "turbo3",
                                        attn_dims=laguna)

        all_grow = KVDims(arch="laguna", block_count=48, full_attention_interval=1,
                           n_head_kv=8, key_length=128, value_length=128,
                           sliding_window=None)
        kv_all_grow = estimate_kv_cache_mib(250_000, 73_395_171_776, "turbo3",
                                             "turbo3", attn_dims=all_grow)
        ratio = kv_all_grow / kv_swa
        assert 3.5 < ratio < 4.5, f"expected ~4x, got {ratio:.2f}x ({kv_all_grow} vs {kv_swa})"

    def test_non_sliding_model_unaffected_by_swa_code_path(self):
        """A model with sliding_window=None must produce the EXACT same
        estimate the pre-SWA-aware formula gave -- this is the fixture from
        TestCalibration.test_dims_path_first_principles (already asserts
        2441); re-asserted here with intent, next to the SWA tests, so a
        future editor sees both invariants together."""
        non_sliding_dims = KVDims(arch="qwen35", block_count=64, full_attention_interval=4,
                         n_head_kv=4, key_length=256, value_length=256)
        assert non_sliding_dims.n_swa_layers == 0
        kv = estimate_kv_cache_mib(250_000, 7233 * 1024 * 1024, "turbo3", "turbo2",
                                    attn_dims=non_sliding_dims)
        assert kv == 2441

    def test_default_kvdims_never_crashes_the_swa_guard(self):
        """THE crash: on a default KVDims (sliding_window=None,
        n_swa_layers=0 via property), 0 * min(ctx, None) does NOT
        short-circuit in Python -- min(ctx, None) raises TypeError before
        the multiply. Must not raise."""
        default_dims = KVDims("qwen35", 64, 0, 4, 256, 256)
        assert default_dims.sliding_window is None
        assert default_dims.n_swa_layers == 0
        # must not raise
        kv = estimate_kv_cache_mib(32768, 668_788_096, "f16", attn_dims=default_dims)
        assert kv > 0

    def test_swa_term_scales_k_and_v_by_their_own_quant(self):
        """K and V halves of the SWA term must scale independently, same as
        the existing growing-layer term -- not both by a single shared
        scale."""
        dims = KVDims(arch="laguna", block_count=8, full_attention_interval=4,
                       n_head_kv=8, key_length=128, value_length=128,
                       sliding_window=64)
        kv_symmetric = estimate_kv_cache_mib(1000, 1024 * 1024 * 1024, "f16", "f16",
                                              attn_dims=dims)
        kv_asymmetric = estimate_kv_cache_mib(1000, 1024 * 1024 * 1024, "turbo3",
                                               "f16", attn_dims=dims)
        assert kv_asymmetric < kv_symmetric


# --- tensor-info offset-delta reader -------------------------------------

def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k: str, v: str) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)


def _kv_u32(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _tensor_info(name: str, dims: "list[int]", ttype: int, offset: int) -> bytes:
    out = _gguf_string(name)
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", ttype)  # never decoded by the reader -- any value ok
    out += struct.pack("<Q", offset)
    return out


def _build_gguf_with_tensors(kvs: "list[bytes]", tensors: "list[bytes]",
                              alignment: int = 32) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    header += b"".join(kvs) + b"".join(tensors)
    return header


class TestTensorOffsetReader:
    def _synthetic_moe_gguf(self, tmp_path, custom_type_id=47):
        """3 tensors: 1 non-expert, 2 expert (blk.0 and blk.2 -- blk.1 has
        none, mirroring laguna's dense-first-layer shape), using a CUSTOM
        quant type-id the reader must never decode."""
        sizes = [100, 4000, 200]  # arbitrary but must be positive & consistent
        offsets = [0, 100, 4100]
        tensors = [
            _tensor_info("token_embd.weight", [10, 10], 0, offsets[0]),
            _tensor_info("blk.0.ffn_down_exps.weight", [10, 10], custom_type_id, offsets[1]),
            _tensor_info("blk.2.ffn_up_exps.weight", [10, 10], custom_type_id, offsets[2]),
        ]
        kvs = [_kv_str("general.architecture", "testmoe"),
               _kv_u32("general.alignment", 32)]
        header = _build_gguf_with_tensors(kvs, tensors)
        data_start = len(header) + ((-len(header)) % 32)
        total_data = sum(sizes)
        body = header + b"\x00" * (data_start - len(header)) + b"\xAB" * total_data
        p = tmp_path / "moe.gguf"
        p.write_bytes(body)
        return p

    def test_closure_holds_and_expert_bytes_split_by_block(self, tmp_path):
        p = self._synthetic_moe_gguf(tmp_path)
        layout = read_expert_layout(p)
        assert layout is not None
        assert layout.non_expert_bytes == 100
        assert layout.expert_bytes_by_block == {0: 4000, 2: 200}
        assert layout.total_expert_bytes == 4200
        assert layout.fsize == p.stat().st_size

    def test_never_decodes_type_field(self, tmp_path):
        """A custom engine quant type-id (e.g. type-id 47, or anything else)
        must not affect the result at all -- the reader only ever consults
        offset."""
        p1 = self._synthetic_moe_gguf(tmp_path, custom_type_id=47)
        p2_path = tmp_path / "moe2.gguf"
        p2_path.write_bytes(p1.read_bytes())  # identical bytes; type-id irrelevant either way
        layout1 = read_expert_layout(p1)
        layout2 = read_expert_layout(p2_path)
        assert layout1 == layout2

    def test_split_expert_bytes_uses_block_index_not_position(self):
        """--n-cpu-moe N offloads the first N BLOCK INDICES, not the first N
        entries of expert_bytes_by_block. A dense-first-layer model (block 0
        has no experts, mirrors laguna) must not shift the split."""
        from turbohaul._gguf_tensor_offsets import ExpertLayout
        layout = ExpertLayout(
            expert_bytes_by_block={1: 100, 2: 200, 3: 300},  # block 0 dense
            non_expert_bytes=50, fsize=1000, data_start=100,
        )
        offloaded, resident = split_expert_bytes(layout, n_cpu_moe=3)
        # blocks 1, 2 < 3 offload; block 3 stays resident
        assert offloaded == 300  # 100 + 200
        assert resident == 300

    def test_malformed_file_returns_none_not_raise(self, tmp_path):
        p = tmp_path / "garbage.gguf"
        p.write_bytes(b"NOT A GGUF FILE")
        assert read_expert_layout(p) is None

    def test_nonexistent_file_returns_none_not_raise(self, tmp_path):
        assert read_expert_layout(tmp_path / "does_not_exist.gguf") is None

    def test_expert_tensor_without_block_index_refuses_whole_derivation(self, tmp_path):
        """An '_exps' tensor with no parseable blk.N. prefix must refuse the
        WHOLE derivation (return None), not silently drop/misattribute it."""
        offsets = [0, 100]
        tensors = [
            _tensor_info("token_embd.weight", [1], 0, offsets[0]),
            _tensor_info("weird_exps.weight", [1], 0, offsets[1]),  # no blk.N.
        ]
        kvs = [_kv_str("general.architecture", "testmoe"),
               _kv_u32("general.alignment", 32)]
        header = _build_gguf_with_tensors(kvs, tensors)
        data_start = len(header) + ((-len(header)) % 32)
        body = header + b"\x00" * (data_start - len(header)) + b"\xAB" * 200
        p = tmp_path / "weird.gguf"
        p.write_bytes(body)
        assert read_expert_layout(p) is None


# --- moe_ram_fit gate ------------------------------------------------------

class TestMoeRamFitGate:
    def test_defaults_are_a_pure_noop(self, monkeypatch):
        r = check_moe_ram_fit()
        assert r.ok is True
        assert r.name == "moe_ram_fit"

    def test_refuses_when_need_exceeds_available(self, monkeypatch):
        import turbohaul.safety as safety_mod
        monkeypatch.setattr(safety_mod, "_read_meminfo_kib",
                             lambda: {"MemAvailable": 1024 * 1024})  # 1 GiB
        r = check_moe_ram_fit(expert_offloaded_mib=2000, kv_ram_mib=0)
        assert r.ok is False
        assert "offloaded experts=2000" in r.detail

    def test_passes_when_need_fits(self, monkeypatch):
        import turbohaul.safety as safety_mod
        monkeypatch.setattr(safety_mod, "_read_meminfo_kib",
                             lambda: {"MemAvailable": 64 * 1024 * 1024})  # 64 GiB
        r = check_moe_ram_fit(expert_offloaded_mib=2000, kv_ram_mib=500)
        assert r.ok is True

    def test_no_probe_passes_open(self, monkeypatch):
        import turbohaul.safety as safety_mod
        monkeypatch.setattr(safety_mod, "_read_meminfo_kib", lambda: {})
        r = check_moe_ram_fit(expert_offloaded_mib=999999, kv_ram_mib=0)
        assert r.ok is True
        assert "no-probe" in r.detail

    def test_combines_expert_and_kv_ram(self, monkeypatch):
        import turbohaul.safety as safety_mod
        monkeypatch.setattr(safety_mod, "_read_meminfo_kib",
                             lambda: {"MemAvailable": 3000 * 1024})  # 3000 MiB
        # neither alone exceeds 3000, but together they do
        r = check_moe_ram_fit(expert_offloaded_mib=2000, kv_ram_mib=1500)
        assert r.ok is False
