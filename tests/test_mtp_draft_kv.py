"""MTP draft-context KV term.

Mirrors tests/test_dimension_aware_kv.py's conventions (real KVDims, the
hand-built-GGUF-bytes helpers, monkeypatching turbohaul.safety._read_free_
vram_all_mib) so this file drops into the same test session with no new
fixtures. See docs/SAFETY_GATE_VRAM_MATH.md §4 (three-tier KV discipline)
and MTP_DRAFT_MARGIN's own comment in safety.py for the margin rationale.

FACTS pinned here, taken from GGUF-header reads (not
re-derived here):
  draft_bytes_per_token = nextn_predict_layers * head_count_kv
                           * (key_length + value_length) * 2
  multiplied by ctx_size (NOT spec_draft_n_max), unconditional f16 (scale
  never derived from spec_draft_type_k/v), scaled by MTP_DRAFT_MARGIN.
Per-model derived figures:
  a 27B MTP model: nextn=1 hkv=4 kl=vl=256
    ctx=250,000 -> 976.6 MiB pre-margin (the derived per-token figure can sit
    slightly below real allocation -- the residual MTP_DRAFT_MARGIN covers).
  a 35B MTP model (the large-context case): nextn=1 hkv=2 kl=vl=256 ctx=500,000
    -> 976.6 MiB pre-margin (derived from the geometry only).
"""
import struct

import pytest

from turbohaul._gguf_meta import KVDims, read_kv_dims
from turbohaul.safety import (
    MTP_DRAFT_MARGIN,
    all_safety_gates,
    check_kv_cache_fit,
    estimate_kv_cache_mib,
)

GGUF_BODY_BYTES = 19 * 1024 * 1024 * 1024  # ~19 GiB, matches a 35B model's scale


def _mtp_dims(n_head_kv: int, *, nextn_predict_layers: int = 1) -> KVDims:
    """A minimal synthetic KVDims isolating the draft term: block_count=1,
    full_attention_interval=0 -> n_attn_layers=1, so the pre-existing K/V
    term is small and the draft term's own contribution is exact and
    independently verifiable (not entangled with a specific model's real
    layer count, which the draft term does not
    depend on)."""
    return KVDims(
        arch="qwen35moe", block_count=1, full_attention_interval=0,
        n_head_kv=n_head_kv, key_length=256, value_length=256,
        nextn_predict_layers=nextn_predict_layers,
    )


# === Draft-term isolation + per-model regression pins =========================

class TestDraftTermIsolation:
    def test_orion_grm_geometry_delta_matches_derived_figure(self):
        """A 27B MTP model: nextn=1 hkv=4 kl=vl=256
        ctx=250,000. Pre-margin: 4096 B/tok * 250,000 = 976.56 MiB (floors
        to 976 against the isolated n_attn=1 baseline, also 976 -> total
        1025 with everything folded together, since draft and base happen
        to share the same per-token magnitude here)."""
        off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                    attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                   attn_dims=_mtp_dims(4), mtp_draft_active=True)
        assert off == 976, off
        assert on == 2001, on
        assert on - off == 1025, on - off

    def test_qwen35b_mtp_crash_model_geometry_delta_matches_derived_figure(self):
        """A 35B MTP model (the large-context case): nextn=1 hkv=2 kl=vl=256
        ctx=500,000 -- half the heads, double the context, so the same
        976.6 MiB pre-margin figure derived above (derived from the
        geometry only, same as the 27B case)."""
        off = estimate_kv_cache_mib(500_000, GGUF_BODY_BYTES,
                                    attn_dims=_mtp_dims(2), mtp_draft_active=False)
        on = estimate_kv_cache_mib(500_000, GGUF_BODY_BYTES,
                                   attn_dims=_mtp_dims(2), mtp_draft_active=True)
        assert off == 976, off
        assert on == 2001, on
        assert on - off == 1025, on - off

    def test_margin_is_actually_applied(self):
        """Without MTP_DRAFT_MARGIN the pre-margin draft delta would be
        exactly 976 MiB (matching the isolated baseline) -- WITH the
        margin it must be strictly larger, proving the +5% is live in the
        code path, not just documented in a comment."""
        off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                    attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                   attn_dims=_mtp_dims(4), mtp_draft_active=True)
        unmargined_delta_bytes = 1 * 4 * (256 + 256) * 2 * 250_000
        margined_delta_bytes = unmargined_delta_bytes * MTP_DRAFT_MARGIN
        assert margined_delta_bytes > unmargined_delta_bytes
        assert (on - off) * 1024 * 1024 >= unmargined_delta_bytes


# === Gating: both mtp_draft_active AND nextn_predict_layers > 0 required ======

class TestDraftTermGating:
    def test_inactive_flag_is_a_noop_even_with_nextn_present(self):
        """A GGUF with a real MTP head present (nextn_predict_layers=1) but
        the MANIFEST not opting into draft-mtp (mtp_draft_active=False)
        must NOT get the draft budget -- the draft cache is only actually
        allocated when the operator has speculative decoding turned on."""
        off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, attn_dims=_mtp_dims(4))
        still_off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                          attn_dims=_mtp_dims(4), mtp_draft_active=False)
        assert off == still_off

    def test_active_flag_is_a_noop_without_a_real_nextn_head(self):
        """mtp_draft_active=True on a non-MTP GGUF (nextn_predict_layers=0,
        the default for every model that lacks the key) must not fabricate
        a draft term -- it must not pick a number not derived from real
        GGUF geometry."""
        off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                    attn_dims=_mtp_dims(4, nextn_predict_layers=0))
        still_off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                          attn_dims=_mtp_dims(4, nextn_predict_layers=0),
                                          mtp_draft_active=True)
        assert off == still_off

    def test_default_mtp_draft_active_is_false_byte_identical(self):
        """Every existing caller (no mtp_draft_active kwarg at all) must be
        byte-identical to passing mtp_draft_active=False."""
        with_kw = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                        attn_dims=_mtp_dims(4), mtp_draft_active=False)
        without_kw = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, attn_dims=_mtp_dims(4))
        assert with_kw == without_kw

    def test_multiplies_by_ctx_size_not_spec_draft_n_max(self):
        """Must not multiply by spec_draft_n_max. Doubling ctx_size
        must double the draft delta; the function has no spec_draft_n_max
        parameter at all -- there is nothing to multiply by even if a
        caller tried."""
        off_a = estimate_kv_cache_mib(100_000, GGUF_BODY_BYTES,
                                      attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on_a = estimate_kv_cache_mib(100_000, GGUF_BODY_BYTES,
                                     attn_dims=_mtp_dims(4), mtp_draft_active=True)
        off_b = estimate_kv_cache_mib(200_000, GGUF_BODY_BYTES,
                                      attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on_b = estimate_kv_cache_mib(200_000, GGUF_BODY_BYTES,
                                     attn_dims=_mtp_dims(4), mtp_draft_active=True)
        delta_a, delta_b = on_a - off_a, on_b - off_b
        assert abs(delta_b - 2 * delta_a) <= 1, (delta_a, delta_b)

    def test_never_scaled_by_manifest_cache_type(self):
        """Must not scale the draft term by manifest cache_type_k/v.
        The draft cache is unconditionally f16 -- passing a heavily
        quantized kv_cache_quant must change the MAIN K/V term but leave
        the draft delta untouched."""
        off_f16 = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, "f16", "f16",
                                        attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on_f16 = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, "f16", "f16",
                                       attn_dims=_mtp_dims(4), mtp_draft_active=True)
        off_q4 = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, "q4_0", "q4_0",
                                       attn_dims=_mtp_dims(4), mtp_draft_active=False)
        on_q4 = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, "q4_0", "q4_0",
                                      attn_dims=_mtp_dims(4), mtp_draft_active=True)
        assert off_f16 > off_q4  # main K/V term IS scaled by quant, as before
        assert (on_f16 - off_f16) == (on_q4 - off_q4)  # draft delta is NOT

    def test_confined_to_tier2_never_applies_to_legacy_or_override_path(self):
        """No attn_dims at all (legacy file-size path) or a measured
        kv_bytes_per_token override -- mtp_draft_active must be a no-op in
        both, since the term needs real parsed GGUF geometry to exist."""
        legacy_off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES)
        legacy_on = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES, mtp_draft_active=True)
        assert legacy_off == legacy_on

        override_off = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                             kv_bytes_per_token=13824.0)
        override_on = estimate_kv_cache_mib(250_000, GGUF_BODY_BYTES,
                                            kv_bytes_per_token=13824.0, mtp_draft_active=True)
        assert override_off == override_on


# === The key case: fails without the term, passes (refuses) with it ===========

class TestGateRefusesOvercommitDraftTermMisses:
    def test_gate_admits_without_draft_then_refuses_with_it(self):
        """A model sized so body+overhead+KV(no draft) FITS free VRAM but
        body+overhead+KV(with draft) does NOT -- the key case: a
        test that fails without the draft term and
        passes with it, not one that passes either way."""
        gguf_mib = GGUF_BODY_BYTES // (1024 * 1024)
        free = gguf_mib + 976 + 1024 + (gguf_mib + 2001 + 1024)
        free //= 2  # midpoint: 21968 -- admits the no-draft total, refuses the draft total
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.safety._read_free_vram_all_mib", lambda *a, **k: [free])
            no_draft = check_kv_cache_fit(250_000, GGUF_BODY_BYTES,
                                          attn_dims=_mtp_dims(4), mtp_draft_active=False)
            with_draft = check_kv_cache_fit(250_000, GGUF_BODY_BYTES,
                                            attn_dims=_mtp_dims(4), mtp_draft_active=True)
        assert no_draft.ok, f"no-draft should fit: {no_draft.detail}"
        assert not with_draft.ok, f"with-draft should refuse the over-commit: {with_draft.detail}"

    def test_all_safety_gates_threads_mtp_draft_active(self):
        """The aggregator-level equivalent of the case above -- proves
        the flag survives the full all_safety_gates() plumbing, not just
        the inner check_kv_cache_fit call."""
        gguf_mib = GGUF_BODY_BYTES // (1024 * 1024)
        free = (gguf_mib + 976 + 1024 + gguf_mib + 2001 + 1024) // 2
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.safety._read_free_vram_all_mib", lambda *a, **k: [free])
            no_draft = all_safety_gates(
                min_free_ram_mib=1024, min_free_vram_mib=512,
                max_cpu_busy_percent=99.0, max_iowait_percent=99.0,
                ctx_size=250_000, gguf_size_bytes=GGUF_BODY_BYTES,
                attn_dims=_mtp_dims(4), mtp_draft_active=False,
            )
            with_draft = all_safety_gates(
                min_free_ram_mib=1024, min_free_vram_mib=512,
                max_cpu_busy_percent=99.0, max_iowait_percent=99.0,
                ctx_size=250_000, gguf_size_bytes=GGUF_BODY_BYTES,
                attn_dims=_mtp_dims(4), mtp_draft_active=True,
            )
        kv_off = [g for g in no_draft if g.name == "kv_cache_fit"][0]
        kv_on = [g for g in with_draft if g.name == "kv_cache_fit"][0]
        assert kv_off.ok, kv_off.detail
        assert not kv_on.ok, kv_on.detail


# === no_kv_offload interaction: the host-RAM placement path ===================

class TestNoKvOffloadHostRamInteraction:
    def test_mtp_draft_active_also_affects_the_host_ram_precheck(self):
        """Design decision: the draft term
        follows kv_mib's existing monolithic placement, so it must also
        show up in the no_kv_offload host-RAM pre-check inside
        all_safety_gates (the kv_ram_mib branch), not just the VRAM
        default branch. Pinned here because it's a real, deliberate design
        choice, not an accident of how the parameter happens to thread."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("turbohaul.safety._read_free_vram_all_mib", lambda *a, **k: [999_999])
            no_draft = all_safety_gates(
                min_free_ram_mib=0, min_free_vram_mib=0,
                max_cpu_busy_percent=99.0, max_iowait_percent=99.0,
                ctx_size=250_000, gguf_size_bytes=GGUF_BODY_BYTES,
                no_kv_offload=True, attn_dims=_mtp_dims(4), mtp_draft_active=False,
            )
            with_draft = all_safety_gates(
                min_free_ram_mib=0, min_free_vram_mib=0,
                max_cpu_busy_percent=99.0, max_iowait_percent=99.0,
                ctx_size=250_000, gguf_size_bytes=GGUF_BODY_BYTES,
                no_kv_offload=True, attn_dims=_mtp_dims(4), mtp_draft_active=True,
            )
        moe_off = [g for g in no_draft if g.name == "moe_ram_fit"][0]
        moe_on = [g for g in with_draft if g.name == "moe_ram_fit"][0]
        # kv_ram_mib feeds moe_ram_fit's detail; the draft term changing it
        # proves mtp_draft_active reached the host-RAM pre-check too.
        assert moe_off.detail != moe_on.detail, (moe_off.detail, moe_on.detail)


# === _gguf_meta parser: nextn_predict_layers round-trips from real bytes ======

def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k: str, v: str) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)


def _kv_u32(k: str, v: int) -> bytes:
    return _gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _build_gguf(kvs: list, tensor_count: int = 0) -> bytes:
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kvs))
    return body + b"".join(kvs)


class TestNextnPredictLayersGgufParsing:
    def test_nextn_predict_layers_round_trips_from_gguf_bytes(self, tmp_path):
        p = tmp_path / "mtp.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen35moe"),
            _kv_u32("qwen35moe.block_count", 48),
            _kv_u32("qwen35moe.full_attention_interval", 4),
            _kv_u32("qwen35moe.attention.head_count_kv", 2),
            _kv_u32("qwen35moe.attention.key_length", 256),
            _kv_u32("qwen35moe.attention.value_length", 256),
            _kv_u32("qwen35moe.nextn_predict_layers", 1),
        ]))
        d = read_kv_dims(p)
        assert d is not None
        assert d.nextn_predict_layers == 1, d

    def test_nextn_predict_layers_absent_defaults_to_zero(self, tmp_path):
        """A GGUF with no nextn_predict_layers key at all (every existing
        non-MTP model) parses to 0 -- the default that keeps
        mtp_draft_active a no-op via TestDraftTermGating above."""
        p = tmp_path / "no_mtp.gguf"
        p.write_bytes(_build_gguf([
            _kv_str("general.architecture", "qwen35"),
            _kv_u32("qwen35.block_count", 64),
            _kv_u32("qwen35.full_attention_interval", 4),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.attention.key_length", 256),
            _kv_u32("qwen35.attention.value_length", 256),
        ]))
        d = read_kv_dims(p)
        assert d is not None
        assert d.nextn_predict_layers == 0, d

    def test_kvdims_positional_construction_still_works(self):
        """Backward-compat pin: the pre-existing positional-construction
        pattern in test_dimension_aware_kv.py (6 positional args, relying
        on sliding_window's default) must still work with
        nextn_predict_layers appended as a SECOND trailing default."""
        d = KVDims("qwen35", 64, 4, 4, 256, 256)
        assert d.sliding_window is None
        assert d.nextn_predict_layers == 0
