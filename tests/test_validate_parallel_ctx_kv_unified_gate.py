"""_validate_parallel_ctx: the per-slot split checks only apply WITHOUT a
unified KV pool.

Before this fix, the divisibility and PER_SLOT_CTX_FLOOR checks ran
unconditionally whenever parallel > 1 -- but step 2 (a few lines above them)
already RAISES unless kv_unified is true, so past that raise kv_unified is
guaranteed true. Per the vendored engine (llama-context.cpp:230-233),
kv_unified true means n_ctx_seq = n_ctx -- the FULL context per slot, not a
1/parallel split. The two downstream checks encode the split-across-slots
belief that is never actually true in the one path they can run in: they
rejected valid configs 100% of the time they fired.

Fix: gate both downstream checks on `not kv_unified`. The kv_unified
requirement itself (the raise) is untouched -- still unconditional, still
first. Because that requirement forces kv_unified true whenever parallel > 1,
the gated block is now correctly-scoped but presently unreachable through the
public Manifest API -- there is no way to construct a parallel>1 manifest
with kv_unified false without hitting the (intentionally unchanged) raise
first. That is expected, not a coverage gap: the whole point of the fix is
that the split checks never legitimately apply in the only path that reaches
them today.

Non-vacuity: tests 1-2 and the boundary/real-manifest tests below FAIL on
unmodified code (ManifestValidationError on the divisibility or floor
check). The control tests (3) pass on both old and new code -- they assert
the kv_unified requirement itself, which this fix does not touch.

Reactivation proof: the gated block calls _validate_parallel_split(parallel,
ctx_size) -- the divisibility+floor logic extracted into its own function
specifically so it stays directly callable without going through
_validate_parallel_ctx's kv_unified requirement at all. TestSplitLogicItself
below calls it directly to prove the logic is intact and still raises
correctly, rather than merely asserting that it would if it could ever run.
This does not touch or weaken the kv_unified requirement in the product
code -- it calls a different, narrower function that never checks kv_unified
in the first place.
"""
import pytest
from pydantic import ValidationError as PydanticValidationError

from turbohaul.manifest import Manifest, ModelManifest, ManifestValidationError, _validate_parallel_split

_SHA = "a" * 64


def _manifest(**flags):
    return ModelManifest(model_tag="t", gguf_blob_sha256=_SHA, llama_server_flags=flags)


class TestDivisibilityBypassedUnderKvUnified:
    def test_parallel2_kv_unified_ctx249999_validates(self):
        """Odd ctx_size that fails today's divisibility
        check (249999 is not divisible by 2) must validate once kv_unified is
        true, since no split happens."""
        m = _manifest(parallel=2, kv_unified=True, ctx_size=249999)
        assert m.llama_server_flags["ctx_size"] == 249999


class TestFloorBypassedUnderKvUnified:
    def test_parallel2_kv_unified_small_ctx_validates(self):
        """A ctx_size whose split (100 // 2 = 50) would be
        far below PER_SLOT_CTX_FLOOR (8192) must still validate under
        kv_unified, since each slot gets the full 100 tokens, not a split."""
        m = _manifest(parallel=2, kv_unified=True, ctx_size=100)
        assert m.llama_server_flags["ctx_size"] == 100


class TestControlKvUnifiedRequirementUnchanged:
    """The control: with kv_unified absent or explicitly false, the
    requirement at step 2 must still raise, unchanged, proving step 2 was not
    weakened by this fix."""

    def test_kv_unified_absent_still_raises(self):
        with pytest.raises(PydanticValidationError, match="requires kv_unified"):
            _manifest(parallel=2, ctx_size=249999)

    def test_kv_unified_false_still_raises(self):
        with pytest.raises(PydanticValidationError, match="requires kv_unified"):
            _manifest(parallel=2, kv_unified=False, ctx_size=249999)

    def test_parallel_1_is_a_no_op_regardless_of_kv_unified(self):
        """Unchanged back-compat path: parallel<=1 never reaches step 2 at
        all, kv_unified or not."""
        m = _manifest(parallel=1, ctx_size=100)
        assert m.llama_server_flags["ctx_size"] == 100


class TestBoundaryRegressionDetector:
    def test_small_model_par2_exact_floor_boundary_validates(self):
        """Named explicitly in the requirements: ctx_size=16384, parallel=2 ->
        per_slot == PER_SLOT_CTX_FLOOR (8192) EXACTLY. The floor check is
        `per_slot < PER_SLOT_CTX_FLOOR` -- 8192 < 8192 is False, so this must
        validate both before this fix (it already did, on the numbers alone)
        and after (via the new gate). The cheapest possible detector for an
        accidental `<` -> `<=` flip anywhere in this change."""
        m = _manifest(parallel=2, kv_unified=True, ctx_size=16384)
        assert m.llama_server_flags["ctx_size"] == 16384


class TestRealManifestOutcomeUnchanged:
    """Every parallel:2 manifest found across the full
    set of real manifests.
    All of them set kv_unified:true.
    Each already passed today's validator on the numbers (all ctx_size values divide
    evenly by 2 and clear the 8192 floor); each must still pass post-fix, now
    via the new gate rather than by coincidentally satisfying it. Named after
    the shape of a real manifest so a future outcome flip is traceable to which
    class of deployed config it would affect. Names are generic placeholders;
    the ctx_size/parallel pairs are the load-bearing part."""

    @pytest.mark.parametrize(
        "name,ctx_size",
        [
            ("model-a-d4", 250000),
            ("model-b-d4", 250000),
            ("model-c-35b-moe", 500000),
            ("model-d-35b-moe-gpu1", 500000),
            ("model-e-35b", 500000),
            ("model-f-par2", 16384),
        ],
    )
    def test_real_parallel2_manifest_still_validates(self, name, ctx_size):
        m = _manifest(parallel=2, kv_unified=True, ctx_size=ctx_size)
        assert m.llama_server_flags["ctx_size"] == ctx_size, name


class TestSplitLogicItselfIsStillCorrectWhenCalledDirectly:
    """Proves the "reactivates automatically" claim in _validate_parallel_ctx's
    comment is TRUE, not just documented. Calls _validate_parallel_split
    directly with a kv_unified:false scenario -- bypassing
    _validate_parallel_ctx and its kv_unified requirement entirely, not
    weakening it. If the kv_unified requirement is ever relaxed so this path
    becomes reachable again, THIS is the logic that would run, and these two
    tests are the ones that would catch a regression in it today."""

    def test_non_divisible_ctx_size_raises_the_divisibility_error(self):
        with pytest.raises(ManifestValidationError, match="not divisible by"):
            _validate_parallel_split(parallel=2, ctx_size=249999)

    def test_below_floor_after_division_raises_the_floor_error(self):
        # 16000 // 2 = 8000, below PER_SLOT_CTX_FLOOR (8192), and 16000 is
        # evenly divisible by 2 so this exercises the SECOND raise, not the
        # first.
        with pytest.raises(ManifestValidationError, match="PER_SLOT_CTX_FLOOR"):
            _validate_parallel_split(parallel=2, ctx_size=16000)

    def test_valid_split_does_not_raise(self):
        """Control for the control: the same logic must NOT raise for a
        genuinely valid split, so the two tests above are catching real
        violations, not just any input."""
        _validate_parallel_split(parallel=2, ctx_size=16384)
