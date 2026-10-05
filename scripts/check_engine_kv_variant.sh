#!/usr/bin/env bash
# check_engine_kv_variant.sh -- build-time guard against a silent quality defect: a
# libllama.so built from a source revision that only WARNS about turbo-K
# quality risk on high-GQA-ratio models instead of actually upgrading K to
# q8_0 -- a different, worse-quality engine that ships silently, with no
# crash, no error, and a green build.
#
# THE DEFECT: llama-kv-cache.cpp's constructor auto-upgrades K from a turbo
# type to q8_0 when the model's GQA ratio (n_head / n_head_kv) is >= 6 and
# K == V, because turbo-K's quantization error is amplified by the GQA
# broadcast factor on those models, which can degrade perplexity substantially on
# models with a 7:1 ratio. The vendored source (engine/) carries ONLY this
# real-upgrade code path: `LLAMA_LOG_WARN(... "upgrading K from %s to q8_0
# to prevent quality degradation" ...); type_k = GGML_TYPE_Q8_0;`. A stale
# revision of this same file instead carried a WARN-ONLY variant that logged
# "auto-asymmetric risk: ... — configured K type is honored as-is" and left
# type_k untouched -- same log-site shape, opposite behavior. Both variants
# compile, link, boot, and serve identically; the difference is invisible
# until someone notices a quality regression on a high-GQA model, days
# later, from perplexity, not from a stack trace.
#
# WHY A NEW SCRIPT, NOT AN EXTENSION of check_engine_symbols.sh (the
# existing symbols guard already wired into this same Dockerfile): that script
# targets a DIFFERENT library (libllama-server-impl*, via
# impl_candidates=("${BUILD_DIR}"/libllama-server-impl*)) than the one that
# actually diverges here (libllama.so.0.0.0). If no libllama-server-impl
# artifact is present in a build dir, that script prints "nothing to check,
# OK" and exits 0 -- extending it to also assert the K-variant would have
# produced a guard that is vacuously green on exactly the build shape this
# defect lives in. The two checks also answer genuinely different questions
# -- does this artifact RESOLVE (symbol completeness) vs WHICH VARIANT is
# this (a behavioral fact about the code, not its linkage) -- so overloading
# one script with both would let a future reader believe a green run proved
# both when a missing artifact already made the first vacuously true.
#
# WHAT THIS CHECKS: the target library's string table for the literal
# substring "upgrading K from" -- the log-format prefix that is unique to
# the real-upgrade code path (confirmed absent from the warn-only variant,
# which uses a structurally different message: "auto-asymmetric risk").
# A strings-table check is intentionally the same class of static check
# check_engine_symbols.sh already uses (that script inspects nm's dynamic
# symbol table; this one inspects the string table) -- both are "does this
# specific artifact carry evidence of the code path we need", neither
# executes the binary or any model.
#
# WHY THIS COSTS NOTHING NEW TO RUN: `strings` ships in the SAME `binutils`
# package check_engine_symbols.sh already installs transiently (installed,
# used, purged, apt lists cleaned, all in one RUN layer) in
# the image build. This script is meant to run in that exact layer,
# right alongside the existing check -- extending the GUARD INFRASTRUCTURE,
# even though it is a separate script asserting a separate fact.
#
# VALIDATION: the check is exercised in both directions. A library built from a
# source revision that only warns fails it, and a library compiled from the
# vendored engine source passes it. A control confirms the check can still
# FAIL, not merely pass, so a green result carries information. The check is a
# plain string-table test over the built library: no model is loaded and no
# binary is executed, so the same library gives the same verdict on any
# build host. It is a static check of one artifact, in the same class as
# the symbols guard.
#
# WARN/ENFORCE TOGGLE: this check ENFORCES by default (ENFORCE_DEFAULT=1 below). A
# warn-only mode exists for builds that start from a prebuilt library which
# predates the real upgrade path: there a hard failure could be worse than the
# degraded-but-running engine it replaces. Set CHECK_ENGINE_KV_VARIANT_ENFORCE=0
# to downgrade a failure to a loud warning. The only in-repo caller,
# Dockerfile.cuda-multi, COMPILES FROM SOURCE every time, and a from-source
# build carries the real upgrade path, so that caller has no need for the warn
# asymmetry. Enforcing is therefore the default: a missing code path fails the
# build instead of shipping a degraded engine silently. Changing the default
# changes build-failure behaviour, so it is a deliberate decision, not a side
# effect of a cleanup -- see the note on ENFORCE_DEFAULT below.
#
# ENFORCE_DEFAULT is the SINGLE point of control for callers that do not set
# CHECK_ENGINE_KV_VARIANT_ENFORCE themselves. To change the default mode, flip
# this one value (1 = enforce, 0 = warn) -- do NOT "fix" it with a per-callsite
# override instead; that leaves a second place to remember, the exact failure
# mode this default exists to avoid.
# NOTE: this script's own rule is that a caller which builds a known-good .so
# from source every time may enforce, and the sole caller,
# Dockerfile.cuda-multi, is exactly that. Enforcement is therefore safe there:
# a from-source build PASSES this check, and with ENFORCE=1 a library lacking
# the upgrade path exits 1 (refused) while a from-source library exits 0
# (allowed), so the mechanism discriminates rather than merely passing.
# (To warn instead of refuse, set CHECK_ENGINE_KV_VARIANT_ENFORCE=0.)
ENFORCE_DEFAULT=1
#
# Usage:   check_engine_kv_variant.sh [--require-present] [path-to-libllama.so]
#          (default path: /opt/turboquant/build/bin/libllama.so.0.0.0)
# Exit 0:  the library carries the real auto-asymmetric K-upgrade code path;
#          OR the target file is absent/empty AND --require-present was NOT
#          given (nothing to check is not a failure -- not every build/context
#          produces this file at this path; mirrors check_engine_symbols.sh's
#          own convention for that case); OR the library is missing the code
#          path but the effective mode is WARN (see ENFORCE_DEFAULT /
#          CHECK_ENGINE_KV_VARIANT_ENFORCE above) -- in this last case a loud
#          warning is printed to stderr, this is a DEFERRED failure, not a
#          pass, and the fix is named in it.
# Exit 1:  the library exists, does NOT carry the upgrade code path, AND the
#          effective mode is ENFORCE; OR the target file is absent/empty AND
#          --require-present WAS given (inside a build the engine must
#          exist, so a missing one there is a build failure, not "OK").

set -euo pipefail

require_present=0
LIB=""
for arg in "$@"; do
    case "$arg" in
        --require-present)
            require_present=1
            ;;
        *)
            LIB="$arg"
            ;;
    esac
done
LIB="${LIB:-/opt/turboquant/build/bin/libllama.so.0.0.0}"

if [ -n "${CHECK_ENGINE_KV_VARIANT_ENFORCE:-}" ]; then
    enforce="$CHECK_ENGINE_KV_VARIANT_ENFORCE"
else
    enforce="$ENFORCE_DEFAULT"
fi

if [ ! -s "$LIB" ]; then
    if [ "$require_present" = "1" ]; then
        echo "check_engine_kv_variant: ${LIB} absent/empty -- --require-present was set, this build must produce it." >&2
        exit 1
    fi
    echo "check_engine_kv_variant: ${LIB} absent/empty -- nothing to check, OK"
    exit 0
fi

if ! command -v strings >/dev/null 2>&1; then
    echo "check_engine_kv_variant: required tool 'strings' not found on PATH" >&2
    exit 1
fi

# NOTE: `grep -c` here, not `grep -q` -- `-q` exits the instant it finds a
# match, closing its end of the pipe while `strings` may still be mid-write;
# under `set -o pipefail` that SIGPIPE (128+13=141) on `strings` becomes the
# PIPELINE's reported exit status even though grep found what it was looking
# for, so `if strings ... | grep -q ...` can report FALSE on a genuine match.
# `-c` always reads its input to completion (it has to, to count correctly),
# so `strings` exits 0 normally and no SIGPIPE occurs. Exercise this script on a
# library that carries the upgrade path as well as one that lacks it: a check
# that only demonstrates failing correctly, never passing correctly, is not
# proven.
match_count="$(strings "$LIB" 2>/dev/null | grep -c "upgrading K from" || true)"
if [ "${match_count:-0}" -gt 0 ]; then
    echo "check_engine_kv_variant: ${LIB} carries the auto-asymmetric K-upgrade code path, OK"
    exit 0
fi

{
    echo "check_engine_kv_variant: ${LIB} is MISSING the auto-asymmetric K-upgrade code path."
    echo "  This build would silently keep turbo K quantization on high-GQA-ratio models (n_head /"
    echo "  n_head_kv >= 6) instead of auto-upgrading K to q8_0 -- a known cause of a large perplexity"
    echo "  increase on 7:1-ratio models. No crash, no error at build or"
    echo "  boot time; the only symptom is a silent quality regression, discovered later from"
    echo "  perplexity, not from a stack trace."
} >&2

if [ "$enforce" = "1" ]; then
    echo "check_engine_kv_variant: ENFORCE mode -- refusing to ship this engine." >&2
    exit 1
fi

# WARN mode is an EXPLICIT OPT-OUT (CHECK_ENGINE_KV_VARIANT_ENFORCE=0), not the
# default. Every in-repo caller compiles the engine from source (see
# Dockerfile.cuda-multi), so enforcement is safe there, and ENFORCE_DEFAULT is 1:
# a build that wants to continue anyway has to ask for it.
# A warn that is silent is just a disabled guard, so this stays exactly as loud
# as the FAIL branch and states plainly that it is not a pass.
{
    echo "check_engine_kv_variant: WARNING -- WARN mode was explicitly requested"
    echo "  (CHECK_ENGINE_KV_VARIANT_ENFORCE=0), continuing build with a KNOWN-DEGRADED"
    echo "  engine. This is a DEFERRED failure, not a pass. The default is ENFORCE"
    echo "  (see ENFORCE_DEFAULT): unset that variable and this build hard-fails instead,"
    echo "  which is what you almost certainly want. Rebuild ${LIB} from vendored"
    echo "  source to get the real upgrade path."
} >&2
exit 0
