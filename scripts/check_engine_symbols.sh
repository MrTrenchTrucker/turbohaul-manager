#!/usr/bin/env bash
# check_engine_symbols.sh -- build-time guard against a mixed/partial engine build.
#
# A mixed build can happen when a Docker layer copies a freshly-built libllama-server-impl over a stale
# libllama.so that is missing llama_state_seq_save_file_capped. Every gate the build
# already has would pass -- it compiles, it boots, it serves, /health is green -- because
# the missing symbol is only ever touched on the FIRST KV save, in production. This
# script exists to catch that class of defect at BUILD time, before the image ships.
#
# Symbol versions are normalised before comparing. `nm -D --undefined-only` prints a
# versioned undefined reference as `name@VERSION` (ONE @); `nm -D --defined-only`
# prints that same symbol's resolving definition as `name@@VERSION` (TWO @ -- marks the
# default version unversioned references bind to). Those two strings can never be
# string-equal, so without normalisation EVERY strong undefined libc symbol in ANY
# real binary would look unresolved -- not sometimes, always. A test fixture that
# calls no versioned symbols would never reveal this. The script therefore strips
# everything from the first `@` on both sides before comparing (see NORMALISE below).
# This means the check proves "the symbol exists somewhere reachable", not "the exact
# glibc ABI version matches" -- that's the right guarantee for the
# missing-symbol defect class (total absence of a symbol), not a stricter
# ABI-version-skew check this script does
# not attempt.
#
# The dynamic loader is scanned separately. `__tls_get_addr@GLIBC_2.3` is provided by the
# DYNAMIC LOADER (e.g. ld-linux-x86-64.so.2), not by any library. `ldd` shows the loader
# as an absolute path with NO `=>` arrow, so a scan limited to `=>` lines would miss it,
# and passing that line through the same libc/libstdc++/libgcc_s/libm filter used for
# resolved dependencies would discard it, since the loader's filename never matches.
# The script instead scans the loader's own defined-symbol table separately,
# unfiltered by name (see LOADER below); without that, a healthy build that uses
# thread-local storage would report `__tls_get_addr@GLIBC_2.3` as unresolved,
# although it resolves at runtime.
#
# WHAT THIS CHECKS: every STRONG undefined ("U") dynamic symbol in the built
# server-impl library resolves (by bare name, version-normalised) against (1) a sibling
# lib*-prefixed file in the same build directory, (2) the runtime library stack
# (libc/libstdc++/libgcc_s/libm) via ldd, or (3) the dynamic loader itself (whatever
# ldd reports as its non-`=>` absolute-path entry -- see LOADER below).
#
# Weak-undefined symbols (nm type "w"/"v", e.g. compiler-emitted hooks like
# __gmon_start__) are deliberately NOT
# required to resolve -- they default to a safe null value when absent, by ELF
# specification, and are not the defect class here; treating them as required would
# false-positive on ordinary GCC-built C++ .so files (a gate that false-fires on healthy
# builds is exactly as worthless as one that never fires on broken ones -- both get
# switched off). This exclusion is NOT silent: the weak-undefined count is always
# printed, and CHECK_ENGINE_SYMBOLS_VERBOSE=1 prints their names too, so a future reader
# can tell "considered and excluded" apart from "this script forgot weak symbols exist."
#
# ARTIFACT NAMING: the built server-impl library is NOT guaranteed a `.so` extension on
# every build (an assumption of one would be wrong). This script looks for anything named
# `libllama-server-impl*` in the build dir (matching CMake's `lib`-prefix convention,
# which IS consistently applied, rather than the suffix, which is not) and likewise
# scans `lib*` -- not `*.so*` -- for sibling providers, so detection survives whichever
# suffix convention a given build produces.
#
# DELIBERATE SCOPE LIMIT: this does NOT chase sibling-to-sibling reference rings (e.g.
# a dropped libggml-cpu.so that some OTHER sibling references, but libllama-server-impl
# itself never touches). CPU/CUDA/NCCL accelerator libs are environment-dependent -- a
# bare build host legitimately lacks some of them, and a stricter transitive-closure
# check would false-positive there. This script only proves the server-impl library's
# OWN direct undefined-symbol set resolves; it does not prove the whole dependency
# graph is closed. That is a narrower, cheaper, and more reliable guarantee than a full
# transitive check -- and it is the exact guarantee the missing-symbol defect class needs.
#
# VALIDATION REQUIREMENT: this script must be proven against the REAL engine artifacts
# in place in a real build dir, not a fixture copied out of one (copying can break ldd's
# resolution) and not a synthetic fixture alone. A
# synthetic fixture is a legitimate can-it-go-red CONTROL; it is never the validation.
#
# Usage:   check_engine_symbols.sh [build-bin-dir]
#          (default build-bin-dir: /opt/turboquant/build/bin)
#          CHECK_ENGINE_SYMBOLS_VERBOSE=1  also names the excluded weak-undefined
#          symbols, not just their count.
# Exit 0:  consistent build, OR the server-impl library is absent/empty (nothing to
#          check is not a failure -- not every build produces this target).
# Exit 1:  unresolved symbols found -- lists them (with their original version suffix,
#          for greppability) on stderr.

set -euo pipefail

BUILD_DIR="${1:-/opt/turboquant/build/bin}"

# Find the server-impl library by lib-prefix naming convention, not by extension --
# the real built artifact has been observed with NO `.so` suffix at all.
shopt -s nullglob
impl_candidates=("${BUILD_DIR}"/libllama-server-impl*)
shopt -u nullglob
IMPL=""
for c in "${impl_candidates[@]}"; do
    if [ -s "$c" ]; then
        IMPL="$c"
        break
    fi
done

if [ -z "$IMPL" ]; then
    echo "check_engine_symbols: no libllama-server-impl artifact in ${BUILD_DIR} (looked for libllama-server-impl*) -- nothing to check, OK"
    exit 0
fi

for tool in nm ldd; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "check_engine_symbols: required tool '$tool' not found on PATH" >&2
        exit 1
    fi
done

# Every symbol IMPL needs from outside itself. Type "U" only (strong/global undefined)
# -- deliberately excludes weak-undefined ("w"/"v"), see header. Both sets are extracted
# from the SAME nm invocation so the legibility report below can never drift from what
# was actually excluded. `_raw` keeps the original (possibly versioned) name for display;
# `_norm` strips from the first `@` for matching -- see NORMALISE in the header.
all_undefined="$(nm -D --undefined-only "$IMPL" 2>/dev/null || true)"
undefined_syms_raw="$(printf '%s\n' "$all_undefined" | awk '$1 == "U" {print $2}' | sort -u)"
undefined_syms_norm="$(printf '%s\n' "$undefined_syms_raw" | sed -E 's/@.*//' | sort -u)"
weak_syms="$(printf '%s\n' "$all_undefined" | awk '$1 ~ /^[wv]$/ {print $2}' | sort -u)"

if [ -z "$weak_syms" ]; then
    weak_count=0
else
    weak_count="$(printf '%s\n' "$weak_syms" | wc -l | tr -d ' ')"
fi
echo "check_engine_symbols: ${weak_count} weak-undefined symbol(s) excluded by design (ELF-spec null-default when absent, not the missing-symbol class)"
if [ "$weak_count" -gt 0 ] && [ -n "${CHECK_ENGINE_SYMBOLS_VERBOSE:-}" ]; then
    printf '%s\n' "$weak_syms" | sed 's/^/    weak (excluded): /'
fi

if [ -z "$undefined_syms_norm" ]; then
    echo "check_engine_symbols: ${IMPL} has zero required (strong) undefined symbols, OK"
    exit 0
fi

sibling_defined="$(mktemp)"
trap 'rm -f "$sibling_defined"' EXIT

# 1) Every defined dynamic symbol across every OTHER lib*-prefixed file in the same
# build dir (lib-prefix, not .so-extension -- see ARTIFACT NAMING in the header).
shopt -s nullglob
for so in "${BUILD_DIR}"/lib*; do
    [ "$so" = "$IMPL" ] && continue
    [ -f "$so" ] || continue
    nm -D --defined-only "$so" 2>/dev/null | awk '{print $NF}' | sed -E 's/@.*//' >> "$sibling_defined" || true
done
shopt -u nullglob

# 2) The runtime stack: resolve the shared libs ldd reports for IMPL itself (honours
# its RPATH/RUNPATH, same as production resolution -- point LD_LIBRARY_PATH at the
# build dir too so sibling deps of IMPL resolve during this probe, not just
# libc/libstdc++), then pull each candidate's defined dynamic symbols. Limited to
# libc/libstdc++/libgcc_s/libm on purpose -- see the scope-limit note above.
ldd_out="$(LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH:-}" ldd "$IMPL" 2>/dev/null || true)"
runtime_libs="$(printf '%s\n' "$ldd_out" | awk '/=>/{print $3}' | grep -E '/lib(c|stdc\+\+|gcc_s|m)\.so' || true)"

# 3) The dynamic LOADER itself (e.g. ld-linux-x86-64.so.2). ldd shows it as an
# ABSOLUTE PATH WITH NO `=>` ARROW, because it's the ELF interpreter, not a resolved
# dependency -- and it provides its own runtime symbols (__tls_get_addr, the _dl_*
# family, ...) that NEITHER the sibling scan above NOR the `=>` lines just above ever
# see. The awk step captures the loader's non-`=>` line correctly, but a
# libc/libstdc++/libgcc_s/libm name filter would then throw it away,
# since the loader's filename never matches that pattern --
# on a real healthy build that calls __tls_get_addr (any TLS use), the loader-provided
# symbol would read as unresolved. So this branch is NOT name-filtered at all: ldd's own
# `=>`-vs-not distinction already identifies the interpreter (the other non-`=>` case,
# vdso, is excluded by requiring an absolute path -- vdso has no on-disk file). This
# scans the loader's FULL defined-symbol table rather than allow-listing
# `__tls_get_addr` by name, on purpose -- do not assume it is the only loader-provided
# symbol a binary can need; a name allow-list rots, a full scan does not.
loader="$(printf '%s\n' "$ldd_out" | awk '!/=>/{if ($1 ~ /^\//) print $1}')"

for lib in $runtime_libs $loader; do
    [ -f "$lib" ] && { nm -D --defined-only "$lib" 2>/dev/null | awk '{print $NF}' | sed -E 's/@.*//' >> "$sibling_defined" || true; }
done

sort -u -o "$sibling_defined" "$sibling_defined"

unresolved_norm="$(comm -23 <(printf '%s\n' "$undefined_syms_norm") "$sibling_defined")"

if [ -n "$unresolved_norm" ]; then
    echo "check_engine_symbols: UNRESOLVED symbols in ${IMPL} (mixed/partial engine build):" >&2
    while IFS= read -r bare; do
        matches="$(printf '%s\n' "$undefined_syms_raw" | grep -E "^${bare}(@|\$)" || true)"
        if [ -n "$matches" ]; then
            printf '%s\n' "$matches" | sed 's/^/  /' >&2
        else
            echo "  ${bare}" >&2
        fi
    done <<< "$unresolved_norm"
    exit 1
fi

echo "check_engine_symbols: ${IMPL} -- all undefined symbols resolve, OK"
exit 0

