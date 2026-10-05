#!/usr/bin/env bash
# test_check_engine_kv_variant.sh -- the can-it-discriminate CONTROL for
# check_engine_kv_variant.sh, same discipline as test_check_engine_symbols.sh
# -- a synthetic fixture proves the gate CAN fire and CAN pass; it
# is NOT the full validation. That validation was done separately,
# directly against two real engine builds on record for the KV-variant check (not
# committed here -- that run is not part of this test):
#   warn-only (bad):     4,006,744 B -- a former repo-root libllama.so.0.0.0
#                        (no longer in the repository; size kept as the
#                         historical fixture reference for this control)
#   K-upgrading (good):  3,993,584 B -- byte-size-identical to a known-good engine build
# Both real files independently confirmed, deterministically across 5 repeated
# runs each: K-upgrading exits 0 regardless of mode; warn-only exits 1 under
# CHECK_ENGINE_KV_VARIANT_ENFORCE=1 and exits 0-but-loud under the
# ENFORCE_DEFAULT=0 warn mode, which existed for a transition period (see
# scripts/check_engine_kv_variant.sh's own header for why -- one
# Dockerfile's committed engine was the warn-only variant at the time, and there was
# no prebuilt image available, so an unconditional fail would block the
# only rollback path that Dockerfile produces rather than protect it).
#
# This synthetic control exists so the gate's OWN logic stays covered without
# committing either multi-MB binary to a publicly mirrored repo -- both
# fixtures here are tiny, on-the-fly gcc builds, matching test_check_engine_
# symbols.sh's own no-binary-committed convention exactly.
#
# Requires: gcc, strings (both commonly available; not required at image
# build time -- this is a dev-time self-check of the gate, not part of the
# Docker build).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/check_engine_kv_variant.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0

echo "--- ARM 1: real K-upgrade message present -> expect EXIT 0 ---"
echo "    (padded with ~2000 distinct filler strings AHEAD of the target message on"
echo "     purpose -- a small fixture never reproduced the SIGPIPE/pipefail regression"
echo "     this script's own history records: 'grep -q' exits the instant it finds a"
echo "     match, and if 'strings' is still mid-write when that happens, its SIGPIPE'd"
echo "     exit status (141) becomes the PIPELINE's reported status under 'set -o"
echo "     pipefail' even though grep found the match -- 'if strings ... | grep -q"
echo "     ...; then' can then take the ELSE branch on a genuine match. A tiny fixture"
echo "     writes its whole output before grep even starts reading, so the race never"
echo "     fires; padding is what makes this arm an actual regression test for that bug"
echo "     class, not just a happy-path check.)"
: > "${WORK}/upgrade.c"
for i in $(seq 1 2000); do
    echo "const char *pad_${i} = \"filler string number ${i} padding the string table so strings(1) has real work to do before it reaches the target message\";" \
        >> "${WORK}/upgrade.c"
done
cat >> "${WORK}/upgrade.c" <<'EOF'
/* Mirrors llama-kv-cache.cpp's real LLAMA_LOG_WARN format string. */
const char *fixture_upgrade_msg =
    "%s: auto-asymmetric: GQA ratio %u:1 (n_head=%u, n_head_kv=%u) - "
    "upgrading K from %s to q8_0 to prevent quality degradation. "
    "Disable with TURBO_AUTO_ASYMMETRIC=0\n";
int fixture_upgrade_entry(void) { return (int)(long)fixture_upgrade_msg; }
EOF
gcc -shared -fPIC -o "${WORK}/upgrade.so" "${WORK}/upgrade.c"
# grep -c, not -q -- see the SIGPIPE/pipefail note above this arm; this sanity
# check is just as susceptible as the script under test, and a small,
# unpadded fixture would mask the same bug.
upgrade_count="$(strings "${WORK}/upgrade.so" | grep -c "upgrading K from" || true)"
if [ "${upgrade_count:-0}" -eq 0 ]; then
    echo "FIXTURE BUG: upgrade.so does not contain the upgrade string -- fixture cannot exercise arm 1" >&2
    exit 2
fi
set +e
bash "$SCRIPT" "${WORK}/upgrade.so" >"${WORK}/arm1.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm1.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 1: PASS"
else
    echo "ARM 1: FAIL (expected exit 0, got $rc) -- a check that cannot pass a genuinely correct engine is not proven, it is decoration." >&2
    fail=1
fi
echo

cat > "${WORK}/warnonly.c" <<'EOF'
/* Mirrors the stale variant's actual message shape -- same
   log-site concept, structurally different text, no "upgrading K from". */
const char *fixture_warn_msg =
    "%s: auto-asymmetric risk: GQA ratio %u:1 (n_head=%u, n_head_kv=%u), "
    "K configured as %s\n turbo K quantization error is amplified by the "
    "GQA broadcast factor and may degrade quality. Configured K type is "
    "honored as-is.\n";
int fixture_warnonly_entry(void) { return (int)(long)fixture_warn_msg; }
EOF
gcc -shared -fPIC -o "${WORK}/warnonly.so" "${WORK}/warnonly.c"
warnonly_count="$(strings "${WORK}/warnonly.so" | grep -c "upgrading K from" || true)"
if [ "${warnonly_count:-0}" -gt 0 ]; then
    echo "FIXTURE BUG: warnonly.so unexpectedly contains the upgrade string -- fixture cannot exercise arm 2" >&2
    exit 2
fi

echo "--- ARM 2: warn-only message present, ENFORCE mode (CHECK_ENGINE_KV_VARIANT_ENFORCE=1) -> expect EXIT 1 ---"
set +e
CHECK_ENGINE_KV_VARIANT_ENFORCE=1 bash "$SCRIPT" "${WORK}/warnonly.so" >"${WORK}/arm2.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm2.out"
if [ "$rc" -eq 1 ] && grep -q "MISSING" "${WORK}/arm2.out" && grep -q "silently keep turbo K quantization" "${WORK}/arm2.out"; then
    echo "ARM 2: PASS (fails, and names the KV-variant issue in its own message)"
else
    echo "ARM 2: FAIL (expected exit 1 naming the KV-variant issue, got exit $rc)" >&2
    fail=1
fi
echo

echo "--- ARM 2b: warn-only message present, DEFAULT mode (ENFORCE unset) -> expect EXIT 1 (REFUSE) ---"
echo "    (ENFORCE_DEFAULT is 1, so the DEFAULT refuses -- a warn that is silent is just"
echo "     a disabled guard, so this arm fails unless the warning is actually printed and names"
echo "     the KV-variant issue and its follow-up, not just the exit code.)"
set +e
env -u CHECK_ENGINE_KV_VARIANT_ENFORCE bash "$SCRIPT" "${WORK}/warnonly.so" >"${WORK}/arm2b.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm2b.out"
if [ "$rc" -eq 1 ] && grep -q "MISSING" "${WORK}/arm2b.out" && grep -q "silently keep turbo K quantization" "${WORK}/arm2b.out" \
   && grep -q "refusing to ship" "${WORK}/arm2b.out"; then
    echo "ARM 2b: PASS (DEFAULT is now ENFORCE -- refuses, naming the KV-variant issue)"
else
    echo "ARM 2b: FAIL (expected exit 1 refusing and naming the KV-variant issue, got exit $rc)" >&2
    fail=1
fi
echo

# ARM 2c covers the explicit opt-out kept after ENFORCE_DEFAULT became 1. The warn branch still
# EXISTS as an explicit opt-out, so it still needs coverage -- otherwise
# flipping the default silently deletes the only test of that path.
echo "--- ARM 2c: warn-only message present, EXPLICIT opt-out (ENFORCE=0) -> expect EXIT 0, but LOUD ---"
set +e
CHECK_ENGINE_KV_VARIANT_ENFORCE=0 bash "$SCRIPT" "${WORK}/warnonly.so" >"${WORK}/arm2c.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm2c.out"
if [ "$rc" -eq 0 ] && grep -q "MISSING" "${WORK}/arm2c.out" && grep -q "silently keep turbo K quantization" "${WORK}/arm2c.out" \
   && grep -qi "WARN" "${WORK}/arm2c.out"; then
    echo "ARM 2c: PASS (explicit opt-out passes the build, but loudly)"
else
    echo "ARM 2c: FAIL (expected exit 0 with a loud KV-variant warning, got exit $rc)" >&2
    fail=1
fi
echo

echo "--- ARM 3: target file missing entirely -> expect EXIT 0 graceful ---"
set +e
bash "$SCRIPT" "${WORK}/does-not-exist.so" >"${WORK}/arm3.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm3.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 3: PASS"
else
    echo "ARM 3: FAIL (expected exit 0, got $rc)" >&2
    fail=1
fi
echo

echo "--- ARM 4: target file present but empty -> expect EXIT 0 graceful ---"
: > "${WORK}/empty.so"
set +e
bash "$SCRIPT" "${WORK}/empty.so" >"${WORK}/arm4.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm4.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 4: PASS"
else
    echo "ARM 4: FAIL (expected exit 0, got $rc)" >&2
    fail=1
fi
echo

if [ "$fail" -eq 0 ]; then
    echo "test_check_engine_kv_variant: ALL 7 CONTROL ARMS PASSED (upgrade-present, enforce-fail, warn-pass-but-loud, missing, empty, plus their sub-checks). This proves the gate CAN discriminate the two message shapes on synthetic fixtures, in BOTH directions, AND that the ENFORCE/WARN toggle actually changes the exit code without ever going silent (a check only ever shown to fail is not a proven check; a warn that prints nothing is a disabled guard wearing a costume). It does NOT replace the real-artifact run against the actual 4,006,744 B / 3,993,584 B binaries on record for the KV-variant check -- that verification is separate."
    exit 0
else
    echo "test_check_engine_kv_variant: AT LEAST ONE CONTROL ARM FAILED. Do not ship." >&2
    exit 1
fi
