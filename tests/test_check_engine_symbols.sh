#!/usr/bin/env bash
# test_check_engine_symbols.sh -- the can-it-go-red CONTROL for check_engine_symbols.sh.
#
# This is NOT the real-artifact validation -- it is a synthetic fixture, and it may
# only ever be the control that proves the gate CAN fire, never the
# proof it does not FALSE-fire on a real build. That proof requires the real engine
# artifacts in place (arms 1-3 against the demo image), run separately.
#
# Unlike an earlier fixture, this one deliberately carries a REAL GLIBC-VERSIONED
# undefined symbol (waitpid, via libc) alongside an own-code one (fixture_provided_
# symbol). The earlier fixture carried zero versioned symbols, which is exactly why its
# clean-room pass proved nothing about the bug class that shipped -- a control that
# cannot exercise the bug class is decoration, not a control.
#
# Requires: gcc, nm, ldd (all present in this sandbox; not required at image build
# time -- this is a dev-time self-check of the gate, not part of the Docker build).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/check_engine_symbols.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cat > "${WORK}/provider.c" <<'EOF'
int fixture_provided_symbol(void) { return 42; }
EOF
cat > "${WORK}/impl.c" <<'EOF'
#include <sys/wait.h>
#include <unistd.h>
extern int fixture_provided_symbol(void);
int impl_entry(pid_t p, int *status, int options) {
    int a = fixture_provided_symbol();
    int b = waitpid(p, status, options);
    return a + b;
}
EOF

BUILD="${WORK}/builddir"
mkdir -p "$BUILD"
gcc -shared -fPIC -o "${BUILD}/libprovider_fixture.so" "${WORK}/provider.c"
# No .so extension on purpose: mirrors the real artifact name observed in
# the demo image. --unresolved-symbols=ignore-all mirrors the real build's shape (two
# independently-built layers; the provider isn't linked in, only co-located at runtime).
gcc -shared -fPIC -Wl,--unresolved-symbols=ignore-all \
    -o "${BUILD}/libllama-server-impl" "${WORK}/impl.c"

fail=0

echo "--- sanity: fixture carries both an own-code AND a GLIBC-versioned undefined symbol ---"
if ! nm -D --undefined-only "${BUILD}/libllama-server-impl" | grep -q "fixture_provided_symbol"; then
    echo "FIXTURE BUG: fixture_provided_symbol not undefined in impl" >&2
    exit 2
fi
if ! nm -D --undefined-only "${BUILD}/libllama-server-impl" | grep -qE "waitpid@[A-Za-z0-9_.]+"; then
    echo "FIXTURE BUG: no versioned waitpid undefined ref in impl -- fixture does not exercise the bug class" >&2
    exit 2
fi
echo "OK: fixture exercises both the sibling-resolution path and the @ / @@ versioning path."
echo

echo "--- ARM 1: consistent (impl + provider both present) -> expect EXIT 0 ---"
set +e
bash "$SCRIPT" "$BUILD" >"${WORK}/arm1.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm1.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 1: PASS"
else
    echo "ARM 1: FAIL (expected exit 0, got $rc) -- an earlier version of this fixture never ran this arm; a fail here means the versioning fix is not working." >&2
    fail=1
fi
echo

echo "--- ARM 2: broken (provider MOVED OUT of the build dir, not renamed in place) -> expect EXIT 1 naming fixture_provided_symbol, NOT waitpid ---"
BROKEN="${WORK}/broken"
mkdir -p "$BROKEN"
cp "${BUILD}/libllama-server-impl" "$BROKEN/"
HIDDEN="$(mktemp -d)"
cp "${BUILD}/libprovider_fixture.so" "$HIDDEN/"  # left OUT of $BROKEN entirely
set +e
bash "$SCRIPT" "$BROKEN" >"${WORK}/arm2.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm2.out"
if [ "$rc" -eq 1 ] && grep -q "fixture_provided_symbol" "${WORK}/arm2.out" && ! grep -q "waitpid" "${WORK}/arm2.out"; then
    echo "ARM 2: PASS (fails for the right symbol, and does not spuriously flag the still-resolvable versioned one)"
else
    echo "ARM 2: FAIL (expected exit 1 naming fixture_provided_symbol only, got exit $rc)" >&2
    fail=1
fi
rm -rf "$HIDDEN"
echo

echo "--- ARM 3: impl missing entirely -> expect EXIT 0 graceful ---"
EMPTY="$(mktemp -d)"
set +e
bash "$SCRIPT" "$EMPTY" >"${WORK}/arm3.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm3.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 3: PASS"
else
    echo "ARM 3: FAIL (expected exit 0, got $rc)" >&2
    fail=1
fi
rm -rf "$EMPTY"
echo

echo "--- ARM 4: healthy build that calls a LOADER-provided symbol (TLS access -> __tls_get_addr) -> expect EXIT 0 ---"
echo "    (this is the arm that caught a real bug in an earlier version of the check: arm 1 against the REAL artifacts failed"
echo "     EXIT 1 on __tls_get_addr@GLIBC_2.3 -- provided by ld-linux-x86-64.so.2, not any library --"
echo "     because the earlier ldd scan filtered the loader's non-'=>' line through the libc-family name"
echo "     filter, which the loader's own filename never matches. No synthetic fixture before this"
echo "     one exercised that path; this fixture exists so it can't regress silently again.)"
TLSWORK="${WORK}/tls"
mkdir -p "$TLSWORK"
cat > "${TLSWORK}/tls_impl.c" <<'EOF'
__thread int fixture_tls_var = 0;
int bump_tls(void) { return ++fixture_tls_var; }
EOF
TLSBUILD="${TLSWORK}/builddir"
mkdir -p "$TLSBUILD"
gcc -shared -fPIC -ftls-model=global-dynamic -o "${TLSBUILD}/libllama-server-impl" "${TLSWORK}/tls_impl.c"
if ! nm -D --undefined-only "${TLSBUILD}/libllama-server-impl" | grep -q "__tls_get_addr"; then
    echo "FIXTURE BUG: this compiler/flag combo did not emit a __tls_get_addr reference -- arm 4 cannot exercise the bug it exists to catch" >&2
    exit 2
fi
set +e
bash "$SCRIPT" "$TLSBUILD" >"${WORK}/arm4.out" 2>&1
rc=$?
set -e
cat "${WORK}/arm4.out"
if [ "$rc" -eq 0 ]; then
    echo "ARM 4: PASS"
else
    echo "ARM 4: FAIL (expected exit 0, got $rc -- the loader-scan fix has regressed)" >&2
    fail=1
fi
echo

if [ "$fail" -eq 0 ]; then
    echo "test_check_engine_symbols: ALL 4 CONTROL ARMS PASSED. This proves the gate CAN discriminate on fixtures that carry a versioned symbol AND a loader-provided one. It does NOT prove arms 1-3 against the real in-place artifacts -- that verification is separate and required before ship."
    exit 0
else
    echo "test_check_engine_symbols: AT LEAST ONE CONTROL ARM FAILED. Do not ship." >&2
    exit 1
fi

