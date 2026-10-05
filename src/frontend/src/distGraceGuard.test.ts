/* Packaging-layer guard — DOES THE SHIPPED BUNDLE CONTAIN THE GRACE RENDER?
 *
 * ⛔ WHY THIS FILE EXISTS:
 * A stale built bundle is a packaging defect, not an FE logic bug.
 * Dockerfile.engine-src COPYs the committed src/frontend/dist
 * and never runs a build, so the shipped bundle can lag the source: the
 * grace render (ResidentCard's `model.phase ===
 * 'GRACE'` branch and its `resident-countdown` element)
 * can exist in SOURCE — and graceCardRender.test.tsx tests it — but that
 * test imports the SOURCE component, so it goes green on an artifact that is
 * never actually shipped. A stale committed bundle contains ZERO occurrences
 * of `phase`; a rebuilt bundle contains the whole branch.
 *
 * WHAT THIS FILE PROVES: it reads the BUILT artifact under dist/ — the exact
 * bytes Dockerfile.engine-src would COPY — and asserts the grace render is in
 * there. It deliberately imports NO FE source component: importing source is
 * the exact shape that lets this defect go unnoticed.
 *
 * The markers checked are "phase", `phase==="GRACE"` and
 * "resident-countdown"; a stale committed dist contains none of them and a
 * freshly rebuilt one contains all three, so this test fails on a stale dist
 * and passes on a rebuilt dist.
 *
 * This file is ADDITIVE at the packaging layer. graceCardRender.test.tsx
 * (source-level wiring test) stays untouched and remains the instrument for
 * the component itself.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';

/* Locate the frontend root: the directory holding the built dist/index.html.
 * `npm test` runs from src/frontend (process.cwd()), but stay robust when
 * invoked from a parent/child directory: walk up from cwd until
 * dist/index.html is found. A missing dist/ is itself a packaging-defect
 * class (nothing to ship) — loud failure, never a silent skip. */
function findFrontendRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 6; i++) {
    if (existsSync(join(dir, 'dist', 'index.html'))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error(
    'Packaging guard: no dist/index.html under or above ' +
      process.cwd() +
      ' — there is no built artifact to guard (vite build never ran, or the wrong root). Rebuild with: cd src/frontend && npm run build.'
  );
}

const root = findFrontendRoot();
const assetsDir = join(root, 'dist', 'assets');

/* The built JS bundles only — never .js.map: source maps carry original source
 * text and would let the guard pass on a bundle that lacks the feature. */
const bundles = existsSync(assetsDir)
  ? readdirSync(assetsDir)
      .filter((f) => f.endsWith('.js') && !f.endsWith('.map'))
      .map((f) => join(assetsDir, f))
  : [];

const code = bundles.map((f) => readFileSync(f, 'utf8')).join('\n');

test('Packaging guard: a built JS bundle exists under dist/assets', () => {
  assert.ok(
    bundles.length >= 1,
    `no built JS bundle in ${assetsDir} — dist/ is empty or missing; the Dockerfile would COPY nothing shippable. Rebuild: cd src/frontend && npm run build`
  );
});

test('Packaging guard: the built bundle carries the per-resident phase resolver', () => {
  const hits = (code.match(/phase/g) ?? []).length;
  assert.ok(
    hits > 0,
    `the built bundle contains ZERO occurrences of "phase" — it predates the grace render. The committed dist is stale; rebuild with: cd src/frontend && npm run build`
  );
});

test('Packaging guard: the built bundle contains the GRACE branch of residentCountdown', () => {
  assert.match(
    code,
    /\.phase\s*===\s*["']GRACE["']/,
    'the built bundle lacks the `model.phase === "GRACE"` branch — during grace the card would fall back to the legacy state and read "busy" again. Stale dist; rebuild required.'
  );
});

test('Packaging guard: the built bundle renders the resident-countdown element', () => {
  assert.ok(
    code.includes('resident-countdown'),
    'the built bundle lacks the "resident-countdown" element — the countdown would not show. Stale dist; rebuild required.'
  );
});
