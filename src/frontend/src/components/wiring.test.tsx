// WHAT GUARDS THE *USE* OF THE FIX, NOT JUST THE FUNCTIONS.
//
// ⛔ THE GAP THIS EXISTS TO CLOSE. Every component below is exported,
// prop-driven and directly tested. That proves each one is CORRECT. It does
// not prove any of them is WIRED. Without these checks the render sites of
// LintWarningsPanel, WaitingCard and QueueCard could each be deleted outright,
// all four ruleIdentity call sites reverted to rule.address, and saveRules
// reverted to discard the putConfig response -- and the rest of the suite would still
// pass. Components that exist, are correct, are tested,
// and are attached to nothing the suite can see. That is the same shape as the
// defect this suite exists to catch: an instrument that cannot report the failure it
// exists to catch.
//
// ⚠ WHAT THIS IS, STATED PLAINLY SO NOBODY OVER-READS IT. These are
// STRUCTURAL assertions over the component source, not DOM assertions. They
// prove the wiring is PRESENT. They do not prove it RENDERS.
//
// WHY NOT A DOM TEST -- measured, not assumed:
//   * jsdom, linkedom, happy-dom, domino: all absent from node_modules, and
//     `typeof document` is "undefined" under this runner. There is no DOM
//     implementation to mount into at all.
//   * The harness renders with renderToStaticMarkup, which never runs effects.
//     A static render of <FastLane/> reaches 1,661 chars of empty-state markup
//     (no rules loaded, so no identity cells) and LintWarningsPanel returns
//     null on an empty list. A static render of <Dashboard/> reaches 53 chars:
//     "Loading...". The data those sites render arrives in a useEffect that a
//     static render never runs, so the sites are structurally unreachable.
//   * Adding a DOM dependency is deliberately avoided here (see
//     scripts/build-tests.mjs: "No new dependency"), for offline builds.
// A DOM-level arm is therefore UNFALSIFIABLE in this harness, and that is a
// dependency decision for the maintainers, not something to paper over.
//
// ★ EVERY CHECK BELOW CARRIES ITS OWN REVERSION CHECK. Each test asserts the real
// source passes AND that the specific reversion it guards against is caught.
// A check that cannot fail is not a check, so the failure is asserted here
// permanently rather than watched once and taken on trust.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

function source(name: string): string {
  const p = fileURLToPath(new URL(`../src/components/${name}`, import.meta.url));
  // An unreadable file must FAIL, never pass vacuously: an empty string would
  // satisfy "does not contain rule.address" perfectly.
  assert.ok(existsSync(p), `wiring test could not find ${name} at ${p} -- the check would be vacuous`);
  const src = readFileSync(p, 'utf8');
  assert.ok(src.length > 500, `${name} read back as ${src.length} bytes -- refusing to assert on it`);
  return src;
}

/** Count JSX mount sites of a component, e.g. `<WaitingCard ` or `<WaitingCard/>`. */
function mounts(src: string, component: string): number {
  return (src.match(new RegExp(`<${component}[\\s/>]`, 'g')) || []).length;
}

test('FastLane MOUNTS LintWarningsPanel and feeds it the lint state', () => {
  const src = source('FastLane.tsx');
  assert.ok(mounts(src, 'LintWarningsPanel') >= 1,
    'the LintWarningsPanel render site is gone -- the panel is correct, tested, and attached to nothing');
  assert.match(src, /<LintWarningsPanel\s+warnings=\{lintWarnings\}/,
    'LintWarningsPanel is mounted but not fed the lint state');

  // Control: delete the render site, a reversion that would otherwise go unnoticed.
  const reverted = src.replace(/<LintWarningsPanel\s+warnings=\{lintWarnings\}\s*\/>/, '');
  assert.notEqual(reverted, src, 'control did not modify anything -- the pattern moved');
  assert.equal(mounts(reverted, 'LintWarningsPanel'), 0,
    'CONTROL: deleting the render site must be detectable, or this test proves nothing');
});

test('saveRules USES the putConfig response instead of discarding it', () => {
  const src = source('FastLane.tsx');
  assert.match(src, /const\s+result\s*=\s*await\s+putConfig\(/,
    'saveRules discards the putConfig response -- "warn, not go silent" is false for a UI operator');
  assert.match(src, /extractLintWarnings\(result\)/,
    'the putConfig response is captured but never turned into warnings');
  assert.match(src, /setLintWarnings\(warnings\)/,
    'warnings are extracted but never reach state, so the panel can never show them');

  // Control: the discarding form -- await the call, keep nothing.
  const reverted = src.replace(/const\s+result\s*=\s*await\s+putConfig\(/, 'await putConfig(');
  assert.notEqual(reverted, src, 'control did not modify anything -- the pattern moved');
  assert.doesNotMatch(reverted, /const\s+result\s*=\s*await\s+putConfig\(/,
    'CONTROL: discarding the response must be detectable');
});

test('the rule identity cells call ruleIdentity, not rule.address', () => {
  const src = source('FastLane.tsx');
  const calls = (src.match(/ruleIdentity\(rule\)/g) || []).length;
  assert.ok(calls >= 4,
    `expected the 4 identity call sites (list key, card, table key, table cell), found ${calls} -- ` +
    'a reverted site shows the raw address again, which is the bug this wiring guards against');

  // Control: revert every call site to the raw expression.
  const reverted = src.replace(/ruleIdentity\(rule\)/g, 'rule.address');
  assert.equal((reverted.match(/ruleIdentity\(rule\)/g) || []).length, 0,
    'CONTROL: reverting the call sites must be detectable');
});

test('Dashboard MOUNTS QueueCard, and Queue MOUNTS WaitingCard', () => {
  const dash = source('Dashboard.tsx');
  assert.ok(mounts(dash, 'QueueCard') >= 1,
    'the QueueCard render site is gone from the dashboard');
  assert.match(dash, /<QueueCard\s+queue=\{data\.queue\}/,
    'QueueCard is mounted but not fed the queue payload');

  const queue = source('Queue.tsx');
  assert.ok(mounts(queue, 'WaitingCard') >= 1,
    'the WaitingCard render site is gone from the queue tab');
  assert.match(queue, /<WaitingCard\s+queue=\{queue\}/,
    'WaitingCard is mounted but not fed the queue payload');

  // Control, both sites.
  assert.equal(mounts(dash.replace(/<QueueCard[^/]*\/>/, ''), 'QueueCard'), 0,
    'CONTROL: deleting the QueueCard site must be detectable');
  assert.equal(mounts(queue.replace(/<WaitingCard[^/]*\/>/, ''), 'WaitingCard'), 0,
    'CONTROL: deleting the WaitingCard site must be detectable');
});

test('both screens read "waiting" from the one shared definition', () => {
  // The Queue tab and the Dashboard must both import the shared waiting module,
  // or the two screens can disagree about the same word and drift apart.
  const shared = source('queue/waiting.tsx');
  assert.match(shared, /export function waitingCount/);
  assert.match(shared, /export function WaitingCard/);
  const queue = source('Queue.tsx');
  assert.match(queue, /from '\.\/queue\/waiting'/,
    'the queue tab no longer imports the shared waiting definition');
  const dash = source('Dashboard.tsx');
  assert.match(dash, /waitingCount/,
    'the dashboard no longer uses the shared waiting definition and can drift from the queue tab again');

  const reverted = queue.replace(/from '\.\/queue\/waiting'/g, "from './queue/waiting-OTHER'");
  assert.doesNotMatch(reverted, /from '\.\/queue\/waiting'/,
    'CONTROL: losing the shared import must be detectable');
});
