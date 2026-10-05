// On-theme tag-priority dropdown, replacing a native <select>
// whose OPEN popup is OS-rendered on mobile (it looked like the phone's
// default) with a hand-rolled WAI-ARIA select-only combobox.
//
// This runner has no DOM/jsdom -- renderToStaticMarkup produces a plain HTML
// string with no event listeners attached, so click/keydown interactions
// cannot be simulated here (same limitation as every other test file in this
// package). What CAN be tested rigorously, with real dynamic assertions and
// no DOM at all, is the interaction STATE MACHINE itself: stepListbox is a
// pure function of (state, action) -> (state, commit?), so a full "user
// arrows through several options, then presses Enter" sequence can be
// replayed here exactly as it would fire from real onKeyDown handlers, and
// the keyboard rule ("arrow-browsing must never silently commit; only Enter/
// Space/click may") is asserted directly against that replay, not just
// against static markup. Wiring the real onClick/onKeyDown handlers in
// TagRankListbox to call this exact function is verified by direct code
// reading (FastLane.tsx's dispatch()), the same "no browser, disclosed and
// code-read-verified instead" pattern used for the parts of this file that
// genuinely need a real browser (position: fixed layout, scroll/resize).
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import FastLane, {
  alreadyInFastLane,
  applyTagRank,
  CLOSED_LISTBOX_STATE,
  computePopupPosition,
  DiscoveredNameCell,
  discoveredRuleFor,
  DiscoveredTableHead,
  extractLintWarnings,
  formatRefusalsByReason,
  LintWarningsPanel,
  formatTagRankOption,
  rulesAfterAddingDiscovered,
  ruleIdentity,
  RulesTableHead,
  stepListbox,
  TagRankListbox,
} from './FastLane';
import type { FastLaneDiscoverable, FastLaneRule } from '../api';

const rule = (address: string, ranks: Partial<Record<string, number | null>> = {}): FastLaneRule => ({
  address,
  label: '',
  tag_ranks: { main: null, curator: null, compression: null, sub_agent: null, unclassified: null, ...ranks },
});

// --- applyTagRank (pure data transform) -------------------------------

test('applyTagRank sets the rank for the targeted row and tag only', () => {
  const rules = [rule('a'), rule('b')];
  const next = applyTagRank(rules, 1, 'curator', 3);
  assert.equal(next[0].tag_ranks?.curator, null, 'row 0 must be untouched');
  assert.equal(next[1].tag_ranks?.curator, 3);
});

test('applyTagRank leaves the other four tags on the same row untouched', () => {
  const rules = [rule('a', { main: 5 })];
  const next = applyTagRank(rules, 0, 'curator', 2);
  assert.equal(next[0].tag_ranks?.main, 5, 'main must survive a curator-only edit');
  assert.equal(next[0].tag_ranks?.curator, 2);
});

test('applyTagRank(rank=null) clears a rank back to unranked ("--")', () => {
  const rules = [rule('a', { compression: 4 })];
  const next = applyTagRank(rules, 0, 'compression', null);
  assert.equal(next[0].tag_ranks?.compression, null);
});

test('applyTagRank falls back to empty ranks when tag_ranks was null on the row', () => {
  const rules: FastLaneRule[] = [{ address: 'a', label: '', tag_ranks: null }];
  const next = applyTagRank(rules, 0, 'sub_agent', 1);
  assert.equal(next[0].tag_ranks?.sub_agent, 1);
  assert.equal(next[0].tag_ranks?.main, null, 'the other four tags must default to null, not crash');
});

test('applyTagRank returns a new array and a new row object (no in-place mutation)', () => {
  const rules = [rule('a')];
  const next = applyTagRank(rules, 0, 'main', 1);
  assert.notEqual(next, rules);
  assert.notEqual(next[0], rules[0]);
  assert.equal(rules[0].tag_ranks?.main, null, 'the original array must be unaffected');
});

// --- stepListbox (the interaction state machine, keyboard rule) --------

test('closed by default', () => {
  assert.deepEqual(CLOSED_LISTBOX_STATE, { open: false, activeIndex: -1 });
});

test('open sets open=true at the given index', () => {
  const { state } = stepListbox(CLOSED_LISTBOX_STATE, { type: 'open', activeIndex: 2 });
  assert.deepEqual(state, { open: true, activeIndex: 2 });
});

test('close always resets to CLOSED_LISTBOX_STATE, regardless of prior activeIndex', () => {
  const { state } = stepListbox({ open: true, activeIndex: 4 }, { type: 'close' });
  assert.deepEqual(state, CLOSED_LISTBOX_STATE);
});

test('move is a no-op while closed (arrow keys before the popup is open do not silently move a hidden cursor)', () => {
  const { state, commitIndex } = stepListbox(CLOSED_LISTBOX_STATE, { type: 'move', delta: 1, optionCount: 6 });
  assert.deepEqual(state, CLOSED_LISTBOX_STATE);
  assert.equal(commitIndex, undefined);
});

test('move clamps at the last option going down (does not overflow past rank 5)', () => {
  const { state } = stepListbox({ open: true, activeIndex: 5 }, { type: 'move', delta: 1, optionCount: 6 });
  assert.equal(state.activeIndex, 5);
});

test('move clamps at index 0 going up (does not go negative past "--")', () => {
  const { state } = stepListbox({ open: true, activeIndex: 0 }, { type: 'move', delta: -1, optionCount: 6 });
  assert.equal(state.activeIndex, 0);
});

test('home jumps to index 0, end jumps to the last option, only while open', () => {
  assert.equal(stepListbox({ open: true, activeIndex: 3 }, { type: 'home' }).state.activeIndex, 0);
  assert.equal(stepListbox({ open: true, activeIndex: 0 }, { type: 'end', optionCount: 6 }).state.activeIndex, 5);
  assert.deepEqual(stepListbox(CLOSED_LISTBOX_STATE, { type: 'home' }).state, CLOSED_LISTBOX_STATE);
});

test('commitActive while closed does nothing and never returns a commitIndex', () => {
  const { state, commitIndex } = stepListbox(CLOSED_LISTBOX_STATE, { type: 'commitActive' });
  assert.deepEqual(state, CLOSED_LISTBOX_STATE);
  assert.equal(commitIndex, undefined);
});

test('commitActive while open commits the currently highlighted index and closes', () => {
  const { state, commitIndex } = stepListbox({ open: true, activeIndex: 3 }, { type: 'commitActive' });
  assert.equal(commitIndex, 3);
  assert.deepEqual(state, CLOSED_LISTBOX_STATE, 'must close on commit');
});

test('commitIndex (a direct option click) commits that index regardless of the highlighted one, and closes', () => {
  const { state, commitIndex } = stepListbox({ open: true, activeIndex: 1 }, { type: 'commitIndex', index: 4 });
  assert.equal(commitIndex, 4);
  assert.deepEqual(state, CLOSED_LISTBOX_STATE);
});

test('arrow-browsing through every option and then Escaping never produces a commitIndex at any step', () => {
  let state = CLOSED_LISTBOX_STATE;
  const commits: number[] = [];
  const sequence: Array<{ type: 'open'; activeIndex: number } | { type: 'move'; delta: 1 | -1; optionCount: number } | { type: 'close' }> = [
    { type: 'open', activeIndex: 0 },
    { type: 'move', delta: 1, optionCount: 6 },
    { type: 'move', delta: 1, optionCount: 6 },
    { type: 'move', delta: 1, optionCount: 6 },
    { type: 'close' }, // Escape: browsed to index 3 (rank 3) but never committed
  ];
  for (const action of sequence) {
    const result = stepListbox(state, action);
    state = result.state;
    if (result.commitIndex !== undefined) commits.push(result.commitIndex);
  }
  assert.deepEqual(commits, [], 'browsing + Escape must never fire a commit, or a rank would silently apply and save with no user confirmation');
  assert.deepEqual(state, CLOSED_LISTBOX_STATE);
});

test('positive case: open, arrow down twice, Enter commits the option actually landed on (index 2), not the original value', () => {
  let state = stepListbox(CLOSED_LISTBOX_STATE, { type: 'open', activeIndex: 0 }).state;
  state = stepListbox(state, { type: 'move', delta: 1, optionCount: 6 }).state;
  state = stepListbox(state, { type: 'move', delta: 1, optionCount: 6 }).state;
  assert.equal(state.activeIndex, 2);
  const { commitIndex } = stepListbox(state, { type: 'commitActive' });
  assert.equal(commitIndex, 2);
});

// --- computePopupPosition (pure layout math, no real DOM needed) ------

const VIEWPORT_390 = { width: 390, height: 844 };

test('places the popup below the trigger when there is room', () => {
  const r = computePopupPosition({ top: 100, bottom: 144, left: 20, right: 84, width: 64, height: 44 }, 264, VIEWPORT_390);
  assert.equal(r.placement, 'below');
  assert.equal(r.top, 144 + 4);
});

test('flips the popup above the trigger when it would overflow the bottom of the viewport', () => {
  const r = computePopupPosition({ top: 800, bottom: 844, left: 20, right: 84, width: 64, height: 44 }, 264, VIEWPORT_390);
  assert.equal(r.placement, 'above');
  assert.equal(r.top, 800 - 264 - 4);
});

test('clamps left so the popup never overflows the right edge at 390px', () => {
  const r = computePopupPosition({ top: 100, bottom: 144, left: 350, right: 386, width: 36, height: 44 }, 264, VIEWPORT_390);
  assert.ok(r.left + 64 <= VIEWPORT_390.width, `left=${r.left} would overflow a 390px viewport`);
});

test('clamps left so the popup never goes off the left edge', () => {
  const r = computePopupPosition({ top: 100, bottom: 144, left: -10, right: 54, width: 64, height: 44 }, 264, VIEWPORT_390);
  assert.ok(r.left >= 4);
});

// --- TagRankListbox structural markup (what renderToStaticMarkup CAN prove) --

test('closed listbox: aria-expanded=false, no popup in the DOM, no aria-activedescendant', () => {
  const html = renderToStaticMarkup(
    <TagRankListbox id="t1" value={null} label="Main priority" triggerClassName="" onCommit={() => {}} />,
  );
  assert.ok(html.includes('aria-expanded="false"'));
  assert.ok(!html.includes('role="listbox"'), 'popup must not be in the markup while closed');
  assert.ok(!html.includes('aria-activedescendant'));
});

test('role=combobox, aria-haspopup=listbox, and the aria-label exactly matches the label prop', () => {
  const html = renderToStaticMarkup(
    <TagRankListbox id="t2" value={2} label="Compression priority for 192.0.2.10" triggerClassName="" onCommit={() => {}} />,
  );
  assert.ok(html.includes('role="combobox"'));
  assert.ok(html.includes('aria-haspopup="listbox"'));
  assert.ok(html.includes('aria-label="Compression priority for 192.0.2.10"'));
});

test('trigger carries the 44px touch-target floor on both call sites (normalise both to 44px)', () => {
  const html = renderToStaticMarkup(
    <TagRankListbox id="t3" value={1} label="x" triggerClassName="min-w-[3.75rem] justify-center" onCommit={() => {}} />,
  );
  assert.ok(html.includes('min-h-[44px]'), 'the table-render trigger was min-h-[36px] before; must be 44px now');
});

test('trigger uses rounded-lg, not the file-inconsistent rounded', () => {
  const html = renderToStaticMarkup(
    <TagRankListbox id="t4" value={null} label="x" triggerClassName="" onCommit={() => {}} />,
  );
  assert.ok(html.includes('rounded-lg'));
});

test('displayed value text reflects the current value ("--" for null, the number otherwise)', () => {
  assert.equal(formatTagRankOption(null), '—');
  assert.equal(formatTagRankOption(3), '3');
  const htmlUnranked = renderToStaticMarkup(<TagRankListbox id="t5" value={null} label="x" triggerClassName="" onCommit={() => {}} />);
  const htmlRanked = renderToStaticMarkup(<TagRankListbox id="t6" value={5} label="x" triggerClassName="" onCommit={() => {}} />);
  assert.ok(htmlUnranked.includes('>—<'));
  assert.ok(htmlRanked.includes('>5<'));
});

test('never renders a native <select> at all -- the OS-popup element this component exists to remove', () => {
  const html = renderToStaticMarkup(<TagRankListbox id="t7" value={4} label="x" triggerClassName="" onCommit={() => {}} />);
  assert.ok(!html.includes('<select'));
});

// --- the server's lint warnings must REACH the operator -----------
//
// saveRules used to `await putConfig(...)` and discard the result, so a save
// that was accepted AND misconfigured looked identical on screen to a clean
// one -- "warn, not go silent" was false for anyone using the UI.

test('extractLintWarnings: pulls the server warnings through', () => {
  assert.deepEqual(
    extractLintWarnings({ status: 'ok', warnings: ['rule 3 never observed', 'dup address'] }),
    ['rule 3 never observed', 'dup address'],
  );
});

test('extractLintWarnings: a clean save yields none', () => {
  assert.deepEqual(extractLintWarnings({ status: 'ok', warnings: [] }), []);
  assert.deepEqual(extractLintWarnings({ status: 'ok' }), []);
});

test('extractLintWarnings: never throws inside a save handler on a hostile payload', () => {
  // This reads a SERVER body. Throwing here would turn a SUCCESSFUL save into
  // a failed one on screen -- worse than the bug being fixed.
  for (const bad of [null, undefined, 'nope', 42, [], { warnings: 'not-an-array' }, { warnings: null }]) {
    assert.deepEqual(extractLintWarnings(bad), []);
  }
});

test('extractLintWarnings: drops non-strings so nothing renders as [object Object]', () => {
  assert.deepEqual(
    extractLintWarnings({ warnings: ['real', { a: 1 }, null, '', 42, 'also real'] }),
    ['real', 'also real'],
  );
});

test('ruleIdentity: an address rule renders exactly as it always did', () => {
  assert.equal(ruleIdentity({ address: '192.0.2.5' }), '192.0.2.5');
});

test('ruleIdentity: a container_name rule shows the NAME, not an empty cell', () => {
  assert.equal(ruleIdentity({ container_name: 'gateway-svc' }), 'gateway-svc');
  // control: neither set must be empty, NOT the string "undefined" -- the
  // same phantom-value class as str(None) on the manager side.
  assert.equal(ruleIdentity({}), '');
  assert.notEqual(ruleIdentity({}), 'undefined');
});

// --- the warnings actually reach the SCREEN ---------------

test('LintWarningsPanel renders every warning the server returned', () => {
  const html = renderToStaticMarkup(
    <LintWarningsPanel warnings={['rule index 0 names container x, which currently resolves to NO addresses', 'dup address']} />,
  );
  assert.match(html, /data-testid="fastlane-lint-warnings"/);
  assert.match(html, /resolves to NO addresses/);
  assert.match(html, /dup address/);
  // the count must be stated, or an operator cannot tell one problem from six
  assert.match(html, /flagged 2/);
});

test('LintWarningsPanel renders NOTHING when the save was clean', () => {
  // Control: without this, a panel that always rendered would pass the arm
  // above and would also nag on every clean save until operators ignored it.
  const html = renderToStaticMarkup(<LintWarningsPanel warnings={[]} />);
  assert.equal(html, '');
});

test('LintWarningsPanel says "problem" for one and "problems" for many', () => {
  const one = renderToStaticMarkup(<LintWarningsPanel warnings={['a']} />);
  assert.match(one, /flagged 1 problem\b/);
  assert.doesNotMatch(one, /problems/);
});

// --- the identity column must not be headed "Address" ------
// A rule identifies its client by container name (on the docker network) or
// by address (off it), so a column headed "Address" is a false statement
// about what is below it (for example `gateway-svc`).
//
// A header of `>Address</th>` with `>Identity</th>` absent fails this check.

test('the rules table identity column is headed "Identity", not "Address"', () => {
  const html = renderToStaticMarkup(<RulesTableHead />);
  assert.ok(html.includes('>Identity</th>'),
    `identity column header missing from: ${html}`);
  assert.ok(!html.includes('>Address</th>'),
    'the rules table still heads the identity column "Address", which is false ' +
    'for a container_name rule');
});

test('the same markup still carries the columns that did NOT change', () => {
  // Without this, the assertions above would both hold against a component
  // that rendered nothing at all -- an empty string contains no "Address"
  // either. This proves the probe is reading a real header row.
  const html = renderToStaticMarkup(<RulesTableHead />);
  // TAG_LABELS are title-case in the MARKUP ('Main', 'Sub-agent'); the ALL-CAPS
  // a reader sees in the browser is the `uppercase` CSS class on the <thead>.
  // Asserting the on-screen casing here fails against correct markup -- dump
  // the values before writing the matcher.
  for (const col of ['>#</th>', '>Label</th>', '>Main</th>', '>Curator</th>',
                     '>Compression</th>', '>Sub-agent</th>', '>Unclassified</th>']) {
    assert.ok(html.includes(col), `control column ${col} missing — the probe is not ` +
      `reading the real header row, so the Identity assertion proves nothing: ${html}`);
  }
});

// ---------------------------------------------------------------------------
// Fast Lane FE: counter strip, dedupe, container name creation, lint-on-mount
// gating. No-DOM rule: pure functions and the
// component's INITIAL static markup get real dynamic red/green coverage;
// click/mount-effect wiring does not (see the final block below, which says
// so explicitly rather than manufacturing a green for it).
// ---------------------------------------------------------------------------

// --- formatRefusalsByReason (never a fabricated 0) ----------------

test('formatRefusalsByReason: null (never fetched, or backend without the field) reads "not yet available"', () => {
  assert.equal(formatRefusalsByReason(null), 'not yet available');
});

test('formatRefusalsByReason: {} (Fast Lane ran, refused nothing) reads "none" -- NOT the same string as null', () => {
  const empty = formatRefusalsByReason({});
  assert.equal(empty, 'none');
  assert.notEqual(empty, formatRefusalsByReason(null),
    '"never checked" and "checked, nothing wrong" must not render identically');
});

test('formatRefusalsByReason: a single reason renders "reason: count"', () => {
  assert.equal(formatRefusalsByReason({ budget: 3 }), 'budget: 3');
});

test('formatRefusalsByReason: multiple reasons are all listed, comma-separated', () => {
  const out = formatRefusalsByReason({ budget: 3, other_reason: 1 });
  assert.match(out, /budget: 3/);
  assert.match(out, /other_reason: 1/);
});

// --- alreadyInFastLane (trust the backend's `assigned`) -----------

test('alreadyInFastLane: assigned=true reads Added', () => {
  const entry = { address: '192.0.2.5', assigned: true } as FastLaneDiscoverable;
  assert.equal(alreadyInFastLane(entry), true);
});

test('alreadyInFastLane: assigned=false reads not-Added', () => {
  const entry = { address: '192.0.2.5', assigned: false } as FastLaneDiscoverable;
  assert.equal(alreadyInFastLane(entry), false);
});

test('alreadyInFastLane: assigned absent from the payload reads not-Added, not a crash', () => {
  const entry = { address: '192.0.2.5' } as FastLaneDiscoverable;
  assert.equal(alreadyInFastLane(entry), false);
});

test('a container_name-covered address (assigned=true) reads Added even though no rule.address equals it', () => {
  // An address-only predicate, `rules.some((r) => r.address
  // === entry.address)`, fails here: container_name rules have address:null,
  // so it NEVER matches, regardless of entry.assigned -- every
  // container_name-covered row showed an enabled "Add to Fast Lane". This is
  // exactly that shape: the backend marks the row assigned=true (it resolved
  // a container_name rule to this address), and there is no address
  // field on any rule to match against at all.
  const entry = { address: '10.0.0.5', assigned: true } as FastLaneDiscoverable;
  const rulesWithNoAddresses: FastLaneRule[] = [
    { container_name: 'gateway-svc', label: '', tag_ranks: null },
  ];
  // alreadyInFastLane does not take `rules` at all -- that IS the fix: the
  // old bug was structural (comparing against a field that container_name
  // rules never populate), not a missed edge case in a still-address-based
  // check. Asserting the signature doesn't accept `rules` documents that a
  // regression can't quietly reintroduce the address-comparison shape.
  assert.equal(alreadyInFastLane.length, 1, 'must take only the census entry, not rules -- the whole fix is not needing rules at all');
  assert.equal(alreadyInFastLane(entry), true);
  void rulesWithNoAddresses; // shown only to document the scenario being regression-guarded
});

// --- Full-page initial-render markup: renderToStaticMarkup --
// never fires useEffect (deferred to a DOM commit that never happens here),
// so this is safe -- it renders FastLane's INITIAL state only (enabled=false,
// rules=[], refusalsByReason=null), never a real fetch. That's still enough
// to prove the new markup exists with honest pre-fetch content.

test('the counter strip renders real fields, not the old stub text', () => {
  const html = renderToStaticMarkup(<FastLane />);
  assert.match(html, /Counters/);
  assert.match(html, /Refusals by reason/);
  assert.match(html, /not yet available/, 'pre-fetch state must read "not yet available", not a fabricated 0 or blank');
  assert.match(html, /Served \/ jumped per class/);
  assert.match(html, /Fairness firings/);
  assert.doesNotMatch(html, /counter strip are not yet wired/,
    'the counter-strip clause of the old stub must be GONE, not merely joined by new markup');
  // The part of the stub that is still true must survive.
  assert.match(html, /Live jump panel and last-ten-jumps log are not yet wired/);
});

test('a container-name rule can be created from the UI: the input and button exist', () => {
  const html = renderToStaticMarkup(<FastLane />);
  assert.match(html, /placeholder="container name \(docker network\)"/);
  assert.match(html, />Add by container name</);
});

// --- Mount-time lint warnings: honestly unfalsifiable in this harness ------------------------
//
// lintWarnings on mount needs a real backend read-path (GET /api/config has
// no equivalent to PUT's lint_rules() call -- an open gap). Separately: even if a
// mount-time source existed, whether
// the mount effect actually POPULATES lintWarnings is effect-timing behavior
// this renderer cannot observe at all (useEffect never fires under
// renderToStaticMarkup) -- that half is UNFALSIFIABLE in this
// harness regardless of the backend question, same as the counter strip's
// and the neighbouring fetch/click wiring above. Verified instead by direct code
// reading, the same disclosed-not-faked pattern this file already uses for
// TagRankListbox's real-browser-only concerns (see the top of this file).

// --- Discovered rows: confirmed container name ------------------------------

const EMPTY_RANKS = { main: null, curator: null, compression: null, sub_agent: null, unclassified: null };

const named = (name: unknown, address = '192.0.2.7'): FastLaneDiscoverable =>
  ({ address, container_name: name } as unknown as FastLaneDiscoverable);

test('discoveredRuleFor: a named row saves a rule by name with no address key', () => {
  const r = discoveredRuleFor(named('app-one'));
  assert.deepEqual(r, { container_name: 'app-one', label: '', tag_ranks: EMPTY_RANKS });
  assert.equal('address' in r, false);
});

test('discoveredRuleFor: the name is trimmed', () => {
  const r = discoveredRuleFor(named('  app-one \t'));
  assert.equal(r.container_name, 'app-one');
  assert.equal('address' in r, false);
});

test('discoveredRuleFor: a row without a usable name saves the address rule with no container_name key', () => {
  const cases: unknown[] = [null, undefined, '', '   ', 42];
  for (const c of cases) {
    const r = discoveredRuleFor(named(c, '192.0.2.9'));
    assert.deepEqual(r, { address: '192.0.2.9', label: '', tag_ranks: EMPTY_RANKS }, `name=${String(c)}`);
    assert.equal('container_name' in r, false, `name=${String(c)}`);
  }
  const noKey = discoveredRuleFor({ address: '192.0.2.9' } as FastLaneDiscoverable);
  assert.deepEqual(noKey, { address: '192.0.2.9', label: '', tag_ranks: EMPTY_RANKS });
  assert.equal('container_name' in noKey, false);
});

test('discoveredRuleFor: each call returns a fresh tag_ranks object', () => {
  const e = named('app-one');
  const a = discoveredRuleFor(e);
  const b = discoveredRuleFor(e);
  assert.notEqual(a.tag_ranks, b.tag_ranks);
  (a.tag_ranks as Record<string, number | null>).main = 3;
  assert.equal((b.tag_ranks as Record<string, number | null>).main, null);
  const c = discoveredRuleFor(named(null));
  const d = discoveredRuleFor(named(null));
  assert.notEqual(c.tag_ranks, d.tag_ranks);
});

test('alreadyInFastLane(entry, rules): backend assigned=true reads Added whatever the rules are', () => {
  const entry = { address: '192.0.2.7', container_name: 'app-one', assigned: true } as FastLaneDiscoverable;
  assert.equal(alreadyInFastLane(entry, []), true);
});

test('alreadyInFastLane(entry, rules): a named row with a rule of that exact name reads Added', () => {
  const entry = named('app-one');
  const rules: FastLaneRule[] = [{ container_name: 'app-one', label: '', tag_ranks: null }];
  assert.equal(alreadyInFastLane(entry, rules), true);
});

test('alreadyInFastLane(entry, rules): a rule with a different name does not make the row Added', () => {
  const entry = named('app-one');
  const rules: FastLaneRule[] = [{ container_name: 'worker-two', label: '', tag_ranks: null }];
  assert.equal(alreadyInFastLane(entry, rules), false);
});

test('alreadyInFastLane(entry, rules): an unnamed row is not made Added by a name rule', () => {
  const rules: FastLaneRule[] = [{ container_name: 'app-one', label: '', tag_ranks: null }];
  assert.equal(alreadyInFastLane(named(null), rules), false);
  assert.equal(alreadyInFastLane({ address: '192.0.2.7' } as FastLaneDiscoverable, rules), false);
});

test('alreadyInFastLane: one-argument form with assigned absent still reads not-Added', () => {
  assert.equal(alreadyInFastLane(named('app-one')), false);
});

test('DiscoveredTableHead: Name comes right after Address and the other columns remain', () => {
  const html = renderToStaticMarkup(<DiscoveredTableHead />);
  const heads = [...html.matchAll(/<th[^>]*>([^<]*)<\/th>/g)].map((m) => m[1]);
  const at = heads.indexOf('Address');
  assert.notEqual(at, -1);
  assert.equal(heads[at + 1], 'Name');
  for (const h of ['Requests', 'Last seen', 'Tag classes', 'Models']) {
    assert.ok(heads.includes(h), `missing column ${h}`);
  }
  assert.ok(html.indexOf('>Name<') > html.indexOf('>Address<'));
  assert.ok(html.indexOf('>Name<') < html.indexOf('>Requests<'));
});

test('DiscoveredNameCell: shows the confirmed name, and a dash when there is none', () => {
  const cell = (e: FastLaneDiscoverable) =>
    renderToStaticMarkup(<table><tbody><tr><DiscoveredNameCell entry={e} /></tr></tbody></table>);
  const withName = cell(named('app-one'));
  assert.match(withName, /data-testid="discovered-name"/);
  assert.match(withName, />app-one<\/td>/);
  const without = cell(named(null));
  assert.match(without, /data-testid="discovered-name"/);
  assert.match(without, />—<\/td>/);
  assert.doesNotMatch(without, /app-one/);
  assert.match(cell(named('   ')), />—<\/td>/);
});

// --- rulesAfterAddingDiscovered: what "Add to Fast Lane" saves --------------

const addrRule = (address: string): FastLaneRule => ({ address, label: 'x', tag_ranks: null });
const nameRule = (container_name: string): FastLaneRule => ({ container_name, label: 'y', tag_ranks: null });

test('rulesAfterAddingDiscovered: a named row appends a by-name rule at the end, nothing else changes', () => {
  const rules = [addrRule('192.0.2.1'), nameRule('worker-two')];
  const before = JSON.stringify(rules);
  const out = rulesAfterAddingDiscovered(rules, named('app-one', '192.0.2.7'));
  assert.ok(out !== null);
  assert.notEqual(out, rules);
  assert.equal(out.length, 3);
  assert.deepEqual(out[0], rules[0]);
  assert.deepEqual(out[1], rules[1]);
  assert.deepEqual(out[2], { container_name: 'app-one', label: '', tag_ranks: EMPTY_RANKS });
  assert.equal('address' in out[2], false);
  assert.equal(rules.length, 2);
  assert.equal(JSON.stringify(rules), before);
});

test('rulesAfterAddingDiscovered: an unnamed row appends the address rule at the end', () => {
  const rules = [nameRule('worker-two')];
  const out = rulesAfterAddingDiscovered(rules, named(null, '192.0.2.9'));
  assert.ok(out !== null);
  assert.equal(out.length, 2);
  assert.deepEqual(out[0], rules[0]);
  assert.deepEqual(out[1], { address: '192.0.2.9', label: '', tag_ranks: EMPTY_RANKS });
  assert.equal('container_name' in out[1], false);
  assert.equal(rules.length, 1);
});

test('rulesAfterAddingDiscovered: adding to an empty list gives exactly the one new rule', () => {
  const out = rulesAfterAddingDiscovered([], named('app-one'));
  assert.deepEqual(out, [{ container_name: 'app-one', label: '', tag_ranks: EMPTY_RANKS }]);
});

test('rulesAfterAddingDiscovered: a named row whose name already has a rule is a duplicate (null)', () => {
  assert.equal(rulesAfterAddingDiscovered([addrRule('192.0.2.1'), nameRule('app-one')], named('app-one')), null);
  assert.equal(rulesAfterAddingDiscovered([nameRule('app-one')], named('  app-one ')), null);
});

test('rulesAfterAddingDiscovered: an unnamed row whose address already has a rule is a duplicate (null)', () => {
  assert.equal(rulesAfterAddingDiscovered([nameRule('worker-two'), addrRule('192.0.2.9')], named(null, '192.0.2.9')), null);
});

test('rulesAfterAddingDiscovered: a named row whose address already has an address rule is still added by name', () => {
  const rules = [addrRule('192.0.2.7')];
  const out = rulesAfterAddingDiscovered(rules, named('app-one', '192.0.2.7'));
  assert.ok(out !== null);
  assert.equal(out.length, 2);
  assert.deepEqual(out[1], { container_name: 'app-one', label: '', tag_ranks: EMPTY_RANKS });
});

test('rulesAfterAddingDiscovered: an unnamed row whose address matches a name-only rule is added', () => {
  const rules = [nameRule('app-one')];
  const out = rulesAfterAddingDiscovered(rules, named(null, '192.0.2.7'));
  assert.ok(out !== null);
  assert.equal(out.length, 2);
  assert.deepEqual(out[1], { address: '192.0.2.7', label: '', tag_ranks: EMPTY_RANKS });
});
