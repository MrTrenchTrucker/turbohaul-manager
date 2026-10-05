// Renders ThroughputSection for a given residents[] snapshot and
// asserts the data-testid/data-value hooks against aggregate.ts's
// own pure-function outputs -- proving the JSX actually WIRES to combinedTokS/
// prefillMean/outputFraction/alarmCounts rather than e.g. reading a single
// resident's own fields (the "structurally blind" bug class: blanking the
// whole box would still leave every check green).
//
// residentAlarm/useBusyTimers are injected test
// doubles here, never real severity logic -- these tests prove
// ThroughputSection CALLS the injected classifier and threads busyForS
// through, not that any particular resident field means "stalled" (that
// judgment belongs to the residentAlarm classifier alone).
//
// SSR boundary: renderToStaticMarkup never runs useEffect, so every test here
// exercises a SINGLE synchronous render of a given residents[] snapshot --
// this proves the Hero/split-bar/banner wiring, not the sparkline's own
// tick-to-tick accrual (that is proven directly and more thoroughly by the
// pure advanceCombinedSparkline unit tests in aggregate.test.tsx, which need
// no rendering at all).
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  ThroughputSection,
  SPARK_SAMPLES,
  combinedContext,
  resolveResidentContext,
  mergeLastKnownContext,
} from './ThroughputSection';
import { contextTone } from './contextTone';
import { combinedTokS, outputFraction, prefillMean } from './aggregate';
import type { AlarmToken, BusyTimers, ResidentAlarm, UseBusyTimers } from './aggregate';
import type { GenerationInfo, ResidentModel } from '../../api';

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    ...overrides,
  };
}

function makeResident(overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    model_tag: 'model-a-27b',
    state: 'ACTIVE',
    port: 11500,
    pid: 4242,
    spawn_seq: 1,
    reserved_need_mib: 24000,
    parallel: 1,
    main_gpu: 0,
    split_mode: 'single',
    inflight: 1,
    idle_expires_in_s: null,
    generation: makeGeneration(),
    ...overrides,
  };
}

// Default test doubles for tests that don't care about the alarm machinery.
const noAlarm: ResidentAlarm = () => '';
const noBusyTimers: UseBusyTimers = () => ({});

function tokenByTagDouble(byTag: Record<string, AlarmToken>): ResidentAlarm {
  return (resident) => byTag[resident.model_tag] ?? '';
}

function extractDataValue(html: string, testid: string): string | null {
  const re = new RegExp(`data-testid="${testid}"[^>]*?data-value="([^"]*)"`);
  const m = html.match(re);
  return m ? m[1] : null;
}

// No other test asserted the RENDERED WIDTH, so a hardcoded
// `const prefillWidth = 50;` (a half-full bar) would pass every
// other check -- data-value/text/element-presence all stay honest while
// the bar itself was wrong. This reads
// the real inline style width: `style="width:NN%"` (no space around the colon).
function extractWidthPct(html: string, testid: string): number | null {
  const re = new RegExp(`data-testid="${testid}"[^>]*?style="width:([\\d.]+)%"`);
  const m = html.match(re);
  return m ? Number(m[1]) : null;
}

test('ThroughputSection: 0 residents renders the IDLE panel -- agg-box present, no hero/bar/banner testids, no NaN anywhere', () => {
  const html = renderToStaticMarkup(
    <ThroughputSection residents={[]} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.match(html, /data-testid="agg-box"/);
  assert.match(html, /no active generation/);
  assert.doesNotMatch(html, /data-testid="agg-tok-s"/);
  assert.doesNotMatch(html, /data-testid="agg-prefill-pct"/);
  assert.doesNotMatch(html, /data-testid="agg-output-pct"/);
  assert.doesNotMatch(html, /data-testid="agg-alarm-banner"/);
  assert.doesNotMatch(html, /NaN/);
});

test('ThroughputSection: 1 resident -- agg-tok-s/agg-prefill-pct/agg-output-pct data-value match aggregate.ts exactly', () => {
  const residents = [
    makeResident({ generation: makeGeneration({ state: 'prefill', tok_s: 12, prefill_pct: 40 }) }),
  ];
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.equal(extractDataValue(html, 'agg-tok-s'), String(combinedTokS(residents)));
  assert.equal(extractDataValue(html, 'agg-prefill-pct'), String(prefillMean(residents)));
  assert.equal(extractDataValue(html, 'agg-output-pct'), String(outputFraction(residents)));
});

test(
  'ThroughputSection: nobody prefilling -- agg-prefill-pct stays PRESENT (element + attribute both), ' +
    'data-value is the EMPTY STRING (not "0", not the literal "null", not omitted) -- revised contract. ' +
    'A String(prefillMean(...))-style comparison would wrongly expect the string "null" here; this test ' +
    'checks the actual DOM contract instead.',
  () => {
    const residents = [
      makeResident({ generation: makeGeneration({ state: 'generating', tok_s: 30 }) }),
    ];
    assert.equal(prefillMean(residents), null, 'sanity: aggregate.ts itself agrees nobody is prefilling');
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.match(html, /data-testid="agg-prefill-pct"/, 'the element must still be present');
    assert.equal(extractDataValue(html, 'agg-prefill-pct'), '', 'the empty string, distinguishable from a dropped element (which would read null)');
    assert.match(html, />—<\/span>/, 'the visible text is a dash, never "0%" and never "NaN%"');
    assert.doesNotMatch(html, /NaN/);
  },
);

test(
  'ThroughputSection: the RENDERED WIDTH tracks the real value -- the hardcoded-width mutation catch. ' +
    'no-data must render width:0%, a live value must render a PROPORTIONAL width (not a constant): ' +
    'a hardcoded `const prefillWidth = 50` passed every other check (data-value/text/element-presence) ' +
    'while painting the bar HALF FULL with nothing prefilling -- the one thing a user actually ' +
    'looks at was the one thing nothing pinned. Two distinct points on the line, not one, because a ' +
    'no-data-only assertion is itself killed by a different hardcoded constant (e.g. `width: 0` always).',
  () => {
    const idle = [makeResident({ generation: makeGeneration({ state: 'generating', tok_s: 30 }) })];
    const idleHtml = renderToStaticMarkup(
      <ThroughputSection residents={idle} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractWidthPct(idleHtml, 'agg-prefill-pct'), 0, 'no-data must render 0% width, not a nonzero constant');

    const live = [makeResident({ generation: makeGeneration({ state: 'prefill', tok_s: undefined, prefill_pct: 63 }) })];
    const liveHtml = renderToStaticMarkup(
      <ThroughputSection residents={live} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractWidthPct(liveHtml, 'agg-prefill-pct'), 63, 'a real 63% mean must render width:63%, not any other constant');
  },
);

test(
  'ThroughputSection: a genuinely prefilling resident still shows the real percentage -- ' +
    'regression guard on the non-null path after the null-handling change',
  () => {
    const residents = [
      makeResident({ generation: makeGeneration({ state: 'prefill', tok_s: undefined, prefill_pct: 77 }) }),
    ];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractDataValue(html, 'agg-prefill-pct'), '77');
    assert.match(html, />77%</);
  },
);

test(
  'ThroughputSection: N>=2 residents -- agg-tok-s equals the SUM across residents, not any single resident\'s own ' +
    'value -- distinguishing fixture for the exact "structurally blind, one box, blanked whole" bug class',
  () => {
    // Two residents with DIFFERENT tok_s, chosen so the sum (40) equals
    // neither resident's own individual value (10, 30) -- a wrong wiring
    // that reads e.g. residents[0]'s own tok_s instead of combinedTokS(...)
    // would produce 10 here, not 40.
    const residents = [
      makeResident({ model_tag: 'a', generation: makeGeneration({ tok_s: 10 }) }),
      makeResident({ model_tag: 'b', generation: makeGeneration({ tok_s: 30 }) }),
    ];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractDataValue(html, 'agg-tok-s'), '40');
    assert.equal(combinedTokS(residents), 40, 'sanity: aggregate.ts itself agrees the sum is 40');
  },
);

test('ThroughputSection: alarm banner ABSENT when residentAlarm returns clear for everyone', () => {
  const residents = [makeResident({ model_tag: 'a' }), makeResident({ model_tag: 'b' })];
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.doesNotMatch(html, /data-testid="agg-alarm-banner"/);
});

test('ThroughputSection: alarm banner PRESENT, data-value is the TOTAL, breakdown text distinguishes stalled from no-telemetry', () => {
  const residents = [
    makeResident({ model_tag: 'a' }),
    makeResident({ model_tag: 'b' }),
    makeResident({ model_tag: 'c' }),
  ];
  const residentAlarm = tokenByTagDouble({ a: 'stalled', b: 'no-telemetry', c: '' });
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={residentAlarm} useBusyTimers={noBusyTimers} />,
  );
  // Total across both non-empty tokens, not a "did anything happen" boolean.
  assert.equal(extractDataValue(html, 'agg-alarm-banner'), '2');
  // Both operational facts must be visible, not collapsed into one number.
  assert.match(html, /1 stalled/);
  assert.match(html, /1 no telemetry/);
  assert.match(html, /3 resident/);
  assert.doesNotMatch(html, /data-testid="agg-busy-count"/, 'nobody is busy in this fixture -- the busy-count line must not render at all when the count is 0, same "nothing to show when there\'s nothing to say" convention as the banner itself');
});

test(
  'ThroughputSection busy-only residents render NO alarm banner at all -- busy alone is ordinary ' +
    'traffic, not an alarm (it must not read as "ALARM 1 busy of 2 residents"). The neutral ' +
    'busy-count line DOES still show the count, with no red/alarm styling anywhere near it.',
  () => {
    const residents = [makeResident({ model_tag: 'a' }), makeResident({ model_tag: 'b' })];
    const residentAlarm = tokenByTagDouble({ a: 'busy', b: '' });
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={residentAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.doesNotMatch(html, /data-testid="agg-alarm-banner"/, 'a lone busy resident must never trip the banner');
    assert.equal(extractDataValue(html, 'agg-busy-count'), '1');
    assert.match(html, /1 busy/, 'the count is still visible somewhere, just not as an alarm');
    assert.doesNotMatch(html, /border-red-700/, 'nothing on the page should carry alarm-red styling when the only fact is "busy"');
  },
);

test(
  'ThroughputSection -- busy + one stalled: the banner renders, but shows ONLY ' +
    'the stalled count, both in data-value (1, not 2) and in the breakdown text (no "busy" substring inside ' +
    'the banner itself, even though the page-level busy-count line elsewhere legitimately says "1 busy"). ' +
    'Scoped to the banner\'s own markup specifically so this doesn\'t collide with the separate, correct ' +
    '"1 busy" text in the header -- a whole-page doesNotMatch(/busy/) would wrongly fail on a correct render.',
  () => {
    const residents = [makeResident({ model_tag: 'a' }), makeResident({ model_tag: 'b' })];
    const residentAlarm = tokenByTagDouble({ a: 'busy', b: 'stalled' });
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={residentAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractDataValue(html, 'agg-alarm-banner'), '1', 'busy must not inflate the total alongside a real alarm');
    const bannerStart = html.indexOf('data-testid="agg-alarm-banner"');
    const bannerEnd = html.indexOf('— see cards below', bannerStart);
    assert.ok(bannerStart !== -1 && bannerEnd !== -1, 'sanity: the banner actually rendered');
    const bannerHtml = html.slice(bannerStart, bannerEnd);
    assert.match(bannerHtml, /1 stalled/);
    assert.doesNotMatch(bannerHtml, /busy/, 'the banner\'s own breakdown text must never mention busy, even when a real alarm is also present');
  },
);

test('ThroughputSection: alarm banner counts by CALLING residentAlarm, never by inspecting generation.stalled itself -- distinguishing fixture', () => {
  // generation.stalled is explicitly false and state is 'generating' on both
  // -- a component that still re-derived severity from generation fields
  // (forbidden) would render no banner regardless of the injected verdict.
  const residents = [
    makeResident({ model_tag: 'a', generation: makeGeneration({ stalled: false, state: 'generating' }) }),
  ];
  const residentAlarm: ResidentAlarm = () => 'no-telemetry';
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={residentAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.equal(extractDataValue(html, 'agg-alarm-banner'), '1');
  assert.match(html, /1 no telemetry/);
});

test('ThroughputSection: useBusyTimers is called ONCE over the whole set and its per-resident result reaches residentAlarm via busyForS', () => {
  const residents = [makeResident({ model_tag: 'the-tag' })];
  const timers: BusyTimers = { 'the-tag': 137 };
  const useBusyTimersDouble: UseBusyTimers = (rs) => {
    assert.equal(rs, residents, 'useBusyTimers must receive the same residents array');
    return timers;
  };
  let receivedBusyForS: number | null | undefined;
  const residentAlarm: ResidentAlarm = (_resident, busyForS) => {
    receivedBusyForS = busyForS;
    return busyForS != null && busyForS > 100 ? 'no-telemetry' : '';
  };
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={residentAlarm} useBusyTimers={useBusyTimersDouble} />,
  );
  assert.equal(receivedBusyForS, 137);
  assert.equal(extractDataValue(html, 'agg-alarm-banner'), '1');
});

test(
  'ThroughputSection: the top box is ONE bordered container, not two -- the ' +
    'merge, PINNED after a mutation survived once. Re-wrapping the merged box in the OLD ' +
    '`grid grid-cols-1 lg:grid-cols-3` two-column layout leaves the single p-6 box\'s own class untouched ' +
    '(box-count alone does not catch it) -- so this also asserts NO grid-* class exists anywhere in the ' +
    'render. This same test also covers the number+graph SIBLINGS-IN-A-ROW requirement: ' +
    'renderToStaticMarkup has no layout engine (confirmed), so this proves STRUCTURE only -- a real ' +
    'flex row wraps a shrink-0 number block and a flex-1 min-w-0 sparkline wrapper, in that order, with the ' +
    'old stacked border-t divider completely gone. A test that only checks box-count and hero-before-spark ' +
    'ORDER (the original shape) cannot tell "siblings in a row" from "stacked in the same box", which is ' +
    'exactly the distinction that matters here. The row wrapper\'s className is ' +
    'now `flex flex-col gap-3 sm:flex-row sm:items-center` (App.tsx:41\'s own idiom) so mobile widths stack ' +
    'Hero above a full-width sparkline instead of splitting an ~86px sliver off a shrink-0 hero block -- this ' +
    'assertion is updated to the NEW exact string deliberately, not loosened into a substring/toContain match. ' +
    'This still proves STRUCTURE only (the className string and the shrink-0/flex-1 document order); it does ' +
    'NOT and cannot prove rendered width at 412px -- renderToStaticMarkup has no layout engine, so that half ' +
    'of the proof bar is a live browser geometry gate at 412px/1680px, not this test.',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ tok_s: 30 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    // Exact string, closing quote immediately after p-6 -- distinguishes
    // this box from SplitBar's identically-prefixed p-4 box and from the
    // IDLE box (p-6 followed by more classes), both confirmed by inspecting
    // the actual source: only the merged top box's class ends exactly here.
    const boxOpens = (html.match(/class="rounded-lg border border-slate-700 bg-slate-950 p-6"/g) || []).length;
    assert.equal(boxOpens, 1, 'exactly one top-level bordered box for the merged Hero+sparkline pair, not two');
    // THE ACTUAL FIX: no CSS grid class anywhere. ThroughputSection.tsx
    // uses no legitimate grid layout at all today (confirmed by reading the
    // source) -- any `grid`/`grid-cols` class appearing here means the old
    // 2-column constraint was reintroduced around the merged box, exactly
    // what "undo the merge" looks like even when the box's OWN class is
    // untouched (box-count would still be 1, but this assertion catches it).
    assert.doesNotMatch(html, /\bgrid\b/, 'no grid wrapper anywhere -- box-count alone missed this');
    // NOTE: SplitBar (prefill/output) legitimately keeps its OWN separate
    // box below -- only the Hero +
    // combined-sparkline pair is merged, not the split bar too. Not asserting its
    // absence here would be wrong; it's a different widget by design.
    //
    // A DOCUMENT-ORDER chain, not a siblinghood proof. `combined
    // tok/s (last` (the old order anchor) no longer renders at all
    // post-merge -- reusing it here would silently prove nothing (an absent
    // substring makes every ordering comparison after it vacuously true
    // against -1). Each class string below is confirmed unique within this
    // component's render output; the final anchor is BigSparkline's
    // placeholder text (see sparkIdx below), not its aria-label --
    // BigSparkline itself is untouched (the component was never
    // modified), but its aria-label only exists on the real <svg> branch,
    // and every fixture in this file has samples===[] under SSR (useEffect
    // never runs), so BigSparkline always takes its samples.length < 2
    // placeholder branch here.
    //
    // CAVEAT: indexOf-based ordering can only prove these markers
    // appear in this SEQUENCE in the rendered HTML string -- it cannot
    // distinguish the sparkline wrapper being a true FLEX SIBLING of the
    // shrink-0 block from it being nested INSIDE that block instead (every
    // index below still ascends either way, so that regression stays
    // green here). String matching is the wrong instrument for that
    // distinction. True sibling-vs-nested DOM structure, and the actual
    // side-by-side visual layout, are proven by a live-browser geometry check
    // (asserts the number's and the graph's
    // bounding boxes overlap vertically and the graph sits to the right),
    // not by this unit test.
    const boxIdx = html.indexOf('class="rounded-lg border border-slate-700 bg-slate-950 p-6"');
    const rowIdx = html.indexOf('class="flex flex-col gap-3 sm:flex-row sm:items-center"', boxIdx);
    const shrinkIdx = html.indexOf('class="shrink-0"', rowIdx);
    const heroIdx = html.indexOf('Combined Throughput', shrinkIdx);
    const sparkWrapIdx = html.indexOf('class="flex-1 min-w-0"', heroIdx);
    // samples stays [] in every SSR render here (useEffect never runs), so
    // BigSparkline always takes its samples.length < 2 placeholder branch --
    // the aria-label only exists on the real <svg> branch, which this fixture
    // never reaches. Anchor on the placeholder text instead, which is the one
    // guaranteed to render.
    const sparkIdx = html.indexOf('gathering samples', sparkWrapIdx);
    assert.ok(
      boxIdx < rowIdx &&
        rowIdx < shrinkIdx &&
        shrinkIdx < heroIdx &&
        heroIdx < sparkWrapIdx &&
        sparkWrapIdx < sparkIdx,
      'a flex row opens, then a shrink-0 Hero block, then a flex-1 min-w-0 sparkline wrapper, in that ' +
        'DOCUMENT ORDER, inside the merged box -- proves order and that the divider markup is gone, NOT that ' +
        'the sparkline wrapper is a true flex sibling rather than nested inside Hero (see the live geometry ' +
        'guard, a separate check, for that claim)',
    );
    // The old stacked shape's divider must be gone entirely, not just
    // reordered -- a regression that re-adds `mt-4 pt-4 border-t` around a
    // re-stacked sparkline would still pass every check above it if this
    // were missing (box-count stays 1, grid stays absent, and a sloppy
    // siblings check could still find shrink-0/flex-1 divs INSIDE a
    // border-t wrapper rather than beside each other).
    assert.doesNotMatch(html, /border-t border-slate-800/, 'the stacked-layout divider must not survive the merge');
  },
);

test(
  'ThroughputSection: SPARK_SAMPLES is PINNED at 60 -- a mutation (30->60 on the ' +
    'card side, and this constant is the hero-panel analogue) silently doubling the lookback window ' +
    'survived the entire suite once already. This is the WINDOW the combined graph describes, not a ' +
    'resolution knob -- the layout change stretched the pixel WIDTH deliberately without touching this number.',
  () => {
    assert.equal(SPARK_SAMPLES, 60);
  },
);

test(
  'ThroughputSection: removed captions -- "sum across every active resident" and ' +
    '"instant: X tok/s" are GONE (both redundant post-merge: the first restates what "Combined ' +
    'Throughput" already says, the second is a literal duplicate of the same combined value now shown ' +
    'at text-7xl right beside the graph), while the window size and peak survive, merged into one ' +
    'compact line -- the only two facts in the old three-caption set that were not already visible ' +
    'elsewhere in the merged row.',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ tok_s: 30 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.doesNotMatch(html, /sum across every active resident/, 'redundant with "Combined Throughput"');
    assert.doesNotMatch(html, /instant:/, 'redundant with the text-7xl number now beside it');
    // SSR never runs useEffect (file-header note), so samples stays
    // INITIAL_SPARKLINE_STATE.samples ([]) here and peak is always the
    // documented samples.length===0 fallback of 0 -- fmtTokS(0) === '0.0'.
    assert.match(
      html,
      /last 60 · peak 0\.0 tok\/s/,
      'window size and peak survive, merged into one line, with real fmtTokS/SPARK_SAMPLES values',
    );
  },
);


/* ------------------------------------------------------------------ */
/*  Context-size readout, OVERALL AGGREGATE                              */
/* ------------------------------------------------------------------ */

function extractText(html: string, testid: string): string | null {
  const re = new RegExp(`data-testid="${testid}"[^>]*>([^<]*)<`);
  const m = html.match(re);
  return m ? m[1] : null;
}

test('resolveResidentContext PURE: current n_ctx present -> uses it, never the last-known fallback', () => {
  const resident = makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 40000 }) });
  const result = resolveResidentContext(resident, { used: 1, capacity: 2 });
  assert.deepEqual(result, { used: 40000, capacity: 100000, pct: 40 });
});

test(
  'resolveResidentContext PURE: current n_ctx null, a last-known reading supplied -> uses the fallback -- ' +
    'constraint 2, idle is a remembered value, never blanked',
  () => {
    const resident = makeResident({ generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) });
    const result = resolveResidentContext(resident, { used: 86000, capacity: 100000 });
    assert.deepEqual(result, { used: 86000, capacity: 100000, pct: 86 });
  },
);

test('resolveResidentContext PURE: current null AND no last-known -> null, genuinely never reported', () => {
  const resident = makeResident({ generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) });
  const result = resolveResidentContext(resident, undefined);
  assert.equal(result, null);
});

test(
  'combinedContext PURE: TRUE SUM over residents[], matching a worked example exactly ' +
    '(214,592/250,112 + 197,788/250,112 = 412,380/500,224)',
  () => {
    const residents = [
      makeResident({ model_tag: 'model-a-27b-q4-textonly', generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) }),
      makeResident({ model_tag: 'model-a-27b-q4', port: 11501, generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 197788 }) }),
    ];
    const combined = combinedContext(residents, {});
    assert.deepEqual(combined, { used: 412380, capacity: 500224 });
  },
);

test(
  'combinedContext PURE: DISTINGUISHING -- the sum is NOT either resident\'s own value alone (guards against ' +
    'a status.generation mirror: reading one resident\'s number and calling it ' +
    'the aggregate)',
  () => {
    const residents = [
      makeResident({ model_tag: 'a', generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) }),
      makeResident({ model_tag: 'b', port: 11501, generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 197788 }) }),
    ];
    const combined = combinedContext(residents, {});
    assert.notEqual(combined?.used, 214592, 'must not silently equal resident A alone');
    assert.notEqual(combined?.used, 197788, 'must not silently equal resident B alone');
    assert.equal(combined?.used, 412380, 'must be the true sum of both');
  },
);

test('combinedContext PURE: nobody has EVER reported context -> null, distinct from "0 used of some capacity"', () => {
  const residents = [
    makeResident({ generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) }),
  ];
  assert.equal(combinedContext(residents, {}), null);
});

// There is no worstContextPct here, and no dedicated tests for it:
// worst-of does not drive the
// aggregate's tone, and there are no
// production callers (a helper with no caller becomes a tested ghost: a test
// suite implying a behaviour the UI no longer has). The regression guard for
// an idle resident's remembered near-full context
// lives in ResidentCard.test.tsx, since the anti-dilution
// property it protects belongs to the per-resident card: each card
// colours by its own value, including an idle one on last-known (the
// per-resident card is the ONLY place the property lives, instead of
// being duplicated on the aggregate too).

test(
  'combinedContext PURE + contextTone: a worked example -- 98% + 10% sums/averages to 54%, which is ' +
    'GREEN under the 75/90 thresholds -- and that IS the target behaviour, not a bug to guard against. ' +
    'The combined number is the ONLY thing tone comes from, on ' +
    'purpose, by design -- see the render-level reversal-proof test below for the full end-to-end ' +
    'wiring, not just this pure-function sanity check.',
  () => {
    const residents = [
      makeResident({ model_tag: 'near-full', generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 98000 }) }),
      makeResident({ model_tag: 'mostly-empty', port: 11501, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 10000 }) }),
    ];
    const combined = combinedContext(residents, {});
    const combinedPct = combined ? (combined.used / combined.capacity) * 100 : null;
    assert.equal(combinedPct, 54, 'sanity: the combined/average number really does read 54');
    assert.equal(contextTone(combinedPct!), 'green', 'the combined box reads green off its OWN value even though one resident individually sits at 98%');
  },
);

test('ThroughputSection: 1 resident -- agg-context-used/-capacity/-pct data-value match combinedContext exactly', () => {
  const residents = [makeResident({ generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) })];
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  const combined = combinedContext(residents, {});
  assert.equal(extractDataValue(html, 'agg-context-used'), String(combined!.used));
  assert.equal(extractDataValue(html, 'agg-context-capacity'), String(combined!.capacity));
  assert.equal(extractDataValue(html, 'agg-context-pct'), String((combined!.used / combined!.capacity) * 100));
  assert.match(html, /214,592<\/span> \/ <span[^>]*>250,112/, 'formatted used/capacity line renders with thousands separators (each number in its own span, separator as a bare text node between them)');
});

test('ThroughputSection: 2 residents -- agg-context headline is the TRUE SUM, matching worked numbers', () => {
  const residents = [
    makeResident({ model_tag: 'model-a-27b-q4-textonly', generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) }),
    makeResident({ model_tag: 'model-a-27b-q4', port: 11501, generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 197788 }) }),
  ];
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.equal(extractDataValue(html, 'agg-context-used'), '412380');
  assert.equal(extractDataValue(html, 'agg-context-capacity'), '500224');
  assert.match(html, /412,380<\/span> \/ <span[^>]*>500,224/);
  assert.match(html, />82%</, '412380/500224 rounds to 82%');
});

test(
  'ThroughputSection: the RENDERED WIDTH of agg-context-bar tracks the real combined pct, not a ' +
    'hardcoded value (same lesson as the prefill bar width test)',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 37000 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.equal(extractWidthPct(html, 'agg-context-bar'), 37);
  },
);

test('ThroughputSection: no resident has ever reported context -> "-" text, no NaN, bar at 0 width', () => {
  const residents = [makeResident({ generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) })];
  const html = renderToStaticMarkup(
    <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
  );
  assert.match(html, /agg-context-pct"[^>]*>—</);
  assert.doesNotMatch(html, /NaN/);
  assert.equal(extractWidthPct(html, 'agg-context-bar'), 0);
});

test(
  'ThroughputSection: aggregate-side 75% BOUNDARY fixture -- exactly on the shared amber floor. ' +
    'Mutating contextTone.ts\'s threshold must fire BOTH the card suite (which already has a live 75% fixture) and ' +
    'this aggregate suite, or a duplicate classifier has crept back in. The aggregate\'s own ' +
    'colour comes from its own pct, not from the WORST resident, so a fixture on this boundary pins ' +
    'this file\'s own threshold usage.',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 75000 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.match(html, /agg-context-pct"[^>]*class="[^"]*text-amber-300/, '75% itself is amber, not green -- boundary inclusive on the amber side, same rule contextTone.test.ts pins directly');
    assert.match(html, /agg-context-bar"[^>]*class="[^"]*bg-amber-500/);
  },
);

test(
  'ThroughputSection: Combined Context row sits in its OWN bordered box, DIRECTLY AFTER the ' +
    'combined tok/s box -- used/capacity now sits BESIDE the label (moved up from below ' +
    'the pct number), THEN the pct + bar row underneath. Document-order chain only (see the ' +
    'caveat a few tests up: proves order, not true DOM siblinghood).',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    const tokBoxIdx = html.indexOf('class="rounded-lg border border-slate-700 bg-slate-950 p-6"');
    const ctxBoxIdx = html.indexOf('Combined Context', tokBoxIdx);
    const ctxUsedIdx = html.indexOf('data-testid="agg-context-used"', ctxBoxIdx);
    const ctxPctIdx = html.indexOf('data-testid="agg-context-pct"', ctxUsedIdx);
    const ctxBarIdx = html.indexOf('data-testid="agg-context-bar"', ctxPctIdx);
    assert.ok(
      tokBoxIdx !== -1 && tokBoxIdx < ctxBoxIdx && ctxBoxIdx < ctxUsedIdx && ctxUsedIdx < ctxPctIdx && ctxPctIdx < ctxBarIdx,
      'combined tok/s box, then Combined Context label, then used/capacity (moved up), then pct, then the bar, in that document order',
    );
  },
);


/* ------------------------------------------------------------------ */
/*  Additional coverage for gaps in the tests above. Three gaps are     */
/*  closed below; the fourth is a documented, pre-existing              */
/*  architectural boundary shared with resident-tok-s's own             */
/*  last-marker rendering -- see the comment above                      */
/*  resident-context-used in ResidentCard.tsx.                          */
/* ------------------------------------------------------------------ */

test(
  'mergeLastKnownContext PURE: an idle resident\'s entry is carried forward UNCHANGED across ticks, never ' +
    'dropped -- this is what actually protects useLastKnownContextByResident\'s hook wiring, since hooks ' +
    'cannot be exercised under this repo\'s SSR test harness (mutation-check finding: a "rebuild next={} ' +
    'from only this tick\'s live residents" mutation would be invisible to every OTHER test in this file, all ' +
    'of which hand-construct their lastKnownByKey fixture directly)',
  () => {
    const residentFull = makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 95000 }) });
    const tick1 = mergeLastKnownContext({}, [residentFull]);
    assert.deepEqual(tick1, { 'a:11500': { used: 95000, capacity: 100000 } });

    const residentIdle = makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) });
    const tick2 = mergeLastKnownContext(tick1, [residentIdle]);
    assert.deepEqual(
      tick2,
      { 'a:11500': { used: 95000, capacity: 100000 } },
      'the remembered reading must survive into the next tick even though THIS tick has no live data for it',
    );
  },
);

test('mergeLastKnownContext PURE: a resident that IS currently reporting overwrites its own prior entry with the fresh reading, never stacks/merges numerically', () => {
  const tick1 = mergeLastKnownContext({}, [makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 10000 }) })]);
  const tick2 = mergeLastKnownContext(tick1, [makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 50000 }) })]);
  assert.deepEqual(tick2, { 'a:11500': { used: 50000, capacity: 100000 } });
});

test('mergeLastKnownContext PURE: two DIFFERENT residents accumulate independently, one going idle does not disturb the other\'s live entry', () => {
  const a = makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 40000 }) });
  const b = makeResident({ model_tag: 'b', port: 11501, generation: makeGeneration({ n_ctx: 200000, n_prompt_tokens: 60000 }) });
  const tick1 = mergeLastKnownContext({}, [a, b]);
  const aIdle = makeResident({ model_tag: 'a', port: 11500, generation: makeGeneration({ state: 'idle', n_ctx: undefined, n_prompt_tokens: undefined }) });
  const tick2 = mergeLastKnownContext(tick1, [aIdle, b]);
  assert.deepEqual(tick2, {
    'a:11500': { used: 40000, capacity: 100000 },
    'b:11501': { used: 60000, capacity: 200000 },
  });
});

test(
  'ThroughputSection: agg-context-pct rendered TEXT uses proper ROUNDING (Math.round), not floor/' +
    'truncate -- discriminating fixture where the two disagree (86,500/100,000 = 86.5% -> 87 rounds up; a ' +
    'Math.floor mutation would show 86 here while the data-value attribute stayed correct, which the ' +
    'original 2-resident fixture\'s rounding-invariant 82.4% could not distinguish -- mutation-check finding)',
  () => {
    const residents = [makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 86500 }) })];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.match(html, />87%</, '86.5% must round UP to 87, not floor/truncate to 86');
  },
);

test(
  'ThroughputSection: THE REVERSAL, RENDER-LEVEL PROOF -- the rendered agg-context-bar/agg-context-pct ' +
    'CSS class reflects the COMBINED pct ONLY, even when one resident individually sits red-band and the two ' +
    'land in DIFFERENT tone bands. This is the SAME two-resident fixture an earlier revision used to ' +
    'prove the opposite (worst-resident-driven, red) -- reusing it here makes the reversal\'s before/after ' +
    'unmistakable: same inputs, opposite intended colour, on purpose, per the design clarification ("does ' +
    'not need to be yellow/orange at 40%"). Two fully LIVE residents (no last-known/useEffect needed -- this ' +
    'exercises the actual JSX wiring end to end, not just the pure combinedContext/contextTone sanity check ' +
    'above). A regression back to worst-of driving this box\'s colour -- or a stray tone prop reintroduced -- ' +
    'would flip this red and fail here immediately.',
  () => {
    const residents = [
      makeResident({ model_tag: 'near-full', generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 95000 }) }),
      makeResident({ model_tag: 'mostly-empty', port: 11501, generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 5000 }) }),
    ];
    const html = renderToStaticMarkup(
      <ThroughputSection residents={residents} residentAlarm={noAlarm} useBusyTimers={noBusyTimers} />,
    );
    assert.match(html, />50%</, 'sanity: the combined/average percentage really is 50, green-band');
    assert.match(html, /agg-context-bar"[^>]*class="[^"]*bg-emerald-500/, 'the bar must render GREEN off the combined 50% value, not red off the one near-full resident');
    assert.match(html, /agg-context-pct"[^>]*class="[^"]*text-emerald-300/, 'the headline percentage number itself must also render in the green text tone');
    assert.doesNotMatch(html, /agg-context-bar"[^>]*class="[^"]*bg-red-500/);
    assert.doesNotMatch(html, /agg-context-pct"[^>]*class="[^"]*text-red-400/);
    assert.doesNotMatch(html, /data-testid="agg-context-worst"/, 'no worst-item exists at all -- it was withdrawn, see the file header comment above ContextAggRow');
  },
);
