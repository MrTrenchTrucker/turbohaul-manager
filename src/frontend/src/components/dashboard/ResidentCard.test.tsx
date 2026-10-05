// ResidentCard.tsx previously had ZERO coverage (blanking the whole top box
// still left the suite green). These tests exercise
// the card as a pure function of its props via renderToStaticMarkup, same
// pattern as Dashboard.test.tsx -- no new dependency.
//
// busyForS is supplied directly as a prop in every test here rather than
// through useBusyTimers: React effects do not run under renderToStaticMarkup
// (SSR), so useBusyTimers' internal timer would always read null through
// this harness -- which is exactly why busyForS is a prop that
// ResidentCard receives, rather than a hook it calls internally. That
// move is what makes the no-telemetry path testable at all.
import test from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { ResidentCard, resolveTokSDisplay, resolveContextDisplay, LiveOutputBox, isNearBottom, kvRestoreTone, CARD_SPARK_SAMPLES } from './ResidentCard';
import { contextTone } from './contextTone';
import { BUSY_ESCALATE_S } from './alarms';
import type { GenerationInfo, LoadVerifyRecord, RequestIdentity, ResidentModel } from '../../api';
import type { GenPane } from '../../hooks/useLiveStream';

// tokRateTone is injected (the real dashboard/tokRate.ts,
// not imported here -- see ResidentCard.tsx). This stub mirrors the real
// tokRateTone's signature and tones (the real implementation has its own tests;
// this is only here so ResidentCard's tests can supply the required prop
// with realistic-shaped behaviour, not to test tokRateTone itself).
function stubTokRateTone(gen: GenerationInfo | null | undefined): 'live' | 'idle' | 'stalled' {
  if (!gen) return 'idle';
  if (gen.stalled || gen.state === 'stalled' || gen.prefill_stall_alarm) return 'stalled';
  if (gen.state === 'generating' || gen.state === 'prefill') return 'live';
  return 'idle';
}

function makeGeneration(overrides: Partial<GenerationInfo> = {}): GenerationInfo {
  return {
    state: 'generating',
    prompt_progress: null,
    stalled: false,
    streaming: true,
    generation_id: 'gen-1',
    measured_at_iso: '2026-08-20T00:00:00Z',
    tok_s: 42.5,
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

function makePane(overrides: Partial<GenPane> = {}): GenPane {
  return {
    paneKey: 'model-a-27b',
    generation_id: 'gen-1',
    model_tag: 'model-a-27b',
    text: '',
    done: false,
    lastTokS: 42.5,
    lastFrameAt: Date.now(),
    tokHistory: [],
    ...overrides,
  };
}

function makeLoadVerifyRecord(overrides: Partial<LoadVerifyRecord> = {}): LoadVerifyRecord {
  return {
    event: 'kv_restore',
    trigger: 'reuse',
    model_tag: 'model-a-27b',
    port: 11500,
    pid: 4242,
    process_alive: true,
    health_200: true,
    model_resident: true,
    kv_expected_tokens: null,
    kv_actual_n_past: null,
    restore_attempted: false,
    kv_restore_ok: null,
    retry_count: 0,
    final_status: 'unverified',
    reason: null,
    thread_hash: null,
    session_id: null,
    ...overrides,
  };
}

function dataValue(html: string, testId: string): string | undefined {
  const m = html.match(new RegExp(`data-testid="${testId}"[^>]*data-value="([^"]*)"`));
  return m?.[1];
}

function render(overrides: Partial<Parameters<typeof ResidentCard>[0]> = {}) {
  return renderToStaticMarkup(
    <ResidentCard model={makeResident()} busyForS={null} tokRateTone={stubTokRateTone} {...overrides} />,
  );
}

/* ------------------------------------------------------------------ */
/*  Structural: card root, tag                                          */
/* ------------------------------------------------------------------ */

test('ResidentCard: renders exactly one resident-card root', () => {
  const html = render();
  const count = (html.match(/data-testid="resident-card"/g) || []).length;
  assert.equal(count, 1);
});

test('ResidentCard: resident-tag carries the model_tag text', () => {
  const html = render({ model: makeResident({ model_tag: 'model-d-8b-fast' }) });
  assert.match(html, /data-testid="resident-tag"[^>]*>model-d-8b-fast</);
});

/* ------------------------------------------------------------------ */
/*  resident-tok-s: ALWAYS present (design rule), data-value=0 for      */
/*  generation:null so a missing element can never read as ambiguous.   */
/* ------------------------------------------------------------------ */

test('ResidentCard: generation:null -- card still renders, resident-tok-s present with data-value=0', () => {
  const html = render({ model: makeResident({ state: 'GRACE', generation: null }) });
  assert.match(html, /data-testid="resident-card"/, 'a generation:null resident must still get a card (design rule)');
  assert.equal(dataValue(html, 'resident-tok-s'), '0');
});

test('ResidentCard: live generation -- resident-tok-s data-value equals the real tok_s', () => {
  const html = render({ model: makeResident({ generation: makeGeneration({ tok_s: 17.3 }) }) });
  assert.equal(dataValue(html, 'resident-tok-s'), '17.3');
});

test('ResidentCard: DISTINGUISHING -- two residents with different tok_s render different data-value, not a shared/stale number', () => {
  const htmlA = render({ model: makeResident({ model_tag: 'a', generation: makeGeneration({ tok_s: 5 }) }) });
  const htmlB = render({ model: makeResident({ model_tag: 'b', generation: makeGeneration({ tok_s: 95 }) }) });
  assert.equal(dataValue(htmlA, 'resident-tok-s'), '5');
  assert.equal(dataValue(htmlB, 'resident-tok-s'), '95');
});

/* ------------------------------------------------------------------ */
/*  resident-alarm token wiring -- busyForS supplied as a prop, so the  */
/*  no-telemetry path (otherwise SSR-unreachable) IS exercisable here.  */
/* ------------------------------------------------------------------ */

test('ResidentCard: clean generation, busyForS null -> resident-alarm data-value empty', () => {
  const html = render({ busyForS: null });
  assert.equal(dataValue(html, 'resident-alarm'), '');
});

test('ResidentCard: gen.stalled -> resident-alarm data-value "stalled"', () => {
  const html = render({ model: makeResident({ generation: makeGeneration({ stalled: true }) }), busyForS: null });
  assert.equal(dataValue(html, 'resident-alarm'), 'stalled');
});

test('ResidentCard: busyForS >= BUSY_ESCALATE_S -> resident-alarm data-value "no-telemetry" -- the path SSR cannot reach through a hook, only through the prop', () => {
  const html = render({ busyForS: BUSY_ESCALATE_S });
  assert.equal(dataValue(html, 'resident-alarm'), 'no-telemetry');
});

test('ResidentCard: busyForS under threshold -> resident-alarm data-value "busy"', () => {
  const html = render({ busyForS: 5 });
  assert.equal(dataValue(html, 'resident-alarm'), 'busy');
});

test('ResidentCard: prefill_stall_alarm -> resident-alarm data-value "prefill-hang"', () => {
  const html = render({
    model: makeResident({ generation: makeGeneration({ prefill_stall_alarm: true }) }),
    busyForS: null,
  });
  assert.equal(dataValue(html, 'resident-alarm'), 'prefill-hang');
});

test('ResidentCard PRECEDENCE (through the real component, not just the pure function): stalled + high busyForS -> still "stalled", proving the wiring itself does not reorder precedence', () => {
  const html = render({
    model: makeResident({ generation: makeGeneration({ stalled: true }) }),
    busyForS: BUSY_ESCALATE_S,
  });
  assert.equal(dataValue(html, 'resident-alarm'), 'stalled');
});

/* ------------------------------------------------------------------ */
/*  Lighter live output (design rule): last non-empty line + ago,        */
/*  gated on both gen AND pane being present.                           */
/* ------------------------------------------------------------------ */

test('ResidentCard: pane present -- the FULL buffer renders in the scrolling box (not just the last line, ' +
  'the earlier "lighter" design this replaces)', () => {
  const html = render({
    pane: makePane({ text: 'first line\nsecond line\n\nthird line\n' }),
  });
  assert.match(html, /first line/, 'the box is a real scrolling buffer now, not a last-line-only preview');
  assert.match(html, /third line/);
});

test('ResidentCard: pane absent -> no live-output section rendered at all -- verified via the always-or-never testid, ' +
  'not a text label, since a text-label check would vacuously pass once the labels themselves changed', () => {
  const html = render({ pane: null });
  assert.doesNotMatch(html, /data-testid="resident-live-output-status"/);
});

test('ResidentCard: generation:null but a pane IS present -- the live-output box PERSISTS, showing the pane\'s text -- ' +
  'this is the open-question answer (persist + grey out, never disappear): pane lifecycle now tracks residentTags ' +
  'via useLiveStream, independent of model.generation, precisely so an idle-but-still-resident model does not vanish mid-glance', () => {
  const html = render({
    model: makeResident({ generation: null }),
    pane: makePane({ text: 'last thing this resident said before going idle', done: true }),
  });
  assert.match(html, /last thing this resident said before going idle/, 'the box must persist and show the last text, not hide because generation is null');
  assert.equal(dataValue(html, 'resident-live-output-status'), 'done', 'a done pane renders the DONE/grey state, not LIVE');
});

test('ResidentCard: no pane at all (resident truly has no SSE data yet, or never has) -- no live-output section, not a crash', () => {
  const html = render({
    model: makeResident({ generation: null }),
    pane: null,
  });
  assert.doesNotMatch(html, /DONE|LIVE/);
});

/* ------------------------------------------------------------------ */
/*  RequestIdentityStrip prop-name trap: the prop                       */
/*  is residentModelTag, not cardModelTag -- a wrong name is a SILENT   */
/*  no-op that only tsc would catch on an unknown-prop typo, and it     */
/*  has already bitten a hand-written probe fixture.                    */
/* ------------------------------------------------------------------ */

function makeIdentity(overrides: Partial<RequestIdentity> = {}): RequestIdentity {
  return {
    ip: '10.0.0.5',
    model_tag: 'model-a-27b',
    session_id: 'sess-abc123',
    is_main: true,
    is_sub_agent: false,
    is_curator: false,
    is_compression: false,
    resolved_class: null,
    thread_id: 'thread-xyz',
    ...overrides,
  };
}

test('ResidentCard: RequestIdentityStrip renders the identity at all (basic wiring, does not by itself prove the prop NAME is right)', () => {
  const html = render({
    model: makeResident({ model_tag: 'model-a-27b' }),
    soleResident: true,
    requestIdentity: makeIdentity({ model_tag: 'model-a-27b', label: 'Gateway' }),
  });
  assert.doesNotMatch(html, /no request yet/);
  assert.match(html, /Gateway/);
});

test('ResidentCard: THE PROP-NAME TRAP -- residentModelTag must be the card\'s OWN model_tag, not any other name. ' +
  'A wrong prop name (e.g. cardModelTag) leaves it undefined inside RequestIdentityStrip, which silently ' +
  'reads as "same model" and NEVER shows the "≠ resident" chip even when the request genuinely ran on a ' +
  'different model than this card -- exactly the silent failure a hand-written probe fixture hit. ' +
  'This is the only test in this file that actually exercises residentModelTag\'s wiring, not just whether ' +
  'the strip renders at all.', () => {
  const html = render({
    model: makeResident({ model_tag: 'card-model-A' }),
    soleResident: true, // strip renders regardless of the outer ResidentCard-level tag gate
    requestIdentity: makeIdentity({ model_tag: 'different-model-B' }), // genuinely a different model
  });
  assert.match(
    html,
    /≠ resident/,
    'requestIdentity.model_tag ("different-model-B") differs from the card\'s own model_tag ' +
      '("card-model-A") -- the chip MUST show. If residentModelTag were wired under a wrong prop name, ' +
      'it would read undefined inside RequestIdentityStrip and this chip would silently never appear.',
  );
});

/* ------------------------------------------------------------------ */
/*  Per-resident identity, backward-compat fallback                     */
/* ------------------------------------------------------------------ */

test('ResidentCard: request_identity ABSENT (undefined) -- falls back to the legacy global-prop gate, ' +
  'and the gate still SHOWS when soleResident+global identity are present -- old behaviour, case 1 of 2', () => {
  const html = render({
    model: makeResident({ model_tag: 'model-a-27b', request_identity: undefined }),
    soleResident: true,
    requestIdentity: makeIdentity({ model_tag: 'model-a-27b', label: 'FallbackShows' }),
  });
  assert.match(html, /FallbackShows/, 'old behaviour: soleResident + a global identity must still show it');
});

test('ResidentCard: request_identity ABSENT (undefined) -- the gate still HIDES when neither soleResident ' +
  'nor a tag match holds -- old behaviour, case 2 of 2, proves the fallback did not just default to always-on', () => {
  const html = render({
    model: makeResident({ model_tag: 'card-model-A', request_identity: undefined }),
    soleResident: false,
    requestIdentity: makeIdentity({ model_tag: 'unrelated-model-B', label: 'ShouldNotShow' }),
  });
  assert.doesNotMatch(
    html,
    /ShouldNotShow/,
    'neither soleResident nor a tag match holds -- the legacy gate must still hide the strip, exactly as today',
  );
});

test('ResidentCard: request_identity explicitly null -- shows "no request yet" UNCONDITIONALLY, ' +
  'even when the old gate (soleResident=false, no tag match) would have hidden it', () => {
  const html = render({
    model: makeResident({ model_tag: 'card-model-A', request_identity: null }),
    soleResident: false,
    requestIdentity: makeIdentity({ model_tag: 'unrelated-model-B' }), // global identity irrelevant once the field is present
  });
  assert.match(html, /no request yet/, 'a shipped-but-empty per-resident field must render the placeholder, not hide the strip');
});

test('ResidentCard: request_identity PRESENT -- shows THIS resident\'s own identity unconditionally, ' +
  'even when the old gate (soleResident=false, no tag match against the GLOBAL) would have hidden it', () => {
  const html = render({
    model: makeResident({
      model_tag: 'card-model-A',
      request_identity: makeIdentity({ model_tag: 'card-model-A', label: 'OwnIdentity' }),
    }),
    soleResident: false,
    requestIdentity: makeIdentity({ model_tag: 'totally-different-global', label: 'GlobalShouldNotAppear' }),
  });
  assert.match(html, /OwnIdentity/, 'the per-resident identity must show once present, regardless of the old gate');
  assert.doesNotMatch(html, /GlobalShouldNotAppear/, 'the stale global identity must never leak through once the per-resident field is present');
});

test('ResidentCard: request_identity PRESENT -- residentModelTag is still passed, so the "≠ resident" chip ' +
  'stays live as a diagnostic even for the new per-resident path (not just the legacy fallback)', () => {
  const html = render({
    model: makeResident({
      model_tag: 'card-model-A',
      request_identity: makeIdentity({ model_tag: 'somehow-mismatched-model' }),
    }),
    soleResident: false,
  });
  assert.match(html, /≠ resident/, 'residentModelTag must still reach RequestIdentityStrip on the per-resident path too');
});

/* ------------------------------------------------------------------ */
/*  tokRateTone colour + grey last-known value                          */
/* ------------------------------------------------------------------ */

test('resolveTokSDisplay PURE: current tok_s present -> uses it, not a fallback', () => {
  const result = resolveTokSDisplay(42.5, 10);
  assert.deepEqual(result, { value: 42.5, isLastKnown: false });
});

test('resolveTokSDisplay PURE: current null, a last-known value exists -> uses the fallback, flagged', () => {
  const result = resolveTokSDisplay(null, 17.3);
  assert.deepEqual(result, { value: 17.3, isLastKnown: true });
});

test('resolveTokSDisplay PURE: current null AND no last-known value -> genuinely nothing to show', () => {
  const result = resolveTokSDisplay(null, null);
  assert.deepEqual(result, { value: null, isLastKnown: false });
});

test('resolveTokSDisplay PURE: current tok_s of exactly 0 is a REAL live reading, not treated as absent -- distinguishing fixture', () => {
  const result = resolveTokSDisplay(0, 99);
  assert.deepEqual(result, { value: 0, isLastKnown: false }, '0 must not be confused with null/undefined and fall through to the stale fallback');
});

test('ResidentCard: tokRateTone drives the number\'s colour class -- live/idle/stalled each render a distinct tone', () => {
  const htmlLive = render({ tokRateTone: () => 'live' });
  const htmlIdle = render({ tokRateTone: () => 'idle' });
  const htmlStalled = render({ tokRateTone: () => 'stalled' });
  assert.match(htmlLive, /resident-tok-s"[^>]*class="[^"]*text-emerald-300/);
  assert.match(htmlIdle, /resident-tok-s"[^>]*class="[^"]*text-slate-500/);
  assert.match(htmlStalled, /resident-tok-s"[^>]*class="[^"]*text-red-400/);
});

test('ResidentCard: data-value stays the RAW current tok_s (or 0), independent of tokRateTone -- ' +
  'the tone changes colour, never the number the aggregate-sum calculation reads', () => {
  const html = render({
    model: makeResident({ generation: makeGeneration({ tok_s: 7.2 }) }),
    tokRateTone: () => 'stalled', // even a stalled tone must not alter data-value
  });
  assert.equal(dataValue(html, 'resident-tok-s'), '7.2');
});

test('ResidentCard: generation:null -- data-value is exactly "0" (matching combinedTokS\'s own treatment of an excluded resident), ' +
  'not the (SSR-unreachable) last-known fallback value', () => {
  const html = render({ model: makeResident({ generation: null }) });
  assert.equal(dataValue(html, 'resident-tok-s'), '0');
  assert.match(html, />—</, 'no current reading and no last-known value under a fresh render -> the dash placeholder, not a fabricated number');
});

/* ------------------------------------------------------------------ */
/*  Expand/collapse the live-output box (a user-requested layout        */
/*  option). isNearBottom PURE tests are what actually prove the        */
/*  scroll-maths risk a height change creates -- the real               */
/*  useLayoutEffect-driven auto-scroll is not reachable under this      */
/*  repo's renderToStaticMarkup harness (same limitation useBusyTimers/ */
/*  useLiveStream's connection effect/useLastKnownTokS already have).   */
/* ------------------------------------------------------------------ */

test('isNearBottom PURE: at the very bottom (distance 0) -> true', () => {
  assert.equal(isNearBottom(300, 260, 40), true);
});

test('isNearBottom PURE: relying on the DEFAULT threshold specifically -- distance 20 (strictly between ' +
  '4 and 40) is near under the real default (20 < 40) -- the test that actually distinguishes the ' +
  'default value, unlike the fixtures above/below which either land far outside any plausible threshold ' +
  'or pass thresholdPx explicitly and so can never catch a change to the default itself', () => {
  assert.equal(isNearBottom(300, 260, 20), true); // 300-260-20 = 20, no 4th arg -- uses the default
});

test('isNearBottom PURE: far from the bottom -> false', () => {
  assert.equal(isNearBottom(1000, 0, 128), false);
});

test('isNearBottom PURE: exact boundary (distance == threshold) -> false, the check is strictly-less-than', () => {
  assert.equal(isNearBottom(200, 60, 100, 40), false); // 200-60-100 = 40
});

test('isNearBottom PURE: one under the boundary (distance == threshold-1) -> true', () => {
  assert.equal(isNearBottom(200, 61, 100, 40), true); // 200-61-100 = 39
});

test('isNearBottom HEIGHT: the SAME scrollTop/scrollHeight pair gives a DIFFERENT, correct verdict at two ' +
  'different clientHeight values -- this is what proves the boundary math adapts to collapsed (h-32) vs ' +
  'expanded (h-64), not an assumption baked in for one fixed size', () => {
  const scrollHeight = 300;
  const scrollTop = 130;
  const collapsedClientHeight = 128; // ~h-32
  const expandedClientHeight = 256; // ~h-64
  assert.equal(
    isNearBottom(scrollHeight, scrollTop, collapsedClientHeight),
    false,
    'at the collapsed size this position is 42px from the bottom -- not near',
  );
  assert.equal(
    isNearBottom(scrollHeight, scrollTop, expandedClientHeight),
    true,
    'the identical scroll position reads as at/past the bottom once the box is taller -- the formula must pick this up, not stay pinned to the collapsed answer',
  );
});

test('LiveOutputBox: defaultExpanded false (the default) -> h-32 height class, collapsed toggle state', () => {
  const html = renderToStaticMarkup(<LiveOutputBox pane={makePane({ text: 'hello' })} />);
  assert.match(html, /h-32 overflow-y-auto/);
  assert.doesNotMatch(html, /h-64 overflow-y-auto/);
  assert.match(html, /data-testid="resident-live-output-toggle"[^>]*data-value="collapsed"/);
  assert.match(html, /aria-label="Expand live output"/);
  // The toggle must show the WORD "Expand", not the old glyph -- this is a product requirement, not a detail.
  assert.match(html, /<button[^>]*data-testid="resident-live-output-toggle"[^>]*>Expand<\/button>/);
});

test('LiveOutputBox: defaultExpanded true -> h-64 height class, expanded toggle state, aria-label flips', () => {
  const html = renderToStaticMarkup(<LiveOutputBox pane={makePane({ text: 'hello' })} defaultExpanded={true} />);
  assert.match(html, /h-64 overflow-y-auto/);
  assert.doesNotMatch(html, /h-32 overflow-y-auto/);
  assert.match(html, /data-testid="resident-live-output-toggle"[^>]*data-value="expanded"/);
  assert.match(html, /aria-label="Retract live output"/);
  assert.match(html, /<button[^>]*data-testid="resident-live-output-toggle"[^>]*>Retract<\/button>/);
});

test(
  'LiveOutputBox: the toggle ALWAYS carries a solid emerald background -- the plain grey toggle ' +
    'was invisible next to the LIVE badge, so the green box IS the ' +
    'requirement, not a strippable styling detail. Pinned HERE with the button (by design) so this ' +
    'test proves the requirement and does not depend on a later change. ' +
    'Pins the INTENT (a STANDALONE bg-emerald-N token) not a brittle exact class string.',
  () => {
    for (const expanded of [false, true]) {
      const html = renderToStaticMarkup(
        <LiveOutputBox pane={makePane({ text: 'hello' })} defaultExpanded={expanded} />,
      );
      const buttonTag = html.match(/<button[^>]*data-testid="resident-live-output-toggle"[^>]*>/);
      assert.ok(buttonTag, `toggle button not found (expanded=${expanded})`);
      const classAttr = buttonTag![0].match(/class="([^"]*)"/);
      assert.ok(classAttr, `toggle button has no class attribute (expanded=${expanded})`);
      // STANDALONE token, deliberately. A plain /bg-emerald-\d+/ also matches the
      // substring inside "hover:bg-emerald-600", so it would ACCEPT a button that is
      // grey until you point at it -- the exact failure this pins against.
      assert.match(
        classAttr![1],
        /(^|\s)bg-emerald-\d+(\s|$)/,
        `must carry a standalone bg-emerald-N background in its DEFAULT state (expanded=${expanded}) -- ` +
          'hover-only green does not satisfy it: the box must read green before the pointer ever moves',
      );
      assert.match(
        classAttr![1],
        /hover:bg-emerald-\d+/,
        `must keep a hover response (expanded=${expanded}) -- that is what distinguishes this ACTION ` +
          'from the inert LIVE status badge now that both are green',
      );
    }
  },
);

test('LiveOutputBox: expanded state does not disturb any EXISTING behaviour -- badge, tok/s, last-frame age all still render', () => {
  const html = renderToStaticMarkup(
    <LiveOutputBox
      pane={makePane({ text: 'still generating', done: false, lastTokS: 12.3, lastFrameAt: Date.now() })}
      defaultExpanded={true}
    />,
  );
  assert.match(html, /data-testid="resident-live-output-status"[^>]*data-value="live"/);
  assert.match(html, /12\.3 tok\/s/);
  assert.match(html, /still generating/);
  assert.match(html, /s ago/);
});

test('LiveOutputBox: toggle button is always present regardless of state -- always-rendered per the ' +
  'same "ambiguous absence" principle as resident-tok-s/resident-alarm', () => {
  const collapsed = renderToStaticMarkup(<LiveOutputBox pane={makePane()} />);
  const expanded = renderToStaticMarkup(<LiveOutputBox pane={makePane()} defaultExpanded={true} />);
  assert.match(collapsed, /data-testid="resident-live-output-toggle"/);
  assert.match(expanded, /data-testid="resident-live-output-toggle"/);
});

/* ------------------------------------------------------------------ */
/*  tok/s row sparkline stretches -- flex-1/min-w-0, not a              */
/*  fixed w-24. Checked at BOTH residency counts the system supports    */
/*  (1 resident, 2 residents): flex-1                                   */
/*  without min-w-0 is the exact bug that reads fine at one width and   */
/*  clips/overflows at another (min-w-0 overrides a flex child's        */
/*  default min-width:auto, which otherwise refuses to shrink below its */
/*  intrinsic content size). renderToStaticMarkup has no CSS layout      */
/*  engine -- it cannot measure real overflow -- so this proves the      */
/*  STRUCTURAL fix is present and count-independent, not literal pixels; */
/*  a manual browser check is what confirms the pixels. ResidentsPanel's */
/*  own grid is grid-cols-1 (cards always stack full-width, one per     */
/*  row), so "1-up/2-up" here means resident COUNT in one render pass,  */
/*  not a responsive column breakpoint.                                 */
/* ------------------------------------------------------------------ */

function sparklineClass(html: string): string[] {
  // The sparkline is the only element in this component carrying
  // aria-hidden="true" (SSR never accumulates samples -- no effects run --
  // so it always renders the placeholder branch here, one per card).
  const re = /class="([^"]*)"\s+aria-hidden="true"/g;
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) out.push(m[1]);
  return out;
}

test('ResidentCard: 1 resident -- sparkline is flex-1 min-w-0, never the old fixed w-24', () => {
  const html = render();
  const classes = sparklineClass(html);
  assert.equal(classes.length, 1, 'exactly one card, one sparkline');
  assert.match(classes[0], /\bflex-1\b/);
  assert.match(classes[0], /\bmin-w-0\b/);
  assert.doesNotMatch(classes[0], /\bw-24\b/);
});

test('ResidentCard: 2 residents -- EACH card independently gets flex-1 min-w-0, not just the first', () => {
  const htmlA = render({ model: makeResident({ model_tag: 'a' }) });
  const htmlB = render({ model: makeResident({ model_tag: 'b' }) });
  const combined = htmlA + htmlB; // two independent card renders, same as ResidentsPanel's .map()
  const classes = sparklineClass(combined);
  assert.equal(classes.length, 2, 'two cards rendered, two sparklines');
  for (const cls of classes) {
    assert.match(cls, /\bflex-1\b/, 'the fix must apply to every card, not be index-dependent');
    assert.match(cls, /\bmin-w-0\b/);
    assert.doesNotMatch(cls, /\bw-24\b/);
  }
});

/*  kvRestoreTone -- the KV-restore 4-branch state machine.             */
/*  Reads restore_attempted AND kv_restore_ok, never final_status alone */
/*  (except in the fallback branch, where neither field is trustworthy).*/
/* ------------------------------------------------------------------ */

test('kvRestoreTone: restore_attempted === false -> no-data, "nothing to restore", NOT a verdict', () => {
  const tone = kvRestoreTone(makeLoadVerifyRecord({ restore_attempted: false, kv_restore_ok: null }));
  assert.equal(tone.state, 'no-data');
  assert.equal(tone.label, 'NOTHING TO RESTORE');
  assert.match(tone.dot, /slate/);
});

test('kvRestoreTone: restore_attempted === true, kv_restore_ok === true -> ok, GREEN -- the explicit requirement', () => {
  const tone = kvRestoreTone(makeLoadVerifyRecord({ restore_attempted: true, kv_restore_ok: true }));
  assert.equal(tone.state, 'ok');
  assert.equal(tone.label, 'OK');
  assert.match(tone.dot, /emerald/);
});

test('kvRestoreTone: restore_attempted === true, kv_restore_ok === false -> failed, a genuine miss, must stay visible', () => {
  const tone = kvRestoreTone(makeLoadVerifyRecord({ restore_attempted: true, kv_restore_ok: false }));
  assert.equal(tone.state, 'failed');
  assert.equal(tone.label, 'FAILED');
  assert.match(tone.dot, /rose/);
});

test(
  'kvRestoreTone: THE BRANCH THAT IS EASY TO LOSE -- restore_attempted === true, kv_restore_ok === null ' +
    '-> pending/VERIFYING. Attempted-but-unmeasured is neither "nothing to restore" nor a failure; it ' +
    'must be distinct from BOTH neighbors, not collapsed into either.',
  () => {
    const tone = kvRestoreTone(makeLoadVerifyRecord({ restore_attempted: true, kv_restore_ok: null }));
    assert.equal(tone.state, 'pending');
    assert.equal(tone.label, 'VERIFYING');
    assert.match(tone.dot, /amber/);
    assert.notEqual(tone.label, 'NOTHING TO RESTORE');
    assert.notEqual(tone.label, 'FAILED');
  },
);

test(
  'kvRestoreTone: restore_attempted undefined (older record predating the field) -> falls back to ' +
    'loadVerifyTone(final_status) rather than inventing a claim -- a record that cannot answer the ' +
    'question must not be given an answer',
  () => {
    const rec = makeLoadVerifyRecord({ final_status: 'ok' });
    delete (rec as { restore_attempted?: boolean | null }).restore_attempted;
    const tone = kvRestoreTone(rec);
    assert.equal(tone.label, 'OK');
    assert.equal(tone.state, 'ok');
  },
);

test('kvRestoreTone: restore_attempted explicitly null (not just undefined) -> same fallback path', () => {
  const tone = kvRestoreTone(makeLoadVerifyRecord({ restore_attempted: null, final_status: 'failed' }));
  assert.equal(tone.label, 'FAILED');
  assert.equal(tone.state, 'failed');
});

test(
  'ResidentCard + LoadVerifyWidget: a kv_restore record with restore_attempted:false renders ' +
    'load-verify-kv-status data-value="no-data", NOT the word "UNVERIFIED" -- proves the fix is actually ' +
    'wired in, not just correct in isolation',
  () => {
    const html = render({
      loadVerify: [makeLoadVerifyRecord({ event: 'kv_restore', restore_attempted: false, kv_restore_ok: null })],
    });
    assert.equal(dataValue(html, 'load-verify-kv-status'), 'no-data');
    assert.match(html, /NOTHING TO RESTORE/);
    assert.doesNotMatch(html, />UNVERIFIED</);
  },
);

test(
  'ResidentCard + LoadVerifyWidget: a model_load record with final_status "unverified" renders ' +
    'BYTE-IDENTICALLY to before this change -- loadVerifyTone(), not kvRestoreTone(), and carries NO ' +
    'load-verify-kv-status testid (that hook is KV-row-only)',
  () => {
    const html = render({
      loadVerify: [
        makeLoadVerifyRecord({ event: 'model_load', final_status: 'unverified', restore_attempted: undefined }),
      ],
    });
    assert.match(html, /model load/);
    assert.match(html, />UNVERIFIED</);
    assert.doesNotMatch(html, /load-verify-kv-status/);
  },
);

test(
  'ResidentCard: CARD_SPARK_SAMPLES is PINNED at 30 -- a mutation (30->60, ' +
    'silently doubling the card graph\'s lookback window) survived the entire suite once already. ' +
    'This is the WINDOW the graph describes, not a resolution knob -- the sparkline-width change stretched the pixel ' +
    'WIDTH deliberately without touching this number; changing this number changes what the graph ' +
    'CLAIMS ("how far back"), which is exactly the silent-meaning-change the open question refused.',
  () => {
    assert.equal(CARD_SPARK_SAMPLES, 30);
  },
);


/* ------------------------------------------------------------------ */
/*  Context-size readout (used / capacity), PER RESIDENT                 */
/* ------------------------------------------------------------------ */

// Lesson from the aggregate bar tests, reapplied: a hardcoded/wrong width can survive
// every data-value/text check while the one thing a viewer actually looks
// at (the bar) lies. Same shape as ThroughputSection.test.tsx's
// extractWidthPct.
function extractWidthPct(html: string, testId: string): number | null {
  const re = new RegExp(`data-testid="${testId}"[^>]*?style="width:([\\d.]+)%"`);
  const m = html.match(re);
  return m ? Number(m[1]) : null;
}

test('resolveContextDisplay PURE: current n_ctx present -> uses it, not a fallback', () => {
  const result = resolveContextDisplay(250112, 214592, { used: 1, capacity: 2 });
  assert.deepEqual(result, { used: 214592, capacity: 250112, pct: (214592 / 250112) * 100, isLastKnown: false });
});

test(
  'resolveContextDisplay PURE: current n_ctx null, a last-known reading exists -> uses the fallback, ' +
    'flagged isLastKnown -- design constraint: idle is a marked LAST-KNOWN value, never a dash. This is the ' +
    'exact assertion a "blank to dash on idle" mutation (dropping this fallback branch) must break.',
  () => {
    const result = resolveContextDisplay(null, 0, { used: 214592, capacity: 250112 });
    assert.deepEqual(result, { used: 214592, capacity: 250112, pct: (214592 / 250112) * 100, isLastKnown: true });
  },
);

test('resolveContextDisplay PURE: current null AND no last-known value -> genuinely nothing to show', () => {
  const result = resolveContextDisplay(null, null, null);
  assert.equal(result, null);
});

test(
  'resolveContextDisplay PURE: current used of exactly 0 with a real n_ctx is a REAL live reading (fresh ' +
    'generation, nothing processed yet), not treated as absent -- distinguishing fixture',
  () => {
    const result = resolveContextDisplay(250112, 0, { used: 999, capacity: 999 });
    assert.deepEqual(result, { used: 0, capacity: 250112, pct: 0, isLastKnown: false }, '0 used must not fall through to the stale fallback');
  },
);

test('resolveContextDisplay PURE: capacity of 0 never divides by zero -- pct is 0, not NaN/Infinity', () => {
  const result = resolveContextDisplay(0, 0, null);
  assert.deepEqual(result, { used: 0, capacity: 0, pct: 0, isLastKnown: false });
});

test(
  'ResidentCard: generation:null -- card still renders, resident-context-used/-capacity present ' +
    'with data-value=0 (same "always present" rule as resident-tok-s -- a missing element ' +
    'is ambiguous between "correctly excluded" and "a bug dropped it")',
  () => {
    const html = render({ model: makeResident({ state: 'GRACE', generation: null }) });
    assert.equal(dataValue(html, 'resident-context-used'), '0');
    assert.equal(dataValue(html, 'resident-context-capacity'), '0');
  },
);

test('ResidentCard: live generation -- resident-context-used/-capacity data-value equal the real n_prompt_tokens/n_ctx', () => {
  const html = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) }) });
  assert.equal(dataValue(html, 'resident-context-used'), '214592');
  assert.equal(dataValue(html, 'resident-context-capacity'), '250112');
  assert.match(html, />214,592</, 'the formatted number (with thousands separators) renders too, not just the raw data-value');
  assert.match(html, /· 86%</, 'pct rounds to the nearest whole percent (214592/250112 = 85.8ish -> 86)');
});

test(
  'ResidentCard: DISTINGUISHING -- two residents with different context render different ' +
    'data-values, not a shared/stale number',
  () => {
    const htmlA = render({ model: makeResident({ model_tag: 'a', generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 1000 }) }) });
    const htmlB = render({ model: makeResident({ model_tag: 'b', generation: makeGeneration({ n_ctx: 200000, n_prompt_tokens: 2000 }) }) });
    assert.equal(dataValue(htmlA, 'resident-context-used'), '1000');
    assert.equal(dataValue(htmlB, 'resident-context-used'), '2000');
    assert.equal(dataValue(htmlA, 'resident-context-capacity'), '100000');
    assert.equal(dataValue(htmlB, 'resident-context-capacity'), '200000');
  },
);

test('ResidentCard: context fullness bar rendered WIDTH matches the real pct, not a hardcoded value (lesson from the aggregate bar tests reapplied)', () => {
  const htmlLow = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 10000 }) }) });
  const htmlHigh = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 95000 }) }) });
  assert.equal(extractWidthPct(htmlLow, 'resident-context-bar'), 10);
  assert.equal(extractWidthPct(htmlHigh, 'resident-context-bar'), 95);
});

test(
  'ResidentCard: context fullness bar tone follows the SAME thresholds as contextTone -- green/amber/red bar fill class',
  () => {
    const green = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 50000 }) }) }); // 50%
    const amber = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 75000 }) }) }); // 75%
    const red = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 95000 }) }) }); // 95%
    assert.match(green, /resident-context-bar"[^>]*class="[^"]*bg-emerald-500/);
    assert.match(amber, /resident-context-bar"[^>]*class="[^"]*bg-amber-500/);
    assert.match(red, /resident-context-bar"[^>]*class="[^"]*bg-red-500/);
  },
);

test(
  'ResidentCard regression guard (the aggregate does not take a worst-of, so the anti-dilution ' +
    'property this guards lives HERE: a near-full resident, INCLUDING AN IDLE ONE SHOWING LAST-KNOWN, ' +
    'must render RED on its OWN card). THE "clean up null handling" REFACTOR TRAP: a permanent guard, not a run-once ' +
    'mutation control: a future edit that special-cases contextDisplay.isLastKnown out of the tone/colour ' +
    'path (e.g. "dim it, it\'s stale" or "skip colouring an idle reading") would regress this silently -- ' +
    'every OTHER test in this file would stay green.\n\n' +
    'Proven as TWO composed pieces, with the same SSR-boundary limitation noted in ' +
    'ResidentCard.tsx: first, PURE, below -- resolveContextDisplay\'s last-known fallback correctly resolves ' +
    'and classifies a remembered near-full reading as red, proven directly since a live render can never ' +
    'populate useLastKnownContext\'s ref under this repo\'s SSR harness; second, the adjacent ' +
    '"context fullness bar tone" test two tests up proves a red-classified pct actually PAINTS the bar red, ' +
    'through a live (non-idle) render. The two compose into the full property because ContextFullnessBar\'s ' +
    'signature is `{ pct: number | null }` alone -- it has no isLastKnown parameter to special-case without ' +
    'a visible signature change, so whatever pct resolveContextDisplay hands it (idle-fallback or live, ' +
    'indistinguishable to ContextFullnessBar) is coloured by the exact same, single, non-branching path.',
  () => {
    const nearFullRemembered = resolveContextDisplay(null, null, { used: 95000, capacity: 100000 });
    assert.deepEqual(
      nearFullRemembered,
      { used: 95000, capacity: 100000, pct: 95, isLastKnown: true },
      'the idle resident\'s remembered 95% reading must still resolve, flagged isLastKnown',
    );
    assert.equal(contextTone(nearFullRemembered!.pct), 'red', 'a remembered near-full reading must classify red, not be diluted or defaulted to green because it is stale');

    // A second discriminating fixture at 86% (an amber-band example) -- proves this
    // isn't just a boundary artifact of the 95% case: idle-remembered readings classify by the SAME
    // thresholds as live ones across the whole scale, not just at the extreme.
    const idleAmberRemembered = resolveContextDisplay(null, null, { used: 86000, capacity: 100000 });
    assert.equal(idleAmberRemembered!.pct, 86);
    assert.equal(contextTone(idleAmberRemembered!.pct), 'amber', 'must not be diluted to green -- 86 sits in the amber band under the 75/90 thresholds');
  },
);

test(
  'ResidentCard: used number stays FULL WEIGHT (never dimmed) even though tok/s one row up ' +
    'DOES dim on idle -- design constraint: a stale-but-true context reading is current occupancy, not a ' +
    'faded fact, so only a small marker (not the number\'s color) may say "stale". Capacity is the one ' +
    'exception, and deliberately so (see the capacity-dimming test below) -- it is a reference ceiling, not the ' +
    'moving number this rule is about.',
  () => {
    const html = render({ model: makeResident({ generation: makeGeneration({ state: 'idle', n_ctx: 100000, n_prompt_tokens: 50000 }) }) });
    assert.match(
      html,
      /resident-tok-s"[^>]*class="[^"]*text-slate-500/,
      'sanity check: tok/s really does dim on this idle-state fixture, confirming the contrast is real',
    );
    assert.match(html, /resident-context-used"[^>]*class="[^"]*text-slate-200/);
    assert.doesNotMatch(html, /resident-context-used"[^>]*class="[^"]*text-slate-500/);
  },
);

test(
  'ResidentCard: capacity number is dimmed to the exact " / " separator idiom ' +
    '(text-slate-500), not the loud text-slate-200 the used number keeps -- this is a DELIBERATE reversal ' +
    'of an assertion that was pinned the opposite way before this change (a doesNotMatch on this exact ' +
    'class), rewritten rather than deleted because the later design change is what makes the old pin wrong.',
  () => {
    const html = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 100000, n_prompt_tokens: 50000 }) }) });
    assert.match(html, /resident-context-capacity"[^>]*class="[^"]*text-slate-500/);
    assert.match(html, /resident-context-capacity"[^>]*class="[^"]*text-xl font-bold tabular-nums/, 'size/weight/tabular-nums unchanged -- only colour moved');
  },
);

test(
  'ResidentCard: per-turn progress bar (CompactProgress) now sits ABOVE the context ' +
    'block, reversed from the earlier ordering -- a deliberate bar-order decision -- and the ' +
    'context block moved as one unit below it. Document-order chain only (see the caveat on the ' +
    'ThroughputSection.test.tsx equivalent: this proves ORDER, not true DOM siblinghood or visual stacking).',
  () => {
    const html = render({
      model: makeResident({ generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592, state: 'generating' }) }),
    });
    const tokIdx = html.indexOf('data-testid="resident-tok-s"');
    const turnBarIdx = html.indexOf('h-1.5 bg-slate-800 rounded overflow-hidden', tokIdx);
    const ctxLabelIdx = html.indexOf('>context<', turnBarIdx);
    const ctxUsedIdx = html.indexOf('data-testid="resident-context-used"', ctxLabelIdx);
    const ctxBarIdx = html.indexOf('data-testid="resident-context-bar"', ctxUsedIdx);
    assert.ok(
      tokIdx !== -1 && turnBarIdx !== -1 && tokIdx < turnBarIdx && turnBarIdx < ctxLabelIdx &&
        ctxLabelIdx < ctxUsedIdx && ctxUsedIdx < ctxBarIdx,
      'tok/s row, then the per-turn progress bar, then the context label, then the context numbers, ' +
        'then the context fullness bar, in that document order',
    );
  },
);

test(
  'ResidentCard: the context block\'s own wrapper DROPPED the flex-row classes it shared ' +
    'with its bar before this round -- a real class-attribute fact (not an order-only inference): only the ' +
    'tok/s row keeps the "flex items-center gap-3" shape now, since the context bar moved to its own ' +
    'full-width line underneath the numbers instead of squeezing in beside them.',
  () => {
    const html = render({ model: makeResident({ generation: makeGeneration({ n_ctx: 250112, n_prompt_tokens: 214592 }) }) });
    const flexRowClass = 'mt-2 pt-2 border-t border-slate-800 flex items-center gap-3';
    const occurrences = html.split(flexRowClass).length - 1;
    assert.equal(
      occurrences,
      1,
      'exactly one row (tok/s) still uses this flex-row shape; the context block\'s own border-t wrapper no longer does',
    );
  },
);

/* ------------------------------------------------------------------ */
/*  current_request_identity -- CURRENT, not LAST                       */
/* ------------------------------------------------------------------ */

test('ResidentCard: current_request_identity null -- renders the "current request" row with the ' +
  'idle placeholder, distinguishable from the "last request" row by heading AND by its OWN testid', () => {
  const html = render({
    model: makeResident({
      model_tag: 'card-model-A',
      request_identity: makeIdentity({ model_tag: 'card-model-A', label: 'LastOne' }),
      current_request_identity: null,
    }),
  });
  assert.match(html, /last request/, 'the LAST row must still render its heading');
  assert.match(html, /current request/, 'the CURRENT row must render its own heading');
  assert.match(html, /LastOne/, 'the last-request identity content must still show');
  assert.match(html, /idle/, 'a null current identity renders the idle placeholder, not "no request yet"');
  assert.match(
    html,
    /data-testid="resident-current-request-identity"/,
    'the current row must carry its OWN testid, distinct from resident-request-identity',
  );
});

test('ResidentCard: current_request_identity PRESENT -- shows the live identity, distinct from ' +
  'a (possibly different) last request, proving current is not just mirroring last', () => {
  const html = render({
    model: makeResident({
      model_tag: 'card-model-A',
      request_identity: makeIdentity({ model_tag: 'card-model-A', label: 'FinishedRequest', session_id: 'sess-old' }),
      current_request_identity: makeIdentity({ model_tag: 'card-model-A', label: 'LiveRequest', session_id: 'sess-now' }),
    }),
  });
  assert.match(html, /FinishedRequest/, 'the last row keeps showing the OLD request');
  assert.match(html, /LiveRequest/, 'the current row shows the NEW, actually-live request');
  assert.match(html, /sess-old/);
  assert.match(html, /sess-now/);
});

test('ResidentCard: current_request_identity ABSENT (undefined, an old-backend response) -- the ' +
  'current row still renders (idle placeholder), never crashes, never silently disappears', () => {
  const html = render({
    model: makeResident({ model_tag: 'card-model-A', current_request_identity: undefined }),
  });
  assert.match(
    html,
    /data-testid="resident-current-request-identity"/,
    'the current row hook must be present even when the backend has not shipped the field yet',
  );
});

test('ResidentCard NO-FLIP (machine-readable): two residents each carry BOTH resident-request-identity ' +
  'AND resident-current-request-identity hooks -- exactly one of each, never a shared/collapsed count', () => {
  const htmlA = render({
    model: makeResident({
      model_tag: 'card-model-A',
      request_identity: makeIdentity({ model_tag: 'card-model-A' }),
      current_request_identity: makeIdentity({ model_tag: 'card-model-A' }),
    }),
  });
  const lastHooks = [...htmlA.matchAll(/data-testid="resident-request-identity"/g)];
  const currentHooks = [...htmlA.matchAll(/data-testid="resident-current-request-identity"/g)];
  assert.equal(lastHooks.length, 1, 'exactly one "last" hook per card');
  assert.equal(currentHooks.length, 1, 'exactly one "current" hook per card -- not merged with, not duplicating, the last hook');
});
