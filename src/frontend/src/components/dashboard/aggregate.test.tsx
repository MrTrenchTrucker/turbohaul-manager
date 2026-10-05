// Aggregate math as pure functions, tested independently of JSX.
// The prefill-boundary decision and the injection contract:
// synthesizeResident is dependency-injected, never imported or
// reimplemented here.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  advanceCombinedSparkline,
  alarmCounts,
  combinedTokS,
  INITIAL_SPARKLINE_STATE,
  outputFraction,
  prefillMean,
  selectResidents,
  totalAlarmCount,
} from './aggregate';
import type { AlarmToken, ResidentAlarm } from './aggregate';
import type { GenerationInfo, ResidentModel, StatusSnapshot } from '../../api';

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
    model_tag: 'model-b-27b',
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

function makeStatus(overrides: Partial<StatusSnapshot> = {}): StatusSnapshot {
  return {
    queue: { acceptance_buffer_depth: 0, staging_queue_depth: 0, staging_queue_max: 8 },
    active: null,
    loading: null,
    grace: null,
    idle_hot: null,
    parallel_slots: { used: 0, max: 4 },
    residents: [],
    vram: null,
    vram_total_mib: null,
    generation: null,
    ...overrides,
  };
}

/* ------------------------------------------------------------------ */
/*  selectResidents -- THE ONE RULE                                    */
/* ------------------------------------------------------------------ */

test('selectResidents: residents[] non-empty returns it as-is and never calls the legacy fallback', () => {
  const r = makeResident({ model_tag: 'the-real-one' });
  const status = makeStatus({ residents: [r] });
  let calls = 0;
  const result = selectResidents(status, () => {
    calls += 1;
    return makeResident({ model_tag: 'should-not-be-used' });
  });
  assert.deepEqual(result, [r]);
  assert.equal(calls, 0, 'synthesizeResident must not be invoked when residents[] is non-empty');
});

test('selectResidents: residents[] EMPTY falls back to legacy synthesizeResident(status) -- cap<=1 must not blank', () => {
  const status = makeStatus({ residents: [] });
  const legacy = makeResident({ model_tag: 'legacy-single-residency' });
  let calls = 0;
  const result = selectResidents(status, (s) => {
    calls += 1;
    assert.equal(s, status, 'synthesizeResident must receive the same status object');
    return legacy;
  });
  assert.deepEqual(result, [legacy]);
  assert.equal(calls, 1);
});

test('selectResidents: residents[] EMPTY and legacy synthesizeResident returns null -> [] (not [null])', () => {
  const status = makeStatus({ residents: [] });
  const result = selectResidents(status, () => null);
  assert.deepEqual(result, []);
});

test('selectResidents + combinedTokS end-to-end: cap<=1 (legacy) fallback still produces a live number, the regression maintainers fear most', () => {
  const status = makeStatus({
    residents: [],
    active: { slot_id: 's0', model_tag: 'model-b-27b', state: 'ACTIVE', thread_id_prefix: 'th-', pid: 1, port: 11500 },
    generation: makeGeneration({ tok_s: 42.5 }),
  });
  const legacy = makeResident({ generation: makeGeneration({ tok_s: 42.5 }) });
  const residents = selectResidents(status, () => legacy);
  assert.equal(residents.length, 1);
  assert.equal(combinedTokS(residents), 42.5, 'a cap<=1 dashboard must not read 0/blank when a real generation is live');
});

/* ------------------------------------------------------------------ */
/*  combinedTokS                                                       */
/* ------------------------------------------------------------------ */

test('combinedTokS: empty array -> 0', () => {
  assert.equal(combinedTokS([]), 0);
});

test('combinedTokS: single resident returns its tok_s', () => {
  const r = makeResident({ generation: makeGeneration({ tok_s: 12.3 }) });
  assert.equal(combinedTokS([r]), 12.3);
});

test('combinedTokS: N>=2 residents SUMS, does not average -- distinguishing fixture', () => {
  const a = makeResident({ generation: makeGeneration({ tok_s: 10 }) });
  const b = makeResident({ generation: makeGeneration({ tok_s: 30 }) });
  // sum = 40, mean = 20 -- a wrong impl that averages would return 20 here.
  assert.equal(combinedTokS([a, b]), 40);
});

test('combinedTokS: a generation:null resident (card-only, by design) contributes 0, is excluded', () => {
  const live = makeResident({ generation: makeGeneration({ tok_s: 15 }) });
  const cardOnly = makeResident({ state: 'GRACE', generation: null });
  assert.equal(combinedTokS([live, cardOnly]), 15);
});

test('combinedTokS: tok_s undefined on a live-generation resident guards to 0, never NaN', () => {
  const r = makeResident({ generation: makeGeneration({ tok_s: undefined }) });
  const result = combinedTokS([r]);
  assert.equal(result, 0);
  assert.equal(Number.isNaN(result), false);
});

/* ------------------------------------------------------------------ */
/*  outputFraction                                                     */
/* ------------------------------------------------------------------ */

test('outputFraction: empty array -> 0', () => {
  assert.equal(outputFraction([]), 0);
});

test('outputFraction: max_tokens 0 is excluded (no divide by zero), lone resident -> 0', () => {
  const r = makeResident({ generation: makeGeneration({ n_decoded: 5, max_tokens: 0 }) });
  const result = outputFraction([r]);
  assert.equal(result, 0);
  assert.equal(Number.isFinite(result), true);
});

test('outputFraction: max_tokens absent is excluded (no divide by zero), lone resident -> 0', () => {
  const r = makeResident({ generation: makeGeneration({ n_decoded: 5, max_tokens: undefined }) });
  const result = outputFraction([r]);
  assert.equal(result, 0);
  assert.equal(Number.isFinite(result), true);
});

test('outputFraction: sums n_decoded and max_tokens across residents with known capacity', () => {
  const a = makeResident({ generation: makeGeneration({ n_decoded: 20, max_tokens: 100 }) });
  const b = makeResident({ generation: makeGeneration({ n_decoded: 30, max_tokens: 100 }) });
  assert.equal(outputFraction([a, b]), 0.25); // (20+30)/(100+100)
});

test('outputFraction: a resident with n_decoded but NO known max_tokens must not inflate the fraction -- distinguishing fixture', () => {
  // Orphan resident: 200 decoded tokens but capacity unknown (max_tokens null).
  // A wrong impl that sums n_decoded globally regardless of max_tokens would
  // compute (200+10)/100 = 2.1 (210%) here -- an obviously wrong out-of-range
  // fraction that is not NaN and so would slip past a NaN-only guard.
  const orphan = makeResident({ generation: makeGeneration({ n_decoded: 200, max_tokens: null as unknown as undefined }) });
  const known = makeResident({ generation: makeGeneration({ n_decoded: 10, max_tokens: 100 }) });
  const result = outputFraction([orphan, known]);
  assert.equal(result, 0.1, 'orphan resident (unknown capacity) must be excluded from both sums');
  assert.ok(result <= 1, 'fraction must never exceed 100%');
});

test('outputFraction: n_decoded undefined on a resident with valid max_tokens guards to 0 contribution', () => {
  const r = makeResident({ generation: makeGeneration({ n_decoded: undefined, max_tokens: 100 }) });
  assert.equal(outputFraction([r]), 0);
});

test('outputFraction: a generation:null resident is excluded entirely', () => {
  const cardOnly = makeResident({ state: 'IDLE_HOT', generation: null });
  const known = makeResident({ generation: makeGeneration({ n_decoded: 10, max_tokens: 100 }) });
  assert.equal(outputFraction([cardOnly, known]), 0.1);
});

test(
  'outputFraction: DISPLAY CLAMP -- a resident IN the output set whose n_decoded ' +
    'transiently exceeds its own max_tokens, mixed with a normal one, must clamp to exactly 1.0, never ' +
    'above; the lower bound is not spuriously reachable (0 residents still reads exactly 0).',
  () => {
    // Overshoot resident: legitimately in the set (valid positive max_tokens),
    // but its own n_decoded (150) exceeds its own cap (100) -- the paired
    // guard cannot exclude this one, it belongs in the set.
    const overshoot = makeResident({ generation: makeGeneration({ n_decoded: 150, max_tokens: 100 }) });
    // Normal resident: within its own cap, does not dilute enough to bring
    // the combined ratio back under 1.
    const normal = makeResident({ generation: makeGeneration({ n_decoded: 90, max_tokens: 100 }) });
    // Unclamped: (150+90)/(100+100) = 240/200 = 1.2 -- must clamp to 1.0.
    const result = outputFraction([overshoot, normal]);
    assert.equal(result, 1, 'unclamped math is 1.2; the clamp must bring it to exactly 1.0');
    assert.ok(result <= 1, 'clamped fraction must never exceed 1.0');

    // Lower bound: 0 residents must still read exactly 0, not be "clamped"
    // into looking like a valid in-range value.
    assert.equal(outputFraction([]), 0);
  },
);

/* ------------------------------------------------------------------ */
/*  prefillMean                                                        */
/* ------------------------------------------------------------------ */

test(
  'prefillMean: empty array -> null (no data, not a real zero) -- contract change. ' +
    'An empty residents[] hits the exact same prefilling.length===0 branch as "residents exist but none ' +
    'are prefilling", so it must resolve the same new way: null, not the old hard 0.',
  () => {
    assert.equal(prefillMean([]), null);
  },
);

test(
  'prefillMean: residents exist but NONE are prefilling (all decoding) -> null, not 0 -- ' +
    'the flapping case: agg-prefill-pct read 0 while a genuinely-different resident ' +
    'was mid-prefill a moment later, because the "nobody prefilling right now" instant is common with ' +
    '2+ residents cycling independently, and a hard 0 there lies as "0% complete" instead of "no data".',
  () => {
    const decoding = makeResident({ generation: makeGeneration({ state: 'generating', tok_s: 30 }) });
    assert.equal(prefillMean([decoding]), null);
  },
);

test('prefillMean: single prefilling resident returns its prefill_pct', () => {
  const r = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 63 }) });
  assert.equal(prefillMean([r]), 63);
});

test('prefillMean: N>=2 prefilling residents MEANS, does not sum -- distinguishing fixture', () => {
  const a = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 20 }) });
  const b = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 60 }) });
  // mean = 40, sum = 80 -- a wrong impl that sums would return 80 here.
  assert.equal(prefillMean([a, b]), 40);
});

test('prefillMean: a generation:null resident mixed among prefilling ones is excluded from the denominator', () => {
  const prefilling = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 50 }) });
  const cardOnly = makeResident({ state: 'LOADING', generation: null });
  assert.equal(prefillMean([prefilling, cardOnly]), 50);
});

test(
  'prefillMean: THE decisive fixture -- gates on generation.state, never on prefill_pct itself. ' +
    'A decoding resident carrying a stale prefill_pct<100 (the documented fork behavior, ' +
    'live_monitor.py / Dashboard.tsx derivePill) must NOT be pulled into the mean.',
  () => {
    // Stale: already generating (state flipped once total_inst > 0), but its
    // prefill_pct is still frozen below 100 from before the flip -- exactly
    // the fork behavior derivePill's own comment documents.
    const decodingWithStalePct = makeResident({
      generation: makeGeneration({ state: 'generating', prefill_pct: 45, n_decoded: 12 }),
    });
    const genuinelyPrefilling = makeResident({
      generation: makeGeneration({ state: 'prefill', prefill_pct: 80 }),
    });
    const result = prefillMean([decodingWithStalePct, genuinelyPrefilling]);
    // A prefill_pct-based implementation (gate on `prefill_pct != null` or
    // `prefill_pct < 100`) would include the decoding resident too and
    // return (45+80)/2 = 62.5. The correct, state-gated answer is 80.
    assert.equal(result, 80);
    assert.notEqual(result, 62.5);
  },
);

test('prefillMean: a state==="prefill" resident with prefill_pct still null (startup race) counts in the denominator, contributes 0', () => {
  const justStarted = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: null }) });
  assert.equal(prefillMean([justStarted]), 0);

  const alsoPrefilling = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 100 }) });
  // denominator = 2 (both prefilling), sum = 0 + 100 -> mean 50, not 100.
  assert.equal(prefillMean([justStarted, alsoPrefilling]), 50);
});

test('prefillMean: the prefill_pct===100-with-zero-tokens-emitted boundary (an open question at design time) is correctly included', () => {
  // state is still 'prefill' at this instant per live_monitor.py (flips to
  // 'generating' only once total_inst > 0) -- this resident is genuinely
  // ~done prefilling and must be counted, contributing ~100.
  const atTheBoundary = makeResident({ generation: makeGeneration({ state: 'prefill', prefill_pct: 100 }) });
  assert.equal(prefillMean([atTheBoundary]), 100);
});

test('prefillMean: no NaN/Infinity when every field on every resident is absent/null/undefined', () => {
  const bare = makeResident({
    generation: {
      state: 'prefill',
      prompt_progress: null,
      stalled: false,
      streaming: false,
      generation_id: null,
      measured_at_iso: '2026-08-20T00:00:00Z',
      // tok_s, n_decoded, max_tokens, prefill_pct all intentionally absent
    },
  });
  const result = prefillMean([bare]);
  assert.equal(result, 0);
  assert.equal(Number.isNaN(result), false);
  assert.equal(Number.isFinite(result), true);
});

/* ------------------------------------------------------------------ */
/*  alarmCounts / totalAlarmCount (per-resident alarm classifier      */
/*  is injected)                                                      */
/* ------------------------------------------------------------------ */

test('alarmCounts: empty array -> all-zero counts', () => {
  assert.deepEqual(alarmCounts([], () => '', {}), { busy: 0, 'prefill-hang': 0, 'no-telemetry': 0, stalled: 0 });
});

test('alarmCounts: counts by CALLING residentAlarm, never by inspecting generation fields itself -- distinguishing fixture for the injected-classifier contract', () => {
  // Both residents' OWN generation fields say "perfectly healthy" -- not
  // stalled, not null, state==='generating'. If alarmCounts re-derived
  // severity from generation fields (forbidden by the injection contract) it would count 0
  // here regardless of what the injected classifier says. The injected
  // stub below says otherwise for model 'b' -- a real verdict this
  // component cannot see the inputs for (e.g. ActiveInfo timing the card's
  // residentAlarm has access to and this file does not).
  const a = makeResident({ model_tag: 'a', generation: makeGeneration({ stalled: false, state: 'generating' }) });
  const b = makeResident({ model_tag: 'b', generation: makeGeneration({ stalled: false, state: 'generating' }) });
  const residentAlarm: ResidentAlarm = (r) => (r.model_tag === 'b' ? 'no-telemetry' : '');
  const counts = alarmCounts([a, b], residentAlarm, {});
  assert.equal(counts['no-telemetry'], 1, 'the count must come from the injected verdict, not from re-deriving it');
  assert.equal(totalAlarmCount(counts), 1);
});

test('alarmCounts: BY TOKEN, not lumped -- 1 stalled and 1 no-telemetry are different operational facts and must be distinguishable', () => {
  const residents = [
    makeResident({ model_tag: 'busy-one' }),
    makeResident({ model_tag: 'hang-one' }),
    makeResident({ model_tag: 'telemetry-one' }),
    makeResident({ model_tag: 'stalled-one' }),
    makeResident({ model_tag: 'clear-one' }),
  ];
  const tokenByTag: Record<string, AlarmToken> = {
    'busy-one': 'busy',
    'hang-one': 'prefill-hang',
    'telemetry-one': 'no-telemetry',
    'stalled-one': 'stalled',
    'clear-one': '',
  };
  const residentAlarm: ResidentAlarm = (r) => tokenByTag[r.model_tag] ?? '';
  const counts = alarmCounts(residents, residentAlarm, {});
  // A lumped single-number implementation would collapse this to "N
  // alarming" and lose which kind -- assert each bucket individually.
  // alarmCounts itself is untouched by the banner-token rule -- it still tallies all
  // four buckets, busy included; counts stays exactly what it always was.
  assert.deepEqual(counts, { busy: 1, 'prefill-hang': 1, 'no-telemetry': 1, stalled: 1 });
  // The aggregate total excludes busy: busy is a normal-operation
  // fact, not trouble, so it does not contribute to the aggregate's total
  // (BANNER_ALARM_TOKENS, aggregate.ts) even though alarmCounts above still
  // tallies it.
  assert.equal(totalAlarmCount(counts), 3, 'busy no longer counts toward the total -- 1 prefill-hang + 1 no-telemetry + 1 stalled = 3, not 4');
});

test(
  'alarmCounts/totalAlarmCount: no double-counting for the three tokens the aggregate treats as ALARMING -- ' +
    'one resident lands in exactly one bucket, and that bucket is the whole total, regardless of which of ' +
    'the three it is',
  () => {
    const r = makeResident();
    for (const token of ['prefill-hang', 'no-telemetry', 'stalled'] as const) {
      const counts = alarmCounts([r], () => token, {});
      assert.equal(counts[token], 1);
      assert.equal(totalAlarmCount(counts), 1);
    }
  },
);

test(
  'totalAlarmCount PURE -- a busy-only resident tallies counts.busy===1 (alarmCounts itself is ' +
    'unchanged, still counts every token) but contributes ZERO to the total (busy is excluded from ' +
    'BANNER_ALARM_TOKENS) -- pinned as its OWN case, not folded into the loop above, so a future edit that ' +
    'accidentally re-adds busy to that list breaks THIS test by name, not a generic loop iteration',
  () => {
    const r = makeResident();
    const counts = alarmCounts([r], () => 'busy', {});
    assert.equal(counts.busy, 1, 'alarmCounts still tallies busy -- the per-card badge depends on this count staying accurate');
    assert.equal(totalAlarmCount(counts), 0, 'but the aggregate total must read 0 -- a lone busy resident must never trip the red banner');
  },
);

test('alarmCounts: passes the per-resident busyForS lookup (keyed by model_tag) through to residentAlarm untouched', () => {
  const r = makeResident({ model_tag: 'the-tag' });
  let received: number | null | undefined;
  const residentAlarm: ResidentAlarm = (_resident, busyForS) => {
    received = busyForS;
    return '';
  };
  alarmCounts([r], residentAlarm, { 'the-tag': 137 });
  assert.equal(received, 137, 'busyForS must come from the busyTimers map keyed by model_tag, not be recomputed');
});

test('alarmCounts: a resident absent from the busyTimers map gets null busyForS, not undefined or 0 -- guards against a false BUSY_ESCALATE_S comparison downstream', () => {
  const r = makeResident({ model_tag: 'not-in-the-map' });
  let received: number | null | undefined;
  const residentAlarm: ResidentAlarm = (_resident, busyForS) => {
    received = busyForS;
    return '';
  };
  alarmCounts([r], residentAlarm, {});
  assert.equal(received, null);
});

/* ------------------------------------------------------------------ */
/*  advanceCombinedSparkline                                          */
/* ------------------------------------------------------------------ */

test('advanceCombinedSparkline: first real tick appends exactly one sample, the combinedTokS value', () => {
  const r = makeResident({ generation: makeGeneration({ tok_s: 25, measured_at_iso: 't1' }) });
  const next = advanceCombinedSparkline(INITIAL_SPARKLINE_STATE, [r], 60);
  assert.deepEqual(next.samples, [25]);
});

test('advanceCombinedSparkline: DEDUP -- an identical tick key (same backend snapshot re-rendered) returns the SAME object reference, samples unchanged', () => {
  const r = makeResident({ generation: makeGeneration({ tok_s: 25, measured_at_iso: 't1' }) });
  const first = advanceCombinedSparkline(INITIAL_SPARKLINE_STATE, [r], 60);
  const second = advanceCombinedSparkline(first, [r], 60);
  assert.equal(second, first, 'a duplicate tick must return the exact same reference, not just equal contents');
  assert.equal(second.samples.length, 1);
});

test('advanceCombinedSparkline: a NEW tick (measured_at_iso advances) appends a second sample', () => {
  const r1 = makeResident({ generation: makeGeneration({ tok_s: 25, measured_at_iso: 't1' }) });
  const r2 = makeResident({ generation: makeGeneration({ tok_s: 40, measured_at_iso: 't2' }) });
  const first = advanceCombinedSparkline(INITIAL_SPARKLINE_STATE, [r1], 60);
  const second = advanceCombinedSparkline(first, [r2], 60);
  assert.deepEqual(second.samples, [25, 40]);
  assert.notEqual(second, first);
});

test('advanceCombinedSparkline: REORDERING the same residents (no new data) does not append -- proves the sort-before-join is load-bearing, distinguishing fixture', () => {
  // DELIBERATELY different measured_at_iso per resident ('t1' vs 't2') --
  // if both shared one timestamp, joining WITHOUT sorting would coincidentally
  // produce the same string regardless of order, making this fixture inert
  // (a version of this fixture with one shared timestamp could not tell
  // sorted from unsorted joins).
  const a = makeResident({ model_tag: 'aaa', port: 1, generation: makeGeneration({ tok_s: 10, measured_at_iso: 't1' }) });
  const b = makeResident({ model_tag: 'bbb', port: 2, generation: makeGeneration({ tok_s: 20, measured_at_iso: 't2' }) });
  const first = advanceCombinedSparkline(INITIAL_SPARKLINE_STATE, [a, b], 60);
  // Same two residents, same data, just swapped array order -- a wrong impl
  // that joins measured_at_iso WITHOUT sorting first would compute a
  // different key here ("t2|t1" vs "t1|t2") and wrongly append a duplicate.
  const second = advanceCombinedSparkline(first, [b, a], 60);
  assert.equal(second, first, 'reordering alone must not register as a new tick');
  assert.equal(second.samples.length, 1);
});

test('advanceCombinedSparkline: FIFO-trims to maxSamples, dropping the OLDEST sample first', () => {
  let state = INITIAL_SPARKLINE_STATE;
  for (let i = 0; i < 5; i++) {
    const r = makeResident({ generation: makeGeneration({ tok_s: i, measured_at_iso: `t${i}` }) });
    state = advanceCombinedSparkline(state, [r], 3);
  }
  // Ticks were 0,1,2,3,4 -- capped at 3 samples, oldest (0,1) dropped, newest kept.
  assert.deepEqual(state.samples, [2, 3, 4]);
});

test('advanceCombinedSparkline: NEVER resets when the resident SET changes entirely -- the decisive property', () => {
  const modelA = makeResident({ model_tag: 'model-a', generation: makeGeneration({ tok_s: 10, measured_at_iso: 't1' }) });
  let state = advanceCombinedSparkline(INITIAL_SPARKLINE_STATE, [modelA], 60);
  assert.deepEqual(state.samples, [10]);

  // Total resident-set churn: model-a is gone, two entirely different
  // residents (different tags, different generation_ids) have appeared.
  // A reset-on-identity-change implementation would clear the buffer here;
  // the combined series must not -- it's the machine's continuous history.
  const modelB = makeResident({ model_tag: 'model-b', generation: makeGeneration({ tok_s: 15, measured_at_iso: 't2', generation_id: 'gen-b' }) });
  const modelC = makeResident({ model_tag: 'model-c', generation: makeGeneration({ tok_s: 5, measured_at_iso: 't2', generation_id: 'gen-c' }) });
  state = advanceCombinedSparkline(state, [modelB, modelC], 60);
  assert.deepEqual(state.samples, [10, 20], 'buffer must keep growing across a total resident-set change, not reset');
});
