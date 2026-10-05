import { useEffect, useRef, useState } from 'react';
import type { ResidentModel } from '../../api';
import {
  advanceCombinedSparkline,
  alarmCounts,
  BANNER_ALARM_TOKENS,
  combinedTokS,
  INITIAL_SPARKLINE_STATE,
  outputFraction,
  prefillMean,
  totalAlarmCount,
} from './aggregate';
import type { AlarmTokenCounts, ResidentAlarm, SparklineState, UseBusyTimers } from './aggregate';
import {
  CONTEXT_TONE_BAR_CLASSES,
  CONTEXT_TONE_TEXT_CLASSES,
  contextTone,
} from './contextTone';

/* ================================================================== */
/*  AGGREGATE TOP BOX -- describes the WHOLE MACHINE across every       */
/*  resident, not one engine. Per-resident detail lives on the          */
/*  per-resident ResidentCard/ResidentsPanel.                           */
/* ================================================================== */

// This is the WINDOW the combined graph describes ("how far back
// does this look"), NOT a resolution/pixel-density knob. Stretching the
// sparkline's rendered WIDTH is a cosmetic change; changing this
// number silently changes what the graph MEANS -- 60 samples reads as "the
// last 60 backend ticks", and doubling it would silently claim twice that
// history while looking like an unrelated polish tweak. Pinned by a direct
// test (ThroughputSection.test.tsx), because changing this value (30->60)
// silently doubles the lookback and no other assertion would catch it --
// only an explicit assertion on this exact value does.
// Exported ONLY so that test can assert the real constant, not a copy of it.
export const SPARK_SAMPLES = 60;

function fmtTokS(v: number): string {
  return v.toFixed(1);
}

function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

/* ------------------------------------------------------------------ */
/*  Context-size readout, OVERALL AGGREGATE                              */
/* ------------------------------------------------------------------ */

// green <75% -- amber 75-90% -- red >90%.
// The rule itself lives in ./contextTone -- a dependency-free classifier owned
// by neither component, so this aggregate module never has to import the
// per-resident card file (see the file header above) to share it.

function contextKey(resident: ResidentModel): string {
  return `${resident.model_tag}:${resident.port}`;
}

export interface ResidentContext {
  used: number;
  capacity: number;
  pct: number;
}

/**
 * PURE. Resolves ONE resident's current-or-last-known context, or null if
 * this resident has never reported one. Reads
 * resident.generation directly, NEVER status.generation -- that field
 * mirrors ONE resident, not a sum. lastKnown is supplied by the
 * caller rather than read from a ref directly, so this half stays fully
 * unit-testable without a live effect cycle -- same split as
 * ResidentCard.tsx's resolveContextDisplay/useLastKnownContext.
 */
export function resolveResidentContext(
  resident: ResidentModel,
  lastKnown: { used: number; capacity: number } | undefined,
): ResidentContext | null {
  const nCtx = resident.generation?.n_ctx;
  if (nCtx != null) {
    const used = resident.generation?.n_prompt_tokens ?? 0;
    return { used, capacity: nCtx, pct: nCtx > 0 ? (used / nCtx) * 100 : 0 };
  }
  if (lastKnown != null) {
    return { ...lastKnown, pct: lastKnown.capacity > 0 ? (lastKnown.used / lastKnown.capacity) * 100 : 0 };
  }
  return null;
}

/**
 * PURE. Mirrors combinedTokS's shape (aggregate.REFERENCE.ts): loop over
 * residents[], skip no-data, accumulate. Sums the SAME current-or-last-known
 * reading resolveResidentContext resolves per resident -- by design,
 * an idle resident's last-known context still occupies real window,
 * so it counts toward the true sum exactly like a live one (excluding it
 * would silently undercount, the same shape of bug as reading
 * status.generation). Returns null only when NO resident has ever reported
 * context -- nothing to show yet, distinct from "0 used of some capacity".
 */
export function combinedContext(
  residents: ResidentModel[],
  lastKnownByKey: Record<string, { used: number; capacity: number }>,
): { used: number; capacity: number } | null {
  let used = 0;
  let capacity = 0;
  let any = false;
  for (const resident of residents) {
    const ctx = resolveResidentContext(resident, lastKnownByKey[contextKey(resident)]);
    if (!ctx) continue;
    any = true;
    used += ctx.used;
    capacity += ctx.capacity;
  }
  return any ? { used, capacity } : null;
}

/**
 * PURE reducer, exported: merges this tick's CURRENT readings into the
 * previous last-known map. A resident NOT currently reporting (idle) has
 * its prior entry carried forward UNCHANGED, never dropped; a resident that
 * IS currently reporting has its entry overwritten with the fresh reading.
 * Same accumulate-don't-replace shape aggregate.ts's own
 * advanceCombinedSparkline already uses for its rolling FIFO (this file's
 * own header comment credits that function's pure-reducer tests as proving
 * "the sparkline's own tick-to-tick accrual... needs no rendering at all") --
 * applied here for the same reason: useLastKnownContextByResident's
 * useEffect body can never run under this repo's SSR test harness, so this
 * split is what makes tick-to-tick persistence testable at all without a
 * live render. A reducer that rebuilt next={} from only this tick's live residents
 * instead of merging into prev would silently drop every idle
 * resident's memory one tick after it goes idle. That would be invisible to
 * every combinedContext test (they all hand-construct their lastKnownByKey
 * fixture directly and never exercise this reducer); only this
 * function's own direct two-tick tests below would catch it.
 */
export function mergeLastKnownContext(
  prev: Record<string, { used: number; capacity: number }>,
  residents: ResidentModel[],
): Record<string, { used: number; capacity: number }> {
  const next = { ...prev };
  for (const resident of residents) {
    const nCtx = resident.generation?.n_ctx;
    if (nCtx != null) {
      next[contextKey(resident)] = { used: resident.generation?.n_prompt_tokens ?? 0, capacity: nCtx };
    }
  }
  return next;
}

// Same ref+effect shape as ResidentCard.tsx's useLastKnownContext, hoisted to
// the WHOLE resident set in one hook call -- same generalization
// useBusyTimers already applies to per-resident busy-state (one hook call
// over the set, not one per resident inside a loop -- rules-of-hooks). This
// is the aggregate's OWN last-known memory, entirely separate from each
// ResidentCard instance's own ref: same duality as BigSparkline's rolling
// samples already being independent state from each card's useTokSHistory,
// over the same underlying data. Thin wiring only -- all the real logic is
// in mergeLastKnownContext above.
function useLastKnownContextByResident(
  residents: ResidentModel[],
): Record<string, { used: number; capacity: number }> {
  const lastKnown = useRef<Record<string, { used: number; capacity: number }>>({});
  useEffect(() => {
    lastKnown.current = mergeLastKnownContext(lastKnown.current, residents);
  }, [residents]);
  return lastKnown.current;
}

// The combined box colours by the COMBINED value ONLY: one near-full
// resident next to a nearly empty one must not turn the combined box
// amber/red while the combined value itself is calm. There is no worst-of
// tone rule for this box; the anti-
// dilution protection lives on each
// resident's OWN card (ResidentCard.tsx), which already colours by its own
// value including an idle one on last-known -- see the near-full-resident
// regression test in ResidentCard.test.tsx. No worst-item lives here: at
// N=1 it would be pure noise (worst IS combined, by identity). resolvedTone
// below is computed LOCALLY from pct, not caller-supplied, on purpose.
//
// Row shape otherwise matches the combined-tok/s row: used/capacity
// sits up beside the "Combined Context" label (one line saved,
// same as the tok/s box), pct is the shrink-0 hero
// number, fullness bar is the flex-1 filler.
function ContextAggRow({
  combined,
}: {
  combined: { used: number; capacity: number } | null;
}) {
  const pct = combined && combined.capacity > 0 ? (combined.used / combined.capacity) * 100 : null;
  const resolvedTone = pct != null ? contextTone(pct) : 'green';
  const width = pct != null ? Math.min(100, Math.max(0, pct)) : 0;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-5">
      {/* p-5, not p-6 -- SplitBar below already claims p-4 on this
          same rounded-lg/border-slate-700/bg-slate-950 combo so that each
          independently-countable box in this section has its own class
          fingerprint (see the "exactly one p-6 box" test a few lines up in
          ThroughputSection.test.tsx, which this deliberately does not
          collide with). */}
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-xs uppercase tracking-wide text-slate-500">Combined Context</div>
        <div className="text-sm text-slate-500 tabular-nums">
          <span data-testid="agg-context-used" data-value={combined?.used ?? 0}>
            {combined != null ? fmtInt(combined.used) : '—'}
          </span>
          {' / '}
          <span data-testid="agg-context-capacity" data-value={combined?.capacity ?? 0}>
            {combined != null ? fmtInt(combined.capacity) : '—'}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span
          data-testid="agg-context-pct"
          data-value={pct ?? ''}
          className={`shrink-0 text-4xl font-bold tabular-nums leading-none ${CONTEXT_TONE_TEXT_CLASSES[resolvedTone]}`}
        >
          {pct != null ? `${Math.round(pct)}%` : '—'}
        </span>
        <div className="flex-1 min-w-0">
          <div className="h-2 w-full overflow-hidden rounded bg-slate-800">
            <div
              data-testid="agg-context-bar"
              className={`h-full ${CONTEXT_TONE_BAR_CLASSES[resolvedTone]} transition-all`}
              style={{ width: `${width}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Combined sparkline (240x48) ─────────────────────────────────────────
function BigSparkline({ samples }: { samples: number[] }) {
  const W = 240;
  const H = 48;
  if (samples.length < 2) {
    return (
      <div className="flex h-12 items-center justify-center text-xs text-slate-600">
        — gathering samples —
      </div>
    );
  }
  const max = Math.max(...samples, 1);
  const step = W / (SPARK_SAMPLES - 1);
  const points = samples
    .map((v, i) => {
      const x = i * step;
      const y = H - (v / max) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-12 w-full"
      role="img"
      aria-label="combined tokens per second history"
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        className="text-emerald-400"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// ── Hero: combined tok/s ─────────────────────────────────────────────────
// Not its own bordered box -- the caller wraps Hero AND the
// combined sparkline in ONE shared border (the two boxes were merged into
// one, full width). It is a SIBLING of the sparkline in one flex
// row (matching the resident-card row shape) -- shrink-0 so it
// keeps its natural width and leaves the rest of the row to the graph, same
// division of labour as ResidentCard's own label+number column. Dropped
// "sum across every active resident": redundant once the row reads as one
// unit -- "Combined Throughput" already says combined, and the section
// header above already states the resident count.
function Hero({ combined }: { combined: number }) {
  return (
    <div className="shrink-0">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">Combined Throughput</div>
      <div className="flex items-end gap-3">
        <span
          data-testid="agg-tok-s"
          data-value={combined}
          className="text-7xl font-bold tabular-nums leading-none text-emerald-300"
        >
          {fmtTokS(combined)}
        </span>
        <span className="text-2xl font-medium text-slate-500 pb-1">tok/s</span>
      </div>
    </div>
  );
}

// ── Split bar: prefill (left) | output (right), rendered simultaneously ──
// data-value carries the RAW function output, unnormalized --
// agg-prefill-pct is prefillMean()'s native 0-100 scale, agg-output-pct is
// outputFraction()'s native 0-1 clamped scale. Deliberately NOT the same
// scale (normalizing would be the exact
// silent asymmetry a gate gets wrong) -- the display text below handles the
// human-facing formatting; data-value stays each function's own contract.
//
// prefillPct is number|null -- null means "nobody is prefilling
// right now" (prefillMean's own contract, aggregate.ts), a distinct fact
// from a real 0%. Rendered as an EXPLICIT no-data state: dash text, 0-width
// fill (nothing to show either way, made explicit rather than relying on
// Math.max's implicit null->0 coercion, which would not even typecheck).
// data-testid AND data-value stay ALWAYS present (data-value="" for
// no-data) -- do NOT omit the attribute. Two reasons, both checked in the
// tree: (1) ThroughputSection.test.tsx's extractDataValue can't distinguish
// an omitted attribute from a missing element (both regex-miss to null), so
// a dropped SplitBar would silently read as this healthy no-data state; (2)
// ResidentCard.tsx's resident-tok-s already rules the same way for the same
// reason ("a missing element is ambiguous between 'correctly excluded' and
// 'a bug dropped it'") -- two widgets in the same panel must not answer that
// question differently.
function SplitBar({ prefillPct, outputFrac }: { prefillPct: number | null; outputFrac: number }) {
  const outputPct = outputFrac * 100;
  const prefillWidth = prefillPct != null ? Math.min(100, Math.max(0, prefillPct)) : 0;
  const prefillText = prefillPct != null ? `${Math.round(prefillPct)}%` : '—';
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
      <div className="flex items-baseline justify-between mb-2 text-xs uppercase tracking-wide text-slate-500">
        <span>prefill · mean of active prefills</span>
        <span>output · decoded / max, all residents</span>
      </div>
      <div className="flex h-3 w-full gap-1">
        <div className="h-full w-1/2 overflow-hidden rounded bg-slate-800">
          <div
            data-testid="agg-prefill-pct"
            data-value={prefillPct ?? ''}
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${prefillWidth}%` }}
          />
        </div>
        <div className="h-full w-1/2 overflow-hidden rounded bg-slate-800">
          <div
            data-testid="agg-output-pct"
            data-value={outputFrac}
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, outputPct))}%` }}
          />
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs font-mono text-slate-400 tabular-nums">
        <span>{prefillText}</span>
        <span>{Math.round(outputPct)}%</span>
      </div>
    </div>
  );
}

// ── Alarm banner: COUNTS, never a state pill ─────────────────────────────
// Design decision: no aggregate StatePill -- there is no single
// coherent STATE for N residents in different states at once, and faking
// one would be a confident lie. But dropping the stall/no-telemetry signal
// entirely with nobody told to carry it lets a dead engine behind an ACTIVE
// slot read as calm -- so this narrow banner carries COUNTS only, one per
// AlarmToken ("1 stalled and 1 no-telemetry are
// different operational facts"), never a single lumped number and never a
// re-derived severity. data-value on the root stays the TOTAL (matching the
// data-value contract, which predates the per-token breakdown); the
// breakdown is in the display text,
// which is free to change without touching that contract. Per-resident
// detail (which one, how long) is on the per-resident ResidentCard.
//
// The breakdown iterates BANNER_ALARM_TOKENS (stalled/no-telemetry/
// prefill-hang), not every key in ALARM_LABELS -- 'busy' keeps a label here
// (documentation for the reader, harmless if unused) but is never listed,
// so a busy+stalled fixture reads "1 stalled", never "1 busy, 1 stalled".
// Using the SAME list totalAlarmCount sums is what makes it impossible for
// the total and the breakdown text to disagree about which tokens count.
const ALARM_LABELS: Record<keyof AlarmTokenCounts, string> = {
  stalled: 'stalled',
  'no-telemetry': 'no telemetry',
  'prefill-hang': 'prefill hang',
  busy: 'busy',
};

function AlarmBanner({ total, counts }: { total: number; counts: AlarmTokenCounts }) {
  const grandTotal = totalAlarmCount(counts);
  if (grandTotal === 0) return null;
  const breakdown = BANNER_ALARM_TOKENS
    .filter((token) => counts[token] > 0)
    .map((token) => `${fmtInt(counts[token])} ${ALARM_LABELS[token]}`)
    .join(', ');
  return (
    <div
      data-testid="agg-alarm-banner"
      data-value={grandTotal}
      className="rounded border-2 border-red-700 bg-red-950/40 p-3"
    >
      <div className="flex items-center gap-2 text-red-300">
        <span className="font-mono text-sm">⚠ ALARM</span>
        <span className="text-xs">
          {breakdown} of {fmtInt(total)} resident{total === 1 ? '' : 's'} — see cards below
        </span>
      </div>
    </div>
  );
}

export function ThroughputSection({
  residents,
  residentAlarm,
  useBusyTimers,
}: {
  residents: ResidentModel[];
  // The per-resident alarm module is the single author of the alarm verdict.
  // Injected exactly like SynthesizeResident -- dashboard/alarms.ts
  // didn't exist in the tree when this was written, so an import would
  // fail to compile; DI needs no path and can't silently regress to "always
  // clear" the way a stale/missing import could. residentAlarm is pure over
  // (resident, busyForS); useBusyTimers is the ONE hook call that supplies
  // busyForS per resident -- it must be called unconditionally here, not
  // per-resident inside a loop (rules-of-hooks).
  residentAlarm: ResidentAlarm;
  useBusyTimers: UseBusyTimers;
}) {
  // Rolling combined-throughput sparkline. Pure accumulation logic lives in
  // advanceCombinedSparkline (aggregate.ts, directly unit-tested); this hook
  // is thin wiring only -- store the returned state, re-render only when it
  // actually changed (advanceCombinedSparkline returns the SAME reference on
  // a deduped/no-op tick).
  const sparkState = useRef<SparklineState>(INITIAL_SPARKLINE_STATE);
  const [, forceTick] = useState(0);

  useEffect(() => {
    const next = advanceCombinedSparkline(sparkState.current, residents, SPARK_SAMPLES);
    if (next !== sparkState.current) {
      sparkState.current = next;
      forceTick((t) => t + 1);
    }
  }, [residents]);

  // Called unconditionally, above the early return, same invariant as every
  // other hook in this component -- the busy-timer state for the WHOLE
  // resident set in one call, not one hook call per resident.
  const busyTimers = useBusyTimers(residents);

  // Same invariant -- called unconditionally, above the early
  // return, one hook call for the whole resident set.
  const lastKnownContextByResident = useLastKnownContextByResident(residents);

  if (residents.length === 0) {
    return (
      <div className="space-y-3" data-testid="agg-box">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-slate-300">LIVE INFERENCE</h2>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-6 text-sm italic text-slate-500">
          — no active generation —
        </div>
      </div>
    );
  }

  const combined = combinedTokS(residents);
  const prefill = prefillMean(residents);
  const output = outputFraction(residents);
  const total = residents.length;
  const counts = alarmCounts(residents, residentAlarm, busyTimers);
  const samples = sparkState.current.samples;
  const peak = samples.length > 0 ? Math.max(...samples) : 0;
  const combinedCtx = combinedContext(residents, lastKnownContextByResident);

  return (
    <div className="space-y-3" data-testid="agg-box">
      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <h2 className="text-sm font-semibold text-slate-300">LIVE INFERENCE</h2>
          <span className="text-xs font-mono text-slate-500">
            {fmtInt(total)} resident{total === 1 ? '' : 's'}
          </span>
          {/* Optional: busy no longer alarms (see
              BANNER_ALARM_TOKENS, aggregate.ts), but the count is still
              worth showing SOMEWHERE -- neutral, same slate tone as the
              resident count beside it, no red/alarm styling anywhere near
              it. Shown only when > 0, same "nothing to show when there's
              nothing to say" convention AlarmBanner itself already uses --
              a permanent "0 busy" would just be a second, quieter clock
              nobody asked for. */}
          {counts.busy > 0 && (
            <span className="text-xs font-mono text-slate-500" data-testid="agg-busy-count" data-value={counts.busy}>
              · {fmtInt(counts.busy)} busy
            </span>
          )}
        </div>
      </div>

      <AlarmBanner total={total} counts={counts} />

      {/* ONE box, not two nested. The number
          and the sparkline are SIBLINGS in one flex row (matching the
          resident-card row shape) instead of stacked -- the old
          border-t divider is dropped entirely, since a divider between two
          stacked blocks has nothing left to separate once they share a
          line. BigSparkline itself is UNTOUCHED (only the caller
          layout changes, not the component) -- flex-1 min-w-0 lives on this wrapper
          div, and BigSparkline's own w-full fills it, same as its
          "gathering samples" placeholder, which has no width class of its
          own and so also just fills whatever wraps it.

          Below sm (mobile, e.g. 412px), a plain side-by-side row would
          split into a shrink-0 Hero block and a leftover sliver for the
          graph -- starved of width, not overlapping. How starved depends
          on the value: Hero is sized by the digits, so at "0.0" the graph
          would get 86px of 412, at "26.8" (measured live mid-generation) it
          gets 36px, and at triple-digit tok/s it goes to ~0. A mobile-only
          negative margin borrowing the ~70px of dead space beside the
          number is not used: 36+40 is still worse than the 86 of the
          idle case. flex-col
          stacks Hero on its own full-width row with the sparkline wrapper
          on a full-width row below it; sm:flex-row sm:items-center restores
          today's exact side-by-side row unchanged at >=640px. Reuses
          App.tsx:41's own flex-col/sm:flex-row idiom rather than inventing
          a new one -- no useMediaQuery/matchMedia, breakpoint class only. */}
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <Hero combined={combined} />
          <div className="flex-1 min-w-0">
            <BigSparkline samples={samples} />
          </div>
        </div>
        {/* "instant" dropped -- exact duplicate of the same
            `combined` value already shown at text-7xl right next to the
            graph now, so it was a full line for zero new information.
            "last N" (the window) and peak (genuinely new, shown nowhere
            else) survive, merged into one compact line. */}
        <div className="mt-2 text-xs text-slate-500 tabular-nums">
          last {SPARK_SAMPLES} · peak {fmtTokS(peak)} tok/s
        </div>
      </div>

      <ContextAggRow combined={combinedCtx} />

      <SplitBar prefillPct={prefill} outputFrac={output} />
    </div>
  );
}
