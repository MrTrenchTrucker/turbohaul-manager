import type { GenerationInfo, ResidentModel, StatusSnapshot } from '../../api';

/**
 * Resolves a ResidentModel from the legacy single-residency fields (active /
 * loading / grace / idle_hot / generation) when residents[] is empty.
 *
 * Injected rather than imported so this module does not depend on where
 * synthesizeResident lives.
 * Callers pass the real synthesizeResident in at the call site
 * -- this file never imports it and never reimplements its body, so there is
 * exactly one definition and no path to silently diverge.
 */
export type SynthesizeResident = (status: StatusSnapshot) => ResidentModel | null;

/**
 * THE ONE RULE: aggregate over status.residents[] when it is non-empty; fall
 * back to the legacy synthesizeResident(status) path when it is empty.
 * residents[] is empty by design at max_parallel_sidecars<=1 -- a
 * residents[]-only implementation blanks every cap<=1 dashboard.
 */
export function selectResidents(
  status: StatusSnapshot,
  synthesizeResident: SynthesizeResident,
): ResidentModel[] {
  if (status.residents.length > 0) return status.residents;
  const legacy = synthesizeResident(status);
  return legacy ? [legacy] : [];
}

function liveGeneration(resident: ResidentModel): GenerationInfo | null {
  return resident.generation;
}

function finiteOrZero(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Sum of generation.tok_s over residents with a live (non-null) generation. */
export function combinedTokS(residents: ResidentModel[]): number {
  let total = 0;
  for (const resident of residents) {
    const gen = liveGeneration(resident);
    if (!gen) continue;
    total += finiteOrZero(gen.tok_s);
  }
  return total;
}

/**
 * Global sum(n_decoded)/sum(max_tokens) over residents with a live
 * generation, paired per-resident: a resident contributes to EITHER sum only
 * if it carries a valid positive max_tokens. A resident with n_decoded but no
 * known max_tokens has unknown capacity and must not inflate the fraction --
 * counting its n_decoded alone can push the ratio past 100%, which is a real
 * bug even though it never produces NaN/Infinity outright.
 * 0 residents / all-unknown-capacity -> 0, never NaN or Infinity.
 *
 * Display clamp: the paired guard above removes the
 * unknown-capacity case, but a resident legitimately IN the set can still
 * have n_decoded transiently exceed its own max_tokens -- live_monitor.py
 * (:674) clamps the identical nd/effective_max ratio for the same reason on
 * the per-resident value, first-party precedent this aggregate mirrors.
 */
export function outputFraction(residents: ResidentModel[]): number {
  let decoded = 0;
  let max = 0;
  for (const resident of residents) {
    const gen = liveGeneration(resident);
    if (!gen) continue;
    const maxTokens = gen.max_tokens;
    if (typeof maxTokens !== 'number' || !Number.isFinite(maxTokens) || maxTokens <= 0) continue;
    decoded += finiteOrZero(gen.n_decoded);
    max += maxTokens;
  }
  const frac = max > 0 ? decoded / max : 0;
  return Math.min(1, Math.max(0, frac));
}

/**
 * Mean of generation.prefill_pct over residents currently prefilling, or
 * NULL when nobody is prefilling right now.
 *
 * null is a DISTINCT fact from 0: an
 * average over an empty set is undefined, not zero. Each resident cycles
 * prefill -> decode independently, so with 2+ residents the "someone is
 * mid-prefill" set is intermittently empty even while prefill work is
 * genuinely happening across the machine -- a hard 0 there rendered as
 * "0% complete", a lie distinct from "nothing is prefilling this instant".
 * Callers must treat null as no-data and render it explicitly as such, never
 * coerce it to a number (see ThroughputSection.tsx's SplitBar).
 *
 * "Currently prefilling" gates on generation.state === 'prefill', NEVER on
 * prefill_pct's own value: on this fork prefill_pct can read <100 through all
 * of healthy decode once prefill completes (documented in the live monitor),
 * so a pct-based test misclassifies an already-decoding resident as still
 * prefilling. A prefilling resident whose prefill_pct hasn't populated yet
 * (startup race, admission_ctx_len momentarily 0) still counts toward the
 * denominator -- it IS prefilling -- but contributes 0, never NaN.
 */
export function prefillMean(residents: ResidentModel[]): number | null {
  const prefilling = residents.filter((resident) => liveGeneration(resident)?.state === 'prefill');
  if (prefilling.length === 0) return null;
  const sum = prefilling.reduce(
    (acc, resident) => acc + finiteOrZero(liveGeneration(resident)?.prefill_pct),
    0,
  );
  return sum / prefilling.length;
}

/**
 * Injection contract: the card component is the SINGLE
 * AUTHOR of the per-resident alarm verdict -- its ResidentCard and this
 * aggregate banner must call the SAME classifier, never two independently
 * -derived ones, or they can silently disagree on screen with no test able
 * to catch it (each half would be internally consistent).
 *
 * residentAlarm is deliberately NOT pure over `resident` alone: busy-vs-
 * no-telemetry is a TIME distinction (how long telemetry has been silent
 * behind an ACTIVE slot), which cannot live inside a per-resident call made
 * from inside a loop without breaking rules-of-hooks (a variable number of
 * hook calls per render). So the timer state is hoisted into ONE hook call
 * over the whole set -- `useBusyTimers(residents)` -- and its per-resident
 * result is passed into the otherwise-pure `residentAlarm(resident,
 * busyForS)`. Both live in a sibling module, dashboard/alarms.ts -- same
 * pattern as SynthesizeResident: the types are declared here from the
 * published SIGNATURE, the real functions are injected, never imported here
 * and never reimplemented.
 */
export type AlarmToken = '' | 'busy' | 'prefill-hang' | 'no-telemetry' | 'stalled';
export type ResidentAlarm = (resident: ResidentModel, busyForS: number | null) => AlarmToken;
/** Keyed by model_tag, per useBusyTimers' published signature. */
export type BusyTimers = Record<string, number | null>;
export type UseBusyTimers = (residents: ResidentModel[]) => BusyTimers;

export type AlarmTokenCounts = {
  busy: number;
  'prefill-hang': number;
  'no-telemetry': number;
  stalled: number;
};

function emptyAlarmTokenCounts(): AlarmTokenCounts {
  return { busy: 0, 'prefill-hang': 0, 'no-telemetry': 0, stalled: 0 };
}

/**
 * Per-token counts across residents. Counts BY CALLING the injected
 * residentAlarm -- never re-derives severity (the injection contract forbids this in
 * both ThroughputSection.tsx and this file). Deliberately broken down BY
 * TOKEN rather than lumped into one number: 1 stalled and 1 no-telemetry
 * are different operational facts, and collapsing them would hide which one
 * an operator needs to act on. This file makes zero judgment about WHICH
 * token a resident gets (that's residentAlarm's job, i.e. the card's) or how
 * urgent one token is relative to another (that's the card's severity ordering,
 * not this function's) -- it only tallies what it was told.
 */
export function alarmCounts(
  residents: ResidentModel[],
  residentAlarm: ResidentAlarm,
  busyTimers: BusyTimers,
): AlarmTokenCounts {
  const counts = emptyAlarmTokenCounts();
  for (const resident of residents) {
    const busyForS = busyTimers[resident.model_tag] ?? null;
    const token = residentAlarm(resident, busyForS);
    if (token !== '') counts[token] += 1;
  }
  return counts;
}

/**
 * Which tokens the AGGREGATE treats as genuinely alarming.
 * 'busy' is deliberately excluded -- a resident mid-generation is normal
 * operation, not trouble, and summing it into the red banner's total made
 * it flicker on ordinary traffic. The per-card AlarmBadge still shows
 * 'busy' unchanged -- this list governs only the aggregate's OWN use of the
 * verdict, in this file and in ThroughputSection.tsx's breakdown text, not
 * the verdict itself. ONE list, consumed by both totalAlarmCount below and
 * AlarmBanner's breakdown filter, so the two cannot independently drift the
 * way two separately-hardcoded exclusions could (same "one fact used
 * twice, one place it lives" shape as contextTone.ts).
 */
export const BANNER_ALARM_TOKENS = ['stalled', 'no-telemetry', 'prefill-hang'] as const satisfies readonly Exclude<AlarmToken, ''>[];

/** Sum across every token the aggregate treats as alarming -- "is there anything genuinely wrong to show". */
export function totalAlarmCount(counts: AlarmTokenCounts): number {
  return BANNER_ALARM_TOKENS.reduce((sum, token) => sum + counts[token], 0);
}

/** Rolling combined-throughput sparkline state: a fixed-size FIFO of
 * combinedTokS(residents) samples, plus the tick key the last sample was
 * taken under (see advanceCombinedSparkline). */
export interface SparklineState {
  samples: number[];
  key: string;
}

export const INITIAL_SPARKLINE_STATE: SparklineState = { samples: [], key: '' };

/**
 * "Did the backend produce a new tick since we last sampled" key: a sorted
 * join of every resident's own generation.measured_at_iso. All live
 * residents are refreshed together once per backend poll tick --
 * live_monitor.py's LiveResidentsSupervisor._tick() gathers every resident
 * concurrently (asyncio.gather) and writes them all before the next
 * asyncio.sleep(self._interval) -- so this key changes exactly when real new
 * data arrived, and stays identical across a duplicate/WS-triggered re-fetch
 * that lands before the next real tick (same race the legacy single-gen
 * `measured_at_iso` dedup guarded against, generalized to N residents).
 * Sorted first so a mere reorder of the SAME residents (no new data) can
 * never be mistaken for a new tick.
 */
function sparklineTickKey(residents: ResidentModel[]): string {
  return residents
    .slice()
    .sort((a, b) => `${a.model_tag}:${a.port}`.localeCompare(`${b.model_tag}:${b.port}`))
    .map((resident) => liveGeneration(resident)?.measured_at_iso ?? '')
    .join('|');
}

/**
 * Advance the combined-throughput rolling sparkline by one tick. Deliberately
 * NEVER resets on residency/generation-identity churn (the
 * reasoning below): unlike a single generation's own
 * history -- where blending across an unrelated NEXT burst is misleading --
 * the combined series has no single "burst" identity to protect; it IS the
 * machine's continuous rolling activity, and a peak that persists briefly
 * across one resident's generation boundary while others keep running is
 * correct, not stale. Dedupes same-tick re-renders via sparklineTickKey
 * (returns the SAME object reference, not just equal values, when deduped --
 * callers can use that to skip a wasted re-render). FIFO-trims to maxSamples.
 */
export function advanceCombinedSparkline(
  prev: SparklineState,
  residents: ResidentModel[],
  maxSamples: number,
): SparklineState {
  const key = sparklineTickKey(residents);
  if (key === prev.key) return prev;
  const next = [...prev.samples, combinedTokS(residents)];
  const trimmed = next.length > maxSamples ? next.slice(next.length - maxSamples) : next;
  return { samples: trimmed, key };
}
