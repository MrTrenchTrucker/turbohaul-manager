import { useEffect, useRef, useState } from 'react';
import type { GenerationInfo, ResidentModel } from '../../api';

// The stall/no-telemetry escalation
// machinery ported from ThroughputSection.tsx's derivePill (the aggregate
// box), scoped to a resident set. ThroughputSection.tsx is not edited by
// this file or its consumers; it could adopt this module too.
//
// residentAlarm() is the single source of truth for a resident's alarm
// state: ResidentCard.tsx renders it per-card, and the aggregate banner
// consumes the SAME verdict for its per-token count, so the card and the
// banner cannot disagree about which residents are alarmed.
//
// Split deliberately into a PURE half and a STATEFUL half:
// 'busy' vs 'no-telemetry' is a TIME distinction -- elapsed seconds is not
// readable from a ResidentModel snapshot alone, so residentAlarm() takes it
// as an explicit parameter rather than trying to derive it internally.
// useBusyTimers() is the one hook call that produces that parameter for an
// entire resident set at once (never one hook call per resident inside a
// .map()) -- ResidentsPanel resolves it once and passes each card its own
// value, the same shape as the `panes` prop.

export type AlarmToken = '' | 'busy' | 'prefill-hang' | 'no-telemetry' | 'stalled';

export const BUSY_ESCALATE_S = 120; // longest legit engine op (4.6GB KV restore) is ~30s; ThroughputSection.tsx's own value
export const PREFILL_STALL_AFTER_S = 60; // must match BE PREFILL_STALL_AFTER_S; ThroughputSection.tsx's own value

/**
 * PURE. Precedence when more than one condition is simultaneously true:
 * stalled > no-telemetry > prefill-hang > busy.
 *
 * In practice `stalled` and `no-telemetry` are structurally mutually
 * exclusive in production (a stalled generation is never classified
 * "quiet" by useBusyTimers' internal phase tracking, so busyForS stays
 * null whenever stalled is true) -- the explicit precedence order is kept
 * anyway so the function is correct even if that invariant is ever broken
 * elsewhere, and so `prefill-hang` (a genuinely independent boolean) is
 * never silently masked by a differently-ordered set of early returns.
 */
export function residentAlarm(model: ResidentModel, busyForS: number | null): AlarmToken {
  const gen = model.generation;
  if (!gen) return '';

  // A resident in its grace window would otherwise read 'busy'. This is WHY
  // it said busy, and it was not a mislabel.
  //
  // `busyForS` comes from useBusyTimers, which detects TELEMETRY GOING QUIET. The
  // backend starts a resident's grace window in _serve_on_resident at TURN COMPLETION --
  // which is precisely the moment telemetry goes quiet. So a detector whose only
  // evidence is quiet MUST classify every grace window as "engine busy -- telemetry
  // paused". Nothing in the payload distinguished "quiet because the turn ended" from
  // "quiet because the engine is stuck", so the card confidently showed the one other
  // thing that looks identical from the outside.
  //
  // `phase` is that missing discriminator. Suppress ONLY `busy`, and ONLY on GRACE:
  // stalled / no-telemetry / prefill-hang are untouched below, so a genuinely wedged
  // engine still alarms exactly as before. A backend that does not send `phase` leaves
  // it undefined, which is not 'GRACE', so busy fires unchanged -- older backends
  // keep their previous behaviour.
  const graceQuiet = model.phase === 'GRACE';

  const stalled = gen.stalled || gen.state === 'stalled';
  const noTelemetry = busyForS != null && busyForS >= BUSY_ESCALATE_S;
  const prefillHang = !!gen.prefill_stall_alarm;
  const busy = busyForS != null && !noTelemetry && !graceQuiet;

  if (stalled) return 'stalled';
  if (noTelemetry) return 'no-telemetry';
  if (prefillHang) return 'prefill-hang';
  if (busy) return 'busy';
  return '';
}

// ── useBusyTimers -- the one stateful hook, for the whole set ──────────────

// Bursty workloads hard-cycle gen.state generating->finishing->idle->generating
// every ~20s (same fork behaviour ThroughputSection's useActivityPhase names).
// A resident only counts as "quiet" once it has gone this long without a live
// tick -- otherwise every normal inter-burst gap would flap the busy timer.
const RECENT_HOLD_MS = 10000;

const LIVE_STATES: ReadonlySet<string> = new Set<string>([
  'generating',
  'prefill',
  'finishing',
  'loading',
  'grace',
  'stalled',
]);

function isLiveGeneration(gen: GenerationInfo): boolean {
  return gen.stalled || LIVE_STATES.has(gen.state);
}

interface TimerEntry {
  lastLiveAt: number; // 0 = never observed live
  busySince: number | null;
}

/**
 * ONE hook call for an entire resident set, keyed by model_tag (unique per
 * resident by construction -- the manager keys the resident registry by
 * model_tag). Recomputes
 * every resident's quiet-window/busy state once per commit rather than
 * owning a per-tag setTimeout: useStatus.ts polls at ~1Hz (POLL_INTERVAL_MS
 * = 1000), two orders of magnitude under BUSY_ESCALATE_S (120s) and one
 * under RECENT_HOLD_MS (10s), so piggybacking on the natural poll-driven
 * re-render costs at most ~1s of slop against either threshold -- far
 * cheaper and simpler than hand-rolling N independent timers whose set
 * changes as residents come and go.
 *
 * Entries for tags no longer present are pruned every tick, so a resident
 * that unloads and later reloads under the SAME tag starts its escalation
 * clock fresh rather than inheriting a stale one from a different, earlier
 * instance that happened to share the tag.
 */
export function useBusyTimers(residents: ResidentModel[]): Record<string, number | null> {
  const entries = useRef<Record<string, TimerEntry>>({});
  const [, forceTick] = useState(0);

  useEffect(() => {
    const now = Date.now();
    const seen = new Set(residents.map(r => r.model_tag));
    for (const tag of Object.keys(entries.current)) {
      if (!seen.has(tag)) delete entries.current[tag];
    }
    for (const resident of residents) {
      const tag = resident.model_tag;
      const gen = resident.generation;
      const entry = entries.current[tag] ?? { lastLiveAt: 0, busySince: null };
      if (gen && isLiveGeneration(gen)) entry.lastLiveAt = now;
      const quiet = !gen || entry.lastLiveAt === 0 || now - entry.lastLiveAt > RECENT_HOLD_MS;
      const engineBusy = quiet && resident.state === 'ACTIVE';
      if (engineBusy) {
        if (entry.busySince == null) entry.busySince = now;
      } else {
        entry.busySince = null;
      }
      entries.current[tag] = entry;
    }
    forceTick(t => t + 1);
  }, [residents]);

  const now = Date.now();
  const result: Record<string, number | null> = {};
  for (const resident of residents) {
    const tag = resident.model_tag;
    const entry = entries.current[tag];
    const busySince = entry?.busySince ?? null;
    result[tag] =
      busySince != null ? Math.max(0, Math.round((now - busySince) / 1000)) : null;
  }
  return result;
}
