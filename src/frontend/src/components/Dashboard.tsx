import { useMemo } from 'react';
import { useStatus } from '../hooks/useStatus';
import { useLiveStream } from '../hooks/useLiveStream';
import { synthesizeResident } from './dashboard/synthesizeResident';
import { selectResidents } from './dashboard/aggregate';
import { tokRateTone } from './dashboard/tokRate';
import { residentAlarm, useBusyTimers } from './dashboard/alarms';
import { Card, KV } from './dashboard/primitives';
import { ThroughputSection } from './dashboard/ThroughputSection';
import { ResidentsPanel } from './dashboard/ResidentsPanel';

export { RequestIdentityStrip } from './dashboard/RequestIdentityStrip';
import { waitingCount } from './queue/waiting';
export { SpecDowngradeWidget } from './dashboard/ResidentCard';

/* ------------------------------------------------------------------ */
/*  Tok/s sparkline (per-pane, hand-rolled inline SVG)                  */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Live Output Panes (one per loaded model)                             */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Queue + Parallel Slots mini-card (kept from old dashboard)            */
/* ------------------------------------------------------------------ */

export function QueueCard({
  queue,
  parallelSlots,
}: {
  queue: {
    acceptance_buffer_depth: number;
    staging_queue_depth: number;
    staging_queue_max: number;
    queue_depth_total?: number;
  };
  parallelSlots: { used: number; max: number };
}) {
  // THE HEADLINE NUMBER IS THE WAITING TOTAL, NOT THE STAGING
  // DEPTH. Staging is transient by design -- a request sits there for
  // milliseconds before a free resident takes it -- so a card keyed on it
  // reads 0 almost always and blinks to 1 for a fraction of a second, which
  // is why a card keyed on it appeared to flap every 10 seconds while the box was
  // genuinely busy with six requests queued. The total counts requests held
  // behind a busy resident, which never enter staging at all, and it stays
  // put for as long as they are really waiting.
  //
  // Staging and the acceptance buffer are kept below as secondary detail:
  // they are real numbers and the N / max shape is the familiar one.
  //
  // ⛔ waitingCount is IMPORTED, never reimplemented here. This card and the
  // Queue tab's WaitingCard must use one shared definition of "waiting" so
  // the two screens cannot drift apart. One definition, two screens.
  const waiting = waitingCount(queue);
  const pct =
    queue.staging_queue_max > 0
      ? Math.min(100, ((waiting ?? queue.staging_queue_depth) / queue.staging_queue_max) * 100)
      : 0;
  return (
    <Card
      title="Queue"
      tone={waiting !== null && waiting > 0 ? 'border-amber-700' : 'border-slate-700'}
    >
      <div className="text-lg font-semibold text-slate-200" data-testid="dashboard-queue-waiting">
        {waiting ?? '—'}
      </div>
      <div className="text-xs text-slate-500 mb-2">
        {/* null is an OLDER MANAGER that does not send the field, not an idle
            box. Saying so beats rendering a fabricated 0, which is
            indistinguishable from "nothing is waiting" and is wrong in the
            one direction that matters. */}
        {waiting === null ? 'not reported by this manager' : 'waiting'}
      </div>
      <div className="h-2 bg-slate-800 rounded mb-3 overflow-hidden">
        <div className="h-full bg-amber-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <KV k="staging" v={`${queue.staging_queue_depth} / ${queue.staging_queue_max}`} />
      <KV k="acceptance buffer" v={queue.acceptance_buffer_depth} />
      <KV k="parallel slots" v={`${parallelSlots.used} / ${parallelSlots.max}`} />
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Dashboard (split view)                                          */
/* ------------------------------------------------------------------ */

export default function Dashboard() {
  const { data, error, lastUpdate } = useStatus();

  // When residents[] is empty (single-residency, cap<=1), synthesize a
  // partial ResidentModel from the legacy active/loading/grace/idle_hot
  // fields + the generation alias. This bridges the split-view frontend
  // back to the data the operator wants to see.
  // THE ONE RULE lives in exactly ONE place: selectResidents (dashboard/aggregate.ts),
  // so the dashboard cannot diverge from it: a fix to selectResidents
  // reaches the dashboard automatically.
  const effectiveResidents = useMemo(
    () => (data ? selectResidents(data, synthesizeResident) : []),
    [data],
  );

  // One SSE connection PER RESIDENT, keyed off the same list the cards render
  // from -- so a fresh page load opens every resident's stream immediately.
  // The old no-arg call followed the manager's single "most-recently-active"
  // anchor, so `panes` only filled in for residents the tab happened to
  // witness while open: after a refresh it showed exactly one live output box
  // no matter how many residents were generating.
  const { panes, connected: streamConnected } = useLiveStream(
    effectiveResidents.map((r) => r.model_tag),
  );

  if (!data) {
    return (
      <div className="text-slate-400">
        {error ? (
          <div className="text-amber-400">Error fetching /status: {error.message}</div>
        ) : (
          <div>Loading…</div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top row: queue + parallel slots */}
      <QueueCard queue={data.queue} parallelSlots={data.parallel_slots} />

      {/* Full-width LIVE INFERENCE / throughput block — reads data.generation
          (the primary gen block) directly. Single-residency + series show the
          full picture; double-parallel shows the primary + a concurrency caption. */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-4">
        <ThroughputSection
          residents={effectiveResidents}
            residentAlarm={residentAlarm}
            useBusyTimers={useBusyTimers}
          />
      </div>

      {/* Each resident card carries its OWN live output box, so there is no
          separate full-height Live Output column -- keeping both would show
          the same output twice. Residents run full width. */}
      <div>
        <ResidentsPanel
            residents={effectiveResidents}
          vram={data.vram}
          vramTotal={data.vram_total_mib}
          parallelSlots={data.parallel_slots}
          requestIdentity={data.request_identity}
          loadVerify={data.load_verify}
          specDowngrade={data.spec_downgrade}
          panes={panes}
          tokRateTone={tokRateTone}
        />
      </div>

      {/* Status footer */}
      <div className="text-xs text-slate-500 flex items-center gap-3">
        <span>
          last update:{' '}
          <span className="font-mono">{lastUpdate?.toISOString() ?? '—'}</span>
        </span>
        <span>
          sse:{' '}
          <span className={streamConnected ? 'text-emerald-400' : 'text-amber-400'}>
            {streamConnected ? 'connected' : 'disconnected'}
          </span>
        </span>
        {error && <span className="text-amber-400">⚠ {error.message} (retrying…)</span>}
      </div>
    </div>
  );
}
