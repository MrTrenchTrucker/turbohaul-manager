import { Navigate, useLocation } from 'react-router-dom';
import type { QueueWaitingRow } from '../api';
import { useStatus } from '../hooks/useStatus';
import { SubTabs } from './SubTabs';
import FastLane from './FastLane';
import { WaitingCard } from './queue/waiting';
import { QueueClaimRows } from './queue/claims';
import { RequestIdentityStrip } from './Dashboard';

// Exported and prop-driven -- same reasoning
// WaitingCard above already documents: inline in the page
// body this could not be rendered by the static harness; exporting it lets
// a render test cover it.
export function QueueWaitingRows({ waiting }: { waiting?: QueueWaitingRow[] }) {
  if (waiting === undefined) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        This manager does not report the waiting-request surface.
      </div>
    );
  }
  if (waiting.length === 0) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        Nothing waiting.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-900 text-xs uppercase text-slate-500">
          <tr>
            <th className="text-left px-4 py-2">Client</th>
            <th className="text-left px-4 py-2">Tag / rank</th>
            <th className="text-left px-4 py-2">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {waiting.map((row) => {
            const client = row.fastlane?.label || row.thread_id_prefix;
            const tagRank = row.fastlane ? `${row.fastlane.rank}` : '—';
            const status =
              row.likely_victim !== null
                ? `waiting for a turn to complete (likely victim: ${row.likely_victim})`
                : 'waiting for a turn to complete';
            return (
              <tr key={row.slot_id} className="text-slate-300" data-testid="queue-waiting-row">
                <td className="px-4 py-2 font-mono">{client}</td>
                <td className="px-4 py-2 font-mono text-center">{tagRank}</td>
                <td className="px-4 py-2 text-slate-400">
                  {status}
                  <span className="text-slate-500"> ({row.state})</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Re-exported so existing importers (and Queue.test.tsx) keep working while
// the definition itself lives in one shared place -- see ./queue/waiting.
export { WaitingCard, waitingCount } from './queue/waiting';

// The subtab key that used to be 'fastline' (route
// /queue/fastline) is now 'fastlane' (/queue/fastlane) -- an existing
// bookmark/history entry to the old URL would otherwise silently soft-land
// on the Overview tab (SubTabs.tsx's resolveActiveKey falls back to
// defaultKey for any unrecognized segment, it does not error), with nothing
// telling the person they didn't land where they meant to. Pure and
// exported so the match itself is directly testable without a routing context.
export function isLegacyFastLaneQueuePath(pathname: string): boolean {
  return pathname === '/queue/fastline' || pathname.startsWith('/queue/fastline/');
}

function QueueOverview() {
  const { data, error, lastUpdate } = useStatus();

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

  const { queue, parallel_slots, active, grace, idle_hot } = data;
  const stagedPct =
    queue.staging_queue_max > 0
      ? Math.min(100, (queue.staging_queue_depth / queue.staging_queue_max) * 100)
      : 0;

  // Grace row — "held during serve — re-arms after turn"
  // when an ACTIVE serve is holding the engine (backend suppresses the
  // countdown during serve so FE never shows a mid-prefill 'unload in Ns' clock).
  const showGraceHeld = active && active.state === 'ACTIVE' && grace;
  const graceDisplay = showGraceHeld
    ? 'held during serve — re-arms after turn'
    : grace
      ? `${grace.remaining_s}s (ext ${grace.extension_count}/${grace.max_extensions})`
      : '—';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Queue state</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              Staging queue
            </div>
            <div className="text-2xl font-bold text-slate-200 mb-3">
              {queue.staging_queue_depth}
              <span className="text-base font-normal text-slate-500"> / {queue.staging_queue_max}</span>
            </div>
            <div className="h-2 bg-slate-800 rounded overflow-hidden">
              <div
                className="h-full bg-amber-500 transition-all"
                style={{ width: `${stagedPct}%` }}
              />
            </div>
          </div>
          <WaitingCard queue={queue} />
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              Acceptance buffer
            </div>
            <div className="text-2xl font-bold text-slate-200">
              {queue.acceptance_buffer_depth}
            </div>
            <div className="text-xs text-slate-500 mt-2">
              In-flight requests being placed into staging (FIFO).
            </div>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Slot occupancy</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Parallel sidecars</span>
            <span className="font-mono text-slate-200">
              {parallel_slots.used} / {parallel_slots.max}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Active model</span>
            <span className="font-mono text-slate-200">
              {active?.model_tag ?? '—'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Active state</span>
            <span className="font-mono text-slate-200">{active?.state ?? '—'}</span>
          </div>
          <div className="mt-1">
            {/* CURRENT renders
                ABOVE last, so the row that matters is not buried under
                a FINISHED session that the label would otherwise show as if
                it were live. Unconditional, same as the per-resident strip
                on the Dashboard: null when idle renders the "idle"
                placeholder rather than being omitted, so a missing element
                is never ambiguous with a dropped one. */}
            <RequestIdentityStrip
              identity={data.current_request_identity}
              heading="current request"
              testId="resident-current-request-identity"
            />
          </div>
          {data.request_identity && (
            <div className="mt-1">
              {/* Who is holding the slot — same identity line
                  the Dashboard renders (ip · fastlane label · model · role ·
                  session). Label only when the operator saved one for that
                  IP; the strip renders nothing for it otherwise. */}
              <RequestIdentityStrip identity={data.request_identity} />
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-slate-400">Grace model</span>
            <span className="font-mono text-slate-200">{grace?.model_tag ?? '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Grace remaining</span>
            <span className="font-mono text-slate-200">{graceDisplay}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Idle hot-load</span>
            <span className="font-mono text-slate-200">
              {idle_hot ? `${idle_hot.model_tag} (${idle_hot.remaining_s}s)` : '— cold —'}
            </span>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Waiting requests</h2>
        <QueueWaitingRows waiting={queue.waiting} />
      </div>

      {/* A DIFFERENT population from the
          table above -- claims registered via _defer_unroutable that have not
          reached staging yet, so queue.waiting is blind to them. Its own
          labeled section on purpose: presenting it merged with Waiting
          requests would conflate the two. */}
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Fast Lane claims</h2>
        <QueueClaimRows claims={queue.fastlane_claims_snapshot} />
      </div>

      <div className="text-xs text-slate-500">
        last update: <span className="font-mono">{lastUpdate?.toISOString() ?? '—'}</span>
      </div>
    </div>
  );
}

// Queue is the SubTabs host for Queue (this tab's content, in
// QueueOverview) and Fast Lane. The host itself
// must always render (never gated behind /status loading/error) so the
// Fast Lane subtab is never hidden -- only QueueOverview owns that guard now.
export default function Queue() {
  const location = useLocation();
  if (isLegacyFastLaneQueuePath(location.pathname)) {
    return <Navigate to="/queue/fastlane" replace />;
  }

  return (
    <SubTabs
      basePath="/queue"
      defaultKey="overview"
      tabs={[
        { key: 'overview', label: 'Queue', element: <QueueOverview /> },
        { key: 'fastlane', label: 'Fast Lane', element: <FastLane /> },
      ]}
    />
  );
}
