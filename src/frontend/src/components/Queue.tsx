import { useStatus } from '../hooks/useStatus';

export default function Queue() {
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
          <div className="flex justify-between">
            <span className="text-slate-400">Grace model</span>
            <span className="font-mono text-slate-200">{grace?.model_tag ?? '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Grace remaining</span>
            <span className="font-mono text-slate-200">
              {grace ? `${grace.remaining_s}s` : '—'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Idle hot-load</span>
            <span className="font-mono text-slate-200">
              {idle_hot ? `${idle_hot.model_tag} (${idle_hot.remaining_s}s)` : '— cold —'}
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        <strong className="text-slate-300">Per-slot detail:</strong> the FIFO list of
        staged thread ids is not yet exposed by /status; W19 will add /api/queue with
        per-slot rows (position, model_tag, thread_id_prefix, state, ETA).
      </div>

      <div className="text-xs text-slate-500">
        last update: <span className="font-mono">{lastUpdate?.toISOString() ?? '—'}</span>
      </div>
    </div>
  );
}
