import { useEffect, useMemo, useState } from 'react';
import { useLogging, MAX_ACCUMULATED_EVENTS, type LogEvent } from '../hooks/useLogging';

const REDACTED_KEYS = ['prompt', 'response', 'context', 'stderr', 'stdout', 'messages'];
const INPUT_CLS = 'block mt-1 rounded bg-slate-900 border border-slate-700 px-2 py-1 text-xs text-slate-200 font-mono';
const BTN_CLS = 'rounded bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs text-slate-200 disabled:opacity-50';

function categorize(t: string): string {
  if (/_fail|_error|loading_fail|force_cold/.test(t)) return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
  if (/^safety_gate|slot_evicted/.test(t)) return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
  if (/^background_|^audit_/.test(t)) return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
  if (/^slot_|^submit|^stage_|^active|^grace|^idle|^teardown|^boot_/.test(t)) return 'bg-sky-500/20 text-sky-300 border-sky-500/40';
  return 'bg-slate-500/20 text-slate-300 border-slate-500/40';
}

function fmtUtc(iso: string): string {
  try { return new Date(iso).toISOString().replace('T', ' ').replace(/\.\d+Z$/, 'Z'); }
  catch { return iso; }
}

function EventRow({ ev }: { ev: LogEvent }) {
  return (
    <details className="rounded border border-slate-800 bg-slate-950 p-2">
      <summary className="flex flex-wrap items-center gap-2 cursor-pointer text-xs">
        <span className="font-mono text-slate-400" title={ev.occurred_at}>{fmtUtc(ev.occurred_at)}</span>
        <span className={`px-1.5 py-0.5 rounded border text-[10px] font-medium ${categorize(ev.event_type)}`}>{ev.event_type}</span>
        <span className="font-mono text-slate-300">{ev.slot_id ?? <em className="text-slate-500">system</em>}</span>
        <span className="ml-auto text-[10px] text-slate-600">#{ev.event_id}</span>
      </summary>
      <pre className="mt-2 text-[11px] font-mono bg-slate-900 rounded p-2 overflow-x-auto text-slate-300 whitespace-pre-wrap">{JSON.stringify(ev.payload, null, 2)}</pre>
    </details>
  );
}

export default function Logs() {
  const [slotInput, setSlotInput] = useState('');
  const [typeInput, setTypeInput] = useState('');
  const [limitInput, setLimitInput] = useState(100);
  const [debounced, setDebounced] = useState({ slot_id: '', event_type: '', limit: 100 });

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced({ slot_id: slotInput, event_type: typeInput, limit: limitInput }), 300);
    return () => window.clearTimeout(t);
  }, [slotInput, typeInput, limitInput]);

  const filters = useMemo(() => ({ slot_id: debounced.slot_id, event_type: debounced.event_type, limit: debounced.limit }),
    [debounced.slot_id, debounced.event_type, debounced.limit]);
  const { events, loading, error, hasMore, oversized, loadMore, refresh } = useLogging(filters);
  const atCap = events.length >= MAX_ACCUMULATED_EVENTS;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-200">Logs</h2>
      <div className="rounded border border-slate-700 bg-slate-900/60 p-3 text-xs text-slate-400">
        <span className="text-slate-300 font-medium">Server-side redaction:</span> keys <span className="font-mono text-slate-300">{`{${REDACTED_KEYS.join(', ')}}`}</span> are stripped before transmission per ARCHITECTURE.md §11.3 — these fields will never appear in payload below.
      </div>
      <div className="flex flex-wrap items-end gap-3 rounded border border-slate-700 bg-slate-950 p-3">
        <label className="text-xs text-slate-400">slot_id
          <input type="text" value={slotInput} onChange={(e) => setSlotInput(e.target.value)} placeholder="exact match" className={`${INPUT_CLS} w-48`} />
        </label>
        <label className="text-xs text-slate-400">event_type
          <input type="text" value={typeInput} onChange={(e) => setTypeInput(e.target.value)} placeholder="exact match" className={`${INPUT_CLS} w-48`} />
        </label>
        <label className="text-xs text-slate-400">limit
          <input type="number" value={limitInput} onChange={(e) => setLimitInput(Math.min(500, Math.max(1, Number(e.target.value) || 1)))} min={1} max={500} className={`${INPUT_CLS} w-24`} />
        </label>
        <button type="button" onClick={refresh} disabled={loading} className={BTN_CLS}>{loading ? 'Loading…' : 'Refresh'}</button>
      </div>
      {oversized && <div className="rounded border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">This page contained one event larger than the 79KB budget. Tighten filters or expect partial paging on subsequent pages.</div>}
      {error && <div className="rounded border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-200">/v1/logging failed: {error.message}</div>}
      {!loading && events.length === 0 && !error && <div className="rounded border border-slate-700 bg-slate-950 p-4 text-sm text-slate-500">No events match filters.</div>}
      <div className="space-y-1">{events.map((ev) => <EventRow key={ev.event_id} ev={ev} />)}</div>
      {events.length > 0 && (
        <div className="pt-2">
          {atCap ? <div className="text-xs text-slate-500">Showing latest {MAX_ACCUMULATED_EVENTS} — refresh with tighter filter to load older</div>
            : hasMore ? <button type="button" onClick={loadMore} disabled={loading} className={BTN_CLS}>{loading ? 'Loading…' : 'Load more'}</button>
            : <div className="text-xs text-slate-500">End of audit log.</div>}
        </div>
      )}
    </div>
  );
}
