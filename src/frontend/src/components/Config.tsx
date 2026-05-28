import { useCallback, useEffect, useState } from 'react';
import { getConfig } from '../api';

interface ConfigShape {
  server: Record<string, unknown>;
  storage: Record<string, unknown>;
  runtime: Record<string, unknown>;
  ui: Record<string, unknown>;
  queue: Record<string, unknown>;
  pull: Record<string, unknown>;
}

export default function Config() {
  const [cfg, setCfg] = useState<ConfigShape | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [editText, setEditText] = useState('');
  const [editError, setEditError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const raw = await getConfig();
      const shaped = raw as unknown as ConfigShape;
      setCfg(shaped);
      setEditText(
        JSON.stringify({ queue: shaped.queue, pull: shaped.pull }, null, 2),
      );
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runPut = useCallback(async () => {
    setEditError(null);
    setStatus(null);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(editText);
    } catch (e) {
      setEditError(`JSON parse error: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    setBusy(true);
    try {
      const r = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      const text = await r.text();
      setStatus(`PUT HTTP ${r.status}: ${text.slice(0, 400)}`);
      if (r.ok) await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }, [editText, refresh]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Boot config (read-only)</h2>
        <div className="text-sm text-slate-500 mb-3">
          Boot fields are immutable at runtime — PUT /api/config returns HTTP 403 if you try to
          mutate them. They are baked in from /etc/turbohaul/turbohaul.yaml + env overrides.
        </div>
        {error && <div className="text-amber-400 text-sm mb-3">⚠ {error.message}</div>}
        {cfg && (
          <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
            <ReadOnlySection title="server" data={cfg.server} />
            <ReadOnlySection title="storage" data={cfg.storage} />
            <ReadOnlySection title="runtime (binary + port_base)" data={cfg.runtime} />
            <ReadOnlySection title="ui" data={cfg.ui} />
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Runtime config (editable)</h2>
        <div className="text-sm text-slate-500 mb-3">
          Editing queue + pull settings. Submit JSON with the keys you want to change. Only
          runtime fields are accepted; including boot fields will return 403.
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            spellCheck={false}
            className="w-full h-96 rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
          />
          {editError && (
            <div className="mt-2 text-sm text-rose-400">⚠ {editError}</div>
          )}
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => void runPut()}
              disabled={busy}
              className="px-4 py-2 rounded-md bg-emerald-700 text-white text-sm font-medium hover:bg-emerald-600 disabled:bg-slate-700"
            >
              {busy ? 'Saving…' : 'Apply'}
            </button>
            <button
              onClick={() => void refresh()}
              disabled={busy}
              className="px-3 py-2 rounded-md bg-slate-800 text-slate-300 text-sm hover:bg-slate-700"
            >
              Reload (discard edits)
            </button>
          </div>
        </div>
      </div>

      {status && (
        <pre className="rounded-lg border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 whitespace-pre-wrap break-all">
{status}
        </pre>
      )}

      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400">
        <strong className="text-slate-300">Per-model manifest editor:</strong> deferred to a
        polish wave. /api/manifests CRUD + ETag/If-Match concurrency are wired on the BE;
        W20+ adds a model-picker UI that consumes them.
      </div>
    </div>
  );
}

function ReadOnlySection({ title, data }: { title: string; data: Record<string, unknown> }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">{title}</div>
      <pre className="text-xs font-mono text-slate-400 bg-slate-900 rounded-md p-2 overflow-x-auto">
{JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
