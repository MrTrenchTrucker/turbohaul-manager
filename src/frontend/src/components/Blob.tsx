import { useCallback, useEffect, useState } from 'react';
import type { ModelTag } from '../api';
import { getTags } from '../api';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

type PullSource = 'url' | 'hf' | 'import';

async function postJSON(path: string, body: unknown): Promise<Response> {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function deleteJSON(path: string, body: unknown): Promise<Response> {
  return fetch(path, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export default function Blob() {
  const [tags, setTags] = useState<ModelTag[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const [source, setSource] = useState<PullSource>('url');
  const [pullUrl, setPullUrl] = useState('');
  const [pullSha, setPullSha] = useState('');
  const [pullTag, setPullTag] = useState('');
  const [hfRepo, setHfRepo] = useState('');
  const [hfFile, setHfFile] = useState('');
  const [hfTag, setHfTag] = useState('');
  const [importPath, setImportPath] = useState('');
  const [importTag, setImportTag] = useState('');

  const refresh = useCallback(async () => {
    try {
      const r = await getTags();
      setTags(r.models);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runPull = useCallback(async () => {
    setBusy(true);
    setStatus(null);
    try {
      let r: Response;
      if (source === 'url') {
        r = await postJSON('/api/pull-url', {
          url: pullUrl,
          expected_sha256: pullSha || undefined,
          tag: pullTag || undefined,
        });
      } else if (source === 'hf') {
        r = await postJSON('/api/pull-hf', {
          repo: hfRepo,
          file: hfFile,
          tag: hfTag || undefined,
        });
      } else {
        r = await postJSON('/api/import', {
          path: importPath,
          tag: importTag || undefined,
        });
      }
      const text = await r.text();
      setStatus(`HTTP ${r.status}: ${text.slice(0, 300)}`);
      if (r.ok) await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }, [source, pullUrl, pullSha, pullTag, hfRepo, hfFile, hfTag, importPath, importTag, refresh]);

  const runDelete = useCallback(
    async (digest: string) => {
      if (!window.confirm(`Delete blob ${digest.slice(0, 16)}…?`)) return;
      setBusy(true);
      try {
        const r = await deleteJSON('/api/delete', { digest });
        setStatus(`DELETE HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
        await refresh();
      } catch (e) {
        setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-slate-200">Installed models</h2>
          <button
            onClick={() => void refresh()}
            className="px-3 py-1 rounded-md bg-slate-800 text-slate-300 text-sm hover:bg-slate-700"
            disabled={busy}
          >
            Refresh
          </button>
        </div>
        {error && (
          <div className="text-amber-400 text-sm mb-3">
            ⚠ {error.message}
          </div>
        )}
        {tags === null ? (
          <div className="text-slate-500 text-sm italic">Loading…</div>
        ) : tags.length === 0 ? (
          <div className="text-slate-500 text-sm italic">No models installed yet. Pull one below.</div>
        ) : (
          <div className="rounded-lg border border-slate-700 bg-slate-950 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-xs uppercase text-slate-500">
                <tr>
                  <th className="text-left px-4 py-2">Name</th>
                  <th className="text-right px-4 py-2">Size</th>
                  <th className="text-left px-4 py-2">Digest</th>
                  <th className="text-left px-4 py-2">Modified</th>
                  <th className="text-right px-4 py-2"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {tags.map((m) => (
                  <tr key={m.digest} className="text-slate-300">
                    <td className="px-4 py-2 font-mono">{m.name}</td>
                    <td className="px-4 py-2 font-mono text-right">{formatBytes(m.size)}</td>
                    <td className="px-4 py-2 font-mono text-slate-500">
                      {m.digest.slice(0, 16)}…
                    </td>
                    <td className="px-4 py-2 text-slate-500">{m.modified_at ?? '—'}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        className="px-2 py-1 rounded-md bg-rose-900/40 text-rose-300 text-xs hover:bg-rose-900/60"
                        disabled={busy}
                        onClick={() => void runDelete(m.digest)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Pull model</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
          <div className="flex gap-2 mb-4">
            {(['url', 'hf', 'import'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSource(s)}
                className={
                  source === s
                    ? 'px-3 py-1 rounded-md bg-emerald-700 text-white text-sm'
                    : 'px-3 py-1 rounded-md bg-slate-800 text-slate-400 text-sm hover:bg-slate-700'
                }
              >
                {s === 'url' ? 'URL' : s === 'hf' ? 'HuggingFace' : 'Local import'}
              </button>
            ))}
          </div>

          {source === 'url' && (
            <div className="space-y-2">
              <Field label="URL (https only)" value={pullUrl} onChange={setPullUrl} placeholder="https://..." />
              <Field label="Expected sha256 (optional)" value={pullSha} onChange={setPullSha} placeholder="hex 64 chars" />
              <Field label="Tag (optional)" value={pullTag} onChange={setPullTag} placeholder="e.g. my-model:8b" />
            </div>
          )}
          {source === 'hf' && (
            <div className="space-y-2">
              <Field label="HF repo" value={hfRepo} onChange={setHfRepo} placeholder="owner/repo" />
              <Field label="File in repo" value={hfFile} onChange={setHfFile} placeholder="model-q4.gguf" />
              <Field label="Tag (optional)" value={hfTag} onChange={setHfTag} placeholder="e.g. my-model:8b" />
            </div>
          )}
          {source === 'import' && (
            <div className="space-y-2">
              <Field label="Absolute path (must be under import_allowed_root)" value={importPath} onChange={setImportPath} placeholder="/var/lib/turbohaul/import-staging/foo.gguf" />
              <Field label="Tag (optional)" value={importTag} onChange={setImportTag} placeholder="e.g. my-model:8b" />
            </div>
          )}

          <div className="mt-4">
            <button
              onClick={() => void runPull()}
              disabled={busy}
              className="px-4 py-2 rounded-md bg-emerald-700 text-white text-sm font-medium hover:bg-emerald-600 disabled:bg-slate-700"
            >
              {busy ? 'Working…' : `Run ${source}`}
            </button>
          </div>
        </div>
      </div>

      {status && (
        <pre className="rounded-lg border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 whitespace-pre-wrap break-all">
{status}
        </pre>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="block text-xs text-slate-400 mb-1">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-1.5 text-sm font-mono text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-600"
      />
    </label>
  );
}
