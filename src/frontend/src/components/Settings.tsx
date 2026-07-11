import { useEffect, useState } from 'react';
import type { VersionInfo, StatusSnapshot } from '../api';
import { getVersion, getConfig, putConfig, getStatus } from '../api';

export default function Settings() {
  const [ver, setVer] = useState<VersionInfo | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [persistMaxGiB, setPersistMaxGiB] = useState<string>('40');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [persistKV, setPersistKV] = useState<StatusSnapshot['persist_kvcache'] | null>(null);

  const GiB = 1024 ** 3;

  useEffect(() => {
    let cancelled = false;
    getVersion()
      .then((v) => {
        if (!cancelled) setVer(v);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e : new Error(String(e)));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((c) => {
        if (!cancelled) {
          setConfig(c);
          const persist = (c.persist as Record<string, unknown>) || {};
          const maxBytes = (persist.max_bytes as number) || 42949672960;
          setPersistMaxGiB(String(Math.round(maxBytes / GiB)));
        }
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getStatus()
      .then((s) => {
        if (!cancelled && s.persist_kvcache) {
          setPersistKV(s.persist_kvcache);
        }
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  const handlePersistSave = async () => {
    const val = parseInt(persistMaxGiB, 10);
    if (isNaN(val) || val < 0) {
      setSaveMsg({ type: 'error', text: 'Invalid value — must be a non-negative integer (GiB)' });
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      await putConfig({ persist: { max_bytes: val * GiB } });
      setSaveMsg({ type: 'success', text: `Saved: ${val} GiB — applies immediately; reverts to the configured default on manager restart` });
      // Refresh config to show actual value
      const c = await getConfig();
      setConfig(c);
      const persist = (c.persist as Record<string, unknown>) || {};
      const maxBytes = (persist.max_bytes as number) || 42949672960;
      setPersistMaxGiB(String(Math.round(maxBytes / GiB)));
    } catch (e) {
      setSaveMsg({ type: 'error', text: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSaving(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes >= GiB) return `${(bytes / GiB).toFixed(2)} GiB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KiB`;
    return `${bytes} B`;
  };

  const getCurrentUsage = () => {
    if (!persistKV) return '—';
    return formatBytes(persistKV.total_bytes);
  };

  const getHeadroom = () => {
    if (!persistKV) return '—';
    return formatBytes(persistKV.headroom_bytes);
  };

  const getOverCapIndicator = () => {
    if (!persistKV) return null;
    if (persistKV.over_cap) {
      return <span className="text-amber-400 ml-2">⚠ OVER CAP</span>;
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">About</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-3 text-sm">
          {error && <div className="text-amber-400">⚠ {error.message}</div>}
          {ver ? (
            <>
              <Row k="version" v={ver.version} />
              <Row k="backend" v={ver.backend} />
              <Row k="backend SHA pinned" v={String(ver.backend_sha_pinned)} />
              <Row k="api compat" v={ver.api_compat} />
              <Row k="user-agent" v={ver.user_agent} />
            </>
          ) : (
            <div className="text-slate-500 italic">Loading…</div>
          )}
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Persist KV Cache (SSD)</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-4">
          <div className="space-y-2">
            <label className="block text-sm text-slate-300">
              Maximum SSD footprint for persisted KV caches
            </label>
            <div className="flex items-center gap-4 flex-wrap">
              <input
                type="number"
                min="0"
                step="1"
                value={persistMaxGiB}
                onChange={(e) => setPersistMaxGiB(e.target.value)}
                className="w-24 px-3 py-2 rounded bg-slate-800 border border-slate-600 text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                disabled={saving}
              />
              <span className="text-slate-400 font-mono text-sm">GiB</span>
              <button
                onClick={handlePersistSave}
                disabled={saving}
                className="px-4 py-2 rounded bg-emerald-600 text-emerald-50 text-sm font-medium hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
              {saveMsg && (
                <span className={`text-sm ${saveMsg.type === 'success' ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {saveMsg.text}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500">
              Cap applies to the <code className="font-mono text-slate-400">SLOT_PERSIST_DIR</code> archive
              (sum of <code className="font-mono text-slate-400">.bin</code> files). Oldest triplets evicted first when over cap.
              Set to <code className="font-mono text-slate-400">0</code> to disable ceiling (age/count GC still runs).
            </p>
          </div>

          <div className="pt-4 border-t border-slate-700 space-y-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-slate-400">Configured cap:</span>
              <span className="font-mono text-slate-200 text-right truncate">
                {config && config.persist
                  ? formatBytes((config.persist as Record<string, unknown>).max_bytes as number)
                  : '—'}
              </span>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-slate-400">Current SSD usage:</span>
              <span className="font-mono text-slate-200 text-right truncate">
                {getCurrentUsage()}
                {getOverCapIndicator()}
              </span>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-slate-400">Headroom:</span>
              <span className="font-mono text-slate-200 text-right truncate">
                {getHeadroom()}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Licenses + attribution</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400 space-y-2">
          <p>
            Turbohaul-Manager v0.2 — MIT-licensed wrapper around the inference engine.
          </p>
          <p>
            Inference backend: <span className="font-mono text-slate-300">llama-server</span>{' '}
            built from Tom&apos;s TurboQuant fork of llama.cpp (MIT).
          </p>
          <p>
            See <span className="font-mono text-slate-300">THIRD_PARTY_LICENSES</span> in the
            container at <span className="font-mono text-slate-300">/usr/share/doc/turbohaul/</span>.
          </p>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-slate-400">{k}</span>
      <span className="font-mono text-slate-200 text-right truncate">{v}</span>
    </div>
  );
}
