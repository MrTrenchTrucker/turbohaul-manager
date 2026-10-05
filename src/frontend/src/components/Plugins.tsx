import { useEffect, useState } from 'react';
import { getConfig, getConfigSchema, getPlugins, putConfig } from '../api';
import type { ConfigSchema, PluginListEntry } from '../api';

// The Plugins tab (Media Hook). Structure/classes are copied from
// FastLane.tsx rather than invented.

export type ConfiguredIndicator = { text: string; tone: 'ok' | 'pending' };

// `configured: false` means the manifest's resource_key isn't in the boot
// registry -- which is genuinely EITHER a typo OR a half-finished setup, and
// the data alone can't tell them apart. Rendered as amber "not configured
// yet", never red/error: the backend's own design already treats
// a plugin manifest existing before its registry entry as a legitimate,
// expected state (rejecting it at write time would make first-time setup a
// chicken-and-egg problem). The UI must not paint a
// fault the backend itself refuses to assert.
//
// The true branch says 'Configured', not 'OK', because 'OK' would overclaim.
// The backend computes this from a registry
// lookup that NEVER OPENS A SOCKET, so 'OK' -- the word an operator actually
// reads as "this works" -- would be wrong. 'Configured' says exactly
// what was established and nothing more. Reachability is only ever learned by
// invoking (502 `unreachable`), never by this listing.
export function formatConfiguredIndicator(configured: boolean): ConfiguredIndicator {
  return configured
    ? { text: 'Configured', tone: 'ok' }
    : { text: 'Not configured yet', tone: 'pending' };
}

// The KEY-READ lives HERE, in a pure exported function,
// rather than inline at the two JSX call sites -- deliberately.
//
// This runner has no DOM (see Plugins.test.tsx's header), so an inline
// `p.configured` inside JSX is unreachable by any test. And an untestable
// key-read is EXACTLY how a renamed wire key could ship green while every
// badge silently flipped to amber: JSON is untyped at runtime, so reading a
// key the backend no longer emits yields `undefined`, takes the falsy branch,
// and throws nothing. Putting the read in a covered pure function is what
// makes "wire renamed, FE not" a test failure.
export function indicatorForEntry(entry: PluginListEntry): ConfiguredIndicator {
  return formatConfiguredIndicator(entry.configured);
}

// Pure, immutable map update -- keyed by model_tag
// (resource_key is NOT unique per manifest, two plugin manifests
// can share one boot-registry entry, so keying by resource_key would make
// disabling one silently disable the other too. model_tag is unique by
// construction -- "should THIS plugin run" is a manifest-level policy
// decision, independent of "where does it physically connect").
export function toggleEnabled(
  enabled: Record<string, boolean>,
  modelTag: string,
  value: boolean,
): Record<string, boolean> {
  return { ...enabled, [modelTag]: value };
}

export type RuntimeKnobsResult =
  | { valid: true; payload: { max_concurrent: number; no_progress_timeout_s: number } }
  | { valid: false; error: string };

// Pure so it's testable with real dynamic assertions, no DOM needed -- same
// extraction pattern as Settings.tsx's validateSchedulingFields. Bounds come
// from the schema when available and are simply skipped when not: the server
// remains the bounds authority regardless of what the client can check up
// front (same degrade-gracefully contract as every other schema-driven field
// in this app).
export function validatePluginRuntimeFields(
  maxConcurrentInput: string,
  noProgressTimeoutSInput: string,
  pluginRuntimeSchema: ConfigSchema[string] | undefined,
): RuntimeKnobsResult {
  const maxConcurrent = parseInt(maxConcurrentInput, 10);
  if (Number.isNaN(maxConcurrent)) {
    return { valid: false, error: 'Max concurrent must be a whole number' };
  }
  const maxConcurrentBounds = pluginRuntimeSchema?.max_concurrent;
  if (maxConcurrentBounds && maxConcurrentBounds.minimum != null && maxConcurrent < maxConcurrentBounds.minimum) {
    return { valid: false, error: `Max concurrent must be >= ${maxConcurrentBounds.minimum}` };
  }
  if (maxConcurrentBounds && maxConcurrentBounds.maximum != null && maxConcurrent > maxConcurrentBounds.maximum) {
    return { valid: false, error: `Max concurrent must be <= ${maxConcurrentBounds.maximum}` };
  }

  const noProgressTimeoutS = parseFloat(noProgressTimeoutSInput);
  if (Number.isNaN(noProgressTimeoutS)) {
    return { valid: false, error: 'No-progress timeout must be a number' };
  }
  const timeoutBounds = pluginRuntimeSchema?.no_progress_timeout_s;
  if (timeoutBounds && timeoutBounds.minimum != null && noProgressTimeoutS < timeoutBounds.minimum) {
    return { valid: false, error: `No-progress timeout must be >= ${timeoutBounds.minimum}` };
  }
  if (timeoutBounds && timeoutBounds.maximum != null && noProgressTimeoutS > timeoutBounds.maximum) {
    return { valid: false, error: `No-progress timeout must be <= ${timeoutBounds.maximum}` };
  }

  return {
    valid: true,
    payload: { max_concurrent: maxConcurrent, no_progress_timeout_s: noProgressTimeoutS },
  };
}

const INDICATOR_CLS: Record<ConfiguredIndicator['tone'], string> = {
  ok: 'bg-emerald-950/40 text-emerald-400',
  pending: 'bg-amber-950/40 text-amber-400',
};

export default function Plugins() {
  const [plugins, setPlugins] = useState<PluginListEntry[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [maxConcurrentInput, setMaxConcurrentInput] = useState('4');
  const [noProgressTimeoutSInput, setNoProgressTimeoutSInput] = useState('600');
  const [schema, setSchema] = useState<ConfigSchema | undefined>(undefined);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getPlugins()
      .then((r) => {
        if (cancelled) return;
        setPlugins(r.plugins);
        setLoaded(true);
      })
      .catch((e) => {
        if (cancelled) return;
        setListError(e instanceof Error ? e.message : String(e));
        setLoaded(true);
      });
    getConfig()
      .then((c) => {
        if (cancelled) return;
        const pr = (c.plugin_runtime as Record<string, unknown>) || {};
        setEnabled((pr.enabled as Record<string, boolean>) || {});
        setMaxConcurrentInput(String(pr.max_concurrent ?? 4));
        setNoProgressTimeoutSInput(String(pr.no_progress_timeout_s ?? 600));
      })
      .catch(console.error);
    getConfigSchema()
      .then((s) => {
        if (!cancelled) setSchema(s);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  // Whole-section save: config_put.py's runtime merge is shallow, one level
  // -- plugin_runtime is replaced wholesale on every PUT, never
  // field-merged (same contract FastLane.tsx's saveRules documents for
  // `fastlane.rules`). Every save sends the full {enabled, max_concurrent,
  // no_progress_timeout_s} object, never a partial one, or an untouched
  // field would silently revert to its Pydantic default.
  const saveRuntime = async (
    nextEnabled: Record<string, boolean>,
    maxConcurrentStr: string,
    timeoutStr: string,
  ) => {
    const result = validatePluginRuntimeFields(maxConcurrentStr, timeoutStr, schema?.plugin_runtime);
    if (!result.valid) {
      setSaveMsg({ type: 'error', text: result.error });
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      await putConfig({
        plugin_runtime: {
          enabled: nextEnabled,
          max_concurrent: result.payload.max_concurrent,
          no_progress_timeout_s: result.payload.no_progress_timeout_s,
        },
      });
      setEnabled(nextEnabled);
      setSaveMsg({ type: 'success', text: 'Saved' });
    } catch (e) {
      setSaveMsg({ type: 'error', text: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSaving(false);
    }
  };

  const onToggleEnabled = (modelTag: string, value: boolean) => {
    const next = toggleEnabled(enabled, modelTag, value);
    setEnabled(next); // optimistic; saveRuntime below reconciles/reports failure
    void saveRuntime(next, maxConcurrentInput, noProgressTimeoutSInput);
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-amber-700/50 bg-amber-950/30 p-3 text-sm text-amber-300">
        Work in progress — feedback and contributions welcome
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-300">
        Resource plugins route agent requests (transcription, OCR, TTS, image generation) to
        operator-declared external containers — Turbohaul ships no media tooling itself. This tab
        lists what is configured and lets you enable or disable each one and tune the shared
        runtime knobs.
      </div>

      {listError && (
        <div className="rounded-lg border border-rose-700/50 bg-rose-950/30 p-4 text-sm text-rose-300">
          Could not load plugins: {listError}
        </div>
      )}

      {saveMsg && (
        <div className={`text-sm ${saveMsg.type === 'success' ? 'text-emerald-400' : 'text-amber-400'}`}>
          {saveMsg.text}
        </div>
      )}

      {loaded && !listError && plugins.length === 0 && (
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-500 italic">
          No plugins configured yet. An operator adds one by declaring a boot-registry entry (host,
          port, health check) and saving a plugin manifest whose resource_key names it.
        </div>
      )}

      {plugins.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-slate-200 mb-4">Plugins</h2>

          {/* Mobile card layout (< md), matching FastLane.tsx's convention. */}
          <div className="md:hidden space-y-3">
            {plugins.map((p) => {
              const indicator = indicatorForEntry(p);
              return (
                <div key={p.model_tag} className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="font-mono text-slate-200 break-all">{p.model_tag}</div>
                      <div className="text-xs text-slate-500 mt-1">{p.lane.toUpperCase()} lane</div>
                    </div>
                    <span className={`text-xs shrink-0 rounded px-2 py-1 ${INDICATOR_CLS[indicator.tone]}`}>
                      {indicator.text}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">
                    {p.capabilities.length > 0 ? p.capabilities.join(' · ') : '—'}
                  </div>
                  <label className="flex items-center gap-2 min-h-[44px]">
                    <input
                      type="checkbox"
                      checked={enabled[p.model_tag] ?? true}
                      onChange={(e) => onToggleEnabled(p.model_tag, e.target.checked)}
                      disabled={saving}
                      className="h-4 w-4 accent-emerald-500"
                    />
                    <span className="text-sm text-slate-300">Enabled</span>
                  </label>
                </div>
              );
            })}
          </div>

          {/* Desktop table (>= md). */}
          <div className="hidden md:block rounded-lg border border-slate-700 bg-slate-950 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-xs uppercase text-slate-500">
                <tr>
                  <th className="text-left px-4 py-2">Model tag</th>
                  <th className="text-left px-4 py-2">Lane</th>
                  <th className="text-left px-4 py-2">Capabilities</th>
                  <th className="text-left px-4 py-2">Status</th>
                  <th className="text-left px-4 py-2">Enabled</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {plugins.map((p) => {
                  const indicator = indicatorForEntry(p);
                  return (
                    <tr key={p.model_tag} className="text-slate-300">
                      <td className="px-4 py-2 font-mono">{p.model_tag}</td>
                      <td className="px-4 py-2 text-xs uppercase">{p.lane}</td>
                      <td className="px-4 py-2 text-slate-500">
                        {p.capabilities.length > 0 ? p.capabilities.join(', ') : '—'}
                      </td>
                      <td className="px-4 py-2">
                        <span className={`text-xs rounded px-2 py-1 ${INDICATOR_CLS[indicator.tone]}`}>
                          {indicator.text}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        <input
                          type="checkbox"
                          checked={enabled[p.model_tag] ?? true}
                          onChange={(e) => onToggleEnabled(p.model_tag, e.target.checked)}
                          disabled={saving}
                          className="h-4 w-4 accent-emerald-500"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Runtime settings</h2>
        <p className="text-xs text-slate-500 mb-3">
          Applies to every plugin. The boot registry (where each resource_key connects to) is
          operator-configured at boot and is not editable here.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md">
          <label className="block">
            <span className="block text-xs uppercase tracking-wide text-slate-500 mb-1">Max concurrent</span>
            <input
              type="number"
              value={maxConcurrentInput}
              onChange={(e) => setMaxConcurrentInput(e.target.value)}
              onBlur={() => void saveRuntime(enabled, maxConcurrentInput, noProgressTimeoutSInput)}
              className="w-full h-11 bg-slate-900 border border-slate-700 rounded px-3 text-sm text-slate-200"
            />
          </label>
          <label className="block">
            <span className="block text-xs uppercase tracking-wide text-slate-500 mb-1">
              No-progress timeout (s)
            </span>
            <input
              type="number"
              value={noProgressTimeoutSInput}
              onChange={(e) => setNoProgressTimeoutSInput(e.target.value)}
              onBlur={() => void saveRuntime(enabled, maxConcurrentInput, noProgressTimeoutSInput)}
              className="w-full h-11 bg-slate-900 border border-slate-700 rounded px-3 text-sm text-slate-200"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
