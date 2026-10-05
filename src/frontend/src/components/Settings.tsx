import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import type { ConfigSchema, VersionInfo, StatusSnapshot } from '../api';
import { getVersion, getConfig, getConfigSchema, putConfig, getStatus } from '../api';
import { SubTabs } from './SubTabs';
import Config from './Config';
import Schema from './Schema';
import Logs from './Logs';

// Settings is the SubTabs host for General (GeneralSettings),
// Config, Schema and Logs.

// The 7 queue-section fields that share the "same
// policy" as Fast Lane's own scheduling scalars. Shown read-only in the
// curated card below — they are ALREADY live-editable via Settings ->
// Config -> queue (its schema-driven generic editor renders every queue
// field), so giving them a second independently-stateful editable copy
// here would go stale the moment someone edits one on Config and tabs
// back here without a page reload (SubTabs keeps every visited pane
// mounted forever, per its own module comment — GeneralSettings' config
// fetch runs once on mount and never refreshes).
export const QUEUE_SCHEDULING_FIELDS: { key: string; label: string }[] = [
  { key: 'grace_seconds', label: 'grace_seconds' },
  { key: 'max_grace_extensions', label: 'max_grace_extensions' },
  { key: 'idle_hot_load_seconds', label: 'idle_hot_load_seconds' },
  { key: 'max_consecutive_same_model', label: 'max_consecutive_same_model' },
  { key: 'max_other_model_wait_s', label: 'max_other_model_wait_s' },
  { key: 'main_lane_reserved', label: 'main_lane_reserved' },
  { key: 'staging_queue_depth', label: 'staging_queue_depth' },
];

export function formatQueueFieldValue(v: unknown): string {
  if (v === undefined || v === null) return '—';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

export type SchedulingFieldsResult =
  | { valid: true; payload: { max_normal_wait_s: number; cross_model_switches_per_min: number } }
  | { valid: false; error: string };

// Pure so it can be tested with real dynamic assertions (no DOM needed) —
// same extraction pattern as stepListbox. Bounds come from the
// schema when available and are simply skipped when not (schema fetch
// failed), matching Config.tsx's own degrade-gracefully contract: the
// server remains the bounds authority regardless of what the client can
// check up front.
export function validateSchedulingFields(
  maxNormalWaitSInput: string,
  crossModelSwitchesPerMinInput: string,
  fastlaneSchema: ConfigSchema[string] | undefined,
): SchedulingFieldsResult {
  const maxWait = parseFloat(maxNormalWaitSInput);
  if (Number.isNaN(maxWait)) {
    return { valid: false, error: 'Anti-starvation wait must be a number' };
  }
  const waitBounds = fastlaneSchema?.max_normal_wait_s;
  if (waitBounds && waitBounds.minimum != null && maxWait < waitBounds.minimum) {
    return { valid: false, error: `Anti-starvation wait must be >= ${waitBounds.minimum}` };
  }
  if (waitBounds && waitBounds.maximum != null && maxWait > waitBounds.maximum) {
    return { valid: false, error: `Anti-starvation wait must be <= ${waitBounds.maximum}` };
  }

  const switches = parseInt(crossModelSwitchesPerMinInput, 10);
  if (Number.isNaN(switches)) {
    return { valid: false, error: 'Cross-model switch rate must be a whole number' };
  }
  const switchBounds = fastlaneSchema?.cross_model_switches_per_min;
  if (switchBounds && switchBounds.minimum != null && switches < switchBounds.minimum) {
    return { valid: false, error: `Cross-model switch rate must be >= ${switchBounds.minimum}` };
  }
  if (switchBounds && switchBounds.maximum != null && switches > switchBounds.maximum) {
    return { valid: false, error: `Cross-model switch rate must be <= ${switchBounds.maximum}` };
  }

  return {
    valid: true,
    payload: { max_normal_wait_s: maxWait, cross_model_switches_per_min: switches },
  };
}

export interface FastLaneLoadState {
  configLoaded: boolean;
  configError: string | null;
}

/** Decide whether a GET /api/config response's
 * fastlane section may be certified as this deployment's real settings.
 *
 * Extracted (same reason validateSchedulingFields above was) so it is
 * testable without rendering GeneralSettings, which this file's own test
 * suite documents cannot be statically rendered here.
 *
 * A 200 response is not sufficient: the server drops the whole fastlane
 * section when it fails to load or fully salvage at boot, and serves CODE
 * DEFAULTS in its place. `fastlane.config_error` is how it says so
 * (FastLane.tsx already reads it for its own tab; this card did not).
 * Certifying those defaults as loaded is not cosmetic -- one Save PUTs them
 * as a deliberate operator choice, and persisted config always outranks the
 * yaml at every future boot, so it would mask the operator's real settings
 * PERMANENTLY, including after they fix whatever config_error names.
 */
export function computeFastLaneLoadState(
  fastlane: Record<string, unknown>,
): FastLaneLoadState {
  const configError = (fastlane.config_error as string | null | undefined) ?? null;
  if (configError) {
    return {
      configLoaded: false,
      configError:
        `Fast Lane settings could not be loaded -- the server is showing ` +
        `CODE DEFAULTS, not this deployment's real settings: ${configError}`,
    };
  }
  return { configLoaded: true, configError: null };
}

export default function Settings() {
  return (
    <SubTabs
      basePath="/settings"
      defaultKey="general"
      tabs={[
        { key: 'general', label: 'General', element: <GeneralSettings /> },
        { key: 'config', label: 'Config', element: <Config /> },
        { key: 'schema', label: 'Schema', element: <Schema /> },
        { key: 'logs', label: 'Logs', element: <Logs /> },
      ]}
    />
  );
}

function GeneralSettings() {
  const [ver, setVer] = useState<VersionInfo | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [persistMaxGiB, setPersistMaxGiB] = useState<string>('40');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [persistKV, setPersistKV] = useState<StatusSnapshot['persist_kvcache'] | null>(null);
  const [scaffoldStrip, setScaffoldStrip] = useState<boolean>(true);
  const [savingScaffoldStrip, setSavingScaffoldStrip] = useState(false);
  const [scaffoldStripMsg, setScaffoldStripMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [fastLaneEnabled, setFastLaneEnabled] = useState<boolean>(false);
  const [savingFastLane, setSavingFastLane] = useState(false);
  const [fastLaneMsg, setFastLaneMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  // These two seed the inputs for the paint BEFORE getConfig() resolves, and
  // they are what a failed getConfig() leaves in place, so they must match the
  // backend defaults in config.py FastLaneConfig (max_normal_wait_s 3600.0,
  // cross_model_switches_per_min 2). They were correct when written and went
  // stale when the backend defaults moved; this mirror is not covered by any
  // behavioural test, so keep it in step by hand when those defaults change.
  const [maxNormalWaitS, setMaxNormalWaitS] = useState<string>('3600');
  const [crossModelSwitchesPerMin, setCrossModelSwitchesPerMin] = useState<string>('2');
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  // Whether GET /api/config actually came back. Until it does, the two inputs
  // below are showing seed values rather than this deployment's settings, and
  // the toggle is showing `false` rather than whether Fast Lane is really on.
  // Saving in that state would PUT all three, silently overwriting live
  // configuration -- including switching the feature off -- with values the
  // operator never chose and had no way to know were placeholders.
  const [configLoaded, setConfigLoaded] = useState<boolean>(false);
  const [configError, setConfigError] = useState<string | null>(null);

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
          const kv = (c.kv as Record<string, unknown>) || {};
          setScaffoldStrip(kv.covered_scaffold_strip === undefined ? true : Boolean(kv.covered_scaffold_strip));
          const fastlane = (c.fastlane as Record<string, unknown>) || {};
          setFastLaneEnabled(Boolean(fastlane.enabled));
          setMaxNormalWaitS(String(fastlane.max_normal_wait_s ?? 3600));
          setCrossModelSwitchesPerMin(String(fastlane.cross_model_switches_per_min ?? 2));
          // A 200 here does not certify these three
          // fields as this deployment's real settings -- see
          // computeFastLaneLoadState's own docstring for why.
          const loadState = computeFastLaneLoadState(fastlane);
          setConfigLoaded(loadState.configLoaded);
          if (loadState.configError) setConfigError(loadState.configError);
        }
      })
      .catch((e) => {
        // Was `.catch(console.error)`. A failure here is not a console-only
        // event: this effect has no retry and `[]` deps, and the Settings tab
        // stays mounted when you navigate away, so the seeded values persist
        // for the rest of the session with nothing on screen to say so.
        // Self-describing (not just the raw error) so it reads correctly
        // alongside the fastlane.config_error message above, which
        // shares this same banner but is not a fetch failure at all.
        if (!cancelled) {
          setConfigError(
            `Could not load the current settings (GET /api/config failed: ` +
            `${e instanceof Error ? e.message : String(e)}).`
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getConfigSchema()
      .then((s) => {
        if (!cancelled) setSchema(s);
      })
      .catch((e) => {
        if (!cancelled) setSchemaError(e instanceof Error ? e.message : String(e));
      });
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

  const handleScaffoldStripSave = async () => {
    setSavingScaffoldStrip(true);
    setScaffoldStripMsg(null);
    try {
      await putConfig({ kv: { covered_scaffold_strip: scaffoldStrip } });
      setScaffoldStripMsg({ type: 'success', text: `Saved: ${scaffoldStrip ? 'enabled' : 'disabled'}` });
    } catch (e) {
      setScaffoldStripMsg({ type: 'error', text: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSavingScaffoldStrip(false);
    }
  };

  const handleFastLaneSave = async () => {
    // Belt and braces with the disabled button below: this PUT writes every
    // field on the card, so it must never run against unloaded state.
    if (!configLoaded) {
      setFastLaneMsg({
        type: 'error',
        text: 'Current settings have not loaded, so saving would overwrite them. Reload the page and try again.',
      });
      return;
    }
    const result = validateSchedulingFields(maxNormalWaitS, crossModelSwitchesPerMin, schema?.fastlane);
    if (!result.valid) {
      setFastLaneMsg({ type: 'error', text: result.error });
      return;
    }
    setSavingFastLane(true);
    setFastLaneMsg(null);
    try {
      await putConfig({ fastlane: { enabled: fastLaneEnabled, ...result.payload } });
      setFastLaneMsg({
        type: 'success',
        text: `Saved: ${fastLaneEnabled ? 'enabled' : 'disabled'}, anti-starvation wait ${result.payload.max_normal_wait_s}s, switch rate ${result.payload.cross_model_switches_per_min}/min`,
      });
    } catch (e) {
      setFastLaneMsg({ type: 'error', text: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSavingFastLane(false);
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
        <h2 className="text-xl font-semibold text-slate-200 mb-4">KV Cache Behavior</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-4">
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={scaffoldStrip}
                onChange={(e) => setScaffoldStrip(e.target.checked)}
                className="h-4 w-4 accent-emerald-500"
                disabled={savingScaffoldStrip}
              />
              Strip reasoning scaffold from saved KV cache
            </label>
            <div className="flex items-center gap-4 flex-wrap">
              <button
                onClick={handleScaffoldStripSave}
                disabled={savingScaffoldStrip}
                className="px-4 py-2 rounded bg-emerald-600 text-emerald-50 text-sm font-medium hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {savingScaffoldStrip ? 'Saving…' : 'Save'}
              </button>
              {scaffoldStripMsg && (
                <span className={`text-sm ${scaffoldStripMsg.type === 'success' ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {scaffoldStripMsg.text}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500">
              Makes the saved KV cache byte-match what the client re-sends on the next turn, so
              reasoning models reuse their cache across tool calls instead of paying a full
              re-prefill. Leave enabled unless you are debugging cache behavior.
            </p>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Fast Lane</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-4">
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={fastLaneEnabled}
                onChange={(e) => setFastLaneEnabled(e.target.checked)}
                className="h-4 w-4 accent-emerald-500"
                disabled={savingFastLane || !configLoaded}
              />
              Enable Fast Lane request priority
            </label>
            <p className="text-xs text-slate-500">
              Fast Lane changes the order requests are served in, and overrides the grace period,
              the idle-unload timer, and regular queue ordering while a rule is in effect. It is
              admission-only — it never interrupts a turn that is already running. Configure rules
              under Queue → Fast Lane.
            </p>
          </div>

          <div className="pt-4 border-t border-slate-700 space-y-3">
            <h3 className="text-sm font-semibold text-slate-300">Scheduling &amp; priority</h3>
            {configError && (
              <div className="text-red-400 text-xs rounded-md border border-red-900/50 bg-red-950/30 px-3 py-2">
                ⚠ {configError} The values below are defaults, not this deployment's settings, so
                saving is disabled — it would overwrite your live configuration. Reload the page
                to retry.
              </div>
            )}
            {schemaError && (
              <div className="text-amber-400 text-xs rounded-md border border-amber-900/50 bg-amber-950/30 px-3 py-2">
                ⚠ Per-field bounds are unavailable (GET /api/config/schema failed: {schemaError}).
                Values below are still editable; the server remains the bounds authority.
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="flex flex-col gap-1 text-sm text-slate-300">
                Anti-starvation wait — max_normal_wait_s (seconds)
                <input
                  type="number"
                  step="1"
                  min={schema?.fastlane?.max_normal_wait_s?.minimum ?? undefined}
                  max={schema?.fastlane?.max_normal_wait_s?.maximum ?? undefined}
                  value={maxNormalWaitS}
                  onChange={(e) => setMaxNormalWaitS(e.target.value)}
                  disabled={savingFastLane || !configLoaded}
                  className="min-h-11 w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm text-slate-300">
                Cross-model switch rate — cross_model_switches_per_min
                <input
                  type="number"
                  step="1"
                  min={schema?.fastlane?.cross_model_switches_per_min?.minimum ?? undefined}
                  max={schema?.fastlane?.cross_model_switches_per_min?.maximum ?? undefined}
                  value={crossModelSwitchesPerMin}
                  onChange={(e) => setCrossModelSwitchesPerMin(e.target.value)}
                  disabled={savingFastLane || !configLoaded}
                  className="min-h-11 w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </label>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 space-y-1.5">
              <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">
                Also part of this policy (queue section — read-only here)
              </div>
              {QUEUE_SCHEDULING_FIELDS.map(({ key, label }) => (
                <div key={key} className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="font-mono text-slate-400">{label}</span>
                  <span className="font-mono text-slate-200 text-right truncate">
                    {formatQueueFieldValue((config?.queue as Record<string, unknown> | undefined)?.[key])}
                  </span>
                </div>
              ))}
              <Link
                to="/settings/config"
                className="min-h-11 mt-1 flex items-center text-xs text-emerald-400 hover:text-emerald-300 underline underline-offset-2"
              >
                Edit these under Settings → Config → queue
              </Link>
            </div>

            <div className="flex items-center gap-4 flex-wrap">
              <button
                onClick={handleFastLaneSave}
                disabled={savingFastLane || !configLoaded}
                className="px-4 py-2 rounded bg-emerald-600 text-emerald-50 text-sm font-medium hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {savingFastLane ? 'Saving…' : 'Save'}
              </button>
              {fastLaneMsg && (
                <span className={`text-sm ${fastLaneMsg.type === 'success' ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {fastLaneMsg.text}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-slate-200 mb-4">Licenses + attribution</h2>
        <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-400 space-y-2">
          <p>
            Turbohaul-Manager v0.8.0 — MIT-licensed wrapper around the inference engine.
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
