export type SlotState =
  | 'IDLE_COLD'
  | 'PRE_LOADING'
  | 'LOADING'
  | 'READY'
  | 'ACTIVE'
  | 'GRACE'
  | 'GRACE_BUSY'
  | 'POPPED'
  | 'IDLE_HOT'
  | 'LOADING_FAIL';

export interface ActiveInfo {
  slot_id: string;
  model_tag: string;
  state: SlotState;
  thread_id_prefix: string;
  pid: number;
  port: number;
  // Named engine operation for FE Dashboard pill
  engine_op?: string;
}

export interface GraceInfo {
  remaining_s: number;
  extension_count: number;
  max_extensions: number;
  thread_id_prefix: string;
  model_tag: string;
}

export interface IdleHotInfo {
  remaining_s: number;
  model_tag: string;
}

export interface LoadingInfo {
  slot_id: string;
  model_tag: string;
  state: SlotState;
  thread_id_prefix: string;
  elapsed_s: number;
  pid: number | null;
  port: number | null;
  // Named engine operation for FE Dashboard pill
  engine_op?: string;
}

export interface QueueWaitingRowFastLane {
  rule_index: number;
  rank: number;
  label: string;
  fastlane_rule: string;
}

// One row per waiting request -- client, tag,
// rank, and state for the whole wait, redacted (thread_id_prefix only, never
// a full thread_id or client_meta). fastlane is null for an unlisted (or floor-only) row.
// likely_victim is display-only, recomputed fresh on every /status read --
// see manager.py's _likely_victim_model_tag, never cached.
export interface QueueWaitingRow {
  position: number;
  slot_id: string;
  model_tag: string;
  thread_id_prefix: string;
  state: string;
  waited_s: number;
  fastlane: QueueWaitingRowFastLane | null;
  floor_promoted: boolean;
  likely_victim: string | null;
}

// One row per live Fast Lane claim,
// served by /status as fastlane_claims_snapshot. A DIFFERENT population from
// QueueWaitingRow: a claim is registered via _defer_unroutable while
// genuinely deferring and can PRECEDE staging, so queue.waiting (backed by
// queue._staging) is blind to it. fastlane is null for an unlisted
// (or floor-only) claim; the shape is identical to QueueWaitingRowFastLane,
// so that type is reused rather than duplicated.
export interface FastlaneClaimRow {
  slot_id: string;
  model_tag: string;
  thread_id_prefix: string;
  fastlane: QueueWaitingRowFastLane | null;
  reason: string;
  registered_at: string;
  waited_s: number;
}

export interface QueueInfo {
  acceptance_buffer_depth: number;
  staging_queue_depth: number;
  staging_queue_max: number;
  // Requests ACTUALLY waiting, including those handed straight to a
  // busy resident's inbox -- those never enter _staging and so are invisible
  // to the two fields above, which is why the staging number can read 0 while
  // work is genuinely queued. Optional: an older manager does not send it,
  // and we render "unknown" rather than a confidently wrong number.
  queue_depth_total?: number;
  // /status serves a real snapshot via manager.status_snapshot, one row per
  // live claim.
  // Optional: an older manager does not send the key at all.
  fastlane_claims_snapshot?: FastlaneClaimRow[];
  // The waiting-request surface, up to 50
  // rows. Optional: an older manager does not send it.
  waiting?: QueueWaitingRow[];
}

export interface ParallelSlots {
  used: number;
  max: number;
}

// --- P2: per-resident + vram types (consumed from /status) ---

export interface GenerationInfo {
  state: string;
  tok_s?: number;
  tok_s_instant?: number;
  n_decoded?: number;
  max_tokens?: number;
  n_remain?: number;
  n_prompt_tokens?: number;
  // Widened from number|undefined to number|null -- the wire
  // sends literal null when a resident goes idle (n_prompt_cache two lines
  // below already carries this exact type for the same reason). No
  // consumers of n_ctx existed before the widening (checked repo-wide), so
  // this is a type-only change with no blast radius beyond the two call
  // sites that read it.
  n_ctx?: number | null;
  prompt_progress: string | null;
  // Integer prefill %% from the /slots headline (BE live_monitor).
  prefill_pct?: number | null;
  // Raw prefill counters (cache-restored + newly-processed).
  n_prompt_proc?: number | null;
  n_prompt_cache?: number | null;
  pct?: number;
  eta_s?: number;
  stalled: boolean;
  streaming: boolean;
  generation_id: string | null;
  riders?: number;
  measured_at_iso: string;
  // Mid-prefill hang alarm (observability-only, FE renders red banner)
  prefill_stall_alarm?: boolean;
}

export interface ResidentModel {
  model_tag: string;
  state: string;
  port: number;
  pid: number;
  spawn_seq: number;
  reserved_need_mib: number;
  parallel: number;
  main_gpu: number;
  split_mode: string;
  inflight: number;
  // This resident's own pending-request
  // count -- same accessor manager.py's own inbox-depth reads already use.
  // Optional: an older manager does not send the key at all.
  inbox_depth?: number;
  idle_expires_in_s: number | null;
  // The ONE resolved phase + its countdown. `phase` carries the
  // backend ResidentState value names VERBATIM (RESERVED_LOADING / ACTIVE / GRACE /
  // IDLE_EVICTABLE / DEAD) -- deliberately NOT a new vocabulary, because this card
  // already branches on a mix of SlotState names and invented ones, and adding a
  // fourth spelling is the defect this field exists to remove.
  // `idle_expires_in_s` above is now DERIVED from the same backend resolver, not
  // computed separately -- it stays for one release for consumers that predate this.
  // All three are optional: undefined on any backend that hasn't shipped them.
  phase?: string;
  remaining_s?: number | null;
  phase_resolved_from?: string;
  generation: GenerationInfo | null;
  // Named engine operation for FE Dashboard pill
  engine_op?: string;
  // This resident's OWN LAST-request identity, additive --
  // regardless of whether it is still running. undefined on any backend
  // that hasn't shipped this field yet (ResidentCard must fall back to the
  // legacy global identity in that case); null once the field exists but
  // this resident hasn't served a request yet.
  request_identity?: RequestIdentity | null;
  // This resident's identity RIGHT NOW -- null when this
  // resident is idle. NOT the same fact as request_identity above: a
  // finished session must not keep showing as live. undefined on any backend
  // that hasn't shipped this field yet.
  current_request_identity?: RequestIdentity | null;
}

// Last per-request
// structured identity — display/observability only, all fields nullable.
export interface RequestIdentity {
  ip: string | null;
  // The operator-saved Fast Lane rule label for this ip
  // (fastlane.rules[].label), null/absent when the ip has no saved label —
  // render NOTHING for it (no placeholder, no 'unknown').
  label?: string | null;
  model_tag: string | null;
  session_id: string | null;
  is_main: boolean | null;
  is_sub_agent: boolean | null;
  is_curator: boolean | null;
  is_compression: boolean | null;
  resolved_class: string | null;
  thread_id: string | null;
}

// Observability: one record per model (re)spawn + KV restore —
// the REAL end-state so operators + FE can SEE whether a load truly took,
// instead of trusting a bare 200. Display-only, all fields nullable.
export interface LoadVerifyRecord {
  event: 'model_load' | 'kv_restore' | string;
  trigger: string;
  model_tag: string;
  port: number;
  pid: number | null;
  process_alive: boolean | null;
  health_200: boolean | null;
  model_resident: boolean | null;
  kv_expected_tokens: number | null;
  kv_actual_n_past: number | null;
  restore_attempted?: boolean | null;
  kv_restore_ok: boolean | null;
  retry_count: number;
  final_status: 'ok' | 'retried_ok' | 'failed' | string;
  reason: string | null;
  thread_hash: string | null;
  session_id: string | null;
}

// Observability: one record per model where speculative decoding
// ran WITHOUT the requested acceleration -- INFORMATIONAL, never an error.
// The model loaded and serves normally; this only says the speedup was not
// applied and why. A genuine load FAILURE is represented exclusively by
// ResidentModel.state === 'LOADING_FAIL' -- a different field, on purpose,
// so the two can never collapse into one indicator. `source` distinguishes
// where the downgrade was detected: 'engine_log' (parsed from the engine's
// own SPEC_DOWNGRADED line after the target model already loaded) or
// 'manager_config' (the manager caught an unrunnable spec_type/draft-model
// pairing before ever spawning the drafter). `reason_code` is an open,
// append-only vocabulary -- do not assume it is a closed set. Display-only,
// all fields nullable except model_tag/source.
export interface SpecDowngradeRecord {
  model_tag: string;
  source: 'engine_log' | 'manager_config' | string;
  arch: string | null;
  component: 'draft' | 'mtp' | string | null;
  reason_code: string | null;
  detail: string | null;
}

export interface StatusSnapshot {
  queue: QueueInfo;
  active: ActiveInfo | null;
  loading: LoadingInfo | null;
  grace: GraceInfo | null;
  idle_hot: IdleHotInfo | null;
  parallel_slots: ParallelSlots;
  // P2: per-resident array + vram
  residents: ResidentModel[];
  vram: number[] | null;
  vram_total_mib: number[] | null;
  // P2: legacy single-generation alias (kept for compat)
  generation: GenerationInfo | null;
  // LAST request's structured identity, additive --
  // regardless of whether it is still running.
  request_identity?: RequestIdentity | null;
  // The identity actually driving the box RIGHT NOW --
  // null when idle. LOAD-BEARING at cap<=1: residents[] is empty in that
  // configuration, so this top-level key is the ONLY place the feature is
  // visible there.
  current_request_identity?: RequestIdentity | null;
  // Last N load/restore verify records (newest last), additive.
  load_verify?: LoadVerifyRecord[] | null;
  // Last N SPEC_DOWNGRADE records (newest last), additive.
  spec_downgrade?: SpecDowngradeRecord[] | null;
  // Persisted KV cache SSD usage snapshot
  persist_kvcache?: {
    total_bytes: number;
    cap_bytes: number;
    file_count: number;
    headroom_bytes: number;
    over_cap: boolean;
  } | null;
}

export interface ModelTag {
  name: string;
  size: number;
  digest: string;
  modified_at?: string;
  details?: {
    format?: string;
    context_length?: number;
    expected_vram_bytes?: number;
    display_name?: string;
    description?: string;
    // Additive, best-effort, all nullable —
    // absent on older BE responses, must degrade gracefully.
    parameter_count?: number | null;
    size_label?: string | null;
    architecture?: string | null;
    is_moe?: boolean | null;
    expert_count?: number | null;
    modality?: 'text' | 'vision' | null;
  };
  revision?: number;
}

export interface VersionInfo {
  version: string;
  backend: string;
  backend_sha_pinned: boolean;
  api_compat: string;
  user_agent: string;
}

// Per-model manifest editor types
export interface Manifest {
  model_tag: string;
  display_name?: string;
  description?: string;
  gguf_blob_sha256: string;
  gguf_size_bytes?: number;
  context_size?: number;
  expected_vram_bytes?: number;
  revision?: number;
  // Additive: hidden is a real manifest field (false today); surfaced here so the
  // Models tab can render hide/unhide without hand-editing yaml on disk.
  hidden?: boolean;
  llama_server_flags?: Record<string, unknown>;
  prompt_template?: {
    system_default?: string;
    stop_tokens?: string[];
  };
}

export interface ManifestWithEtag {
  manifest: Manifest;
  etag: string;
}

export interface ManifestSaveResult {
  status: string;
  model_tag: string;
  revision: number;
  restart_required: boolean;
}

// P2: SSE live-output frame shape
export interface LiveOutputFrame {
  generation_id: string | null;
  text: string;
  done: boolean;
  reset: boolean;
  idle?: boolean;
  tok_s?: number;
  model_tag?: string | null;
  [key: string]: unknown;
}

// Config tab schema-driven editor. Additive: describes the
// per-field {type, default, min/max} contract GET /api/config/schema
// serves, built off Pydantic model_json_schema() + model_dump().
export type ConfigFieldType = 'integer' | 'number' | 'boolean' | 'string' | 'array';

export interface ConfigFieldSchema {
  type: ConfigFieldType;
  default: unknown;
  minimum?: number;
  maximum?: number;
}

export type ConfigSectionSchema = Record<string, ConfigFieldSchema>;

export type ConfigSchema = Record<string, ConfigSectionSchema>;

// Fast Lane rules + discovered-address census. Additive: FastLaneRule/FastLaneTagRanks mirror config.py; the census
// entry shape comes from TurbohaulManager.fastlane_census_snapshot() on the
// backend and is read defensively here as an open record since its exact
// field set is owned elsewhere.
export interface FastLaneTagRanks {
  main: number | null;
  curator: number | null;
  compression: number | null;
  sub_agent: number | null;
  unclassified: number | null;
}

export interface FastLaneRule {
  // Exactly one of address / container_name identifies the client.
  // `container_name` MUST be carried here even though the FE never edits it
  // directly: the rules table PUTs the WHOLE array back, so a field this
  // type does not know about is dropped on the next operator save.
  address?: string | null;
  container_name?: string | null;
  label: string | null;
  tag_ranks: FastLaneTagRanks | null;
}

/** What PUT /api/config actually returns. `warnings` carries the Fast Lane
 *  lint output -- a save can succeed AND be misconfigured, so the caller
 *  must render these rather than treating HTTP 200 as "clean". */
export interface PutConfigResult extends Record<string, unknown> {
  status?: string;
  warnings?: string[];
}

export interface FastLaneDiscoverable {
  address: string;
  container_name?: string | null;
  [key: string]: unknown;
}

export interface FastLaneCensusResponse {
  backend_pending: boolean;
  entries: FastLaneDiscoverable[];
}

// Media Hook (FE): GET /api/plugins shape, byte-matches
// api/plugins.py's list_plugins response. Deliberately has NO host/port field
// -- the backend redacts those server-side (same rationale as GET /api/config's
// plugins.registry_keys) and this type must not grow one back in.
export interface PluginListEntry {
  model_tag: string;
  lane: 'cpu' | 'gpu';
  capabilities: string[];
  // The backend computes this from a REGISTRY LOOKUP that never opens a
  // socket, so it means "an entry exists
  // for this resource_key and its address is permitted" -- never "the
  // container is up". This name must keep matching api/plugins.py's wire key
  // exactly: a mismatch does not throw here, it silently reads `undefined`,
  // renders the amber branch, and reports every plugin as unconfigured.
  configured: boolean;
}

export interface PluginListResponse {
  plugins: PluginListEntry[];
  total: number;
}

const BASE = '';

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json() as Promise<T>;
}

export const getStatus = () => getJSON<StatusSnapshot>('/status');
export const getTags = () => getJSON<{ models: ModelTag[] }>('/api/tags');
export const getVersion = () => getJSON<VersionInfo>('/api/version');
export const getConfig = () => getJSON<Record<string, unknown>>('/api/config');
export const getConfigSchema = () => getJSON<ConfigSchema>('/api/config/schema');
export const getFastLaneCensus = () => getJSON<FastLaneCensusResponse>('/api/fastlane/census');
export const getPlugins = () => getJSON<PluginListResponse>('/api/plugins');

export async function putConfig(payload: Record<string, unknown>): Promise<PutConfigResult> {
  const r = await fetch(`${BASE}/api/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(`PUT /api/config ${r.status}${detail ? ` — ${detail}` : ''}`);
  }
  return (await r.json()) as Promise<Record<string, unknown>>;
}

// Manifest CRUD with ETag handling
export async function getManifest(tag: string): Promise<ManifestWithEtag> {
  const r = await fetch(`${BASE}/api/manifests/${encodeURIComponent(tag)}`);
  if (!r.ok) throw new Error(`GET /api/manifests/${tag} ${r.status}`);
  const etag = r.headers.get('etag') || '';
  const manifest = (await r.json()) as Manifest;
  return { manifest, etag };
}

export async function putManifest(
  tag: string,
  manifest: Manifest,
  ifMatch: string | null,
): Promise<ManifestSaveResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (ifMatch) headers['If-Match'] = ifMatch;
  const r = await fetch(`${BASE}/api/manifests/${encodeURIComponent(tag)}`, {
    method: 'PUT',
    headers,
    body: JSON.stringify(manifest),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(`PUT /api/manifests/${tag} ${r.status}${detail ? ` — ${detail}` : ''}`);
  }
  return (await r.json()) as ManifestSaveResult;
}

// Narrow PATCH: flips ONLY the hidden field on a manifest. Touches no root
// fields, no llama_server_flags — a hide toggle must never rewrite unrelated
// settings. If-Match carries the revision; 412 on mismatch like PUT.
export async function patchManifestHidden(
  tag: string,
  hidden: boolean,
  ifMatch: string | null,
): Promise<ManifestSaveResult & { hidden: boolean }> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (ifMatch) headers['If-Match'] = ifMatch;
  const r = await fetch(`${BASE}/api/manifests/${encodeURIComponent(tag)}`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify({ hidden }),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(`PATCH /api/manifests/${tag} ${r.status}${detail ? ` — ${detail}` : ''}`);
  }
  return (await r.json()) as ManifestSaveResult & { hidden: boolean };
}

// Management listing — the ONLY surface that includes hidden manifests.
// /api/tags omits hidden models by design; reading hidden from it is a trap.
export interface ManifestSummaryRow {
  model_tag: string;
  // manifests.py's list_manifests has ALWAYS sent
  // this field ("kind": m.kind) -- it was simply never typed
  // on the FE side, which had no prior consumer that needed to tell model
  // and plugin manifest rows apart from the summary listing alone. Adding
  // the type does not change what the backend sends.
  kind: 'model' | 'plugin' | null; // null only alongside hidden:null (unreadable row)
  hidden: boolean | null; // null = unreadable manifest; error field carries why
  revision: number; // display only
  etag: string; // exact quoted If-Match value the backend wants ('"3"')
  // These two were added so the L1 tile grid
  // can be drawn from ONE listing call instead of an N+1 Promise.all over
  // every manifest (the server already read the full
  // manifest and was discarding both fields). null for the unreadable-row
  // sentinel (alongside kind:null/hidden:null) -- kept present-but-null
  // rather than omitted so the row shape stays uniform. display_name can
  // ALSO be legitimately null on an otherwise-readable model manifest that
  // simply has none set (Manifest.display_name is optional) -- guard for
  // null, never assume a string.
  display_name: string | null;
  gguf_blob_sha256: string | null;
  // File size is a manifest-level fallback for the tile's size row when
  // GET /api/blobs (the canonical source) is not available.
  gguf_size_bytes?: number | null;
  error?: string;
}

export async function getManifests(): Promise<{
  manifests: ManifestSummaryRow[];
  total: number;
  hidden_count: number;
}> {
  const r = await fetch(`${BASE}/api/manifests`);
  if (!r.ok) throw new Error(`GET /api/manifests ${r.status}`);
  return (await r.json()) as {
    manifests: ManifestSummaryRow[];
    total: number;
    hidden_count: number;
  };
}

// Clears a single model's KV/checkpoint overrides so it follows the global defaults.
// Empty `cleared` means it was already on defaults — success, not an error.
export async function restoreManifestDefaults(
  tag: string,
  ifMatch: string | null,
): Promise<{
  status: string;
  model_tag: string;
  revision: number;
  cleared: Record<string, unknown>;
  now_defaults: Record<string, unknown>;
  takes_effect: string;
}> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (ifMatch) headers['If-Match'] = ifMatch;
  const r = await fetch(
    `${BASE}/api/manifests/${encodeURIComponent(tag)}/restore-defaults`,
    { method: 'POST', headers },
  );
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(
      `POST /api/manifests/${tag}/restore-defaults ${r.status}${detail ? ` — ${detail}` : ''}`,
    );
  }
  return (await r.json()) as {
    status: string;
    model_tag: string;
    revision: number;
    cleared: Record<string, unknown>;
    now_defaults: Record<string, unknown>;
    takes_effect: string;
  };
}

// Enumerate blobs on disk (GET /api/blobs -> {"blobs":[{"digest","size_bytes"}...],"total"}).
// Not available on older backends -- a caller MUST catch this and degrade (see
// Models.tsx's groupIntoTiles: a null blob list means "zero-manifest blobs
// don't get a tile," not "no models.").
export async function getBlobs(): Promise<{
  blobs: { digest: string; size_bytes: number; description: string | null }[];
  total: number;
}> {
  const r = await fetch(`${BASE}/api/blobs`);
  if (!r.ok) throw new Error(`GET /api/blobs ${r.status}`);
  return (await r.json()) as {
    blobs: { digest: string; size_bytes: number; description: string | null }[];
    total: number;
  };
}

// A human description
// on the model tile itself, editable from inside the opened model; added
// alongside GET /api/blobs. Empty string / null CLEARS it;
// unknown digest 404s; malformed digest 400s; >2000 chars 400s.
export async function putBlobDescription(digest: string, description: string): Promise<void> {
  const r = await fetch(`${BASE}/api/blobs/${encodeURIComponent(digest)}/description`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ description }),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(`PUT /api/blobs/${digest}/description ${r.status}${detail ? ` — ${detail}` : ''}`);
  }
}

// Model-level delete: deletes the blob by digest.
// /api/delete performs a reference check -- it refuses
// with 409 when any manifest still names this digest in gguf_blob_sha256,
// mmproj_blob_sha256 or spec_draft_gguf_blob_sha256, and the detail names
// those manifests. The caller cannot delete a blob out from under a
// manifest, even by choosing to keep it.
// Reading `detail` is the whole point of the backend naming them: without it
// the user sees "DELETE /api/delete 409" and has nothing to act on. Same
// shape as every other write path in this file.
export async function deleteBlobApi(digest: string): Promise<void> {
  const r = await fetch(`${BASE}/api/delete`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ digest }),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const b = await r.json();
      detail = (b as { detail?: string }).detail || '';
    } catch {
      // ignore
    }
    throw new Error(`DELETE /api/delete ${r.status}${detail ? ` — ${detail}` : ''}`);
  }
}

export async function deleteManifestApi(tag: string): Promise<void> {
  const r = await fetch(`${BASE}/api/manifests/${encodeURIComponent(tag)}`, {
    method: 'DELETE',
  });
  if (!r.ok) throw new Error(`DELETE /api/manifests/${tag} ${r.status}`);
}
