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
  // Named engine operation for the Dashboard pill
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
  // Named engine operation for the Dashboard pill
  engine_op?: string;
}

export interface QueueInfo {
  acceptance_buffer_depth: number;
  staging_queue_depth: number;
  staging_queue_max: number;
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
  n_ctx?: number;
  prompt_progress: string | null;
  // Integer prefill %% from the /slots headline (backend live monitor).
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
  // Mid-prefill hang alarm (observability-only; the UI renders a red banner)
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
  idle_expires_in_s: number | null;
  model_resident: boolean | null;
  generation: GenerationInfo | null;
}

// Last per-request structured identity — display/observability only,
// all fields nullable.
export interface RequestIdentity {
  ip: string | null;
  model_tag: string | null;
  session_id: string | null;
  is_main: boolean | null;
  is_sub_agent: boolean | null;
  is_curator: boolean | null;
  is_compression: boolean | null;
  resolved_class: string | null;
  thread_id: string | null;
}

// Observability: one record per model (re)spawn + KV restore — the real
// end-state so operators and the UI can see whether a load truly took,
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
  kv_restore_ok: boolean | null;
  retry_count: number;
  final_status: 'ok' | 'retried_ok' | 'failed' | string;
  reason: string | null;
  thread_hash: string | null;
  session_id: string | null;
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
  // Last request's structured identity, additive.
  request_identity?: RequestIdentity | null;
  // Last N load/restore verify records (newest last), additive.
  load_verify?: LoadVerifyRecord[] | null;
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

export async function putConfig(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
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

export async function deleteManifestApi(tag: string): Promise<void> {
  const r = await fetch(`${BASE}/api/manifests/${encodeURIComponent(tag)}`, {
    method: 'DELETE',
  });
  if (!r.ok) throw new Error(`DELETE /api/manifests/${tag} ${r.status}`);
}
