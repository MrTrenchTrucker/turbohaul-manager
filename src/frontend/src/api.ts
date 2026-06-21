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

export type GenerationState =
  | 'generating'
  | 'prefill'
  | 'finishing'
  | 'idle'
  | 'loading'
  | 'grace'
  | 'stalled'
  | 'transitioning';

export interface GenerationInfo {
  state: GenerationState;
  tok_s: number | null; // EWMA tok/s; null = first-decode pending (show '—', never a fake low number)
  tok_s_instant: number; // unsmoothed last-tick rate (use for the sparkline)
  n_decoded: number; // tokens generated so far
  max_tokens: number | null; // null = unbounded (no progress bar; count up)
  n_remain: number | null;
  n_prompt_tokens: number; // CURRENT REQUEST context (prompt + generated-so-far)
  n_ctx: number | null; // model context-window CAPACITY (e.g. 250112); null = unknown
  prompt_progress: number | null; // 0..1 during prefill
  pct: number | null; // 0..100; null = unbounded
  eta_s: number | null;
  stalled: boolean;
  streaming: boolean; // false => live text unavailable (non-streaming request)
  generation_id: string | null; // 8 hex chars; use as the reset key + SSE query param
  riders: number; // concurrent slots (parallel:N); 1 for series
  measured_at_iso: string;
}

export interface StatusSnapshot {
  queue: QueueInfo;
  active: ActiveInfo | null;
  loading: LoadingInfo | null;
  grace: GraceInfo | null;
  idle_hot: IdleHotInfo | null;
  parallel_slots: ParallelSlots;
  generation?: GenerationInfo | null;
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
