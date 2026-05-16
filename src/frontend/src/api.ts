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

export interface QueueInfo {
  acceptance_buffer_depth: number;
  staging_queue_depth: number;
  staging_queue_max: number;
}

export interface ParallelSlots {
  used: number;
  max: number;
}

export interface StatusSnapshot {
  queue: QueueInfo;
  active: ActiveInfo | null;
  grace: GraceInfo | null;
  idle_hot: IdleHotInfo | null;
  parallel_slots: ParallelSlots;
}

export interface ModelTag {
  name: string;
  size: number;
  digest: string;
  modified_at?: string;
}

export interface VersionInfo {
  version: string;
  backend: string;
  backend_sha_pinned: boolean;
  api_compat: string;
  user_agent: string;
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
