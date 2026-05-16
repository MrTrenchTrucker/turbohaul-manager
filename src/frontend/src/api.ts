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

export interface StatusSnapshot {
  active: { tag: string | null; port: number | null; state: SlotState | null };
  grace: { tag: string | null; deadline_ms: number | null };
  queue: { depth: number; head_tag: string | null };
  idle_hot: { tag: string | null; expires_ms: number | null };
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
