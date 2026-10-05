import type { ResidentModel, StatusSnapshot } from '../../api';

/**
 * Synthesize a partial ResidentModel from legacy single-residency fields
 * (active / loading / grace / idle_hot / generation).
 *
 * Under cap<=1, residents[] is empty by design — the inference data lives
 * on the legacy fields. This bridges that gap so the dashboard panels
 * (model name, state, tok/s, tok/s graph, live output) still render.
 *
 * Priority order matches the clean FE LoadedBanner:
 *   active > loading > grace > idle_hot
 */
export function synthesizeResident(data: StatusSnapshot): ResidentModel | null {
  const source =
    data.active ??
    data.loading ??
    (data.grace ? { model_tag: data.grace.model_tag, state: 'GRACE' as const } : null) ??
    (data.idle_hot ? { model_tag: data.idle_hot.model_tag, state: 'IDLE_HOT' as const } : null);

  if (!source) return null;

  return {
    model_tag: source.model_tag,
    state: source.state,
    port: (source as any).port ?? 0,
    pid: (source as any).pid ?? 0,
    spawn_seq: 0,
    reserved_need_mib: 0,
    parallel: 1,
    main_gpu: 0,
    split_mode: 'single',
    inflight: 0,
    // 'unload in Ns' is only truthful when the model is actually
    // parked — never while a serve is in flight (this badge + the stale advertised
    // grace clock would otherwise read as the grace timer firing mid-prefill).
    idle_expires_in_s:
      source.state === 'GRACE' || source.state === 'IDLE_HOT'
        ? (data.grace?.remaining_s ?? data.idle_hot?.remaining_s ?? null)
        : null,
    generation: data.generation,
    // engine_op from active/loading info
    engine_op: (source as any).engine_op,
  };
}
