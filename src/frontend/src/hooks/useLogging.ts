// /v1/logging hook. Inline fetch + co-located
// types per #3 (no api.ts addition for a single route). Race-guards
// via epoch-counter + AbortController. NO autoRefresh (#1 dropped).
import { useCallback, useEffect, useRef, useState } from 'react';

export interface LogEvent {
  event_id: number;
  slot_id: string | null;
  event_type: string;
  payload: Record<string, unknown>;
  occurred_at: string;
}

interface LoggingEnvelope {
  events: LogEvent[];
  next_since: number | null;
  oversized: boolean;
}

export interface LoggingFilters {
  slot_id?: string;
  event_type?: string;
  limit: number;
}

export interface UseLoggingResult {
  events: LogEvent[];
  loading: boolean;
  error: Error | null;
  hasMore: boolean;
  oversized: boolean;
  loadMore: () => void;
  refresh: () => void;
}

// F8: hard cap on accumulated events to prevent mobile-jank under heavy paging.
export const MAX_ACCUMULATED_EVENTS = 2000;

function buildUrl(filters: LoggingFilters, since: number): string {
  // F4: omit slot_id/event_type when empty — BE matches literal "" not omitted.
  const params = new URLSearchParams();
  params.append('since', String(since));
  params.append('limit', String(filters.limit));
  if (filters.slot_id && filters.slot_id.trim()) {
    params.append('slot_id', filters.slot_id.trim());
  }
  if (filters.event_type && filters.event_type.trim()) {
    params.append('event_type', filters.event_type.trim());
  }
  return `/v1/logging?${params.toString()}`;
}

export function useLogging(filters: LoggingFilters): UseLoggingResult {
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [nextSince, setNextSince] = useState<number | null>(0);
  const [oversized, setOversized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // F5 race-guard: epoch counter invalidates in-flight responses on overlap.
  const epochRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const fetchPage = useCallback(
    async (since: number, mode: 'reset' | 'append') => {
      const myEpoch = ++epochRef.current;
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      try {
        const r = await fetch(buildUrl(filters, since), { signal: ac.signal });
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
        const env = (await r.json()) as LoggingEnvelope;
        if (myEpoch !== epochRef.current) return; // stale — newer fetch superseded
        setEvents((prev) => {
          // BE returns ASC by event_id (#4). loadMore appends NEWER below.
          const merged = mode === 'reset' ? env.events : [...prev, ...env.events];
          // F8 cap — keep the last MAX_ACCUMULATED_EVENTS (newest tail).
          return merged.length > MAX_ACCUMULATED_EVENTS
            ? merged.slice(merged.length - MAX_ACCUMULATED_EVENTS)
            : merged;
        });
        setNextSince(env.next_since);
        setOversized(env.oversized);
        setError(null);
      } catch (e) {
        if (myEpoch !== epochRef.current) return;
        if ((e as DOMException).name === 'AbortError') return;
        setError(e instanceof Error ? e : new Error(String(e)));
      } finally {
        if (myEpoch === epochRef.current) setLoading(false);
      }
    },
    [filters.slot_id, filters.event_type, filters.limit],
  );

  // Initial + filter-change: reset and start from since=0.
  useEffect(() => {
    void fetchPage(0, 'reset');
    return () => {
      abortRef.current?.abort();
    };
  }, [fetchPage]);

  // F3: cursor predicate MUST be `!== null && !== undefined` — event_id 0 is
  // a valid cursor and a plain `if (nextSince)` would falsy-trap on it.
  const hasMore =
    nextSince !== null &&
    nextSince !== undefined &&
    events.length < MAX_ACCUMULATED_EVENTS;

  const loadMore = useCallback(() => {
    if (!hasMore || loading) return;
    void fetchPage(nextSince as number, 'append');
  }, [fetchPage, hasMore, loading, nextSince]);

  const refresh = useCallback(() => {
    void fetchPage(0, 'reset');
  }, [fetchPage]);

  return { events, loading, error, hasMore, oversized, loadMore, refresh };
}
