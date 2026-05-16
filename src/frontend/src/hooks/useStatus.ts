import { useCallback, useEffect, useRef, useState } from 'react';
import type { StatusSnapshot } from '../api';
import { getStatus } from '../api';
import { subscribeWsState } from '../ws';

export interface UseStatusResult {
  data: StatusSnapshot | null;
  loading: boolean;
  error: Error | null;
  lastUpdate: Date | null;
  refresh: () => void;
}

const POLL_INTERVAL_MS = 2000;

export function useStatus(): UseStatusResult {
  const [data, setData] = useState<StatusSnapshot | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const mounted = useRef(true);

  const fetchOnce = useCallback(async () => {
    try {
      const s = await getStatus();
      if (!mounted.current) return;
      setData(s);
      setError(null);
      setLastUpdate(new Date());
    } catch (e) {
      if (!mounted.current) return;
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void fetchOnce();
    const interval = window.setInterval(() => {
      void fetchOnce();
    }, POLL_INTERVAL_MS);
    const sub = subscribeWsState(() => {
      void fetchOnce();
    });
    return () => {
      mounted.current = false;
      window.clearInterval(interval);
      sub.close();
    };
  }, [fetchOnce]);

  return {
    data,
    loading: data === null && error === null,
    error,
    lastUpdate,
    refresh: () => {
      void fetchOnce();
    },
  };
}
