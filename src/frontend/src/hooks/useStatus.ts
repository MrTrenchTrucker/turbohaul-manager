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

const POLL_INTERVAL_MS = 1000; // ~1Hz per P2 spec

export function useStatus(): UseStatusResult {
  const [data, setData] = useState<StatusSnapshot | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const mounted = useRef(true);
  const intervalRef = useRef<ReturnType<typeof window.setInterval> | null>(null);

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
     intervalRef.current = interval;
     let wsTimer: ReturnType<typeof setTimeout> | null = null;
     const sub = subscribeWsState(() => {
       // Debounce WS-triggered fetch — if we're within 200ms of the
       // next interval tick, skip the extra poll. Otherwise fetch now and
       // restart the interval to avoid double-polling.
       if (wsTimer !== null) {
         clearTimeout(wsTimer);
         wsTimer = null;
       }
       void fetchOnce();
       // Reset the interval timer to avoid a WS-triggered fetch racing
       // with the next interval tick (the double-poll bug).
       clearInterval(intervalRef.current!);
       wsTimer = setTimeout(() => {
         intervalRef.current = window.setInterval(() => {
           void fetchOnce();
         }, POLL_INTERVAL_MS);
         wsTimer = null;
       }, POLL_INTERVAL_MS);
     });
     return () => {
       mounted.current = false;
       clearInterval(intervalRef.current!);
       if (wsTimer !== null) clearTimeout(wsTimer);
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
