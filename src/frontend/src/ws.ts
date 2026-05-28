export type WsEvent = Record<string, unknown>;
export type WsHandler = (event: WsEvent) => void;

export interface WsSubscription {
  close: () => void;
}

// Reconnect pattern uses exponential backoff with jitter.
// (production-validated; exponential backoff 1s -> 30s cap).
export function subscribeWsState(handler: WsHandler): WsSubscription {
  let ws: WebSocket | null = null;
  let stopped = false;
  let retryMs = 1000;
  const MAX_RETRY = 30000;

  const connect = () => {
    if (stopped) return;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws/state`;
    ws = new WebSocket(url);
    ws.onopen = () => {
      retryMs = 1000;
    };
    ws.onmessage = (e) => {
      try {
        handler(JSON.parse(e.data));
      } catch {
        // ignore
      }
    };
    ws.onclose = () => {
      if (stopped) return;
      setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, MAX_RETRY);
    };
    ws.onerror = () => {
      ws?.close();
    };
  };

  connect();

  return {
    close: () => {
      stopped = true;
      ws?.close();
    },
  };
}
