// Live-output SSE hook for the Dashboard live panel. Opens ONE persistent
// EventSource against /ui/live/output/stream (NO query params) and FOLLOWS the
// CURRENT live generation automatically — the "anchor". Frame contract:
//   { generation_id, text, done:false, reset:true }            -> NEW generation: CLEAR pane, then show replay tail
//   { generation_id, text, done:false }                        -> incremental delta: APPEND text
//   { generation_id, text:"", done:true }                      -> generation ENDED: mark ended, KEEP listening
//   { generation_id:null, text:"", done:false, reset:true, idle:true } -> IDLE: clear + show idle
//   ': keep-alive' comment lines                               -> heartbeats (never reach onmessage)
// Reconnect-on-error mirrors the exponential backoff in ws.ts (1s -> 30s cap).
// Text is bounded to ~16KiB to match the server-side tail and keep the DOM light.
import { useEffect, useRef, useState } from 'react';

// Keep the accumulated tail in lock-step with the server's "last ~16KiB" tail.
export const MAX_OUTPUT_CHARS = 16 * 1024;

interface LiveOutputFrame {
  generation_id: string | null;
  text: string;
  done: boolean;
  reset?: boolean;
  idle?: boolean;
}

export interface UseLiveOutputResult {
  text: string;
  generationId: string | null;
  available: boolean; // a connection is open and following the anchor
  ended: boolean; // current generation signalled done (stream still listening)
  idle: boolean; // no live generation right now
}

function clampTail(s: string): string {
  return s.length > MAX_OUTPUT_CHARS ? s.slice(s.length - MAX_OUTPUT_CHARS) : s;
}

export function useLiveOutput(): UseLiveOutputResult {
  const [text, setText] = useState('');
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const [ended, setEnded] = useState(false);
  const [idle, setIdle] = useState(false);

  // Mutable text mirror so incremental deltas accumulate without re-subscribing.
  const textRef = useRef('');
  // Anti-flicker: on a fresh generation we KEEP the previous output visible and
  // only replace it when the NEXT generation's first token actually arrives, so
  // a bursty workload (generate -> gap -> generate) never blanks the pane.
  const pendingClear = useRef(false);

  useEffect(() => {
    let es: EventSource | null = null;
    let stopped = false;
    let retryMs = 1000;
    const MAX_RETRY = 30000;
    let retryTimer: number | undefined;

    const connect = () => {
      if (stopped) return;
      // NO query params — the stream follows the current live generation.
      es = new EventSource('/ui/live/output/stream');

      es.onopen = () => {
        retryMs = 1000;
        setAvailable(true);
      };

      es.onmessage = (e) => {
        let frame: LiveOutputFrame;
        try {
          frame = JSON.parse(e.data) as LiveOutputFrame;
        } catch {
          return; // ignore malformed frames (keep-alive comments never reach here)
        }

        if (frame.idle) {
          // Went idle: KEEP the last output visible (no blank flash); just mark
          // idle. The previous generation's text stays until the next one starts.
          // Clear any deferred-clear flag: an idle frame that arrives before the
          // next generation's first delta must not later wipe the retained text.
          pendingClear.current = false;
          setGenerationId(null);
          setIdle(true);
          setEnded(true);
          return;
        }

        if (frame.reset) {
          // A NEW generation started. If a replay tail came with it (reconnect
          // mid-generation) show it immediately; otherwise KEEP the prior output
          // and defer the clear to the first delta of the new generation, so a
          // bursty generate->gap->generate cycle never blanks the pane.
          setGenerationId(frame.generation_id);
          setIdle(false);
          setEnded(false);
          if (frame.text) {
            textRef.current = clampTail(frame.text);
            setText(textRef.current);
            pendingClear.current = false;
          } else {
            pendingClear.current = true;
          }
          return;
        }

        if (frame.text) {
          // Incremental delta. If a fresh generation is pending, replace the old
          // output now (on the first real token) instead of blanking earlier.
          if (pendingClear.current) {
            textRef.current = '';
            pendingClear.current = false;
          }
          textRef.current = clampTail(textRef.current + frame.text);
          setText(textRef.current);
        }

        if (frame.done) {
          // Current generation ended — mark ended but KEEP the connection open;
          // more generations will follow and arrive as fresh `reset` frames.
          setEnded(true);
        }
      };

      es.onerror = () => {
        // EventSource auto-reconnects, but we manage backoff explicitly to
        // mirror ws.ts and avoid hammering the endpoint.
        setAvailable(false);
        es?.close();
        if (stopped) return;
        retryTimer = window.setTimeout(connect, retryMs);
        retryMs = Math.min(retryMs * 2, MAX_RETRY);
      };
    };

    connect();

    return () => {
      stopped = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      es?.close();
    };
  }, []);

  return { text, generationId, available, ended, idle };
}
