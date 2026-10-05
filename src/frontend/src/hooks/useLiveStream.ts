import { useEffect, useRef, useState } from 'react';
import type { LiveOutputFrame } from '../api';

// One SSE connection PER RESIDENT (?model_tag=<tag>), replacing
// the single-anchor-follow design. The backend already supported this
// (live_stream.py's ?model_tag= param) -- it was never wired up here.
//
// Why one connection per resident: a single fetch(SSE_URL) with no param
// follows the manager's global "most-recently-active resident" alias. With
// two residents generating, the anchor switches between them over time and
// the FE would ACCUMULATE one pane per resident as differently-tagged frames
// arrive under that one connection -- so a long-lived tab would show both
// boxes purely by having witnessed both anchors at some point. `panes`
// starts empty on mount, so a fresh page load would only ever show whichever
// resident the anchor happened to be on at that instant: one box.
//
// Now every resident gets its OWN connection, pinned to that resident from
// the moment it opens -- there is no shared anchor to lose track of.

const TOK_HISTORY_CAP = 40;

// Grace-hold (unchanged from the original single-stream design, now
// applied PER RESIDENT instead of globally): an `idle` frame arrives in the
// brief gaps between a model's rapid follow-up / sub-agent calls (observed
// ~5-6s). Flipping done:true on the first idle frame made the LIVE/DONE
// badge oscillate every gap. Instead, an idle frame arms a timer local to
// THIS resident's own stream; only a SUSTAINED idle (no reset/delta for
// RECENT_HOLD_MS) flips that one pane to done. A per-resident timer (not one
// shared timer flipping every pane at once, as before) is required now that
// each resident's live/idle state is genuinely independent of the others'.
const RECENT_HOLD_MS = 10000;

export interface GenPane {
  paneKey: string;
  generation_id: string;
  model_tag: string | null;
  text: string;
  done: boolean;
  lastTokS: number | null;
  lastFrameAt: number;
  tokHistory: number[];
}

export interface UseLiveStreamResult {
  panes: Record<string, GenPane>;
  connected: boolean;
  error: string | null;
}

const SSE_URL = '/ui/live/output/stream';
const RECONNECT_MS = 2000;

/**
 * PURE. Applies one already-parsed, non-idle SSE frame for a given resident
 * tag to the previous panes record. Exported and tested directly (two
 * residents' frames applied here must populate two INDEPENDENT
 * keys, never collide into one) because this is the one piece of
 * useLiveStream's per-connection logic that is meaningfully testable
 * without a DOM/effect-running harness -- the connection-diffing effect and
 * the fetch/stream-reading loop around this function are not).
 * `tag` is the resident this frame's OWN connection was opened
 * for, never parsed from the frame -- see the comment at streamResident's
 * call site for why that's sound.
 */
export function applyFrame(
  tag: string,
  prev: Record<string, GenPane>,
  frame: Pick<LiveOutputFrame, 'generation_id' | 'text' | 'done' | 'reset' | 'tok_s'>,
): Record<string, GenPane> {
  const genId = frame.generation_id;
  if (!genId) return prev;

  const existing = prev[tag];
  if (frame.reset) {
    return {
      ...prev,
      [tag]: {
        paneKey: tag,
        generation_id: genId,
        model_tag: tag,
        text: frame.text,
        done: frame.done ?? false,
        lastTokS: frame.tok_s ?? null,
        lastFrameAt: Date.now(),
        tokHistory: frame.tok_s != null ? [frame.tok_s] : [],
      },
    };
  }
  const prevHistory = existing?.tokHistory ?? [];
  const tokHistory =
    frame.tok_s != null ? [...prevHistory, frame.tok_s].slice(-TOK_HISTORY_CAP) : prevHistory;
  return {
    ...prev,
    [tag]: {
      paneKey: tag,
      generation_id: genId,
      model_tag: tag,
      text: (existing?.text ?? '') + frame.text,
      done: frame.done ?? existing?.done ?? false,
      lastTokS: frame.tok_s ?? existing?.lastTokS ?? null,
      lastFrameAt: Date.now(),
      tokHistory,
    },
  };
}

/** PURE. Flips one tag's existing pane to done:true; a no-op if that tag has no pane yet. */
export function markPaneDone(tag: string, prev: Record<string, GenPane>): Record<string, GenPane> {
  const existing = prev[tag];
  if (!existing) return prev;
  return { ...prev, [tag]: { ...existing, done: true } };
}

/**
 * ONE hook call, called once from Dashboard(). Internally opens/closes a
 * connection PER TAG in `residentTags` via a single effect that diffs the
 * desired tag set against what's currently open -- it does NOT call a hook
 * per tag (that would vary this hook's own call count across renders, a
 * genuine rules-of-hooks violation, unlike calling hooks inside a separate
 * child component instantiated via .map(), which is fine). The per-tag
 * connection logic below (`streamResident`) is a plain async function, not
 * a hook, so calling it a variable number of times is safe.
 *
 * Effect dependency is `residentTags`' CONTENT (sorted + joined), not the
 * array's identity: `residentTags` is a fresh array reference on every
 * ~1s /status poll tick even when the underlying resident SET hasn't
 * changed, and keying on identity would tear down and reopen every stream
 * every second.
 */
export function useLiveStream(residentTags: string[]): UseLiveStreamResult {
  const [panes, setPanes] = useState<Record<string, GenPane>>({});
  const [connectedTags, setConnectedTags] = useState<Record<string, boolean>>({});
  const [errorsByTag, setErrorsByTag] = useState<Record<string, string>>({});
  const connections = useRef<Record<string, AbortController>>({});

  const tagsKey = [...residentTags].sort().join('\u0000');

  useEffect(() => {
    const desired = new Set(residentTags);

    // Close the connection AND drop the pane for any tag no longer present
    // (the resident unloaded) -- "close on disappear, no leak". This is a
    // DIFFERENT event from a still-resident model going idle: idle keeps its
    // pane (done:true, greyed), this removes it entirely.
    for (const tag of Object.keys(connections.current)) {
      if (desired.has(tag)) continue;
      connections.current[tag].abort();
      delete connections.current[tag];
      setPanes(prev => {
        if (!(tag in prev)) return prev;
        const { [tag]: _drop, ...rest } = prev;
        return rest;
      });
      setConnectedTags(prev => {
        if (!(tag in prev)) return prev;
        const { [tag]: _drop, ...rest } = prev;
        return rest;
      });
      setErrorsByTag(prev => {
        if (!(tag in prev)) return prev;
        const { [tag]: _drop, ...rest } = prev;
        return rest;
      });
    }

    // Open a connection for any newly-appeared tag. Tags already open are
    // left completely alone -- no thrash on an unchanged set.
    for (const tag of desired) {
      if (connections.current[tag]) continue;
      const controller = new AbortController();
      connections.current[tag] = controller;
      void streamResident(tag, controller.signal, {
        onFrame: updater => setPanes(updater),
        onConnected: v => setConnectedTags(prev => ({ ...prev, [tag]: v })),
        onError: msg =>
          setErrorsByTag(prev => {
            if (msg === null) {
              if (!(tag in prev)) return prev;
              const { [tag]: _drop, ...rest } = prev;
              return rest;
            }
            return { ...prev, [tag]: msg };
          }),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tagsKey]);

  // True unmount: close every open connection.
  useEffect(() => {
    return () => {
      for (const controller of Object.values(connections.current)) controller.abort();
    };
  }, []);

  // Vacuously true with zero residents (nothing to be disconnected from);
  // otherwise true only when every currently-desired stream is connected.
  const connected = residentTags.every(tag => connectedTags[tag] === true);
  const error = residentTags.map(tag => errorsByTag[tag]).find(e => e != null) ?? null;

  return { panes, connected, error };
}

interface StreamCallbacks {
  onFrame: (updater: (prev: Record<string, GenPane>) => Record<string, GenPane>) => void;
  onConnected: (v: boolean) => void;
  onError: (msg: string | null) => void;
}

/**
 * Plain async function (NOT a hook) managing one resident's own SSE
 * connection + reconnect loop, called once per tag from the effect above.
 * Frame-reading logic (buffering, `data:` parsing, reset/idle/done
 * handling) is the same protocol handling the original single-stream design
 * had -- only the connection topology changed, not the wire format.
 */
async function streamResident(tag: string, signal: AbortSignal, cb: StreamCallbacks): Promise<void> {
  let idleTimer: number | undefined;
  const clearIdleTimer = () => {
    if (idleTimer !== undefined) {
      window.clearTimeout(idleTimer);
      idleTimer = undefined;
    }
  };
  signal.addEventListener('abort', clearIdleTimer);

  while (!signal.aborted) {
    try {
      const resp = await fetch(`${SSE_URL}?model_tag=${encodeURIComponent(tag)}`, {
        headers: { Accept: 'text/event-stream' },
        signal,
      });

      if (!resp.ok) {
        cb.onError(`SSE ${resp.status}`);
        cb.onConnected(false);
      } else {
        const reader = resp.body?.getReader();
        if (!reader) {
          cb.onError('No reader');
          cb.onConnected(false);
        } else {
          cb.onConnected(true);
          cb.onError(null);

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed.startsWith(':')) continue;
              if (!trimmed.startsWith('data: ')) continue;

              const raw = trimmed.slice(6);
              try {
                const frame = JSON.parse(raw) as LiveOutputFrame;

                // Idle: this resident has no active generation right now.
                // Grace-hold (see RECENT_HOLD_MS above) before flipping to
                // done, local to this resident's own timer.
                if (frame.idle) {
                  clearIdleTimer();
                  idleTimer = window.setTimeout(() => {
                    idleTimer = undefined;
                    cb.onFrame(prev => markPaneDone(tag, prev));
                  }, RECENT_HOLD_MS);
                  continue;
                }
                if (!frame.generation_id) continue;

                clearIdleTimer();

                // `tag` is the resident THIS CONNECTION WAS OPENED FOR, not
                // parsed from the frame. Every frame on a ?model_tag=<tag>
                // stream belongs to that resident by construction (verified
                // against live_stream.py's _tag_for_gid before building
                // this) -- no more generation_id fallback. That fallback
                // existed only because the old single-stream design had to
                // infer identity from the frame itself; this design knows
                // it from which connection the frame arrived on, before
                // even parsing the frame.
                cb.onFrame(prev => applyFrame(tag, prev, frame));
              } catch {
                // skip malformed frames
              }
            }
          }
          cb.onConnected(false);
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === 'AbortError') return;
      cb.onError(e instanceof Error ? e.message : String(e));
      cb.onConnected(false);
    }

    if (signal.aborted) return;
    await new Promise(r => setTimeout(r, RECONNECT_MS));
  }
}
