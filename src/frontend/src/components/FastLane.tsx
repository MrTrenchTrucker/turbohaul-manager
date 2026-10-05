import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { getConfig, getFastLaneCensus, getStatus, putConfig } from '../api';
import type { FastLaneDiscoverable, FastLaneRule, FastLaneTagRanks } from '../api';

const BTN_CLS = 'rounded bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed';
// Same visual family as BTN_CLS, but this one is used OUTSIDE the desktop-only
// table (the Discovered section's Refresh control is rendered in both card
// and table modes, not toggled by hidden md:block/md:hidden) so it has to
// clear the 44px touch floor on mobile while staying compact at md+.
const TOUCH_BTN_CLS = 'rounded bg-slate-800 hover:bg-slate-700 px-3 min-h-[44px] md:min-h-0 md:py-1.5 text-sm md:text-xs text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed';
const TAG_KEYS: (keyof FastLaneTagRanks)[] = ['main', 'curator', 'compression', 'sub_agent', 'unclassified'];
const TAG_LABELS: Record<keyof FastLaneTagRanks, string> = {
  main: 'Main', curator: 'Curator', compression: 'Compression',
  sub_agent: 'Sub-agent', unclassified: 'Unclassified',
};

function emptyTagRanks(): FastLaneTagRanks {
  return { main: null, curator: null, compression: null, sub_agent: null, unclassified: null };
}

// ---------------------------------------------------------------------------
// On-theme tag-priority dropdown. A native <select>'s OPEN popup
// is rendered entirely by the OS on mobile -- no CSS reaches it -- so this is
// a hand-rolled WAI-ARIA "select-only combobox" (focus stays on the trigger
// the whole time; the popup is presentational, driven by
// aria-activedescendant, never itself focused). That single property is what
// makes "focus returns to the trigger on close" automatic rather than
// something to re-implement.
//
// The interaction state machine (stepListbox) and the data/position math
// (applyTagRank, computePopupPosition) are plain, pure, exported functions
// so they're directly unit-testable with node:test even though this repo's
// test harness has no DOM/jsdom to simulate real click/keydown events against
// the rendered component -- see FastLane.test.tsx.
// ---------------------------------------------------------------------------

const TAG_RANK_OPTIONS: (number | null)[] = [null, 1, 2, 3, 4, 5];

export function LintWarningsPanel({ warnings }: { warnings: string[] }) {
  // Exported and prop-driven ON PURPOSE. Inline in the page
  // body this was unreachable by any test: the harness renders statically and
  // cannot drive the async save handler that sets the state. As a component
  // taking props it renders in one line, the same way ResidentCard.test.tsx
  // already exercises LiveOutputBox. The subject is not changed to make a test
  // PASS -- it is made observable so a test can FAIL.
  if (warnings.length === 0) return null;
  return (
    <div
      data-testid="fastlane-lint-warnings"
      className="mb-3 rounded-lg border border-amber-700/60 bg-amber-950/30 p-3"
    >
      <div className="text-xs font-semibold uppercase tracking-wide text-amber-400 mb-1">
        Saved, but the manager flagged {warnings.length}{' '}
        {warnings.length === 1 ? 'problem' : 'problems'} with these rules
      </div>
      <ul className="list-disc list-inside space-y-1">
        {warnings.map((w, i) => (
          <li key={i} className="text-xs text-amber-200 font-mono break-words">
            {w}
          </li>
        ))}
      </ul>
    </div>
  );
}

// The RULES table's header row. Extracted so it can be asserted on directly:
// the whole FastLane component fetches on mount, which this package's runner
// (renderToStaticMarkup, no DOM, no fetch stub) cannot drive -- the same
// reason LintWarningsPanel is exported.
//
// The identity column is NOT "Address" any more. A rule identifies its
// client by container name (on the docker network) or by address (off it), so
// a header reading "Address" above a cell reading `gateway-svc` states
// something false about what is under it. `ruleIdentity` is what fills the
// cell; this names it.
//
// ⛔ The DISCOVERED census table further down has its own <th>Address</th> and
// it is CORRECT -- that table really does list observed addresses. Do not
// "fix" it to match this one.
export function RulesTableHead() {
  return (
    <thead className="bg-slate-900 text-xs uppercase text-slate-500">
      <tr>
        <th className="text-left px-4 py-2">#</th>
        <th className="text-left px-4 py-2">Identity</th>
        <th className="text-left px-4 py-2">Label</th>
        {TAG_KEYS.map((tag) => (
          <th key={tag} className="text-center px-2 py-2">{TAG_LABELS[tag]}</th>
        ))}
        <th className="text-right px-4 py-2"></th>
      </tr>
    </thead>
  );
}

export function ruleIdentity(rule: { address?: string | null; container_name?: string | null }): string {
  // What this rule identifies its client BY. Address rules keep
  // rendering exactly as before; a container_name rule shows the name
  // instead of an empty cell. Also used as the React key -- keying on
  // `address` alone would give every name rule the same undefined key.
  return rule.container_name || rule.address || '';
}

export function extractLintWarnings(result: unknown): string[] {
  // Defensive on purpose: this reads a SERVER payload. An older manager, a
  // proxy that rewrites the body, or a future shape change must degrade to
  // "no warnings to show" rather than throwing inside a save handler and
  // turning a successful save into a failed one on screen. Non-string
  // entries are dropped rather than String()-ed, so a nested object can
  // never render as "[object Object]" to an operator.
  if (!result || typeof result !== 'object') return [];
  const raw = (result as { warnings?: unknown }).warnings;
  if (!Array.isArray(raw)) return [];
  return raw.filter((w): w is string => typeof w === 'string' && w.length > 0);
}

export function formatTagRankOption(n: number | null): string {
  return n === null ? '—' : String(n);
}

export function applyTagRank(
  rules: FastLaneRule[],
  i: number,
  tag: keyof FastLaneTagRanks,
  rank: number | null,
): FastLaneRule[] {
  return rules.map((r, idx) =>
    idx === i ? { ...r, tag_ranks: { ...(r.tag_ranks ?? emptyTagRanks()), [tag]: rank } } : r,
  );
}

export type ListboxState = { open: boolean; activeIndex: number };

export type ListboxAction =
  | { type: 'open'; activeIndex: number }
  | { type: 'close' }
  | { type: 'move'; delta: 1 | -1; optionCount: number }
  | { type: 'home' }
  | { type: 'end'; optionCount: number }
  | { type: 'commitActive' }
  | { type: 'commitIndex'; index: number };

export const CLOSED_LISTBOX_STATE: ListboxState = { open: false, activeIndex: -1 };

// Pure: given the current open/highlight state and one discrete user action,
// returns the next state plus, ONLY on an actual commit (Enter/Space on the
// active option, or a direct option click), the index that was committed.
// This is the deliberate, single, testable commit path the keyboard-safety
// requirement calls for -- arrow-key browsing (`move`/`home`/`end`) NEVER returns
// commitIndex, only `commitActive`/`commitIndex` do, so highlighting an
// option while arrowing through it can never be silently treated as picking
// it.
export function stepListbox(
  state: ListboxState,
  action: ListboxAction,
): { state: ListboxState; commitIndex?: number } {
  switch (action.type) {
    case 'open':
      return { state: { open: true, activeIndex: action.activeIndex } };
    case 'close':
      return { state: CLOSED_LISTBOX_STATE };
    case 'move': {
      if (!state.open) return { state };
      const next = Math.min(Math.max(state.activeIndex + action.delta, 0), action.optionCount - 1);
      return { state: { open: true, activeIndex: next } };
    }
    case 'home':
      return state.open ? { state: { open: true, activeIndex: 0 } } : { state };
    case 'end':
      return state.open ? { state: { open: true, activeIndex: action.optionCount - 1 } } : { state };
    case 'commitActive':
      if (!state.open || state.activeIndex < 0) return { state };
      return { state: CLOSED_LISTBOX_STATE, commitIndex: state.activeIndex };
    case 'commitIndex':
      return { state: CLOSED_LISTBOX_STATE, commitIndex: action.index };
    default:
      return { state };
  }
}

type Rect = { top: number; bottom: number; left: number; right: number; width: number; height: number };

// Pure: no real layout engine needed to test this -- feed it plain numbers
// shaped like a DOMRect. Flips the popup above the trigger when there isn't
// room below, and clamps left so it never overflows the right edge of the
// viewport. `position: fixed` (not absolute) is what lets the result escape
// the table wrapper's `overflow-x-auto` clipping -- fixed is relative to the
// viewport, not any scrolling/overflow ancestor.
export function computePopupPosition(
  triggerRect: Rect,
  popupHeight: number,
  viewport: { width: number; height: number },
): { top: number; left: number; placement: 'above' | 'below' } {
  const fitsBelow = triggerRect.bottom + popupHeight <= viewport.height;
  const placement: 'above' | 'below' = fitsBelow || triggerRect.top < popupHeight ? 'below' : 'above';
  const top = placement === 'below' ? triggerRect.bottom + 4 : Math.max(4, triggerRect.top - popupHeight - 4);
  const minWidth = Math.max(triggerRect.width, 64);
  const left = Math.min(Math.max(4, triggerRect.left), Math.max(4, viewport.width - minWidth - 4));
  return { top, left, placement };
}

export function TagRankListbox({
  id,
  value,
  label,
  triggerClassName,
  onCommit,
}: {
  id: string;
  value: number | null;
  label: string;
  triggerClassName: string;
  onCommit: (rank: number | null) => void;
}) {
  const [state, setState] = useState<ListboxState>(CLOSED_LISTBOX_STATE);
  const [popupPos, setPopupPos] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const popupRef = useRef<HTMLUListElement | null>(null);

  const dispatch = (action: ListboxAction) => {
    const result = stepListbox(state, action);
    setState(result.state);
    if (result.commitIndex !== undefined) {
      onCommit(TAG_RANK_OPTIONS[result.commitIndex]);
    }
  };

  const openFromCurrentValue = () => {
    const idx = TAG_RANK_OPTIONS.indexOf(value);
    dispatch({ type: 'open', activeIndex: idx >= 0 ? idx : 0 });
  };

  // Measure + position after the popup mounts (real layout, so this cannot
  // run under renderToStaticMarkup), then keep it glued to
  // the trigger across scroll (any ancestor, via capture-phase listening --
  // scroll events don't bubble but do fire during capture) and window
  // resize, rAF-throttled so a scroll gesture doesn't spam layout reads.
  useLayoutEffect(() => {
    if (!state.open) {
      setPopupPos(null);
      return;
    }
    let raf = 0;
    const reposition = () => {
      const trigger = triggerRef.current;
      const popup = popupRef.current;
      if (!trigger || !popup) return;
      const r = trigger.getBoundingClientRect();
      const popupHeight = popup.offsetHeight || TAG_RANK_OPTIONS.length * 44;
      const result = computePopupPosition(r, popupHeight, { width: window.innerWidth, height: window.innerHeight });
      setPopupPos({ top: result.top, left: result.left });
    };
    reposition();
    const onScrollOrResize = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        reposition();
      });
    };
    window.addEventListener('scroll', onScrollOrResize, true);
    window.addEventListener('resize', onScrollOrResize);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScrollOrResize, true);
      window.removeEventListener('resize', onScrollOrResize);
    };
  }, [state.open]);

  useEffect(() => {
    if (!state.open) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target)) return;
      if (popupRef.current?.contains(target)) return;
      dispatch({ type: 'close' });
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.open]);

  return (
    <>
      <button
        ref={triggerRef}
        id={id}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={state.open}
        aria-controls={state.open ? `${id}-popup` : undefined}
        aria-activedescendant={state.open && state.activeIndex >= 0 ? `${id}-opt-${state.activeIndex}` : undefined}
        aria-label={label}
        className={`inline-flex items-center gap-1.5 rounded-lg bg-slate-900 border border-slate-700 px-3 min-h-[44px] text-sm text-slate-300 ${triggerClassName}`}
        onClick={() => (state.open ? dispatch({ type: 'close' }) : openFromCurrentValue())}
        onKeyDown={(e) => {
          switch (e.key) {
            case 'ArrowDown':
              e.preventDefault();
              if (!state.open) openFromCurrentValue();
              else dispatch({ type: 'move', delta: 1, optionCount: TAG_RANK_OPTIONS.length });
              break;
            case 'ArrowUp':
              e.preventDefault();
              if (!state.open) openFromCurrentValue();
              else dispatch({ type: 'move', delta: -1, optionCount: TAG_RANK_OPTIONS.length });
              break;
            case 'Home':
              if (state.open) {
                e.preventDefault();
                dispatch({ type: 'home' });
              }
              break;
            case 'End':
              if (state.open) {
                e.preventDefault();
                dispatch({ type: 'end', optionCount: TAG_RANK_OPTIONS.length });
              }
              break;
            case 'Enter':
            case ' ':
              e.preventDefault();
              if (state.open) dispatch({ type: 'commitActive' });
              else openFromCurrentValue();
              break;
            case 'Escape':
              if (state.open) {
                e.preventDefault();
                dispatch({ type: 'close' });
              }
              break;
            case 'Tab':
              if (state.open) dispatch({ type: 'close' });
              break;
            default:
              break;
          }
        }}
      >
        <span>{formatTagRankOption(value)}</span>
        <svg aria-hidden="true" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" className="h-3.5 w-3.5 shrink-0 text-slate-500">
          <path d="M6 8l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {state.open && (
        <ul
          ref={popupRef}
          id={`${id}-popup`}
          role="listbox"
          aria-label={label}
          className="z-50 min-w-[4.5rem] rounded-lg border border-slate-700 bg-slate-900 py-1 text-sm text-slate-300 shadow-lg"
          style={{ position: 'fixed', top: popupPos?.top ?? -9999, left: popupPos?.left ?? -9999 }}
        >
          {TAG_RANK_OPTIONS.map((opt, idx) => (
            <li
              key={opt ?? 'none'}
              id={`${id}-opt-${idx}`}
              role="option"
              aria-selected={value === opt}
              className={`flex min-h-[44px] cursor-pointer items-center px-3 ${idx === state.activeIndex ? 'bg-slate-800 text-slate-100' : ''} ${value === opt ? 'font-semibold' : ''}`}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => dispatch({ type: 'commitIndex', index: idx })}
            >
              {formatTagRankOption(opt)}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

// Census timestamps are seconds since epoch (per manager.py's time.time()),
// possibly fractional, and possibly missing -- narrowed defensively the same
// way this file already handles other `unknown`-typed census fields
// (String(...)/Array.isArray(...)). `text` is what a human scans; `title` is
// the exact value for anyone who needs it precisely, never discarded.
function formatRelativeTime(value: unknown): { text: string; title: string } {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return { text: '—', title: '—' };
  }
  const ms = value * 1000;
  const date = new Date(ms);
  const title = date.toLocaleString();
  const diffS = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (diffS < 5) return { text: 'just now', title };
  if (diffS < 60) return { text: `${diffS}s ago`, title };
  const diffMin = Math.round(diffS / 60);
  if (diffMin < 60) return { text: `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`, title };
  const diffH = Math.round(diffMin / 60);
  if (diffH < 24) return { text: `${diffH} hour${diffH === 1 ? '' : 's'} ago`, title };
  const diffD = Math.round(diffH / 24);
  return { text: `${diffD} day${diffD === 1 ? '' : 's'} ago`, title };
}

function formatCount(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString() : String(value ?? '—');
}

// "not yet available" for a promised counter this page has no backend
// source for at all, vs "none" for a real {} (Fast Lane ran, refused
// nothing) -- never a fabricated 0, which would read as "healthy, nothing
// refused" indistinguishably from an actually-healthy run.
export function formatRefusalsByReason(refusals: Record<string, number> | null): string {
  if (refusals === null) return 'not yet available';
  const entries = Object.entries(refusals);
  if (entries.length === 0) return 'none';
  return entries.map(([reason, count]) => `${reason}: ${formatCount(count)}`).join(', ');
}

// The container name the manager has confirmed for a Discovered row, or
// null when there is none (absent, null, empty, whitespace, or not a string).
export function discoveredName(entry: FastLaneDiscoverable): string | null {
  const raw = (entry as { container_name?: unknown }).container_name;
  if (typeof raw !== 'string') return null;
  const name = raw.trim();
  return name ? name : null;
}

// The rule "Add to Fast Lane" saves for a Discovered row: by container name
// when the row has a confirmed one (it survives a restart that changes the
// address), by address only when it has none. A name rule carries no
// `address` key at all.
export function discoveredRuleFor(entry: FastLaneDiscoverable): FastLaneRule {
  const name = discoveredName(entry);
  if (name !== null) return { container_name: name, label: '', tag_ranks: emptyTagRanks() };
  return { address: entry.address, label: '', tag_ranks: emptyTagRanks() };
}

// The rules list after "Add to Fast Lane" on a Discovered row, or null when
// the add is a duplicate: a rule with the same container name for a named
// row, or the same address for an unnamed row. The new rule goes at the END;
// existing rules are untouched and in order, and the input is not mutated.
export function rulesAfterAddingDiscovered(
  rules: FastLaneRule[],
  entry: FastLaneDiscoverable,
): FastLaneRule[] | null {
  const name = discoveredName(entry);
  const duplicate = name !== null
    ? rules.some((r) => r.container_name === name)
    : rules.some((r) => r.address === entry.address);
  if (duplicate) return null;
  return [...rules, discoveredRuleFor(entry)];
}

// The Discovered table head, extracted so a test can render it.
export function DiscoveredTableHead() {
  return (
    <thead className="bg-slate-900 text-xs uppercase text-slate-500">
      <tr>
        <th className="text-left px-4 py-2">Address</th>
        <th className="text-left px-4 py-2">Name</th>
        <th className="text-right px-4 py-2">Requests</th>
        <th className="hidden lg:table-cell text-left px-4 py-2">First seen</th>
        <th className="text-left px-4 py-2">Last seen</th>
        <th className="text-left px-4 py-2">Tag classes</th>
        <th className="text-left px-4 py-2">Models</th>
        <th className="text-right px-4 py-2"></th>
      </tr>
    </thead>
  );
}

// The Name cell of a Discovered row: the confirmed container name, or a dash.
export function DiscoveredNameCell({ entry }: { entry: FastLaneDiscoverable }) {
  return (
    <td className="px-4 py-2 font-mono" data-testid="discovered-name">
      {discoveredName(entry) ?? '—'}
    </td>
  );
}

export function alreadyInFastLane(
  entry: FastLaneDiscoverable,
  rules: FastLaneRule[] = [],
): boolean {
  // Trust the backend's own `assigned` field (fastlane_census_snapshot(),
  // manager.py) instead of re-deriving address equality here: `assigned`
  // already accounts for a container_name rule's RESOLVED addresses,
  // which this component has no way to see on its own -- a client-side
  // `r.address === entry.address` check is structurally blind to those, so
  // every container_name-covered row read as addable no matter what.
  //
  // A row with a confirmed name also counts as added when a rule already
  // names that container: a by-name save shows as Added at once, before the
  // backend has resolved the rule's addresses.
  if (Boolean((entry as { assigned?: unknown }).assigned)) return true;
  const name = discoveredName(entry);
  return name !== null && rules.some((r) => r.container_name === name);
}

export default function FastLane() {
  const [enabled, setEnabled] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [rules, setRules] = useState<FastLaneRule[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error' | 'warning'; text: string } | null>(null);
  const [lintWarnings, setLintWarnings] = useState<string[]>([]);
  const [census, setCensus] = useState<FastLaneDiscoverable[]>([]);
  const [censusBackendPending, setCensusBackendPending] = useState(false);
  const [censusError, setCensusError] = useState<string | null>(null);
  const [maxParallelSidecars, setMaxParallelSidecars] = useState<number | null>(null);
  // null = not yet fetched, or fetched from a backend that doesn't have this
  // field yet -- rendered as "not yet available", never a fabricated 0.
  const [refusalsByReason, setRefusalsByReason] = useState<Record<string, number> | null>(null);
  const [newContainerName, setNewContainerName] = useState('');

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((c) => {
        if (cancelled) return;
        const fastlane = (c.fastlane as Record<string, unknown>) || {};
        setEnabled(Boolean(fastlane.enabled));
        setConfigError((fastlane.config_error as string | null) ?? null);
        const loadedRules = (fastlane.rules as FastLaneRule[]) || [];
        setRules(loadedRules.map((r) => ({ ...r, tag_ranks: r.tag_ranks ?? emptyTagRanks() })));
        setLoaded(true);
      })
      .catch(console.error);
    getStatus()
      .then((s) => {
        if (cancelled) return;
        setMaxParallelSidecars(s.parallel_slots?.max ?? null);
        // Read defensively (cast, not a typed import) the same way
        // getConfig()'s response is read above -- the
        // api.ts type for StatusSnapshot has not been regenerated for this
        // field yet. Absent -> null ("not yet available"), never a
        // fabricated {} that would be indistinguishable from a real
        // "ran, refused nothing".
        const fastlaneStatus = (s as unknown as { fastlane?: { refusals_by_reason?: Record<string, number> } })
          .fastlane;
        setRefusalsByReason(fastlaneStatus?.refusals_by_reason ?? null);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshCensus = () => {
    getFastLaneCensus()
      .then((r) => {
        setCensusBackendPending(Boolean(r.backend_pending));
        setCensus(r.entries || []);
        setCensusError(null);
      })
      .catch((e) => setCensusError(e instanceof Error ? e.message : String(e)));
  };

  useEffect(() => {
    refreshCensus();
  }, []);

  // Whole-array save: config_put.py's runtime merge is a shallow, one-level
  // merge, so `rules` is replaced wholesale on every PUT, never
  // element-merged. Every row edit, reorder, add or remove saves the entire
  // array, not a partial patch.
  const saveRules = async (next: FastLaneRule[]) => {
    setSaving(true);
    setSaveMsg(null);
    setLintWarnings([]);
    try {
      const result = await putConfig({ fastlane: { rules: next } });
      setRules(next);
      // The server lints the rules and returns what it found. This
      // call used to discard the result, so a save that was accepted AND
      // misconfigured looked identical to a clean one -- "warn, not go
      // silent" was false for anyone using the UI.
      const warnings = extractLintWarnings(result);
      setLintWarnings(warnings);
      setSaveMsg(
        warnings.length > 0
          ? { type: 'warning', text: `Saved with ${warnings.length} warning${warnings.length === 1 ? '' : 's'}` }
          : { type: 'success', text: 'Saved' },
      );
    } catch (e) {
      setSaveMsg({ type: 'error', text: `Save failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSaving(false);
    }
  };

  const moveRule = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= rules.length) return;
    const next = [...rules];
    [next[i], next[j]] = [next[j], next[i]];
    void saveRules(next);
  };

  const removeRule = (i: number) => {
    const next = rules.filter((_, idx) => idx !== i);
    void saveRules(next);
  };

  const updateRuleLabel = (i: number, label: string) => {
    const next = rules.map((r, idx) => (idx === i ? { ...r, label } : r));
    setRules(next); // local only until Save
  };

  // Commit-on-select: applyTagRank + saveRules fire together the instant a
  // rank is picked (option click, or Enter/Space on the active option) --
  // not deferred to a later blur. The old <select>'s onBlur-driven persist
  // doesn't carry over cleanly to this widget (a select-only combobox never
  // moves DOM focus off the trigger while browsing options, so a stale
  // "blur later" hook would silently never fire for a pick that's the last
  // interaction before the user closes the tab). In practice this isn't a
  // behavior change: each of the 5 tag columns is its own independent
  // control, and the old onBlur already fired per-field the moment focus
  // left THAT field for a sibling one -- there was never real batching
  // across fields to begin with.
  const commitTagRank = (i: number, tag: keyof FastLaneTagRanks, rank: number | null) => {
    const next = applyTagRank(rules, i, tag, rank);
    setRules(next);
    void saveRules(next);
  };

  const addFromDiscovered = (entry: FastLaneDiscoverable) => {
    const next = rulesAfterAddingDiscovered(rules, entry);
    if (next === null) return;
    void saveRules(next);
  };

  // The identity model -- matching by container name first
  // -- was hand-editable only; the UI could never EMIT a container_name
  // rule. The Discovered table now offers a name on its own when the
  // manager has confirmed one for the row; this typed box stays for a
  // container the list cannot name (not yet confirmed, or not a container on
  // the shared network) and for rules added before that lookup ran. The
  // operator already knows their own docker network's container names, the
  // same way they already trust the label/tag-rank fields below.
  // Same shape as addFromDiscovered: append at the END of the CURRENT
  // client-held `rules` array, one whole-array PUT via saveRules -- does
  // NOT reorder any existing rule.
  const addContainerNameRule = () => {
    const name = newContainerName.trim();
    if (!name) return;
    if (rules.some((r) => r.container_name === name)) return;
    const next = [...rules, { container_name: name, label: '', tag_ranks: emptyTagRanks() }];
    setNewContainerName('');
    void saveRules(next);
  };

  // A container_name rule has no address, so it cannot be sharing one -- the
  // ambiguous-gateway warning below is meaningless for it and must not fire.
  const sharedAddress = (address: string | null | undefined) =>
    !!address && census.filter((e) => e.address === address).length > 1;

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-4 text-sm text-slate-300">
        Fast Lane changes the order requests are served in, and overrides the grace period, the
        idle-unload timer, and regular queue ordering while a rule is in effect. It is
        admission-only — it never interrupts a turn that is already running.
      </div>

      {!enabled && loaded && (
        <div className="rounded-lg border border-amber-700/50 bg-amber-950/30 p-4 text-sm text-amber-300">
          Fast Lane is off — switch it on in Settings.
        </div>
      )}

      {configError && (
        <div className="rounded-lg border border-rose-700/50 bg-rose-950/30 p-4 text-sm text-rose-300">
          The saved Fast Lane config was rejected and the feature booted disabled: {configError}
        </div>
      )}

      {maxParallelSidecars !== null && maxParallelSidecars >= 2 && (
        <div className="rounded-lg border border-sky-700/50 bg-sky-950/30 p-4 text-sm text-sky-300">
          Multiple parallel sidecars are configured (max {maxParallelSidecars}). Fast Lane reorders
          the shared queue; it does not pin a request to a specific sidecar.
        </div>
      )}

      <div className={enabled ? '' : 'opacity-50 pointer-events-none'}>
        <div className="space-y-6">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-200">Rules (priority order)</h2>
              {saveMsg && (
                <span
                  className={`text-sm ${
                    saveMsg.type === 'success'
                      ? 'text-emerald-400'
                      : saveMsg.type === 'warning'
                        ? 'text-amber-400'
                        : 'text-red-400'
                  }`}
                >
                  {saveMsg.text}
                </span>
              )}
            </div>
            <LintWarningsPanel warnings={lintWarnings} />
            <p className="text-xs text-slate-500 mb-3">
              Position in this list is the priority — row 1 is served before row 2. Blank on all
              five tag ranks means everything from that client is equal.
            </p>
            <div className="flex flex-col sm:flex-row gap-2 mb-4">
              <input
                type="text"
                value={newContainerName}
                onChange={(e) => setNewContainerName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') addContainerNameRule();
                }}
                className="flex-1 h-11 sm:h-9 bg-slate-900 border border-slate-700 rounded px-3 text-sm text-slate-200"
                placeholder="container name (docker network)"
              />
              <button
                className={TOUCH_BTN_CLS}
                disabled={saving || newContainerName.trim().length === 0}
                onClick={addContainerNameRule}
              >
                Add by container name
              </button>
            </div>
            {rules.length === 0 ? (
              <div className="text-slate-500 text-sm italic">
                No rules yet. Add an address from the Discovered table below, or a container name above.
              </div>
            ) : (
              <>
                {/* Mobile card layout (< md). One card per rule; every rank
                    select carries its own visible <label> since the <th>
                    text that labels the desktop table's columns doesn't
                    exist once rows aren't table rows. */}
                <div className="md:hidden space-y-3">
                  {rules.map((rule, i) => (
                    <div key={ruleIdentity(rule) || i} className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-baseline gap-2">
                            <span className="text-xs font-mono text-slate-500">{i + 1}</span>
                            <span className="font-mono text-slate-200 break-all">{ruleIdentity(rule)}</span>
                          </div>
                          {sharedAddress(rule.address) && (
                            <div className="text-xs text-amber-400 mt-1">
                              A rule on the address alone will apply to all callers sharing it — if
                              these are different callers they cannot be told apart here; add a tag
                              to narrow it.
                            </div>
                          )}
                        </div>
                        <div className="flex gap-2 shrink-0">
                          <button
                            className="rounded bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] min-w-[44px]"
                            disabled={saving || i === 0}
                            onClick={() => moveRule(i, -1)}
                            title="Move up"
                          >
                            ↑
                          </button>
                          <button
                            className="rounded bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] min-w-[44px]"
                            disabled={saving || i === rules.length - 1}
                            onClick={() => moveRule(i, 1)}
                            title="Move down"
                          >
                            ↓
                          </button>
                        </div>
                      </div>

                      <input
                        type="text"
                        value={rule.label ?? ''}
                        onChange={(e) => updateRuleLabel(i, e.target.value)}
                        onBlur={() => void saveRules(rules)}
                        className="w-full h-11 bg-slate-900 border border-slate-700 rounded px-3 text-sm text-slate-200"
                        placeholder="operator note"
                      />

                      <div>
                        <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Tag priority</div>
                        <div className="grid grid-cols-2 gap-3">
                          {TAG_KEYS.map((tag) => (
                            <label key={tag} className="block">
                              <span className="block text-xs uppercase tracking-wide text-slate-500 mb-1">
                                {TAG_LABELS[tag]}
                              </span>
                              <TagRankListbox
                                id={`tagrank-card-${i}-${tag}`}
                                value={rule.tag_ranks?.[tag] ?? null}
                                label={`${TAG_LABELS[tag]} priority`}
                                triggerClassName="w-full justify-between"
                                onCommit={(rank) => commitTagRank(i, tag, rank)}
                              />
                            </label>
                          ))}
                        </div>
                      </div>

                      <div className="flex justify-end">
                        <button
                          className="rounded bg-rose-900/40 text-rose-300 hover:bg-rose-900/60 disabled:opacity-50 min-h-[44px] px-4 text-sm"
                          disabled={saving}
                          onClick={() => removeRule(i)}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Desktop table (>= md). Same content as before; only the
                    wrapper (overflow-hidden -> overflow-x-auto, so content
                    wider than the viewport scrolls instead of being
                    unreachable) and the select touch target changed. */}
                <div className="hidden md:block rounded-lg border border-slate-700 bg-slate-950 overflow-x-auto">
                  <table className="w-full text-sm">
                    <RulesTableHead />
                    <tbody className="divide-y divide-slate-800">
                      {rules.map((rule, i) => (
                        <tr key={ruleIdentity(rule) || i} className="text-slate-300">
                          <td className="px-4 py-2 font-mono text-slate-500">{i + 1}</td>
                          <td className="px-4 py-2 font-mono">
                            {ruleIdentity(rule)}
                            {sharedAddress(rule.address) && (
                              <div className="text-xs text-amber-400 mt-1">
                                A rule on the address alone will apply to all callers sharing it — if
                                these are different callers they cannot be told apart here; add a tag
                                to narrow it.
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-2">
                            <input
                              type="text"
                              value={rule.label ?? ''}
                              onChange={(e) => updateRuleLabel(i, e.target.value)}
                              onBlur={() => void saveRules(rules)}
                              className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                              placeholder="operator note"
                            />
                          </td>
                          {TAG_KEYS.map((tag) => (
                            <td key={tag} className="px-2 py-2 text-center">
                              <TagRankListbox
                                id={`tagrank-table-${i}-${tag}`}
                                value={rule.tag_ranks?.[tag] ?? null}
                                label={`${TAG_LABELS[tag]} priority for ${rule.address}`}
                                triggerClassName="min-w-[3.75rem] justify-center"
                                onCommit={(rank) => commitTagRank(i, tag, rank)}
                              />
                            </td>
                          ))}
                          <td className="px-4 py-2 text-right whitespace-nowrap">
                            <button
                              className={BTN_CLS}
                              disabled={saving || i === 0}
                              onClick={() => moveRule(i, -1)}
                              title="Move up"
                            >
                              ↑
                            </button>{' '}
                            <button
                              className={BTN_CLS}
                              disabled={saving || i === rules.length - 1}
                              onClick={() => moveRule(i, 1)}
                              title="Move down"
                            >
                              ↓
                            </button>{' '}
                            <button
                              className="rounded bg-rose-900/40 text-rose-300 text-xs hover:bg-rose-900/60 px-3 py-1.5 disabled:opacity-50"
                              disabled={saving}
                              onClick={() => removeRule(i)}
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            <p className="text-xs text-slate-500 mt-3">
              Embedding requests carry no address and are always normal priority.
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-200">Discovered</h2>
              <button className={TOUCH_BTN_CLS} onClick={refreshCensus}>Refresh</button>
            </div>
            {censusError && (
              <div className="text-amber-400 text-sm mb-3">⚠ {censusError}</div>
            )}
            {censusBackendPending && (
              <div className="text-slate-500 text-sm italic mb-3">
                Census backend not yet available.
              </div>
            )}
            {census.length === 0 ? (
              <div className="text-slate-500 text-sm italic">No addresses observed yet.</div>
            ) : (
              <>
                {/* Mobile card layout (< md). First seen is dropped here too
                    (not just at the desktop md tier) -- of the two
                    timestamps, last-seen is the one that answers "is this
                    live?"; models collapse behind a count so 8+ names don't
                    dominate a card the way they dominated a ~90px column. */}
                <div className="md:hidden space-y-3">
                  {census.map((entry) => {
                    const lastSeen = formatRelativeTime(entry.last_seen);
                    const already = alreadyInFastLane(entry, rules);
                    const confirmedName = discoveredName(entry);
                    const models = Array.isArray(entry.models_seen) ? entry.models_seen : [];
                    const tagClasses = Array.isArray(entry.tag_classes_seen) ? entry.tag_classes_seen : [];
                    return (
                      <div key={entry.address} className="rounded-lg border border-slate-700 bg-slate-950 p-4 space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <span className="font-mono text-slate-200 break-all">{entry.address}</span>
                          <span className="font-mono text-slate-400 text-sm tabular-nums shrink-0">
                            {formatCount(entry.request_count)} reqs
                          </span>
                        </div>
                        {confirmedName !== null && (
                          <div className="font-mono text-xs text-slate-500 break-all" data-testid="discovered-name">
                            {confirmedName}
                          </div>
                        )}
                        <div className="text-xs text-slate-500" title={lastSeen.title}>
                          last seen {lastSeen.text}
                        </div>
                        <div className="text-xs text-slate-500">
                          {tagClasses.length > 0 ? tagClasses.join(' · ') : '—'}
                        </div>
                        {models.length > 0 ? (
                          <details className="text-xs text-slate-500">
                            <summary className="cursor-pointer">
                              {models.length} model{models.length === 1 ? '' : 's'}
                            </summary>
                            <div className="mt-1 text-slate-400">{models.join(', ')}</div>
                          </details>
                        ) : (
                          <div className="text-xs text-slate-500">—</div>
                        )}
                        <button
                          className="w-full h-11 rounded bg-slate-800 hover:bg-slate-700 text-sm text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                          data-testid="discovered-add"
                          disabled={saving || already}
                          onClick={() => addFromDiscovered(entry)}
                        >
                          {already ? 'Added' : 'Add to Fast Lane'}
                        </button>
                      </div>
                    );
                  })}
                </div>

                {/* Desktop table (>= md). Same content as before; wrapper
                    scrolls instead of clipping, timestamps are relative
                    (exact value in title), Models gets a width cap, and
                    First seen drops at md and comes back at lg. */}
                <div className="hidden md:block rounded-lg border border-slate-700 bg-slate-950 overflow-x-auto">
                  <table className="w-full text-sm">
                    <DiscoveredTableHead />
                    <tbody className="divide-y divide-slate-800">
                      {census.map((entry) => {
                        const firstSeen = formatRelativeTime(entry.first_seen);
                        const lastSeen = formatRelativeTime(entry.last_seen);
                        return (
                          <tr key={entry.address} className="text-slate-300">
                            <td className="px-4 py-2 font-mono">{entry.address}</td>
                            <DiscoveredNameCell entry={entry} />
                            <td className="px-4 py-2 font-mono text-right">
                              {String(entry.request_count ?? '—')}
                            </td>
                            <td className="hidden lg:table-cell px-4 py-2 text-slate-500" title={firstSeen.title}>
                              {firstSeen.text}
                            </td>
                            <td className="px-4 py-2 text-slate-500" title={lastSeen.title}>
                              {lastSeen.text}
                            </td>
                            <td className="px-4 py-2 text-slate-500">
                              {Array.isArray(entry.tag_classes_seen) ? entry.tag_classes_seen.join(', ') : '—'}
                            </td>
                            <td className="px-4 py-2 text-slate-500 max-w-[22rem]">
                              {Array.isArray(entry.models_seen) ? entry.models_seen.join(', ') : '—'}
                            </td>
                            <td className="px-4 py-2 text-right">
                              <button
                                className={BTN_CLS}
                                data-testid="discovered-add"
                                disabled={saving || alreadyInFastLane(entry, rules)}
                                onClick={() => addFromDiscovered(entry)}
                              >
                                {alreadyInFastLane(entry, rules) ? 'Added' : 'Add to Fast Lane'}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>

          <div className="rounded-lg border border-slate-700 bg-slate-950 p-4">
            <h2 className="text-sm font-semibold text-slate-300 mb-3">Counters</h2>
            <dl className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Refusals by reason</dt>
                <dd className="font-mono text-slate-200 mt-1">{formatRefusalsByReason(refusalsByReason)}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Served / jumped per class</dt>
                <dd className="text-slate-600 italic mt-1">not yet available</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Fairness firings</dt>
                <dd className="text-slate-600 italic mt-1">not yet available</dd>
              </div>
            </dl>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4 text-sm text-slate-500 italic">
            Live jump panel and last-ten-jumps log are not yet wired to a backend data source and
            are intentionally omitted rather than backed by fabricated data.
          </div>
        </div>
      </div>
    </div>
  );
}
