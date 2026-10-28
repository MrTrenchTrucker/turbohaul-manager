// Blob tab model delete. The Blob tab delete uses the same two-step prompt
// as the Models tab, so a 409 from the backend (blob still referenced by a
// manifest) is shown to the operator rather than lost in a status line:
// look up the manifests that reference
// the blob (hidden ones included), ask drop-or-keep, then delete in the right
// order and show a notice that carries the backend's reason.
//
// The module under test is read through a namespace import so that a missing
// export fails the behaviour test with an AssertionError rather than a
// bundle/import error.
// All fakes append to ONE shared call log, so the cross-function order
// (every manifest delete strictly before the blob delete) is observable.
// No DOM and no timers: renderToStaticMarkup and injected fakes only.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { ComponentType } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import * as BlobMod from './Blob';
import { DeleteBlobPrompt } from './Models';

const mod = BlobMod as unknown as Record<string, unknown>;

type Row = { model_tag: string; gguf_blob_sha256: string | null; hidden?: boolean };
type Plan = { ok: boolean; message: string; manifestTags: string[] };
type Result = { ok: boolean; message: string };
type TabDeleteArgs = { digest: string; name: string; manifestTags: string[]; dropManifests: boolean };
type RowProps = {
  colSpan: number;
  manifestCount: number;
  busy: boolean;
  onDropManifests: () => void;
  onKeepManifests: () => void;
  onCancel: () => void;
};

function exported<T>(name: string): T {
  const fn = mod[name];
  assert.equal(typeof fn, 'function', `Blob.tsx must export ${name}`);
  return fn as T;
}

const DIGEST = 'sha256:abc';

// Fakes that write into one shared log. Each call yields to the event loop
// once (setImmediate: deterministic ordering, no wall-clock time) so that a
// parallel implementation would show up as two calls in flight at once.
function makeDeleteFakes(opts: {
  failManifest?: Record<string, Error>;
  failBlob?: Error;
}) {
  const log: string[] = [];
  let inflight = 0;
  let maxInflight = 0;
  const enter = async () => {
    inflight += 1;
    maxInflight = Math.max(maxInflight, inflight);
    await new Promise<void>((r) => setImmediate(r));
    inflight -= 1;
  };
  const deleteManifestApi = async (tag: string): Promise<void> => {
    log.push(`manifest:${tag}`);
    await enter();
    const err = opts.failManifest?.[tag];
    if (err) throw err;
  };
  const deleteBlobApi = async (digest: string): Promise<void> => {
    log.push(`blob:${digest}`);
    await enter();
    if (opts.failBlob) throw opts.failBlob;
  };
  return { log, deleteManifestApi, deleteBlobApi, maxInflight: () => maxInflight };
}

const ROWS: Row[] = [
  { model_tag: 'a', gguf_blob_sha256: 'abc' },
  { model_tag: 'b', gguf_blob_sha256: 'abc', hidden: true },
  { model_tag: 'c', gguf_blob_sha256: 'zzz' },
  { model_tag: 'd', gguf_blob_sha256: null },
  { model_tag: 'a', gguf_blob_sha256: 'abc' },
];

// ---------------------------------------------------------------------------
// W1 manifestTagsForDigest
// ---------------------------------------------------------------------------

test('W1: manifestTagsForDigest keeps hidden rows, drops duplicates and null digests, accepts both digest spellings', () => {
  const fn = exported<(rows: Row[], digest: string) => string[]>('manifestTagsForDigest');
  assert.deepEqual(fn(ROWS, 'sha256:abc'), ['a', 'b'], 'W1: prefixed digest');
  assert.deepEqual(fn(ROWS, 'abc'), ['a', 'b'], 'W1: bare digest');
  assert.deepEqual(fn(ROWS, 'sha256:nomatch'), [], 'W1: no match gives an empty list');
  assert.deepEqual(fn([], 'sha256:abc'), [], 'W1: no rows gives an empty list');
});

test('W1: manifestTagsForDigest matches the digest exactly, never by prefix, and keeps listing order', () => {
  const fn = exported<(rows: Row[], digest: string) => string[]>('manifestTagsForDigest');
  assert.deepEqual(fn(ROWS, 'sha256:ab'), [], 'W1: "ab" is a prefix of "abc" and must not match');
  assert.deepEqual(
    fn([{ model_tag: 'abcd', gguf_blob_sha256: 'abcd' }, ...ROWS], 'sha256:abc'),
    ['a', 'b'],
    'W1: a longer digest that merely starts with the wanted one does not match',
  );
  const order = [
    { model_tag: 'z', gguf_blob_sha256: 'abc' },
    { model_tag: 'm', gguf_blob_sha256: 'abc' },
    { model_tag: 'b', gguf_blob_sha256: 'abc' },
  ];
  assert.deepEqual(fn(order, 'abc'), ['z', 'm', 'b'], 'W1: listing order is kept, not sorted');
});

// ---------------------------------------------------------------------------
// W2 loadBlobDeletePlan
// ---------------------------------------------------------------------------

test('W2: loadBlobDeletePlan success reads the manifest list exactly once and returns the matching tags', async () => {
  const fn = exported<(digest: string, deps: { getManifests: () => Promise<unknown> }) => Promise<Plan>>('loadBlobDeletePlan');
  let calls = 0;
  const plan = await fn(DIGEST, {
    getManifests: async () => {
      calls += 1;
      return { manifests: ROWS, total: ROWS.length, hidden_count: 1 };
    },
  });
  assert.equal(calls, 1, 'W2: getManifests must be called exactly once');
  assert.deepEqual(plan, { ok: true, message: '', manifestTags: ['a', 'b'] }, 'W2: success plan');
});

test('W2: loadBlobDeletePlan never throws; a failing manifest list gives ok=false with the error text and a retry hint', async () => {
  const fn = exported<(digest: string, deps: { getManifests: () => Promise<unknown> }) => Promise<Plan>>('loadBlobDeletePlan');
  const plan = await fn(DIGEST, {
    getManifests: async () => {
      throw new Error('GET /api/manifests 503');
    },
  });
  assert.equal(plan.ok, false, 'W2: ok must be false');
  assert.ok(plan.message.startsWith('Could not read the manifest list:'), `W2: message start, got: ${plan.message}`);
  assert.ok(plan.message.includes('GET /api/manifests 503'), `W2: message must carry the error text, got: ${plan.message}`);
  assert.ok(plan.message.includes('hit Refresh and retry'), `W2: message must carry the retry hint, got: ${plan.message}`);
  assert.deepEqual(plan.manifestTags, [], 'W2: no tags on failure');
});

// ---------------------------------------------------------------------------
// W3-W7 performBlobTabDelete
// ---------------------------------------------------------------------------

test('W3: performBlobTabDelete with dropManifests deletes every manifest, one at a time, strictly before the blob', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const f = makeDeleteFakes({});
  const result = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: ['a', 'b'], dropManifests: true },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.deepEqual(f.log, ['manifest:a', 'manifest:b', 'blob:sha256:abc'], 'W3: call order, blob last with its sha256: prefix');
  assert.equal(f.maxInflight(), 1, 'W3: deletes must run sequentially, never two in flight');
  assert.deepEqual(result, { ok: true, message: 'Deleted foo:8b and 2 manifest(s).' }, 'W3: result');
});

test('W4: a failing manifest delete aborts the whole delete: no later manifest, no blob, and the error text reaches the message', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const f = makeDeleteFakes({ failManifest: { b: new Error('DELETE /api/manifests/b 500') } });
  const result = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: ['a', 'b', 'c'], dropManifests: true },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.ok(!f.log.includes('blob:sha256:abc'), `W4: the blob delete must never be attempted, log: ${f.log.join(',')}`);
  assert.ok(!f.log.includes('manifest:c'), `W4: "c" must never be attempted, log: ${f.log.join(',')}`);
  assert.deepEqual(f.log, ['manifest:a', 'manifest:b'], 'W4: a was deleted, b was attempted, nothing else');
  assert.equal(result.ok, false, 'W4: ok must be false');
  assert.ok(result.message.startsWith('Delete failed:'), `W4: message start, got: ${result.message}`);
  assert.ok(result.message.includes('/api/manifests/b 500'), `W4: message must carry the error, got: ${result.message}`);
  assert.ok(result.message.endsWith('hit Refresh and retry.'), `W4: retry hint, got: ${result.message}`);
});

test('W5: the 409 from the blob delete reaches the notice with its backend detail', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const detail =
    'DELETE /api/delete 409 — blob sha256:abc is still referenced by 1 manifest(s): hidden-one. Delete or re-point them first (DELETE /api/manifests/<tag>), then retry.';
  const f = makeDeleteFakes({ failBlob: new Error(detail) });
  const result = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: ['a'], dropManifests: true },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.deepEqual(f.log, ['manifest:a', 'blob:sha256:abc'], 'W5: the blob delete was attempted after the manifest');
  assert.equal(result.ok, false, 'W5: ok must be false');
  assert.ok(result.message.startsWith('Delete failed:'), `W5: message start, got: ${result.message}`);
  assert.ok(result.message.includes('409'), `W5: the status must not be swallowed, got: ${result.message}`);
  assert.ok(result.message.includes('hidden-one'), `W5: the referencing tag from the backend must not be replaced by generic text, got: ${result.message}`);
  assert.ok(result.message.endsWith('hit Refresh and retry.'), `W5: retry hint, got: ${result.message}`);
});

test('W6: the keep path calls only the blob delete and words the notice for orphaned manifests', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const f = makeDeleteFakes({});
  const kept = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: ['a', 'b'], dropManifests: false },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.deepEqual(f.log, ['blob:sha256:abc'], 'W6: keep must not delete any manifest');
  assert.deepEqual(
    kept,
    { ok: true, message: 'Deleted foo:8b, kept 2 manifest(s) (now orphaned until re-linked).' },
    'W6: keep result with manifests',
  );

  const f2 = makeDeleteFakes({});
  const none = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: [], dropManifests: false },
    { deleteManifestApi: f2.deleteManifestApi, deleteBlobApi: f2.deleteBlobApi },
  );
  assert.deepEqual(f2.log, ['blob:sha256:abc'], 'W6: keep with no tags');
  assert.deepEqual(none, { ok: true, message: 'Deleted foo:8b.' }, 'W6: keep result without manifests');
});

test('W6: the keep path with a 409 rejection is a failure that carries the backend detail', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const f = makeDeleteFakes({
    failBlob: new Error('DELETE /api/delete 409 — blob sha256:abc is still referenced by 2 manifest(s): x, y.'),
  });
  const result = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: ['x', 'y'], dropManifests: false },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.deepEqual(f.log, ['blob:sha256:abc'], 'W6: only the blob delete was attempted');
  assert.equal(result.ok, false, 'W6: ok must be false');
  assert.ok(result.message.includes('409') && result.message.includes('x, y'), `W6: detail must be in the message, got: ${result.message}`);
});

test('W7: drop with zero manifest tags still deletes just the blob and says 0 manifest(s)', async () => {
  const fn = exported<(a: TabDeleteArgs, d: unknown) => Promise<Result>>('performBlobTabDelete');
  const f = makeDeleteFakes({});
  const result = await fn(
    { digest: DIGEST, name: 'foo:8b', manifestTags: [], dropManifests: true },
    { deleteManifestApi: f.deleteManifestApi, deleteBlobApi: f.deleteBlobApi },
  );
  assert.deepEqual(f.log, ['blob:sha256:abc'], 'W7: only the blob delete');
  assert.deepEqual(result, { ok: true, message: 'Deleted foo:8b and 0 manifest(s).' }, 'W7: result');
});

// ---------------------------------------------------------------------------
// W8 BlobDeleteRow markup
// ---------------------------------------------------------------------------

function renderRow(props: Partial<RowProps>): string {
  const Row = exported<ComponentType<RowProps>>('BlobDeleteRow');
  return renderToStaticMarkup(
    <table>
      <tbody>
        <Row
          colSpan={8}
          manifestCount={2}
          busy={false}
          onDropManifests={() => {}}
          onKeepManifests={() => {}}
          onCancel={() => {}}
          {...props}
        />
      </tbody>
    </table>,
  );
}

function disabledButtons(html: string): number {
  return (html.match(/<button[^>]*\sdisabled(?:=""|\s|>)/g) ?? []).length;
}

test('W8: BlobDeleteRow spans the table, shows the plural prompt and the three actions', () => {
  const html = renderRow({ manifestCount: 2 });
  assert.match(html, /<tr><td colSpan="8"/, `W8: a full-width cell inside a row, got: ${html}`);
  assert.ok(html.includes('2 manifests reference this model'), `W8: plural line, got: ${html}`);
  assert.ok(html.includes('Delete blob + 2 manifests'), `W8: drop button, got: ${html}`);
  assert.ok(html.includes('Delete blob, keep manifests'), `W8: keep button, got: ${html}`);
  assert.ok(html.includes('Cancel'), `W8: cancel button, got: ${html}`);
  assert.equal(disabledButtons(html), 0, `W8: nothing is disabled while idle, got: ${html}`);
});

test('W8: BlobDeleteRow with one manifest uses the singular wording of the Models-tab prompt', () => {
  const html = renderRow({ manifestCount: 1 });
  assert.ok(html.includes('1 manifest reference this model'), `W8: singular line, got: ${html}`);
  assert.ok(html.includes('Delete blob + 1 manifest'), `W8: singular drop button, got: ${html}`);
  assert.ok(!html.includes('1 manifests'), `W8: no "1 manifests", got: ${html}`);
});

test('W8: BlobDeleteRow passes colSpan through and disables all three buttons while busy', () => {
  const wide = renderRow({ colSpan: 5 });
  assert.match(wide, /colSpan="5"/, `W8: colSpan is a prop, not a constant, got: ${wide}`);
  const html = renderRow({ busy: true });
  assert.equal(disabledButtons(html), 3, `W8: all three buttons carry disabled while busy, got: ${html}`);
});

// ---------------------------------------------------------------------------
// W9-W11 pins
// ---------------------------------------------------------------------------

// Strips // and /* */ comments but leaves string and template literals alone,
// so a "//" inside "https://..." is not mistaken for a comment. Quote strings
// end at a newline: an apostrophe in JSX text then cannot swallow later code.
function stripComments(src: string): string {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === '/' && d === '/') {
      while (i < n && src[i] !== '\n') i += 1;
      continue;
    }
    if (c === '/' && d === '*') {
      const end = src.indexOf('*/', i + 2);
      i = end < 0 ? n : end + 2;
      out += ' ';
      continue;
    }
    if (c === "'" || c === '"' || c === '`') {
      out += c;
      i += 1;
      while (i < n && src[i] !== c && (c === '`' || src[i] !== '\n')) {
        if (src[i] === '\\' && i + 1 < n) {
          out += src[i];
          i += 1;
        }
        out += src[i];
        i += 1;
      }
      if (i < n && src[i] === c) {
        out += c;
        i += 1;
      }
      continue;
    }
    out += c;
    i += 1;
  }
  return out;
}

function readBlobSource(): string {
  return stripComments(readFileSync(join(process.cwd(), 'src/components/Blob.tsx'), 'utf8'));
}

test('W9: Blob.tsx no longer uses window.confirm, window.alert or the raw deleteJSON helper', () => {
  const source = readBlobSource();
  assert.ok(!source.includes('window.confirm'), 'W9: window.confirm must be gone from Blob.tsx');
  assert.ok(!source.includes('window.alert'), 'W9: window.alert must not appear in Blob.tsx');
  assert.ok(!source.includes('deleteJSON('), 'W9: the raw deleteJSON call must be gone from Blob.tsx');
});

test('W9: Blob.tsx wires the two-step flow through the api helpers and the shared prompt', () => {
  const source = readBlobSource();
  for (const needle of [
    'deleteBlobApi',
    'deleteManifestApi',
    'getManifests',
    'loadBlobDeletePlan(',
    'performBlobTabDelete(',
    '<BlobDeleteRow',
  ]) {
    assert.ok(source.includes(needle), `W9: Blob.tsx must contain ${needle}`);
  }
  assert.match(
    source,
    /import\s+(?:Models\s*,\s*)?\{[^}]*\bDeleteBlobPrompt\b[^}]*\}\s*from\s*['"]\.\/Models['"]/,
    'W9: DeleteBlobPrompt must be imported from ./Models',
  );
});

test('W10: the comment stripper used by the scans drops comments, keeps code and keeps "//" inside strings', () => {
  const stripped = stripComments(
    [
      '// window.confirm in a line comment',
      '/* window.alert in a block comment */',
      "const u = 'https://x.y'; window.confirm('real');",
    ].join('\n'),
  );
  assert.ok(!stripped.includes('window.alert'), 'W10: block comment content must be removed');
  assert.ok(!stripped.includes('line comment'), 'W10: line comment content must be removed');
  assert.ok(stripped.includes("window.confirm('real')"), 'W10: real code after a URL string must survive');
});

function walkSource(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      out.push(...walkSource(full));
    } else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name)) {
      out.push(full);
    }
  }
  return out;
}

test('W10: no non-test source file under src/ calls window.confirm or window.alert', () => {
  const root = join(process.cwd(), 'src');
  const files = walkSource(root);
  const names = files.map((f) => relative(root, f));
  assert.ok(files.length > 10, `W10: the walk must find the source tree, found ${files.length} files`);
  assert.ok(names.includes(join('components', 'Blob.tsx')), 'W10: the walk must include components/Blob.tsx');
  assert.ok(names.includes(join('components', 'Models.tsx')), 'W10: the walk must include components/Models.tsx');
  const offenders = files
    .filter((f) => /window\.(confirm|alert)/.test(stripComments(readFileSync(f, 'utf8'))))
    .map((f) => relative(root, f));
  assert.deepEqual(offenders, [], `W10: window.confirm / window.alert found in: ${offenders.join(', ')}`);
});

test('W11: DeleteBlobPrompt is still exported by Models.tsx and prints the plural count line', () => {
  assert.equal(typeof DeleteBlobPrompt, 'function', 'W11: Models.tsx must export DeleteBlobPrompt');
  const html = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={3} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  assert.ok(html.includes('3 manifests reference this model'), `W11: got: ${html}`);
});

// ---------------------------------------------------------------------------
// Wiring inside BlobDeleteRow and BlobContents. Static markup
// prints no event handlers and there is no DOM here, so these tests read the
// element tree of the hook-free BlobDeleteRow directly and pin the component
// wiring as source text (comments stripped, whitespace tolerant). Each pin
// guards one wrong-wiring class and its failure message names that class.
// ---------------------------------------------------------------------------

type ElementLike = { type?: unknown; props?: { children?: unknown } & Record<string, unknown> };

function findElementByType(node: unknown, type: unknown): ElementLike | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findElementByType(child, type);
      if (hit) return hit;
    }
    return null;
  }
  if (node === null || typeof node !== 'object') return null;
  const el = node as ElementLike;
  if (el.type === type) return el;
  return findElementByType(el.props?.children, type);
}

test('W12: BlobDeleteRow hands each of its own props to the matching DeleteBlobPrompt prop, not a neighbour', () => {
  const Row = exported<(p: RowProps) => unknown>('BlobDeleteRow');
  for (const busy of [false, true]) {
    const dropSpy = () => {};
    const keepSpy = () => {};
    const cancelSpy = () => {};
    const tree = Row({
      colSpan: 8,
      manifestCount: 2,
      busy,
      onDropManifests: dropSpy,
      onKeepManifests: keepSpy,
      onCancel: cancelSpy,
    });
    const prompt = findElementByType(tree, DeleteBlobPrompt);
    assert.ok(prompt !== null, 'W12: the returned tree must contain a DeleteBlobPrompt element');
    const props = prompt!.props as Record<string, unknown>;
    assert.equal(props.onDropManifests, dropSpy, 'W12: onDropManifests must be forwarded as given (wrong-callback wiring)');
    assert.equal(props.onKeepManifests, keepSpy, 'W12: onKeepManifests must be forwarded as given, not the drop callback (wrong-callback wiring)');
    assert.equal(props.onCancel, cancelSpy, 'W12: onCancel must be forwarded as given (wrong-callback wiring)');
    assert.equal(props.manifestCount, 2, 'W12: manifestCount must be forwarded unchanged');
    assert.equal(props.busy, busy, `W12: busy=${busy} must be forwarded unchanged`);
    assert.notEqual(props.onDropManifests, props.onKeepManifests, 'W12: the drop and keep callbacks must stay distinct');
  }
});

// Text between two anchors of the comment-stripped source, so a pin looks only
// at the function it is about (an exported helper with the same name elsewhere
// in the file cannot satisfy it). Missing anchors fail the test, never skip it.
function sliceBetween(source: string, from: string, to: string, label: string): string {
  const start = source.indexOf(from);
  assert.ok(start >= 0, `${label}: Blob.tsx must contain "${from}"`);
  const end = source.indexOf(to, start);
  assert.ok(end > start, `${label}: Blob.tsx must contain "${to}" after "${from}"`);
  return source.slice(start, end);
}

const COMPONENT_RETURN = '<div className="space-y-6">';

function startDeleteBody(source: string, label: string): string {
  return sliceBetween(source, 'const startDelete', 'const confirmDelete', label);
}

function confirmDeleteBody(source: string, label: string): string {
  return sliceBetween(source, 'const confirmDelete', COMPONENT_RETURN, label);
}

test('W13: startDelete asks loadBlobDeletePlan about the CLICKED row with the real getManifests (not a canned empty plan)', () => {
  const body = startDeleteBody(readBlobSource(), 'W13');
  assert.match(
    body,
    /await\s+loadBlobDeletePlan\(\s*m\.digest\s*,\s*\{\s*getManifests\s*\}\s*\)/,
    'W13: startDelete must `await loadBlobDeletePlan(m.digest, { getManifests })`; a hard-coded plan would show "0 manifests" and then 409 (canned-plan class)',
  );
});

test('W14: confirmDelete clears the pending row and then awaits refresh(), in that order', () => {
  const body = confirmDeleteBody(readBlobSource(), 'W14');
  assert.match(
    body,
    /setPendingDelete\(\s*null\s*\)\s*;\s*await\s+refresh\(\s*\)/,
    'W14: confirmDelete must `setPendingDelete(null); await refresh()`; without the awaited refresh the deleted model stays in the list, which is the stale-list failure (missing-refresh class)',
  );
});

test('W15: confirmDelete derives the notice error flag from the result, never a constant', () => {
  const body = confirmDeleteBody(readBlobSource(), 'W15');
  assert.match(
    body,
    /setNoticeErr\(\s*!\s*result\.ok\s*\)/,
    'W15: confirmDelete must call setNoticeErr(!result.ok) (error-flag-constant class)',
  );
  assert.ok(
    !/setNoticeErr\(\s*(?:false|true)\s*\)/.test(body),
    'W15: confirmDelete must not set the error flag to a constant (error-flag-constant class)',
  );
});

test('W16: the drop button deletes manifests and the keep button does not, and confirmDelete passes that choice to performBlobTabDelete', () => {
  const source = readBlobSource();
  assert.match(
    source,
    /onDropManifests=\{\s*\(\)\s*=>\s*void\s+confirmDelete\(\s*true\s*\)\s*\}/,
    'W16: onDropManifests must be `() => void confirmDelete(true)` (swapped-button class)',
  );
  assert.match(
    source,
    /onKeepManifests=\{\s*\(\)\s*=>\s*void\s+confirmDelete\(\s*false\s*\)\s*\}/,
    'W16: onKeepManifests must be `() => void confirmDelete(false)` (swapped-button class)',
  );
  const body = confirmDeleteBody(source, 'W16');
  assert.match(body, /await\s+performBlobTabDelete\(/, 'W16: confirmDelete must await performBlobTabDelete( (dropped-call class)');
  assert.match(body, /dropManifests\s*:\s*drop\b/, 'W16: confirmDelete must pass dropManifests: drop (choice-not-forwarded class)');
  assert.match(
    body,
    /deleteManifestApi\s*,\s*deleteBlobApi/,
    'W16: confirmDelete must pass the real deleteManifestApi and deleteBlobApi as deps (fake-deps class)',
  );
});

test('W17: the notice is red for a failure and green for a success (classes not swapped)', () => {
  const source = readBlobSource();
  const m = source.match(/noticeErr\s*\?\s*'(bg-red-950\/40[^']*)'\s*:\s*'(bg-emerald-950\/40[^']*)'/);
  assert.ok(
    m !== null,
    "W17: the notice className must be `noticeErr ? 'bg-red-950/40 ...' : 'bg-emerald-950/40 ...'` (red on the error branch, emerald on the other: swapped-colour class)",
  );
  assert.ok(
    m![1].includes('border-red-700') && m![1].includes('text-red-200'),
    `W17: the error branch must carry the red border and text, got: ${m![1]}`,
  );
  assert.ok(
    m![2].includes('border-emerald-700') && m![2].includes('text-emerald-200'),
    `W17: the success branch must carry the emerald border and text, got: ${m![2]}`,
  );
});

test('W18: the prompt row opens under the pending row only: it checks the digest AND the name', () => {
  const source = readBlobSource();
  const at = source.indexOf('<BlobDeleteRow');
  assert.ok(at >= 0, 'W18: Blob.tsx must render <BlobDeleteRow');
  const open = source.lastIndexOf('</tr>', at);
  assert.ok(open >= 0 && open < at, 'W18: <BlobDeleteRow must follow the model row');
  const condition = source.slice(open, at);
  assert.match(condition, /pendingDelete\s*!==\s*null/, 'W18: the row condition must test pendingDelete !== null');
  assert.match(condition, /pendingDelete\.digest\s*===\s*m\.digest/, 'W18: the row condition must compare the digest');
  assert.match(
    condition,
    /pendingDelete\.name\s*===\s*m\.name/,
    'W18: the row condition must also compare the name, or two tags on one blob both open the prompt (name-check-dropped class)',
  );
});

test('W19: the row Delete button starts the delete for THAT row, unchanged', () => {
  assert.match(
    readBlobSource(),
    /onClick=\{\s*\(\)\s*=>\s*void\s+startDelete\(\s*m\s*\)\s*\}/,
    'W19: the Delete button must be onClick={() => void startDelete(m)} (altered-row-argument class)',
  );
});
