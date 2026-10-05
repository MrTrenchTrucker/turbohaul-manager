// Regression: Blob -> Models -> Edit showed
// exactly one spec_type control (in the "Primary" block) and no "Speculative
// Decoding" heading anywhere, so a user hunting for speculative-decode
// settings by category name found nothing. spec_type is now `dualRender`
// (flagsSchema.ts) and renders in BOTH places, driven by the same
// flagValues/enabledFlags state -- these tests prove the category section
// actually contains it, prove the other four primary flags (cache_type_k,
// cache_type_v, temp, top_p) are UNCHANGED (still Primary-only), and prove
// that driving both render sites with the same value produces the same
// selection (the "not a copy" requirement), without inventing an interactive
// harness this test runner can't do (renderToStaticMarkup has no DOM events).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  CategorySection,
  FlagInput,
  computeReasoningBudgetConflict,
  ReasoningBudgetConflictHint,
  REASONING_BUDGET_HINT_ID,
  groupIntoTiles,
  selectModelManifestRows,
  summaryRowToDetail,
  performRename,
  ModelTileCard,
  ModelInfoCard,
  ManifestRow,
  InlineTextPrompt,
  DeleteBlobPrompt,
  type ManifestDetailLite,
  type TileGroup,
  type BlobEntry,
} from './Models';
import { FLAGS_SCHEMA, getCategorySectionFlags } from '../flagsSchema';
import type { ManifestSummaryRow } from '../api';

const specTypeSpec = FLAGS_SCHEMA.find((f) => f.name === 'spec_type')!;
const nPredictSpec = FLAGS_SCHEMA.find((f) => f.name === 'n_predict')!;

test('spec_type is present in the Speculative Decoding category section flags', () => {
  const names = getCategorySectionFlags('Speculative Decoding').map((f) => f.name);
  assert.ok(
    names.includes('spec_type'),
    `expected 'spec_type' in the Speculative Decoding category section, got: ${names.join(', ')}`,
  );
});

test('spec_type appears exactly once in the Speculative Decoding category section flags', () => {
  const names = getCategorySectionFlags('Speculative Decoding').map((f) => f.name);
  const count = names.filter((n) => n === 'spec_type').length;
  assert.equal(count, 1, `expected exactly one 'spec_type' entry, got ${count} (${names.join(', ')})`);
});

test('spec_type still appears in the Primary set (Primary block is unaffected)', () => {
  const primaryNames = FLAGS_SCHEMA.filter((f) => f.primary).map((f) => f.name);
  assert.ok(primaryNames.includes('spec_type'), `expected 'spec_type' still primary, got: ${primaryNames.join(', ')}`);
});

test('the other four primary flags stay Primary-only -- NOT duplicated into their category sections', () => {
  const cases: Array<{ cat: 'KV Cache' | 'Sampling'; name: string }> = [
    { cat: 'KV Cache', name: 'cache_type_k' },
    { cat: 'KV Cache', name: 'cache_type_v' },
    { cat: 'Sampling', name: 'temp' },
    { cat: 'Sampling', name: 'top_p' },
  ];
  for (const { cat, name } of cases) {
    const names = getCategorySectionFlags(cat).map((f) => f.name);
    assert.ok(!names.includes(name), `expected '${name}' to stay OUT of the ${cat} category section, got: ${names.join(', ')}`);
  }
});

test('CategorySection renders a visible "Speculative Decoding" heading with spec_type inside it', () => {
  const flags = getCategorySectionFlags('Speculative Decoding');
  const html = renderToStaticMarkup(
    <CategorySection
      cat="Speculative Decoding"
      flags={flags}
      values={{ spec_type: 'draft-dflash' }}
      enabledFlags={new Set(['spec_type'])}
      onChange={() => {}}
      onToggle={() => {}}
      defaultOpen={true}
    />,
  );
  assert.match(html, /Speculative Decoding/, `expected the category heading text, got: ${html}`);
  assert.match(html, /spec_type/, `expected the spec_type row inside the category section, got: ${html}`);
});

test('CategorySection\'s spec_type control reflects the selected value (draft-dspark) via a real <select>', () => {
  const flags = getCategorySectionFlags('Speculative Decoding');
  const html = renderToStaticMarkup(
    <CategorySection
      cat="Speculative Decoding"
      flags={flags}
      values={{ spec_type: 'draft-dspark' }}
      enabledFlags={new Set(['spec_type'])}
      onChange={() => {}}
      onToggle={() => {}}
      defaultOpen={true}
    />,
  );
  assert.match(
    html,
    /<option[^>]*value="draft-dspark"[^>]*selected[^>]*>draft-dspark<\/option>/,
    `expected draft-dspark selected in the category section's select, got: ${html}`,
  );
});

test('the Primary-block control and the category-section control render the SAME selection when driven by the same state (not a copy)', () => {
  const sharedValue = 'draft-dflash';
  const sharedEnabled = true;

  // What the ★ Primary block renders for spec_type (Models.tsx primaryFlags.map -> FlagInput).
  const primaryHtml = renderToStaticMarkup(
    <FlagInput spec={specTypeSpec} value={sharedValue} enabled={sharedEnabled} onChange={() => {}} onToggle={() => {}} />,
  );

  // What the category section renders for spec_type, driven by the identical value/enabled.
  const categoryHtml = renderToStaticMarkup(
    <CategorySection
      cat="Speculative Decoding"
      flags={[specTypeSpec]}
      values={{ spec_type: sharedValue }}
      enabledFlags={new Set(sharedEnabled ? ['spec_type'] : [])}
      onChange={() => {}}
      onToggle={() => {}}
      defaultOpen={true}
    />,
  );

  const selectedOptionPattern = /<option[^>]*value="draft-dflash"[^>]*selected[^>]*>draft-dflash<\/option>/;
  assert.match(primaryHtml, selectedOptionPattern, `Primary control did not show draft-dflash selected: ${primaryHtml}`);
  assert.match(categoryHtml, selectedOptionPattern, `Category control did not show draft-dflash selected: ${categoryHtml}`);
});

// ---------------------------------------------------------------------------
// reasoning_budget >= n_predict inline conflict hint.
// A manifest can hold a thinking budget larger than the output ceiling the
// caller actually receives -- the model spends its whole budget thinking
// and returns nothing. Both values are individually legal (see each field's
// own bounds in flagsSchema.ts), so nothing else catches this. n_predict:
// -1 (unbounded) must NEVER trip the hint -- that is the required negative
// control, since a false alarm there would train operators to ignore it.
// ---------------------------------------------------------------------------

test('computeReasoningBudgetConflict: REQUIRED negative control -- n_predict -1 (unbounded) never fires, regardless of reasoning_budget size', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: -1, reasoning_budget: 999_999 },
    new Set(['n_predict', 'reasoning_budget']),
  );
  assert.equal(result, null, `expected no conflict for n_predict=-1, got: ${JSON.stringify(result)}`);
});

test('computeReasoningBudgetConflict: fires when reasoning_budget >= n_predict and n_predict > 0', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: 2048, reasoning_budget: 3192 },
    new Set(['n_predict', 'reasoning_budget']),
  );
  assert.deepEqual(result, { nPredict: 2048, reasoningBudget: 3192 });
});

test('computeReasoningBudgetConflict: fires at exact equality (reasoning_budget === n_predict, the ">=" boundary)', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: 2048, reasoning_budget: 2048 },
    new Set(['n_predict', 'reasoning_budget']),
  );
  assert.deepEqual(result, { nPredict: 2048, reasoningBudget: 2048 });
});

test('computeReasoningBudgetConflict: no conflict when reasoning_budget is safely below n_predict', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: 4096, reasoning_budget: 1024 },
    new Set(['n_predict', 'reasoning_budget']),
  );
  assert.equal(result, null, `expected no conflict, got: ${JSON.stringify(result)}`);
});

test('computeReasoningBudgetConflict: no conflict when reasoning_budget is -1 (unlimited), even with a small n_predict', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: 100, reasoning_budget: -1 },
    new Set(['n_predict', 'reasoning_budget']),
  );
  assert.equal(result, null, `expected no conflict, got: ${JSON.stringify(result)}`);
});

test('computeReasoningBudgetConflict: gated on enabledFlags, not just flagValues -- a DISABLED flag leaving a stale value behind must not fire', () => {
  // ModelEditor's onToggleFlag disable path removes the name from
  // enabledFlags but does NOT clear flagValues (see Models.tsx) -- a
  // conflict check that only looked at flagValues would false-positive here.
  const result = computeReasoningBudgetConflict(
    { n_predict: 2048, reasoning_budget: 999_999 }, // reasoning_budget value is STALE
    new Set(['n_predict']), // reasoning_budget is NOT enabled
  );
  assert.equal(result, null, `expected no conflict when reasoning_budget is disabled, got: ${JSON.stringify(result)}`);
});

test('computeReasoningBudgetConflict: no conflict when only n_predict is enabled (reasoning_budget omitted entirely)', () => {
  const result = computeReasoningBudgetConflict(
    { n_predict: 100 },
    new Set(['n_predict']),
  );
  assert.equal(result, null);
});

test('computeReasoningBudgetConflict: no conflict when only reasoning_budget is enabled (n_predict omitted entirely)', () => {
  const result = computeReasoningBudgetConflict(
    { reasoning_budget: 999_999 },
    new Set(['reasoning_budget']),
  );
  assert.equal(result, null);
});

test('ReasoningBudgetConflictHint renders both numbers and plain-English cause+effect text', () => {
  const html = renderToStaticMarkup(
    <ReasoningBudgetConflictHint nPredict={2048} reasoningBudget={3192} />,
  );
  assert.match(html, /3192/, `expected reasoning_budget value in the hint, got: ${html}`);
  assert.match(html, /2048/, `expected n_predict value in the hint, got: ${html}`);
  assert.match(html, /returns nothing/, `expected the plain-English effect statement, got: ${html}`);
  assert.ok(
    html.includes(`id="${REASONING_BUDGET_HINT_ID}"`),
    `expected the hint's own id attribute, got: ${html}`,
  );
});

test('ReasoningBudgetConflictHint is a polite live region (can appear while focus stays in a number input)', () => {
  const html = renderToStaticMarkup(
    <ReasoningBudgetConflictHint nPredict={2048} reasoningBudget={3192} />,
  );
  assert.match(html, /aria-live="polite"/, `expected aria-live="polite", got: ${html}`);
});

test('FlagInput applies aria-describedby to the widget when describedById is passed', () => {
  const html = renderToStaticMarkup(
    <FlagInput
      spec={nPredictSpec}
      value={2048}
      enabled={true}
      onChange={() => {}}
      onToggle={() => {}}
      describedById={REASONING_BUDGET_HINT_ID}
    />,
  );
  assert.match(
    html,
    new RegExp(`aria-describedby="${REASONING_BUDGET_HINT_ID}"`),
    `expected aria-describedby on the n_predict widget, got: ${html}`,
  );
});

test('FlagInput omits aria-describedby entirely when describedById is not passed (the other ~106 flags, unaffected)', () => {
  const html = renderToStaticMarkup(
    <FlagInput
      spec={nPredictSpec}
      value={2048}
      enabled={true}
      onChange={() => {}}
      onToggle={() => {}}
    />,
  );
  assert.ok(!html.includes('aria-describedby'), `expected no aria-describedby attribute at all, got: ${html}`);
});

// =============================================================================
// /blob/models three-level overhaul.
// renderToStaticMarkup/jsdom has NO layout engine and does not fire useEffect
// or DOM click events -- every test below proves STRUCTURE (which elements,
// which classes, which text, in which state-driven configuration), never
// rendered pixel geometry or a simulated click sequence. Page height, the size
// of the Edit button, and the actual click-through L1->L2->L3 flow are
// verified separately in a live browser, outside this harness -- disclosed
// here, not silently implied covered.
// =============================================================================

function makeSummaryRow(overrides: Partial<ManifestSummaryRow> = {}): ManifestSummaryRow {
  return {
    model_tag: 'alpha',
    kind: 'model',
    hidden: false,
    revision: 1,
    etag: '"1"',
    display_name: 'Alpha Model',
    gguf_blob_sha256: 'blob-a',
    ...overrides,
  };
}

function makeDetail(overrides: Partial<ManifestDetailLite> = {}): ManifestDetailLite {
  return {
    model_tag: 'alpha',
    displayName: 'Alpha Model',
    blobSha256: 'aaaa1111',
    hidden: false,
    revision: 1,
    etag: '"1"',
    fileSizeBytes: null,
    ...overrides,
  };
}

// -----------------------------------------------------------------------
// selectModelManifestRows -- the plugin-exclusion claim, tested
// directly rather than trusted from a component read. The old bottom list had
// NO kind filter, so plugin manifests appeared on /blob/models.
// -----------------------------------------------------------------------

test('selectModelManifestRows: drops kind==="plugin" rows -- the six plugin manifests never reach this page', () => {
  const rows = [
    makeSummaryRow({ model_tag: 'ffmpeg-video', kind: 'plugin' }),
    makeSummaryRow({ model_tag: 'model-a-27b', kind: 'model' }),
  ];
  const out = selectModelManifestRows(rows);
  assert.deepEqual(out.map((r) => r.model_tag), ['model-a-27b']);
});

test('selectModelManifestRows: drops the hidden===null "unreadable manifest" sentinel row', () => {
  const rows = [
    makeSummaryRow({ model_tag: 'corrupt-one', kind: null, hidden: null, error: 'unreadable' }),
    makeSummaryRow({ model_tag: 'model-a-27b', kind: 'model', hidden: false }),
  ];
  const out = selectModelManifestRows(rows);
  assert.deepEqual(out.map((r) => r.model_tag), ['model-a-27b']);
});

test('selectModelManifestRows: KEEPS hidden===true model rows -- hide/show must still reach L2 or nothing could un-hide them', () => {
  const rows = [makeSummaryRow({ model_tag: 'hidden-one', kind: 'model', hidden: true })];
  const out = selectModelManifestRows(rows);
  assert.deepEqual(out.map((r) => r.model_tag), ['hidden-one']);
});

// -----------------------------------------------------------------------
// summaryRowToDetail -- the server now adds
// display_name/gguf_blob_sha256 directly to the GET /api/manifests listing
// (the server already held the full manifest and was discarding both), so
// L1/L2 build from ONE listing call instead of an N+1 Promise.all over
// getManifest(tag) per row. getManifest(tag) is KEPT, but only for the L3
// accordion when a manifest is actually opened -- both fetch paths are
// intentionally kept.
// -----------------------------------------------------------------------

test('summaryRowToDetail: a null display_name passes through as null -- never coerced to an empty string or crashed on', () => {
  const detail = summaryRowToDetail(makeSummaryRow({ display_name: null }));
  assert.notEqual(detail, null, 'expected a detail object, not a skip');
  assert.equal(detail!.displayName, null);
});

test('summaryRowToDetail: a null gguf_blob_sha256 (the unreadable-row sentinel shape) is SKIPPED, not fabricated into a bogus tile', () => {
  const detail = summaryRowToDetail(makeSummaryRow({ gguf_blob_sha256: null }));
  assert.equal(detail, null, `expected null (skip), got: ${JSON.stringify(detail)}`);
});

test('summaryRowToDetail: a normal row maps display_name/gguf_blob_sha256/hidden/revision/etag straight through, no fetch involved', () => {
  const detail = summaryRowToDetail(
    makeSummaryRow({ model_tag: 'model-a-27b', display_name: 'Example Model 27B', gguf_blob_sha256: 'blob-x', hidden: true, revision: 4, etag: '"4"' }),
  );
  assert.deepEqual(detail, {
    model_tag: 'model-a-27b',
    displayName: 'Example Model 27B',
    blobSha256: 'blob-x',
    hidden: true,
    revision: 4,
    etag: '"4"',
    fileSizeBytes: null,
  });
});

test('summaryRowToDetail: gguf_size_bytes maps straight through when present (the file-size fallback for when GET /api/blobs is not live)', () => {
  const detail = summaryRowToDetail(makeSummaryRow({ gguf_size_bytes: 22_200_000_000 }));
  assert.equal(detail!.fileSizeBytes, 22_200_000_000);
});

// -----------------------------------------------------------------------
// groupIntoTiles -- pure and network-free.
// -----------------------------------------------------------------------

test('groupIntoTiles: two manifests sharing a blobSha256 group into ONE tile', () => {
  const tiles = groupIntoTiles(
    [
      makeDetail({ model_tag: 'model-a-27b', blobSha256: 'blob-a' }),
      makeDetail({ model_tag: 'model-a-27b-video', blobSha256: 'blob-a' }),
    ],
    null,
  );
  assert.equal(tiles.length, 1, `expected exactly one tile, got ${tiles.length}`);
  assert.equal(tiles[0].manifests.length, 2);
});

test('groupIntoTiles: manifests within a tile are sorted by model_tag, independent of input order', () => {
  const tiles = groupIntoTiles(
    [
      makeDetail({ model_tag: 'zulu', blobSha256: 'blob-a' }),
      makeDetail({ model_tag: 'alpha', blobSha256: 'blob-a' }),
    ],
    null,
  );
  assert.deepEqual(
    tiles[0].manifests.map((m) => m.model_tag),
    ['alpha', 'zulu'],
  );
});

test('groupIntoTiles: displayName is picked from whichever manifest in the group has one, even if it is not the first pushed', () => {
  const tiles = groupIntoTiles(
    [
      makeDetail({ model_tag: 'variant-a', blobSha256: 'blob-a', displayName: null }),
      makeDetail({ model_tag: 'variant-b', blobSha256: 'blob-a', displayName: 'Example-Model-35B' }),
    ],
    null,
  );
  assert.equal(tiles[0].displayName, 'Example-Model-35B');
});

test('groupIntoTiles: blobs===null (blob listing not deployed) -- tiles come ONLY from manifests, never manufactured from nothing', () => {
  const tiles = groupIntoTiles(
    [makeDetail({ model_tag: 'a', blobSha256: 'blob-a' })],
    null,
  );
  assert.deepEqual(tiles.map((t) => t.digest), ['blob-a']);
});

test('groupIntoTiles: a blob in the /api/blobs list with ZERO manifests still gets a tile', () => {
  const blobs: BlobEntry[] = [
    { digest: 'blob-a', sizeBytes: 100, description: null },
    { digest: 'blob-empty', sizeBytes: 200, description: null },
  ];
  const tiles = groupIntoTiles([makeDetail({ model_tag: 'a', blobSha256: 'blob-a' })], blobs);
  const empty = tiles.find((t) => t.digest === 'blob-empty');
  assert.ok(empty, 'expected a tile for the zero-manifest blob');
  assert.equal(empty!.manifests.length, 0);
  assert.equal(empty!.displayName, null);
});

test('groupIntoTiles: a blob entry for a digest that ALREADY has manifests merges sizeBytes -- does not create a second tile', () => {
  const blobs: BlobEntry[] = [{ digest: 'blob-a', sizeBytes: 555, description: null }];
  const tiles = groupIntoTiles([makeDetail({ model_tag: 'a', blobSha256: 'blob-a' })], blobs);
  assert.equal(tiles.length, 1, `expected one tile, not two, got ${tiles.length}`);
  assert.equal(tiles[0].sizeBytes, 555);
  assert.equal(tiles[0].manifests.length, 1);
});

// -----------------------------------------------------------------------
// ModelTileCard -- L1. Includes the empty-tile label
// behaviour.
// -----------------------------------------------------------------------

const emptyTile: TileGroup = { digest: 'abcdef0123456789fedcba', sizeBytes: null, displayName: null, description: null, manifests: [] };
const namedTile: TileGroup = {
  digest: 'abcdef0123456789fedcba',
  sizeBytes: null,
  displayName: 'Example-Model-35B-A3B (Mini)',
  description: null,
  manifests: [makeDetail(), makeDetail({ model_tag: 'beta' })],
};

test('ModelTileCard: zero-manifest tile shows "Unnamed model" -- never a fabricated name', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={emptyTile} onOpen={() => {}} />);
  assert.match(html, /Unnamed model/, `expected the disclosed placeholder, got: ${html}`);
});

test('ModelTileCard: zero-manifest tile is CTA-forward -- "click to add the first one", not a bare naming puzzle', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={emptyTile} onOpen={() => {}} />);
  assert.match(html, /No manifests yet.*add the first one/, `expected the CTA copy, got: ${html}`);
});

test('ModelTileCard: reuses the file\'s OWN existing truncated-digest micro-copy convention ("sha: {16 hex}…"), not new copy', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={emptyTile} onOpen={() => {}} />);
  assert.match(html, /sha: abcdef0123456789…/, `expected the existing 16-char truncation convention, got: ${html}`);
});

test('ModelTileCard: named tile shows display_name as the heading and its manifest count', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={namedTile} onOpen={() => {}} />);
  assert.match(html, /Example-Model-35B-A3B \(Mini\)/);
  assert.match(html, /2 manifests/);
});

test('ModelTileCard: carries NO Delete affordance (Delete is not offered on the tile face, for safety)', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={namedTile} onOpen={() => {}} />);
  assert.ok(!html.includes('>Delete<'), `expected no Delete control on the tile, got: ${html}`);
});

test('ModelTileCard: bg-slate-950, matching the dashboard (tiles read the same colour as the page behind them)', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={namedTile} onOpen={() => {}} />);
  assert.match(html, /class="[^"]*bg-slate-950[^"]*"/, `expected bg-slate-950, got: ${html}`);
  assert.ok(!html.includes('bg-slate-900'), `expected no bg-slate-900 anywhere on the tile, got: ${html}`);
  assert.ok(!html.includes('bg-slate-800'), `expected no lighter hover fill either (dropped for the same "lightest thing on the page" reason as ManifestRow), got: ${html}`);
});

// -----------------------------------------------------------------------
// ModelInfoCard -- L2. The model's own identity +
// own actions, in a card the user reads as "about the model" without a
// label saying so. Edit description and Delete model must NOT be
// adjacent or share an alignment column (the same fat-finger rule
// as the Back-button fix).
// -----------------------------------------------------------------------

const infoCardTile: TileGroup = {
  digest: 'abcdef0123456789fedcba',
  sizeBytes: 22_200_000_000,
  displayName: 'Example Model 27B',
  description: 'A dense 27B text+vision model.',
  manifests: [],
};

test('ModelInfoCard: renders name, description, size, and sha', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(html, /Example Model 27B/);
  assert.match(html, /A dense 27B text\+vision model\./);
  assert.match(html, /22\.20 GB/);
  assert.match(html, /abcdef0123456789…/);
});

test('ModelInfoCard: a null description shows the "No description yet." placeholder -- distinct from ModelTileCard\'s null-renders-nothing rule, since this IS the place editing happens, so an empty state needs to invite it', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={{ ...infoCardTile, description: null }} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(html, /No description yet\./);
});

test('ModelInfoCard: both "Edit description" and "Delete model" are present', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(html, />Edit description</);
  assert.match(html, />Delete model</);
});

test('ModelInfoCard: "Edit description" matches "Delete model"\'s SIZE (px-3 py-1 text-sm font-medium) but keeps its OWN color -- not Delete model\'s rose, since it is not destructive', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(
    html,
    /class="[^"]*px-3 py-1 rounded text-sm font-medium border border-slate-600 text-slate-300[^"]*">\s*Edit description/,
    `expected Edit description to match Delete model's size classes but keep the routine (non-rose) palette, got: ${html}`,
  );
  assert.ok(!/class="[^"]*rose[^"]*">\s*Edit description/.test(html), `expected Edit description to NOT take Delete model's rose color, got: ${html}`);
});

test('ModelInfoCard: Delete model sits in its OWN wrapper div carrying its own border-t divider, immediately preceding the button (not just "a divider somewhere earlier in the card")', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(
    html,
    /<div class="[^"]*border-t[^"]*">\s*<button[^>]*>\s*Delete model/,
    `expected Delete model's own immediate wrapper div to carry a border-t divider, got: ${html}`,
  );
});

test('ModelInfoCard: Edit description is NOT inside Delete model\'s own divider wrapper -- they are different DOM containers, not the same flex row', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  const deleteWrapperMatch = html.match(/<div class="[^"]*border-t[^"]*">\s*<button[^>]*>\s*Delete model[\s\S]*?<\/div>/);
  assert.ok(deleteWrapperMatch, `expected to find Delete model's wrapper, got: ${html}`);
  assert.ok(
    !deleteWrapperMatch![0].includes('Edit description'),
    `expected Edit description to be OUTSIDE Delete model's own wrapper div, got: ${deleteWrapperMatch![0]}`,
  );
});

// -----------------------------------------------------------------------
// ManifestRow -- L2. The Edit/Close and description buttons were removed,
// the surviving buttons were enlarged, and the row was
// split into a two-line info block (manifest's OWN displayName
// primary, model_tag secondary) plus a separate flex-wrap button row, once
// the one-line nowrap row overflowed on a phone.
// -----------------------------------------------------------------------

test('ManifestRow: NO "Edit ✎" / "▲ Close" button anywhere, collapsed or expanded (removed entirely; the row itself is the only toggle)', () => {
  const collapsed = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const expanded = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={true} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.ok(!collapsed.includes('Edit ✎'), `expected no Edit button, got: ${collapsed}`);
  assert.ok(!expanded.includes('▲ Close'), `expected no Close button, got: ${expanded}`);
});

test('ManifestRow: NO "✎ description" button either (removed same round, it duplicated Edit)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.ok(!html.includes('description'), `expected no description-edit control on the row, got: ${html}`);
});

test('ManifestRow: an expanded row is visually distinct from a collapsed one via the row\'s OWN border color, NOT a lighter background (bg-slate-950 stays constant in every state -- a lighter expanded fill would become the lightest thing on the page)', () => {
  const collapsed = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const expanded = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={true} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(expanded, /class="[^"]*bg-slate-950[^"]*border-emerald-600[^"]*"/, `expected the expanded row to keep bg-slate-950 and switch to the emerald border, got: ${expanded}`);
  assert.match(collapsed, /class="[^"]*bg-slate-950[^"]*border-slate-800[^"]*"/, `expected the collapsed row to keep bg-slate-950 with the slate border, got: ${collapsed}`);
  assert.ok(
    !/(?<!hover:)border-emerald-600/.test(collapsed),
    `expected a collapsed row's border-emerald-600 to appear ONLY as a hover: variant, not always-on, got: ${collapsed}`,
  );
});

test('ManifestRow: NO state (collapsed, expanded, or hover) uses a background lighter than bg-slate-950 -- fill never differentiates, only the border does', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={true} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.ok(!html.includes('bg-slate-900'), `expected no bg-slate-900 anywhere (that was the "lightest thing on the page" bug), got: ${html}`);
  assert.ok(!html.includes('bg-slate-800'), `expected no bg-slate-800 anywhere, got: ${html}`);
});

test('ManifestRow: Duplicate/Rename/Hide/Delete are all ONE uniform grown size (px-5 py-3 text-base font-semibold), Delete keeping its own rose color as the one meaningful difference', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  for (const label of ['Duplicate', 'Rename', 'Hide']) {
    const re = new RegExp(`class="[^"]*px-5 py-3 rounded-md text-base font-semibold border border-slate-600[^"]*">${label}<`);
    assert.match(html, re, `expected ${label} to use the uniform grown class, got: ${html}`);
  }
  assert.match(
    html,
    /class="[^"]*px-5 py-3 rounded-md text-base font-semibold border border-rose-900[^"]*">Delete</,
    `expected Delete to use the SAME padding\\/text-size as its siblings but keep its rose border, got: ${html}`,
  );
});

test('ManifestRow: the button row uses flex-wrap with no shrink-0 on any button (structural proof of "wrap, not overflow" -- the actual pixel non-overflow at 412px is a live browser measurement, not provable here)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const buttonRowMatch = html.match(/<div class="flex flex-wrap justify-end gap-2">(.*)<\/div>\s*<\/div>$/);
  assert.ok(buttonRowMatch, `expected a flex-wrap button row, got: ${html}`);
  assert.ok(!buttonRowMatch![1].includes('shrink-0'), `expected no shrink-0 on any button (that was the overflow cause) -- the status dot elsewhere in the row legitimately keeps its own shrink-0, got: ${buttonRowMatch![1]}`);
});

test('ManifestRow: the button row is right-aligned (justify-end), per the stated design preference', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /class="flex flex-wrap justify-end gap-2"/, `expected the button row to be right-aligned, got: ${html}`);
});

// -----------------------------------------------------------------------
// A restrained depth treatment -- soft shadow, a faint
// light ring, and a real press/active response -- applied identically to
// ModelTileCard and ManifestRow (both real click targets), and explicitly
// NOT to ModelInfoCard (not clickable). The class strings are assertable;
// whether it actually READS as "restrained," survives a dark background,
// or feels right on press is a manual visual check on a real render
// -- not claimed as tested here.
// -----------------------------------------------------------------------

test('ModelTileCard: carries the depth treatment (shadow-sm, a faint ring, and a real active/press state)', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={namedTile} onOpen={() => {}} />);
  assert.match(html, /shadow-sm/, `expected a soft shadow, got: ${html}`);
  assert.match(html, /ring-1 ring-white\/10/, `expected the faint highlight ring, got: ${html}`);
  assert.match(html, /active:scale-\[0\.98\] active:shadow-inner/, `expected a real press/active response (hover alone does nothing on a touchscreen), got: ${html}`);
});

test('ManifestRow: carries the SAME depth treatment as ModelTileCard (same on both, not a flatter row)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /shadow-sm/, `expected a soft shadow, got: ${html}`);
  assert.match(html, /ring-1 ring-white\/10/, `expected the faint highlight ring, got: ${html}`);
  assert.match(html, /active:scale-\[0\.98\] active:shadow-inner/, `expected a real press/active response, got: ${html}`);
});

test('ModelInfoCard: carries NONE of the depth treatment -- it is not a click target and was explicitly excluded', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.ok(!html.includes('shadow-sm'), `expected no shadow-sm on ModelInfoCard's root, got: ${html}`);
  assert.ok(!html.includes('active:scale'), `expected no active/press state on ModelInfoCard's root (it is not clickable), got: ${html}`);
});

test('ModelInfoCard: bg-slate-950, matching the dashboard (one of the four boxes a live measurement flagged as still slate-900)', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(html, /class="[^"]*bg-slate-950[^"]*"/, `expected bg-slate-950, got: ${html}`);
  assert.ok(!html.includes('bg-slate-900'), `expected no bg-slate-900 anywhere on the info card, got: ${html}`);
});

test('ManifestRow: hidden=true shows "Show"; hidden=false shows "Hide"', () => {
  const hiddenHtml = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ hidden: true })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const visibleHtml = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ hidden: false })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(hiddenHtml, />Show</);
  assert.match(visibleHtml, />Hide</);
});

test('ManifestRow: exactly the four remaining actions (Duplicate, Rename, Hide/Show, Delete) are present -- no Restore-defaults duplicate (it already lives in the L3 editor)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  for (const label of ['Duplicate', 'Rename', 'Hide', 'Delete']) {
    assert.ok(html.includes(`>${label}`), `expected a "${label}" control, got: ${html}`);
  }
  assert.ok(!html.includes('Restore defaults'), 'expected NO Restore-defaults button on the L2 row (lives in L3 only)');
});

// --- two-line content, per-manifest displayName ---

test('ManifestRow: when displayName is set, it is the PRIMARY line and model_tag renders underneath in mono (two lines, so a user can tell what the manifest is without clicking on it)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'model-c-35b-moe', displayName: 'Example Model 35B (MoE)' })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /<p class="text-sm text-slate-100 truncate">Example Model 35B \(MoE\)<\/p>/, `expected displayName as the primary line, got: ${html}`);
  assert.match(html, /<p class="font-mono text-xs text-slate-500 truncate"[^>]*>model-c-35b-moe<\/p>/, `expected model_tag as a secondary mono line, got: ${html}`);
});

test('ManifestRow: when displayName is null, model_tag becomes the primary line and there is NO second line (never an empty one)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'model-c-35b-moe', displayName: null })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /<p class="text-sm text-slate-100 truncate">model-c-35b-moe<\/p>/, `expected model_tag as the primary line when displayName is null, got: ${html}`);
  assert.ok(!html.includes('font-mono text-xs text-slate-500 truncate'), `expected NO secondary line when displayName is null, got: ${html}`);
});

test('ManifestRow: two manifests sharing a similar tag but DIFFERENT displayName are now distinguishable without opening either', () => {
  const moe = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'model-c-35b-moe', displayName: 'Example Model 35B (MoE)' })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const mtp = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'model-c-35b-mtp', displayName: 'Example Model 35B (MTP)' })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.notEqual(moe, mtp);
  assert.match(moe, /\(MoE\)/);
  assert.match(mtp, /\(MTP\)/);
});

// -----------------------------------------------------------------------
// InlineTextPrompt (Add manifest, Rename) and DeleteBlobPrompt
// -----------------------------------------------------------------------

test('InlineTextPrompt: pre-fills the input with `initial` and shows the confirm label', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="New tag" initial="model-a-27b" confirmLabel="Rename" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /value="model-a-27b"/, `expected the initial value pre-filled, got: ${html}`);
  assert.match(html, />Rename</);
});

test('InlineTextPrompt: renders the note line only when one is passed (the "callers will 404" warning)', () => {
  const withNote = renderToStaticMarkup(
    <InlineTextPrompt label="New tag" initial="x" confirmLabel="Rename" note="Callers using the old name will get a 404." busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  const withoutNote = renderToStaticMarkup(
    <InlineTextPrompt label="New manifest tag" initial="" confirmLabel="Create" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(withNote, /will get a 404/);
  assert.ok(!withoutNote.includes('404'), 'expected no warning text on the plain Add-manifest prompt');
});

test('DeleteBlobPrompt: informational "N manifests reference this" line, singular vs plural, matches the warn-not-block design', () => {
  const one = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={1} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  const three = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={3} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  assert.match(one, /1 manifest reference this/);
  assert.match(three, /3 manifests reference this/);
});

test('DeleteBlobPrompt: both options are present, not a single destructive default', () => {
  const html = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={2} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /Delete blob \+ 2 manifests/);
  assert.match(html, /Delete blob, keep manifests/);
});

// -----------------------------------------------------------------------
// The repo's own 44px minimum touch target
// (min-h-11 sm:min-h-0, matching FastLane.tsx's own pattern), applied to
// the seven controls that were under it. flex-wrap
// on the two prompt button rows. Pixel confirmation (does min-h-11 ACTUALLY
// render >=44px, does the row actually stop overflowing) is a manual
// browser check -- these tests prove the classes are present, not the
// rendered geometry.
// -----------------------------------------------------------------------

test('ModelInfoCard: Edit description and Delete model both carry the 44px touch-target floor (min-h-11 sm:min-h-0)', () => {
  const html = renderToStaticMarkup(
    <ModelInfoCard tile={infoCardTile} onEditDescriptionClick={() => {}} onDeleteModelClick={() => {}} />,
  );
  assert.match(html, /class="[^"]*min-h-11 sm:min-h-0[^"]*">\s*Edit description/, `expected Edit description to carry the touch floor, got: ${html}`);
  assert.match(html, /class="[^"]*min-h-11 sm:min-h-0[^"]*">\s*Delete model/, `expected Delete model to carry the touch floor, got: ${html}`);
});

test('InlineTextPrompt: Create/Cancel both carry the 44px touch-target floor (fixes both call sites, rename and add-manifest, in one place)', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="x" initial="" confirmLabel="Create" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /class="[^"]*min-h-11 sm:min-h-0[^"]*">Create</, `expected Create to carry the touch floor, got: ${html}`);
  assert.match(html, /class="[^"]*min-h-11 sm:min-h-0[^"]*">Cancel</, `expected Cancel to carry the touch floor, got: ${html}`);
});

test('InlineTextPrompt: the button row allows wrapping (flex-wrap)', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="x" initial="" confirmLabel="Create" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /class="flex flex-wrap items-center gap-2"/, `expected the button row to allow wrapping, got: ${html}`);
});

// -----------------------------------------------------------------------
// The shared component's box gets the same
// bg-slate-950 recolor. A `multiline` variant is added (an early-return
// branch, not a merged conditional) for the description editor, which
// needs a real textarea -- while rename and add-manifest, the two
// EXISTING callers, must stay single-line <input> BY CONSTRUCTION. These
// tests prove that directly: render with exactly the props those two
// callers use today (no `multiline`) and assert an <input> is present and
// NO <textarea> exists anywhere -- not just an assertion of intent.
// -----------------------------------------------------------------------

test('InlineTextPrompt: bg-slate-950, matching the dashboard', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="x" initial="" confirmLabel="Create" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /class="[^"]*bg-slate-950[^"]*"/, `expected bg-slate-950, got: ${html}`);
});

test('InlineTextPrompt: the EXISTING single-line usage (no `multiline` prop -- exactly what rename and add-manifest pass today) renders an <input> and NEVER a <textarea>, proving the tag call sites are structurally unaffected', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="New tag" initial="model-a-27b" confirmLabel="Rename" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /<input/, `expected an <input>, got: ${html}`);
  assert.ok(!html.includes('<textarea'), `expected NO <textarea> in the single-line (default) branch, got: ${html}`);
});

test('InlineTextPrompt: the single-line <input> itself carries the 44px touch floor (it measured 26px when the prompt was actually OPEN, not just resting)', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="x" initial="" confirmLabel="Create" busy={false} onConfirm={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /<input[^>]*class="[^"]*min-h-11 sm:min-h-0[^"]*"/, `expected the input itself to carry the touch floor, got: ${html}`);
});

test('InlineTextPrompt: `multiline` renders a <textarea rows="3" maxlength="2000">, full width, clearing the touch floor, and NO <input> at all', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="Model description" initial="" confirmLabel="Save" busy={false} onConfirm={() => {}} onCancel={() => {}} multiline />,
  );
  assert.match(html, /<textarea[^>]*rows="3"[^>]*maxLength="2000"/, `expected a 3-row textarea with the 2000-char server cap, got: ${html}`);
  assert.match(html, /<textarea[^>]*class="[^"]*w-full min-h-11[^"]*"/, `expected the textarea to be full width and clear the touch floor, got: ${html}`);
  assert.ok(!html.includes('<input'), `expected NO <input> in the multiline branch, got: ${html}`);
});

test('InlineTextPrompt: `multiline`\'s value pre-fills from `initial`, same as the single-line branch (the description editor must show the CURRENT description, not start blank)', () => {
  const html = renderToStaticMarkup(
    <InlineTextPrompt label="Model description" initial="A dense 27B text+vision model." confirmLabel="Save" busy={false} onConfirm={() => {}} onCancel={() => {}} multiline />,
  );
  assert.match(html, /<textarea[^>]*>A dense 27B text\+vision model\.<\/textarea>/, `expected the textarea to pre-fill with the initial description, got: ${html}`);
});

test('DeleteBlobPrompt: all three buttons carry the 44px touch-target floor', () => {
  const html = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={2} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  for (const label of ['Delete blob \\+ 2 manifests', 'Delete blob, keep manifests', 'Cancel']) {
    const re = new RegExp(`class="[^"]*min-h-11 sm:min-h-0[^"]*">${label}`);
    assert.match(html, re, `expected "${label}" to carry the touch floor, got: ${html}`);
  }
});

test('DeleteBlobPrompt: the button row allows wrapping (flex-wrap)', () => {
  const html = renderToStaticMarkup(
    <DeleteBlobPrompt manifestCount={2} busy={false} onDropManifests={() => {}} onKeepManifests={() => {}} onCancel={() => {}} />,
  );
  assert.match(html, /class="flex flex-wrap gap-2"/, `expected the button row to allow wrapping, got: ${html}`);
});

test('ManifestRow: a manifest whose displayName EQUALS its model_tag suppresses the secondary line (never renders the same text twice)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'draft-nmax15', displayName: 'draft-nmax15' })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const occurrences = (html.match(/draft-nmax15/g) ?? []).length;
  assert.equal(occurrences, 1, `expected "draft-nmax15" to appear exactly once (no duplicate line), got ${occurrences} in: ${html}`);
});

test('ManifestRow: a manifest with a GENUINELY different displayName still shows both lines, model_tag rendered as the secondary line\'s actual TEXT CONTENT, not merely present somewhere in the markup (e.g. a title attribute)', () => {
  // The secondary <p> keeps
  // `title={row.model_tag}` regardless of what its CHILD content renders,
  // so a loose "does the model_tag appear anywhere in the html" check
  // would still pass even if the visible text were wrongly swapped to
  // displayName -- the title attribute alone would satisfy it. Anchored
  // to the specific <p> and its rendered text instead, matching the
  // stricter pattern already used a few tests up.
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail({ model_tag: 'model-c-35b-moe', displayName: 'Example Model 35B (MoE)' })} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /<p class="text-sm text-slate-100 truncate">Example Model 35B \(MoE\)<\/p>/, `expected displayName as the primary line's text, got: ${html}`);
  assert.match(html, /<p class="font-mono text-xs text-slate-500 truncate"[^>]*>model-c-35b-moe<\/p>/, `expected model_tag as the secondary line's own rendered TEXT (not just its title attribute), got: ${html}`);
});

// -----------------------------------------------------------------------
// performRename -- two regressions that must be caught: inverting the operation order to DELETE-old THEN
// PUT-new, and stripping the required 404 warning from the success
// message. Extracted as performRename specifically so both safety
// properties are directly assertable, not just implied by a manual check.
// -----------------------------------------------------------------------

function makeRenameDeps(callLog: string[], opts: { deleteFails?: boolean } = {}) {
  return {
    getManifest: async (tag: string) => {
      callLog.push(`get:${tag}`);
      return { manifest: { model_tag: tag, gguf_blob_sha256: 'blob-a' }, etag: '"1"' };
    },
    putManifest: async (tag: string) => {
      callLog.push(`put:${tag}`);
      return { status: 'ok', model_tag: tag, revision: 1, restart_required: false };
    },
    deleteManifestApi: async (tag: string) => {
      callLog.push(`delete:${tag}`);
      if (opts.deleteFails) throw new Error('delete failed');
    },
  };
}

test('performRename: CALL ORDER -- putManifest(new tag) is observed BEFORE deleteManifestApi(old tag), never after', async () => {
  const callLog: string[] = [];
  const result = await performRename('old-tag', 'new-tag', makeRenameDeps(callLog));
  assert.ok(result.ok, `expected a successful rename, got: ${JSON.stringify(result)}`);
  const putIdx = callLog.indexOf('put:new-tag');
  const deleteIdx = callLog.indexOf('delete:old-tag');
  assert.notEqual(putIdx, -1, `expected putManifest to have been called, log: ${callLog.join(', ')}`);
  assert.notEqual(deleteIdx, -1, `expected deleteManifestApi to have been called, log: ${callLog.join(', ')}`);
  assert.ok(
    putIdx < deleteIdx,
    `expected put BEFORE delete (safe direction: a failed step leaves both tags, never neither), got order: ${callLog.join(', ')}`,
  );
});

test('performRename: a delete-then-put (inverted order) implementation is reproduced and shown RED against the order assertion above', async () => {
  // Reproduces that inverted order directly in the test, rather
  // than only in a throwaway source edit, so the order assertion's ability
  // to catch it is permanently on record (and re-runs on every `npm test`).
  async function invertedPerformRename(
    oldTag: string,
    newTag: string,
    deps: ReturnType<typeof makeRenameDeps>,
  ) {
    await deps.deleteManifestApi(oldTag); // inverted order: delete moved first
    const { manifest } = await deps.getManifest(oldTag);
    await deps.putManifest(newTag, { ...manifest, model_tag: newTag } as never, null as never);
    return { ok: true, message: `${oldTag} renamed to ${newTag}. Callers using the old name will get a 404.` };
  }
  const callLog: string[] = [];
  await invertedPerformRename('old-tag', 'new-tag', makeRenameDeps(callLog));
  const putIdx = callLog.indexOf('put:new-tag');
  const deleteIdx = callLog.indexOf('delete:old-tag');
  assert.ok(
    !(putIdx < deleteIdx),
    'expected the inverted mutation to fail the "put before delete" property (proving the test bites)',
  );
});

test('performRename: success message CONTAINS the required 404 warning (property, not a literal full-sentence match)', async () => {
  const callLog: string[] = [];
  const result = await performRename('old-tag', 'new-tag', makeRenameDeps(callLog));
  assert.ok(result.ok);
  assert.match(
    result.message,
    /404/,
    `expected the success message to warn callers of the old name about a 404, got: ${result.message}`,
  );
});

test('performRename: on a failed delete, ok=false and the message says both tags now exist (never silently reports success)', async () => {
  const callLog: string[] = [];
  const result = await performRename('old-tag', 'new-tag', makeRenameDeps(callLog, { deleteFails: true }));
  assert.equal(result.ok, false);
  assert.match(result.message, /both now exist/);
});

// =============================================================================
// Tile and row interaction: the ROW is the toggle (a separate "Edit (large)"
// row action never said so; clicking the tag label or the row container used
// to leave the page height unchanged, and only the Edit button expanded it).
// Also covered: tile information/shape restoration, Delete moved off the tile
// face entirely, a bigger/separated Back button, and dashboard-style
// clickability.
// =============================================================================

// --- Tile stats: the ctx/VRAM
// agree-vs-differ aggregation was replaced by a human description + file size
// -- it sidesteps the whole which-manifest-number problem.
// summarizeTileStat/fmtTileStat were removed entirely along with their
// tests, following the practice of deleting a
// feature once it's no longer needed rather than leaving it as dead code. ---

test('ModelTileCard: renders the tile description when set', () => {
  const tile: TileGroup = { digest: 'blob-a', sizeBytes: 22_200_000_000, displayName: 'Example Model 27B', description: 'A dense 27B text+vision model.', manifests: [] };
  const html = renderToStaticMarkup(<ModelTileCard tile={tile} onOpen={() => {}} />);
  assert.match(html, /A dense 27B text\+vision model\./, `expected the description text, got: ${html}`);
});

test('ModelTileCard: a null description renders NOTHING -- never a placeholder sentence like "no description" (an empty tile that says \'no description\' is noise)', () => {
  const html = renderToStaticMarkup(<ModelTileCard tile={emptyTile} onOpen={() => {}} />);
  assert.ok(!/no description/i.test(html), `expected no placeholder text at all, got: ${html}`);
});

test('ModelTileCard: still renders file size from the blob (unaffected by dropping ctx/vram)', () => {
  const tile: TileGroup = { digest: 'blob-a', sizeBytes: 22_200_000_000, displayName: 'Example Model 27B', description: null, manifests: [] };
  const html = renderToStaticMarkup(<ModelTileCard tile={tile} onOpen={() => {}} />);
  assert.match(html, /22\.20 GB/, `expected the file size, got: ${html}`);
});

test('ModelTileCard: NO edit-description affordance on the tile face -- description display is read-only, editing is L2-only (open question, answered: purely click-to-open)', () => {
  const tile: TileGroup = { digest: 'blob-a', sizeBytes: null, displayName: 'Example Model 27B', description: 'x', manifests: [] };
  const html = renderToStaticMarkup(<ModelTileCard tile={tile} onOpen={() => {}} />);
  const buttonCount = (html.match(/<button/g) ?? []).length;
  assert.equal(buttonCount, 0, `expected zero buttons on the tile even with a description present, got ${buttonCount} in: ${html}`);
});

// --- Delete moved off the tile (also see the earlier "carries NO
// Delete affordance" test above) ---

test('ModelTileCard: onOpen is the ONLY interactive prop -- no onDeleteClick or any other action callback remains on the tile', () => {
  // Compile-time proof: if a caller had to pass onDeleteClick again, this
  // file would fail to typecheck. Runtime companion: the tile's own click
  // target is exactly its outer role="button" container, not a nested
  // action element competing for the same click.
  const html = renderToStaticMarkup(<ModelTileCard tile={namedTile} onOpen={() => {}} />);
  const buttonCount = (html.match(/<button/g) ?? []).length;
  assert.equal(buttonCount, 0, `expected zero nested <button> elements on the tile (it IS the button), got ${buttonCount} in: ${html}`);
});

// --- the row is the toggle, styled to look clickable ---
// renderToStaticMarkup fires no DOM events and does not serialize React's
// onClick handlers into the HTML string, so click/stopPropagation behavior
// itself cannot be proven by this harness (no jsdom in this repo's test
// toolchain) -- stated plainly, same disclosed limit as every interactive
// claim here. What IS provable structurally: the row's root carries
// the clickable affordance (role, tabIndex, cursor-pointer) matching
// ModelTileCard's own convention, and every nested action is a real,
// separate <button> element (a precondition for stopPropagation to mean
// anything at all -- if these were plain <span>s there would be nothing to
// guard).

test('ManifestRow: the row root itself carries role="button", tabIndex, and a clickable cursor -- the toggle target, not just the Edit button', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /<div[^>]*role="button"[^>]*tabindex="0"[^>]*cursor-pointer/, `expected the row's OUTER container to be the clickable target, got: ${html}`);
});

test('ManifestRow: exactly 4 nested <button> elements now (Edit and description removed), each a real distinct button -- the structural precondition for stopPropagation to matter', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  const buttonCount = (html.match(/<button/g) ?? []).length;
  assert.equal(buttonCount, 4, `expected exactly 4 nested <button> elements (Duplicate/Rename/Hide/Delete), got ${buttonCount} in: ${html}`);
});

test('ManifestRow: styled to match ModelTileCard\'s own clickable-card convention (border + hover treatment), not a plain unstyled row', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.match(html, /class="[^"]*rounded-lg border[^"]*"/, `expected a Card-like rounded border treatment, got: ${html}`);
  assert.match(html, /border-slate-800/, `expected the collapsed border color, got: ${html}`);
  assert.match(html, /hover:border-emerald-600/, `expected the same hover-border convention ModelTileCard uses, got: ${html}`);
});

test('ManifestRow: the dead `mb-1.5 last:mb-0` pair is GONE -- it never worked (each row is the sole child of its own key-wrapper, so :last-child matched every row and zeroed the margin every time) and a dead class pair that looks load-bearing misleads the next reader. Spacing between rows is now the CONTAINER\'s job (flex flex-col gap-3, same token the tile grid uses)', () => {
  const html = renderToStaticMarkup(
    <ManifestRow row={makeDetail()} expanded={false} busy={false} onToggleEdit={() => {}} onToggleHide={() => {}} onDuplicate={() => {}} onRenameClick={() => {}} onDelete={() => {}} />,
  );
  assert.ok(!html.includes('mb-1.5'), `expected no mb-1.5 on the row itself, got: ${html}`);
  assert.ok(!html.includes('last:mb-0'), `expected no last:mb-0 on the row itself, got: ${html}`);
});

// -----------------------------------------------------------------------
// L1 tile grid breakpoint -- grid consistency decision (lg:/xl:,
// matching ResidentsPanel's own lg: gate so a phone in desktop-request mode
// stays one column). Reverting the grid breakpoint must be caught by
// the suite. The grid lives inline in Models()'s default export,
// which owns its own data-fetching state (unlike ResidentsPanel, a pure
// presentational component taking residents as props) -- Models() always
// renders its "Loading models…" branch under renderToStaticMarkup, since
// useEffect never fires during SSR, so the grid literally cannot be
// reached by rendering the component the way ResidentsPanel.test.tsx pins
// its own grid. This test pins the exact className as SOURCE TEXT instead
// -- reads Models.tsx directly and asserts the literal substring is
// present. It goes red if the string is reverted, the same as a rendered pin.
// -----------------------------------------------------------------------

function readModelsSource() {
  return readFileSync(join(process.cwd(), 'src/components/Models.tsx'), 'utf8');
}

test('Models.tsx L1 grid: the className is EXACTLY "grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3" -- source-text pin, not a rendered one (Models() cannot be rendered standalone in this harness; see comment above)', () => {
  const source = readModelsSource();
  assert.ok(
    source.includes('<div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">'),
    'expected the exact L1 grid className string to be present verbatim in Models.tsx',
  );
});

test('Models.tsx L1 grid: the OLD sm:/lg:-only breakpoint string is GONE -- proves this is an update, not an addition alongside the stale one', () => {
  const source = readModelsSource();
  assert.ok(
    !source.includes('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4'),
    'expected the old grid className to no longer be present anywhere in Models.tsx',
  );
});

test('Models.tsx Manifests section: bg-slate-950, source-text pin (same fill as the dashboard AND the same fill as the rows nested inside it, matching ThroughputSection\'s own border-only nesting)', () => {
  const source = readModelsSource();
  assert.ok(
    source.includes('<div className="rounded-lg border border-slate-700 bg-slate-950 p-3" data-testid="manifests-section">'),
    'expected the Manifests section wrapper to carry bg-slate-950 verbatim',
  );
});

// -----------------------------------------------------------------------
// Refresh moved off the always-rendered page
// header into level-scoped boxes (boxed at both L1 and L2,
// not left loose anywhere); the manifest-row list container gained a real
// flex+gap-3 (display:block never supported gap at all). All source-text
// pins for the same reason as the grid/section pins above -- this logic
// lives inline in Models()'s default export, which owns its own
// data-fetching state and can't be rendered standalone in this harness.
// -----------------------------------------------------------------------

test('Models.tsx: Refresh is NOT unconditionally rendered in the always-visible page header any more (it used to be the one loose, unboxed control on the page)', () => {
  const source = readModelsSource();
  const headerBlock = source.slice(source.indexOf('<h2 className="text-xl font-bold'), source.indexOf('{err &&'));
  assert.ok(!headerBlock.includes('↻ Refresh'), `expected no unconditional Refresh in the always-rendered header, got: ${headerBlock}`);
});

test('Models.tsx: L1 carries its own boxed Refresh control (bg-slate-950, bordered) -- present only when a tile is not selected', () => {
  const source = readModelsSource();
  assert.ok(
    source.includes('<div className="rounded-lg border border-slate-700 bg-slate-950 p-2 flex justify-end">'),
    'expected the L1 Refresh-only box to be present verbatim',
  );
});

test('Models.tsx: L2 carries ONE box holding both Back to models and Refresh together', () => {
  const source = readModelsSource();
  assert.ok(
    source.includes('<div className="rounded-lg border border-slate-700 bg-slate-950 p-2 flex items-center justify-between">'),
    'expected the L2 Back+Refresh box to be present verbatim',
  );
});

test('Models.tsx: the manifest-row list container is a real flex container using the SAME gap-3 token the tile grid uses -- display:block (what it was before) does not support gap at all', () => {
  const source = readModelsSource();
  assert.ok(
    source.includes('<div className="flex flex-col gap-3">'),
    'expected the manifest-row list wrapper to be a flex container with gap-3',
  );
});
