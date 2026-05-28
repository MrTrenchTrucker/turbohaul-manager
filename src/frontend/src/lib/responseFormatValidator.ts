/**
 * Client-side preflight validator for `response_format` payloads sent to
 * `POST /v1/chat/completions` and `POST /api/chat`.
 *
 * **Contract:** This is FAST-FEEDBACK ONLY. The server's `_validate_json_schema`
 * in `src/turbohaul/api/chat_completion.py` is the source of truth — if the
 * server rejects with HTTP 422 `schema_validation_failed`, the UI shows the
 * server's `detail.message` verbatim. This module exists so the user catches
 * obvious mistakes before the round-trip.
 *
 * The 10 server-side rejection reasons are:
 *   1.  jsonschema_lib_unavailable                  (SERVER-ONLY — N/A client)
 *   2.  missing_or_malformed_json_schema_field      (preflightable)
 *   3.  missing_or_malformed_schema_field           (preflightable)
 *   4.  schema_not_json_serializable                (preflightable)
 *   5.  schema_size_exceeded:<bytes>                (preflightable)
 *   6.  schema_depth_exceeded                       (preflightable)
 *   7.  schema_property_count_exceeded              (preflightable)
 *   8.  schema_contains_ref_unsupported             (preflightable)
 *   9.  schema_missing_additionalProperties_guard   (preflightable)
 *   10. schema_compile_failed:<exctype>             (SERVER-ONLY — Draft202012Validator)
 *
 * #1 and #10 are server-only — they require runtime checks the client can't
 * perform without re-implementing the BE jsonschema library. The friendly-text
 * map in `SchemaEditor.tsx` still handles those codes when they arrive on a
 * 422 response.
 *
 * Limits mirror the BE constants exactly:
 *   SCHEMA_SIZE_MAX_BYTES        = 65536
 *   SCHEMA_DEPTH_MAX             = 16
 *   SCHEMA_PROPERTY_COUNT_MAX    = 64
 *
 * If the BE limits drift, update both sides in lock-step and bump this module
 * via a follow-on RC.
 */

export const SCHEMA_SIZE_MAX_BYTES = 65536;
export const SCHEMA_DEPTH_MAX = 16;
export const SCHEMA_PROPERTY_COUNT_MAX = 64;

/** Discriminated union of the three accepted response_format shapes. */
export type ResponseFormat =
  | null
  | { type: 'text' }
  | { type: 'json_object' }
  | { type: 'json_schema'; json_schema: { name: string; schema: Record<string, unknown> } };

/** Editor mode — drives which inputs are visible. */
export type EditorMode = 'text' | 'json_object' | 'json_schema';

/** Reason codes mirror BE strings exactly (excluding size/compile suffix payloads). */
export type RejectionReason =
  | 'jsonschema_lib_unavailable'
  | 'missing_or_malformed_json_schema_field'
  | 'missing_or_malformed_schema_field'
  | 'schema_not_json_serializable'
  | 'schema_size_exceeded'
  | 'schema_depth_exceeded'
  | 'schema_property_count_exceeded'
  | 'schema_contains_ref_unsupported'
  | 'schema_missing_additionalProperties_guard'
  | 'schema_compile_failed';

export interface ValidationResult {
  ok: boolean;
  /** Stable code matching BE rejection-reason string (without `:<suffix>`). */
  reason?: RejectionReason;
  /** Human-readable preflight detail (size value, depth value, etc.). */
  detail?: string;
}

const OK: ValidationResult = { ok: true };

/**
 * Validate a full `response_format` envelope as the BE does.
 *
 * For `text` / `json_object` modes the envelope is shape-only and always ok.
 * For `json_schema` mode the inner `schema` is walked recursively against the
 * 8 client-preflightable rules (BE checks #2–#9).
 */
export function validateResponseFormat(rf: ResponseFormat): ValidationResult {
  if (rf === null) return OK;
  if (rf.type === 'text' || rf.type === 'json_object') return OK;
  if (rf.type !== 'json_schema') {
    // Unknown type — BE would emit 400 `response_format_unsupported_type`;
    // the editor's mode selector prevents this in practice, but guard anyway.
    return { ok: false, reason: 'missing_or_malformed_json_schema_field' };
  }

  // BE rule #2: outer `json_schema` must be a non-null object
  const js = rf.json_schema as unknown;
  if (!isPlainObject(js)) {
    return { ok: false, reason: 'missing_or_malformed_json_schema_field' };
  }

  // BE rule #3: inner `schema` must be a non-null object
  const inner = (js as Record<string, unknown>)['schema'];
  if (!isPlainObject(inner)) {
    return { ok: false, reason: 'missing_or_malformed_schema_field' };
  }
  const schema = inner as Record<string, unknown>;

  return validateInnerSchema(schema);
}

/**
 * Validate just the inner schema object (BE checks #4–#9).
 *
 * Exposed separately so the editor can run preflight on the parsed `schema`
 * object directly during keystroke validation, without rebuilding the full
 * envelope.
 */
export function validateInnerSchema(schema: Record<string, unknown>): ValidationResult {
  // BE rule #4: must be JSON-serializable.
  // In JS, JSON.stringify throws on cycles and silently drops functions/undefined;
  // we treat a throw as the rejection signal. We also reject any value whose
  // serialized form is `undefined` (top-level non-serializable).
  let serialized: string;
  try {
    serialized = JSON.stringify(schema);
  } catch {
    return { ok: false, reason: 'schema_not_json_serializable' };
  }
  if (typeof serialized !== 'string') {
    return { ok: false, reason: 'schema_not_json_serializable' };
  }

  // BE rule #5: serialized byte length cap.
  // Use TextEncoder for accurate UTF-8 byte count (parity with Python `.encode("utf-8")`).
  const byteLen = new TextEncoder().encode(serialized).length;
  if (byteLen > SCHEMA_SIZE_MAX_BYTES) {
    return {
      ok: false,
      reason: 'schema_size_exceeded',
      detail: `${byteLen} bytes (max ${SCHEMA_SIZE_MAX_BYTES})`,
    };
  }

  // BE rule #6: recursive nesting depth cap.
  const depth = measureDepth(schema);
  if (depth > SCHEMA_DEPTH_MAX) {
    return {
      ok: false,
      reason: 'schema_depth_exceeded',
      detail: `depth ${depth} (max ${SCHEMA_DEPTH_MAX})`,
    };
  }

  // BE rule #7: total property count cap.
  const propCount = countProperties(schema);
  if (propCount > SCHEMA_PROPERTY_COUNT_MAX) {
    return {
      ok: false,
      reason: 'schema_property_count_exceeded',
      detail: `${propCount} properties (max ${SCHEMA_PROPERTY_COUNT_MAX})`,
    };
  }

  // BE rule #8: $ref unsupported anywhere in the tree.
  if (containsRef(schema)) {
    return { ok: false, reason: 'schema_contains_ref_unsupported' };
  }

  // BE rule #9: every object-typed subschema must declare additionalProperties:false.
  const guardMissing = findMissingAdditionalPropertiesGuard(schema);
  if (guardMissing) {
    return { ok: false, reason: 'schema_missing_additionalProperties_guard' };
  }

  return OK;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Max depth of nested objects/arrays. Top-level object counts as depth 1. */
function measureDepth(node: unknown, current = 1): number {
  if (Array.isArray(node)) {
    let max = current;
    for (const item of node) {
      max = Math.max(max, measureDepth(item, current + 1));
    }
    return max;
  }
  if (isPlainObject(node)) {
    let max = current;
    for (const v of Object.values(node)) {
      max = Math.max(max, measureDepth(v, current + 1));
    }
    return max;
  }
  return current;
}

/**
 * Count every `properties: {...}` key across the whole tree. Mirrors the BE
 * intent ("total properties > 64"): each named property in any nested
 * `properties` block counts once.
 */
function countProperties(node: unknown): number {
  let total = 0;
  walk(node, (n) => {
    if (isPlainObject(n)) {
      const props = n['properties'];
      if (isPlainObject(props)) {
        total += Object.keys(props).length;
      }
    }
  });
  return total;
}

/** Does any subtree contain a `$ref` key? */
function containsRef(node: unknown): boolean {
  let found = false;
  walk(node, (n) => {
    if (found) return;
    if (isPlainObject(n) && Object.prototype.hasOwnProperty.call(n, '$ref')) {
      found = true;
    }
  });
  return found;
}

/**
 * Return true if any object-typed subschema lacks `additionalProperties:false`.
 * BE rule #9: any node with `type === "object"` MUST set
 * `additionalProperties: false` (or any other explicit falsy literal).
 */
function findMissingAdditionalPropertiesGuard(node: unknown): boolean {
  let missing = false;
  walk(node, (n) => {
    if (missing) return;
    if (isPlainObject(n) && n['type'] === 'object') {
      const ap = n['additionalProperties'];
      if (ap !== false) missing = true;
    }
  });
  return missing;
}

/** DFS walk that visits every plain object and array element. */
function walk(node: unknown, visit: (n: unknown) => void): void {
  visit(node);
  if (Array.isArray(node)) {
    for (const item of node) walk(item, visit);
  } else if (isPlainObject(node)) {
    for (const v of Object.values(node)) walk(v, visit);
  }
}
