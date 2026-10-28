# Module: slot_meta_projection

## Purpose
Keep the state database from holding, and the boot census from re-reading, the whole request body of every slot. Two
pure pieces. The identity projection says which client-metadata keys are persisted. The census projection says which
keys the boot rebuild needs and how to read them back from the stored body without the body ever reaching Python.

## Owns
- The list of client-metadata keys worth persisting for a slot (`IDENTITY_KEYS`) and the function that reduces a
  client-metadata dict to them (`identity_client_meta`).
- The reduction of a body that is already stored, given as JSON text, to the same keys (`strip_stored_meta`).
- The list of keys the boot census reads (`CENSUS_KEYS`), the SQL that reads exactly those from the stored body
  (`CENSUS_SLOTS_SQL`), and the parse of one projected value back into the small dict the census uses
  (`parse_census_meta`).

## Does Not Own
- Running the SQL, building or merging census rows, and every log line: manager.py, legacy.
- Writing slot rows (`upsert_slot`, the slot-end rewrite in `mark_slot_ended`): state.py, legacy. `upsert_slot` still stores
  whatever client metadata it is given.
- Choosing which rows to reduce, and when (the slot end, the retention sweep): state.py and manager.py, legacy.
- What the labels mean and how a class is chosen from them (`_class_from_label`): kv_classify.py.
- The in-memory client metadata of a running request. Only what is written to the database is reduced.

## Public Interface
- `IDENTITY_KEYS`: the persisted keys, in order.
- `CENSUS_KEYS`: the keys of the census projection, in the order of the projected array.
- `CENSUS_SLOTS_SQL`: the census SELECT over the slots table. It returns `slot_id, thread_id, model_tag, created_at` and
  one projected column.
- `identity_client_meta(client_meta) -> dict`: the identity keys that are present; `{}` for anything that is not a dict.
- `parse_census_meta(projected) -> dict`: the projected array as a dict of the non-null keys; `{}` for NULL, empty, not
  JSON, or an array of the wrong length.
- `strip_stored_meta(stored) -> str | None`: the identity-only JSON text for a stored body, byte-identical to what
  `json.dumps(identity_client_meta(body))` writes; None for NULL, empty, not JSON, or JSON that is not an object.

## Depends On
- Nothing inside TurboHaul. Standard library only. Values in, a value out.

## Invariants
- Every census key is an identity key. A key the census reads must be persisted, or the census would change silently.
- `CENSUS_SLOTS_SQL` never selects the stored body itself, only an expression over it, and returns NULL for a body that
  is not valid JSON instead of failing the statement.
- `identity_client_meta` returns only identity keys, never changes its input, and keeps a present key whose value is null.
- `parse_census_meta` and `strip_stored_meta` never raise.
- `strip_stored_meta` is idempotent: a body that is already identity-only comes back unchanged, so a caller can compare
  and skip the write.
- Pure: no I/O, no logging, no globals, no clock.

## Test Locations
- Unit: tests/unit/slot_meta_projection/
- Contract: tests/contract/test_slot_meta_projection_contract.py
- The manager's use of the module (legacy code): tests/test_boot_census_projection.py, tests/test_slot_identity_persistence.py

## Known Gotchas
- A body that is valid JSON but not an object, or that contains `NaN` or `Infinity`, now gives a row with no address and is
  skipped. Reading the body in Python used to clear the whole census for the first case and parse the second.
- The projection is evaluated by SQLite, which still reads every stored body to do it. The heap no longer holds the bodies;
  the scan itself remains until old bodies are purged or the address becomes its own column.
- Rows written before the identity projection still hold a full body. They keep working: the census reads the same keys
  from them.
