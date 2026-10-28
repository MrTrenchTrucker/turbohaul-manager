# slot_meta_projection

Keeps the state database from holding, and the boot census from re-reading, the whole request body of every slot.

## What it does
Every request becomes a slot row, and the row used to store the full request body: all messages, the system prompt, the
tool list. That is almost all of the database. At boot the manager rebuilds the Fast Lane census from those rows, and it
needs only the client's address and a few labels from each one. This module holds the two rules that stop the waste.

## How it works
1. **Persist identity only.** `identity_client_meta` reduces a request's client metadata to the keys worth keeping: the
   address, the session, the role, the class labels and the per-role save flag. The manager passes the result to the
   database when it writes a slot.
2. **Read only what the census needs.** `CENSUS_SLOTS_SQL` asks SQLite to pull the address and the five class keys out
   of the stored body and hand back one small array per row. The body never reaches Python. A row whose body is not valid
   JSON gives NULL and is treated as a row with no address.
3. **Rebuild the dict.** `parse_census_meta` turns that array back into the small dict the census reads.
4. **Reduce what is already stored.** `strip_stored_meta` takes the stored text of an older, full body and returns the same
   identity-only text a new slot would have been written with, or nothing if the text is not a JSON object. The caller
   decides which rows to reduce and when.

## What it does not do
- It does not run the SQL, build the census or log. manager.py does.
- It does not choose rows or write them. Old rows keep their full bodies until the caller reduces them, and the census
  reads the same keys from either form.

## Who calls it
manager.py: the boot census rebuild, and the two places a slot row is written. The public names are listed in AGENTS.md.
