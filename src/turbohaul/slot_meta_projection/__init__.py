"""What the manager keeps of a slot's client metadata, and what the boot census reads back, as one module.

A slot's client metadata carries the whole request body. Two small, pure pieces keep the state
database from holding and re-reading all of it. The identity projection says which keys are
worth persisting. The census projection says which keys the boot rebuild needs and how to read
them from the database without ever loading the body. The caller runs the SQL, writes the rows
and builds the census. This module imports nothing from turbohaul and logs nothing.
"""

from __future__ import annotations

from .projection import (
    CENSUS_KEYS,
    CENSUS_SLOTS_SQL,
    IDENTITY_KEYS,
    identity_client_meta,
    parse_census_meta,
    strip_stored_meta,
)

__all__ = [
    "CENSUS_KEYS",
    "CENSUS_SLOTS_SQL",
    "IDENTITY_KEYS",
    "identity_client_meta",
    "parse_census_meta",
    "strip_stored_meta",
]
