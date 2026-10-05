# Module: fastlane_client_names

## Purpose
Decide which container name, if any, the Fast Lane Discovered list may show for a client address, so an operator can add a
rule by name from that list (decisions/ADR-003). A rule by name keeps matching when the container gets a new address on
restart; a rule by address does not.

## Owns
- The name candidates taken from one reverse lookup answer: the whole name first, then the same name with its trailing
  dot-separated labels removed one at a time (`a.b.c` gives `a.b.c`, `a.b`, `a`).
- Which candidates are confirmed: a candidate counts only when a forward lookup of it returned the client's own address.
- Which confirmed name is shown: the shortest, with ties broken alphabetically. No confirmed candidate means no name.

## Does Not Own
- Doing any lookup, the cache, the retry backoff and the background task that runs the lookups: legacy code beside the
  forward name resolver (fastlane_resolve.py), started by the API at boot (api/main.py).
- Normalizing an address string: fastlane.py (`normalize_address`). Callers pass normalized addresses in.
- The Discovered list itself (the census) and its `container_name` field: manager.py, served by api/fastlane_census.py.
- Matching a request to a rule. The shown name is display only; it never reaches the matcher.
- What the frontend does with the name: src/frontend/src/components/FastLane.tsx and src/frontend/src/api.ts.

## Public Interface
- `name_candidates(reverse_answer: str) -> tuple[str, ...]`: the candidates for one reverse lookup answer, whole name first.
  An empty or malformed answer (whitespace, an empty label, or an IP address literal) gives an empty tuple; one trailing dot
  is dropped first.
- `confirmed_name(address, forward: Mapping[str, Iterable]) -> str | None`: the name to show for `address`, given each
  candidate's forward-lookup addresses (normalized). `None` when no candidate's addresses contain `address`.

## Depends On
- Nothing inside TurboHaul. Plain strings and address values in, a name or `None` out.

## Invariants
- A name is never shown unless a forward lookup of exactly that name returned the client's own address.
- Among confirmed names the shortest wins, so the name shown is the one an operator would type, without a network suffix
  when a shorter form also confirms.
- The same inputs always give the same answer (ties are broken alphabetically, never by input order).
- No candidate is longer than the reverse answer, and the number of candidates is bounded by its label count.
- Pure functions: no I/O, no lookups, no logging, no clock, no reading of live state.

## Test Locations
- Unit: tests/unit/fastlane_client_names/
- Contract: tests/contract/test_fastlane_client_names_contract.py

## Known Gotchas
- A reverse answer on a container network usually carries the network as a suffix (`name.network`). The bare name is the
  one a rule should use, and it confirms only when the bare name resolves forward from the service's own network.
- An address a container has just released can be handed to another container. That is why a shown name must be confirmed
  again from time to time by the caller; this module only judges one set of lookup results.
- When more than one candidate confirms, the shortest-then-alphabetical rule still picks one name, and the forward lookup
  guarantees it points at this client's address.
