# ADR-003: The Fast Lane Discovered list shows a confirmed container name, and adding from it saves a rule by name (module fastlane_client_names)

**Status:** accepted
**Date:** 2026-10-01

## Context
The Discovered list on the Fast Lane page shows the address each request came from. "Add to Fast Lane" on a row saved a
rule by that address. For a client in a container that address is leased by the container runtime and can change on
restart, so the manager answered such a save with a warning to prefer a container name, which the list could not offer.
The rule then stopped matching after the next restart. Rules by container name already existed (a typed box), but the
operator had to know the name.

## Decision
- A background task looks up the container name behind each Discovered address (reverse lookup), off the request path,
  with a short timeout, a cap on rows per tick and a retry backoff. It fails closed: no confirmed name, no name shown.
- A name is shown only after a forward lookup of it returns that row's own address. Among confirmed names the shortest is
  shown. This rule is the module `fastlane_client_names` (src/turbohaul/fastlane_client_names): pure, with the lookup
  results passed in.
- The Discovered row carries the name (`container_name`, null until confirmed). "Add to Fast Lane" saves a rule by name
  when the row has one, and by address only when it has none.
- The name is display only. Matching a request to a rule is unchanged: a rule by name matches through the existing forward
  resolver.

## Reasons
- It removes the warning at its cause and gives the operator the rule that keeps working across restarts, from the place
  they actually add clients.
- Confirming forward keeps a stale or wrong reverse answer from naming the wrong client.
- Reverse lookup stays out of matching: the forward resolver alone decides matches.

## Consequences
- One more background task and its lookups. A request never waits on them.
- A row can show an address first and a name a few seconds later, once the lookup has confirmed it.
- Clients that are not containers, or whose name does not resolve from the service's network, keep showing an address, and
  adding them still saves a rule by address.
