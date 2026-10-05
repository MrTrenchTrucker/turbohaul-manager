# fastlane_client_names

Lets the Fast Lane Discovered list show a client's container name, so a rule can be added by name.

## Why
The Discovered list sees only the address a request came from. A Fast Lane rule added by address stops matching when the
container restarts and gets a new address, and the manager warns about exactly that ("prefer container_name"). With a
confirmed name on the row, "Add to Fast Lane" saves a rule by name instead.

## What this module decides
Given the answer of one reverse lookup (address to name) and the results of forward lookups (name to addresses), it decides
which name to show:
- candidates are the whole answer and the answer with its trailing labels removed one at a time;
- a candidate counts only if its forward lookup returned the client's own address;
- the shortest confirmed candidate is shown (ties alphabetical); with none, the row keeps showing only the address.

## What it does not do
It performs no lookups and keeps no cache. The lookups run in a background task beside the forward name resolver, never on
the request path, and the name is used for display and for building the rule only, never for matching a request.
