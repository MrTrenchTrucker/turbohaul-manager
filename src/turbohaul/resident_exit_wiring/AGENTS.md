# Module: resident_exit_wiring

## Purpose
The manager-facing side of the resident exit feature: measure the live inputs for the keep-warm decision and ask
resident_exit, write the one exit log line for a resident death, and pick the exit reason when the caller did not stamp one.

## Owns
- Measuring the keep-warm inputs from the live manager view (same-model active siblings, the designated victim, other-model
  capacity demand, the parallel sidecar limit), calling `decide_retention`, and writing the hold-warm log line when it says hold.
- Writing the exit line: which fields are read from the resident, the thread and port fallbacks, the info-level write, and
  the best-effort rule that a failure is logged at debug and never raised.
- Resolving the reason of a death when the caller left it unstamped.

## Does Not Own
- The keep-warm decision itself, the exit line format and the `ExitReason` names: resident_exit.
- Evicting, deregistering or parking a resident, the registry lock, and the order of the calls in a death: manager.py.
- The stamps (`<resident>.exit_reason = ...`) and the call sites of the exit line: manager.py.
- The make-room eviction line and every other log line: manager.py.

## Public Interface
- `hold_warm_window(host, r, idle_window, hold_window_s, *, active_state) -> float | None`: the window to park a finishing
  resident with, or None to evict it now. `host` is read through a Protocol view; `hold_window_s` is the hold the caller
  offers; `active_state` is the caller's ACTIVE state value (the state enum lives in manager.py, which this module cannot
  import).
- `log_resident_exit(r, reason, state_before, port=None, *, format_line, state_after) -> None`: write the exit line for one
  death. The caller passes the line formatter and the DEAD state in.
- `resolve_exit_reason(r, idle) -> ExitReason`: the stamped reason, else SELF_IDLE_TIMEOUT when `idle` is true, else
  UNSPECIFIED.

## Depends On
- resident_exit, through its package entry point only.

## Invariants
- It may log, on the logger named `turbohaul.manager`, so existing handlers and filters keep matching. It may read the live
  manager through a Protocol view. It holds no state of its own and starts no tasks.
- The hold-warm check keeps the order of the original reads: the active siblings, then the designated victim, then the
  fast-lane claims and the pending reclaim, then the decision. The claims and the pending reclaim are read with a default,
  because a manager built without its constructor does not have them.
- The exit line's formatter is called with keyword arguments only. A failing formatter never escapes: it leaves one debug
  record with the traceback, and no exit line.
- The thread of the line comes from the grace tip, then the active slot, then the idle thread. The port is the one passed in,
  else the resident handle's port.

## Test Locations
- Unit: tests/unit/resident_exit_wiring/
- Contract: tests/contract/test_resident_exit_wiring_contract.py
- Manager level (the legacy wiring): the existing resident exit wiring and guard tests under tests/ (the two files whose
  names end in resident_exit_wiring.py and resident_exit_guards.py), which drive the real manager through its wrapper methods

## Known Gotchas
- manager.py keeps thin methods with the old names (`_hold_warm_window_locked`, `_log_resident_exit`) that call this module.
  They read the hold window and the formatter from the manager's own module globals at call time, because tests patch those
  globals.
- `hold_warm_window` is called under the registry lock and must stay sync. A new `await` in it would break the lock rule.
- The exit reason of a death is stamped by manager.py just before it unloads; this module only supplies the fallback.
