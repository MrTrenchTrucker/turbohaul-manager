# resident_exit_wiring

The part of the resident exit feature that touches the live manager. The decision and the line format live in
resident_exit and stay pure; this module does the measuring and the logging around them.

## What it does
- **Measures for the keep-warm decision.** When a finishing engine would be torn down at once, it counts the other active
  engines of the same model, checks whether this engine is the designated victim, checks whether another model or a pending
  reclaim is waiting for the memory, reads the parallel sidecar limit, and asks resident_exit. If the answer is "hold" it writes one
  hold-warm line and returns the window to park with. Otherwise it returns nothing, and the engine is let go.
- **Writes the exit line.** When a resident dies for a cause that needs a line, it gathers the fields (key, model, thread,
  port, state before and after), asks resident_exit's formatter for the text, and writes it at info level. If anything in
  that fails it writes a debug record instead and carries on, so a logging problem can never break a death.
- **Picks the reason when none was given.** A resident the caller did not label is reported as an idle timeout when it was
  idle, and as unspecified otherwise.

## What it does not do
- It does not decide anything about keeping an engine warm, and it does not shape the line. That is resident_exit.
- It does not stop, park or remove an engine, take the registry lock, or set the reason labels. The manager does.
- It keeps no state and starts no background work.

## How it is used
The manager calls these three functions from short methods of its own. It passes in what lives in the manager (the hold
window, the line formatter, the state values), so this module never imports the manager. The log lines are written on the
manager's logger name, so nothing that reads the logs notices where the code moved to.
