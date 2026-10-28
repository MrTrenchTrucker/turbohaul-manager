# resident_exit

Two small decisions that the manager needs when a model's resident engine ends its work.

## What it decides
- **Keep warm or let go.** When a request for a model finishes and its keep-alive would tear the engine down at once, but
  another engine of the same model is still serving requests on the box, this module can say "keep it warm for a short,
  fixed time instead." The hold is bounded, so a waiting request for more memory can still take that engine back. It holds
  only when nothing outranks the engine, no other model is waiting for its memory, and the box has room for two engines.
  A box that allows one engine never holds, so its behaviour does not change.
- **What the exit line says.** Every time a resident engine dies, one `RESIDENT_EXIT` line is written, with the reason, the
  resident key, the model, the thread, the port, and the state before and after. The line has the same fields in the same
  order every time, so a death can be traced to its cause from one line.

## What it does not decide
- It does not read the live state. The manager measures everything under its lock and passes plain values in.
- It does not stop, park or remove an engine, and it does not log. The manager does all three.
- It does not decide whether the cache of a finished request is saved. That depends on the kind of request, and it has
  nothing to do with whether the engine stays alive.
- The make-room path already writes its own eviction line, so this module never writes a second one for it.

## How it is used
The manager calls `decide_retention` with its measurements when a finishing engine would be torn down at once, and it uses
the window it is given if the answer is a hold. It calls `format_resident_exit` and writes the result at info level
whenever a resident dies for a cause other than make-room, which `needs_exit_line` tells it. The module is pure, so it is
tested without a running manager.
