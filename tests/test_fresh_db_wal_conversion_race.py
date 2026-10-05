"""The FIRST open of a fresh state.sqlite must be serialised.

`PRAGMA journal_mode=WAL` takes an EXCLUSIVE lock to convert the journal. That
conversion happens exactly once per database; on an already-WAL database the
same pragma is a no-op read that takes no lock. Production reaches this through
`init_audit_pool` -> `open_state_db`, so on a genuinely fresh DB every thread's
first pool init races that one-time conversion and losers raise
OperationalError "database is locked".

⛔ NON-VACUITY — this test was watched RED on unmodified code before the fix:
   an unfixed tree raises OperationalError at the journal_mode pragma in a
   sizeable fraction of trials at N=4, and a fixed tree raises none, at both
   4 and 32 threads.
   At TRIALS=60 and a per-trial failure rate of even 20%, an unfixed tree fails
   this test with probability 1 - 0.8**60 ~= 1 - 2e-6. On a fixed tree it cannot fail: the
   conversion is serialised, so there is no loser to raise.

◊ SCOPED TO OperationalError ON PURPOSE. At higher thread counts the unfixed
  code also produces IntegrityError "UNIQUE constraint failed:
  schema_version.version" from the check-then-insert in state.py. That
  is a SEPARATE defect, out of scope here; serialising the first
  open happens to hide it in-process but does not repair it, so this test must
  not quietly assert its absence. Other exception types are TALLIED and shown
  in the failure message for context, never asserted on.

◊ Trial directories are deliberately NOT deleted during the run. Freeing them
  would let the filesystem reuse an inode, and `_INITIALISED` is keyed on
  (st_dev, st_ino) — a reused inode would look like an already-converted
  database, skip the lock, and make THIS test flaky. pytest's tmp_path cleanup
  handles them afterwards.
"""
import sqlite3
import threading
from collections import Counter

from turbohaul.state import open_state_db


THREADS = 4
TRIALS = 60


def _one_trial(db_path, n_threads):
    """Release n_threads simultaneously against one fresh DB.

    Returns (failures, completed) where `failures` is a list of
    (exception_type_name, message) and `completed` counts threads that reached
    a verdict at all — a barrier that breaks means the threads never actually
    overlapped and the trial proved nothing.
    """
    barrier = threading.Barrier(n_threads)
    failures = []
    completed = []
    guard = threading.Lock()

    def worker():
        try:
            barrier.wait(timeout=30)
        except threading.BrokenBarrierError:
            return
        try:
            open_state_db(db_path).close()
            with guard:
                completed.append(True)
        except Exception as exc:  # noqa: BLE001 - the tally is the point
            with guard:
                completed.append(True)
                failures.append((type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return failures, len(completed)


def test_concurrent_first_open_of_a_fresh_db_does_not_raise_database_is_locked(
    tmp_path,
):
    """N threads opening ONE fresh state.sqlite together must all succeed.

    Fails on unmodified code because the losers of the one-time WAL conversion
    race raise OperationalError "database is locked".
    """
    locked = []
    other = Counter()
    thin_trials = []

    for i in range(TRIALS):
        db_dir = tmp_path / f"trial{i}"
        db_dir.mkdir()
        failures, completed = _one_trial(db_dir / "state.sqlite", THREADS)
        if completed != THREADS:
            thin_trials.append((i, completed))
        for kind, msg in failures:
            if kind == sqlite3.OperationalError.__name__ and "locked" in msg:
                locked.append(f"trial {i}: {kind}: {msg}")
            else:
                other[f"{kind}: {msg}"] += 1

    # The fixture must have BUILT the shape before its verdict means anything:
    # every trial has to have released all THREADS together. A broken barrier
    # would make every trial a sequence of single opens, which cannot contend
    # and would pass on broken code.
    assert not thin_trials, (
        "VACUOUS RUN — the barrier broke, so these trials never ran their "
        f"threads concurrently and could not have contended: {thin_trials}. "
        "This test proves nothing until that is fixed."
    )

    assert not locked, (
        f"The one-time WAL conversion on a fresh state.sqlite is not "
        f"serialised: {len(locked)} of {TRIALS * THREADS} threads across "
        f"{TRIALS} trials lost the race and raised "
        f'OperationalError "database is locked".\n'
        + "\n".join(locked[:10])
        + (f"\n... and {len(locked) - 10} more" if len(locked) > 10 else "")
        + (
            "\nOther exception types seen (NOT asserted on — a separate defect owns "
            f"the schema_version TOCTOU): {dict(other)}"
            if other
            else ""
        )
    )


def test_a_deleted_and_recreated_db_is_serialised_again(tmp_path):
    """A DB that is deleted and recreated is FRESH again, and must re-serialise.

    `_INITIALISED` remembers which databases have already been converted, so it
    decides whether the lock is taken. Remembering the PATH SPELLING would be
    wrong: delete state.sqlite and recreate it and the path is unchanged while
    the database is brand new — the entry would still match, the lock would be
    skipped, and the conversion race would be back on a genuinely fresh DB.
    Keying on (st_dev, st_ino) is what makes that case a miss.

    ⛔ This is the test that distinguishes the two keys. Without it, replacing
    the identity key with `str(state_db_path)` would pass the rest of the suite,
    so this test is what catches that regression.

    A delete-then-recreate at the same path normally allocates a new inode,
    so the miss is reliable rather than lucky.
    Any trial where it IS reused is counted and reported, because that — not a
    regression — would be the explanation for a failure here.
    """
    locked = []
    inode_reused = []

    for i in range(TRIALS):
        db_dir = tmp_path / f"recreate{i}"
        db_dir.mkdir()
        db_path = db_dir / "state.sqlite"

        # Convert it once, so this database is registered as initialised.
        open_state_db(db_path).close()
        before = db_path.stat().st_ino
        for suffix in ("", "-wal", "-shm"):
            stale = db_path.with_name(db_path.name + suffix)
            if stale.exists():
                stale.unlink()

        # Same path, new database. Every thread below races a fresh conversion.
        failures, _ = _one_trial(db_path, THREADS)
        if db_path.stat().st_ino == before:
            inode_reused.append(i)
        for kind, msg in failures:
            if kind == sqlite3.OperationalError.__name__ and "locked" in msg:
                locked.append(f"trial {i}: {kind}: {msg}")

    assert not locked, (
        f"A deleted-and-recreated state.sqlite was treated as already "
        f"converted, so its fresh WAL conversion ran unserialised: "
        f"{len(locked)} threads across {TRIALS} trials raised "
        f'OperationalError "database is locked".\n'
        + "\n".join(locked[:10])
        + f"\ninode was reused (making the miss impossible) in trials: "
        f"{inode_reused or 'none — so this is a real regression, not FS luck'}"
    )
