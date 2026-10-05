# Contributing

Thank you for wanting to help. Contributions are welcome, from people and from AI tools alike.
Issues and pull requests go to
[turbohaul-manager](https://github.com/MrTrenchTrucker/turbohaul-manager).

We ask two things of every change:

1. **Make it modular.** Put the code in a small, registered module with one clear job. This
   project is in early days, so for now we can only accept modularized pull requests. The
   standard is [module-sop](https://github.com/MrTrenchTrucker/module-sop): please read it and
   shape your change to it before you open a PR.
2. **Ship a test with it.** Every change comes with a test in the same PR, and the test must
   fail without your change.

**Why we ask.** We would like to reduce low-quality, machine-written changes ("AI slop"), and
we do not blame anyone for them: they are easy to produce and hard to review. Small, modular
changes help. Each one is a piece that fits in a single sitting, so even smaller, weaker models
and people with only a little time can work on one piece at a time, and a test shows that the
piece does what it says.

Everything below is the detail. The short version: open an issue, keep the change small, put
it in a module, add a test, run the checks.

## Steps

1. Open an issue first describing what you want to change and why.
2. Keep PRs scoped to one concern (fix, feature, or doc -- not all three).
3. Run `pytest` locally and make sure it passes before you open a PR. It runs the `tests`
   folder by default. Run `python3 tools/sop_check.py` too (see below).
4. Follow the existing comment style: no comments on the WHAT (the code says that),
   only on the WHY (intent, invariant, or non-obvious constraint).
5. New runtime dependencies must be MIT-compatible. No copyleft (GPL/AGPL/LGPL).
   Add the new entry to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) in the same PR.

## Modular changes

Modularized means the code lives inside a module registered in
[modules.toml](modules.toml) at the repository root.

1. A module is a folder at `src/turbohaul/<module>/`. Its registry entry in `modules.toml`
   declares its `path`, what it `owns`, what it `does_not_own`, what it `depends_on`, its
   `public` names and its `card`.
2. The card is a short `AGENTS.md` inside the module folder, at
   `src/turbohaul/<module>/AGENTS.md`. It has eight sections: Purpose, Owns, Does Not Own,
   Public Interface, Depends On, Invariants, Test Locations and Known Gotchas. The module also
   has a `README.md` that says in plain words what it does and how it works.
3. The card and the registry entry must say the same thing: the same public names and the same
   `depends_on`.
4. Keep chunks small. Where you can, one module per change, and one concern per module. Code
   files have a 300-line soft cap and a 500-line hard cap. A module can be exempt from the caps
   only through a `line_cap_exempt` entry, which needs a written decision (see
   [decisions/README.md](decisions/README.md)).
5. Respect the boundaries. A module may import another module only if its `depends_on` lists
   it, and only through that module's package entry point (its `__init__.py`), never one of its
   inner files. Legacy code is held to the entry-point rule too.
6. Code outside a registered module is legacy until it is moved into one. Moving a piece into a
   module is a good change of its own.

For examples, see the cards of the modules that exist today:
[engine_launch](src/turbohaul/engine_launch/AGENTS.md),
[engine_budget](src/turbohaul/engine_budget/AGENTS.md) and
[fastlane_client_names](src/turbohaul/fastlane_client_names/AGENTS.md).

## A test with every change

1. Every change ships with a test in the same PR: a fix, a feature, or a piece moved into a
   module.
2. The test must fail without your change. A test that passes both with and without the change
   proves nothing.
3. To check it, run the new test against the code as it was before your change. For example,
   set your source changes aside with `git stash push -- <your source files>`, keeping the
   test, and run the test. It should fail, and for the reason you expect (a typo in the test is
   not a reason). Then bring your change back and run it again. It should pass.
4. A module's tests live where its card's Test Locations section says: `tests/unit/<module>/`
   and one contract test, `tests/contract/test_<module>_contract.py`.

## What `tools/sop_check.py` checks

Run `python3 tools/sop_check.py` before you open a PR. It exits 1 on any failure. It checks:

- the module registry: every registered `path` exists, and `depends_on` names only known
  modules;
- the cards: each exists with all eight sections, and agrees with the registry;
- that each module has a `README.md`;
- the import boundaries from item 5 above, legacy code included;
- the 300-line soft and 500-line hard file caps;
- the `ARCHITECTURE.md` word budget, which can shrink but never grow;
- that the two generated files, `MODULE_MAP.md` and `.sop/function_index.json`, are fresh. If it
  reports one as stale or missing, run it once with `--write`, which regenerates them, and then
  run it again.

Decisions that shaped the module system are in [decisions/README.md](decisions/README.md).

Contributors are still recorded in [CONTRIBUTORS.md](CONTRIBUTORS.md), as before.
