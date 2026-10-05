# Vendored: llama-cpp-turboquant (the turboquant / heavily-modified llama.cpp engine)

- **Source repo:** https://github.com/TheTom/llama-cpp-turboquant
- **Base SHA:** 86771a58d -- the commit this tree is BASED ON, not a SHA the
  shipped directory byte-matches. `engine.lock` records the same value as
  `turboquant_engine_sha = 86771a58d` with `turboquant_engine_sha_role = base` and
  `turboquant_engine_matches_sha = false`.
- **What this is:** a SOURCE SNAPSHOT with no git history. It was taken from the fork
  tree at the base SHA and then carried further; it is not a plain `git archive` of
  that one commit.
- **Reachability:** the base SHA and the fork commits referred to below are not
  resolvable from the source-repo URL above; only the upstream-project commits are.
  A reader with access to the fork can reproduce this comparison.

## What the shipped directory actually contains

Measured by comparing the shipped tree against the base commit and against every
commit of both this fork and the upstream project, blob by blob. Of the 73 paths
that differ from the base commit (71 modified, 2 added), every one falls into one of
three groups:

- **Fork integration line -- 30 paths.** The shipped content is byte-identical to the
  tip of the fork's integration line. That line does not descend from the base SHA --
  the two diverged at an earlier common commit -- so this content was not applied on
  top of the base.
- **Earlier upstream work -- 5 paths.** The shipped content matches older
  upstream-project commits only. It arrives with the upstream work this engine tracks,
  not with the base commit, whose own copies of those files differ.
- **An unattributed working-set layer -- 37 paths.** These match no commit at file
  level in the fork or in the upstream project, anywhere, on any branch: on the order
  of 60 individual hunks of shipped source with no commit behind them. The layer
  includes the draft-model architecture implementation (`src/models/dflash.cpp`,
  absent from every fork ref), a new public capped/prefix sequence-save API threaded
  through the whole memory hierarchy, a structured drafter-downgrade signal, and
  parent-aware draft memory measurement. This layer is described in
  `docs/ENGINE_CHANGELOG.md`; its individual changes are not separately attributable
  to commits.

The other 2951 paths are byte-identical to the base commit and are the "base" content
in the ordinary sense.

Because of the unattributed layer, this directory cannot be reproduced from any single
commit, and a future re-sync must diff against a preserved copy rather than against a
SHA.

- **Re-sync to a newer tree:** `git -C <llama-cpp-turboquant clone> archive <newSHA> | tar -x -C engine/llama-cpp-turboquant`,
  then reconcile the unattributed layer by hand. Note what extraction actually does over
  an existing directory: it overwrites the 36 unattributed paths the archive contains,
  reverting them to the archived content, and leaves `src/models/dflash.cpp` in place
  because no archive contains that path. Nothing is deleted, so the tree looks freshly
  extracted while 36 files have silently lost unattributed work and one stale file is
  left over from the previous tree.
