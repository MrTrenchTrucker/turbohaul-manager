## Retrospective: Wave 4b.7 — Round 8 Streaming ACTIVE_MATCH Fix
**Date:** 2026-05-17
**DAG Branches Explored:** 3 hypotheses (H1 event/state divergence, H2 route-level-deadline-vs-ACTIVE_MATCH, H3 Hermes streaming flag drift)
**Branch Failures:** 0 (single iteration to success)
**Final Branch:** H1 confirmed via direct code read + RBSRS/SCOUT triangulation; H2 + H3 dissolved into H1's actual mechanism
**Forgejo commit:** `de0602e`
**GitHub:** UNPUSHED (HARD GATE per Cmdr standing rule)
**RBSRS gates:** Failure Predictor #16 (red-hat) 8/10 post-MOD, Simplicity Advocate #15 (human-thinker) 7.5/10
**Smoke verdict:** RELAY HANDS 5/5 CLEAN PASS @ 1m 45s wall on 3-tool prompt

---

### Delta Analysis

- **Initial prompt** (Cmdr 22:13Z): "I cannot get Hermes to do multiple tool calls without the model going into grace timer and then hot loaded." Multi-tool-call agent loops N>2 failed every turn ≥ 2 with `Slot did not reach ACTIVE within 600.0s` despite Wave 4b.5+4b.6+manifest reasoning_budget=500 having fixed the 2-tool case.
- **Final solution**: ~48 LoC in `manager.py` `_process_slot` ACTIVE_MATCH branch. Mirror the anchor's streaming-branch logic (`stream_handle = handle; stream_ready_event.set(); await stream_done_event`) inside the warm-slot reuse promotion path. Predicate on `client_meta["stream"]` (data semantics), assert events as invariants (eliminates TOCTOU race that the Failure Predictor caught). CancelledError handler signals `stream_done_event` before terminal-park.
- **Path efficiency:** DIRECT — single fix-iteration to success. But only because Wave 4b.5+4b.6 had previously narrowed the failure surface (eliminated all idle_hot / keep_alive / KV-persistence hypotheses), letting Round 8 start from a discriminating empirical signature ("audit shows slot ACTIVE in 1s + LLM completes in 10-15s, but Hermes sees 600s timeout").

---

### Branch Hypothesis Log (none were "failures" — all three converged to H1)

| Hypothesis | Status | What Eliminated/Confirmed It |
|---|---|---|
| **H1**: ACTIVE_MATCH path missing `stream_ready_event.set()` → matched slot's event never set → route hangs 600s | **CONFIRMED** | Direct code read: grep showed only ONE `.set()` site (manager.py:776, anchor only) and ONE `.wait()` site (chat_completion.py:376, route). ACTIVE_MATCH branch had ZERO `is_streaming` check. Audit DB confirmed slots transitioning ACTIVE in 1s while route hung. |
| **H2**: Route-level 600s deadline ignores ACTIVE_MATCH semantics | Dissolved into H1 | The deadline timer IS the symptom mechanism, but H1 is the cause (event is never set, so deadline always fires). H2 would have been a separate bug if the event WAS set but the deadline still misfired — not the case here. |
| **H3**: Hermes `streaming: false` config drift — Hermes actually sending stream=true | Partially true but orthogonal | Hermes' OpenAI Python SDK defaults to stream=true on chat completions for SSE thinking-content support. Hermes config's `providers.<id>.streaming: false` appears to be cosmetic/non-enforcing on this code path. BUT this is not the bug — it's a Hermes-side config-truth issue. Even with stream=true (which Turbohaul correctly handles), the ACTIVE_MATCH branch was broken. |

**Pattern:** A bug isolated to ONE code path can mimic several different higher-level hypotheses. Code-read discrimination beats hypothesis ranking from logs alone.

---

### Branch Failure Log (zero this cycle)

| Branch | Failure Cause | Prevention Rule |
|---|---|---|
| (none) | (none) | (none) |

---

### Process Notes

#### What worked well
1. **DAG-first discipline** — Cmdr's 22:13Z directive mandated DAG SOP → GitNexus → SCOUT → fix → smoke loop. Following it strictly meant we triangulated from code + audit + advisor BEFORE writing a line.
2. **SCOUT Conv 8 parallel consult** — Advisor's Mode D quarantine doc + msg 47 endorsement landed the same hypothesis my code-read converged on. Independent corroboration in <10min.
3. **RBSRS dual-gate (Failure Predictor + Simplicity Advocate)** — Failure Predictor #16 caught CRIT-1 TOCTOU race that would have made the fix intermittent (only ~25% of slot pickups timing-window-overlap). Simplicity Advocate #15 reshaped the fix to dissolve CRIT-1 entirely via predicate change, saving ~15 LoC and avoiding a defensive refactor in `submit()` that would have widened blast-radius.
4. **GitNexus blast-radius** — Even with stale index on Wave 3+ symbols, the `Slot` class impact (4 files, LOW) and `worker_loop` test-caller list (12 callers) gave concrete bounds on what could break.
5. **Single-iteration smoke** — pytest 71/71 + RELAY HANDS 5/5 first time. The triangulation paid off in zero rework.

#### What was suboptimal
1. **Stale GitNexus index** — Required re-indexing on Wave 3+ symbols (`_complete_fn`, `submit_for_streaming`, `stream_ready_event` all "not found"). Re-index kicked off in background but probes couldn't wait. Recommendation: re-index after every Forgejo push (or once-daily cron).
2. **SCOUT subprocess "Argument list too long" errors** — Conv 8 has 50+ msgs now with embedded code blocks. Multiple advisor responses came back as `[SCOUT ERROR: subprocess spawn failed: [Errno 7] Argument list too long]`. RELAY filed RC-03FC1B for SCOUT auto-compression; shelved by Cmdr.
3. **Two false-start bash heredocs** — Python inside `bash -c` inside `py -c` inside `ssh exec_command` is fragile. Spent ~5min debugging quoting. Eventually moved to write-script-to-file + sftp + exec pattern, which is bulletproof. Lesson: NEVER nest 3 levels of heredoc-style quoting — write the inner script to a file.
4. **Bare-curl localhost POST RST mystery** — `POST localhost:11401/v1/chat/completions` returns Connection reset by peer at 0.0s wall, regardless of prompt size or stream flag. ZeroTier path (`10.244.136.121:11401`) works fine. Routed test traffic via RELAY HANDS instead of bare-curl. Should be investigated as separate RC.
5. **Initial wiki misread** — During the catch-up wake, the block after a `─────────────` divider in Cmdr's message turned out to be HANDS pane output (the failing agent trying to look up Logbook FE), not a separate ask to PL. PL spent ~10min writing a Logbook FE wiki before Cmdr course-corrected. Wiki saved at `C:/Users/Matthew/AppData/Local/Temp/turbohaul_handoff/LOGBOOK_FE_WIKI.md` as a pre-bake for when HANDS recovers and tries again. Not loss, but not the deliverable.

---

### Knowledge Base Updates

- [x] **Memory file**: `project_wave4b7_round8_streaming_active_match_shipped_20260517.md` (PL memory index)
- [x] **Memory cross-references** updated for `[[project-wave4b5-multi-turn-persistence-shipped-20260517]]` etc.
- [x] **Forgejo commit message** carries full root-cause + fix narrative for `git log` archaeology (de0602e)
- [x] **This RETRO doc** filed to `Solutions/Retrospectives/RETRO_2026-05-17_wave4b7_round8_streaming_active_match.md`
- [x] **Turbohaul-Manager repo `docs/`**: same RETRO copied into the repo (so devs reading the codebase find it)
- [ ] **CHANGELOG.md** for Turbohaul-Manager repo — should add Wave 4b.7 entry (follow-on RC, not blocking)
- [ ] **SOP doctrine update** (NEW rule, recommendation pending Cmdr go):
  > When adding a NEW execution branch (e.g. streaming-path) to an existing code path that has multiple entry points (anchor + ACTIVE_MATCH + cancel-recover + …), explicitly enumerate ALL entry points and add a test that covers each. Wave 3 SSE pass-through landed the anchor's streaming branch but missed the ACTIVE_MATCH branch — a code-review checklist item for future protocol-shape additions.

---

### Recommendations

1. **Predicates on data semantics outlive predicates on object identity.** `client_meta["stream"]` (a logical claim about what the caller wants) outlives `slot.stream_ready_event is not None` (an object-existence claim that's subject to TOCTOU). Use the former; assert the latter as invariant.
2. **Mirror existing-passing-branch logic BEFORE refactoring into a helper.** Two duplicated branches that work is better than one helper that bundles a bug fix with an abstraction guess. Defer abstraction until both branches have passing integration tests; THEN unify.
3. **Bug-isolation through code-read + SCOUT triangulation + RBSRS dual-gate beats single-line fix.** Advisor pre-locked a one-line `event.set()` fix (msg 47). It was correct but incomplete — would have unblocked the route while ALSO making worker call `_complete_fn` in parallel, violating Failure Predictor #16's single-slot invariant. The structural fix (skip `_complete_fn` on streaming-promoted match) was visible only after reading the FULL ACTIVE_MATCH branch. Two of three reviewers liked the one-liner; the code-read found the right shape.
4. **Re-index GitNexus after every major Forgejo push.** Stale index on Wave 3+ symbols cost ~5min of "symbol not found" → grep fallback. A post-push trigger or 4-hourly cron would keep this current.
5. **Single Cmdr-mandated SOP execution > parallel hypothesis exploration.** The DAG → GitNexus → SCOUT → fix → smoke loop is opinionated and serial. Following it strictly produced single-iteration success vs the multi-iteration Wave 2.3 cycle where we tried 7+ angles in parallel before finding Path 3.
6. **Heredoc-quote nesting hazard.** Three-level nested heredocs (`bash -c "py -c \"...\""`) are fragile. Always write inner scripts to files + sftp + exec by path. Already documented in memory `feedback_shell_quoting_backticks_use_python_file_via_sftp` — re-affirmed here.

---

### Future Improvements for Similar Tasks

- **Pre-fix smoke target catalog**: maintain a list of "minimal commands that exercise each FSM branch" so future fixes can run a 30s sanity smoke (per branch) without firing a full HANDS dispatch.
- **Audit-DB query helper**: bake `audit_query.py` into the container's `/opt/venv/bin/` so any agent can `docker exec turbohaul-demo audit_query` and see last N events. The escape-quote-hell pattern of inline sqlite queries cost time today.
- **ACTIVE_MATCH-streaming integration test**: add `test_streaming_active_match_warm_reuse_passes_handle` to `tests/test_worker_loop.py`. The unit-test bar is met (71/71) but the *integration* path (streaming submit → grace → matched-streaming-submit → ACTIVE_MATCH → handoff) wasn't asserted in pytest. Smoke caught it; tests should too. Filing as follow-on RC.

---

### Cross-references
- Predecessor retro: `RETRO_2026-05-17_wave4b5_4b6_multi_turn_persistence.md`
- SCOUT consult: Conv 8 session `36eec780-f4ff-4783-a050-98b4a46cac87` thread `5689213f-a1ce-4b6e-b1ab-8e6e6fd4c668` msg 41 (PL question) ↔ msg 47 (Advisor endorsement)
- RBSRS scoring: Failure Predictor #16 task entry (red-hat) + Simplicity Advocate #15 task entry (human-thinker)
- Memory: `project_wave4b7_round8_streaming_active_match_shipped_20260517.md`
- SOP: `SOP_110_Percent_Rule.md` (this very SOP)

---

*Generated 2026-05-17 by PL Claude per Cmdr 23:38Z directive "110% retrospective SOP, Commit to Forgejo".*
