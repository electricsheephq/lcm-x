# Refactor window

Structural refactors land in a protected window between a GA tag and the next rc1, so the next release gauntlet covers them.
After the v0.27.0 GA the window is 5 working days, before v0.29 feature work.

## What may land in the window
- PRs labelled `refactor-only`: one move or one extraction each, with no behaviour change. Adjacent phases of one function,
  extracted in place within the same module and pinned by one golden trace that covers both, count as one extraction; a
  move to a different module, or a second function or family, is its own PR. The 200 non-generated-line stop signal still
  applies per PR.
- Footgun fixes that already have a failing (strict xfail) probe test, as separate fix PRs.

## Rules for a refactor-only PR
1. **Characterize first.** The tests that pin current behaviour land before the move, in the same PR or an earlier one.
2. **Byte-identical against the immediate pre-refactor commit.** Run the differential with `--baseline <pre-refactor sha>`; every
   oracle must match exactly. Emitted projection bytes are pinned (`tests/compat/golden`), because they are a replay protocol.
3. **Previous-GA comparison runs separately** and must not regress.
4. **Size.** About one agent-day of work at most. A window may be only a few days long, so larger items are sliced across windows.
5. **Side effects move; they are not removed.** When a pure planner is extracted, its caller keeps every write the old code made
   (cursor and reconciliation updates, carrier and emission registration).
6. **No new mixins, no family splits before stage extraction**, and no re-export shims that let module monkeypatches pass while
   testing nothing.
7. **Touched tests move onto the shared harness.** When a refactor PR edits an existing engine test, that test moves to the
   `tests/harness` helpers in the same PR. There is no bulk migration (see `tests/harness/README.md`).
8. **New modules are strictly typed.** A module created by an extraction passes `mypy --strict` on that file alone, and the PR
   body records the command. Existing modules are not held to this.
9. **Name the invariants.** The PR body lists the IDs from [docs/invariants.md](../invariants.md) that the moved code serves,
   and the characterization tests that pin them.
10. **Stack depth is behaviour where code walks the stack.** The auxiliary-agent checks in `aux_session.py` walk at most
    32 frames from their caller. A move that puts a call to them (directly or through a wrapper) inside a new helper frame
    adds a regression test that drives the deepest real auxiliary path through that helper and asserts it is still
    classified as auxiliary. If any measured path comes within about 4 frames of the budget, the call stays in its
    original frame. Changing the budget is a behaviour change and needs its own issue, never a refactor PR.

## Order for the first window (v0.27.0 GA → v0.28.0 rc1 / v0.29)
1. Bypassed-child parent-state proof and rebind equivalence (tests) — they gate every per-compress state-lifetime change.
2. Verified footgun fixes (each with its probe flipping from xfail to pass).
3. Survival-fit projection codec extraction, after the survival contract and projection replay tests. survival_fit.py has the
   fewest open PRs touching it of the turn-path modules.
4. Assembly planner extractions (`select_prefix`, `select_tail`) only after the queued engine.py PRs have merged.

Background compaction (#787) default-on is planned for v0.31.0. Its slice 1 (default off) merges in the v0.28.0 queue. Per-call
compaction-state refactors are not accelerated for it, and compaction.py refactors rebase after it.
