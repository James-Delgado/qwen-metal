# Agent Operation Procedure — qwen-metal

> The per-task SOP. The minimal user prompt is *"Pick up the next ready task from
> docs/PRIORITIES.yaml."* That is sufficient — this document is the rest.
> This SOP is authoritative for this repo and overrides global defaults.

## The procedure (every task)

1. **Orient.** Read `CLAUDE.md`, this file, `docs/PRIORITIES.yaml`,
   `docs/METHODOLOGY.md`, and the current phase spec in `docs/phases/`. Find the
   lowest-`rank` task with `status: ready`. That is your task — do not pick another.
   If the task belongs to a phase with no spec, stop and say so (specs are written
   just-in-time; see PLAN.md).
2. **Plan.** Post a short plan as your first message: restate the deliverable,
   approach, which standards apply (incl. relevant PLAN.md invariants / hard rules),
   how you'll verify, anticipated follow-ups, and any decision that genuinely needs
   James. If none, proceed without waiting.
3. **Claim it.** Set the task `status: in_progress` + `started_at` (UTC). Commit
   THIS change alone: `chore(priorities): mark <ID> in_progress`.
4. **Execute.** Build the deliverable. Tests land with the code (red→green→refactor).
   GPU kernels: correctness test vs CPU reference BEFORE any optimization (hard rule 3).
5. **Verify — show the output.** Run the tests / build and paste the actual results.
   Never paraphrase ("all green") — quote it.
6. **Post-task review.** Re-read the deliverable as if reviewing someone else's code;
   cross-check against METHODOLOGY.md and the phase spec; note deviations honestly.
7. **Append discovered follow-ups** to `docs/PRIORITIES.yaml` (unique id, rank,
   status, a `notes` line linking where it surfaced). If none: say so explicitly.
8. **Mark done.** Set `status: done` + `completed_at`; flip any now-unblocked
   dependents from `blocked` to `ready`.
9. **Commit the deliverable** (code + tests + docs + the priorities update) in one
   commit: `phaseN(<scope>): <subject>` with a "Closes <ID>" line (commit style per
   CLAUDE.md session discipline).
10. **Session log:** if the session decided or measured anything, append the dated
    entry to DECISIONS.md (this repo's session-log mechanism — see CLAUDE.md).
11. **Report.** One scannable message: what shipped, commits, test status,
    follow-ups, what's next. For SPEC-Pn (phase spec) tasks additionally
    (decided by James, 2026-08-26): report every veto-flagged decision —
    judgment-derived gate constants, new pins/schemas — directly in the
    conversation, item by item with the chosen value and its rationale,
    never only as a pointer into DECISIONS.md.

## When to pause

Irreversible or convention-setting decisions wait for James: model pinning, packed
weight layout changes, numeric gate values, new top-level dirs/modules, methodology
deviations, anything touching the benchmark protocol pins. Reversible,
convention-following decisions proceed — surface them in the plan, don't block.

Tasks marked `owner: james` in the backlog (device work, on-phone benchmarks,
Xcode deployment) are never executed by agents — prepare the code/harness for them
and stop.
