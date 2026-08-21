# TODOS.md — deferred work with context

## CI: GitHub Actions macOS arm64 runner for the engine test suite

- **What:** Workflow running the XCTest suite (module tests, parser/config edge tests,
  Metal toy-kernel tests) on a macOS arm64 runner on every push.
- **Why:** The oracle chain only protects the project if it runs without anyone
  remembering to run it. GitHub's M-series macOS runners have working Metal, so even
  GPU-diff tests can run in CI.
- **Pros:** Regressions caught at commit time across all phases; the fast-oracle
  investment (Accelerate wrapper) pays out automatically.
- **Cons:** macOS runner minutes are ~10× Linux pricing. The full logit test needs the
  ~3 GB checkpoint (not in git), so CI needs either weight caching or a weights-free
  subset.
- **Context (as of 2026-08-20):** Repo not yet scaffolded. Likely shape: CI runs
  everything except the full end-to-end logit test — per-module activation fixtures
  ARE checked into git (~13 MB) and cover most of the correctness surface; full
  end-to-end stays a local target. Decided as "capture, revisit after Phase 1" in the
  2026-08-20 eng review (D10).
- **Depends on / blocked by:** Phase 1 test suite existing; repo scaffold (/project-init).
