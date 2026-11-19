---
name: fea-verify
description: Run and interpret the repository's explicit, resource-limited FEA build and verification workflow.
---

# FEA verification

1. Follow `AGENTS.md` and read `docs/verification.md` for scientific scope and `docs/code-and-runs.md` for operational details. Do not infer verification coverage beyond those documents.
2. From the repository root, inspect available targets with `./scripts/verify.sh --list-targets`. Select only targets appropriate to the approved task; never run an implicit full suite.
3. Use, for example:

   ```bash
   ./scripts/verify.sh env build-mms mms-small
   ./scripts/verify.sh build-uniaxial uniaxial-small
   ./scripts/verify.sh build-resolved smoke-resolved
   ./scripts/verify.sh build-hopg smoke-hopg
   ```

   `build` selects all four builds. Defaults are two build jobs, two solver threads, sequential tests, and target-specific timeouts. Record any explicit resource overrides.
4. Treat `PASS` as meaningful only after inspecting the target's numerical summary and completion checks. `FAIL`, `TIMEOUT`, and prerequisite `SKIPPED` results are not passes. Preserve their logs and partial outputs; do not weaken scientific inputs or tolerances to obtain a pass.
5. Read the printed `verification-runs/<run-id>/summary.json`, `manifest.json`, `commands.json`, target logs, and numerical artifacts. Report actual commands, revision, wall time, resource usage, deferred cases, and remaining scientific limitations.
6. The reduced targets intentionally exclude the 64,000-cell MMS level and the 80/160-step uniaxial cases. Do not run those or large production studies without separate approval.
7. Confirm tracked inputs and archived results were not changed. Do not commit, refactor, or change scientific behavior as part of running this Skill.
