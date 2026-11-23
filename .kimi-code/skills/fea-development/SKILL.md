---
name: fea-development
description: Refactor, document, or extend the repository's scientific FEA code using its architecture, Doxygen, change-control, and A/B development policies.
---

# Scientific FEA development

1. Read `AGENTS.md`, `docs/development-policy.md`, and the current-state documents relevant to the task. Inspect Git status, branch, and recent commits. Identify the approved scope and acceptance criteria before editing.
2. Classify the work as behavior-preserving engineering or numerical/scientific change. Do not turn an approved refactor into a constitutive, material, boundary-condition, solver, tolerance, seed, or output-schema change.
3. Propose an operative plan before modifying code. Identify module boundaries, operation ordering, ownership, and the A/B reference. Wait for approval when required by `AGENTS.md`.
4. Keep orchestration methods at one abstraction level. Put implementation work in the responsible module and avoid helper fragmentation or unapproved cross-fork deduplication. Follow the naming and scientific notation rules in the policy.
5. Document header declarations with Doxygen: purpose, parameters, returns, units, coordinate frames, state changes, and non-obvious ownership or preconditions. Use local `//` comments only for implementation reasoning. Do not duplicate API contracts at definitions or preserve historical debugging narratives.
6. Build affected programs and the applicable Doxygen target. Use the `fea-verify` skill for explicit target selection, resource-limited execution, and interpretation. Run `./scripts/verify.sh check` for numerical changes. Resolved-grain refactors additionally use:

   ```bash
   ./scripts/verify.sh build-resolved ab-resolved \
     --reference-run verification-runs/<accepted-reference-run>
   ```

7. Inspect actual numerical summaries and logs; a zero exit code alone is insufficient. Never weaken inputs, convergence criteria, or comparison tolerances to obtain a pass. Preserve failed-run evidence. For solver performance work, use `scripts/benchmark_solvers.py` and consult the dated measurements in `docs/studies.md` before proposing new optimizations; process wall time, setup, solve work, outer iterations, memory, and numerical histories must be evaluated together.
8. Review the complete diff, confirm protected inputs/results are unchanged, update current-state documentation, and report remaining limitations and deliberately preserved behavior.
