# FEA nuclear-graphite crystallite model

## Session protocol

At the start of every new task:
1. Read AGENTS.md and the relevant current-state documents under docs/.
2. Inspect git status, current branch, and recent commits.
3. Identify the approved task and its acceptance criteria.
4. Do not infer unfinished work from an old conversation.
5. If repository state and task description disagree, stop and ask.
6. Propose an operative plan before modifying code.

Before completing a task:
1. Run the applicable deterministic verification script.
2. Record actual commands, environment/image version, and outcomes.
3. Review the complete Git diff.
4. Update current-state documentation when behavior or status changed.
5. Preserve dated study evidence separately from current-state guidance.
6. Report remaining failures and scientific uncertainties explicitly.

## Purpose and current milestone

This repository develops a crystallite-based, small-strain thermo-mechanical model for nuclear graphite. Its purpose is to connect crystallite anisotropy, orientational disorder, residual stress, and an idealized basal-crack damage variable to homogenized response. The longer-term research program includes crystallites, domains, and binder/filler-scale structure; this repository currently implements the crystallite-scale model.

The current implementations are:

- `resolvedGrainCooling/`: randomly oriented crystallites with scalar damage and periodic, free-surface, or zero-displacement boundary conditions.
- `hopgCooling/`: a fork for HOPG-like mosaic orientations; its tracked baseline is damage-free thermoelasticity.

The resolved-grain **behavior-preserving modular refactor** was completed on 2026-11-18: implementation is split into focused modules, comments are audited, and focused Doxygen documentation and nine-case A/B regression coverage are available. HOPG remains a separate single-file implementation. Sharing code between the forks is a separate architectural task; scientific changes require separate approval.

This current-state documentation was audited against archived commit `67dd766` (2026-10-12). No build or run was performed during the documentation audit because the environment lacked CMake, Docker, and deal.II; archived verification data are identified as such.

## Source-of-truth policy

Use information in this order:

1. The relevant program entry point and module sources (`include/` and `src/` where present), together with the exact run `input.json`, determine actual behavior.
2. This file and `docs/` are the curated current-state explanation and operational guide.
3. Folder READMEs, derivations, optimization reports, and `results/` are dated evidence or design history. They are not automatically current instructions.

If these disagree, inspect code and Git history, record the discrepancy, and correct the curated documentation. Do not silently copy claims between forks or revive recommendations from old reports. Preserve the scientific context of historical results.

Current-state documents:

- [Model formulation and assumptions](docs/model.md)
- [Code architecture, builds, runs, and outputs](docs/code-and-runs.md)
- [Verification status and regression expectations](docs/verification.md)
- [Material-data provenance and validity](docs/material-data.md)
- [Study evidence, performance, and documentation status](docs/studies.md)

## Repository map

| Path | Status and role |
|---|---|
| `resolvedGrainCooling/` | Current resolved-polycrystal implementation and its experiment drivers. |
| `hopgCooling/` | Current HOPG/mosaic implementation, forked from resolved-grain code. |
| `mmsSandbox/` | Separate zero-damage, temperature-independent MMS verification program. |
| `uniaxialDamageTest/` | Separate displacement-controlled damage/ODE verification program. |
| `materialData/` | Digitized source data, fit-generation scripts, model-ready JSON fits, and provenance notes. |
| `polycrystalCooling/` | Earlier physically meaningful implementation; historical, not the current baseline. |
| root `main.cpp`, `cubeTest.cpp`, `oldCoolingModel.cpp` | Legacy development programs; do not use them to describe the current model. |
| `results/` under model folders | Archived configurations, logs, CSV data, plots, and reports. Read each with its own material and solver configuration. |
| root notebooks and `tmp_sol.json` | Legacy/generated artifacts, not operational documentation. |
| `compile_commands.json` | Ignored absolute symlink to a developer machine; generated local configuration, not a portable project file. |
| `.vscode/` | Generic local editor setup; the GCC active-file task is not a deal.II build. |

The mesh and `groupFile` copies used by the programs are identical archived assets. Each program is nevertheless built independently from its own directory.

## Non-negotiable conventions

- Use SI units in the model: metres, pascals, kelvin, and dimensionless strain/damage. Some reports print stress in MPa.
- `11`, `22`, `33` denote **local crystal axes**; `xx`, `yy`, `zz` denote **global laboratory axes**. Axis 3 is the local c-axis and basal-plane normal.
- The damage driver is local `sigma_33`, never global `sigma_zz` unless the two happen to coincide.
- Rotation matrices stored per cell map global tensors to local coordinates; local basis vectors are matrix rows.
- Constitutive assembly uses Kelvin notation. Shear diagonal entries are `2*C44`, `2*C44`, and `2*C66`, with `C66 = (C11-C12)/2`. Do not replace these with Voigt entries.
- CTE inputs are **secant** coefficients referenced to the stress-free initial temperature: thermal strain is `alpha_sec(T)*(T-T0)`, not the integral of the instantaneous CTE evaluated implicitly by the solver.
- Production cooling models use a spatially uniform prescribed temperature. Only the MMS program uses a spatially varying manufactured temperature field.
- Damage is non-decreasing, cooling-only, capped numerically at `0.999`, and driven only by local tensile `sigma_33` above threshold.

See [model.md](docs/model.md) for the equations and scope of these assumptions.

## Build and run

Each model/verification folder is a separate deal.II CMake project requiring deal.II 9.8.0. AMG also requires the Trilinos ML support in the deal.II installation; the direct solver requires UMFPACK.

From one program directory, for example:

```bash
cd resolvedGrainCooling
cmake -S . -B build -DDEAL_II_DIR=/path/to/deal.II
cmake --build build --parallel 4
./build/main input.json
```

Run from the program directory so relative mesh and `groupFile` paths resolve. The two current production programs accept an optional parameter-file argument; without it they read `./input.json`. Always pass a run-specific configuration for sweeps. Do not write through a `build/input.json` symlink or overwrite the tracked baseline to create an experiment.

The tracked HOPG baseline has 64,000 cells and is not a small smoke test. Use reduced configurations and explicit time/resource limits for trial runs. See [code-and-runs.md](docs/code-and-runs.md) and [verification.md](docs/verification.md).

## Collaboration and change control

- The user and supervisors choose scientific priorities. The architect turns them into operative tasks and checks scientific/architectural consistency. Implementation starts only after the user approves the operative task.
- Review changes as focused Git diffs. Separate behavior-preserving edits from constitutive, material-data, boundary-condition, and solver changes.
- Actively identify contradictions and uncertain assumptions. When project intent is missing, ask the user rather than inventing a purpose or silently selecting new physics.
- `hopgCooling/main.cpp` and the resolved-grain modules under `resolvedGrainCooling/include/` and `src/` are independent implementations derived from a duplicated fork. A shared bug fix must be checked deliberately in both; do not assume editing one updates the other. Deduplication itself is an architectural task requiring approval.
- MMS and uniaxial tests are also separate source copies. Passing them does not automatically test later edits in either production fork.
- Preserve exact input, Git revision, seed, mesh resolution, solver options, and output files with every scientific run.
- Treat warnings about high damage, non-convergence, fit-range violations, or stale homogenized stiffness as validity issues, not cosmetic log messages.
- Keep credentials, including the DeepSeek API key, outside the repository and out of Dockerfiles, devcontainer files, logs, and committed environment files.

## Prohibited unapproved changes

During refactoring or documentation work, do not:

- change the kinetic law, damage coupling, stress-free temperature, material fits, or fitted temperature ranges;
- switch secant CTE to instantaneous CTE or integrated thermal strain;
- add healing, crack closure, shear-driven damage, basal slip, irradiation, chemistry, or pore/QI-particle physics;
- change grain assignment, orientation sampling, periodic homogenization, solver tolerances, or acceleration algorithms;
- reinterpret `MosaicSpreadDeg` as FWHM;
- delete historical reports or present their numerical results as configuration-independent truths;
- claim verification beyond the actual PDE, material, damage, and source-copy scope described in [verification.md](docs/verification.md).

## Baseline acceptance and open scientific decisions

A refactor is acceptable only after building the affected programs, rerunning the applicable verification programs, and comparing representative current-model outputs. Serial and WorkStream assembly paths should agree to round-off. Periodic exact and frozen-`Cbar` paths must be compared when that machinery changes. If `Cbar` itself is reported after a frozen-`Cbar` run, rebuild it at the converged state.

The following remain scientific decisions, not agent discretion:

1. Whether the resolved-grain tracked baseline should retain its constant Steward-derived CTEs or adopt the available Marsden fits referenced to 2000 K.
2. Calibration and physical justification of `StressThreshold = 200 MPa`, `A = 4`, and `m = 2`.
3. The intended domain-formalization observables and acceptance criteria for the crystallite model.
4. When crack closure, tension/compression asymmetry, shear mechanisms, or integrated thermal strain should enter the model.
5. Which material digitizations and incomplete citations must be replaced before publication-quality predictions.

These open items do not block a behavior-preserving refactor; changing them does require a separate scientific task.
