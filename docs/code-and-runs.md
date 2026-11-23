# Code architecture, builds, runs, and outputs

## Executable layout

Every executable folder has its own `CMakeLists.txt` and builds a target named `main`. The root `CMakeLists.txt` belongs to the legacy root `main.cpp`, not to the current production models.

| Program | Current role | Parameter-file argument |
|---|---|---|
| `resolvedGrainCooling` | Random-polycrystal cooling, damage, homogenization, solver optimization | Optional `argv[1]`; default `./input.json` |
| `hopgCooling` | Resolved-grain fork with mosaic orientations, damage switch, stress statistics | Optional `argv[1]`; default `./input.json` |
| `mmsSandbox` | Separate zero-damage MMS verification | No run-specific argument used by its driver; default input |
| `uniaxialDamageTest` | Separate damage integration/coupling verification | No run-specific argument used by its driver; default input |
| `polycrystalCooling` | Earlier implementation retained for history | Legacy |

Resolved-grain `main.cpp` now contains only process initialization, parameter-file selection, problem construction, and exception handling. Its implementation is divided by responsibility:

- `include/resolved_grain/`: parameter schema, material tensors, the standalone damage law, safe convergence measures, orientations, quadrature-point state, cell assembly, WorkStream scratch/copy data, output postprocessors, and the `ElasticProblem` interface;
- `src/parameters.cpp` and `src/material.cpp`: parameter declarations and local constitutive tensors;
- `src/cell_assembly.cpp`: the production and manufactured-solution cell kernel;
- `src/damage_law.cpp`: the independently testable scalar backward-Euler damage update;
- `src/setup.cpp`: mesh/group input, grain and boundary setup, caches, constraints, and refinement-cycle preparation;
- `src/assembly.cpp`, `src/solvers.cpp`, and `src/damage.cpp`: cell assembly, linear solves, damage/coupling updates, and converged-damage commitment;
- `src/periodic.cpp`: periodic condensation and homogenization;
- `src/output.cpp` and `src/run.cpp`: field/CSV/PVD output and converged-state recording, separated from run/refinement/temperature-step orchestration.

`ElasticProblem::run()` contains only `initializeRun()`, `initializeMicrostructure()`, `runRefinementCycles()`, and `finalizeRunOutput()`. Lower-level helpers retain the original operation order. The abstraction and module-boundary rules are defined in [development-policy.md](development-policy.md).

HOPG remains a single-file fork. There is no shared library that synchronizes later changes between the two production implementations. Within resolved-grain, `libresolved_grain.a` links both the production executable and test drivers, so unit and production-MMS tests exercise the current numerical sources rather than copies.

## Build requirements and commands

The CMake files request deal.II 9.8.0. Use a deal.II installation with the features required by the selected solver. Resolved-grain testing additionally uses GoogleTest 1.11+, CTest, Python 3, and SymPy; these are located through CMake and never downloaded implicitly. Use `-DBUILD_TESTING=OFF` only for a deliberately model-only build.

- `direct`: UMFPACK;
- `cg`: built-in CG with SSOR;
- `amg`: CG with Trilinos ML, and MPI initialization even for a one-rank run.

Example for either current model:

```bash
cd resolvedGrainCooling       # or hopgCooling
cmake -S . -B build -DDEAL_II_DIR=/path/to/deal.II
cmake --build build --parallel 4
./build/main input.json
```

If deal.II is already discoverable in the container, `-DDEAL_II_DIR=...` may be omitted. Run from the model directory because `MeshFile`, `CellToMaterialFile`, and default parameter paths are relative. The 2026-11-16 documentation audit could not demonstrate a build because its environment had no CMake, Docker, or deal.II. On 2026-11-17, all four current programs were built successfully in the existing container; see the dated environment and results in [studies.md](studies.md#fresh-local-bootstrap-2026-11-17).

The generic `.vscode/tasks.json` single-file GCC command cannot link this deal.II project. `compile_commands.json` at repository root is an ignored, machine-specific absolute symlink and may be broken after cloning.

### Resolved-grain API documentation

Doxygen is optional for the executable build. When it is installed, CMake defines a `docs` target:

```bash
cd resolvedGrainCooling
cmake -S . -B build
cmake --build build --target docs
```

HTML is generated under `build/doxygen/html/`; Graphviz diagrams are disabled so `dot` is not required. Doxygen markup and parameter documentation warnings are errors. Every function and method declaration must have Doxygen documentation; undocumented-member warnings remain disabled to avoid annotation-only churn for trivial data members. Document contracts at declarations, including units, frames, state changes, and lifetimes. Implementation comments retain only local reasoning. See [development-policy.md](development-policy.md) for the complete documentation and design rules.

### Production-linked unit and MMS tests

With the default `BUILD_TESTING=ON`, the build creates `resolved_grain_tests` and `resolved_grain_mms`. GoogleTest cases cover constitutive response, rotations, damage roots, stability, and zero-safe convergence measures. `tests/symbolic/generate_oracles.py` generates the independent SymPy references, including analytic MMS displacement gradients and body forces with temperature-dependent stiffness and secant CTE. No generated header is tracked. The exact 14 test cases, manufactured fields, analytical production cases, numerical tolerances, and coverage limits are documented in [verification.md](verification.md#implemented-tests-setup-oracle-and-pass-criteria).

For a local development loop:

```bash
cd resolvedGrainCooling
ctest --test-dir build --output-on-failure --no-tests=error
./build/resolved_grain_tests --gtest_filter='DamageLaw.*'
```

The authoritative repository-level result remains `verify.sh check`, because CTest alone does not check process isolation, production CSV/PVD artifacts, or the other executables.

## Deterministic local verification

`scripts/verify.sh` is the resource-limited build and verification entry point. Targets are always explicit; invoking it without a target is an error, and there is no implicit full study suite.

```bash
./scripts/verify.sh --list-targets
./scripts/verify.sh check
./scripts/verify.sh analytical-resolved
./scripts/verify.sh production-mms-resolved
./scripts/verify.sh env build
./scripts/verify.sh mms-small uniaxial-small smoke-resolved smoke-hopg
./scripts/verify.sh damage-resolved
./scripts/verify.sh build-resolved ab-resolved \
  --reference-run verification-runs/<accepted-reference-run>
```

The complete approved bootstrap gate can also be selected in one invocation:

```bash
./scripts/verify.sh env build mms-small uniaxial-small smoke-resolved smoke-hopg
```

`build` selects independent Release builds of `mmsSandbox`, `uniaxialDamageTest`, `resolvedGrainCooling`, and `hopgCooling`. The individual `build-mms`, `build-uniaxial`, `build-resolved`, and `build-hopg` targets are also available. Tests require an existing executable and report `SKIPPED` if its build was not selected or completed.

`check` is the unified required gate: environment, all four builds, 14 production-linked GoogleTest cases, production-source temperature-dependent MMS, the reduced copied-source MMS/uniaxial suites, active resolved-grain damage, HOPG smoke, and seven analytical resolved-grain cases. It returns zero only if all required targets pass and protected files remain unchanged. Build errors, test failures, timeouts, and prerequisite skips make the result fail. `summary.json` contains target-level and numerical results; `junit.xml` contains individual GoogleTest cases and target-level integration results. With `--reference-run`, existing resolved-grain A/B targets are also compared; newly introduced analytical targets absent from that reference report A/B as unavailable without failing their independent oracle.

The reduced targets have deliberately limited scope:

- `mms-small`: 1,000 and 8,000 cells only, with all ten temperature steps at each level;
- `uniaxial-small`: pseudo-time steps 0.1, 0.05, and 0.025 only (10, 20, and 40 steps);
- `smoke-resolved`: 1,000 cells and two steps from 2000 to 1960 K;
- `smoke-hopg`: 1,000 cells and two steps from 2500 to 2400 K;
- `damage-resolved`: the complete ten-step, 1,000-cell resolved-grain history from 2000 to 1800 K, with current material fits and an explicitly selected 60 MPa damage threshold instead of the tracked 200 MPa value. Only the isolated input is changed; nonzero monotone damage is required.
- `production-mms-resolved`: the current production cell kernel with spatially varying temperature, temperature-dependent elastic constants and secant CTE, full Dirichlet data, and SymPy-generated forcing on 64, 512, and 4096 cells. Both successive L2 and H1 rates are checked.
- `analytical-resolved`: uniform periodic, uniform clamped, uniform traction-free, two-phase periodic laminate, and exact zero-CTE response under all three boundary-condition families. The laminate reference is obtained independently with SymPy from compatibility, traction continuity, and zero average stress.

`ab-resolved` selects nine resolved-grain cases: short WorkStream and serial smoke tests, complete active-damage histories on both assembly paths, exact periodic condensation, zero-displacement and free-surface boundaries, uniform-orientation analytical response, and an 8,000-cell initial-refinement case. With `--reference-run`, every macroscopic CSV cell is compared (integer columns exactly), along with PVD sequences, convergence/macroscopic iteration logs, and VTU contents. Only the deal.II generation timestamp line is normalized in VTU comparisons; numerical field bytes remain strict. The uniform case independently checks `Ebar` and the Kelvin `Cbar` diagonal. Without a reference, only the intrinsic convergence/output and analytical checks run.

The 64,000-cell MMS level, the 80/160-step uniaxial cases, the 64,000-cell HOPG baseline, and full studies are not selected by these targets and require separate approval.

Defaults are two build jobs, two solver threads, sequential tests, an 8 GiB per-process address-space limit, and a 30-second termination grace period. Timeouts are 120 seconds for `env`, 600 seconds for each build, production smoke test, and `damage-resolved`, 600 seconds for `mms-small`, and 900 seconds for `uniaxial-small`. Use `--jobs`, `--threads`, `--memory-gib`, and `--set-timeout TARGET=SECONDS` only as explicit, recorded resource choices. `--deal-ii-dir` selects another compatible installation.

Every invocation creates an ignored `verification-runs/<run-id>/` directory containing the Git revision and status, source hashes (including resolved-grain headers and module sources), exact commands, logs, inputs, numerical summaries, timing/resource records, and generated outputs. The manifest records `--reference-run` when supplied. Verification drivers and production programs run in isolated workspaces; tracked inputs, archived JSON/plots, and `results/` are hash-checked for contamination. A zero exit code is insufficient: the runner also checks expected step/mesh counts, numerical convergence and regression criteria, final temperatures, finite CSV values, and PVD/field consistency. Failed and timed-out runs retain partial evidence.

The project Skill `.kimi-code/skills/fea-development/SKILL.md` applies the architecture, Doxygen, and change-control policy during implementation. `.kimi-code/skills/fea-verify/SKILL.md` provides target-selection and result-interpretation guidance. Neither skill duplicates model theory or archived results.

### Solver profiling and benchmark studies

Production logs include cumulative linear-solve counts, total CG iterations over all right-hand sides, `Cbar` rebuild counts, and timer sections for solver setup/solves, average-stress integration, damage prediction, convergence checks, diagnostics, and output. Timer sections can be nested: for example, `Convergence check` includes its matrix assembly, and `Process solution` includes an average-stress calculation. Do not sum nested sections as if they were disjoint.

`scripts/benchmark_solvers.py` runs explicit solver/thread/refinement Cartesian products in isolated workspaces. For example:

```bash
python3 scripts/benchmark_solvers.py \
  --solvers cg amg --threads 1 2 4 6 --refinements 1 \
  --final-temperature 1960 --timeout 600 --memory-gib 6 \
  --label '8,000-cell two-step solver/thread sweep'
```

For the current-material active-damage history, add `--final-temperature 1800 --stress-threshold 60e6 --require-damage`. The default is periodic frozen-`Cbar` quasi-Newton with Aitken coupling; `--exact-cbar` and `--coupling-acceleration none` select explicit alternatives. Linear tolerance, materials, mesh, and seed otherwise come from `--base-input`. Benchmarks reduce optional tensor output but retain the final VTU/PVD and complete macroscopic history.

Every invocation creates a separate ignored `verification-runs/<run-id>/` with host/environment data, Git state and source hashes, exact per-case inputs and commands, logs, numerical outputs, and protected-file checks. `summary.json` and `cases.json` record process status, completion checks, full timer sections, cumulative solver work, outer iterations, CPU time, per-case peak RSS, and macroscopic-history hashes. The per-case resource report measures the executable child alone; the manifest's global RSS is only a high-water mark.

A benchmark `PASS` means that the run completed and passed the standard production artifact/convergence checks, not that it was fast or that different solver configurations produced identical iterates. Compare wall time together with total CG work, outer iterations, memory, and macroscopic histories. Solver and tolerance choices are numerical changes and require separate scientific acceptance; this driver does not modify the tracked baseline or form part of `verify.sh check`. The completed 2026-11-20 SSOR/AMG/UMFPACK measurements and their limits are indexed in [studies.md](studies.md#solver-profiling-and-benchmarks-2026-11-20).

## Tracked baselines

These are configurations, not universal physical defaults.

| Setting | `resolvedGrainCooling/input.json` | `hopgCooling/input.json` |
|---|---|---|
| Temperature | 2000 to 1800 K, 20 K increments | 2500 to 1000 K, 50 K increments |
| Boundary condition | Periodic | Periodic |
| Crystallites | 1,000 | 1,000 |
| Elements per crystallite | 1 | 64 |
| Total cells | 1,000 | 64,000 |
| Orientation | Uniform random on sphere | Gaussian mosaic about global z |
| Orientation seed | 20260905 | 20260905 |
| Mosaic component sigma | Not applicable | 0.5 degrees |
| Damage | Enabled | Disabled |
| Elastic data | Polewczyk MD fits, reference 2000 K | Same elastic fits |
| CTE data | Constant Steward-derived values | Marsden secant fits, reference 2500 K |
| Linear solver | CG + SSOR | Trilinos ML AMG |
| Displacement–damage acceleration | Aitken | Aitken configured but damage disabled |
| Periodic macroscopic solve | Frozen-`Cbar` quasi-Newton | Exact seven-solve condensation |
| Threads | 12 | 12 |
| Output directory/prefix | `.` / `polycrystalCooling` | `.` / `hopg` |

The resolved-grain output prefix is historical; it does not mean the executable is `polycrystalCooling/main.cpp`.

## Safe experiment configurations

Do not modify the tracked baseline merely to run a sweep. Create a complete run-specific JSON file in the run output directory and pass it as `argv[1]`. Both current drivers follow this rule:

```bash
cd resolvedGrainCooling
python3 run_sweep.py --help
python3 run_sweep.py --bc-sweep --dry-run
python3 run_sweep.py --seed-sweep --dry-run
```

```bash
cd hopgCooling
python3 run_mosaic.py --help
python3 analyze_mosaic.py --help
```

The resolved-grain driver supports BC/resolution and seed sweeps. The HOPG driver varies mosaic spread, normally with a shared seed so the same sampled standard-normal pattern is scaled. Analysis scripts reconstruct summaries from archived logs/CSV files; they do not make an incomplete run scientifically complete.

`hopgCooling/configure_input.py` **does rewrite** `hopgCooling/input.json`; it is a baseline generator, not a per-run driver. It selects the 2500 K-referenced CTE file and checks its reference and temperature validity. Review the Git diff after running it.

The full tracked HOPG case is not a smoke test. For a small trial, use a separate configuration with `InitialGlobalRefinements = 0`, a shortened cooling interval, a unique output directory/prefix, and an external timeout. For example, from `hopgCooling/` after building:

```bash
python3 - <<'PY'
import json
from pathlib import Path
out = Path('smoke-output')
out.mkdir(exist_ok=True)
cfg = json.loads(Path('input.json').read_text())
cfg['MeshRefinementParameters']['InitialGlobalRefinements']['value'] = '0'
cfg['ModelParameters']['FinalTemperature']['value'] = '2450'
cfg['OutputParameters']['OutputDirectory']['value'] = str(out.resolve())
cfg['OutputParameters']['OutputFileNamePrefix']['value'] = 'smoke'
(out / 'input.json').write_text(json.dumps(cfg, indent=2) + '\n')
PY
timeout 10m ./build/main smoke-output/input.json
```

The timeout only bounds execution; it does not guarantee convergence or scientific validity. Inspect the log and final temperature before using output.

## Numerical structure and performance controls

- Shared-memory WorkStream/TBB assembly is optional through `UseWorkStream`; distributed MPI parallelism is not implemented for the model.
- Serial and WorkStream paths should produce round-off-equivalent results.
- Periodic exact condensation amortizes one matrix/preconditioner over seven right-hand sides.
- Frozen-`Cbar` quasi-Newton can remove most six-column rebuilds in resolved-grain runs. Its output `Cbar` may be stale even when displacement, damage, and effective CTE agree with the exact path.
- Aitken relaxation is most valuable when the displacement–damage fixed point dominates cost.
- HOPG's near-single-crystal anisotropy motivated AMG and exact condensation in the archived study; do not transfer solver rankings automatically to random polycrystals.
- `NumberOfRefinementCycles > 1` restarts the full temperature history and resets damage. It is not continuation with mapped damage.

Dated measurements and their limitations are indexed in [studies.md](studies.md).

## Outputs and reproducibility

Production runs can write:

- `.vtu` field snapshots and a `.pvd` time series;
- `<prefix>_macroscopic_history.csv` with temperature/cooling, macroscopic strain, effective CTE, homogenized-stiffness diagonal, damage, and stress summaries;
- HOPG `<prefix>_stress_statistics.csv` with volume-weighted invariant stress statistics;
- console logs containing convergence, solver, warning, and timing information;
- `tmp_sol.json`, a ParameterHandler dump, in the process working directory.

Optional full local/global stress and strain tensors are controlled by `WriteFullStressTensor` and `WriteFullStrainTensor`. Output strain is total strain; elastic strain requires subtracting thermal eigenstrain.

For every scientific run, retain the exact input, executable Git revision, seed, mesh/refinement settings, solver options, log, and required CSV/field outputs. Run directories should not share mutable field files. Check that a completed run reached `FinalTemperature`; a last CSV row from an interrupted run is not a final-state result.

## Utilities and generated artifacts

- `resolvedGrainCooling/compare_runs.py`: tabulates completed sweep summaries and macroscopic histories.
- `resolvedGrainCooling/plot_cte_comparison.py`: plots the archived damage–CTE coupling comparison.
- `hopgCooling/analyze_mosaic.py`: filters completed runs, plots stress histories, and fits mosaic scaling exponents.
- `mmsSandbox/plot_mms_verification.py`: optionally reruns MMS, then plots archived convergence data.
- `uniaxialDamageTest/plot_verification.py`: optionally reruns step sizes, computes an independent SciPy reference, and plots errors.
- `uniaxialDamageTest/timing_study.py`: historical mesh/timing experiment, not a correctness test.
- `materialData/*/*.py`: digitization, fitting, and input-update utilities; see [material-data.md](material-data.md).
- `paraview_setup.py` files: visualization helpers, not model definitions.

Python analysis/fitting scripts require their imported scientific packages, notably NumPy, SciPy, and Matplotlib. There is no pinned Python environment in the archive.

Root notebooks, `tmp_sol.json`, build trees, and field dumps are generated or legacy artifacts. They should not become hand-maintained sources of truth.

## Remaining documentation and architecture debt

The resolved-grain modules now use focused Doxygen comments and an optional documentation target. The following remain separate cleanup or design tasks:

- stale local/global notation in the recovery section of the two `homogenization.md` copies: local normal stress is `sigma_33`;
- inherited narrative and any stale comments in the independent HOPG fork;
- duplicated implementations across resolved-grain and HOPG, whose deduplication requires a separately approved A/B task;
- the resolved-grain mesh reader's legacy `test.msh` side file and placeholder suffix, retained in this refactor solely to preserve observable behavior.

Do not turn documentation cleanup into constitutive or numerical changes. Compare behavior before and after cleanup.
