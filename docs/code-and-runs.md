# Code architecture, builds, runs, and outputs

## Executable layout

Every executable folder has its own `CMakeLists.txt` and builds a target named `main` from its local `main.cpp`. The root `CMakeLists.txt` belongs to the legacy root `main.cpp`, not to the current production models.

| Program | Current role | Parameter-file argument |
|---|---|---|
| `resolvedGrainCooling` | Random-polycrystal cooling, damage, homogenization, solver optimization | Optional `argv[1]`; default `./input.json` |
| `hopgCooling` | Resolved-grain fork with mosaic orientations, damage switch, stress statistics | Optional `argv[1]`; default `./input.json` |
| `mmsSandbox` | Separate zero-damage MMS verification | No run-specific argument used by its driver; default input |
| `uniaxialDamageTest` | Separate damage integration/coupling verification | No run-specific argument used by its driver; default input |
| `polycrystalCooling` | Earlier implementation retained for history | Legacy |

The production implementations are predominantly single-file programs. Resolved-grain `main.cpp` contains parameter declaration/reading, material tensors, grain and orientation setup, assembly, linear solvers, damage updates, homogenization, output, and run control. HOPG was forked from it on 2026-10-12; the initial fork differed by only a few hundred lines, but there is no shared library that keeps later changes synchronized.

## Build requirements and commands

The CMake files request deal.II 9.8.0. Use a deal.II installation with the features required by the selected solver:

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

If deal.II is already discoverable in the container, `-DDEAL_II_DIR=...` may be omitted. Run from the model directory because `MeshFile`, `CellToMaterialFile`, and default parameter paths are relative. No portable build was demonstrated during the 2026-11-16 documentation audit: the available environment had no CMake, Docker, or deal.II.

The generic `.vscode/tasks.json` single-file GCC command cannot link this deal.II project. `compile_commands.json` at repository root is an ignored, machine-specific absolute symlink and may be broken after cloning.

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

## Known comment/documentation debt for the refactor

The following are cleanup targets, not feature requests:

- current source comments saying periodic homogenization is not implemented, although it is;
- current source comments saying inputs still contain placeholder material flags, although current elastic fits and HOPG CTE fits are installed;
- inherited comments with outdated performance or temperature-independent-material assumptions;
- stale local/global notation in the recovery section of the two `homogenization.md` copies: local normal stress is `sigma_33`;
- tutorial-derived CMake boilerplate and excessive narrative comments;
- duplicated code across resolved-grain and HOPG forks.

Do not turn these edits into constitutive or numerical changes. Compare behavior before and after cleanup.
