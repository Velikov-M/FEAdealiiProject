# Study and verification evidence, performance, and documentation status

## How to use archived studies

Files under `results/`, run records under `verification-runs/`, and most folder READMEs are dated study or verification reports. Their values apply only to the archived input, material data, code revision, seed, mesh, and solver path. Preserve them as scientific evidence; do not rewrite them into timeless model properties.

The curated current-state documents describe present code. A study report can remain scientifically useful even where its future-work recommendations are obsolete. Current test definitions and acceptance criteria belong in [verification.md](verification.md); this document indexes their dated supporting evidence rather than redefining the suite.

## Verification run history

This section records dated outcomes for specific source states. It is evidence, not the definition of the current test suite; see [verification.md](verification.md) for test setups, oracles, and acceptance criteria. Each `verification-runs/<run-id>/` directory is the primary record when it is available locally. These directories are Git-ignored, so the summaries below preserve the run index and key results without implying that old evidence covers later source changes.

### Solver profiling and benchmarks: 2026-11-20

This study measures the instrumented resolved-grain solver at Git revision `964c60474f241056824e9aa5240f70b5fb4ce96b` with the current uncommitted source changes. The host was the **current Ryzen 5 4500U** (6 physical cores, 8 MiB L3, 7.2 GiB RAM), not the older Ryzen 5 7500F/15 GiB machine used in `scaling_options.md`. The build was Release with deal.II `9.9.0-pre` and Trilinos ML. Do not compare these absolute times directly with archived runs using other hosts, materials, damage states, or output settings.

Common conditions were periodic boundaries, seed 20260905, frozen-`Cbar` quasi-Newton, Aitken coupling, relative linear tolerance `1e-8`, current Polewczyk elastic fits and constant secant CTEs, one final reduced field output, and sequential cases. Active-damage cases used the isolated 60 MPa threshold. AMG used two smoother sweeps unless explicitly marked otherwise; SSOR used relaxation 1.2. Process wall time is the primary timing. Timer sections can be nested and must not be summed as disjoint costs. Every accepted case below passed completion/convergence/artifact checks and left protected files unchanged. Each configuration was run once, so the tables contain measurements, not statistical timing distributions.

Instrumentation preservation was checked before benchmarking:

- nine-case A/B plus build: `verification-runs/20261120T113951.299836Z-964c60474f24-93961/`; all ten targets passed, every case had zero scaled CSV difference, identical convergence histories, and VTU equality except timestamps;
- unified gate: `verification-runs/20261120T114103.494302Z-964c60474f24-94164/`; all 18 targets passed, and Doxygen generation passed with warnings as errors.

#### Two-step 8,000-cell solver/thread sweep

All cases had 27,783 DoFs, four outer passes, and 22 linear solves. The 200 MPa threshold produced zero damage.

| Solver | Threads | Wall s | Setup s | Solve s | Total CG iterations | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| SSOR CG | 1 | 21.10 | 0.006 | 13.60 | 2254 | 267.9 |
| SSOR CG | 2 | 18.80 | 0.006 | 12.70 | 2254 | 271.6 |
| SSOR CG | 4 | **17.58** | 0.006 | 12.30 | 2254 | 277.3 |
| SSOR CG | 6 | 18.33 | 0.006 | 13.30 | 2254 | 280.8 |
| AMG CG | 1 | 19.19 | 3.86 | 7.70 | 620 | 341.1 |
| AMG CG | 2 | 17.12 | 3.67 | 7.37 | 620 | 353.2 |
| AMG CG | 4 | 15.67 | 3.50 | 6.91 | 620 | 361.5 |
| AMG CG | 6 | **15.13** | 3.49 | 6.75 | 620 | 364.9 |
| AMG CG, one sweep | 6 | 16.53 | 3.62 | 7.98 | 1000 | 360.7 |
| UMFPACK | 6 | 42.20 | 31.90 | 5.31 | 0 | 1574.0 |

AMG reduced CG work by 3.64x but required substantial setup. Its best case was 1.16x faster than the best SSOR case. One smoother sweep was not beneficial. UMFPACK was excluded from larger cases because factorization setup and memory already dominated at this size. CG/AMG macroscopic strain components agreed to approximately `5e-14` absolute; mean-stress residuals were below 0.06 Pa and should not dominate a physical comparison.

Evidence: `verification-runs/20261120T114343.938665Z-964c60474f24-94546/` for the eight two-sweep cases and `verification-runs/20261120T114720.541134Z-964c60474f24-94718/` for UMFPACK and one-sweep AMG.

#### Full active-damage 8,000-cell history

| Solver | Threads | Wall s | Setup s | Solve s | Linear solves | Total CG | Outer passes | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SSOR CG | 4 | **151.66** | 0.12 | 72.6 | 217 | 13733 | 70 | 276.4 |
| AMG CG | 6 | 166.14 | 60.1 | 40.6 | 198 | 3681 | 70 | 362.8 |

AMG reduced solve time by 1.79x but was 9.6% slower overall because hierarchy setup consumed 60.1 s. SSOR/AMG final mean damage was 0.09862998 and maximum damage 0.2948745; final mean/max damage agreed to approximately `1.2e-10`/`7.7e-9`. The complete histories agree in physical quantities to solver-level tolerances; near-zero mean-stress residuals and last-CG counters are not physical model outputs for this comparison.

Evidence: `verification-runs/20261120T115215.113932Z-964c60474f24-94825/` and `verification-runs/20261120T115446.910324Z-964c60474f24-94824/`. An earlier SSOR run (`20261120T114849.938559Z-964c60474f24-94777`) completed numerically but was marked `FAIL` by the benchmark harness because reduced output conflicted with a ten-PVD-snapshot baseline assertion. The driver was corrected to test active damage independently of output frequency and both cases were rerun; this was not a solver failure.

#### Two-step 64,000-cell finalists

All cases had 206,763 DoFs, four outer passes, and 22 linear solves. The 200 MPa threshold produced zero damage. These compare the best-known thread configuration for each solver from the 8,000-cell sweep, not preconditioners at equal thread count.

| Solver | Threads | Wall s | Setup s | Solve s | Total CG | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| SSOR CG | 4 | 252.13 | 0.09 | 222.0 | 5346 | 715.7 |
| AMG CG | 6 | **156.60** | 14.0 | 113.0 | 1333 | 1371.2 |

AMG gave a 1.61x overall speedup and reduced combined setup-plus-solve time from 222.1 to 127.0 s. Its CG count dropped by 4.01x, while peak RSS approximately doubled. Macroscopic `Ebar` differences were at most approximately `5e-13` absolute.

Evidence: `verification-runs/20261120T115832.268043Z-964c60474f24-94895/` and `verification-runs/20261120T120244.506868Z-964c60474f24-94894/`.

#### Full active-damage 64,000-cell history

| Solver | Threads | Wall s (min) | Setup s | Solve s | Linear solves | Total CG | Outer passes | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SSOR CG | 4 | 2186.26 (36.4) | 1.15 | 1360 | 271 | 32951 | 101 | 714.3 |
| AMG CG | 6 | **1817.73 (30.3)** | 349.0 | 771 | 269 | 9431 | 101 | 1418.7 |

AMG gave only a 1.20x overall speedup in the full history: solve time fell by 1.76x and total CG work by 3.49x, but setup consumed 349 s (19.2% of AMG wall time). Both solvers had six `Cbar` rebuilds and identical outer-pass counts by temperature step: `2, 2, 2, 2, 2, 4, 8, 22, 29, 28`. Steps 8–10 therefore account for 79 of 101 passes and are the primary outer-acceleration target.

Final mean damage was 0.08100395 and maximum damage 0.4572455. Across complete histories, maximum damage agreed within `5.1e-9`; excluding residual norms, asymmetry diagnostics, and last-CG counters, the largest scaled physical CSV difference was `7.2e-9`. This supports solver equivalence within the configured nonlinear/linear tolerances, not bitwise identity.

The practical next solver candidate is **safeguarded AMG hierarchy reuse across nearby damage states**, not merely enabling AMG. Even eliminating all AMG setup would reduce this run only from 1818 to about 1469 s (1.24x additional); setup reuse must therefore be combined with solve or outer-iteration improvements for a larger gain. Linear setup plus solve remains about 62% of wall time in both full 64,000-cell cases, so solver-only changes have a strict Amdahl ceiling of roughly 2.6x before other work. High-damage outer acceleration and the substantial damage/stress-integration costs remain independent targets.

Evidence: `verification-runs/20261120T120636.452421Z-964c60474f24-94973/` and `verification-runs/20261120T124302.840951Z-964c60474f24-94972/`. No production default, material input, seed, tolerance, or archived result was changed. No matrix-free, MPI, monolithic-Newton, or adaptive-tolerance implementation was benchmarked.

### Fresh local bootstrap: 2026-11-17

The final unified bootstrap gate passed at Git revision `8536acf704b18734fb5032841e38bdfe1cf160b9` with the new uncommitted workflow files and the pre-existing uncommitted `AGENTS.md` session-protocol edit. No C++ source, tracked input, or archived result was changed.

Environment recorded inside the existing container:

- Ubuntu 22.04.5 LTS, GCC 11.4.0, CMake 3.28.3, and Python 3.10.12;
- deal.II `9.9.0-pre`, short revision `20f665c7ab`, installed at `/usr/local`; this satisfies the CMake minimum of 9.8.0 and is not a 9.8.0 build;
- MPI, TBB, UMFPACK, Trilinos, and Trilinos ML enabled;
- NumPy 1.21.5, SciPy 1.8.0, and Matplotlib 3.5.1.

Actual final command:

```bash
./scripts/verify.sh env build mms-small uniaxial-small smoke-resolved smoke-hopg
```

Evidence: `verification-runs/20261117T141315.087543Z-8536acf704b1-1806/`. This directory is intentionally Git-ignored local evidence; its `summary.json`, `manifest.json`, `commands.json`, logs, exact inputs, CSV files, and plots are the primary run record.

| Target | Fresh result |
|---|---|
| Four builds | PASS, Release; incremental builds in the final gate, after successful clean builds in the earlier bootstrap invocation |
| `mms-small` | PASS in 22.2 s; 20 completed rows, 1,000/8,000 cells; L2/H1/L-infinity rates 1.9996/1.0196/1.9674; largest relative final-error difference from the archived rows was 0.000324% |
| `uniaxial-small` | PASS in 51.4 s; 10/20/40 steps completed; final damage 0.687177/0.688889/0.689688; fitted order 1.0779; maximum spatial spread 6.06448e-12; maximum algebraic stress mismatch 0.00211972 Pa |
| `smoke-resolved` | PASS in 2.5 s; 1,000 cells, two converged steps to 1960 K; consistent CSV/PVD output; zero damage in this short interval |
| `smoke-hopg` | PASS in 3.8 s; 1,000 cells, two steps to 2400 K; consistent macroscopic/stress-statistics CSV and PVD output; damage identically zero |

The final run used two build jobs, two solver threads, and sequential tests. Its largest-child RSS high-water mark was 248,396 KiB; this is not a measurement of aggregate host memory use. The earlier clean-build invocation reached 1,720,580 KiB during compilation.

Explicitly skipped under the approved resource limit:

- the 64,000-cell MMS level;
- uniaxial step sizes 0.0125 and 0.00625 (80 and 160 steps);
- all large production baselines and refined damage studies.

These fresh results cover only the reduced configurations listed here. The resolved-grain smoke interval did not activate damage, and neither smoke test is experimental validation or a replacement for the copied-source MMS/uniaxial checks.

### Current-material active-damage baseline: 2026-11-18

The pre-refactor resolved-grain baseline is reproducible with:

```bash
./scripts/verify.sh damage-resolved
```

Evidence: `verification-runs/20261118T093735.134221Z-8536acf704b1-29756/`, at Git revision `8536acf704b18734fb5032841e38bdfe1cf160b9` with uncommitted workflow changes. The isolated input preserves the tracked periodic 1,000-cell configuration, seed, current Polewczyk elastic fits, constant secant CTEs, solver settings, and ten 20 K steps from 2000 to 1800 K. Its only scientific change is `StressThreshold = 60 MPa`, approved to activate damage with the current materials; the tracked input remains at 200 MPa.

The target passed in 12.7 s using two threads, with a 225,088 KiB largest-child RSS high-water mark. All ten steps converged without warnings, all ten PVD fields and the macroscopic CSV were consistent, no protected files changed, and damage was finite and monotone. Damage first appeared at 1860 K; at 1800 K the mean and maximum damage were 0.117217 and 0.166286. The 60 MPa threshold is a numerical baseline choice, not a calibration or experimental validation.

An initial otherwise-identical 200 MPa run also completed all ten steps but had zero damage, with maximum local `sigma_33 = 88.98 MPa`; its evidence is retained in `verification-runs/20261118T093504.373533Z-8536acf704b1-28404/`. The archived active-damage run under `resolvedGrainCooling/results/bc-resolution-sweep-cte-qn/periodic_01cpg_seed20260905/` used both constant CTEs and older constant elasticity, so it is historical context rather than a numerical regression target for this baseline. The run manifest, exact input, commands, logs, and numerical artifacts hold the detailed evidence.

### Resolved-grain modular refactor A/B: 2026-11-18

The accepted pre-refactor reference is `verification-runs/20261118T112247.213937Z-8536acf704b1-84891/`, generated at Git revision `8536acf704b18734fb5032841e38bdfe1cf160b9` with:

```bash
./scripts/verify.sh ab-resolved
```

All nine reference cases passed in 68.968 s with two solver threads, sequential tests, and an 8 GiB per-process address-space limit. Its largest-child RSS high-water mark was 288,572 KiB. No protected files changed.

The final modular source was built and compared against that unchanged reference with:

```bash
./scripts/verify.sh build-resolved ab-resolved \
  --reference-run verification-runs/20261118T112247.213937Z-8536acf704b1-84891
```

Final evidence: `verification-runs/20261118T114600.232456Z-8536acf704b1-97289/`. The Release build and all nine cases passed in 95.9 s total. The combined invocation's largest-child RSS was 1,295,340 KiB during compilation; the preceding run-only modular comparison peaked at 289,000 KiB. The run used the same resources and deal.II installation as the reference. No protected input or archived result changed.

Across all cases, the comparator covered 34 macroscopic CSV rows and 27 VTU fields. Every CSV value matched exactly at the written precision (`max_scaled_csv_difference = 0`), integer columns and convergence/macroscopic iteration histories were identical, PVD sequences matched, and every VTU byte matched except the independently varying deal.II generation timestamp line. Both active-damage cases ended at mean/max damage `0.11721722717` / `0.16628639651`; the refinement case completed with 8,000 cells. The uniform-orientation case also passed its independent analytical `Ebar` and Kelvin `Cbar` checks.

Doxygen 1.9.1 documentation was generated successfully with warnings treated as errors:

```bash
cd resolvedGrainCooling
cmake --build build --target docs
```

The resulting HTML is under `resolvedGrainCooling/build/doxygen/html/`. Graphviz output is disabled because `dot` is not installed. This gate covers the refactor configurations only: HOPG and the separate MMS/uniaxial programs were not changed or rerun, large production studies remain excluded, and no experimental validation or new physical verification is implied.

#### Orchestration and interface-documentation follow-up: 2026-11-19

At Git revision `964c60474f241056824e9aa5240f70b5fb4ce96b`, `ElasticProblem::run()` was reduced to four orchestration calls, helpers were moved to their responsible modules, and all resolved-grain method declarations received Doxygen contracts. The project policy is `docs/development-policy.md`; the accompanying development skill is `.kimi-code/skills/fea-development/SKILL.md`.

The same nine-case command and accepted reference were rerun:

```bash
./scripts/verify.sh build-resolved ab-resolved \
  --reference-run verification-runs/20261118T112247.213937Z-8536acf704b1-84891
```

Evidence: `verification-runs/20261119T111607.415971Z-964c60474f24-75769/`. The Release build and all nine cases passed in 126.8 s, including 57.2 s of compilation. Two build jobs and two solver threads were used with an 8 GiB per-process limit; largest-child RSS was 1,291,900 KiB during compilation. Every case again reported `max_scaled_csv_difference = 0`, with identical convergence histories and all 27 VTU fields matching apart from generation timestamps. No protected files changed. Doxygen generation also passed with warnings treated as errors.

This follow-up does not extend scientific verification coverage or modify HOPG, the copied MMS/uniaxial programs, tracked inputs, material data, or numerical settings.

### Unified production-linked verification: 2026-11-19

At Git revision `964c60474f241056824e9aa5240f70b5fb4ce96b` with the then-current uncommitted source changes, the authoritative command was:

```bash
./scripts/verify.sh check
```

Evidence: `verification-runs/20261119T135752.200296Z-964c60474f24-163993/`. All 18 required targets passed in 132.2 s of summed command wall time, with two build jobs, two solver threads, sequential tests, and an 8 GiB per-process limit. The largest-child RSS high-water mark was 1,321,104 KiB during compilation. No target was skipped and no protected file changed. The run contains `summary.json`, `junit.xml`, GoogleTest XML, exact inputs, logs, and numerical artifacts.

The gate included 14 production-linked GoogleTest cases, production-source temperature-dependent MMS, the reduced copied-source MMS and uniaxial suites, active resolved-grain damage, HOPG smoke, and seven direct analytical resolved-grain cases. The new MMS used 64, 512, and 4096 cells. L2 errors were `1.024236e-7`, `2.686982e-8`, and `6.805899e-9`, with successive rates 1.9305 and 1.9811; H1 errors were `1.579951e-6`, `7.888607e-7`, and `3.935805e-7`, with rates 1.0020 and 1.0031. SymPy generated the displacement, analytic gradients, spatial temperature, and forcing from the full temperature-dependent thermoelastic stress divergence.

The two-phase laminate used prescribed orthogonal grain orientations and independently solved strain compatibility, interface traction continuity, and zero average stress with SymPy. Its final mean-stress norm was `1.224e-5 Pa` using exact periodic condensation and the direct solver. Clamped, traction-free, periodic-uniform, and exact zero-CTE cases also passed. In the zero-CTE cases, all reported macroscopic strain components were zero to `1e-12`, mean-stress norms were below `1e-6 Pa`, and damage was exactly zero under periodic, free-surface, and clamped boundaries. The checker did not compare every nodal displacement to zero.

The damage-kernel regression covered the coupled-stress counterexample from the audit, for which the old frozen-stress partition missed both admissible roots. The implementation retained the old path when valid and used a coupled-residual bracketing fallback otherwise. Zero-norm convergence and positive-definiteness checks were added without changing previously valid baseline results.

The final nine-case A/B command was:

```bash
./scripts/verify.sh build-resolved ab-resolved \
  --reference-run verification-runs/20261118T112247.213937Z-8536acf704b1-84891
```

Evidence: `verification-runs/20261119T140136.363117Z-964c60474f24-166352/`. The build and all nine cases passed in 54.0 s; every case reported `max_scaled_csv_difference = 0`, convergence and macroscopic iteration histories matched, and all VTU fields matched except timestamps. Largest-child RSS was 290,132 KiB and protected files were unchanged. Doxygen generation and all 15 discovered CTest tests (14 unit plus production MMS) also passed.

A final combined invocation on the delivered driver,

```bash
./scripts/verify.sh check \
  --reference-run verification-runs/20261118T112247.213937Z-8536acf704b1-84891
```

passed all 18 required targets in 109.0 s with a 480,844 KiB largest-child RSS high-water mark; its JUnit report contains 31 tests with zero failures or skips. Evidence: `verification-runs/20261119T140812.883855Z-964c60474f24-170195/`. Existing uniform/damage A/B comparisons again had zero scaled CSV difference, while new analytical cases correctly reported that no historical A/B counterpart was available.

This remains reduced-scale verification. The 64,000-cell copied-source MMS level, 80/160-step uniaxial cases, and large production studies remain explicitly deferred. The copied-source MMS/uniaxial tests do not replace the production-linked tests; experimental validation, general nonuniform-damage analytical solutions, and full HOPG migration remain outside this gate.

### Archived copied-source MMS convergence

At the final temperature step, the archived `mmsSandbox/mms_convergence_data.json` contains:

| Cells | L2 displacement error | H1 seminorm error | L-infinity error |
|---:|---:|---:|---:|
| 1,000 | 1.23566e-6 | 2.70120e-5 | 2.79240e-6 |
| 8,000 | 3.09003e-7 | 1.33240e-5 | 7.14046e-7 |
| 64,000 | 7.71554e-8 | 6.63367e-6 | 1.79742e-7 |

Observed rates across the two successive refinements are:

| Norm | 1,000 to 8,000 | 8,000 to 64,000 | Expected Q1 trend |
|---|---:|---:|---:|
| L2 | 2.000 | 2.002 | 2 |
| H1 seminorm | 1.020 | 1.006 | 1 |
| L-infinity | 1.967 | 1.990 | Approximately 2 here |

These results support correctness of the archived assembled stiffness and thermal-load terms for the manufactured, temperature-independent problem. They do not verify current temperature-dependent elastic/CTE functions, damage, periodic homogenization, random orientations, or the HOPG fork. The reduced regression target compares against the 1,000/8,000-cell values; see [verification.md](verification.md#other-targets-in-the-unified-gate) for current acceptance criteria.

### Archived uniaxial damage convergence

The independent reference gives final damage `0.6904145198`. Archived finite-element results are:

| Pseudo-time step | Final damage | Absolute final error | Observed order from previous row |
|---:|---:|---:|---:|
| 0.1 | 0.687177 | 3.23752e-3 | — |
| 0.05 | 0.688889 | 1.52552e-3 | 1.086 |
| 0.025 | 0.689688 | 7.26520e-4 | 1.070 |
| 0.0125 | 0.690062 | 3.52520e-4 | 1.043 |
| 0.00625 | 0.690241 | 1.73520e-4 | 1.023 |

The least-squares observed order over all five step sizes is approximately 1.056, consistent with backward Euler. The largest archived spatial damage spread is below `1e-11`. The largest absolute algebraic stress mismatch is approximately `3.61e-3 Pa`, about `1.8e-11` times the 200 MPa threshold, using the test's tight linear-solver tolerance.

These results support the archived damage update and stress–damage coupling in a self-limiting, displacement-controlled homogeneous test. They do not verify load-controlled response, general multiaxial damage, shear-driven mechanisms, heating/crack closure, nonuniform damage fields, or physical calibration. The reduced current target uses only the first three step sizes; see [verification.md](verification.md#other-targets-in-the-unified-gate).

## Latest HOPG mosaic-spread study

Primary report: `hopgCooling/results/mosaic-spread/summary.md`

Configuration and scope:

- archived at HEAD `67dd766`, 2026-10-12;
- 1,000 crystallites, 64 elements per crystallite, 64,000 cells;
- periodic boundary conditions;
- cooling from 2500 to 1000 K;
- Polewczyk temperature-dependent elastic fits and Marsden secant CTE fits referenced to 2500 K;
- damage disabled;
- mosaic component standard deviations 0.25, 0.5, and 1 degree;
- one shared orientation seed, so the tilt realization is scaled rather than independently resampled.

At 1000 K, the archived RMS-stress-versus-sigma log-log exponents are:

| Stress measure | Fitted exponent |
|---|---:|
| Basal shear `tauBasal` | 0.999 |
| Local `sigma33` | 1.135 |
| Maximum basal-plane principal stress | 1.861 |
| Minimum basal-plane principal stress | 1.883 |
| von Mises stress | 1.833 |

These support approximately linear basal-shear scaling and near-quadratic scaling of the basal-plane principal and von Mises RMS stresses for this small-tilt, shared-realization study. Local `sigma33` grew more weakly, with exponent 1.135. None of these are universal HOPG laws: realization scatter, mesh sensitivity, larger tilts, damage, and experimental calibration remain separate questions.

The full report contains percentile/extrema tables and links to the archived per-run inputs, logs, and CSV data. `hopgCooling/README.md` supplies additional scientific interpretation and limitations.

## Resolved-grain numerical and performance evidence

### Frozen-`Cbar` quasi-Newton

Report: `resolvedGrainCooling/results/quasi-newton-verification.md`

In the dated nine-run comparison, exact and frozen-`Cbar` paths agreed to a reported worst output disagreement of `6.9e-7`, against a damage fixed-point tolerance of `1e-5`. The largest reported case showed a 7.8-times speedup at 64 elements per grain, with fewer linear solves and one `Cbar` rebuild in that run.

This is configuration-specific performance evidence, not a guaranteed scaling law. The reported `Cbar` can be the frozen value; rebuild it before quoting converged homogenized stiffness.

### Aitken acceleration

Report: `resolvedGrainCooling/results/aitken-study/README.md`

The archived high-damage case reported approximately 2.05-times fewer solves and 2.16-times lower wall time, with nearly identical converged fields. The study found the benefit grew as accumulated damage made the fixed point dominate cost. This does not establish the same benefit in low-damage or damage-free HOPG runs.

### HOPG solver selection

`hopgCooling/configure_input.py` and `hopgCooling/README.md` record short measured comparisons that motivated AMG and exact seven-solve condensation for the near-single-crystal HOPG case. Frozen-`Cbar` iteration performed poorly there because temperature-dependent stiffness changed enough between steps to trigger rebuilds. Do not generalize this ranking to the resolved random polycrystal, where the tracked baseline uses CG and frozen-`Cbar` quasi-Newton.

### Boundary, resolution, seed, and CTE-coupling studies

- `resolvedGrainCooling/results/README.md` and the sweep subfolders contain boundary-condition, resolution, and realization studies.
- `resolvedGrainCooling/results/cte-coupling-README.md` examines damage coupling to `alpha_33` and its weak homogenized-CTE response in those configurations.
- `resolvedGrainCooling/plot_cte_comparison.py` plots that comparison.

Several headline results predate the current elastic fits or the present tracked CTE baseline. Earlier claims about damage localization and effective CTE were explicitly corrected in the reports; do not resurrect the superseded interpretations.

## Current documentation-status index

| Existing document | How to use it now |
|---|---|
| `hopgCooling/README.md` | Valuable detailed companion to the 2026-10-12 HOPG study. Use with its configuration and limitations; root `AGENTS.md`/`docs/` govern current-state conventions. |
| `resolvedGrainCooling/README.md` | Useful solver, grain/mesh, and uniform-orientation notes. Statements that later features are inherited unchanged are historically scoped. |
| `polycrystalCooling/README.md` | Early model rationale and project history. Contains older material data, performance estimates, and plans; not current operational guidance. |
| `resolvedGrainCooling/homogenization.md` and `polycrystalCooling/homogenization.md` | Byte-identical detailed periodic derivation. Retain as mathematical background. Their recovery section uses stale local `sigma_zz` notation; current convention is local `sigma_33`. Historical temperature-independent statements are not current baseline descriptions. |
| `resolvedGrainCooling/optimization_review.md` | Dated 2026-09-23 design review. Its frozen-`Cbar` recommendation has since been implemented and verified; it is not a pending-task list. |
| `resolvedGrainCooling/scaling_options.md` | Long-term performance/MPI/matrix-free background. Some projections and recommendations were superseded by the optimization review and later quasi-Newton work. |
| `resolvedGrainCooling/thermodynamic_potential.md` | Useful constitutive-potential derivation and discussion of dissipation/crack closure. Section 7's temperature-independent description is outdated for current elastic data and HOPG CTE; archived consistency observations are not universal proofs. |
| `resolvedGrainCooling/cte_integrated_thermal_strain.md` | Parked proposal dated 2026-10-02. Current code still uses total secant thermal strain; do not describe integrated thermal strain as implemented. |
| `materialData/*/README.md` | Primary detailed provenance companions, summarized and corrected for current baseline usage in [material-data.md](material-data.md). |
| `polycrystalCooling/results/2026-09-11-baseline/README.md` | Historical baseline using older/placeholder CTE data. |

## Conclusions that should not be drawn from the archive

- Adequate-looking residual-stress fields are not experimental validation.
- Mesh convergence in MMS does not verify damage or temperature-dependent production data.
- One orientation seed does not quantify realization uncertainty.
- One solver timing does not establish performance on another boundary condition, anisotropy regime, mesh, or damage state.
- Damage growth under the assumed kinetic law does not establish physical Mrozowski-crack evolution.
- Effective-property trends from older constant/placeholder CTE runs are not automatically current-model results.

## Research and engineering priorities requiring explicit selection

The documentation audit does not assign these priorities itself:

1. Define the observables by which crystallite simulations will formalize a mechanical graphite domain.
2. Decide the resolved-grain CTE baseline and complete its provenance.
3. Calibrate or bound damage parameters and define validation data.
4. Establish a shared regression architecture so verification executes current production code rather than drifting copies.
5. After behavior-preserving cleanup, consider deliberate deduplication of the resolved-grain/HOPG solver lineage.
6. Revisit integrated thermal strain, crack closure, shear/inelastic mechanisms, and geometric realism only as separately approved scientific developments.
