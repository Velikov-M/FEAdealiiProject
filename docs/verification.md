# Verification status and regression expectations

## Status at a glance

The repository has separate PDE and damage-kinetics verification programs plus an explicit resource-limited regression runner. Their coverage is complementary; there is no single test suite that validates every production configuration.

- `mmsSandbox/` verifies a separate zero-damage, temperature-independent thermoelastic implementation against a manufactured solution.
- `uniaxialDamageTest/` verifies a separate homogeneous displacement-controlled damage implementation against an independent scalar ODE solution and an algebraic stress relation.
- Additional resolved-grain checks and solver comparisons are documented in dated README/results files.
- HOPG mosaic studies provide physical/code evidence for a particular configuration, not general validation against experiments.
- No compilation or new run was performed during the documentation audit on 2026-11-16 because the environment lacked CMake, Docker, and deal.II. Dated fresh local runs are identified separately from archived results below.

Because the verification programs and production programs are separate source copies, a passing archived test does not automatically cover later changes in the resolved-grain modules or `hopgCooling/main.cpp`.

## Fresh local bootstrap: 2026-11-17

This is new, dated local evidence, separate from the archived results below. The final unified gate passed at Git revision `8536acf704b18734fb5032841e38bdfe1cf160b9` with the new uncommitted workflow files and the pre-existing uncommitted `AGENTS.md` session-protocol edit. No C++ source, tracked input, or archived result was changed.

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

## Current-material active-damage baseline: 2026-11-18

The pre-refactor resolved-grain baseline is reproducible with:

```bash
./scripts/verify.sh damage-resolved
```

Evidence: `verification-runs/20261118T093735.134221Z-8536acf704b1-29756/`, at Git revision `8536acf704b18734fb5032841e38bdfe1cf160b9` with uncommitted workflow changes. The isolated input preserves the tracked periodic 1,000-cell configuration, seed, current Polewczyk elastic fits, constant secant CTEs, solver settings, and ten 20 K steps from 2000 to 1800 K. Its only scientific change is `StressThreshold = 60 MPa`, approved to activate damage with the current materials; the tracked input remains at 200 MPa.

The target passed in 12.7 s using two threads, with a 225,088 KiB largest-child RSS high-water mark. All ten steps converged without warnings, all ten PVD fields and the macroscopic CSV were consistent, no protected files changed, and damage was finite and monotone. Damage first appeared at 1860 K; at 1800 K the mean and maximum damage were 0.117217 and 0.166286. The 60 MPa threshold is a numerical baseline choice, not a calibration or experimental validation.

An initial otherwise-identical 200 MPa run also completed all ten steps but had zero damage, with maximum local `sigma_33 = 88.98 MPa`; its evidence is retained in `verification-runs/20261118T093504.373533Z-8536acf704b1-28404/`. The archived active-damage run under `resolvedGrainCooling/results/bc-resolution-sweep-cte-qn/periodic_01cpg_seed20260905/` used both constant CTEs and older constant elasticity, so it is historical context rather than a numerical regression target for this new baseline. The run manifest, exact input, commands, logs, and numerical artifacts hold the detailed evidence; this section is the compact current-state index.

## Resolved-grain modular refactor A/B: 2026-11-18

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

## MMS: zero-damage thermoelastic PDE

### What it exercises

`mmsSandbox/main.cpp` prescribes a smooth displacement, derives a body force from the corresponding manufactured stress, and applies full Dirichlet boundary data. The manufactured temperature varies in space so that the thermal term contributes to the interior PDE residual. The displacement amplitude is deliberately small enough that elastic and thermal contributions are both visible in the error.

The archived input has:

- three mesh levels: 1,000, 8,000, and 64,000 cells;
- temperature-independent elastic constants and CTE;
- zero damage;
- ten temperature steps per mesh level;
- a spatially varying temperature field in the manufactured thermal strain.

The body force was derived with SymPy in `mmsSandbox/myNotebook.ipynb` and cross-checked numerically. The executable explicitly rejects temperature-dependent material functions because the manufactured body force does not contain the required additional terms.

### Archived results

At the final temperature step, `mms_convergence_data.json` contains:

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

This supports correctness of the assembled stiffness and thermal-load terms for the manufactured, temperature-independent problem. It does **not** verify current temperature-dependent elastic/CTE functions, damage, periodic homogenization, random orientations, or the HOPG fork.

### Reproduction

```bash
cd mmsSandbox
cmake -S . -B build -DDEAL_II_DIR=/path/to/deal.II
cmake --build build --parallel 4
python3 plot_mms_verification.py --run
```

The driver copies the built executable and required assets to a temporary directory, runs the configured sweep, writes `mms_convergence_data.json`, and generates the verification plots. Without `--run`, it only plots the archived JSON. Use a timeout/resource limit when experimenting; the archived three-level sweep is not a minimal smoke test.

## Uniaxial displacement-controlled damage test

### What it exercises

`uniaxialDamageTest/main.cpp` uses a homogeneous cube with zero CTE and axial displacement control. The temperature-step variable is dimensionless pseudo-time. In the homogeneous setting, the algebraic axial stress relation is

\[
\sigma_{zz}=\varepsilon_{zz}
\left[C_{33}^0(1-\omega)-\frac{2(C_{13}^0)^2}{C_{11}^0+C_{12}^0}\right].
\]

Here `zz` is appropriate because the test's loading axis and crystal c-axis coincide in the reduced problem; this does not change the production local/global naming rule.

`plot_verification.py` independently integrates the reduced scalar damage ODE with SciPy `solve_ivp` (`rtol=1e-12`, `atol=1e-14`). It compares the finite-element backward-Euler trajectories at step sizes 0.1, 0.05, 0.025, 0.0125, and 0.00625. It also checks stress against the algebraic relation at the same damage and checks spatial uniformity.

### Archived results

The independent reference gives final damage `0.6904145198`. Archived finite-element results are:

| Pseudo-time step | Final damage | Absolute final error | Observed order from previous row |
|---:|---:|---:|---:|
| 0.1 | 0.687177 | 3.23752e-3 | — |
| 0.05 | 0.688889 | 1.52552e-3 | 1.086 |
| 0.025 | 0.689688 | 7.26520e-4 | 1.070 |
| 0.0125 | 0.690062 | 3.52520e-4 | 1.043 |
| 0.00625 | 0.690241 | 1.73520e-4 | 1.023 |

The least-squares observed order over all five step sizes is approximately 1.056, consistent with backward Euler. The largest archived spatial damage spread is below `1e-11`. The largest absolute algebraic stress mismatch is approximately `3.61e-3 Pa`, about `1.8e-11` times the 200 MPa threshold, using the test's tight linear-solver tolerance.

This supports the damage update and stress–damage coupling in a self-limiting, displacement-controlled homogeneous test. It does not verify load-controlled response, general multiaxial damage, shear-driven mechanisms, heating/crack closure, nonuniform damage fields, or physical calibration.

### Reproduction

```bash
cd uniaxialDamageTest
cmake -S . -B build -DDEAL_II_DIR=/path/to/deal.II
cmake --build build --parallel 4
python3 plot_verification.py --run
```

Without `--run`, the script reconstructs the reference and plots archived `convergence_data.json`. The plotting/reference scripts require NumPy, SciPy, and Matplotlib.

## Other archived checks

The following are useful but not replacements for a shared automated regression suite:

- `resolvedGrainCooling/README.md` documents uniform-orientation checks, including the zero-stress periodic thermoelastic case and expected homogeneous response.
- `resolvedGrainCooling/results/quasi-newton-verification.md` compares exact and frozen-`Cbar` paths over the dated sweep and reports worst disagreement `6.9e-7` in the compared outputs.
- `resolvedGrainCooling/results/aitken-study/README.md` compares accelerated and unaccelerated fixed-point solutions in a high-damage case.
- `hopgCooling/README.md` and `hopgCooling/results/mosaic-spread/summary.md` document orientation sampling and stress-statistics behavior for the 2026-10-12 HOPG configuration.

Read each with its archived input and date. Solver agreement is verification of numerical equivalence, not experimental validation.

## Minimum regression gate after a refactor

Adapt the gate to the files changed, and record actual commands, revisions, and outcomes:

1. Build every affected executable with deal.II 9.8.0 and required solver features.
2. Rerun MMS and uniaxial verification when their copied source or shared algorithms are affected; remember that these runs alone do not cover the production forks.
3. Run small resolved-grain and HOPG cases with one element per crystallite and a shortened temperature interval.
4. Compare serial versus WorkStream assembly to round-off after assembly changes.
5. For periodic changes, compare against exact condensation; test uniform orientation and zero/known analytical response where applicable.
6. For damage changes, compare a complete converged history, not only the first step or a single norm.
7. For solver acceleration changes, compare exact and accelerated outputs and inspect final convergence/warnings.
8. Confirm each run reached `FinalTemperature`, wrote the expected CSV files, and did not mix stale field files.
9. Archive the exact run input and compare outputs at tolerances appropriate to the solver; do not demand bitwise identity after legitimate floating-point reordering, but do not dismiss systematic drift as round-off.

A zero exit code alone is not a successful scientific baseline.

## Missing verification capabilities

The following should remain visible as limitations rather than implied coverage:

- no shared-library tests linked directly against both production forks;
- no automated MMS for temperature-dependent constitutive functions;
- no general nonuniform-damage analytical solution;
- no comprehensive periodic-homogenization regression against an independent implementation;
- no experimental validation of residual stress, damage evolution, or HOPG scaling laws;
- no calibrated uncertainty bounds for digitized material inputs and damage parameters.
