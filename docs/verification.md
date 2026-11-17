# Verification status and regression expectations

## Status at a glance

The archive contains useful PDE and damage-kinetics verification evidence, but not a unified automated test suite for the current production sources.

- `mmsSandbox/` verifies a separate zero-damage, temperature-independent thermoelastic implementation against a manufactured solution.
- `uniaxialDamageTest/` verifies a separate homogeneous displacement-controlled damage implementation against an independent scalar ODE solution and an algebraic stress relation.
- Additional resolved-grain checks and solver comparisons are documented in dated README/results files.
- HOPG mosaic studies provide physical/code evidence for a particular configuration, not general validation against experiments.
- No compilation or new run was performed during the documentation audit on 2026-11-16 because the environment lacked CMake, Docker, and deal.II. All numerical values below come from archived files, not a fresh run.

Because the verification programs and production programs are separate source copies, a passing archived test does not automatically cover later changes in `resolvedGrainCooling/main.cpp` or `hopgCooling/main.cpp`.

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
