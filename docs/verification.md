# Verification strategy and regression expectations

## Purpose and current coverage

This document defines what to verify, how each test works, its acceptance criteria, and the limits of the resulting evidence. Dated run outcomes, software environments, timings, memory measurements, and archived numerical tables belong in [studies.md](studies.md#verification-run-history). Raw commands, inputs, logs, and outputs are held in each `verification-runs/<run-id>/` directory. Do not append new run narratives to this guide.

The repository has separate PDE and damage-kinetics verification programs plus an explicit resource-limited regression runner. Their coverage is complementary; there is no single test suite that validates every production configuration.

- `./scripts/verify.sh check` is the unified required gate. It includes production-linked unit and MMS tests, seven analytical resolved-grain cases, reduced copied-source MMS/uniaxial suites, an active-damage resolved-grain history, and HOPG smoke coverage. The exact setups and pass criteria are listed in [Implemented tests](#implemented-tests-setup-oracle-and-pass-criteria).
- `mmsSandbox/` verifies a separate zero-damage, temperature-independent thermoelastic implementation against a manufactured solution.
- `uniaxialDamageTest/` verifies a separate homogeneous displacement-controlled damage implementation against an independent scalar ODE solution and an algebraic stress relation.
- Additional resolved-grain checks, solver comparisons, and physical studies are indexed in [studies.md](studies.md).
- HOPG mosaic studies provide physical/code evidence for a particular configuration, not general validation against experiments.

Because the copied verification programs and production programs are separate source copies, a passing copied-source test does not automatically cover later changes in the resolved-grain modules or `hopgCooling/main.cpp`. Production-linked tests have their own explicitly stated limits below.

The latest recorded full gate passed all 18 required targets on 2026-11-19. See the [dated summary and source revision](studies.md#unified-production-linked-verification-2026-11-19) and `verification-runs/20261119T140812.883855Z-964c60474f24-170195/`. This is historical evidence for that recorded source state, not a substitute for rerunning `check` after new changes.

## Implemented tests: setup, oracle, and pass criteria

This section is the test-by-test reference for the current suite. Run all required targets with `./scripts/verify.sh check`, or select a subset such as `unit-resolved`, `production-mms-resolved`, or `analytical-resolved`. The executable exit status is only one part of a pass: the runner also applies the numerical and artifact checks below.

There are three deliberately different kinds of evidence:

- **Unit and analytical tests** compare against mathematical identities or independently constructed references.
- **MMS convergence tests** compare against a manufactured exact solution and require the expected refinement rates.
- **A/B tests** compare against accepted outputs from an earlier executable. They detect behavior changes but are not independent proof that the original equations were correct.

### Production-linked GoogleTest unit tests

`unit-resolved` runs `resolvedGrainCooling/build/resolved_grain_tests`. The executable links the same `resolved_grain` static library as production. The runner requires at least one test and rejects every failure or skipped test; the current suite contains exactly 14 cases.

The independent symbolic constants are generated at build time by `resolvedGrainCooling/tests/symbolic/generate_oracles.py`. SymPy uses rational input values and direct index notation rather than calling the C++ material or damage implementations. The generated header is placed under the build directory and is not a tracked source file.

**Material response and stability (3 cases):**

- `Material.ThermoelasticStressMatchesIndependentSymPyIndexNotation` evaluates all six independent stress components at 1800 K, damage 0.2, and a nonspecial symmetric strain containing all normal and shear components. The reference independently forms the secant thermal strain and transversely isotropic stress, including axial damage coupling. Every component must agree within `2e-13 * max(1, |reference|)`.
- `Material.KelvinTensorPassesPhysicalComponentAndSymmetryChecks` checks the transverse-isotropy identities and symmetries, `C_2323 = C44`, and `C_1212 = (C11-C12)/2`. This guards the rank-four tensor components used to construct the Kelvin matrix, whose shear diagonal is `2*C44`, `2*C44`, and `C11-C12`.
- `Material.PositiveDefinitenessLimitMatchesSymbolicDeterminant` compares the computed critical damage with the independent expression `1 - 2*C13^2/(C33*(C11+C12))` to `2e-15`. It also requires the baseline material to be positive definite at damage 0 and 0.9, not positive definite at 0.999, and to throw from the explicit physical-validity check at 0.999.

**Standalone backward-Euler damage law (5 cases):**

- `DamageLaw.CoupledStressCounterexampleFindsSmallerRoot` uses a stress that changes with damage, `sigma(omega) = 1.3 - 0.2*omega` in nondimensional units. SymPy converts the coupled residual to a polynomial and finds two admissible roots. The solver must return the smaller root to `2e-12`, return a value below the second root, and satisfy the original nonlinear residual to `1e-11`. This specifically covers the case missed by the old frozen-stress partition.
- `DamageLaw.ConstantStressLinearExponentMatchesSymbolicRoot` uses old damage 0.1, constant stress 3, threshold 2, prefactor 0.2, exponent `m=1`, and unit time step. The result must match the smaller independent symbolic root to `2e-12`.
- `DamageLaw.SubcriticalAndCompressiveStressDoNotEvolveDamage` requires exactly unchanged damage for stress equal to the threshold and for compressive stress.
- `DamageLaw.ZeroPrefactorIsAnExactNoEvolutionCase` requires exactly unchanged damage when the kinetic prefactor is zero, even above threshold.
- `DamageLaw.RejectsInvalidParametersAndUnbracketedSteps` requires exceptions for a zero time step, a zero kinetic exponent, and a very large step/prefactor for which no admissible root is bracketed. A failed root solve must not silently return a damage value.

**Orientation construction and sampling (2 cases):**

- `Orientations.BasisIsOrthonormalIncludingCoordinateAxes` tests the three coordinate axes, the negative z-axis, and a generic unit c-axis. For each, it requires `R*R^T = I`, `det(R) = 1`, and equality of the third matrix row with the requested global c-axis. The coordinate-axis cases exercise the basis-construction degeneracies.
- `Orientations.SamplingIsReproducibleAndReturnsUnitAxes` generates 32 orientations from each of two `mt19937_64` generators seeded with 12345. Corresponding matrices must be bit-for-bit equal, every sampled c-axis must have unit norm, and every frame must pass the orthonormality/determinant checks. This tests reproducibility and valid frames, not the statistical uniformity of a large sample.

**Convergence scaling (4 cases):**

- `ConvergenceMeasure.AcceptsExactZeroResponse` requires the `0/0` case to return exactly zero, so an exact zero solution is converged rather than NaN.
- `ConvergenceMeasure.RejectsNonzeroIncrementAtZeroScale` requires a nonzero increment with zero scale to return infinity.
- `ConvergenceMeasure.UsesRelativeScaleAwayFromZero` requires increment 2 and scale 4 to return exactly 0.5.
- `ConvergenceMeasure.RejectsNonFiniteAndNegativeInputs` requires NaN increments and negative increments or scales to return infinity rather than be accepted as converged.

These are focused kernel tests. They do not by themselves test complete FE assembly, boundary conditions, coupled production iterations, or random-microstructure statistics.

### Production-source temperature-dependent MMS

`production-mms-resolved` runs the executable built from `resolvedGrainCooling/tests/integration/production_mms.cpp` by the `build-resolved` target. The executable links the production `resolved_grain` library and calls the same `assembleCell()` kernel as the cooling model. It does not use the copied `mmsSandbox` implementation.

On the unit cube, SymPy prescribes the displacement amplitude `A = 1e-6 m` and

\[
\begin{aligned}
u_x &= A\sin(\pi x)\sin(\pi y)\sin(\pi z),\\
u_y &= A\sin(\pi x)\sin(2\pi y)\sin(\pi z),\\
u_z &= A\sin(\pi x)\sin(\pi y)\sin(2\pi z),\\
T(x,y,z) &= 1900 + 20x + 10y + 5z\ \text{K}.
\end{aligned}
\]

The stress-free reference temperature is 2000 K. The test uses nonzero temperature dependence for every elastic constant and both secant CTE components, with zero damage. SymPy differentiates the displacement to obtain the strain, forms `sigma = C(T):(sym(grad(u)) - alpha_sec(T)*(T-2000))` in index notation, and generates the body force `b = -div(sigma)`. Thus the forcing includes derivatives of both the mechanical displacement stress and the spatially varying thermoelastic stress. Analytic displacement gradients are also supplied to the error calculation instead of relying on a parser's derivative approximation.

The numerical problem uses vector Q1 elements, full Dirichlet boundary data from the exact displacement, two-point Gauss quadrature per direction for assembly, and a direct UMFPACK solve. Errors are integrated with four-point Gauss quadrature per direction. Global refinements 2, 3, and 4 produce exactly 64, 512, and 4096 cells.

For each norm, the observed rate between successive meshes is `log2(error_coarse/error_fine)`. The runner requires:

- finite, strictly positive L2 and H1-seminorm errors at every level;
- both successive L2 rates in `[1.8, 2.2]`;
- both successive H1 rates in `[0.8, 1.2]`;
- exactly the three expected cell counts and a successful solver process.

Measured errors and rates from the latest full run are recorded in the [dated run summary](studies.md#unified-production-linked-verification-2026-11-19). This test verifies the shared cell stiffness, thermal load, body-force load, temperature-dependent constitutive evaluation, and Dirichlet solve for the manufactured problem. It does not exercise damage, periodic constraints, homogenization, random orientations, or the production cooling driver with a spatially varying temperature.

### Analytical tests through the production executable

`analytical-resolved` selects seven targets. Unless overridden below, each uses the tracked resolved-grain material and solver configuration, 1,000 cells with one element per crystallite, and two 20 K cooling steps from 2000 to 1960 K. These targets execute the production `main` and therefore cover mesh/grain setup, constraints, assembly, solve, homogenization where applicable, and output.

For every production analytical target, the runner first requires a warning-free/converged log, the expected cell count and two CSV rows, finite CSV values, the requested final temperature, damage in `[0, 0.999]` and monotone to `1e-12`, and existing nonempty fields referenced by the PVD. The target-specific checks are:

- **`uniform-resolved` — periodic homogeneous crystal.** Every grain has identity global-to-local rotation. The exact response is fluctuation-free thermal expansion with zero mean stress. At every temperature, the runner requires `Ebar_11 = Ebar_22 = alpha11_sec(T)*(T-T0)`, `Ebar_33 = alpha33_sec(T)*(T-T0)`, and zero macroscopic shear. Normal strains use relative tolerance `1e-8` and absolute tolerance `1e-11`; shear magnitudes must not exceed `1e-11`. It also checks `meanStressNorm <= 1e-3 Pa` and all six Kelvin homogenized-stiffness diagonal entries against `[C11, C11, C33, 2*C44, 2*C44, C11-C12]`, using relative tolerance `1e-7` and absolute tolerance `1e-3 Pa`.
- **`clamped-uniform-resolved` — zero-displacement homogeneous crystal.** With total strain zero, the independent diagonal stress reference is `sigma_11 = sigma_22 = -((C11+C12)*alpha11 + C13*alpha33)*(T-T0)` and `sigma_33 = -(2*C13*alpha11 + C33*alpha33)*(T-T0)`. All material values are evaluated at the row temperature. The reported mean-stress norm must equal `sqrt(2*sigma_11^2 + sigma_33^2)` with relative tolerance `1e-7` and absolute tolerance `1e-3 Pa`.
- **`free-uniform-resolved` — traction-free homogeneous crystal.** The production free-surface path is run with the direct solver. Its independent assertion is `meanStressNorm <= 1e-3 Pa` at both temperatures. The periodic uniform test above supplies the more detailed free-thermal-strain and stiffness assertions.
- **`laminate-resolved` — two-phase periodic laminate.** An orientation file prescribes equal-volume alternating layers normal to global x. Phase A has local axes `(x,y,z)`; phase B has local axes `(y,z,x)`, so their c-axes are orthogonal. Exact periodic condensation and the direct solver are used. For each temperature, SymPy independently imposes compatible layer-parallel strains `E_y` and `E_z`, zero normal traction in each phase, and zero volume-average `sigma_yy` and `sigma_zz`. It then computes all three macroscopic normal strains, including the average of the different layer-normal strains. The three `Ebar` values must agree with relative tolerance `1e-8` and absolute tolerance `1e-11`; macroscopic shear magnitudes must not exceed `1e-11`, and `meanStressNorm` must not exceed `1e-3 Pa`. This tests nontrivial orientation-dependent load transfer, not merely a homogeneous response.
- **`zero-response-periodic-resolved`, `zero-response-free-resolved`, and `zero-response-clamped-resolved` — exact zero eigenstrain.** Each sets both CTE reference values to zero while retaining the elastic problem and otherwise default random microstructure, then selects periodic, free-surface, or zero-displacement boundaries. For both rows, all six reported macroscopic strain magnitudes must be at most `1e-12`, `meanStressNorm` at most `1e-6 Pa`, and mean/max damage exactly zero. These are macroscopic output checks; they do not assert that every nodal displacement entry is bitwise zero.

The laminate reference is constructed directly in `scripts/verify.py` from the isolated run input, rather than from production homogenization routines. The homogeneous references likewise evaluate the input material expressions independently. A historical A/B counterpart is not required for any of these analytical assertions.

### Other targets in the unified gate

The following checks were already separate verification programs or smoke baselines, but are now also required by `check`:

| Target | What is exercised | Additional pass criteria |
|---|---|---|
| `mms-small` | The separate, zero-damage, temperature-independent `mmsSandbox` on 1,000 and 8,000 cells, with all ten temperature steps per mesh | Complete 20 rows; finite L2/H1/L-infinity errors; H1 rate in `[0.8,1.2]`, the other rates in `[1.8,2.2]`; final errors within 1% of archived values; nonempty plots and FE log |
| `uniaxial-small` | The separate homogeneous displacement-controlled damage program at pseudo-time steps 0.1, 0.05, and 0.025 | Independent SciPy `solve_ivp` ODE reference; final error no more than 1.01 times the archived error; fitted order in `[0.8,1.3]`; spatial spread at most `1e-9`; algebraic stress mismatch at most `1e-8 * StressThreshold`; complete logs and plots |
| `damage-resolved` | The current resolved-grain production executable for all ten steps from 2000 to 1800 K, 1,000 cells, current material fits, and an isolated 60 MPa threshold | Nonzero final mean and maximum damage, monotone bounded damage, ten finite macroscopic rows, ten PVD fields, and no convergence/warning markers; with `--reference-run`, strict A/B comparison also applies |
| `smoke-hopg` | The independent HOPG fork for two steps from 2500 to 2400 K on 1,000 cells | Complete finite macroscopic and stress-statistics CSVs, exactly zero damage, consistent nonempty PVD fields, and no convergence/warning markers; there is no independent analytical stress oracle for this smoke case |

The detailed copied-source MMS and uniaxial formulations are given in the following sections. Their reduced target limits remain explicit: the 64,000-cell copied-source MMS level and 80/160-step uniaxial cases are recorded as deferred, not silently treated as passed.

### Nine-case resolved-grain A/B regression

`ab-resolved` is complementary to the independent tests above. Its cases are `smoke-resolved` (default WorkStream), `serial-resolved`, `damage-resolved`, `damage-serial-resolved`, `exact-resolved` (exact periodic condensation), `zero-resolved` (clamped random microstructure), `free-resolved` (traction-free random microstructure), `uniform-resolved` (also has the analytical oracle above), and `refine-resolved` (8,000 cells). The two damage cases run the full ten-step history; the others use the two-step reduced interval.

With `--reference-run`, the comparator requires identical CSV schemas and row counts; exact `cycle`, `step`, and `cgIterations` columns; all other CSV values within relative tolerance `1e-10` and reference-scaled absolute tolerance `1e-12`; identical PVD field sequences; and byte-identical VTU content after normalizing only the deal.II generation timestamp. It also requires identical temperature/iteration, cell/DoF, CG-iteration, macroscopic-iteration, and `Cbar`-rebuild histories. Thus a zero exit code from the executable alone cannot hide changed numerical output.

Without a reference run, the intrinsic convergence, artifact, and any analytical checks still run, but no claim of historical equivalence is made. A/B equality does not replace the independent unit, MMS, or analytical oracles.

## Copied-source MMS: zero-damage thermoelastic PDE

### What it exercises

`mmsSandbox/main.cpp` prescribes a smooth displacement, derives a body force from the corresponding manufactured stress, and applies full Dirichlet boundary data. The manufactured temperature varies in space so that the thermal term contributes to the interior PDE residual. The displacement amplitude is deliberately small enough that elastic and thermal contributions are both visible in the error.

The archived input has:

- three mesh levels: 1,000, 8,000, and 64,000 cells;
- temperature-independent elastic constants and CTE;
- zero damage;
- ten temperature steps per mesh level;
- a spatially varying temperature field in the manufactured thermal strain.

The body force was derived with SymPy in `mmsSandbox/myNotebook.ipynb` and cross-checked numerically. The executable explicitly rejects temperature-dependent material functions because the manufactured body force does not contain the required additional terms.

### Interpretation and dated results

This test supports correctness of the assembled stiffness and thermal-load terms for the manufactured, temperature-independent problem. It does **not** verify current temperature-dependent elastic/CTE functions, damage, periodic homogenization, random orientations, or the HOPG fork. Archived errors and observed rates are recorded in [studies.md](studies.md#archived-copied-source-mms-convergence); current reduced-target tolerances are specified in [Other targets in the unified gate](#other-targets-in-the-unified-gate).

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

### Interpretation and dated results

This test supports the damage update and stress–damage coupling in a self-limiting, displacement-controlled homogeneous test. It does not verify load-controlled response, general multiaxial damage, shear-driven mechanisms, heating/crack closure, nonuniform damage fields, or physical calibration. Archived final damage, errors, and observed orders are recorded in [studies.md](studies.md#archived-uniaxial-damage-convergence); current reduced-target tolerances are specified in [Other targets in the unified gate](#other-targets-in-the-unified-gate).

### Reproduction

```bash
cd uniaxialDamageTest
cmake -S . -B build -DDEAL_II_DIR=/path/to/deal.II
cmake --build build --parallel 4
python3 plot_verification.py --run
```

Without `--run`, the script reconstructs the reference and plots archived `convergence_data.json`. The plotting/reference scripts require NumPy, SciPy, and Matplotlib.

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
- the copied-source `mmsSandbox` remains temperature-independent; the new production MMS covers the shared cell kernel and constitutive functions, not a spatially varying production cooling driver;
- no general nonuniform-damage analytical solution;
- periodic analytical coverage is limited to homogeneous and two-phase laminate configurations, not arbitrary microstructures;
- no experimental validation of residual stress, damage evolution, or HOPG scaling laws;
- no calibrated uncertainty bounds for digitized material inputs and damage parameters.
