# Current crystallite model

## Scope

This document describes the formulation implemented in `resolvedGrainCooling/main.cpp` together with `resolvedGrainCooling/include/resolved_grain/` and `resolvedGrainCooling/src/`, and in the independent `hopgCooling/main.cpp` fork. It is not a claim that the idealizations are sufficient for nuclear graphite generally. The model is a three-dimensional, small-strain, displacement-based finite-element model with local scalar damage.

## Kinematics and constitutive law

The total small strain is

\[
\varepsilon(\mathbf u)=\operatorname{sym}\nabla \mathbf u,
\]

and, at prescribed uniform temperature \(T\), the local-frame elastic strain and stress are

\[
\varepsilon^e=\varepsilon-\alpha^{\mathrm{sec}}(T,\omega)(T-T_0),
\qquad
\sigma=\mathsf C(T,\omega):\varepsilon^e .
\]

\(T_0=\) `InitialTemperature` is both the CTE reference and the assumed stress-free temperature. There is no body force in the production cooling models. The same temperature-dependent constitutive functions are evaluated at quadrature points from the current scalar step temperature.

In the local crystal frame, axis 3 is the c-axis. Undamaged transverse isotropy is parameterized by \(C_{11},C_{12},C_{13},C_{33},C_{44}\), with

\[
C_{66}=(C_{11}-C_{12})/2.
\]

Damage changes only two entries:

\[
C_{33}(\omega,T)=(1-\omega)C_{33}^{0}(T),
\qquad
\alpha_{33}^{\mathrm{sec}}(\omega,T)=(1-\omega)\alpha_{33}^{0,\mathrm{sec}}(T).
\]

All other elastic constants and the basal CTE \(\alpha_{11}=\alpha_{22}\) are not directly degraded. This is an idealized oriented Mrozowski-crack representation, not a resolved crack model.

The implementation converts the local Voigt constants to a Kelvin matrix before constructing the fourth-order tensor. The shear diagonal is \(2C_{44},2C_{44},2C_{66}\). `checkTransverseIsotropy()` checks both component consistency and rotational invariance about the c-axis.

### Secant versus instantaneous CTE

The constitutive code expects secant CTE:

\[
\alpha^{\mathrm{sec}}(T;T_0)
=\frac{1}{T-T_0}\int_{T_0}^{T}\alpha^{\mathrm{inst}}(\theta)\,d\theta .
\]

Thus `alpha33_functionOfTemperature` and `alpha11_functionOfTemperature` describe a normalized **secant** multiplier when fitted CTE data are used. They must be regenerated/re-referenced if \(T_0\) changes. `hopgCooling/configure_input.py` enforces this for its Marsden data. The tracked resolved-grain baseline instead uses constant CTEs.

With no damage and fixed secant data, the thermoelastic state at a given \(T\) is path-independent; temperature steps then control output resolution. With damage, the backward-Euler kinetic update makes step size part of the numerical approximation.

## Damage kinetics

Each quadrature point stores scalar \(\omega\). At a cooling step of positive magnitude \(\Delta T_c\), the implicit backward-Euler residual is

\[
R(\omega)=
\omega-\omega_{old}
-A\Delta T_c
\left[
\frac{\big(\sigma_{33}(\omega)-\sigma_{th}\big)_+}
{\sigma_{th}(1-\omega)}
\right]^m=0,
\qquad (x)_+=\max(x,0).
\]

The stress is reevaluated at the trial damage, so stiffness degradation and stress relaxation are included in the local root problem. The root search uses `[old_damage, 0.999]`. It retains the historical analytic partition when that partition brackets a valid root and otherwise scans the actual coupled residual for the first sign-changing bracket before calling Boost `toms748`. The analytic partition assumes stress frozen at old damage; the fallback is required because the true stress depends on trial damage. The scan is a numerical bracketing procedure, not a proof that arbitrary new kinetic laws have a unique root.

Consequences of the implemented law:

- only local tensile `sigma_33` above `StressThreshold` activates damage;
- shear, basal slip, and other inelastic mechanisms do not drive damage;
- damage does not heal;
- stiffness loss is bilateral: there is no crack-closure or tension/compression distinction;
- the production model rejects heating rather than treating it as a sign reversal;
- approaching `omega = 1` makes the kinetic term singular and the root bracket narrow; the warning threshold is not a validity proof;
- with only `C33` degraded, positive definiteness additionally requires `(1-omega)*C33^0*(C11+C12) > 2*C13^2` and positive basal/shear eigenvalue conditions. The implementation rejects states outside this range instead of applying CG to an indefinite matrix.

The elastic constitutive structure admits the potential discussed in `resolvedGrainCooling/thermodynamic_potential.md`, but the stress-driven kinetic law is an additional non-potential-derived assumption. That report is not a general proof of thermodynamic consistency for every input or state.

## Coupling and time advancement

The displacement and damage problems are solved in a staggered fixed-point iteration at each temperature step. Convergence uses displacement increment, damage increment, and equilibrium-residual criteria. Optional Aitken/Irons–Tuck relaxation accelerates damage iterates and is clamped to admissible values.

The resolved-grain tracked baseline enables damage. HOPG adds `DamageParameters/DamageEnabled`; its tracked baseline sets it to `false`, making damage identically zero and requiring one mechanical pass per temperature step.

Refinement cycles are independent runs: each cycle restarts the complete temperature history and resets damage. There is no mid-simulation damage transfer. `InitialGlobalRefinements` is applied before a run; this is distinct from `NumberOfRefinementCycles`.

## Geometry, grains, and orientations

The archived mesh is a 10 by 10 by 10 base grid of cubic cells. Grain identity is assigned by cell centroid on a `GrainsPerSide^3` lattice, independently of subsequent mesh refinement. With the tracked `GrainsPerSide = 10`, there are 1,000 crystallites.

| `InitialGlobalRefinements` | Elements per grain | Total cells on the tracked geometry |
|---:|---:|---:|
| 0 | 1 | 1,000 |
| 1 | 8 | 8,000 |
| 2 | 64 | 64,000 |

The elements are three-dimensional vector Q1 elements with `QGauss(fe.degree+1)`, hence eight quadrature points per cell.

For `resolvedGrainCooling`, orientations are sampled uniformly on the sphere unless `UniformOrientation` is selected. Setting `UseOrientationFile` instead reads one explicit global-to-local rotation per grain from `InputFiles/OrientationFile`; this is used for reproducible microstructures and the analytical two-phase laminate. HOPG adds `OrientationDistribution = mosaicGaussian`: two independent zero-mean Gaussian tilt components rotate the c-axis away from global z. `MosaicSpreadDeg` is the standard deviation of **each one-dimensional component**, not the FWHM. For a one-dimensional Gaussian, FWHM is approximately `2.355*sigma`.

Orientation matrices map global to local coordinates. Their rows are the local basis vectors; row index 2 is the c-axis. Reports of local normal stress must therefore use `sigma_33`, not `sigma_zz`.

## Boundary conditions and periodic homogenization

The resolved-grain program supports `zeroDisplacement`, `freeSurface`, and `periodic`. Both current tracked baselines use periodic boundary conditions.

The periodic displacement decomposition is

\[
\mathbf u=\bar{\mathsf E}\,\mathbf x+\widetilde{\mathbf u},
\qquad \widetilde{\mathbf u}\ \text{periodic},
\]

with unknown macroscopic strain \(\bar{\mathsf E}\) chosen to enforce zero volume-average stress:

\[
\langle\sigma\rangle=0.
\]

At frozen damage and temperature, the exact static-condensation path uses one thermal/right-hand-side solve and six unit-macroscopic-strain solves against the same matrix. It constructs \(\bar{\mathsf C}\) from quadrature averages of the six stress responses and obtains the macroscopic strain from the zero-average-stress condition. Effective CTE follows from the resulting macroscopic strain history.

`FrozenCbarQuasiNewton` instead reuses a frozen \(\bar{\mathsf C}\) only for the macroscopic update direction. Each pass still uses the current mechanical matrix and exact quadrature residual. If convergence stalls, \(\bar{\mathsf C}\) is rebuilt and the step retried. A reported frozen \(\bar{\mathsf C}\) is not necessarily the converged instantaneous homogenized stiffness; rebuild it before quoting that quantity as a result.

## HOPG stress statistics

HOPG adds volume-weighted statistics for rotation-invariant local measures:

- local `sigma_33`;
- maximum and minimum basal-plane principal stresses;
- basal shear magnitude `tauBasal = hypot(sigma_13, sigma_23)`;
- von Mises stress.

The CSV files contain means, RMS values, percentiles, and extrema. These statistics avoid dependence on arbitrary rotations of the basal 1–2 axes.

## Deliberate exclusions and validity limits

- Cubic grains/crystallites and ideal interfaces are geometrical idealizations; no real pore, domain, binder/filler, or QI-particle morphology is represented.
- No irradiation or chemical mechanisms are modeled.
- Small strain is questionable when thermal eigenstrain becomes large. The HOPG report specifically cautions about approximately 5% c-axis contraction over its full cooling interval.
- HOPG's assumed stress-free 2500 K state is an initial-condition hypothesis, not an established stress-free history for real material.
- Linear elasticity plus this scalar damage law does not represent crack closure, frictional contact, basal slip, or general inelasticity.
- Material inputs are fitted/digitized model data with limited temperature ranges; see [material-data.md](material-data.md).
- The crystallite model alone does not yet constitute a mechanical definition of a graphite domain. The observables and criteria for that step remain a research decision.
