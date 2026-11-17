# Study evidence, performance, and documentation status

## How to use archived studies

Files under `results/` and most folder READMEs are dated study reports. Their values apply only to the archived input, material data, code revision, seed, mesh, and solver path. Preserve them as scientific evidence; do not rewrite them into timeless model properties.

The curated current-state documents describe present code. A study report can remain scientifically useful even where its future-work recommendations are obsolete.

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
