# Cooling model results, 2026-09-11

Two studies, both at 2000 -> 1800 K in ten steps, `periodic`/`freeSurface`/`zeroDisplacement`
boundary conditions, 1000 grains with random c-axis orientations uniform on the sphere.

CTE is `alpha_a = 1.2e-6`, `alpha_c = 32e-6`, read off the Steward et al. curve at 1800-2000 K.
**Read from a figure, not from source data** -- flagged as such in every `input.json`.

Field dumps (`*.vtu`, `*.pvd`) are gitignored: ~85 MB per 64-elements/grain run, and regenerable
from the `input.json` kept beside each one. Config, console log and macroscopic history are kept.

---

## Study 1: boundary condition x grain resolution (`bc-resolution-sweep/`)

One seed (20260905), 3 BCs x 3 resolutions.

### `<sigma>` confirms both free-cooling formulations

| `norm(<sigma>)` [Pa] | 1 elem/grain | 8 | 64 |
|---|---|---|---|
| `zeroDisplacement` | 5.2e8 | 4.7e8 | 4.4e8 |
| `freeSurface` | 1.11 | 0.70 | 0.47 |
| `periodic` | 0.036 | 0.115 | 0.016 |

Against local stresses of ~2e8 Pa, both free-cooling cases are zero to ~1e-9 relative at every
resolution. `zeroDisplacement`'s 4-5e8 Pa is the spurious clamping tension the formulation predicts.

### Max sigma_33 is pinned at ~207 MPa in all nine runs

Against a 200 MPa threshold. This is the damage feedback closing: growing omega lowers C33, which
lowers sigma_33, which slows further growth. Nine independent runs landing on the same ceiling is a
strong self-consistency check on the coupling.

### Damage localizes, and the MESH sets the band width

| `periodic` | 1 | 8 | 64 |
|---|---|---|---|
| **max** damage | 0.163 | 0.370 | 0.533 |
| **mean** damage | 0.0131 | 0.0069 | 0.0052 |
| volume fraction over threshold | 0.523 | 0.210 | 0.111 |

Max damage rises while mean damage falls: damage concentrates into a shrinking volume at growing
intensity. The over-threshold volume fraction **halves at each refinement** -- ratios 0.40 and 0.53
for `periodic`, 0.41 and 0.53 for `freeSurface` -- which is `~h`, the signature of a damage band
**one element thick**.

This is the classic pathology of a *local* damage model: with no length scale in the constitutive
law, the localization width is set by the discretization rather than the material. **Peak damage
and damaged volume will therefore not converge under refinement** -- they will keep rising and
shrinking. Regularization (nonlocal or gradient damage, or crack-band / fracture-energy scaling) is
what restores mesh objectivity.

Treat this as strong evidence, not proof. The confirming check is to open the `.pvd` in ParaView and
see whether the high-damage region is literally one element thick.

### One element per grain fails QUALITATIVELY, not just quantitatively

`min sigma_33` for `periodic`: **+150 -> +27 -> -109 MPa**.

At one element per grain the model shows **no compression anywhere in the RVE**. Compressive regions
only appear once grains are resolved by several elements. So the 1-element baseline cannot answer
the tension/compression question this study was posed around -- it is not merely inaccurate, it
cannot represent the phenomenon.

Resolution also matters *more* than boundary condition: 1 -> 64 elements/grain roughly doubles max
damage within any BC, while switching BC at fixed resolution moves it by 30-45%.

Ordering across BCs is as predicted at every resolution: `periodic` < `zeroDisplacement` <
`freeSurface`. Clamping adds a spurious triaxial tension pushing every grain toward threshold; free
surfaces let boundary grains relax and concentrate mismatch on their neighbours; periodic has
neither artefact.

The damage warning (threshold 0.9) never triggered -- peak across all nine runs was 0.702 -- so the
kinetic solver stayed inside its trustworthy range throughout.

---

### The effective CTE is resolution-dependent -- but NOT because of damage

**This section previously claimed the effective CTE "inherits the localization problem". That was
wrong, and the correction is more interesting than the error.**

`alphaBar_33` from the `periodic` runs, comparing step 1 (zero damage) against step 10 (damaged):

| resolution | step 1, undamaged | step 10, damaged | change | max damage at step 10 |
|---|---|---|---|---|
| 1 elem/grain | 2.677e-6 | 2.670e-6 | **-0.3%** | 0.163 |
| 8 elem/grain | 3.460e-6 | 3.458e-6 | **-0.0%** | 0.370 |
| 64 elem/grain | 4.138e-6 | 4.141e-6 | **+0.1%** | 0.533 |

**The effective CTE does not respond to damage at all.** Max damage reaches 0.53 and `alphaBar`
moves by 0.1%. Same for the homogenized stiffness: `Cbar_3` goes 195.12 -> 195.01 GPa, **-0.06%**,
over the whole cooling run.

The reason is the localization itself, and the two findings fit together exactly: mean damage at
64 elements/grain only reaches **0.0052**, so the damaged volume is about half a percent. `Cbar` and
`alphaBar` are VOLUME AVERAGES, and a tightly localized damage field barely perturbs a volume
average no matter how intense it is locally.

So the +55% spread across resolutions (2.677 -> 3.460 -> 4.138) is present **at zero damage**. It is
a pure **elastic homogenization convergence** problem: with one trilinear hex per grain the
fluctuation field `u~` cannot represent the strain needed to accommodate neighbour mismatch, which
constrains the RVE artificially and biases `alphaBar` low, toward the Voigt end. Refining relaxes it.

That distinction matters because the two problems have different fixes:

- **Elastic non-convergence of `alphaBar`** is cured by refinement. Increments are +0.78 and +0.68
  between the three resolutions, so it is **not converged even at 64 elements/grain**, and three
  points are too few to extrapolate the limit. A 512-elements/grain run would settle it.
- **Damage localization** is NOT cured by refinement -- it gets worse -- and needs a length scale in
  the constitutive law.

**How the error happened, since it is worth not repeating:** the final `alphaBar` values were
compared across resolutions and the difference attributed to damage without ever checking the
undamaged value. The "undamaged ~2.2e-6" figure quoted as a reference came from the earlier baseline
runs, which used `alpha_c = 27e-6` -- a different material configuration entirely, and not
comparable to these.

### What IS safe to quote from these runs

`Cbar` and `alphaBar` are reliable as **elastic** homogenized properties at a given resolution, since
damage perturbs them by less than 0.3%. What is not reliable is the resolution: they are still
drifting at 64 elements/grain. And the damage field itself -- peak value, damaged volume -- remains
mesh-dependent for the separate reason above.

### Strain output, added 2026-09-14

Full stress AND total strain, both frames, 38 arrays per step. Total strain means
`eps = Ebar + sym(grad u~)`; the elastic strain that generates stress is that minus the thermal
eigenstrain, which is `diag(alpha_a, alpha_a, alpha_c)*(T-T_0)` in the LOCAL frame for every grain,
so it is one known subtraction away and is not duplicated.

Verified against the written files, not against theory: `<eps_global> = Ebar` to 4.9e-8, 2.0e-8 and
7.9e-9 relative at 1, 8 and 64 elements per grain -- i.e. Lemma 1 of `homogenization.md`
(`<sym grad u~> = 0`) holds in the output, and tightens with refinement as it should. Also
`tr(eps_global) = tr(eps_local)` and `c . eps_global . c = eps_33` to float32 precision.

All nine runs reproduced their previous max-damage values EXACTLY, confirming the added output is
pure instrumentation.

## Study 2: realization variability (`seed-sweep/`)

`periodic`, 8 elements/grain, 8 seeds.

| quantity | mean | sd | CoV | range |
|---|---|---|---|---|
| **max damage** | 0.3554 | 0.0321 | **9.0%** | 0.314 - 0.407 |
| effective CTE `alphaBar_33` | 3.343e-6 | 1.56e-7 | 4.7% | 3.01e-6 - 3.47e-6 |
| mean damage | 7.246e-3 | 2.85e-4 | 3.9% | 6.91e-3 - 7.71e-3 |
| volume fraction over threshold | 0.2204 | 7.7e-3 | 3.5% | 0.210 - 0.232 |
| max sigma_33 | 206.9 MPa | 0.243 | **0.1%** | 206.7 - 207.4 |

**The extreme-value statistic is the realization-sensitive one**, as anticipated when this model was
designed: max damage varies by 9%, with a 30% spread between the least and most damaged realization.
Averages are two to three times more stable.

**Max sigma_33 is realization-independent (0.1%)** because it is not a free quantity -- the damage
feedback pins it just above threshold regardless of the microstructure.

**1000 grains is not yet a representative volume.** `alphaBar` still moves 4.7% between
realizations; a genuinely representative cell would hold an effective property to well under 1%.

### How many seeds does a conclusion need?

With CoV ~9% on max damage, separating two models by roughly two standard errors needs

| effect size in max damage | seeds |
|---|---|
| 20% | ~1 |
| 10% | ~4 |
| 5% | ~13 |

So a single realization supports "this changed max damage by 20%" and nothing finer. Note also that
with n = 8 the CoV estimates themselves carry ~27% uncertainty.
