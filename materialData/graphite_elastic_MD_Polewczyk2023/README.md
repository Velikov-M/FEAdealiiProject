# Graphite single-crystal elastic constants vs temperature (MD, Polewczyk et al. 2023)

Reusable temperature dependencies of the five independent elastic constants of a transversely
isotropic graphite crystallite, fitted to molecular-dynamics results.

| file | what it is |
|---|---|
| `digitized_LCBOP_II.csv` | the raw points, read from the paper's figure |
| `fit_elastic.py` | the only thing that produces the fits -- re-run it after correcting any point |
| `elastic_constants_fits.json` | every fit: coefficients, validity range, errors, model-ready strings |
| `fit_check.png` | data vs all three fit families, per constant |

## Provenance -- read this before quoting any number

- **Source:** Polewczyk et al. (2023), molecular dynamics. *Full citation to be filled in.*
- **Potential:** LCBOP_II only. It is the only potential in the figure plotted for all five
  independent constants (REBO and CEDIP are absent from the C33, C13 and C44 panels) and the only one
  with an uncertainty band. Mixing potentials across constants would give a tensor no single model
  supports.
- **Digitized by eye** from a figure image, not from tabulated data. Estimated precision about 1% of
  each panel's y-axis span: C11 +-5, C12 +-2, C33 +-0.4, C13 +-0.15, C44 +-0.008 GPa. The second
  abscissa is read as 500 K (+-50 K) -- the figure has no tick there. **If the paper or its
  supplementary material tabulates these values, replace the CSV and re-run `fit_elastic.py`.**

## Recommended fits

Parsimony rule: the simplest form within 5% max relative error over its own validity range, tried as
linear (300-4000 K) -> linear (1000-4000 K) -> quadratic (300-4000 K).

| | form | valid [K] | max error | value at 2000 K | coefficients (GPa, T in K) |
|---|---|---|---|---|---|
| C11 | linear | 300-4000 | 4.3% | 759.4 | a = 960.8, b = -0.10074 |
| C12 | **quadratic** | 300-4000 | 4.6% | 83.25 | a = 176.85, b = -0.069555, c = 1.1376e-5 |
| C33 | **quadratic** | 300-4000 | 6.5% | 12.61 | a = 32.777, b = -0.013124, c = 1.5202e-6 |
| C13 | linear | **1000**-4000 | 4.6% | 13.06 | a = 7.5714, b = 0.0027429 |
| C44 | linear | **1000**-4000 | 2.4% | 0.480 | a = 0.22286, b = 1.2857e-4 |

The recommended set is **positive definite** (transversely isotropic stability) over 300-4000 K,
checked on a 10 K grid.

**Why not linear everywhere:**

- **C12 and C33 flatten at high T**, so a line over the full range is off by +24% (C12) and +13%
  (C33) *at 2000 K*, inside the modelling window -- not merely at the ends.
- **C13 and C44 have a kink below 1000 K** (the 300 K point sits low). A line fitted only above
  1000 K is excellent there (<=0.4% at 2000 K); a line through all points is off by 22-31% at 300 K.
  So these two are **not valid below 1000 K** -- extrapolating down to room temperature needs a
  different fit.
- **C12's quadratic has a minimum near 3000 K** and turns up after it, following a single uptick at
  4000 K. Do not extrapolate it beyond 4000 K.

## C66 is not independent

Transverse isotropy gives `C66 = (C11 - C12)/2`, which is how the model computes it. Checked against
the C66 panel of the same figure: agreement within +-2.4% up to 3000 K and 4-5% at 3500-4000 K,
i.e. at the level of the digitization error. So the figure is consistent with transverse isotropy
and C66 needs no fit of its own.

## Using a fit

`elastic_constants_fits.json -> constants -> <Cij> -> <family>` contains:

- `coefficients_GPa` -- `C(T) = a + b*T (+ c*T^2)`, for use in any code.
- `model_ready` -- the form this repository's `input.json` consumes,
  `Cij(T) = Cij_0 * Cij_functionOfTemperature(T)`, normalised so that `f(2000 K) = 1`. `Cij_0` is
  then simply the constant's value at 2000 K, and `f` is a muParser string in the variable `T`.

## Effect in the model (resolvedGrainCooling, 2026-10-02)

Applied with `apply_to_input.py`. Verified in a single-crystal run: the model's `Cbar` equals the
MD fit at every temperature step to 7.6e-8 (the digits kept in the muParser coefficients).

Effect, separated from the Kelvin/Voigt shear fix that landed at the same time -- periodic BCs,
seed 20260905, cooling 2000 -> 1800 K:

| | 1 elem/grain | | 8 elem/grain | |
|---|---|---|---|---|
| | max sigma_33 | max damage | max sigma_33 | max damage |
| old constants, buggy shear | 204.7 MPa | 0.090 | 205.8 MPa | 0.243 |
| old constants, fixed shear | 204.6 MPa | 0.087 | 205.9 MPa | 0.215 |
| **MD constants**, fixed shear | **89.0 MPa** | **0** | **102.4 MPa** | **0** |

The shear fix moves damage by 4-11%. The MD constants **halve the peak c-axis stress and eliminate
damage** over this cooling window, because C33 falls from 36.5 to ~13-14 GPa and the mismatch
stress scales with it. (The "old" peak of ~205 MPa was itself pinned at the threshold by the damage
feedback, so the undamaged contrast is larger still.)

**Consequence: the damage threshold sigma_th = 200 MPa is now the most consequential unsourced
number in the model.** It was never sourced, and every damage result so far implicitly depended on
it being reached with the stiffer, experimental C33.
