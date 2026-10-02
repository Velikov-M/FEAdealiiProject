# Graphite crystallite CTEs vs temperature (CCM, Marsden et al. 2018)

The c-axis and basal-plane thermal expansion coefficients, alpha_c and alpha_a, of a graphite
crystallite. They come from the crystal-continuum model (CCM) with acoustic **and optical** modes,
are fitted here, and are integrated to the secant CTE the FE model uses.

| file | what it is |
|---|---|
| `source_figure_alpha_c.png`, `source_figure_alpha_a.png` | the figures as supplied |
| `digitize_alpha_c.py`, `digitize_alpha_a.py` | pixel-colour digitizers; produce the CSVs |
| `digitized_alpha_{c,a}_CCM_acoustic_optical.csv` | 332 / 438 points, 300-2500 K, in 1e-6/K |
| `fit_cte.py` | the only thing that produces the fits -- re-run it after changing data or `T_REF` |
| `cte_fits.json` | every fit: coefficients, errors, instantaneous + secant tables, model-ready strings |
| `fit_check.png` | data vs fits (instantaneous solid, secant dashed) and residuals, both axes |
| `overlay_check_alpha_{c,a}.png` | the full-range fit drawn back onto each source figure (calibration check) |

## Provenance -- read this before quoting any number

- **Source:** B. Marsden, A. Mummery, P. Mummery, *Modelling the coefficient of thermal expansion in
  graphite crystals: implications of lattice strain due to irradiation and pressure*, Proc. R. Soc. A
  **474** (2018) 20180075, doi:10.1098/rspa.2018.0075 (open access, PMC6237503).
  - alpha_c is the red dashed curve, in Fig. 4 according to the PMC text (confirm against the PDF).
  - alpha_a is the grey-blue dashed curve in Fig. 6.
- **Both are instantaneous CTEs.** Eq. (2.6) defines `alpha_c = d e_zz/dT` and
  `alpha_a = d e_xx/dT`. The "acoustic plus optical" curves are the paper's simplification of
  extending the upper integration limit to 2x, so the optical modes are approximated, not computed
  separately.
- **They are model curves, not data, and both sit above the high-temperature measurements:**

  | at 2000 K | this CCM curve | measurements in the same figures | Steward (current input.json) |
  |---|---|---|---|
  | alpha_c | 35.8 | ~33 (Morgan, XRD) | 32 |
  | alpha_a | 1.39 | ~1.15-1.2 (Morgan, Kellett) | 1.2 |
  | **alpha_c - alpha_a** | **34.4** | ~32 | **30.8** |

  Switching `input.json` to these curves would raise the mismatch that drives the stresses by about
  12%.
- **The published curves are polylines.** In Fig. 6 the kinks at 100, 200 and 300 K are clearly
  visible: the CCM was evaluated at about 100 K intervals and the points joined by straight lines.
  Residual structure below about 0.01e-6/K near 300 K is that interpolation, not physics.
- **Digitization** is by pixel colour, with the axes calibrated from the tick marks (see each
  digitizer's docstring). Precision is about half a pixel: +-0.07e-6/K for alpha_c and
  +-0.004e-6/K for alpha_a. Only T >= 300 K is kept. For alpha_a, the legend box is masked: glyph
  edges of the label text pass the colour test.

## Instantaneous vs secant

The FE model computes `eps_th = alpha(T) * (T - T_0)`, so `alpha` there is the **secant from
T_0 = 2000 K** (the stress-free `InitialTemperature`):

    alpha_sec(T) = 1/(T - T_ref) * integral_{T_ref}^{T} alpha_inst(T') dT',   T_ref = 2000 K

Every fit integrates in closed form (checked against numerical quadrature to 5e-14), so the secant
carries no quadrature error. **The secant is tied to T_ref: if `InitialTemperature` changes, set
`T_REF` in `fit_cte.py` and re-run.**

## Fits

Parsimony rule: the first family in the list that is within tolerance on its own range.

- **alpha_c:** 1% max relative error. The digitization is about 0.2%.
- **alpha_a:** 0.03e-6/K max **absolute** error. alpha_a crosses zero near 650 K, so a relative error
  is meaningless there. 0.03e-6/K is 0.1% of the mismatch alpha_c - alpha_a, about 2% of
  alpha_a(2000 K), and about 4 px on the figure.

### alpha_c

| fit | form | valid [K] | max err | inst(2000) | secant at 1800 / 1000 / 300 K |
|---|---|---|---|---|---|
| linear_high | `c0 + c1 u`, u = T-2000 | 1000-2500 | 1.00% (fails) | 35.72 | 35.64 / 35.30 / -- |
| **quadratic_high** | `c0 + c1 u + c2 u^2` | 1000-2500 | 0.32% | 35.81 | 35.75 / 35.33 / -- |
| einstein2_full | `A1 E(th1/T) + A2 E(th2/T)` | **300**-2500 | 0.60% | 35.77 | 35.72 / 35.33 / 34.00 |

- Fitted modes: theta1 = 361 K (A1 = 29.7) and theta2 = 1740 K (A2 = 6.5).

### alpha_a

| fit | form | valid [K] | max err | inst(2000) | secant at 1800 / 1000 / 300 K |
|---|---|---|---|---|---|
| linear_high | `c0 + c1 u` | 1000-2500 | 0.169 (fails) | 1.365 | 1.314 / 1.109 / -- |
| quadratic_high | `c0 + c1 u + c2 u^2` | 1000-2500 | 0.039 (fails) | 1.408 | 1.371 / 1.125 / -- |
| **einstein1c_full** | `c + A E(th/T)` | **300**-2500 | 0.024 | 1.392 | 1.359 / 1.129 / 0.666 |
| cubic_high | cubic in u | 1000-2500 | 0.011 | 1.394 | 1.361 / 1.126 / -- |

- **Why einstein1c rather than einstein2:** a free two-mode fit drives theta1 -> 0, where E -> 1. The
  negative contribution (out-of-plane flexural modes, the membrane effect) is already fully excited
  above 300 K and acts as a **constant**, c = -0.654. The positive in-plane term has
  theta = 2701 K (A = 2.376), between the paper's theta_L = 2562 K and theta_M = 3623 K. So
  3 parameters cover the whole 300-2500 K range.
  - **Do not extrapolate it below 300 K:** the real curve has its minimum near 250 K and returns to
    0 at T = 0.
- **The quadratic fails because it is 1% off at T_ref itself** (1.408 vs 1.392-1.394), not only at
  the ends of its range.

### Which to use

- **Cooling that stays above 1000 K** (includes the current 2000 -> 1800 K window): alpha_c
  quadratic_high and alpha_a einstein1c_full, or einstein for both. Within 1000-2500 K the families
  of each axis agree to within the digitization error.
- **Cooling below 1000 K:** the `_full` fits (einstein2 for alpha_c, einstein1c for alpha_a).
- At 300 K, alpha_a shows why the secant matters: instantaneous -0.630, but secant from 2000 K
  +0.666, with the opposite sign.

### Guarded strings

The Einstein secants are 0/0 at T_ref, so their model-ready strings are guarded with
`if(abs(T-2000)<1e-3, 1, ...)`. deal.II's FunctionParser defines `if`, and it returns only the
chosen branch, so the 0/0 branch never leaks a NaN. The T_ref term is written as the same expression
evaluated at 2000, not as a printed constant, so the cancellation near T_ref is exact. Polynomial
secants are again polynomials, `sum c_k u^k / (k+1)`, with no 0/0.

## Using a fit

`cte_fits.json -> axes.<alpha_c|alpha_a>.fits.<fit>.model_ready` gives

    alpha33_0 / alpha11_0                                  = alpha_0_per_K
    alpha33_functionOfTemperature / alpha11_...            = secant_functionOfTemperature   (f(2000 K) = 1)

**Not applied to `input.json`.**

All 14 strings (7 fits, each instantaneous and secant) were parsed by deal.II's `FunctionParser<1>`
and agree with the Python fits to 6e-10, including at T_ref +- 0.002 K.

## Why the colours swap between Fig. 4 and Fig. 6

The same model curve is red dashed in Fig. 4 but grey-blue dashed in Fig. 6. The pixel colours show
this has no physical meaning:

| | Fig. 4 (alpha_c) | Fig. 6 (alpha_a) |
|---|---|---|
| acoustic only | solid, RGB (94,129,181) | solid, RGB (255,0,0) |
| acoustic + optical | dashed, RGB (255,0,0) | dashed, RGB (94,129,181) |

- **(94,129,181) is Mathematica's default first plot colour**, `ColorData[97][1] =
  RGBColor[0.368417, 0.506779, 0.709798]`, to the last digit. It is not grey: thin dashes
  anti-aliased against white make it look grey.
- **(255,0,0) is the named `Red`**, set explicitly. Every data series is also a named colour (Red,
  Orange, Green, Blue, Purple), consistent across both figures.
- In each figure one model curve was styled `Red` and the other left at the default. The authors
  styled a different curve in each figure, so the colours swap.
- **The line style is the consistent encoding:** dashed = acoustic + optical in both figures.
- Other signs the figures were built independently: the legend wording differs ("acoustic plus
  optical mode" vs "acoustic and optical"), and Fig. 6 is visibly sampled coarser. The red model
  line also clashes with the red Bailey & Yates markers in both figures, which a deliberate colour
  scheme would avoid.
