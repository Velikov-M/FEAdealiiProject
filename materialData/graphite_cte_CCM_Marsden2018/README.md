# Graphite c-axis CTE vs temperature (CCM, Marsden et al. 2018)

The c-axis thermal expansion coefficient of a graphite crystallite, from the crystal-continuum model
(CCM) with acoustic **plus optical** modes. Fitted, then integrated to the secant CTE the FE model
uses.

| file | what it is |
|---|---|
| `source_figure_alpha_c.png` | the figure as supplied (red dashed curve = the digitized one) |
| `digitize_alpha_c.py` | pixel-colour digitizer; produces the CSV |
| `digitized_alpha_c_CCM_acoustic_optical.csv` | 332 points, 300-2500 K, in 1e-6/K |
| `fit_cte.py` | the only thing that produces the fits -- re-run it after changing data or `T_REF` |
| `cte_fits.json` | every fit: coefficients, errors, instantaneous + secant tables, model-ready strings |
| `fit_check.png` | data vs fits (instantaneous solid, secant dashed) and residuals |
| `overlay_check.png` | the einstein2 fit drawn back onto the source figure -- calibration check by eye |

`alpha_a` is not here yet; `fit_cte.py` takes one CSV per axis (`axes_data`), so it slots in as a
second entry.

## Provenance -- read this before quoting any number

- **Source:** B. Marsden, A. Mummery, P. Mummery, *Modelling the coefficient of thermal expansion in
  graphite crystals: implications of lattice strain due to irradiation and pressure*, Proc. R. Soc. A
  **474** (2018) 20180075, doi:10.1098/rspa.2018.0075 (open access, PMC6237503). The figure is
  Fig. 4 according to the PMC text -- confirm against the PDF.
- **It is an instantaneous CTE.** Eq. (2.6): `alpha_c = d e_zz / dT`. It also behaves like one:
  alpha_c -> 0 as T -> 0. The "acoustic plus optical" curve is the paper's simplification of
  extending the upper integration limit to 2x, so the optical modes are approximated, not computed
  separately.
- **It is a model curve, not data.** Above ~1000 K it lies above the high-temperature measurements in
  the same figure: about 35.8 at 2000 K, against ~33 for Morgan (XRD) and ~33 for Steward (our
  previous source). Choosing it over the Steward curve raises alpha_c by ~8% at 2000 K.
- **Digitization:** by pixel colour, not by eye. The axes were calibrated from the tick marks:
  18.5 px = 100 K, and 38.5 px = 5e-6/K. As a check, the curve gives alpha_c(100 K) ~ 17e-6, which
  matches Bailey & Yates. Precision is about half a pixel, i.e. +-0.07e-6/K (+-0.2%). Only
  T >= 300 K is kept: below that the curve runs through the identically red Bailey & Yates markers.

## Instantaneous vs secant

The FE model computes `eps_th = alpha(T) * (T - T_0)`, so `alpha` there is the **secant from
T_0 = 2000 K** (the stress-free `InitialTemperature`):

    alpha_sec(T) = 1/(T - T_ref) * integral_{T_ref}^{T} alpha_inst(T') dT',   T_ref = 2000 K

Both fit families integrate in closed form (checked against numerical quadrature to 1e-14), so the
secant carries no quadrature error. **The secant is tied to T_ref: if `InitialTemperature` changes,
set `T_REF` in `fit_cte.py` and re-run.**

## Fits

Parsimony rule: the simplest form within **1%** max relative error over its own range. This is
tighter than the 5% used for the elastic constants, because this digitization is about 10x more
precise.

| fit | form | valid [K] | max err | alpha_inst(2000) | secant at 1800 / 1000 / 300 K |
|---|---|---|---|---|---|
| linear_high | `c0 + c1 u`, u = T-2000 | 1000-2500 | 1.00% (fails rule) | 35.72 | 35.64 / 35.30 / -- |
| **quadratic_high** (recommended) | `c0 + c1 u + c2 u^2` | 1000-2500 | 0.32% | 35.81 | 35.75 / 35.33 / -- |
| einstein2_full | `A1 E(th1/T) + A2 E(th2/T)` | **300**-2500 | 0.60% | 35.77 | 35.72 / 35.33 / 34.00 |

(units 1e-6/K; `E(x) = x^2 e^x / (e^x - 1)^2`, the Einstein heat-capacity function)

- **quadratic_high** is the recommendation for any cooling that stays above 1000 K, which includes
  the current 2000 -> 1800 K window. Its secant is again a polynomial, `sum c_k u^k / (k+1)`, so
  there is no 0/0 at T_ref.
- **einstein2_full** is the one to use **below 1000 K**. It is physically shaped (Grueneisen:
  alpha ~ heat capacity, one Einstein term per mode group), its residuals look like noise
  (+-0.05e-6/K), and it fitted theta1 = 361 K and theta2 = 1740 K: the same acoustic + optical
  split as the model behind the curve. Its secant is a 0/0 at T_ref, so the model-ready string guards
  it with `if(abs(T-2000)<1e-3, 1, ...)`. deal.II's FunctionParser defines `if` (it returns only the
  chosen branch, so the 0/0 branch never leaks a NaN). The T_ref term is written as the same
  expression evaluated at 2000, not as a printed constant, so the cancellation near T_ref is exact.
- Within 1000-2500 K the two disagree by at most 0.34% (instantaneous) and 0.09% (secant), both
  inside the digitization error. The choice between them only matters below 1000 K.

Over the current 2000 -> 1800 K window, secant and instantaneous differ by only 0.1-0.2%. The
distinction starts to matter at 300 K, where einstein2 gives a secant of 34.0 against an
instantaneous 27.1.

## Using a fit

`cte_fits.json -> axes.alpha_c.fits.<fit>.model_ready` gives

    alpha33_0                       = alpha_0_per_K
    alpha33_functionOfTemperature   = secant_functionOfTemperature     (f(2000 K) = 1)

**Not applied to `input.json`.** `alpha_a` is still missing, and switching alpha_c alone would change
the driving mismatch. All six strings (three fits, instantaneous and secant) were parsed by deal.II's
`FunctionParser<1>` and agree with the Python fits to 5e-10, including at T_ref +- 0.002 K.
