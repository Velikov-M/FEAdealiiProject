# Material-data provenance and validity

## Rule for using material data

Every scientific statement must distinguish the source data, the digitization/fit, and the exact model configuration. A good fit to a digitized curve is not experimental validation of the crystallite model. Do not extrapolate fitted functions beyond their documented ranges without a separate approved assessment.

Model-ready JSON files and generation scripts are under `materialData/`. Input entries contain short provenance strings; this document records the current baseline usage.

## Elastic constants: Polewczyk et al. 2023 MD

Location: `materialData/graphite_elastic_MD_Polewczyk2023/`

Current tracked resolved-grain and HOPG inputs both use the recommended temperature-dependent fits from `elastic_constants_fits.json`:

- source family: molecular dynamics with the LCBOP_II potential;
- values digitized by eye from a figure in `digitized_LCBOP_II.csv`;
- full bibliographic citation remains incomplete in the fit metadata;
- `C11_0`, `C12_0`, `C33_0`, `C13_0`, and `C44_0` are values at 2000 K;
- temperature functions are normalized to one at 2000 K;
- `C66` is derived as `(C11-C12)/2`, not independently fitted.

Recommended-fit diagnostics against the digitized MD values:

| Constant | Fit | Nominal valid range | Maximum relative fit error |
|---|---|---:|---:|
| C11 | `linear_full` | 300–4000 K | 4.31% |
| C12 | `quadratic_full` | 300–4000 K | 4.64% |
| C33 | `quadratic_full` | 300–4000 K | 6.53% |
| C13 | `linear_high` | 1000–4000 K | 4.55% |
| C44 | `linear_high` | 1000–4000 K | 2.38% |

The fit metadata reports that the recommended set is positive definite over 300–4000 K, but the selected C13 and C44 fits are not nominally valid below 1000 K. These errors measure approximation to digitized MD data, not error against real graphite crystals.

The tracked resolved-grain interval 1800–2000 K and HOPG interval 1000–2500 K lie within the selected fit ranges. Replacing eye-digitized values with tabulated source data, if available, remains desirable before publication-quality quantitative claims.

Useful files:

- `fit_elastic.py`: refits constants, checks positive definiteness, and writes model-ready functions.
- `apply_to_input.py`: copies recommended fits into a chosen input JSON; it modifies that file, so review the diff.
- `fit_check.png`: visual fit diagnostic.
- folder `README.md`: detailed provenance and fitting notes.

## CTE: two different current baselines

### Resolved-grain tracked input

`resolvedGrainCooling/input.json` currently uses constant values:

- `alpha11_0 = 1.2e-6/K`;
- `alpha33_0 = 32e-6/K`;
- both temperature functions are `1.0`.

The input documentation attributes these values to visual readings of a Steward et al. curve near 1800–2000 K. The archive does not provide a complete provenance package for that curve comparable to the Marsden folder.

The available Marsden `cte_fits.json`, referenced to 2000 K, is **not applied** to this tracked baseline. Any proposal to adopt it is a material-data change requiring a new run comparison, not a documentation correction.

### HOPG tracked input

`hopgCooling/input.json` uses Marsden et al. 2018 CCM acoustic-plus-optical curves from `materialData/graphite_cte_CCM_Marsden2018/cte_fits_Tref2500.json`:

- source: B. Marsden, A. Mummery, P. Mummery, *Modelling the coefficient of thermal expansion in graphite crystals: implications of lattice strain due to irradiation and pressure*, Proceedings of the Royal Society A 474 (2018) 20180075, DOI `10.1098/rspa.2018.0075`;
- the paper's curves are model results digitized from figures, not direct measurements in this repository;
- the figures provide instantaneous CTE; repository scripts integrate it to the secant CTE required by the constitutive law;
- the reference/stress-free temperature is 2500 K, matching `InitialTemperature`.

Selected instantaneous-fit diagnostics:

| Axis | Fit | Nominal valid range | Fit-selection diagnostic |
|---|---|---:|---|
| c (`alpha33`) | `quadratic_high` | 1000–2500 K | Maximum relative error 0.323% |
| a (`alpha11`) | `einstein1c_full` | 300–2500 K | Maximum absolute error 0.0244e-6/K |

For `alpha_a`, the maximum relative error over the full range is approximately 55% because the curve approaches/crosses small values; the selection criterion was maximum **absolute** error, not a 1% relative-error guarantee. Do not quote the 55% figure without that context or hide it by quoting only the absolute error.

The HOPG 1000–2500 K interval lies within both selected ranges. `configure_input.py` checks this and the 2500 K reference before rewriting the baseline.

Useful files:

- `digitize_alpha_a.py`, `digitize_alpha_c.py`: figure-curve digitization;
- `fit_cte.py`: instantaneous fitting, secant integration, diagnostics, and JSON generation;
- `cte_fits.json`: secant reference 2000 K;
- `cte_fits_Tref2500.json`: same instantaneous fits converted to secant reference 2500 K;
- digitized CSV files, source figures, overlays, and fit-check plots: audit trail;
- folder `README.md`: detailed equations, range notes, and fitting decisions.

## Damage parameters are not calibrated material data

Both current input files retain:

- `StressThreshold = 200e6 Pa`;
- `a_kineticParam = 4.0`;
- `m_kineticParam = 2`.

These values are numerical baseline choices in the archive; no calibration or source has been established by this audit. HOPG disables damage, so these entries do not affect its tracked baseline. Resolved-grain predictions involving damage must be described as conditional on these uncalibrated parameters.

## Required checks when changing data or reference temperature

1. Identify whether each CTE function is instantaneous or secant; the model requires secant thermal strain.
2. Re-reference secant CTE whenever `InitialTemperature` changes.
3. Check every elastic and CTE fit over the complete requested temperature interval, including its lower endpoint.
4. Preserve source, digitization, fit family, range, and fit-error information in the run record.
5. Re-run verification and representative production cases; changing material data can alter damage, conditioning, and solver rankings.
6. Do not compare new results with older `results/` without recording the material-data change.
