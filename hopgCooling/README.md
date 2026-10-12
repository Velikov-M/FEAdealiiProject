# HOPG mosaic cooling

**Question:** can the mosaic spread of HOPG, the small scatter of crystallite c-axes measured by
X-ray or gamma-ray rocking curves, produce noticeable thermal stresses on cooling?

This folder reuses the periodic thermo-elastic solver from `../resolvedGrainCooling`, with a
different interpretation of the model:

- each "grain" is now one **HOPG crystallite**, not a randomly oriented domain of a polycrystal;
- the c-axes are nearly aligned, not uniformly random;
- there are no inter-crystallite cracks and no damage.

## Setup

| | |
|---|---|
| RVE | 10 x 10 x 10 = 1000 cubic crystallites, 64 hexahedral elements each (64 000 cells, 206 763 DoFs) |
| boundary conditions | periodic, macroscopic stress `<sigma> = 0` (free cooling of a large body) |
| temperature | stress-free at **2500 K**, cooled to 1000 K in 50 K steps, uniform in space |
| elastic constants | Polewczyk et al. 2023 MD (LCBOP_II), T-dependent: `../materialData/graphite_elastic_MD_Polewczyk2023/` |
| CTE | Marsden et al. 2018 CCM acoustic + optical, T-dependent, **secant re-integrated from 2500 K**: `../materialData/graphite_cte_CCM_Marsden2018/cte_fits_Tref2500.json` |
| damage | off (`DamageParameters/DamageEnabled = false`) |
| mosaic spread | sigma = 0.25, 0.5, 1 deg, one run each, same seed (20260905) |

`configure_input.py` writes all of this into `input.json` and refuses to run if the CTE file's
reference temperature does not match `InitialTemperature`. Don't edit the CTE strings by hand.

## The orientation model

Each crystallite's c-axis is tilted from the global z axis by (theta_x, theta_y). The two
components are independent Gaussians with standard deviation **sigma**
(`BoundaryConditions/MosaicSpreadDeg`, sampled by `mosaicOrientationGaussian()` in `main.cpp`).

- **Why per component:** a rocking curve rotates the sample about one in-plane axis, so it
  measures the distribution of *one* tilt component. sigma is the std of that 1D marginal.
- **The tilt magnitude** |theta| is then Rayleigh distributed, with mean sigma * sqrt(pi/2), about
  1.25 sigma.
- **sigma is not the FWHM.** Mosaic spreads are usually quoted as the rocking-curve FWHM, which is
  2.355 sigma for a Gaussian. So the three cases correspond to FWHM = 0.59, 1.18 and 2.35 deg.
  **Check which convention the source you compare against uses.**
- **Rotation about the c-axis is not sampled.** It is a material symmetry of the transversely
  isotropic crystallite.
- **The three cases share one seed on purpose.** `std::normal_distribution` returns sigma * z with
  z drawn from the same random stream, so the three meshes carry the same tilt pattern, scaled.
  Differences between cases are the effect of sigma alone, and the stress-vs-sigma exponent comes
  out free of realization noise. Confirmed from the logs: sampled std 0.252221 deg at
  sigma = 0.25 and 1.00888 deg at sigma = 1, exactly a ratio of 4.

## Verification

- **sigma = 0 (perfect crystal), 1 element per crystallite.** The stress is zero to round-off
  (von Mises <= 2e-11 MPa over all 30 steps). Ebar at 1000 K equals the hand-computed thermal
  strain from the CTE fits to every printed digit: Ebar_xx = -1.8557021e-3 (alpha_a,sec = 1.2371e-6)
  and Ebar_zz = -5.3267067e-2 (alpha_c,sec = 35.511e-6). This checks the whole chain: the CTE
  strings, the secant reference, the eigenstrain and the periodic solve.
- **Sampler:** the sampled std of each tilt component and the mean |tilt| match the requested
  sigma and the Rayleigh mean, within sampling error of 1000 crystallites (1%).
- **Solver paths:** the quasi-Newton and exact paths give the same stress statistics to 5e-5.

## Solver choices (measured, see `configure_input.py`)

- **AMG, not CG.** A near-single crystal keeps the full C11/C44 ~ 1400 anisotropy coherent across
  the box, so CG needed 800-1000 iterations per solve. AMG needs 110-275, and two steps took
  240 s instead of 388 s.
- **Exact 7-solve path, not the frozen-Cbar quasi-Newton.** C(T) changes Cbar by ~3% per step.
  The frozen iteration can't reach tolerance in 4 passes, so every step ended in a full rebuild:
  12 solves instead of 7.
- **Step size doesn't affect accuracy.** With no damage and a secant CTE, the state at T does not
  depend on the path, so the 50 K step only sets the output resolution.

## Outputs

`python3 run_mosaic.py` writes one folder per case under `results/mosaic-spread/`. Each contains the
exact `input.json`, the console log, `hopg_macroscopic_history.csv` (Ebar, alphaBar, Cbar per
step), `hopg_stress_statistics.csv`, and the field dumps at 2000, 1500 and 1000 K (gitignored).
`python3 analyze_mosaic.py` makes the plots and `summary.md`.

`hopg_stress_statistics.csv` reports stress measures that don't depend on how the in-plane axes
are chosen, because the local 1 and 2 axes are an arbitrary completion of the c-axis frame. Each
is computed in the crystallite's own frame:

- **sigma33:** normal stress across the basal plane (cleavage / delamination driver)
- **inPlaneMax, inPlaneMin:** principal stresses of the basal-plane 2x2 block
- **tauBasal:** resolved basal shear, sqrt(s13^2 + s23^2) (basal-slip driver)
- **vonMises**

For each: volume-weighted mean, RMS, 1st and 99th percentiles, min and max. Quote the
percentiles. Min and max are single-point extremes at crystallite corners and depend on the mesh.

## Results: `results/mosaic-spread/` (2026-10-12)

Final state is 1000 K (1500 K of cooling). Stresses are in MPa; the in-plane row combines both
principal stresses.

| sigma (FWHM equiv.) | in-plane, 1st-99th percentile | sigma33, 1st-99th percentile | tauBasal, 99th percentile | von Mises RMS |
|---|---|---|---|---|
| 0.25 deg (0.59 deg) | -3.2 ... +2.1 | -0.9 ... +0.9 | 0.43 | 1.3 |
| 0.5 deg (1.18 deg) | -11.7 ... +7.5 | -1.9 ... +1.9 | 0.84 | 4.5 |
| 1 deg (2.35 deg) | -42.7 ... +27.8 | -4.5 ... +4.5 | 1.64 | 16.5 |

How the RMS stress scales with sigma (per interval, 0.25->0.5 and 0.5->1):

| measure | exponent | reading |
|---|---|---|
| tauBasal | 1.00, 1.00 | exactly linear |
| sigma33 | 1.06, 1.21 | linear at small sigma, drifting towards quadratic |
| in-plane | 1.83-1.88, 1.88-1.89 | nearly quadratic, approaching 2 |

- **The basal-plane stresses dominate.** Shear mismatch is first order in the tilt but acts through
  the very soft C44. In-plane mismatch is second order (theta^2) but acts through C11, which is
  ~2500x stiffer. The result is in-plane stresses an order of magnitude above sigma33 and
  tauBasal, and their quadratic growth widens that gap at larger spreads.
- **sigma33 is linear, not quadratic, at small sigma.** The first-order shear mismatch couples into
  the normal stress across the basal plane. At larger sigma the theta^2 term takes over.
- **All measures grow almost linearly with cooling**, except tauBasal, which grows slower:
  C44 in the MD fit softens on cooling (0.55 -> 0.35 GPa from 2500 to 1000 K).
- **The in-plane distribution is skewed**, with a longer compressive tail than tensile tail
  (p01 -43 vs p99 +28 at 1 deg). This is not explained yet; the field dumps would show which
  crystallites carry it.
- **Single-point extremes reach about 3x the percentiles** (in-plane min -125 MPa at 1 deg). They
  sit at crystallite corners and depend on the mesh, so do not quote them.

Whether these levels are "noticeable" depends on strength values the model does not contain:

- tauBasal should be compared with the basal-slip resolved shear strength;
- sigma33 with the interlayer (cleavage) tensile strength;
- in-plane with the basal-plane strength.

All three are still to be sourced.

Cost: 58, 58 and 51 min per case (12 threads, AMG, exact path).

## Limitations: read before quoting a number

- **Cubic crystallites on a regular lattice.** Real HOPG crystallites are platelets: wide in the
  basal plane and thin along c. The aspect ratio changes how in-plane mismatch is constrained, so
  these numbers are for an idealized morphology.
- **No inelasticity.** Graphite's basal shear is very easy, so basal slip would relax tauBasal in
  reality. The elastic tauBasal is an upper bound and should be compared with a basal-slip
  strength, which isn't in this model yet.
- **Stress-free at 2500 K** is an assumption: it is where the material is taken to become elastic.
  Moving it shifts every result, about as much as changing the cooling range by the same amount.
- **Small-strain kinematics** with ~5% uniform c-axis contraction at 1000 K. Only the much smaller
  mismatch part produces stress, but finite-rotation effects of that size are not represented.
- **Material data:** both sources are models, not measurements. The CCM CTE is ~8-12% above the
  high-temperature data in the same figures, and the MD elastic constants were read off a figure.
