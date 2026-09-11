# Baseline: periodic RVE, 1 element per grain

Reference run to compare later work against. Captured at commit `19650d9`, the point where the
periodic `<sigma> = 0` formulation first passed validation.

| | |
|---|---|
| boundary conditions | `periodic` (`u = Ebar.x + u~`, `Ebar` fixed by `<sigma> = 0`) |
| mesh | 10x10x10 hexahedra, **1 element per grain**, 1000 grains, 3993 DoFs |
| orientations | uniform on the sphere, **seed 20260905** |
| cooling | 2000 K -> 1800 K in 10 steps |
| solver | CG + SSOR, 47 iterations; 7 solves per assembly |
| material | graphite crystallite elastic constants; **CTE values are still PLACEHOLDERS** |

`input.json` is the exact configuration. `periodic_seed20260905.csv` is the macroscopic history.

## Headline numbers

- Effective CTE `alphaBar ~ 2.25e-06` (vs `9.7e-06` for a naive orientation average) -- the stiff
  basal directions dominate, so the compliant c-axis expansion is largely suppressed.
- `Cbar` diagonal `384, 364, 348, 234, 235, 193` GPa -- nearly isotropic, from a single crystal
  that is `1060, 1060, 36.5, 0.25, 0.25, 440`.
- `<sigma>` at 6e-8 Pa against local stresses of 2e8, i.e. zero to machine precision.
- Damage just activates by the last step: mean 3.3e-06, max 0.017, 0.05% of volume above threshold.
- `Cbar` relative asymmetry 1.2e-10 throughout (Hill-Mandel check, not enforced anywhere).

## What this is NOT

One realization, one element per grain. Damage initiation is an extreme-value statistic and
genuinely realization-sensitive, and a single trilinear hex cannot resolve grain-boundary
concentrations. Both are known and deliberate at this stage -- see the parent README.

## Measured scaling (for sizing later runs)

CG iterations: 3993 DoFs -> 47; 27783 DoFs -> 138. Solve cost therefore grows as roughly
`N^1.56`, and the solve is 74% of wall time. Extrapolated, for 10 temperature steps:

| grain size | cells | DoFs | time |
|---|---|---|---|
| 1 element | 1 000 | 4 k | ~3 s |
| 8 elements | 8 000 | 24 k | ~1.2 min |
| 16 elements | 16 000 | 48 k | ~3.5 min |
| 64 elements | 64 000 | 192 k | **~30 min** -- past the 15 min wall |
