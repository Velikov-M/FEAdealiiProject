# Coupling alpha_33 to damage: what changed

`alpha_33 = alpha_33(T) * (1 - omega)`, linear, exactly as `C33` was already treated. Nine runs
(3 BCs x 3 resolutions), seed 20260905, 2000 -> 1800 K in ten steps, 1000 grains. The control is
`bc-resolution-sweep/`, which differs ONLY in this coupling.

Plots: `cte_comparison_maxDamage.png`, `..._meanDamage.png`, `..._alphaBar_33.png` -- max damage
against cooling depth `T_0 - T`, dashed = control, solid = coupled, colour = resolution.

## Max damage: large reduction, but it decays with resolution

| | 1 elem/grain | 8 | 64 |
|---|---|---|---|
| `zeroDisplacement` | 0.2304 -> 0.1291 (**-44%**) | 0.4194 -> 0.2747 (**-35%**) | 0.5413 -> 0.3964 (**-27%**) |
| `freeSurface` | 0.2902 -> 0.1776 (**-39%**) | 0.4882 -> 0.3539 (**-28%**) | 0.7022 -> 0.6225 (**-11%**) |
| `periodic` | 0.1631 -> 0.0904 (**-45%**) | 0.3702 -> 0.2430 (**-34%**) | 0.5334 -> 0.4071 (**-24%**) |

The second stabilizing feedback is real: shrinking the c-axis eigenstrain in a damaged region cuts
the mismatch that drives the damage. But **the effect weakens monotonically with resolution**, from
~40-45% at one element per grain to 11-27% at 64, consistently across all three BCs.

That is the localization again. At one element per grain a damaged cell IS the grain, so reducing
its `alpha_33` directly relieves the eigenstrain mismatch against its neighbours -- the feedback
acts on the very thing generating the stress. At 64 elements per grain the damage is a thin layer
near the grain boundary, and softening that layer's CTE barely changes the bulk mismatch between
the two grains still driving it.

Consequence: **the coupling lowers absolute damage but WIDENS the relative spread across
resolutions.** `freeSurface` goes 0.178 -> 0.622 across resolutions with the coupling, against
0.290 -> 0.702 without.

## Mean damage: uniform ~44% reduction, resolution-independent

| | 1 | 8 | 64 |
|---|---|---|---|
| `zeroDisplacement` | -46% | -45% | -45% |
| `freeSurface` | -44% | -43% | -40% |
| `periodic` | -44% | -44% | -44% |

Unlike the peak, the volume-integrated damage drops by a nearly constant fraction at every
resolution. So the coupling removes a fixed proportion of the total damage while redistributing the
peak -- consistent with it acting on the bulk eigenstrain rather than on the localization itself.

## Effect on the fixed-point loop: small

periodic, 64 elements/grain:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total | solves | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 2 | 2 | 2 | 2 | 2 | 5 | 6 | 8 | 15 | 35 | 79 | 553 | 3890 s |
| coupled | 2 | 2 | 2 | 2 | 2 | 5 | 6 | 8 | 15 | 29 | 73 | 511 | 3930 s |

Steps 1-9 are identical; only step 10 improves, 35 -> 29. Total solves fall 7.6% and wall time is
unchanged. So the hoped-for damping of the outer loop is **marginal**, not the second payoff it
looked like mid-run.

## The effective CTE still barely moves

Even with `alpha_33` explicitly tied to damage, `alphaBar_33` changes by well under 1% over a
cooling run, at every resolution. The reason is unchanged and structural: damage occupies under 1%
of the volume outside the clamped case, and `alphaBar` is a volume average.

**So this coupling is not the route to matching a measured bulk CTE.** It changes the damage field
substantially and the homogenized property almost not at all. Getting `alphaBar` to respond would
need either far more damaged volume, or the CTE reduction to act over a length scale rather than
cell by cell.
