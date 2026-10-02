# Frozen-Cbar quasi-Newton: verification and timing

`SolverParameters/FrozenCbarQuasiNewton = true` against the exact 7-solve condensation. Nine runs
(3 BCs x 3 resolutions), seed 20260905, identical in every other respect. Control is
`bc-resolution-sweep-cte/`.

## Solutions coincide

Max relative difference over **all nine runs x all ten temperature steps**:

| | `Ebar_11` | `Ebar_33` | `alphaBar_33` | mean damage | max damage |
|---|---|---|---|---|---|
| `zeroDisplacement` (all 3) | 0 | 0 | 0 | 0 | 0 |
| `freeSurface` (all 3) | 0 | 0 | 0 | 0 | 0 |
| `periodic` 1 elem/grain | 1.4e-09 | 1.6e-09 | 1.5e-09 | 1.5e-08 | 5.9e-09 |
| `periodic` 8 elem/grain | 6.4e-09 | 1.9e-08 | 1.9e-08 | 2.5e-08 | 6.9e-07 |
| `periodic` 64 elem/grain | 1.7e-08 | 1.7e-08 | 1.7e-08 | 2.4e-07 | 3.8e-07 |

**Worst disagreement anywhere: 6.9e-07**, against a damage fixed-point tolerance of 1e-5. The
non-periodic runs are bit-identical because they use `solve()` and never enter the condensation at
all -- the quasi-Newton touches only the periodic path.

## Timing

| run | exact | quasi-Newton | speedup |
|---|---|---|---|
| `periodic` 1 elem/grain | 5.4 s | 3.4 s | 1.6x |
| `periodic` 8 elem/grain | 101.6 s | 28.0 s | **3.6x** |
| `periodic` 64 elem/grain | **3930 s** | **502 s** | **7.8x** |
| `zeroDisplacement`, `freeSurface` | unchanged | unchanged | 1.0x |

At 64 elements/grain: **511 solves -> 128**, with **one** `Cbar` rebuild across 72 fixed-point
passes.

The speedup **grows with problem size**, and overshoots the 3-4.5x projected in
`../optimization_review.md`. Two reasons the projection was low:

- it assumed a solve count of `6 x steps + passes`, i.e. a rebuild every temperature step. The
  self-tuning policy actually rebuilt **once in the entire run** -- `Cbar` is stable enough that
  the staleness detector almost never fires.
- the solve share rises with problem size (92% at 64 elements/grain), so removing solves converts
  more directly into wall time there than in the small case the projection was calibrated on.

## Caveat: the reported `Cbar` is the frozen one

With the quasi-Newton the `Cbar` written to the history is whatever was last rebuilt -- stale by
~1.5e-4 relative. That is within `Cbar`'s own 3.5e-4 drift over a run, and it does NOT enter
`Ebar`, `alphaBar` or damage, which come from the exact residual and match to 1e-7.

But `Cbar` is a publishable output in its own right (the homogenized stiffness). **If it is being
quoted as a result, rebuild it or run the exact path for the final number.** Flagged in the
parameter file via `ReportedCbarIsFrozen`.

## Tuning, which mattered more than expected

`MacroscopicStressTolerance = 1e-10` (the first guess, matching what the exact path achieves) is
counterproductive: it forces a third `Ebar` pass, which trips the staleness detector, which rebuilds
`Cbar` -- ten rebuilds, and only 1.27x. At **1e-8** the loop converges in two passes with one
rebuild and the answer is unchanged.

The floor is **two** `Ebar` passes, not one: the first solve only evaluates the residual at the
`Ebar` carried over from the previous pass; the second confirms the corrected one.

## Note added 2026-10-02: two field files in this archive were overwritten

`periodic_64cpg_seed20260905-1.vtu` and `-2.vtu` were overwritten by a stray run on 2026-10-02 and
have been renamed `*.CONTAMINATED`. Cause: `build/input.json` was a symlink to the tracked
`input.json`, and `run_sweep.py` wrote each run's configuration through it -- so the tracked file
was left pointing its `OutputDirectory` at this archive, and the next plain `./main` wrote into it.
The tracked files here (`input.json`, `.log`, `_macroscopic_history.csv`) were NOT touched: the
history CSV is only written at the end of a run, and the stray run was stopped before that. All
numbers in this document come from the CSVs and logs and are unaffected. Fixed at the root in the
commit that added this note -- see `run_sweep.py`.
