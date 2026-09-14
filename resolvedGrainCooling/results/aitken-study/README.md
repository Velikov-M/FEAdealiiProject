# Aitken relaxation: does it help, or was it overshooting?

`periodic`, 64 elements/grain, seed 20260905, 2000 -> 1800 K in ten steps. Two runs identical in
every respect except `SolverParameters/CouplingAcceleration`.

## Why this was tested

During the sweep, step 10 of the accelerated run showed a **non-monotonic** convergence trace --
`maxDamageInc` jumping back up at iterations 4 and 10 rather than decaying steadily. That looked
like Irons-Tuck overshooting: Aitken is derived for a LINEAR fixed point, and at max damage above
0.5 the displacement-damage coupling is strongly nonlinear. The worry was that the acceleration was
helping in the cheap low-damage regime and hurting in the expensive high-damage one -- exactly
backwards.

**It is not. The worry was wrong, and by a wide margin.**

## Result

Fixed-point iterations per temperature step:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **aitken** | 2 | 2 | 2 | 2 | 2 | 5 | 6 | 8 | 15 | **35** | **79** |
| **none** | 2 | 2 | 2 | 2 | 2 | 6 | 8 | 10 | 34 | **94** | **162** |

| | solves | wall time |
|---|---|---|
| aitken | 553 | 3 894 s (65 min) |
| none | 1 134 | 8 420 s (140 min) |

**2.05x fewer solves, 2.16x faster.** And the margin GROWS with accumulated damage: identical for
steps 1-5, 2.3x at step 9, **2.7x at step 10**.

The answer is unchanged, which is the point of an accelerator:

```
                  aitken                      none
damage mean       0.00519488                  0.00519488
damage max        0.533436                    0.533437
sigma_33 range    -1.09329e8 .. 2.08022e8     identical
alphaBar (diag)   3.8538e-6, 4.23123e-6, ...  identical
```

Only the sixth digit of max damage differs -- the 1e-5 convergence tolerance.

## What the non-monotonic trace actually meant

The upward jumps are the price of an extrapolation that nets out strongly positive, not evidence of
misfiring. A fixed point grinding down monotonically at rate ~0.81 every pass is WORSE than one that
occasionally overshoots and recovers. Reading non-monotonicity as failure was the error.

## The consequential part

**Without Aitken this configuration would not run at the production default.** `MaxNumberOfIterations`
is 100; the unaccelerated run needed **94** at step 10 and would have been within six passes of
throwing. It was raised to 300 for this test purely so a slow run would report a number rather than
an exception.

So Aitken is not a 10% optimisation here -- at 8 elements/grain it measured 10%, but that was a
low-damage case with little to accelerate. At high damage it is the difference between a 65-minute
run and a 140-minute one, and between converging and hitting the iteration cap.

It also reframes the cost model in `../scaling_options.md`: the outer fixed-point loop degrades from
2 to 94 passes across a single cooling run, which dwarfs the `N^0.201` mesh dependence measured on a
low-damage case. **At high damage the fixed point, not the linear solver, is the cost driver.**
