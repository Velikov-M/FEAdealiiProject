# Optimization review, 2026-09-23

Revisits the options with current measurements. Two figures in `scaling_options.md` are now wrong
and are corrected here.

## Where the time actually goes

`periodic`, 64 elements/grain, with the `alpha_33` coupling, 3930 s total:

| section | calls | time | share |
|---|---|---|---|
| **Solve system** | 511 | **3620 s** | **92%** |
| Assemble system | 83 | 53 s | 1.4% |
| **Predict damage** (toms748) | 73 | **41 s** | **1.0%** |
| everything else | | ~200 s | 5% |

`511 solves = 7 x 73 fixed-point iterations`, at 7.08 s each.

**This sets hard ceilings.** Any change to the root-finder is worth at most **1%**. Any change to
assembly at most **1.4%**. Only two things matter: the NUMBER of solves and the COST of one.

---

## The biggest lever is not on the list: stop recomputing `Cbar`

Six of every seven solves exist only to build `Cbar`. Measured, `Cbar` moves **0.035% over the
entire cooling run**:

| | step 1 | step 10 | change |
|---|---|---|---|
| `CbarDiag_1` | 209.698 | 209.627 GPa | -0.034% |
| `CbarDiag_3` | 195.121 | 195.053 GPa | -0.035% |
| `CbarDiag_6` | 114.795 | 114.746 GPa | -0.043% |

So the model spends 86% of its solves recomputing a constant.

**The correct exploitation is a frozen-Jacobian quasi-Newton on `Ebar`, not a cache.** Naively
reusing stale `Y_k = K^-1 L_k` would give a wrong `U`, because `K` does change with damage. Instead:

- refresh `Cbar` occasionally (6 solves),
- each fixed-point pass, solve `K U = F_theta - L Ebar` **once** with the CURRENT `K`,
- update `Ebar <- Ebar - Cbar_frozen^-1 <sigma>_true`, where `<sigma>_true` comes from
  `computeAverageStress()` -- already implemented, quadrature only, no solve.

The frozen `Cbar` is only a Jacobian for the `Ebar` update; the residual driving it is exact. So it
converges to the TRUE answer, at a linear rate of ~3.5e-4 per pass -- effectively one iteration.

| scheme | solves | projected time | speedup |
|---|---|---|---|
| current | 511 | 3930 s | 1.00x |
| refresh `Cbar` per temperature step | 133 | 1252 s | **3.1x** |
| refresh `Cbar` once per run | 79 | 870 s | **4.5x** |

**CORRECTION.** `scaling_options.md` projected "cache `Cbar` while damage is unchanged" at **4.4x**.
That was computed on the old baseline, where damage appeared only at step 10 and iterations were a
uniform ~2.1 per step. With the current CTE values damage starts at step 6 and the late steps
dominate (35 iterations at step 10), so the naive version is now worth **1.12x**. The 3-4.5x above
comes from the quasi-Newton form, which exploits `Cbar` being near-constant *even while damage
changes* -- a different mechanism.

---

## The three options asked about

### 1. Matrix-free

Attacks the 92%. Unchanged from `scaling_options.md`: deal.II's step-37 states the matrix-free
product is **slower than a sparse one for linear elements**, and this code is Q1. Its gain is that
it is compute-bound and therefore threads (the current solve measures 0.96x on 12 threads), and
above all that it enables **geometric multigrid**, the only option that changes the exponent
(`N^1.69 -> ~N^1.2`).

*With healing:* neutral. Matrix-free is operator-agnostic; a healing term changes what the operator
computes, not how it is applied.

**Verdict:** destination, not next step. It deletes all three preconditioner options at once, and
the frozen-`Cbar` change above buys 3-4.5x for a fraction of the work.

### 2. Replacing TOMS 748 with a derivative-based method

Attacks **1.0%** of runtime. Even making the root search free saves 1%.

Worse, it reverses a decision already made for good reason: Newton was tried here and was
unreliable, because the residual is concave and admits **two roots, the larger unphysical**. The
current scheme locates the maximum analytically, brackets on the side with the sign change, and uses
toms748 -- bracket-guaranteed and superlinear.

*With healing:* this is the one place healing genuinely forces work, but for **correctness, not
speed**. Three things break:

- `lowerBound = old_damage` -- damage is no longer non-decreasing within a step, so the bracket's
  lower end must be re-derived.
- the closed-form `omega_max` is specific to the current Rabotnov law; a healing term changes the
  residual's shape and may remove the single-interior-maximum property the bracketing relies on.
- the Aitken relaxation clamps to `[old_damage, 0.999]` and would fight any healing.

**Verdict:** do not change it for speed -- the speed is not there. Re-derive the bracket when the
healing law is written, and keep a bracketing method: a two-root residual is exactly the situation
where a derivative method silently converges to the wrong root.

### 3. Replacing the staggered PDE/ODE fixed point with monolithic Newton

Attacks the iteration count (73). This is the one with real headroom -- step 10 alone needs 29
passes at a linear rate near 0.8, where Newton would take perhaps 5.

But two things cut it down:

- **Aitken is already capturing much of it.** Measured: 162 iterations without it, 73 with. The
  remaining gap to Newton is smaller than the raw 73 suggests.
- **The monolithic tangent is non-symmetric.** `C_tan = C(omega) + (dC/domega)(eps-eps_th) ⊗
  domega/deps`, a rank-one term whose two factors are not parallel -- and the `alpha_33` coupling
  adds another such term. CG dies; GMRES/BiCGStab costs ~1.5-2x per iteration and more memory.

Estimate: ~2.5x fewer iterations x 1.5-2x dearer per solve = **1.3-1.7x net**. Real, but below the
frozen-`Cbar` change and far more invasive.

*With healing:* **this is the argument that may flip.** A staggered fixed point alternating between
a growth branch and a healing branch can oscillate rather than converge -- the iteration count could
degrade sharply, or stop converging at all. Newton does not care which branch it is on, provided the
tangent is consistent. If healing destabilises the outer loop, monolithic goes from "1.3-1.7x" to
"the only thing that works".

**Verdict:** not now. But instrument the fixed-point convergence when healing lands -- if the
iteration counts blow up or oscillate, this becomes the priority rather than an optimization.

---

## Recommended order

1. **Frozen-`Cbar` quasi-Newton on `Ebar`.** 3-4.5x, small and local, justified by a measured
   0.035% drift. Nothing else on the list comes close per unit of work.
2. **Re-derive the damage root-finder bracket for the healing law.** Correctness, not speed.
   Keep bracketing.
3. **Re-measure.** Once the solve count drops 4x, the solve may no longer be 92% and the ranking
   changes.
4. **Monolithic Newton if healing destabilises the outer loop** -- watch the iteration counts.
5. **Matrix-free + geometric multigrid** when the target is 512 elements/grain or a cluster.
