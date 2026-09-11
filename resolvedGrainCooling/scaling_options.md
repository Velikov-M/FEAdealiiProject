# Going faster: MPI, matrix-free, or neither yet

Evaluation of the two candidate rewrites against three questions: **what do they win**, **what gets
rewritten**, and **what does it cost in readability**. Everything numeric below is measured on this
machine unless labelled *projected*.

---

## 0. Where the time actually goes

Threads 1 -> 12, periodic BCs, 8000 cells / 27 783 DoFs:

| section | 1 thread | 12 threads | speedup |
|---|---|---|---|
| assemble | 7.78 s | 1.54 s | **5.1x** |
| damage | 2.15 s | 0.454 s | **4.7x** |
| **solve (cg+SSOR)** | 15.3 s | 16.0 s | **0.96x** |
| **solve (cg+AMG)** | 12.7 s | 12.6 s | **1.01x** |

The two WorkStream loops scale. The solve, which is ~75% of wall time, does not move at all.

### Why — and it is not "no parallel solver"

- `SparseMatrix::vmult` **is** threaded (`parallel::apply_to_subranges`, TBB), as are the vector
  operations. Checked in the deal.II sources.
- `precondition_SSOR` contains no parallel construct: the forward/backward sweeps are inherently
  sequential.
- Trilinos ML on a single rank is serial; its parallelism is MPI, not threads.

So part of the solve is already threaded and still does not speed up. The cause is that **sparse
matrix-vector product is memory-bandwidth-bound**: roughly 2 flops per ~12 bytes loaded, an
arithmetic intensity near 0.17 flop/byte. The memory controllers saturate after 2-3 threads and
further threads only add overhead — which is why CG got marginally *slower* at 12.

### The hardware this has to run on

```
AMD Ryzen 5 7500F   6 physical cores / 12 threads
Socket(s): 1        NUMA node(s): 1     L3: 32 MiB     RAM: 15 GiB
```

**One socket, one NUMA node, one memory system.** Remember this for section 2.

### Measured cell-count scaling

Two things grow with the mesh, and both have to be counted.

**Inner: CG iterations per solve.**

| cells | DoFs | CG+SSOR iterations |
|---|---|---|
| 1 000 | 3 993 | 47 |
| 8 000 | 27 783 | 138 |
| 64 000 | 206 763 | 321 |

`~ N^0.487`.

**Outer: fixed-point iterations per temperature step** (10 steps of 20 K, so a realistic step size):

| cells | iterations/step | max |
|---|---|---|
| 1 000 | 2.10 | 3 |
| 8 000 | 3.10 | 7 |

`~ N^0.201`. Confirmed against the solve count exactly: 21 + 31 solve-sets = 364 solves.

Each temperature step costs `(fixed-point iterations) x 7 solves x (CG iterations)`, so

> **solve cost ~ N^(1 + 0.487 + 0.201) = N^1.69**

Projected wall time for 10 cooling steps, anchored on the measured 65 s at 8 000 cells:
**~1 min** at 8 elements/grain, **~32 min** at 64, **~17 h** at 512.

At 64 000 cells the solve is **92% of wall time** (measured: 1 500 s of 1 640 s), up from 73% at
8 000. It gets worse, not better, with size.

### A note on the convergence criterion

Of the three criteria, `maxDamageInc` is the one that binds; the other two pass by two to three
orders of magnitude. At 8 000 cells the final check reads

```
relSolInc   = 1.79e-07  (tol 1e-05)   passes by 56x
maxDamageInc= 3.17e-06  (tol 1e-05)   passes by 3x     <-- binding
relResidual = 9.76e-08  (tol 1e-04)   passes by 1000x
```

It is a **max-norm over all quadrature points against a fixed absolute tolerance**, so refining adds
samples to the max and makes it harder to satisfy. The code's defence of the choice is sound (damage
is self-limited to [0,1], so an absolute max-norm is physically meaningful) but the side effect is
the `N^0.201` growth above, part of which is a criterion artefact rather than physics.

The effect is mild at a realistic step size. It is NOT mild at a large one: a single 200 K step at
64 000 cells took **28** fixed-point iterations, against 3.1 for ten 20 K steps. So the outer loop
degrades sharply with step size, and step-size choice is itself a performance lever.

---

## 1. Do these first — they need no rewrite at all

### 1a. Cache `Cbar` and the six unit-strain solves while damage is unchanged

The periodic path does **7 solves per assembly**: one thermal, six unit-strain. The six exist only
to build `Cbar`, which depends on the damage field. In the baseline run **damage is identically
zero for 9 of 10 temperature steps**, so `Cbar` and the six `Y_k` are recomputed nine times over
with identical results.

Measured on the baseline: 147 solves over 10 steps (21 solve-sets, 2.1 fixed-point iterations per
step). Reusing `Cbar` and `Y_k` whenever the damage field has not moved:

> **147 solves -> 34. A 4.4x reduction, for a cache and a dirty flag.**

Caveats: `Cbar` also depends on `C(T)`, which is temperature-independent *today* (the
`*_functionOfTemperature` entries are 1.0) but will not be after step 2 of the plan restores it.
Even then the cache is valid across the fixed-point iterations *within* a step, which is still
~2.1x. The dirty test must be on the damage field, not on the step index.

### 1b. Stop assembling the same matrix twice per iteration

42 assemblies for 21 solve-sets. `checkConvergenceCriteria()` re-assembles at the updated damage to
form the residual — correct and necessary — but the *next* loop iteration then assembles again at
that same damage state. One of the two is redundant: **~2x on assembly** by carrying the matrix
forward.

### 1c. Rotate the stiffness once per cell, not once per quadrature point

Verified numerically to 1.3e-16 relative:

> **`C_global(omega) = C_global(0) - omega * C33_0 * (n (x) n (x) n (x) n)`**,  `n` = the c-axis in
> global coordinates.

Damage enters the *global* stiffness as a **rank-one update along the c-axis**. Today
`assemble_cells_*` calls `basis_transformation` on a rank-4 tensor at *every quadrature point*,
although the rotation is identical for all eight points of a cell and only `omega` differs. A rank-4
basis transformation is the dominant per-point cost. Rotate `C_global(0)` once per cell, then apply
the rank-one correction per point.

**Combined, 1a-1c attack the same 75% that the rewrites target, and cost days rather than months.**

---

## 2. MPI (distributed memory)

### What it wins

**On this machine: essentially nothing.** One socket, one NUMA node — MPI ranks contend for exactly
the same memory controllers that threads do. For a bandwidth-bound solve, moving from 12 threads to
12 ranks changes who is waiting on memory, not how much memory bandwidth exists. Expect **1.2-2x**
from better data locality and from removing TBB overhead, not 6x.

**On a cluster: near-linear in nodes**, because every node brings its own memory system. This is the
*only* option that goes past one machine, and it is the reason it sits in the plan at all.

It does **not** change the exponent: cost stays `~N^1.69`, with a smaller constant and more hardware
to spend. 4 nodes turns the 17 h case into ~4 h; it does not turn it into 15 minutes.

### What gets rewritten

Roughly half the file. Mechanical by nature, but it touches nearly every structural concept:

| area | change |
|---|---|
| `triangulation` | -> `parallel::distributed::Triangulation` (p4est) |
| `setup_system` | `locally_owned_dofs` / `locally_relevant_dofs`; constraints on the relevant set |
| matrix, vectors | -> `TrilinosWrappers::SparseMatrix`, `TrilinosWrappers::MPI::Vector`, owned vs ghosted |
| `assemble_cells_*` | `is_locally_owned()` guards, `compress()` |
| `macroCouplingColumns` | six distributed vectors, compressed |
| `computeAverageStress` | `Utilities::MPI::sum` over the 6 stress components and the volume |
| `solve_with_zero_mean_stress` | the 6x6 dense part stays, but must be *identical* on every rank |
| damage storage | `CellDataStorage` per rank, plus `ContinuousQuadratureDataTransfer` on repartition |
| `constrainOneVertex`, `constrainRigidBodyModes` | the support vertex lives on one rank only |
| `process_solution` | min/max/sum become reductions |
| `output_results` | `write_vtu_with_pvtu_record`, subdomain filtering |
| `run()` | console output on rank 0; orientations generated identically on all ranks |

**Untouched:** `getLocalStressTensor`, `kineticResidual`, `update_damage_on_cell`,
`getElasticityTensor`, `assignGrains`, `orientationFromCAxis`. The physics functions are per-cell and
per-quadrature-point, and distribution does not reach them.

### Readability cost: moderate, and well-localised

Ownership guards, `compress()` calls and reductions are noise, but they cluster in setup, assembly
plumbing and output. A reader looking for the mechanics still finds `sigma = C:(eps - alpha dT)` and
the kinetic law written exactly as they are today. The danger is subtler: `<sigma> = 0` and the `Cbar`
symmetry check become *collective* operations, so a missing reduction gives a wrong answer on >1 rank
and a right one on 1 rank. Those need explicit comments saying so.

---

## 3. Matrix-free

### What it wins

The honest starting point is deal.II's own step-37 documentation:

> *"The implementation of the matrix-free matrix-vector product shown in this tutorial is **slower
> than a matrix-vector product using a sparse matrix for linear elements**, but faster for all higher
> order elements... The impact of reduced memory transfer is particularly beneficial when working on
> a multicore processor... an algorithm which is computation bound will show almost perfect parallel
> speedup."*

This code uses **Q1**. So at the current element degree the matrix-free product is *slower per
application* — but it is compute-bound, and therefore it **scales with cores where the present solve
does not** (measured: 0.96x). On 6 physical cores a plausible net is **3-5x**, with real uncertainty
at Q1. Going to Q2 would put it on the favourable side of that quote and improve accuracy per DoF,
but that is a discretization change, not a performance knob.

**The actual prize is not the product — it is the preconditioner.** Matrix-free's natural partner is
geometric multigrid, which gives a **near mesh-independent iteration count**. That replaces `N^0.487`
growth with `O(1)`:

| cells | DoFs | iterations now | with GMG (projected) |
|---|---|---|---|
| 8 000 | 27 783 | 138 | ~10 |
| 64 000 | 206 763 | 321 | ~10 |
| 512 000 | ~1.6 M | ~870 | ~10 |

**That changes the exponent, from `~N^1.69` to `~N^1.2`** (the residual `N^0.2` being the outer
fixed-point loop, which multigrid does not touch). No amount of MPI does that. It is also by
far the largest project here.

### What gets rewritten

| area | change |
|---|---|
| assembly | the stiffness loop disappears, replaced by an operator `local_apply` using `FEEvaluation` |
| RHS and the 6 `L` columns | become matrix-free applications, or stay assembled |
| `system_matrix` | gone — and with it SSOR, AMG **and** UMFPACK; all three preconditioner options die |
| preconditioner | must become Chebyshev+Jacobi (diagonal is computable matrix-free) or geometric multigrid |
| `checkConvergenceCriteria` | `system_matrix.residual()` -> an operator application |
| damage storage | `CellDataStorage` -> `AlignedVector<VectorizedArray<double>>` in MatrixFree's cell-batch layout |
| per-cell material data | `C_global(0)` and `n` stored per **cell batch**, interleaved across SIMD lanes |

The `C(omega)` rank-one identity of section 1c is what makes this tractable at all: without it the
operator would have to rotate a rank-4 tensor at every quadrature point on every application, which
would bury any gain. With it, the per-point work is `C_global(0):eps` minus about ten flops.

`solve_with_zero_mean_stress` survives structurally — the 7-solve condensation does not care how the
operator is applied.

### Readability cost: mixed, and this is the interesting part

**Better in one way.** A matrix-free operator reads much closer to the weak form than an assembly
loop does. Instead of `cell_matrix(i,j) += grad_i : C : grad_j * JxW` you write, at a quadrature
point, "take the symmetric gradient, multiply by C, submit it" — which is literally the constitutive
law. For someone reading for mechanical meaning, step-37-style code is arguably *clearer* than the
i,j double loop.

**Worse in another.** `VectorizedArray<double>` propagates into the types: the stress becomes
`SymmetricTensor<2, dim, VectorizedArray<double>>`, and per-cell quantities (orientation, damage)
must be gathered across SIMD lanes, because a "cell" in MatrixFree is a batch of 4 or 8 geometric
cells processed together. That lane bookkeeping has no mechanical meaning at all and is the part a
non-specialist reader will stumble on.

Net: the *constitutive* code gets clearer, the *data layout* code gets considerably harder.

---

## 4. Recommendation

1. **Do section 1 first.** 4.4x from caching `Cbar`, ~2x on assembly from not assembling twice, plus
   the rank-one rotation. That is days of work against the same bottleneck, and it makes 64
   elements/grain (~32 min today) land near 7 minutes — inside the 15-minute wall, with no rewrite.
2. **Then re-measure.** The bottleneck may well have moved; committing to a rewrite against a
   measurement that is three optimisations out of date would be a mistake.
3. **MPI when the target is a cluster, not before.** On this single-socket machine it buys ~1.5x. It
   is an enabler for different hardware, not a speedup on this one.
4. **Fix the outer loop too.** Multigrid attacks the inner `N^0.487`; the outer `N^0.201` is
   untouched by either rewrite. A mesh-robust damage criterion, or Anderson/Aitken acceleration of
   a fixed point converging linearly at rate ~0.81, is cheap and independent of both options.
5. **Matrix-free + geometric multigrid is the only option that changes the inner exponent**, and it is the
   right end state for 512 000 cells and beyond. It is also the biggest project, it deletes all three
   current preconditioner options, and at Q1 its raw product is slower than what we have. Treat it as
   the destination, not the next step.

