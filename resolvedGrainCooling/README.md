# resolvedGrainCooling

The `polycrystalCooling` model, run at resolutions where a grain is made of **several elements**
rather than one. Forked from it at commit `f842c3a`, immediately after the periodic `<sigma> = 0`
formulation was validated.

Two things change relative to the parent:

1. **Grain identity is decoupled from cell identity.** In `polycrystalCooling` a grain *is* a cell
   (`material_id` = cell index). Here grains are assigned from a coarse grain lattice by centroid
   lookup, so the mesh can be refined without changing the grain count and each grain ends up made
   of several elements. That is what makes grain-boundary stress concentrations representable at
   all -- a single trilinear hex gives essentially constant stress per grain and cannot show them.
2. **The linear solver has to keep up.** See below.

Everything else -- the periodic formulation, the damage kinetics, the output, the index convention
-- is inherited unchanged, so results are directly comparable with
`../polycrystalCooling/results/`.

## Why this needed a solver change first

Measured on the parent, CG + SSOR:

| DoFs | CG iterations |
|---|---|
| 3 993 | 47 |
| 27 783 | 138 |

That is `iterations ~ N^0.56`, so solve cost grows as `~N^1.56` and the solve is already 74% of wall
time. Extrapolated to 10 cooling steps: ~1.2 min at 8 elements/grain, ~3.5 min at 16, and **~30 min
at 64** -- past the 15-minute wall.

`SolverParameters/LinearSolverType` now offers `direct` | `cg` | `amg`, so the three can be compared
on identical problems. See **Measurements** below for what was actually found, including what did
*not* work.

## Notes on the AMG setup

- **`constant_modes` is the setting that matters.** Elasticity is a systems problem: three
  displacement components per node, with the three rigid translations as the operator's near-null
  space. Passing those to ML via `DoFTools::extract_constant_modes` took the iteration count from
  37/76 to **17/30**. Without it ML coarsens as though the problem were scalar and barely improves
  on SSOR.
- **AMG suits this formulation better than a single-solve code.** Its setup is expensive, but the
  periodic path solves seven systems per assembly against one matrix, so the setup amortizes
  roughly sevenfold.
- **Running the solve in native Trilinos containers was tried and is slower.** 20.2 s versus 13.4 s
  of solve time on the same problem: in serial, deal.II's `SparseMatrix::vmult` and plain
  `Vector<double>` inner products beat the Trilinos equivalents by more than the per-application
  vector conversion costs. Worth revisiting only when going distributed, where Trilinos containers
  stop being optional. (A per-element copy loop into a Trilinos vector was slower still -- element
  access there goes through a per-entry lookup; `operator=` does a bulk copy.)
- `main()` now takes `argc`/`argv` and constructs `MPI_InitFinalize`, because Trilinos calls
  `MPI_Comm_size()` while building any of its objects even on one rank. The thread count is left
  as `invalid_unsigned_int` so MPI does not cap threading, leaving
  `ParallelizationParameters/NumberOfThreads` as the single control.

## Measurements

Isolating the refined mesh (run at 1 cycle and at 2, and differencing, since the timer aggregates):

| solver | wall time, 8000-cell cycle | CG iterations @ 27783 DoFs |
|---|---|---|
| `cg` (SSOR) | **18.4 s** | 138 |
| `amg`, `AmgSmootherSweeps=1` | 22.9 s | 45 |
| `amg`, `AmgSmootherSweeps=2` | 23.1 s | 30 |

**AMG cuts iterations 4.6x and still loses on wall time at this size.** It is per-iteration-bound
here, not iteration-bound. But its iteration growth is much better -- `N^0.29` against SSOR's
`N^0.56` -- so there is a crossover. Extrapolating from these two points puts it near **65 k DoFs
(~21 k cells)**, about where 16 elements/grain lands, and gives AMG roughly a 1.65x edge at 64
elements/grain.

That is not enough on its own: it takes the 64-elements/grain case from ~30 min to ~18 min, still
past the 15-minute wall. 8 and 16 elements/grain are comfortable with either solver.

## Verification

`BoundaryConditions/UniformOrientation = true` at three refinement levels. The grain lattice holds
the grain count fixed while the mesh refines under it, and the closed-form answer is reproduced
exactly at every level:

| cells | elements/grain | `Cbar` diagonal (GPa) | asymmetry |
|---|---|---|---|
| 1 000 | 1 | `1060 1060 36.5 0.25 0.25 440` | 1.1e-31 |
| 8 000 | 8 | `1060 1060 36.5 0.25 0.25 440` | 2.3e-31 |
| 64 000 | 64 | `1060 1060 36.5 0.25 0.25 440` | 1.5e-30 |

`Cbar = C` exactly is the single-crystal prediction, so this checks the periodic machinery, the
grain assignment and its survival of refinement all at once. `alphaBar` comes out at
`1e-06, 1e-06, 2.7e-05` (the input crystal CTE) and `Ebar` at `alpha*dT` at every level too, and
the pointwise stress stays at the 1e-6 Pa level against moduli of 1e12 -- so `sigma = 0` holds to
~1e-18 relative even on 64 000 cells, where the condensation runs seven solves over 195 k DoFs.
