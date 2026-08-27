# Polycrystal cooling model (first mechanically meaningful model)

Cooling of a periodic RVE of randomly oriented graphite domains. Thermal-expansion mismatch
between misoriented anisotropic grains is the classic microcracking mechanism in polycrystalline
graphite, so cooling alone produces a non-trivial stress field with no applied load at all.

**Status: scaffold.** `main.cpp` is still a verbatim copy of `../mmsSandbox/main.cpp` and currently
runs the MMS test, not this model. The build is configured and compiles. See the checklist at the
top of `main.cpp`.

## Setup

- Same 10x10x10 hexahedral mesh; **each original cell is one domain** (1000 grains).
- Each domain gets its own orientation and thermo-mechanical properties.
- Orientations initially **fully random**; properties initially **ideal graphite crystallite**.
- **Periodic** boundary conditions (the mesh is periodic, so face pairs match exactly).
- Cooling from 2000 K to 1800 K in 10 steps.

## Design decisions and the reasoning behind them

### Macroscopic strain: free cooling, `<sigma> = 0`

Plain periodicity `u(x+L) = u(x)` forces `<eps> = 0`, so the RVE cannot contract and develops a
uniform triaxial tension of order `3*K*alpha_mean*|dT|` — roughly **tens of MPa** for these
parameters. That is a systematic offset on every grain, and it biases exactly the tension/compression
comparison this study exists to make.

The correct free-cooling form splits the displacement as `u = Ebar.x + u_periodic`, with the
macroscopic strain `Ebar` determined by requiring `<sigma> = 0`. While the problem is linear (damage
is post-processing in step 1) this is a superposition:

1. solve once with `Ebar = 0`  ->  `<sigma>_0`
2. solve six unit-`Ebar` problems  ->  homogenized stiffness `Cbar`
3. `Ebar = -Cbar^-1 : <sigma>_0`, then superpose

Seven solves per temperature step, **all with the same matrix**, so the preconditioner is built once.
`Cbar` and the effective CTE `alphabar` drop out as useful by-products.

**This is only valid while the problem stays linear** — it breaks once damage actually evolves.

### Resolution: start at 1 element/grain, then refine and compare

A single trilinear hex per domain gives essentially constant stress per grain and cannot resolve
grain-boundary concentrations, which is where nucleation happens. So 1 element/grain is not adequate
for locating initiation sites on its own.

It is, however, the right *starting* point: the machinery gets verified before any long computation
is committed. `refine_global(2)` then gives 4x4x4 elements/grain (64000 cells, ~1 min) and
`refine_global(3)` gives 8x8x8 (512000 cells, ~8 min). deal.II propagates `material_id` to children
automatically, so grain identity survives refinement with no bookkeeping. Comparing the two makes the
resolution sensitivity a measured result rather than an assumption.

### Orientation sampling: uniform on the sphere

Only the **domain normal (c-axis) direction** matters — rotation about the c-axis is a symmetry of a
transversely isotropic crystal — so this is a point-on-sphere problem, not SO(3).

Uniformly sampling polar angles is the classic trap: it clusters orientations at the poles. Sample
`cos(theta)` uniform on [-1,1] and `phi` uniform on [0,2pi]. Use a standard construction rather than
improvising, and check the resulting distribution is actually isotropic before trusting any result.
Record the seed.

### Statistics: qualitative first

Damage initiation is an extreme-value statistic and genuinely realization-sensitive, so solid
conclusions would need several seeds and reported distributions. That is deliberately **out of scope
for now**: the present aim is a qualitative grasp of the model's behaviour — enough to judge what
must be added to capture graphite's nano-cracking behaviour. Multiple realizations become necessary
when this turns into a publication, not before.

## Material data

The CTE values in `input.json` are **placeholders**, flagged as such in their `documentation` fields:
`alpha11_0 = 1e-6`, `alpha33_0 = 27e-6`. Real graphite crystallite values are roughly
`alpha_a ~ -1..+1e-6` and `alpha_c ~ 25-30e-6`; the near-zero (possibly negative) basal value against
a large c-axis value is the entire driver of the mismatch, so these must be replaced with sourced
values before any physical conclusion is drawn.

The **elastic constants are already crystallite values** and can stay (C11=1060, C33=36.5, C44=0.25,
C12=180, C13=15 GPa).

## Risks

- **Time wall.** More than ~15 minutes per run calls for extraordinary optimization rather than
  patience. CG iteration counts are now reported per solve, so conditioning trouble will be visible
  immediately. Random orientations with C11/C44 ~ 4240 may hit CG hard even with periodicity helping.
  If so, the Trilinos/AMG work moves up the queue.
- **Superposition assumes linearity** and must be revisited when damage evolution is switched on.

## Where this sits in the wider plan

1. **this model**, simple assumptions
2. restore temperature dependence of properties in the MMS (requires re-deriving the manufactured
   body force for `C(T)` and `alpha(T)`; note the code's thermal strain is `alpha*(T-T0)`, a
   mean/secant CTE, so an instantaneous `alpha(T)` from literature must be integrated first)
3. redo this model with temperature-dependent parameters and compare
4. redo again with averaging-corrected domain properties (Shen et al., *The microstructure and
   texture of Gilsocarbon graphite*, 2019) and compare
5. Trilinos rewrite -> realistic morphology and real compute machines
