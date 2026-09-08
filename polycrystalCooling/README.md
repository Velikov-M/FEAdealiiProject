# Polycrystal cooling model (first mechanically meaningful model)

Cooling of a periodic RVE of randomly oriented graphite domains. Thermal-expansion mismatch
between misoriented anisotropic grains is the classic microcracking mechanism in polycrystalline
graphite, so cooling alone produces a non-trivial stress field with no applied load at all.

**Status: running, on the two simpler boundary conditions.** The MMS machinery is gone, the damage
solver from `../uniaxialDamageTest/` is in, grains carry random c-axis orientations, and both
`zeroDisplacement` and `freeSurface` run end to end in ~2 s for 10 cooling steps at 1 element/grain.
The periodic formulation is not implemented: `applyBoundaryConstraints()`,
`assemble_macroscopic_blocks()` and `solve_with_zero_mean_stress()` carry templates for it and throw
if selected. See `homogenization.md` for the derivation they must implement.

## Running it

```
cd build && cmake .. && make && ./main
```

`build/` holds symlinks to `input.json`, the mesh and `groupFile`, so there is exactly one copy of
each and editing the tracked `input.json` takes effect immediately. (An earlier real *copy* of
`input.json` in `build/` silently shadowed the tracked one, which looks exactly like edits being
ignored -- hence the symlink.)

Switch boundary condition family with `BoundaryConditions/Type`; switch the reference/threaded
assembly paths with `ParallelizationParameters/UseWorkStream`. The two must agree to the last
digit, and that is worth re-checking after any assembly change.

## Setup

- Same 10x10x10 hexahedral mesh; **each original cell is one domain** (1000 grains).
- Each domain gets its own orientation and thermo-mechanical properties.
- Orientations initially **fully random**; properties initially **ideal graphite crystallite**.
- **Periodic** boundary conditions (the mesh is periodic, so face pairs match exactly).
- Cooling from 2000 K to 1800 K in 10 steps.

## Design decisions and the reasoning behind them

### Boundary conditions, in the order they are being built

`BoundaryConditions/Type` selects one of three, sharing every other code path:

| Type | What it is | `<sigma>` | CG iters (1 elem/grain) |
|---|---|---|---|
| `zeroDisplacement` | `u = 0` on the whole boundary | ~4.6e8 Pa at `dT = -200 K` | 19 |
| `freeSurface` | traction-free + minimal 3-2-1 support | ~0.02 Pa (i.e. zero) | 174 |
| `periodic` | `u = Ebar.x + u_per`, `<sigma>=0` | 0 by construction | not implemented |

`zeroDisplacement` is a machinery test, not physics: clamping forbids macroscopic contraction, so
it develops a uniform triaxial tension of tens of MPa. That is visible in the run — the tension
fraction is exactly 1.000 and `<sigma>` grows linearly with `|dT|`, matching the `3*K*alpha*|dT|`
estimate that motivated this whole formulation.

`freeSurface` is a genuine free-cooling state: traction-free boundaries give `<sigma> = 0`
identically by the average stress theorem, and the code measures `|<sigma>|` at ~1e-9 of the local
stress magnitude, which is solver zero. **That is a real correctness check on the formulation, not
just a diagnostic.** What it is not is a *periodic* state — boundary grains relax against a real
free surface, so there is a surface layer with no RVE character.

The measured cost of the floppy support is a 9x jump in CG iterations (19 -> 174), the "floppy BC"
regime already seen in `../uniaxialDamageTest/`. Solve time goes from 2% to 32% of the run.

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

**Superposition does not break because damage exists — only if damage changes *during* the solve.**
The scheme here is staggered (damage frozen while equilibrium is solved, then the two iterated), so
each inner pass is exactly linear in `(u_per, Ebar)` and the 7-solve is exact per pass. Only the
number of times it runs changes. See `homogenization.md` section 7; that section also gives the
reason to *stay* staggered, namely that a monolithic damage tangent is non-symmetric and would cost
us CG.

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

## Visualisation

`./main` writes `polycrystalCooling-<n>.vtu` plus **`polycrystalCooling.pvd`**. Open the `.pvd` --
it carries the whole cooling history as a ParaView time series, with the animation clock showing
*accumulated cooling in K* (0 -> 200). Temperature itself decreases, and ParaView orders a series by
increasing time, so using T directly would run the animation backwards.

Use `freeSurface` for anything meant to be looked at. `zeroDisplacement` clamps the boundary and
buries every field under a uniform triaxial offset.

### What is in the file, and why

| array | frame | meaning |
|---|---|---|
| `displacement` | global | vector; for *Warp By Vector* |
| `c_axis` | — | unit normal to the basal plane, per grain; vector, for glyphs |
| `sigma_caxis_normal_local` | **crystal** | traction normal to the basal planes: **the damage driver** |
| `sigma_basal_shear_local` | **crystal** | `sqrt(s13^2 + s23^2)`; the mode the kinetic law does *not* read |
| `von_mises`, `pressure` | invariant | same in any frame; `pressure` positive in compression |
| `damage_driving_ratio` | — | `sigma_caxis_normal / threshold`; **> 1 means damage is growing now** |
| `damage` | — | omega, the consequence |
| `sigma_global_*` | global | six components, only if `OutputParameters/WriteFullStressTensor` |

On **local vs global**: sigma is one tensor, so given the orientation (also written) the two frames
are redundant *information*. They are not redundant in *practice*, because ParaView cannot apply a
per-cell rotation -- anything wanted in the crystal frame has to be computed in C++. Hence the file
carries frame-independent invariants (nothing to get wrong), the specific local components that
carry physics, and the full global tensor only on request. Every name states its frame, because
"sigma_zz" on its own is genuinely ambiguous here.

**Read the c-axis normal stress correctly.** It is not a component along one fixed direction --
every grain's c points somewhere different. It is well defined (traction normal to *that grain's*
basal planes) and it is exactly what drives damage, so colouring by it is right. But label it
"c-axis normal stress", never "sigma_zz".

### ParaView recipe

Colour + orientation arrows in one view:

1. Open `polycrystalCooling.pvd`, Apply.
2. Colour by `damage_driving_ratio`, rescale to `[0, 1.2]`, diverging colour map centred on 1.0 --
   then warm colours are exactly the grains whose damage is growing.
3. For the c-axis field: **Point Data to Cell Data** -> **Cell Centers** -> **Glyph**.
   Set *Glyph Type* = **Line**, *Orientation Array* = `c_axis`, *Scale Array* = `No scale array`,
   *Scale Factor* ~ 0.08 (about one cell), *Glyph Mode* = All Points.
   The three filters are needed because `c_axis` is written at the 8 vertices of every cell, so
   glyphing the raw points would give 8 glyphs per grain instead of one at its centre.
4. Play the animation to watch stress build, cross the threshold, and damage follow.

**Use a Line glyph, not an Arrow.** The c-axis is a *director*: `c` and `-c` describe the same
crystal, and the sign stored in the file is an artefact of how the direction was sampled. Arrows
would imply a polarity the material does not have.

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
