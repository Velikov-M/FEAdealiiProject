# Scientific FEA development policy

## Scope and change control

This policy applies to new code and to refactoring of the current production models. The modular resolved-grain implementation is the reference application; HOPG remains an independent fork until a separately approved task adopts shared infrastructure.

`AGENTS.md`, `docs/model.md`, and `docs/verification.md` define scientific constraints and verification scope. This document defines implementation and documentation practice; it does not authorize scientific changes.

Classify each task before editing:

- **Behavior-preserving engineering:** code motion, interface clarification, comment/documentation work, and dead-code removal confirmed by inspection.
- **Numerical or scientific change:** any change to equations, material data, parameters, seeds, constraints, solver settings, tolerances, acceleration, scheduling that affects floating-point reduction, or output schemas.

Numerical and scientific changes require separate approval and verification appropriate to their scope. Do not disguise them as refactoring.

## Architecture and abstraction levels

A high-level method should reveal the sequence of operations without exposing low-level implementation details. Each statement in an orchestration method should operate at the same abstraction level.

For example, `ElasticProblem::run()` is restricted to initialization, microstructure setup, refinement cycles, and final output. Mesh parsing, orientation sampling, material checks, cell assembly, output formatting, and damage storage updates belong in lower-level helpers.

Resolved-grain module responsibilities are:

| Path | Responsibility |
|---|---|
| `main.cpp` | Process initialization, parameter-file selection, exception boundary |
| `parameters.*` | Parameter schema, defaults, validation patterns, help text |
| `material.*` | Local constitutive tensors and physical constitutive checks |
| `orientations.h` | Orientation construction and sampling |
| `quadrature_data.h` | Quadrature-point state and pack/unpack representation |
| `quadrature_transfer.*` | Discontinuous Gauss-point damage transfer through mesh adaptation |
| `assembly_data.h` | WorkStream scratch/copy ownership and reusable storage |
| `cell_assembly.*` | Shared production/MMS cell integration at fixed constitutive data |
| `damage_law.*` | Standalone scalar backward-Euler damage residual and root update |
| `convergence.h` | Zero-safe convergence scaling |
| `setup.cpp` | Mesh/grain setup, constraints, caches, refinement-cycle preparation |
| `assembly.cpp` | Mechanical cell assembly in serial and WorkStream paths |
| `solvers.cpp` | Linear solver preparation and individual solves |
| `damage.cpp` | Damage prediction, relaxation, convergence, converged-state commitment |
| `periodic.cpp` | Periodic condensation and homogenized response |
| `output.cpp` | Field/scalar output and converged-state recording |
| `run.cpp` | Run, refinement-cycle, temperature-step, and coupling orchestration |

Keep these boundaries when adding features. Do not create a helper for every few lines: extract when a coherent operation, ownership contract, or abstraction level is clarified. Preserve operation order during code motion, especially constraint construction, random-number consumption, refinement, assembly, and output.

Do not introduce global mutable solver state. Document references whose lifetime must exceed an object's use. Keep serial reference implementations when they provide independent A/B coverage of threaded paths.

Sharing code with HOPG is an architectural task, not an incidental consequence of editing resolved-grain helpers.

## Naming and scientific notation

- Use lower camel case for new methods (`initializeMicrostructure`) and descriptive lower snake case for existing local data where already established. Do not mass-rename legacy interfaces as part of an unrelated refactor.
- Prefer physical names over abbreviations when the meaning is not established by the model documents.
- Use SI units internally: m, Pa, K, and dimensionless strain/damage.
- `11`, `22`, `33` are local crystal axes; `xx`, `yy`, `zz` are global axes. Local axis 3 is the c-axis.
- Rotation matrices map global to local coordinates; local basis vectors are rows.
- Constitutive matrices use Kelvin notation. Shear diagonal entries are `2*C44`, `2*C44`, and `2*C66`.
- CTE values are secant coefficients referenced to the stress-free initial temperature.
- Damage is driven by local tensile `sigma_33`, is cooling-only and non-decreasing, and is capped numerically at `0.999`.

Do not weaken these conventions through ambiguous variable names or undocumented tensor conversions.

## Doxygen documentation

Document at the declaration, normally in a header:

- every class and data structure declared in a header, every free function, and every method declaration, including private methods;
- every function parameter when parameters are documented, using `@param[in]`, `@param[out]`, or `@param[in,out]` as appropriate;
- every non-void return value using `@return`;
- physical units and coordinate frames for physical quantities;
- preconditions, postconditions, state mutations, ownership, lifetime, and side effects when they are not obvious;
- non-obvious member variables and data-structure invariants.

A minimal method comment is:

```cpp
/** Compute the current state's volume-averaged stress.
 * @param[in] temperature Uniform temperature in K.
 * @return Cauchy stress in Pa in global coordinates.
 */
SymmetricTensor<2, dim> computeAverageStress(double temperature) const;
```

Use concise Doxygen comments for contracts, not line-by-line narration. Do not repeat the same API contract at both declaration and definition. Undocumented-member compiler-style warnings remain disabled to avoid requiring annotations on trivial data members; complete function documentation is enforced by review. Malformed Doxygen and parameter-name mismatches are errors.

Implementation comments use `//` and explain only local reasoning that cannot be expressed clearly by the code and API documentation: numerical bracketing, constraint ordering, thread ownership, or a non-obvious deal.II contract. Do not record old bugs, superseded implementations, timing anecdotes, or task history in source comments. Dated evidence belongs in `docs/studies.md` or an archived result.

Long equations and derivations belong in `docs/model.md` or the relevant derivation document, with a focused reference from the API comment.

Build the resolved-grain documentation after documentation changes:

```bash
cd resolvedGrainCooling
cmake --build build --target docs
```

## Verification and performance work

Run the unified mathematical/technical correctness gate for new numerical logic:

```bash
./scripts/verify.sh check
```

It must return nonzero on any required build, test, timeout, or prerequisite failure. Keep unit tests linked to `resolved_grain`; do not add copied-source oracles that can drift from production. Add independent SymPy references to `tests/symbolic/generate_oracles.py` rather than manually transcribing symbolic results. Test-generated files belong in build or verification-run directories.

For a resolved-grain refactor, additionally build the affected target and run the nine-case A/B matrix against an accepted reference:

```bash
./scripts/verify.sh build-resolved ab-resolved \
  --reference-run verification-runs/<accepted-reference-run>
```

Inspect the summary, manifest, commands, and logs. CSV values, integer histories, convergence histories, PVD sequences, and VTU contents must meet the comparator's strict checks. Only the VTU generation timestamp is normalized. Never weaken scientific inputs, convergence criteria, or comparison tolerances to obtain a pass.

Performance changes additionally require recorded timing evidence and a representative resource limit. Keep benchmark results dated and configuration-specific; do not turn them into permanent source commentary or universal solver rankings.

MMS and uniaxial programs are separate source copies. Their passing tests do not establish coverage of later production-fork edits.

## Completion checklist

Before considering engineering work complete:

1. Review the complete diff and confirm that the approved scope was not expanded.
2. Build affected executables and generate documentation when comments changed.
3. Run the applicable deterministic A/B verification and inspect numerical evidence.
4. Confirm tracked inputs and archived results were not modified.
5. Update current-state documentation for changed structure, policy, commands, or status.
6. Report remaining failures, unverified cases, scientific limitations, and deliberately preserved legacy behavior.
