#!/usr/bin/env python3
"""
Mesh-sizing study: how large a problem fits a given wallclock budget on this machine?

Method. Run the test at successively refined meshes and difference the wallclock between
runs to isolate each mesh's cost (a run with N refinement cycles re-solves every coarser
mesh too, and the deal.II timer aggregates over all of them, so differencing is what
separates them). Fit t_step = a*N^b and invert for the budget.

Measured 2026-08-24 on 6 cores / 12 threads, damage active, LinearSolverTolerance 1e-8,
WorkStream on:

     cells   s/temperature step      CG iterations (max)
      1000              0.20                    424
      8000              3.40                   1196
     64000             72.20                   3013

  t_step = 1.13e-5 * N^1.416     (SSOR-CG theory: N^4/3 = N^1.333)
  n_CG   ~ N^0.47                (theory: N^1/3 = N^0.333)

Both exponents run somewhat above theory. The likely reason is the material's extreme
elastic anisotropy (C11/C44 ~ 4240), which SSOR handles poorly; CG already needs 424
iterations on a mere 1000-cell mesh.

Two findings worth more than the extrapolation itself:

  * The ITERATION CAP, not wallclock, is what actually stops you. It was hardcoded at
    1000; 8000 cells already needs 1196, so refining past ~4000 cells failed outright
    with a convergence-failure exception rather than merely running slowly. Now exposed
    as SolverParameters/MaxLinearSolverIterations. Extrapolating, a 430k-cell mesh would
    need ~7400 iterations, so set the cap to ~10000 for production-sized runs.

  * Damage does NOT degrade the conditioning. CG iteration counts came out IDENTICAL
    (424 / 1196 / 3013) with damage active (final omega 0.68) and with damage suppressed
    entirely. Mesh refinement alone drives the count. This was worth checking because the
    opposite is often assumed -- softening reduces C33 by ~3x here and might plausibly
    have wrecked the spectrum, but it does not.

CAVEATS -- these are order-of-magnitude estimates, not guarantees:

  * The coupling iteration count (displacement <-> damage) is ~2 in this test and enters
    the cost linearly. A real problem needing 4-5 iterations per step scales the times up
    by that ratio, i.e. halves the affordable mesh at fixed budget.
  * This test's damage field is spatially uniform. A real problem with localised damage
    may need more coupling iterations, and possibly more CG iterations too.
  * Beyond 64000 cells the numbers are a fitted extrapolation, not measurements.
  * Assembly is charged twice per coupling iteration because checkConvergenceCriteria
    re-assembles to form its residual. That is included in the fit; removing it would buy
    maybe 10-15% at these sizes, since the solve dominates.
  * Memory is not the binding constraint: ~1 KB/DoF for the sparse matrix means ~12M DoFs
    would fit in 12 GB, far beyond what CG iteration counts allow.

Usage
-----
    python3 timing_study.py             # print the sizing table from the measured fit
    python3 timing_study.py --measure   # re-run the sweep (slow: ~7 min) and refit
"""

import argparse, collections, json, math, os, re, shutil, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = {1000: 0.20, 8000: 3.40, 64000: 72.20}   # seconds per temperature step
CG_ITERS = {1000: 424, 8000: 1196, 64000: 3013}


def measure(nsteps=5, tol=1e-8, cap=200000, max_cycles=3):
    build = os.path.join(HERE, "build")
    base = json.load(open(os.path.join(HERE, "input.json")),
                     object_pairs_hook=collections.OrderedDict)
    walls, iters = {}, {}
    with tempfile.TemporaryDirectory() as work:
        for f in ("main", "cube10cellsPerSide.msh", "groupFile"):
            shutil.copy(os.path.join(build, f), work)
        for k in range(1, max_cycles + 1):
            c = json.loads(json.dumps(base), object_pairs_hook=collections.OrderedDict)
            c['MeshRefinementParameters']['NumberOfRefinementCycles']['value'] = str(k)
            c['MeshRefinementParameters']['RefinementStrategy']['value'] = 'uniform'
            c['SolverParameters']['LinearSolverTolerance']['value'] = str(tol)
            c['SolverParameters']['MaxLinearSolverIterations']['value'] = str(cap)
            c['ModelParameters']['TemperatureIncrement']['value'] = str(1.0 / nsteps)
            c['OutputParameters']['OutputFrequency']['value'] = '1000000'
            json.dump(c, open(os.path.join(work, "input.json"), "w"), indent=4)
            t0 = time.time()
            p = subprocess.run(["./main"], cwd=work, capture_output=True, text=True, timeout=14400)
            walls[k] = time.time() - t0
            if p.returncode != 0:
                raise SystemExit(f"run failed at {k} cycles:\n{p.stdout[-1500:]}\n{p.stderr[-800:]}")
            cells = None
            for line in p.stdout.splitlines():
                m = re.search(r"Cells: (\d+),", line)
                if m: cells = int(m.group(1))
                m = re.search(r"CG iterations = (\d+)", line)
                if m and cells: iters[cells] = max(iters.get(cells, 0), int(m.group(1)))
            print(f"  {k} cycle(s): {walls[k]:.1f}s")
    meshes = sorted(iters)
    per = {}
    prev = 0.0
    for k, n in enumerate(meshes, start=1):
        per[n] = (walls[k] - prev) / nsteps
        prev = walls[k]
    return per, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="re-run the sweep instead of using stored numbers")
    a_ = ap.parse_args()
    per, iters = (measure() if a_.measure else (dict(MEASURED), dict(CG_ITERS)))

    ns = sorted(per)
    b = math.log(per[ns[-1]] / per[ns[0]]) / math.log(ns[-1] / ns[0])
    a = per[ns[0]] / ns[0] ** b

    print("\nMeasured cost per temperature step (damage active, tol 1e-8, WorkStream):")
    print(f"  {'cells':>9} {'s/step':>9} {'CG iters':>10}")
    for n in ns:
        print(f"  {n:>9,} {per[n]:>9.2f} {iters.get(n, 0):>10,}")
    print(f"\n  fit: t_step = {a:.3e} * N^{b:.3f}    (SSOR-CG theory: N^1.333)")

    print("\nLargest mesh fitting a wallclock budget:")
    print(f"  {'budget':>8} | {'10 temp steps':>26} | {'100 temp steps':>26}")
    print("  " + "-" * 68)
    for label, secs in (("10 min", 600), ("30 min", 1800), ("2 h", 7200), ("3 h", 10800)):
        cols = []
        for steps in (10, 100):
            N = (secs / steps / a) ** (1.0 / b)
            side = round(N ** (1 / 3))
            cols.append(f"{N:>9,.0f} cells (~{side}^3, {3*(side+1)**3:>9,.0f} DoF)")
        print(f"  {label:>8} | {cols[0]:>26} | {cols[1]:>26}")

    big = (10800 / 10 / a) ** (1.0 / b)
    n_big = iters[ns[-1]] * (big / ns[-1]) ** 0.47
    print(f"\n  At the 3h/10-step size (~{big:,.0f} cells) CG would need ~{n_big:,.0f} iterations,")
    print(f"  so MaxLinearSolverIterations must be raised to ~{round(n_big*1.4, -3):,.0f}.")
    print("  Past this, a better preconditioner (AMG) matters far more than more cores:")
    print("  threading already gives ~5.7x on the parts it covers, but the solve is now the")
    print("  bottleneck and SSOR-CG scales as N^1.4 regardless of thread count.")


if __name__ == "__main__":
    main()
