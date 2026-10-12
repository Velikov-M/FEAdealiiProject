#!/usr/bin/env python3
"""Mosaic-spread study: one run per MosaicSpreadDeg, everything else from input.json.

All cases share ONE OrientationSeed on purpose. std::normal_distribution returns sigma * z with z
drawn from the same engine stream, so the cases are the SAME tilt pattern scaled by sigma -- the
comparison isolates the effect of the spread itself, and the stress-vs-sigma scaling exponent is
read off without realization noise. (Realization scatter is a separate question: vary the seed.)

Each run gets results/mosaic-spread/sig<s>deg_<N>cpg_seed<seed>/ with its exact input.json, the
console log, *_macroscopic_history.csv, *_stress_statistics.csv and the .vtu/.pvd field dumps
(gitignored). The tracked input.json is never written: main takes the run's own copy as argv[1].

Usage (from hopgCooling/):  python3 run_mosaic.py [--sigmas 0.25 0.5 1] [--tag mosaic-spread]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
ap.add_argument("--tag", default="mosaic-spread")
args = ap.parse_args()

base = json.loads((HERE / "input.json").read_text())
seed = base["BoundaryConditions"]["OrientationSeed"]["value"]
cpg = 8 ** int(base["MeshRefinementParameters"]["InitialGlobalRefinements"]["value"])
summary = {}
for sigma in args.sigmas:
    name = f"sig{sigma:g}deg_{cpg}cpg_seed{seed}"
    outdir = HERE / "results" / args.tag / name
    outdir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.vtu", "*.pvd", "*.csv", "*.log"):      # no stale outputs from older runs
        for f in outdir.glob(pattern):
            f.unlink()

    cfg = json.loads(json.dumps(base))
    cfg["BoundaryConditions"]["MosaicSpreadDeg"]["value"] = f"{sigma:g}"
    cfg["OutputParameters"]["OutputDirectory"]["value"] = str(outdir)
    cfg["OutputParameters"]["OutputFileNamePrefix"]["value"] = "hopg"
    run_cfg = outdir / "input.json"
    run_cfg.write_text(json.dumps(cfg, indent=1) + "\n")

    t0 = time.time()
    with open(outdir / "hopg.log", "w") as log:
        rc = subprocess.run([str(HERE / "build" / "main"), str(run_cfg)], cwd=HERE,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    summary[name] = {"sigma_deg": sigma, "returncode": rc, "wall_s": round(wall, 1)}
    print(f"{name}: exit {rc}, {wall:.0f} s", flush=True)
    (HERE / "results" / args.tag / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
