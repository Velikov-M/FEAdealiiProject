#!/usr/bin/env python3
"""
Drive a series of polycrystal cooling runs and put each one's output in its own folder.

Two studies:
  --bc-sweep    3 boundary conditions x 3 grain resolutions, one fixed seed.
                Answers: how much does the boundary treatment change the answer, and how
                much does resolving a grain with 8 or 64 elements change it?
  --seed-sweep  periodic BCs at 8 elements/grain, many seeds.
                Answers: how much of any given number is the realization rather than the model?

Each run gets   results/<study>/<name>/   containing the .pvd + .vtu series, the macroscopic
history CSV, a copy of the exact input.json used, and the console log.  Nothing is shared
between runs, so a later run can never overwrite an earlier one's results.
"""
import argparse, json, collections, pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build"
BASE_INPUT = ROOT / "input.json"

# InitialGlobalRefinements -> elements per grain (hexahedra split eightfold per level)
RESOLUTIONS = {0: 1, 1: 8, 2: 64}


def load_base():
    with open(BASE_INPUT) as f:
        return json.load(f, object_pairs_hook=collections.OrderedDict)


def set_param(cfg, section, key, value):
    """`section` may be a single name or a tuple of nested names."""
    if isinstance(section, str):
        section = (section,)
    node = cfg
    for part in section:
        node = node[part]
    node[key]["value"] = str(value)


def run_one(name, outdir, overrides, dry_run=False):
    cfg = load_base()
    for (section, key), value in overrides.items():
        set_param(cfg, section, key, value)
    # Every run writes into its own directory, under its own prefix.
    set_param(cfg, "OutputParameters", "OutputDirectory", str(outdir))
    set_param(cfg, "OutputParameters", "OutputFileNamePrefix", name)

    outdir.mkdir(parents=True, exist_ok=True)
    # The binary reads ./input.json relative to its own working directory (build/).
    with open(BUILD / "input.json", "w") as f:
        json.dump(cfg, f, indent=1)
    shutil.copy(BUILD / "input.json", outdir / "input.json")

    if dry_run:
        print(f"  [dry run] {name}")
        return 0.0, None

    log = outdir / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run([str(BUILD / "main")], cwd=BUILD, stdout=lf,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  !! {name} FAILED (exit {proc.returncode}); see {log}")
        return elapsed, None
    return elapsed, log


def summarize(log):
    """Pull the headline numbers out of a run's console log."""
    if log is None or not log.exists():
        return {}
    out, text = {}, log.read_text()
    for line in text.splitlines():
        if "damage: mean=" in line:
            out["meanDamage"] = float(line.split("mean=")[1].split(",")[0])
            out["maxDamage"] = float(line.split("max=")[1])
        elif "sigma_33 (local): min=" in line:
            out["minS33"] = float(line.split("min=")[1].split(",")[0])
            out["maxS33"] = float(line.split("max=")[1].split(",")[0])
            out["overThreshold"] = float(line.split("above threshold=")[1])
        elif "alphaBar (diag) =" in line:
            out["alphaBar"] = [float(v) for v in line.split("alphaBar (diag) =")[1].split(",")]
        elif "<sigma> norm=" in line:
            out["meanStressNorm"] = float(line.split("norm=")[1].split(" ")[0])
    out["damageWarning"] = "WARNING: maximum damage" in text
    return out


def bc_sweep(seed, dry_run):
    study = ROOT / "results" / "bc-resolution-sweep"
    rows = []
    for bc in ("zeroDisplacement", "freeSurface", "periodic"):
        for refin, cpg in RESOLUTIONS.items():
            name = f"{bc}_{cpg:02d}cpg_seed{seed}"
            print(f"running {name} ...", flush=True)
            overrides = {
                ("BoundaryConditions", "Type"): bc,
                ("BoundaryConditions", "OrientationSeed"): seed,
                ("MeshRefinementParameters", "InitialGlobalRefinements"): refin,
                ("MeshRefinementParameters", "NumberOfRefinementCycles"): 1,
            }
            secs, log = run_one(name, study / name, overrides, dry_run)
            row = {"bc": bc, "cellsPerGrain": cpg, "seconds": round(secs, 1)}
            row.update(summarize(log))
            rows.append(row)
            print(f"  {secs:7.1f} s   maxDamage={row.get('maxDamage','-')}")
    (study / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {study/'summary.json'}")


def seed_sweep(seeds, dry_run):
    study = ROOT / "results" / "seed-sweep"
    rows = []
    for seed in seeds:
        name = f"periodic_08cpg_seed{seed}"
        print(f"running {name} ...", flush=True)
        overrides = {
            ("BoundaryConditions", "Type"): "periodic",
            ("BoundaryConditions", "OrientationSeed"): seed,
            ("MeshRefinementParameters", "InitialGlobalRefinements"): 1,
            ("MeshRefinementParameters", "NumberOfRefinementCycles"): 1,
        }
        secs, log = run_one(name, study / name, overrides, dry_run)
        row = {"seed": seed, "seconds": round(secs, 1)}
        row.update(summarize(log))
        rows.append(row)
        print(f"  {secs:7.1f} s   maxDamage={row.get('maxDamage','-')}")
    (study / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {study/'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-sweep", action="store_true")
    ap.add_argument("--seed-sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260905, 11, 22, 33, 44, 55, 66, 77])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.bc_sweep or a.seed_sweep):
        ap.error("pick --bc-sweep and/or --seed-sweep")
    if a.bc_sweep:
        bc_sweep(a.seed, a.dry_run)
    if a.seed_sweep:
        seed_sweep(a.seeds, a.dry_run)
