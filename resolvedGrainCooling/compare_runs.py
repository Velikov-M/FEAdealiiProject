#!/usr/bin/env python3
"""Tabulate a finished sweep: what changes with boundary condition, and what changes with
grain resolution.  Reads results/<study>/summary.json plus each run's macroscopic history."""
import json, pathlib, sys, csv, io, statistics

ROOT = pathlib.Path(__file__).resolve().parent


def load_history(run_dir):
    hits = list(run_dir.glob("*_macroscopic_history.csv"))
    if not hits:
        return []
    text = "".join(l for l in hits[0].read_text().splitlines(keepends=True)
                   if not l.startswith("#"))
    return list(csv.DictReader(io.StringIO(text)))


def fmt(v, w=10, p=3):
    return f"{v:{w}.{p}g}" if isinstance(v, (int, float)) else f"{str(v):>{w}}"


def bc_table(study):
    rows = json.loads((study / "summary.json").read_text())
    by = {(r["bc"], r["cellsPerGrain"]): r for r in rows}
    bcs = ["zeroDisplacement", "freeSurface", "periodic"]
    cpgs = sorted({r["cellsPerGrain"] for r in rows})

    for metric, label, scale in [("maxDamage", "max damage", 1),
                                 ("meanDamage", "mean damage", 1),
                                 ("maxS33", "max sigma_33 [MPa]", 1e-6),
                                 ("minS33", "min sigma_33 [MPa]", 1e-6),
                                 ("overThreshold", "volume fraction over threshold", 1),
                                 ("meanStressNorm", "|<sigma>| [Pa]", 1),
                                 ("seconds", "wall time [s]", 1)]:
        print(f"\n{label}")
        print("  " + "".join(f"{('%d cpg'%c):>14}" for c in [""] + cpgs).replace("  cpg", "", 1))
        hdr = f"  {'':<18}" + "".join(f"{('%d elem/grain'%c):>16}" for c in cpgs)
        print(hdr)
        for bc in bcs:
            line = f"  {bc:<18}"
            for c in cpgs:
                r = by.get((bc, c), {})
                v = r.get(metric)
                line += f"{(fmt(v*scale, 14, 4) if isinstance(v,(int,float)) else fmt('-',14))}  "
            print(line)


def seed_table(study):
    rows = json.loads((study / "summary.json").read_text())
    print(f"\n{'seed':>10} {'max damage':>12} {'mean damage':>13} {'max s33 [MPa]':>15} "
          f"{'over thr':>10} {'alphaBar_33':>13}")
    for r in rows:
        ab = r.get("alphaBar", [None]*3)
        print(f"{r['seed']:>10} {fmt(r.get('maxDamage'),12,4)} {fmt(r.get('meanDamage'),13,4)} "
              f"{fmt((r.get('maxS33') or 0)*1e-6,15,5)} {fmt(r.get('overThreshold'),10,3)} "
              f"{fmt(ab[2] if ab and ab[2] is not None else None,13,4)}")
    print()
    for key, label, scale in [("maxDamage","max damage",1), ("meanDamage","mean damage",1),
                              ("maxS33","max sigma_33 [MPa]",1e-6),
                              ("overThreshold","fraction over threshold",1)]:
        vals = [r[key]*scale for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            continue
        m, sd = statistics.mean(vals), statistics.stdev(vals)
        print(f"  {label:<24} mean {m:11.4g}   sd {sd:10.3g}   "
              f"CoV {100*sd/abs(m) if m else float('nan'):6.1f}%   "
              f"range [{min(vals):.4g}, {max(vals):.4g}]")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "bc"
    if which.startswith("bc"):
        bc_table(ROOT / "results" / "bc-resolution-sweep")
    else:
        seed_table(ROOT / "results" / "seed-sweep")
