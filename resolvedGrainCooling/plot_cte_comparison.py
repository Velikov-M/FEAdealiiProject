#!/usr/bin/env python3
"""
Max damage vs cooling depth, with and without the alpha_33-damage coupling.

Reads the per-step macroscopic histories of two sweeps that differ ONLY in that coupling:
    results/bc-resolution-sweep/      alpha_33 independent of damage  (control)
    results/bc-resolution-sweep-cte/  alpha_33 = alpha_33(T) * (1 - omega)

Cooling depth is T_0 - T, so it increases left to right and the curves read as a loading history.
"""
import csv, io, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
CONTROL = ROOT / "results" / "bc-resolution-sweep"
COUPLED = ROOT / "results" / "bc-resolution-sweep-cte"
BCS = ["zeroDisplacement", "freeSurface", "periodic"]
CPGS = [1, 8, 64]


def history(study, bc, cpg):
    d = study / f"{bc}_{cpg:02d}cpg_seed20260905"
    hits = list(d.glob("*_macroscopic_history.csv"))
    if not hits:
        return None
    txt = "".join(l for l in hits[0].read_text().splitlines(keepends=True)
                  if not l.startswith("#"))
    return list(csv.DictReader(io.StringIO(txt)))


def series(rows, key):
    return ([float(r["cooling"]) for r in rows], [float(r[key]) for r in rows])


def main(metric="maxDamage", ylabel="max damage  $\\omega_{max}$"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    colour = {1: "#4C72B0", 8: "#DD8452", 64: "#55A868"}

    for ax, bc in zip(axes, BCS):
        for cpg in CPGS:
            for study, style, lab in ((CONTROL, dict(ls="--", alpha=0.55), "no CTE coupling"),
                                      (COUPLED, dict(ls="-",  lw=2.0),     "with CTE coupling")):
                rows = history(study, bc, cpg)
                if not rows:
                    continue
                x, y = series(rows, metric)
                ax.plot(x, y, color=colour[cpg], marker="o", ms=3, **style)
        ax.set_title(bc, fontsize=11)
        ax.set_xlabel("cooling depth  $T_0 - T$  [K]")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(ylabel)

    # Two legends: colour encodes resolution, line style encodes the model.
    from matplotlib.lines import Line2D
    res = [Line2D([], [], color=colour[c], lw=2, label=f"{c} elem/grain") for c in CPGS]
    mod = [Line2D([], [], color="0.3", ls="--", alpha=0.6, label="no CTE coupling"),
           Line2D([], [], color="0.3", ls="-", lw=2, label="with CTE coupling")]
    axes[0].legend(handles=res, fontsize=8, loc="upper left", frameon=False)
    axes[1].legend(handles=mod, fontsize=8, loc="upper left", frameon=False)

    fig.suptitle("Damage growth during cooling: effect of coupling $\\alpha_{33}$ to damage",
                 fontsize=12)
    fig.tight_layout()
    out = ROOT / f"results/cte_comparison_{metric}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "maxDamage"
    labels = {"maxDamage": "max damage  $\\omega_{max}$",
              "meanDamage": "mean damage  $\\langle\\omega\\rangle$",
              "alphaBar_33": "effective CTE  $\\bar\\alpha_{33}$  [1/K]"}
    main(m, labels.get(m, m))
