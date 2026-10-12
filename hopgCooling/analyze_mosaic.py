#!/usr/bin/env python3
"""Analyse the mosaic-spread study: stress levels vs cooling, and how they scale with sigma.

Reads results/<tag>/sig*deg_*/hopg_stress_statistics.csv and writes into results/<tag>/:
  stress_vs_cooling.png   p01/p99 band of each invariant stress measure vs cooling, per sigma
  scaling.png             RMS at the final step vs sigma, log-log, with the fitted exponent
  summary.md              final-step table + exponents

The scaling exponent is clean because all runs share one seed: the tilt pattern is identical and
only scaled by sigma (see run_mosaic.py). A small-tilt expansion predicts the exponents: the
eigenstrain mismatch between neighbouring crystallites is (alpha_c - alpha_a) dT * n(x)n, and a
tilt theta changes n(x)n by O(theta) in the shear (13, 23) entries but only O(theta^2) in the
normal ones -- so basal shear should go as sigma^1 and the normal stresses as sigma^2, unless the
O(theta) shear mismatch couples into the normal stresses through the anisotropic stiffness.

Usage (from hopgCooling/):  python3 analyze_mosaic.py [--tag mosaic-spread]
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
MEASURES = [
    ("inPlaneMax", "in-plane principal, max"),
    ("inPlaneMin", "in-plane principal, min"),
    ("sigma33", "sigma_33 (across basal plane)"),
    ("tauBasal", "basal shear tau"),
]
# Ordinal single-hue ramp (light -> dark = small -> large spread), plus a line style per series
# so identity never rests on colour alone.
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
STYLES = [":", "--", "-"]


def load(path):
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="mosaic-spread")
args = ap.parse_args()
root = HERE / "results" / args.tag

runs = []
for d in sorted(root.glob("sig*deg_*")):
    f = d / "hopg_stress_statistics.csv"
    if f.exists():
        runs.append((float(re.match(r"sig([\d.]+)deg", d.name).group(1)), d.name, load(f)))
runs.sort()
assert runs, f"no completed runs under {root}"

# --- stress vs cooling ------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (key, title) in zip(axes.flat, MEASURES):
    for (sigma, _, d), color, ls in zip(runs, RAMP, STYLES):
        cool = d["cooling"]
        ax.fill_between(cool, d[f"{key}_p01"], d[f"{key}_p99"], color=color, alpha=0.18, lw=0)
        ax.plot(cool, d[f"{key}_p99"], ls, color=color, lw=2, label=f"sigma = {sigma:g} deg")
        ax.plot(cool, d[f"{key}_p01"], ls, color=color, lw=2)
        ax.annotate(f"{sigma:g} deg", (cool[-1], d[f"{key}_p99"][-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#444444")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("stress [MPa], 1st-99th percentile")
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("cooling from 2500 K [K]")
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("HOPG mosaic, 1000 crystallites x 64 elements, periodic, no damage", fontsize=11)
fig.tight_layout()
fig.savefig(root / "stress_vs_cooling.png", dpi=140)

# --- scaling with sigma -----------------------------------------------------------------------
sig = np.array([r[0] for r in runs])
exponents = {}
fig, ax = plt.subplots(figsize=(6.5, 4.8))
for (key, title), marker in zip(MEASURES + [("vonMises", "von Mises")], "osD^v"):
    rms = np.array([r[2][f"{key}_rms"][-1] for r in runs])
    if len(runs) >= 2:
        exponents[key] = float(np.polyfit(np.log(sig), np.log(rms), 1)[0])
    ax.loglog(sig, rms, marker + "-", ms=8, lw=1.5,
              label=f"{title}" + (f"  (slope {exponents[key]:.2f})" if key in exponents else ""))
ax.set_xlabel("mosaic spread sigma [deg] (std per tilt component)")
ax.set_ylabel("RMS stress at 1000 K [MPa]")
ax.grid(alpha=0.3, which="both", lw=0.5)
ax.legend(fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(root / "scaling.png", dpi=140)

# --- table ------------------------------------------------------------------------------------
lines = ["| sigma [deg] | FWHM equiv. [deg] | measure | p01 | p99 | RMS | min | max |",
         "|---|---|---|---|---|---|---|---|"]
for sigma, name, d in runs:
    for key, title in MEASURES + [("vonMises", "von Mises")]:
        v = {s: d[f"{key}_{s}"][-1] for s in ("p01", "p99", "rms", "min", "max")}
        lines.append(f"| {sigma:g} | {2.3548 * sigma:.2f} | {title} | {v['p01']:+.2f} | {v['p99']:+.2f} | "
                     f"{v['rms']:.2f} | {v['min']:+.2f} | {v['max']:+.2f} |")
lines += ["", "RMS-vs-sigma exponents (log-log fit over the runs above, final step):", ""]
lines += [f"- {k}: {e:.3f}" for k, e in exponents.items()]
(root / "summary.md").write_text(
    "# Mosaic-spread study -- final step (T = 1000 K), stresses in MPa\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
