#!/usr/bin/env python3
"""
Verification plots for the zero-damage thermo-elastic MMS.

What this test checks, and why it is built the way it is:

  * The manufactured solution verifies assemble_system() -- BOTH the stiffness operator
    and the thermal (CTE) load vector.

  * The temperature field varies in space on purpose. With a uniform temperature, a
    homogeneous material and Dirichlet data everywhere, the thermal term is the integral
    of eps(v):C:alpha*dT; with C, alpha and dT all constant that integrand is a constant
    tensor, so the integral collapses to a boundary term that vanishes for every test
    function with zero boundary values. Equivalently div(C:alpha*dT) = 0, so alpha drops
    out of the PDE entirely and the CTE cannot be tested at all.

  * The manufactured displacement amplitude is small (2.382e-4) on purpose too. Getting
    alpha into the equations is not the same as testing it: at unit amplitude the elastic
    load outweighs the thermal one by ~4000x, and doubling the CTE in the assembly moved
    the L2 error by only ~1% while the convergence rate still looked perfect. The
    amplitude is set to the ratio of the two loads so both are genuinely exercised.

    Verified by mutation testing, not by inspection -- with the CTE wrong by 1%,
    convergence stalls outright (rate 0.21 then 0.01) instead of showing 2.00.

Usage
-----
    python3 plot_mms_verification.py            # plot from mms_convergence_data.json
    python3 plot_mms_verification.py --run      # re-run the FEA sweep first, then plot
"""

import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "mms_convergence_data.json")

# --- style -------------------------------------------------------------------
# Panel A: three distinct error norms -> categorical slots 1-3 (the documented palette's
# first three validate on the all-pairs list in both light and dark).
NORM_COLORS = {"L2": "#2a78d6", "H1": "#eb6834", "Linfty": "#1baf7a"}
# Panel C: mesh level is an ORDERED quantity -> one-hue ordinal ramp (blue steps
# 250/450/700), checked for monotone lightness, adjacent dL >= 0.06, light-end contrast
# >= 2:1 and single hue.
MESH_RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def load_params():
    cfg = json.load(open(os.path.join(HERE, "input.json")))
    M = cfg["ModelParameters"]
    return dict(L=float(M["LengthOfTheBody"]["value"]),
                T_0=float(M["InitialTemperature"]["value"]),
                T_end=float(M["FinalTemperature"]["value"]),
                U0=float(M["ManufacturedDisplacementAmplitude"]["value"]))


def run_sweep():
    """Run the MMS once; every refinement cycle and temperature step comes from that run."""
    build = os.path.join(HERE, "build")
    if not os.path.exists(os.path.join(build, "main")):
        raise SystemExit("build/main not found -- build the project first")
    base = json.load(open(os.path.join(HERE, "input.json")),
                     object_pairs_hook=collections.OrderedDict)
    with tempfile.TemporaryDirectory() as work:
        for f in ("main", "cube10cellsPerSide.msh", "groupFile"):
            shutil.copy(os.path.join(build, f), work)
        cfg = json.loads(json.dumps(base), object_pairs_hook=collections.OrderedDict)
        cfg["OutputParameters"]["OutputFrequency"]["value"] = "1000000"  # no field dumps
        json.dump(cfg, open(os.path.join(work, "input.json"), "w"), indent=4)
        proc = subprocess.run(["./main"], cwd=work, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise SystemExit(f"FEA run failed:\n{proc.stdout[-2000:]}")
        out = proc.stdout

    rows, cur, cells, dofs = {}, None, None, None
    for line in out.splitlines():
        m = re.search(r"Refinement cycle (\d+), temperature step (\d+), iteration (\d+)", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"Cells: (\d+), DoFs: (\d+)", line)
        if m:
            cells, dofs = int(m.group(1)), int(m.group(2))
            continue
        m = re.search(r"L2=([\d.eE+-]+), H1=([\d.eE+-]+), Linfty=([\d.eE+-]+)", line)
        if m and cur is not None:
            # later iterations overwrite earlier ones -> keeps the converged value
            rows[cur] = dict(cells=cells, dofs=dofs, L2=float(m.group(1)),
                             H1=float(m.group(2)), Linfty=float(m.group(3)))

    timing = {}
    for line in out.splitlines():
        m = re.search(r"\|\s*(Assemble system|Solve system|Setup of system)\s*\|\s*(\d+)\s*\|\s*([\d.]+)s", line)
        if m:
            timing[m.group(1)] = dict(calls=int(m.group(2)), seconds=float(m.group(3)))
        m = re.search(r"Total wallclock time elapsed since start\s*\|\s*([\d.]+)s", line)
        if m:
            timing["total"] = dict(seconds=float(m.group(1)))
    return {f"{c}|{s}": v for (c, s), v in rows.items()}, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="re-run the FEA before plotting")
    args = ap.parse_args()

    if args.run:
        print("Running MMS sweep (all refinement cycles)...")
        rows, timing = run_sweep()
        json.dump({"rows": rows, "timing": timing}, open(DATA_FILE, "w"), indent=2)
        print(f"  saved -> {DATA_FILE}")
    else:
        if not os.path.exists(DATA_FILE):
            raise SystemExit(f"{DATA_FILE} not found -- run with --run first")
        blob = json.load(open(DATA_FILE))
        rows, timing = blob["rows"], blob["timing"]

    p = load_params()
    parsed = {}
    for k, v in rows.items():
        cy, st = (int(x) for x in k.split("|"))
        parsed[(cy, st)] = v
    cycles = sorted({c for c, _ in parsed})
    last_step = max(s for _, s in parsed)

    # per-cycle values at the final temperature step
    cells = [parsed[(c, last_step)]["cells"] for c in cycles]
    h = [p["L"] / n ** (1.0 / 3.0) for n in cells]
    errs = {k: [parsed[(c, last_step)][k] for c in cycles] for k in ("L2", "H1", "Linfty")}

    fig = plt.figure(figsize=(10.0, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.28)
    ax_conv = fig.add_subplot(gs[:, 0])
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_step = fig.add_subplot(gs[1, 1])

    # --- A: convergence -------------------------------------------------------
    labels = {"L2": r"$L^2$", "H1": r"$H^1$ semi", "Linfty": r"$L^\infty$"}
    rates = {}
    for k in ("L2", "H1", "Linfty"):
        e = errs[k]
        rates[k] = math.log(e[0] / e[-1]) / math.log(h[0] / h[-1])
        ax_conv.plot(h, e, marker="o", ms=5, color=NORM_COLORS[k], mec="white", mew=0.7,
                     zorder=3, label=f"{labels[k]}   (rate {rates[k]:.2f})")

    # Guides are offset away from the data: an unoffset slope-2 guide lands exactly on the
    # Linfty curve and becomes invisible.
    for slope, ref_key, dy in ((2.0, "L2", 0.33), (1.0, "H1", 2.6)):
        guide = np.array(errs[ref_key][0]) * (np.array(h) / h[0]) ** slope * dy
        ax_conv.plot(h, guide, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax_conv.annotate(rf"slope {slope:g}", xy=(h[-1], guide[-1]),
                         xytext=(5, 0), textcoords="offset points",
                         color=INK_SOFT, fontsize=8, va="center")

    ax_conv.set_xscale("log"); ax_conv.set_yscale("log")
    ax_conv.invert_xaxis()
    ax_conv.set_xlabel(r"mesh size  $h = L\,/\,N_{\mathrm{cells}}^{1/3}$")
    ax_conv.set_ylabel("error vs manufactured solution")
    ax_conv.legend(loc="lower left", borderaxespad=0.4)
    # Cell counts sit just above the axes, not floating among the curves.
    for xi, ni in zip(h, cells):
        ax_conv.annotate(f"{ni} cells", xy=(xi, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=7, color=INK_MUTED)

    # --- B: the temperature rise field ---------------------------------------
    # Single-hue warm ramp, truncated away from white so the light end still reads.
    oranges = plt.get_cmap("Oranges")
    warm = matplotlib.colors.LinearSegmentedColormap.from_list(
        "warmTrunc", oranges(np.linspace(0.20, 0.92, 256)))
    xs = np.linspace(0, p["L"], 200); ys = np.linspace(0, p["L"], 200)
    X, Y = np.meshgrid(xs, ys)
    dT = 1.0 * (p["T_end"] - p["T_0"]) * (X / p["L"] + np.exp(-Y / p["L"]))  # at t = 1
    im = ax_temp.pcolormesh(X, Y, np.abs(dT), cmap=warm, shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax_temp, pad=0.03)
    cb.set_label(r"$|\Delta T|$  [K]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax_temp.set_xlabel("$x/L$"); ax_temp.set_ylabel("$y/L$")
    ax_temp.set_title(r"$\Delta T \propto x/L + e^{-y/L}$   (at $t=1$)",
                      fontsize=8, color=INK_SOFT, pad=5)
    ax_temp.set_aspect("equal")

    # --- C: error across the temperature steps -------------------------------
    steps = sorted({s for _, s in parsed})
    for cy, color in zip(cycles, MESH_RAMP):
        e = [parsed[(cy, s)]["L2"] for s in steps]
        ax_step.plot(steps, e, marker="o", ms=3.2, color=color, mec="white", mew=0.5,
                     label=f"{parsed[(cy, last_step)]['cells']} cells")
    ax_step.set_yscale("log")
    ax_step.set_xlabel("temperature step")
    ax_step.set_ylabel(r"$L^2$ error")
    ax_step.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.text(0.008, 0.975, "Thermo-elastic MMS verification (zero damage)",
             fontsize=10, color=INK, va="top", weight="medium")
    fig.text(0.008, 0.932,
             "spatially varying temperature -- a uniform one makes the CTE term vanish identically "
             "and leaves it untested",
             fontsize=8, color=INK_SOFT, va="top")

    fig.subplots_adjust(left=0.075, right=0.87, top=0.835, bottom=0.125)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"mms_verification.{ext}")
        fig.savefig(out); print(f"wrote {out}")

    print("\nConvergence at the final temperature step:")
    print(f"  {'cells':>8} {'h':>8} " + " ".join(f"{k:>12}" for k in ("L2", "H1", "Linfty")))
    for i, c in enumerate(cycles):
        print(f"  {cells[i]:>8} {h[i]:>8.4f} " + " ".join(f"{errs[k][i]:>12.4e}" for k in ("L2", "H1", "Linfty")))
    print("  measured rates: " + ", ".join(f"{k}={rates[k]:.2f}" for k in ("L2", "H1", "Linfty")))
    print("  (expected for Q1: L2 2, H1 1, Linfty 2)")
    if timing:
        print("\nTiming for this run:")
        for k, v in timing.items():
            calls = f" over {v['calls']} calls" if "calls" in v else ""
            print(f"  {k:<18} {v['seconds']:>7.2f}s{calls}")


if __name__ == "__main__":
    main()
