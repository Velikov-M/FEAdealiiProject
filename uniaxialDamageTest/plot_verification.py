#!/usr/bin/env python3
"""
Verification plots for the uniaxial displacement-controlled damage test.

Compares the FEA damage trajectory against an independent high-accuracy reference
solution of the same scalar ODE, and shows the backward-Euler convergence order.

The reference is NOT another FEA run: for this load case the problem reduces
exactly to a scalar ODE (see main.cpp's header for the derivation), which is
integrated here with scipy at tight tolerance. That independence is the point --
it is what makes this a verification rather than a self-consistency check.

Usage
-----
    python3 plot_verification.py            # plot from convergence_data.json
    python3 plot_verification.py --run      # re-run the FEA sweep first (slow), then plot

Requires build/main to exist when using --run.
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "convergence_data.json")
STEP_SIZES = [0.1, 0.05, 0.025, 0.0125, 0.00625]

# --- style -------------------------------------------------------------------
# Ordinal (ordered) ramp for step size -- a single blue hue, light->dark, rather
# than categorical hues: dT is an ordered quantity, so the ramp itself carries
# meaning. Steps 250/350/450/550/700 of the reference blue ramp; validated for
# monotone lightness, adjacent dL >= 0.06, light-end contrast >= 2:1, single hue.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def load_params():
    """Material and loading parameters, read from input.json so plot and run agree."""
    cfg = json.load(open(os.path.join(HERE, "input.json")))
    el = cfg["MaterialParameters"]["LinearElasticityParameters"]
    dp = cfg["MaterialParameters"]["DamageParameters"]
    mp = cfg["ModelParameters"]
    return dict(
        C11_0=float(el["C11_0"]["value"]), C12_0=float(el["C12_0"]["value"]),
        C13_0=float(el["C13_0"]["value"]), C33_0=float(el["C33_0"]["value"]),
        sigma_th=float(dp["StressThreshold"]["value"]),
        A=float(dp["a_kineticParam"]["value"]), m=float(dp["m_kineticParam"]["value"]),
        rate=float(mp["AppliedAxialStrainRate"]["value"]),
    )


def reference_solution(p):
    """Dense high-accuracy solution of the reduced scalar ODE.

    sigma_zz(omega, eps_zz) = eps_zz * [C33_0*(1-omega) - 2*C13_0^2/(C11_0+C12_0)]
    domega/ds = A * (macaulay(sigma_zz - sigma_th)/sigma_th)^m * (1/(1-omega))^m
    """
    transverse = 2.0 * p["C13_0"] ** 2 / (p["C11_0"] + p["C12_0"])

    def rhs(s, y):
        omega = y[0]
        sigma = p["rate"] * s * (p["C33_0"] * (1.0 - omega) - transverse)
        acting = max(sigma - p["sigma_th"], 0.0)
        return [p["A"] * (acting / p["sigma_th"]) ** p["m"] * (1.0 / (1.0 - omega)) ** p["m"]]

    # Tight tolerances: the reference must be far more accurate than anything it
    # is used to judge, otherwise the measured "error" is partly its own.
    return solve_ivp(rhs, [0.0, 1.0], [0.0], rtol=1e-12, atol=1e-14, dense_output=True)


def run_sweep():
    """Run the FEA at each step size, returning damage trajectories keyed by dT."""
    build = os.path.join(HERE, "build")
    if not os.path.exists(os.path.join(build, "main")):
        raise SystemExit("build/main not found -- build the project first")

    base = json.load(open(os.path.join(HERE, "input.json")),
                     object_pairs_hook=collections.OrderedDict)
    results = {}
    with tempfile.TemporaryDirectory() as work:
        for f in ("main", "cube10cellsPerSide.msh", "groupFile"):
            shutil.copy(os.path.join(build, f), work)

        for dT in STEP_SIZES:
            cfg = json.loads(json.dumps(base), object_pairs_hook=collections.OrderedDict)
            cfg["ModelParameters"]["TemperatureIncrement"]["value"] = repr(dT)
            # Suppress per-step field dumps; only the trajectory matters here.
            cfg["OutputParameters"]["OutputFrequency"]["value"] = "1000000"
            json.dump(cfg, open(os.path.join(work, "input.json"), "w"), indent=4)

            proc = subprocess.run(["./main"], cwd=work, capture_output=True,
                                  text=True, timeout=3600)
            # Keep the LAST displacement-damage iteration per step (the converged one).
            per_step, cur = {}, None
            for line in proc.stdout.splitlines():
                m = re.search(r"pseudo-time step (\d+), displacement-damage iteration", line)
                if m:
                    cur = int(m.group(1))
                    continue
                m = re.search(r"damage: mean=([-\d.eE+]+) spread=([-\d.eE+]+)", line)
                if m and cur is not None:
                    per_step[cur] = (float(m.group(1)), float(m.group(2)))

            if not per_step or "Aborting" in proc.stdout:
                raise SystemExit(f"FEA run failed at dT={dT}:\n{proc.stdout[-2000:]}")

            steps = sorted(per_step)
            results[str(dT)] = {
                "s": [n * dT for n in steps],
                "damage": [per_step[n][0] for n in steps],
                "spread": [per_step[n][1] for n in steps],
            }
            print(f"  dT={dT:<9} {len(steps):>4} steps  omega_final={per_step[steps[-1]][0]:.6f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="re-run the FEA sweep before plotting")
    args = ap.parse_args()

    if args.run:
        print("Running FEA sweep:")
        data = run_sweep()
        json.dump(data, open(DATA_FILE, "w"), indent=2)
        print(f"  saved -> {DATA_FILE}")
    else:
        if not os.path.exists(DATA_FILE):
            raise SystemExit(f"{DATA_FILE} not found -- run with --run first")
        data = json.load(open(DATA_FILE))

    params = load_params()
    ref = reference_solution(params)
    dts = sorted((float(k) for k in data), reverse=True)

    fig = plt.figure(figsize=(9.2, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[2.2, 1],
                          hspace=0.12, wspace=0.30)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[1, 0], sharex=ax_traj)
    ax_conv = fig.add_subplot(gs[:, 1])

    # --- trajectory ----------------------------------------------------------
    s_dense = np.linspace(0, 1, 600)
    ax_traj.plot(s_dense, ref.sol(s_dense)[0], color=INK, lw=1.4, zorder=3,
                 label="reference (scipy, rtol $10^{-12}$)")
    for dT, color in zip(dts, RAMP):
        d = data[str(dT)]
        s, omega = np.array(d["s"]), np.array(d["damage"])
        step = max(1, len(s) // 22)  # thin dense series so markers stay readable
        ax_traj.plot(s[::step], omega[::step], ls="none", marker="o", ms=4.0,
                     mfc=color, mec="white", mew=0.6, zorder=4,
                     label=rf"FEA  $\Delta s={dT:g}$")

    ax_traj.set_ylabel(r"damage  $\omega$")
    ax_traj.set_ylim(-0.03, 0.78)
    ax_traj.tick_params(labelbottom=False)
    # Upper left: omega is identically zero until activation, so that block is clear.
    ax_traj.legend(loc="upper left", ncol=1, handletextpad=0.5, borderaxespad=0.3,
                   labelspacing=0.35)

    eps_activate = params["sigma_th"] / (
        params["C33_0"] - 2 * params["C13_0"] ** 2 / (params["C11_0"] + params["C12_0"]))
    s_activate = eps_activate / params["rate"]
    for ax in (ax_traj, ax_resid):
        ax.axvline(s_activate, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # Labelled on the residual panel: the line is shared by both panels, and everything
    # left of it there is empty (no damage yet, so no error to plot).
    ax_resid.text(s_activate - 0.02, 2.2e-2, "damage activates", color=INK_SOFT, fontsize=8,
                  ha="right", va="center")

    # --- error vs pseudo-time ------------------------------------------------
    for dT, color in zip(dts, RAMP):
        d = data[str(dT)]
        s, omega = np.array(d["s"]), np.array(d["damage"])
        err = np.abs(omega - ref.sol(s)[0])
        ax_resid.plot(s, err, color=color, lw=1.4)

    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("pseudo-time  $s$")
    ax_resid.set_ylabel(r"$|\Delta\omega|$")
    ax_resid.set_xlim(0, 1.02)
    ax_resid.set_ylim(1e-5, 1e-1)

    # --- convergence ---------------------------------------------------------
    omega_ref_end = ref.sol(1.0)[0]
    errs = np.array([abs(data[str(dT)]["damage"][-1] - omega_ref_end) for dT in dts])
    dts_arr = np.array(dts)

    slope, intercept = np.polyfit(np.log(dts_arr), np.log(errs), 1)
    guide = np.exp(intercept) * dts_arr ** 1.0
    ax_conv.plot(dts_arr, guide * 1.55, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1,
                 label="first order (slope 1)")
    ax_conv.plot(dts_arr, errs, color="#2a78d6", lw=1.5, marker="o", ms=5.0,
                 mfc="#2a78d6", mec="white", mew=0.7, zorder=3,
                 label=rf"measured (slope {slope:.2f})")

    ax_conv.set_xscale("log")
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel(r"step size  $\Delta s$")
    ax_conv.set_ylabel(r"$|\omega_{\mathrm{FEA}}-\omega_{\mathrm{ref}}|$  at  $s=1$")
    ax_conv.legend(loc="lower right", borderaxespad=0.3)

    fig.text(0.008, 0.975, "Damage ODE verification: uniaxial displacement-controlled test",
             fontsize=10, color=INK, va="top", weight="medium")
    fig.text(0.008, 0.915,
             "backward Euler against an independent scalar-ODE reference; "
             "error decays after the activation kink rather than compounding",
             fontsize=8, color=INK_SOFT, va="top")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.125)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"damage_verification.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")

    print(f"\nConvergence (error at s=1, reference omega={omega_ref_end:.8f}):")
    print(f"  {'ds':>10} {'error':>12} {'ratio':>8} {'order':>7}")
    for i, (dT, e) in enumerate(zip(dts, errs)):
        if i == 0:
            print(f"  {dT:>10.5f} {e:>12.3e} {'-':>8} {'-':>7}")
        else:
            ratio = errs[i - 1] / e
            print(f"  {dT:>10.5f} {e:>12.3e} {ratio:>8.2f} {np.log2(ratio):>7.2f}")
    print(f"  least-squares order over all step sizes: {slope:.3f}")


if __name__ == "__main__":
    main()
