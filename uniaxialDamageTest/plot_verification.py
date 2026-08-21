#!/usr/bin/env python3
"""
Verification plots for the uniaxial displacement-controlled damage test.

Three things are checked, and they are deliberately kept separate:

  1. Stress-damage coupling (algebraic, ODE-free).  At every quadrature point the
     FE stress must satisfy the closed-form relation at the SAME damage value:
         sigma_zz = eps_zz * [C33_0*(1-omega) - 2*C13_0^2/(C11_0+C12_0)]
     This holds regardless of whether omega itself is accurate, so it isolates the
     elastic/coupling algebra from time-integration error.  main.cpp evaluates it
     per point and reports the worst deviation; this script plots that.

  2. Damage trajectory vs an independent reference.  For this load case the problem
     reduces exactly to a scalar ODE, integrated here with scipy at tight tolerance.
     That independence is what makes it verification rather than self-consistency.

  3. Convergence order of the backward-Euler damage integrator.

Note on the stress: sigma_zz is NOT pinned at the threshold once damage activates.
That would make the Macaulay bracket vanish and the damage rate exactly zero.  This
is a rate (creep-type) law -- sustained overstress is what drives growth -- unlike
rate-independent plasticity, where stress really is held at yield.  The sigma_zz
panel shows the resulting rise-peak-decay: the decay is the negative feedback
(damage softens the material, so at fixed applied strain the stress drops), which is
what makes this problem stable.

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


def transverse_term(p):
    """2*C13_0^2/(C11_0+C12_0) -- the damage-independent part of the axial stiffness."""
    return 2.0 * p["C13_0"] ** 2 / (p["C11_0"] + p["C12_0"])


def sigma_zz_closed_form(p, s, omega):
    """sigma_zz = eps_zz * [C33_0*(1-omega) - 2*C13_0^2/(C11_0+C12_0)]."""
    return p["rate"] * s * (p["C33_0"] * (1.0 - omega) - transverse_term(p))


def reference_solution(p):
    """Dense high-accuracy solution of the reduced scalar ODE."""
    def rhs(s, y):
        omega = y[0]
        acting = max(sigma_zz_closed_form(p, s, omega) - p["sigma_th"], 0.0)
        return [p["A"] * (acting / p["sigma_th"]) ** p["m"] * (1.0 / (1.0 - omega)) ** p["m"]]

    # Tight tolerances: the reference must be far more accurate than anything it
    # is used to judge, otherwise the measured "error" is partly its own.
    return solve_ivp(rhs, [0.0, 1.0], [0.0], rtol=1e-12, atol=1e-14, dense_output=True)


def run_sweep():
    """Run the FEA at each step size, returning trajectories keyed by dT."""
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
                if cur is None:
                    continue
                m = re.search(r"damage: mean=([-\d.eE+]+) spread=([-\d.eE+]+)", line)
                if m:
                    per_step.setdefault(cur, {}).update(
                        damage=float(m.group(1)), spread=float(m.group(2)))
                m = re.search(r"sigma_zz: numeric mean~([-\d.eE+]+) spread=([-\d.eE+]+)", line)
                if m:
                    per_step.setdefault(cur, {}).update(sigma=float(m.group(1)))
                m = re.search(r"max\|sigma_zz_FE - sigma_zz_closedform\| = ([-\d.eE+]+)", line)
                if m:
                    per_step.setdefault(cur, {}).update(sigma_alg_err=float(m.group(1)))

            if not per_step or "Aborting" in proc.stdout:
                raise SystemExit(f"FEA run failed at dT={dT}:\n{proc.stdout[-2000:]}")

            steps = sorted(per_step)
            results[str(dT)] = {
                "s":            [n * dT for n in steps],
                "damage":       [per_step[n]["damage"] for n in steps],
                "spread":       [per_step[n]["spread"] for n in steps],
                "sigma":        [per_step[n]["sigma"] for n in steps],
                "sigma_alg_err": [per_step[n]["sigma_alg_err"] for n in steps],
            }
            print(f"  dT={dT:<9} {len(steps):>4} steps  omega_final={per_step[steps[-1]]['damage']:.6f}")
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

    if "sigma" not in next(iter(data.values())):
        raise SystemExit(f"{DATA_FILE} predates the sigma_zz fields -- re-run with --run")

    p = load_params()
    ref = reference_solution(p)
    dts = sorted((float(k) for k in data), reverse=True)

    fig = plt.figure(figsize=(9.6, 5.9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1.9, 1.5, 1.0],
                          hspace=0.16, wspace=0.28)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_sig = fig.add_subplot(gs[1, 0], sharex=ax_traj)
    ax_resid = fig.add_subplot(gs[2, 0], sharex=ax_traj)
    ax_conv = fig.add_subplot(gs[:, 1])

    s_dense = np.linspace(0, 1, 600)
    omega_dense = ref.sol(s_dense)[0]

    # --- damage trajectory ---------------------------------------------------
    ax_traj.plot(s_dense, omega_dense, color=INK, lw=1.4, zorder=3,
                 label="reference (scipy, rtol $10^{-12}$)")
    for dT, color in zip(dts, RAMP):
        d = data[str(dT)]
        s, omega = np.array(d["s"]), np.array(d["damage"])
        step = max(1, len(s) // 22)  # thin dense series so markers stay readable
        ax_traj.plot(s[::step], omega[::step], ls="none", marker="o", ms=4.0,
                     mfc=color, mec="white", mew=0.6, zorder=4,
                     label=rf"FEA  $\Delta s={dT:g}$")
    ax_traj.set_ylabel(r"damage  $\omega$")
    ax_traj.set_ylim(-0.03, 0.80)
    ax_traj.tick_params(labelbottom=False)
    ax_traj.legend(loc="upper left", ncol=2, handletextpad=0.5, columnspacing=1.0,
                   borderaxespad=0.3, labelspacing=0.3)

    eps_activate = p["sigma_th"] / (p["C33_0"] - transverse_term(p))
    s_activate = eps_activate / p["rate"]
    for ax in (ax_traj, ax_sig, ax_resid):
        ax.axvline(s_activate, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)

    # --- axial stress --------------------------------------------------------
    # Shows the negative feedback directly: stress rises elastically, peaks, then
    # decays as damage softens the material at fixed applied strain.
    ax_sig.plot(s_dense, sigma_zz_closed_form(p, s_dense, omega_dense) / 1e6,
                color=INK, lw=1.4, zorder=3)
    for dT, color in zip(dts, RAMP):
        d = data[str(dT)]
        s, sig = np.array(d["s"]), np.array(d["sigma"])
        step = max(1, len(s) // 22)
        ax_sig.plot(s[::step], sig[::step] / 1e6, ls="none", marker="o", ms=4.0,
                    mfc=color, mec="white", mew=0.6, zorder=4)
    ax_sig.axhline(p["sigma_th"] / 1e6, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_sig.text(0.015, p["sigma_th"] / 1e6, r"$\sigma_{th}$", color=INK_SOFT, fontsize=8,
                ha="left", va="bottom", transform=ax_sig.get_yaxis_transform())
    peak = int(np.argmax(sigma_zz_closed_form(p, s_dense, omega_dense)))
    ax_sig.annotate("stress peaks, then damage\nsoftening takes over",
                    xy=(s_dense[peak], sigma_zz_closed_form(p, s_dense[peak],
                                                            omega_dense[peak]) / 1e6),
                    xytext=(s_dense[peak] + 0.10, 120), color=INK_SOFT, fontsize=8,
                    va="center", linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8,
                                    shrinkA=2, shrinkB=4))
    ax_sig.set_ylabel(r"$\sigma_{zz}$  [MPa]")
    ax_sig.set_ylim(0, 360)
    ax_sig.tick_params(labelbottom=False)

    # --- damage error vs pseudo-time ----------------------------------------
    for dT, color in zip(dts, RAMP):
        d = data[str(dT)]
        s, omega = np.array(d["s"]), np.array(d["damage"])
        ax_resid.plot(s, np.abs(omega - ref.sol(s)[0]), color=color, lw=1.4)
    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("pseudo-time  $s$")
    ax_resid.set_ylabel(r"$|\Delta\omega|$")
    ax_resid.set_xlim(0, 1.02)
    ax_resid.set_ylim(1e-5, 1e-1)
    ax_resid.text(s_activate - 0.02, 2.2e-2, "damage activates", color=INK_SOFT,
                  fontsize=8, ha="right", va="center")

    # --- convergence ---------------------------------------------------------
    omega_ref_end = ref.sol(1.0)[0]
    errs = np.array([abs(data[str(dT)]["damage"][-1] - omega_ref_end) for dT in dts])
    dts_arr = np.array(dts)
    slope, intercept = np.polyfit(np.log(dts_arr), np.log(errs), 1)

    ax_conv.plot(dts_arr, np.exp(intercept) * dts_arr * 1.55, color=INK_MUTED, lw=0.9,
                 ls=(0, (4, 3)), zorder=1, label="first order (slope 1)")
    ax_conv.plot(dts_arr, errs, color="#2a78d6", lw=1.5, marker="o", ms=5.0,
                 mfc="#2a78d6", mec="white", mew=0.7, zorder=3,
                 label=rf"measured (slope {slope:.2f})")
    ax_conv.set_xscale("log")
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel(r"step size  $\Delta s$")
    ax_conv.set_ylabel(r"$|\omega_{\mathrm{FEA}}-\omega_{\mathrm{ref}}|$  at  $s=1$")
    ax_conv.legend(loc="lower right", borderaxespad=0.3)

    alg_err = max(max(data[str(dT)]["sigma_alg_err"]) for dT in dts)
    fig.text(0.008, 0.978, "Damage ODE verification: uniaxial displacement-controlled test",
             fontsize=10, color=INK, va="top", weight="medium")
    fig.text(0.008, 0.938,
             "backward Euler against an independent scalar-ODE reference; "
             "error decays after the activation kink rather than compounding",
             fontsize=8, color=INK_SOFT, va="top")
    fig.text(0.008, 0.902,
             rf"stress-damage coupling checked algebraically at every quadrature point:  "
             rf"max$|\sigma_{{zz}}^{{\mathrm{{FE}}}}-\sigma_{{zz}}^{{\mathrm{{alg}}}}|$ = "
             rf"{alg_err:.3g} Pa = {alg_err/p['sigma_th']:.1e}$\,\sigma_{{th}}$ "
             rf"(round-off, at LinearSolverTolerance $10^{{-12}}$)",
             fontsize=8, color=INK_SOFT, va="top")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.868, bottom=0.088)
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

    print("\nStress-damage coupling (algebraic, independent of ODE accuracy):")
    print(f"  worst max|sigma_zz_FE - sigma_zz_closedform| over all runs/steps: "
          f"{alg_err:.4g} Pa = {alg_err/p['sigma_th']:.2e} * sigma_th")
    sig_ref = sigma_zz_closed_form(p, s_dense, omega_dense)
    print(f"  reference sigma_zz peaks at {sig_ref.max()/1e6:.1f} MPa "
          f"= {sig_ref.max()/p['sigma_th']:.2f} * sigma_th at s={s_dense[int(np.argmax(sig_ref))]:.3f}; "
          f"ends at {sig_ref[-1]/p['sigma_th']:.2f} * sigma_th")


if __name__ == "__main__":
    main()
