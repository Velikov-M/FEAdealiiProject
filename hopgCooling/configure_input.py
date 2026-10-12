#!/usr/bin/env python3
"""Write the HOPG baseline into input.json: temperature window, mesh, solver switches, and the
temperature-dependent CTEs from materialData/ (secant re-integrated from T_0 = 2500 K).

The elastic constants are already the Polewczyk 2023 MD fits (inherited from resolvedGrainCooling);
they are absolute functions of T, so they need no re-referencing. The CTE secant DOES depend on the
stress-free temperature, so it is taken from cte_fits_Tref<T0>.json, and this script refuses to run
if that file's T_ref does not match InitialTemperature.

Run from hopgCooling/:  python3 configure_input.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
T0, TEND, DT = 2500.0, 1000.0, 50.0
CTE_JSON = HERE.parent / "materialData" / "graphite_cte_CCM_Marsden2018" / f"cte_fits_Tref{T0:g}.json"


def setv(tree, path, value, doc=None):
    node = tree
    for key in path[:-1]:
        node = node[key]
    entry = node.setdefault(path[-1], {})
    entry["value"] = str(value)
    if doc is not None:
        entry["documentation"] = doc


cfg = json.loads((HERE / "input.json").read_text())

cte = json.loads(CTE_JSON.read_text())
assert cte["T_ref_K"] == T0, f"{CTE_JSON.name} is referenced to {cte['T_ref_K']} K, not T0 = {T0} K"
for axis, key in [("alpha_c", "alpha33"), ("alpha_a", "alpha11")]:
    fit_name = cte["axes"][axis]["recommended"]
    fit = cte["axes"][axis]["fits"][fit_name]
    lo, hi = fit["valid_T_K"]
    assert lo <= TEND and T0 <= hi, f"{axis} {fit_name} valid {lo}-{hi} K does not cover {TEND}-{T0} K"
    m = fit["model_ready"]
    src = (f"Marsden et al. 2018 CCM acoustic+optical, {axis} {fit_name} (valid {lo:g}-{hi:g} K), "
           f"SECANT from T_ref = {T0:g} K. Source: {CTE_JSON.relative_to(HERE.parent)}")
    setv(cfg, ["MaterialParameters", "ThermoElasticParameters", f"{key}_0"], f"{m['alpha_0_per_K']:.10e}",
         f"{axis} at T_ref = {T0:g} K (instantaneous = secant there). {src}")
    setv(cfg, ["MaterialParameters", "ThermoElasticParameters", f"{key}_functionOfTemperature"],
         m["secant_functionOfTemperature"], f"secant f(T), f(T_ref) = 1. {src}")

setv(cfg, ["ModelParameters", "InitialTemperature"], f"{T0:g}",
     "Stress-free temperature: zero thermal strain and the secant CTE reference. Must match the "
     "T_ref of the CTE strings (configure_input.py enforces this).")
setv(cfg, ["ModelParameters", "FinalTemperature"], f"{TEND:g}")
setv(cfg, ["ModelParameters", "TemperatureIncrement"], f"{DT:g}",
     "Thermo-elastic and secant, so the solution at T is path-independent: the step only sets "
     "output resolution, not accuracy.")

setv(cfg, ["MeshRefinementParameters", "InitialGlobalRefinements"], 2)        # 64 elements/crystallite
setv(cfg, ["BoundaryConditions", "Type"], "periodic")
setv(cfg, ["BoundaryConditions", "GrainsPerSide"], 10)                        # 1000 crystallites
setv(cfg, ["BoundaryConditions", "UniformOrientation"], "false")
setv(cfg, ["BoundaryConditions", "OrientationDistribution"], "mosaicGaussian",
     "uniformSphere | mosaicGaussian (HOPG: c-axes scattered about global z).")
setv(cfg, ["BoundaryConditions", "MosaicSpreadDeg"], "0.5",
     "std [deg] of EACH tilt component (rocking-curve marginal), NOT the FWHM (= 2.355 sigma).")
setv(cfg, ["MaterialParameters", "DamageParameters", "DamageEnabled"], "false",
     "Purely thermo-elastic: damage identically zero, one mechanical pass per step.")
setv(cfg, ["OutputParameters", "OutputFrequency"], 10)                        # 2000, 1500, 1000 K
setv(cfg, ["OutputParameters", "OutputFileNamePrefix"], "hopg")
# Solver choice, measured on this problem (64 elements/crystallite, sigma = 1 deg, AMG vs CG and
# quasi-Newton vs exact, 2-3 steps each; stress statistics agree to 5e-5 between paths):
#  * AMG, not CG: a near-single-crystal keeps the full C11/C44 ~ 1400 anisotropy coherent across
#    the box, and CG needed 800-1000 iterations per solve (vs ~140 for the random polycrystal).
#    AMG: 110-275. Wall time 240 s vs 388 s for two steps.
#  * Exact 7-solve path, not the frozen-Cbar quasi-Newton: with no damage, Cbar still changes
#    ~3% per 50 K step through C(T), the frozen iteration contracts too slowly to reach
#    MacroscopicStressTolerance in MaxMacroscopicIterations, and every step ended in a rebuild:
#    12 solves/step vs 7. Wall time 368 s vs 276 s for three steps.
setv(cfg, ["SolverParameters", "LinearSolverType"], "amg")
setv(cfg, ["SolverParameters", "FrozenCbarQuasiNewton"], "false")

(HERE / "input.json").write_text(json.dumps(cfg, indent=1) + "\n")
print(f"input.json: T {T0:g} -> {TEND:g} K in {DT:g} K steps; CTE from {CTE_JSON.name}")
