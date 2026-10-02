"""Fit the instantaneous c-axis CTE (Marsden et al. 2018, CCM acoustic + optical modes) and
integrate it to the SECANT CTE the model uses.

Definitions (T_ref = the model's stress-free InitialTemperature, 2000 K):
    instantaneous   alpha_inst(T) = d eps_th / dT
    thermal strain  eps_th(T)     = integral_{T_ref}^{T} alpha_inst(T') dT'
    secant          alpha_sec(T)  = eps_th(T) / (T - T_ref),   alpha_sec(T_ref) = alpha_inst(T_ref)
The model computes eps_th = alpha(T) * (T - T_0), so it needs alpha_sec -- and alpha_sec depends on
T_ref. If InitialTemperature changes, re-run this script with the new T_REF.

Both fit families integrate in closed form, so the secant carries no quadrature error:
  einstein2      alpha_inst = sum_i A_i E(theta_i/T),  E(x) = x^2 e^x / (e^x - 1)^2
                 (Einstein heat-capacity function: Grueneisen alpha ~ heat capacity, one term per
                 mode group -- the same acoustic + optical split as the CCM curve itself)
                 eps_th = sum_i A_i theta_i [1/(e^{theta_i/T} - 1) - 1/(e^{theta_i/T_ref} - 1)]
  polynomial     alpha_inst = sum_k c_k u^k,  u = T - T_ref
                 alpha_sec  = sum_k c_k u^k / (k + 1)        (no 0/0 at T_ref)
The closed forms are checked against numerical quadrature of alpha_inst below.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

HERE = Path(__file__).parent
T_REF = 2000.0
SCALE = 1e-6           # CSV and coefficients are in 1e-6/K
TOL_REL = 0.01         # parsimony rule: simplest form within 1% max relative error on its range


def E(x):
    return x**2 * np.exp(x) / np.expm1(x) ** 2


# ---------------------------------------------------------------------------------------------
# fit families: each returns (alpha_inst(T), alpha_sec(T), coefficients, muparser strings)
def fit_einstein2(T, a):
    f = lambda T, A1, t1, A2, t2: A1 * E(t1 / T) + A2 * E(t2 / T)
    p, _ = curve_fit(f, T, a, p0=[20, 300, 16, 1500], maxfev=20000)
    A1, t1, A2, t2 = p
    eps = lambda T: sum(A * t * (1 / np.expm1(t / T) - 1 / np.expm1(t / T_REF))
                        for A, t in [(A1, t1), (A2, t2)])

    def sec(T):
        T = np.asarray(T, float)
        u = T - T_REF
        safe = np.where(np.abs(u) < 1e-3, 1.0, u)
        return np.where(np.abs(u) < 1e-3, f(T_REF, *p), eps(T) / safe)

    a0 = f(T_REF, *p)
    inst_str = (f"({A1:.10e})*(({t1:.10e})/T)^2*exp(({t1:.10e})/T)/(exp(({t1:.10e})/T)-1)^2"
                f" + ({A2:.10e})*(({t2:.10e})/T)^2*exp(({t2:.10e})/T)/(exp(({t2:.10e})/T)-1)^2")
    # The T_ref term is written as the same expression evaluated at T_ref, not as a printed
    # constant: eps(T) - eps(T_ref) cancels catastrophically near T_ref, and a 10-digit constant
    # left a 2e-5 error at T = T_ref + 0.002 K. Same literals on both sides cancel exactly.
    eps_str = (f"({A1 * t1:.10e})*(1/(exp(({t1:.10e})/T)-1) - 1/(exp(({t1:.10e})/{T_REF:g})-1))"
               f" + ({A2 * t2:.10e})*(1/(exp(({t2:.10e})/T)-1) - 1/(exp(({t2:.10e})/{T_REF:g})-1))")
    # muparser has no removable-singularity handling: guard T = T_ref explicitly. deal.II's
    # FunctionParser defines if(cond, a, b) (mu_parser_internal.cc), which returns the chosen
    # branch only, so the unused 0/0 branch cannot leak a NaN.
    fsec_str = (f"if(abs(T-{T_REF:g})<1e-3, 1, ({eps_str})/((T-{T_REF:g})*({a0:.10e})))")
    finst_str = f"({inst_str})/({a0:.10e})"
    coeffs = {"A1_1e-6_per_K": A1, "theta1_K": t1, "A2_1e-6_per_K": A2, "theta2_K": t2}
    return (lambda T: f(np.asarray(T, float), *p)), sec, coeffs, finst_str, fsec_str, a0


def fit_poly(T, a, deg):
    c = np.polyfit(T - T_REF, a, deg)[::-1]            # c[k] multiplies u^k
    inst = lambda T: sum(ck * (np.asarray(T, float) - T_REF) ** k for k, ck in enumerate(c))
    sec = lambda T: sum(ck / (k + 1) * (np.asarray(T, float) - T_REF) ** k for k, ck in enumerate(c))
    a0 = c[0]

    def poly_str(coef):
        terms = ["1"] + [f"({ck / a0:.10e})*(T-{T_REF:g})" + (f"^{k}" if k > 1 else "")
                         for k, ck in enumerate(coef) if k > 0]
        return " + ".join(terms)

    coeffs = {f"c{k}_1e-6_per_K_per_K^{k}": ck for k, ck in enumerate(c)}
    coeffs["u"] = f"T - {T_REF:g}"
    return inst, sec, coeffs, poly_str(c), poly_str([ck / (k + 1) for k, ck in enumerate(c)]), a0


# ---------------------------------------------------------------------------------------------
def process_axis(name, csv_path, plot_axes):
    d = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    T, a = d[:, 0], d[:, 1]
    families = [
        ("linear_high", 1000, lambda T, a: fit_poly(T, a, 1)),
        ("quadratic_high", 1000, lambda T, a: fit_poly(T, a, 2)),
        ("einstein2_full", 300, fit_einstein2),
    ]
    out, recommended = {}, None
    grid = np.linspace(300, 2500, 221)
    for fam, lo, fitter in families:
        m = T >= lo
        inst, sec, coeffs, finst, fsec, a0 = fitter(T[m], a[m])
        err = inst(T[m]) - a[m]
        max_rel = float(np.max(np.abs(err / a[m])))

        # closed-form secant vs numerical quadrature of the fitted alpha_inst
        Tchk = grid[(grid >= lo) & (np.abs(grid - T_REF) > 1)]
        num = np.array([quad(lambda s: float(inst(s)), T_REF, t)[0] / (t - T_REF) for t in Tchk])
        sec_check = float(np.max(np.abs(sec(Tchk) - num) / num))
        assert sec_check < 1e-8, (fam, sec_check)
        assert abs(float(sec(T_REF)) - a0) < 1e-9 * a0

        out[fam] = {
            "valid_T_K": [lo, 2500],
            "coefficients": {k: (float(v) if not isinstance(v, str) else v) for k, v in coeffs.items()},
            "rms_1e-6_per_K": float(np.sqrt(np.mean(err**2))),
            "max_abs_error_1e-6_per_K": float(np.max(np.abs(err))),
            "max_rel_error": max_rel,
            "secant_closed_form_vs_quadrature_max_rel": sec_check,
            "instantaneous_at_T_ref_per_K": float(a0) * SCALE,
            "table_1e-6_per_K": {f"{t:.0f}": {"instantaneous": float(inst(t)), "secant": float(sec(t))}
                                 for t in [300, 500, 1000, 1500, 1800, 1900, 2000, 2200, 2500] if t >= lo},
            "model_ready": {
                "alpha_0_per_K": float(a0) * SCALE,
                "secant_functionOfTemperature": fsec,
                "instantaneous_functionOfTemperature": finst,
                "T_ref_K": T_REF,
                "note": "alpha(T) = alpha_0 * f(T). The model's eps_th = alpha(T)*(T - T_0) needs the "
                        "SECANT f, and only for T_0 = T_ref. The instantaneous f is for reference / an "
                        "integrated-eps_th formulation.",
            },
        }
        if recommended is None and max_rel <= TOL_REL:
            recommended = fam
        color = {"linear_high": "C0", "quadratic_high": "C1", "einstein2_full": "C2"}[fam]
        g = grid[grid >= lo]
        plot_axes[0].plot(g, inst(g), "-", color=color, label=f"{fam} instantaneous ({max_rel*100:.2f}%)")
        plot_axes[0].plot(g, sec(g), "--", color=color, label=f"{fam} secant from {T_REF:g} K")
        plot_axes[1].plot(T[m], err, "-", color=color, lw=0.8, label=fam)

    plot_axes[0].plot(T, a, "k.", ms=2, label="digitized (instantaneous)")
    plot_axes[0].axvline(T_REF, color="gray", lw=0.5)
    plot_axes[0].set_ylabel(f"{name} [1e-6/K]")
    plot_axes[0].legend(fontsize=7)
    plot_axes[1].axhline(0, color="gray", lw=0.5)
    plot_axes[1].set_ylabel("fit - data [1e-6/K]")
    plot_axes[1].set_xlabel("T [K]")
    plot_axes[1].legend(fontsize=7)
    return out, recommended


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_data = {"alpha_c": HERE / "digitized_alpha_c_CCM_acoustic_optical.csv"}
    fig, ax = plt.subplots(2, len(axes_data), figsize=(7 * len(axes_data), 7), squeeze=False)
    result = {
        "source": "Marsden et al. (2018), crystal-continuum model (CCM), acoustic plus optical modes "
                  "(red dashed curve) -- full citation to be filled in",
        "provenance": "curve DIGITIZED from a figure image by pixel colour (digitize_alpha_c.py); "
                      "T >= 300 K only",
        "quantity": "the figure plots the INSTANTANEOUS CTE d eps/dT; the secant is derived here by "
                    "closed-form integration from T_ref",
        "units": "coefficients in 1e-6/K, T in K; model_ready alpha_0 in 1/K",
        "T_ref_K": T_REF,
        "parsimony_rule": f"simplest form within {TOL_REL:.0%} max relative error on its own range, "
                          "tried linear_high -> quadratic_high -> einstein2_full",
        "axes": {},
    }
    for i, (name, path) in enumerate(axes_data.items()):
        fits, rec = process_axis(name, path, ax[:, i])
        result["axes"][name] = {"recommended": rec, "fits": fits}
    fig.tight_layout()
    fig.savefig(HERE / "fit_check.png", dpi=130)
    with open(HERE / "cte_fits.json", "w") as f:
        json.dump(result, f, indent=1)
    for name, v in result["axes"].items():
        print(name, "recommended:", v["recommended"])
        for fam, r in v["fits"].items():
            print(f"  {fam:15s} max rel {r['max_rel_error']*100:.2f}%  secant check {r['secant_closed_form_vs_quadrature_max_rel']:.1e}"
                  f"  alpha(2000) {r['instantaneous_at_T_ref_per_K']:.4e}")

    # Overlay: the fitted einstein2 curve drawn back onto the source figure in its own pixel
    # coordinates (same calibration as digitize_alpha_c.py). Checks calibration + fit by eye.
    from PIL import Image, ImageDraw
    img = Image.open(HERE / "source_figure_alpha_c.png").convert("RGB")
    draw = ImageDraw.Draw(img)
    f_inst = lambda T: (lambda e: sum(e[k] * E(e[t] / T) for k, t in
                                      [("A1_1e-6_per_K", "theta1_K"), ("A2_1e-6_per_K", "theta2_K")]))(
        result["axes"]["alpha_c"]["fits"]["einstein2_full"]["coefficients"])
    pts = [(95.5 + T * 18.5 / 100.0, 38.0 + (35.0 - f_inst(T)) * 231.0 / 30.0)
           for T in np.linspace(300, 2500, 400)]
    draw.line(pts, fill=(0, 160, 0), width=1)
    img.resize((img.width * 2, img.height * 2), Image.LANCZOS).save(HERE / "overlay_check.png")
