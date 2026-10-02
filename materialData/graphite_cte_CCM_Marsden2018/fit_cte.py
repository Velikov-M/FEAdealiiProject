"""Fit the instantaneous CTEs alpha_c and alpha_a (Marsden et al. 2018, CCM acoustic + optical
modes) and integrate them to the SECANT CTE the model uses.

Definitions (T_ref = the model's stress-free InitialTemperature, 2000 K):
    instantaneous   alpha_inst(T) = d eps_th / dT
    thermal strain  eps_th(T)     = integral_{T_ref}^{T} alpha_inst(T') dT'
    secant          alpha_sec(T)  = eps_th(T) / (T - T_ref),   alpha_sec(T_ref) = alpha_inst(T_ref)
The model computes eps_th = alpha(T) * (T - T_0), so it needs alpha_sec -- and alpha_sec depends on
T_ref. If InitialTemperature changes, re-run this script with the new T_REF.

Both fit families integrate in closed form, so the secant carries no quadrature error:
  einstein       alpha_inst = c + sum_i A_i E(theta_i/T),  E(x) = x^2 e^x / (e^x - 1)^2
                 (Einstein heat-capacity function: Grueneisen alpha ~ heat capacity, one term per
                 mode group; c is a mode group already fully excited over the fit range, E -> 1)
                 eps_th = c (T - T_ref)
                          + sum_i A_i theta_i [1/(e^{theta_i/T} - 1) - 1/(e^{theta_i/T_ref} - 1)]
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


def E(x):
    return x**2 * np.exp(x) / np.expm1(x) ** 2


# ---------------------------------------------------------------------------------------------
# fit families: each returns (alpha_inst(T), alpha_sec(T), coefficients, muparser strings)
def fit_einstein(T, a, p0, constant):
    """alpha = [c] + sum_i A_i E(theta_i/T); p0 = [c,] A1, th1, A2, th2, ..."""
    def f(T, *p):
        c, modes = (p[0], p[1:]) if constant else (0.0, p)
        return c + sum(modes[2 * i] * E(modes[2 * i + 1] / T) for i in range(len(modes) // 2))

    p, _ = curve_fit(f, T, a, p0=p0, maxfev=50000)
    c, modes = (p[0], p[1:]) if constant else (0.0, p)
    pairs = [(modes[2 * i], modes[2 * i + 1]) for i in range(len(modes) // 2)]
    eps = lambda T: c * (T - T_REF) + sum(A * t * (1 / np.expm1(t / T) - 1 / np.expm1(t / T_REF))
                                          for A, t in pairs)

    def sec(T):
        T = np.asarray(T, float)
        u = T - T_REF
        safe = np.where(np.abs(u) < 1e-3, 1.0, u)
        return np.where(np.abs(u) < 1e-3, f(T_REF, *p), eps(T) / safe)

    a0 = f(T_REF, *p)
    inst_str = " + ".join(([f"({c:.10e})"] if constant else []) +
                          [f"({A:.10e})*(({t:.10e})/T)^2*exp(({t:.10e})/T)/(exp(({t:.10e})/T)-1)^2"
                           for A, t in pairs])
    # The T_ref term is written as the same expression evaluated at T_ref, not as a printed
    # constant: eps(T) - eps(T_ref) cancels catastrophically near T_ref, and a 10-digit constant
    # left a 2e-5 error at T = T_ref + 0.002 K. Same literals on both sides cancel exactly.
    eps_str = " + ".join(([f"({c:.10e})*(T-{T_REF:g})"] if constant else []) +
                         [f"({A * t:.10e})*(1/(exp(({t:.10e})/T)-1) - 1/(exp(({t:.10e})/{T_REF:g})-1))"
                          for A, t in pairs])
    # muparser has no removable-singularity handling: guard T = T_ref explicitly. deal.II's
    # FunctionParser defines if(cond, a, b) (mu_parser_internal.cc), which returns the chosen
    # branch only, so the unused 0/0 branch cannot leak a NaN.
    fsec_str = (f"if(abs(T-{T_REF:g})<1e-3, 1, ({eps_str})/((T-{T_REF:g})*({a0:.10e})))")
    finst_str = f"({inst_str})/({a0:.10e})"
    coeffs = {"c_1e-6_per_K": c} if constant else {}
    for i, (A, t) in enumerate(pairs, 1):
        coeffs[f"A{i}_1e-6_per_K"] = A
        coeffs[f"theta{i}_K"] = t
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
def process_axis(name, spec, plot_axes):
    d = np.loadtxt(spec["csv"], delimiter=",", skiprows=1)
    T, a = d[:, 0], d[:, 1]
    families, criterion, tol = spec["families"], spec["criterion"], spec["tol"]
    out, recommended = {}, None
    grid = np.linspace(300, 2500, 221)
    for fam, lo, fitter in families:
        m = T >= lo
        inst, sec, coeffs, finst, fsec, a0 = fitter(T[m], a[m])
        err = inst(T[m]) - a[m]
        max_abs = float(np.max(np.abs(err)))
        max_rel = float(np.max(np.abs(err / a[m])))
        score = max_rel if criterion == "rel" else max_abs

        # closed-form secant vs numerical quadrature of the fitted alpha_inst
        Tchk = grid[(grid >= lo) & (np.abs(grid - T_REF) > 1)]
        num = np.array([quad(lambda s: float(inst(s)), T_REF, t)[0] / (t - T_REF) for t in Tchk])
        sec_check = float(np.max(np.abs(sec(Tchk) - num)) / np.max(np.abs(num)))
        assert sec_check < 1e-8, (fam, sec_check)
        assert abs(float(sec(T_REF)) - a0) < 1e-9 * a0

        out[fam] = {
            "valid_T_K": [lo, 2500],
            "coefficients": {k: (float(v) if not isinstance(v, str) else v) for k, v in coeffs.items()},
            "rms_1e-6_per_K": float(np.sqrt(np.mean(err**2))),
            "max_abs_error_1e-6_per_K": max_abs,
            "max_rel_error": max_rel,
            "passes_parsimony_tolerance": score <= tol,
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
        if recommended is None and score <= tol:
            recommended = fam
        color = f"C{[f[0] for f in families].index(fam)}"
        g = grid[grid >= lo]
        score_txt = f"{max_rel*100:.2f}%" if criterion == "rel" else f"max |err| {max_abs:.3f}"
        plot_axes[0].plot(g, inst(g), "-", color=color, label=f"{fam} instantaneous ({score_txt})")
        plot_axes[0].plot(g, sec(g), "--", color=color, label=f"{fam} secant from {T_REF:g} K")
        plot_axes[1].plot(T[m], err, "-", color=color, lw=0.8, label=fam)

    plot_axes[0].plot(T, a, "k.", ms=2, label="digitized (instantaneous)")
    plot_axes[0].axvline(T_REF, color="gray", lw=0.5)
    plot_axes[0].axhline(0, color="gray", lw=0.5)
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

    # Parsimony rule, per axis: the first family in the list within tolerance on its own range.
    #  alpha_c: 1% max relative error (the digitization is good to ~0.2%).
    #  alpha_a: crosses zero near 650 K, so relative error is meaningless; 0.03e-6/K max absolute
    #           error instead -- 0.1% of the mismatch alpha_c - alpha_a that drives the stresses,
    #           ~2% of alpha_a(2000 K), ~4 px on the figure.
    axes_spec = {
        "alpha_c": {
            "csv": HERE / "digitized_alpha_c_CCM_acoustic_optical.csv",
            "criterion": "rel", "tol": 0.01,
            "families": [
                ("linear_high", 1000, lambda T, a: fit_poly(T, a, 1)),
                ("quadratic_high", 1000, lambda T, a: fit_poly(T, a, 2)),
                ("einstein2_full", 300, lambda T, a: fit_einstein(T, a, [20, 300, 16, 1500], False)),
            ],
            # figure pixel calibration (digitize_alpha_c.py), for the overlay check
            "px": (lambda T: 95.5 + T * 18.5 / 100.0, lambda al: 38.0 + (35.0 - al) * 231.0 / 30.0),
            "figure": "source_figure_alpha_c.png", "overlay_fit": "einstein2_full",
        },
        "alpha_a": {
            "csv": HERE / "digitized_alpha_a_CCM_acoustic_optical.csv",
            "criterion": "abs", "tol": 0.03,
            "families": [
                ("linear_high", 1000, lambda T, a: fit_poly(T, a, 1)),
                ("quadratic_high", 1000, lambda T, a: fit_poly(T, a, 2)),
                # A free two-mode fit drives theta1 -> 0 (E -> 1): the negative flexural group is
                # fully excited above 300 K and acts as a constant. So fit that form directly.
                ("einstein1c_full", 300, lambda T, a: fit_einstein(T, a, [-0.6, 2.4, 2700], True)),
                ("cubic_high", 1000, lambda T, a: fit_poly(T, a, 3)),
            ],
            "px": (lambda T: 146.0 + T * 638.0 / 2500.0, lambda al: 238.75 - al * 126.3),
            "figure": "source_figure_alpha_a.png", "overlay_fit": "einstein1c_full",
        },
    }
    fig, ax = plt.subplots(2, len(axes_spec), figsize=(7 * len(axes_spec), 7), squeeze=False)
    result = {
        "source": "B. Marsden, A. Mummery, P. Mummery, Modelling the coefficient of thermal expansion in "
                  "graphite crystals: implications of lattice strain due to irradiation and pressure, "
                  "Proc. R. Soc. A 474 (2018) 20180075, doi:10.1098/rspa.2018.0075. CCM acoustic plus "
                  "optical modes: alpha_c red dashed curve (Fig. 4 per PMC text), alpha_a grey-blue "
                  "dashed curve (Fig. 6)",
        "provenance": "curves DIGITIZED from figure images by pixel colour (digitize_alpha_*.py); "
                      "T >= 300 K only",
        "quantity": "the figures plot the INSTANTANEOUS CTE d e/dT (paper eq. 2.6); the secant is derived "
                    "here by closed-form integration from T_ref",
        "units": "coefficients in 1e-6/K, T in K; model_ready alpha_0 in 1/K",
        "T_ref_K": T_REF,
        "axes": {},
    }
    for i, (name, spec) in enumerate(axes_spec.items()):
        fits, rec = process_axis(name, spec, ax[:, i])
        result["axes"][name] = {
            "recommended": rec,
            "parsimony_rule": (f"first of {[f[0] for f in spec['families']]} within "
                               + (f"{spec['tol']:.0%} max relative error" if spec["criterion"] == "rel"
                                  else f"{spec['tol']}e-6/K max absolute error")
                               + " on its own range"),
            "fits": fits,
        }
    fig.tight_layout()
    fig.savefig(HERE / "fit_check.png", dpi=130)
    with open(HERE / "cte_fits.json", "w") as f:
        json.dump(result, f, indent=1)
    for name, v in result["axes"].items():
        print(name, "recommended:", v["recommended"])
        for fam, r in v["fits"].items():
            print(f"  {fam:16s} max abs {r['max_abs_error_1e-6_per_K']:.4f}  max rel {r['max_rel_error']*100:7.2f}%"
                  f"  secant check {r['secant_closed_form_vs_quadrature_max_rel']:.1e}"
                  f"  alpha(2000) {r['instantaneous_at_T_ref_per_K']:.4e}")

    # Overlay: one fitted curve per axis drawn back onto its source figure in that figure's pixel
    # coordinates (the digitizers' calibration). Checks calibration + fit by eye.
    from PIL import Image, ImageDraw
    for name, spec in axes_spec.items():
        d = np.loadtxt(spec["csv"], delimiter=",", skiprows=1)
        fam = dict((f[0], f) for f in spec["families"])[spec["overlay_fit"]]
        m = d[:, 0] >= fam[1]
        inst = fam[2](d[m, 0], d[m, 1])[0]
        img = Image.open(HERE / spec["figure"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        x_of, y_of = spec["px"]
        draw.line([(x_of(T), y_of(float(inst(T)))) for T in np.linspace(300, 2500, 400)],
                  fill=(0, 160, 0), width=1)
        img.resize((img.width * 2, img.height * 2), Image.LANCZOS).save(HERE / f"overlay_check_{name}.png")
