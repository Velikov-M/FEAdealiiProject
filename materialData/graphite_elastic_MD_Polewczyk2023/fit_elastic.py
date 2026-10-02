#!/usr/bin/env python3
"""
Fit temperature dependencies of the graphite single-crystal elastic constants (LCBOP_II MD,
Polewczyk et al. 2023) and write them in reusable forms.

Inputs : digitized_LCBOP_II.csv     (read by eye from the paper's figure -- see its header)
Outputs: elastic_constants_fits.json  every fit, with coefficients, validity range and errors
         fit_check.png                data vs fits, one panel per constant

Three fit families are kept for every constant, because no single family is good for all five:
  linear_full     C = a + b T                  over 300-4000 K
  linear_high     C = a + b T                  over 1000-4000 K  (C13, C44 kink below 1000 K)
  quadratic_full  C = a + b T + c T^2          over 300-4000 K  (C12, C33 flatten at high T)
The RECOMMENDED family follows a PARSIMONY rule: the simplest form whose max relative error over its
own validity range is within TOL (5%), tried in the order linear_full -> linear_high ->
quadratic_full; if none qualifies, the best available. Linear is preferred wherever it is good enough
(the user's stated preference), and a quadratic is used only where a line genuinely fails. Re-run this script after correcting any digitized value; nothing is hand-copied.
"""
import csv, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
T_REF = 2000.0   # normalisation temperature for the model-ready form (the model's T_0)
NAMES = ["C11", "C12", "C33", "C13", "C44"]
TOL = 0.05       # parsimony threshold on max relative error
ORDER = ["linear_full", "linear_high", "quadratic_full"]


def load():
    lines = [l for l in open(HERE / "digitized_LCBOP_II.csv") if not l.startswith("#")]
    rows = list(csv.reader(lines))
    head, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    return data[:, 0], {n: data[:, head.index(n)] for n in head[1:]}


def fit(T, y, deg, tmin, tmax):
    m = (T >= tmin) & (T <= tmax)
    p = np.polyfit(T[m], y[m], deg)                 # highest power first
    r = y[m] - np.polyval(p, T[m])
    coef = list(p[::-1])                            # a, b, (c): constant term first
    return {
        "form": "C(T) = " + " + ".join(["a", "b*T", "c*T^2"][:deg + 1]),
        "coefficients_GPa": dict(zip("abc", [float(c) for c in coef])),
        "valid_T_K": [tmin, tmax],
        "rms_GPa": float(np.sqrt(np.mean(r ** 2))),
        "max_abs_error_GPa": float(abs(r).max()),
        "max_rel_error": float(abs(r / y[m]).max()),
        "_p": p,
    }


def model_ready(p, name):
    """C(T) = C_ref * f(T) with f(T_ref) = 1, written as a muParser string in the variable T --
    exactly the form ElasticityTensor::setFromPrm consumes (Cij_0 * Cij_functionOfTemperature)."""
    c_ref = float(np.polyval(p, T_REF))
    # expand the polynomial about T_REF:  C(T) = sum_k d_k (T - T_REF)^k   (Taylor, exact)
    d = [c_ref]
    if len(p) >= 2:
        d.append(float(np.polyval(np.polyder(p, 1), T_REF)))
    if len(p) >= 3:
        d.append(float(np.polyval(np.polyder(p, 2), T_REF)) / 2.0)
    terms = ["1"]
    if len(d) >= 2:
        terms.append(f"({d[1] / c_ref:.6e})*(T-{T_REF:g})")
    if len(d) >= 3:
        terms.append(f"({d[2] / c_ref:.6e})*(T-{T_REF:g})^2")
    return {"Cij_0_Pa": c_ref * 1e9, "Cij_functionOfTemperature": " + ".join(terms),
            "T_ref_K": T_REF, "note": f"{name}(T) = Cij_0 * f(T), f({T_REF:g} K) = 1"}


def transversely_isotropic_stable(c):
    """Positive-definiteness of a transversely isotropic stiffness."""
    C11, C12, C33, C13, C44 = (c[n] for n in NAMES)
    return C11 > abs(C12) and C44 > 0 and C33 > 0 and (C11 + C12) * C33 > 2 * C13 ** 2


def main():
    T, data = load()
    out = {
        "source": "Polewczyk et al. (2023), molecular dynamics, LCBOP_II potential -- full "
                  "citation to be filled in",
        "provenance": "values DIGITIZED BY EYE from a figure (see digitized_LCBOP_II.csv); "
                      "replace with the paper's tabulated data if available",
        "units": "GPa for coefficients, K for temperature",
        "C66": "not independent: C66 = (C11 - C12)/2 for transverse isotropy; the figure's C66 "
               "agrees with this within +-2.4% up to 3000 K and 4-5% above (digitization level)",
        "constants": {},
    }
    families = {"linear_full": (1, 300, 4000), "linear_high": (1, 1000, 4000),
                "quadratic_full": (2, 300, 4000)}
    best = {}
    for n in NAMES:
        entry = {f: fit(T, data[n], *a) for f, a in families.items()}
        ok = [f for f in ORDER if entry[f]["max_rel_error"] <= TOL]
        rec = ok[0] if ok else min(entry, key=lambda f: entry[f]["max_rel_error"])
        best[n] = entry[rec]["_p"]
        for f in entry:
            entry[f]["model_ready"] = model_ready(entry[f].pop("_p"), n)
        entry["recommended"] = rec
        out["constants"][n] = entry

    # stability of the recommended set across the whole range, finely sampled
    Ts = np.linspace(300, 4000, 371)
    unstable = [t for t in Ts
                if not transversely_isotropic_stable({n: np.polyval(best[n], t) for n in NAMES})]
    out["recommended_set_positive_definite_300_4000K"] = (len(unstable) == 0)
    (HERE / "elastic_constants_fits.json").write_text(json.dumps(out, indent=1))

    # ---------------------------------------------------------------- plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    style = {"linear_full": ("C0", "--"), "linear_high": ("C1", "-"), "quadratic_full": ("C2", "-.")}
    for ax, n in zip(axs.flat, NAMES):
        ax.plot(T, data[n], "k*", ms=7, label="digitized LCBOP_II")
        for f, (c, ls) in style.items():
            e = out["constants"][n][f]
            lo, hi = e["valid_T_K"]; tt = np.linspace(lo, hi, 100)
            k = e["coefficients_GPa"]
            yy = k["a"] + k["b"] * tt + k.get("c", 0.0) * tt ** 2
            lw = 2.4 if out["constants"][n]["recommended"] == f else 1.0
            ax.plot(tt, yy, color=c, ls=ls, lw=lw,
                    label=f + (" (recommended)" if lw > 2 else ""))
        ax.axvspan(1800, 2000, color="0.85", zorder=0)
        ax.set_title(n); ax.set_xlabel("T [K]"); ax.set_ylabel("GPa"); ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, frameon=False)
    c66 = (data["C11"] - data["C12"]) / 2
    ax = axs.flat[5]
    ax.plot(T, data["C66"], "k*", label="C66 read from figure")
    ax.plot(T, c66, "m^-", ms=4, label="(C11 - C12)/2")
    ax.set_title("C66 consistency check"); ax.set_xlabel("T [K]"); ax.grid(alpha=.3)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Graphite elastic constants, LCBOP_II MD (Polewczyk 2023) -- "
                 "grey band = current modelling window", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "fit_check.png", dpi=150)

    print("recommended family per constant:")
    for n in NAMES:
        e = out["constants"][n]; r = e[e["recommended"]]
        print(f"  {n:4s} {e['recommended']:15s} max rel err {100*r['max_rel_error']:4.1f}%  "
              f"Cij_0={r['model_ready']['Cij_0_Pa']/1e9:8.4g} GPa  f(T)={r['model_ready']['Cij_functionOfTemperature']}")
    print("recommended set positive definite over 300-4000 K:",
          out["recommended_set_positive_definite_300_4000K"])


if __name__ == "__main__":
    main()
