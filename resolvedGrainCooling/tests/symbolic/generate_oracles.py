#!/usr/bin/env python3
"""Generate independent closed-form references for the resolved-grain unit tests."""

import argparse
from pathlib import Path

import sympy as sp


def cpp(value):
    return format(float(value), ".17e")


def roots_for_damage(polynomial):
    roots = sp.nroots(polynomial, n=40, maxsteps=200)
    return sorted(
        [root for root in roots if abs(sp.im(root)) < sp.Float("1e-30") and 0 < sp.re(root) < 0.999],
        key=lambda root: float(sp.re(root)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    omega = sp.symbols("omega", real=True)
    coupled_k = sp.Rational("2.450539712794255")
    coupled_residual = omega - coupled_k * ((sp.Rational(3, 10) - omega / 5) / (1 - omega)) ** 2
    coupled_polynomial = sp.numer(sp.together(coupled_residual))
    coupled_roots = roots_for_damage(coupled_polynomial)
    if len(coupled_roots) != 2:
        raise RuntimeError(f"expected two admissible coupled-stress roots, got {coupled_roots}")

    old = sp.Rational(1, 10)
    k = sp.Rational(1, 5)
    acting_ratio = sp.Rational(1, 2)
    constant_residual = omega - old - k * acting_ratio / (1 - omega)
    constant_roots = roots_for_damage(sp.numer(sp.together(constant_residual)))
    if len(constant_roots) != 2:
        raise RuntimeError(f"expected two admissible m=1 roots, got {constant_roots}")

    temperature = 1800
    temperature_change = temperature - 2000
    C11 = sp.Rational("760e9") * (1 + sp.Rational(1, 10000) * temperature_change)
    C12 = sp.Rational("80e9") * (1 - sp.Rational(2, 10000) * temperature_change)
    C13 = sp.Rational("13e9") * (1 + sp.Rational(3, 10000) * temperature_change)
    C33 = sp.Rational("12e9") * (1 - sp.Rational(4, 10000) * temperature_change)
    C44 = sp.Rational("0.48e9") * (1 + sp.Rational(5, 10000) * temperature_change)
    damage = sp.Rational(1, 5)
    alpha11 = sp.Rational("1.2e-6") * (1 + sp.Rational(1, 1000) * temperature_change)
    alpha33 = sp.Rational("32e-6") * (1 - sp.Rational(1, 1000) * temperature_change)

    strain = sp.Matrix([
        [sp.Rational(1, 10000), sp.Rational(6, 10000), sp.Rational(-5, 10000)],
        [sp.Rational(6, 10000), sp.Rational(-2, 10000), sp.Rational(4, 10000)],
        [sp.Rational(-5, 10000), sp.Rational(4, 10000), sp.Rational(3, 10000)],
    ])
    elastic_strain = strain.copy()
    elastic_strain[0, 0] -= alpha11 * temperature_change
    elastic_strain[1, 1] -= alpha11 * temperature_change
    elastic_strain[2, 2] -= (1 - damage) * alpha33 * temperature_change
    C66 = (C11 - C12) / 2
    stress = sp.zeros(3)
    stress[0, 0] = C11 * elastic_strain[0, 0] + C12 * elastic_strain[1, 1] + C13 * elastic_strain[2, 2]
    stress[1, 1] = C12 * elastic_strain[0, 0] + C11 * elastic_strain[1, 1] + C13 * elastic_strain[2, 2]
    stress[2, 2] = C13 * (elastic_strain[0, 0] + elastic_strain[1, 1]) + (1 - damage) * C33 * elastic_strain[2, 2]
    stress[1, 2] = stress[2, 1] = 2 * C44 * elastic_strain[1, 2]
    stress[0, 2] = stress[2, 0] = 2 * C44 * elastic_strain[0, 2]
    stress[0, 1] = stress[1, 0] = 2 * C66 * elastic_strain[0, 1]
    stress_order = [stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]]

    baseline = [
        sp.Rational("7.593581e11"),
        sp.Rational("8.324534e10"),
        sp.Rational("1.305714e10"),
        sp.Rational("1.261083e10"),
        sp.Rational("4.8e8"),
    ]
    critical_damage = 1 - 2 * baseline[2] ** 2 / (baseline[3] * (baseline[0] + baseline[1]))

    x, y, z = sp.symbols("x y z", real=True)
    mms_temperature = 1900 + 20 * x + 10 * y + 5 * z
    mms_change = mms_temperature - 2000
    amplitude = sp.Rational("1e-6")
    mms_displacement = [
        amplitude * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.sin(sp.pi * z),
        amplitude * sp.sin(sp.pi * x) * sp.sin(2 * sp.pi * y) * sp.sin(sp.pi * z),
        amplitude * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.sin(2 * sp.pi * z),
    ]
    coordinates = (x, y, z)
    gradient = sp.Matrix(3, 3, lambda i, j: sp.diff(mms_displacement[i], coordinates[j]))
    mms_strain = (gradient + gradient.T) / 2
    mms_C11 = sp.Rational("760e9") * (1 + sp.Rational(1, 10000) * mms_change)
    mms_C12 = sp.Rational("80e9") * (1 - sp.Rational(2, 10000) * mms_change)
    mms_C13 = sp.Rational("13e9") * (1 + sp.Rational(3, 10000) * mms_change)
    mms_C33 = sp.Rational("12e9") * (1 - sp.Rational(4, 10000) * mms_change)
    mms_C44 = sp.Rational("0.48e9") * (1 + sp.Rational(5, 10000) * mms_change)
    mms_alpha11 = sp.Rational("1.2e-6") * (1 + sp.Rational(1, 1000) * mms_change)
    mms_alpha33 = sp.Rational("32e-6") * (1 - sp.Rational(1, 1000) * mms_change)
    mms_elastic_strain = mms_strain.copy()
    mms_elastic_strain[0, 0] -= mms_alpha11 * mms_change
    mms_elastic_strain[1, 1] -= mms_alpha11 * mms_change
    mms_elastic_strain[2, 2] -= mms_alpha33 * mms_change
    mms_C66 = (mms_C11 - mms_C12) / 2
    mms_stress = sp.zeros(3)
    mms_stress[0, 0] = (mms_C11 * mms_elastic_strain[0, 0]
                        + mms_C12 * mms_elastic_strain[1, 1]
                        + mms_C13 * mms_elastic_strain[2, 2])
    mms_stress[1, 1] = (mms_C12 * mms_elastic_strain[0, 0]
                        + mms_C11 * mms_elastic_strain[1, 1]
                        + mms_C13 * mms_elastic_strain[2, 2])
    mms_stress[2, 2] = (mms_C13 * (mms_elastic_strain[0, 0] + mms_elastic_strain[1, 1])
                        + mms_C33 * mms_elastic_strain[2, 2])
    mms_stress[1, 2] = mms_stress[2, 1] = 2 * mms_C44 * mms_elastic_strain[1, 2]
    mms_stress[0, 2] = mms_stress[2, 0] = 2 * mms_C44 * mms_elastic_strain[0, 2]
    mms_stress[0, 1] = mms_stress[1, 0] = 2 * mms_C66 * mms_elastic_strain[0, 1]
    mms_body_force = [
        -sum(sp.diff(mms_stress[i, j], coordinates[j]) for j in range(3))
        for i in range(3)
    ]
    mms_displacement_gradient = [
        sp.diff(mms_displacement[i], coordinates[j]) for i in range(3) for j in range(3)
    ]

    def parser_code(expression):
        return (sp.cxxcode(sp.expand(expression))
                .replace("std::", "")
                .replace("M_PI", "pi"))

    def expression_array(name, expressions):
        return [
            f"  inline constexpr const char *{name}[{len(expressions)}] = {{",
            *[f"    R\"mms({parser_code(expression)})mms\"," for expression in expressions],
            "  };",
        ]

    lines = [
        "#ifndef RESOLVED_GRAIN_TEST_SYMBOLIC_ORACLES_H",
        "#define RESOLVED_GRAIN_TEST_SYMBOLIC_ORACLES_H",
        "",
        "// Generated by tests/symbolic/generate_oracles.py using independent index notation.",
        "namespace SymbolicOracles",
        "{",
        f"  inline constexpr double coupledStressRoot = {cpp(sp.re(coupled_roots[0]))};",
        f"  inline constexpr double coupledStressSecondRoot = {cpp(sp.re(coupled_roots[1]))};",
        f"  inline constexpr double constantStressM1Root = {cpp(sp.re(constant_roots[0]))};",
        f"  inline constexpr double baselineCriticalDamage = {cpp(critical_damage)};",
        "  inline constexpr double stress[6] = {",
        *[f"    {cpp(value)}," for value in stress_order],
        "  };",
        *expression_array("mmsDisplacement", mms_displacement),
        *expression_array("mmsDisplacementGradient", mms_displacement_gradient),
        *expression_array("mmsBodyForce", mms_body_force),
        f"  inline constexpr const char *mmsTemperature = R\"mms({sp.cxxcode(mms_temperature)})mms\";",
        "  inline constexpr double mmsReferenceTemperature = 2000.0;",
        "}",
        "",
        "#endif",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
