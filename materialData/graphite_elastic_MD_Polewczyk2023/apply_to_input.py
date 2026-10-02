#!/usr/bin/env python3
"""
Write the recommended elastic-constant fits into a deal.II input.json of this repository.

    python3 apply_to_input.py path/to/input.json [--family FAMILY]

Sets, for each of C11 C12 C33 C13 C44:
    MaterialParameters/LinearElasticityParameters/<Cij>_0                     value at T_ref (Pa)
    MaterialParameters/LinearElasticityParameters/<Cij>_functionOfTemperature  f(T), f(T_ref) = 1
and records the provenance in each entry's documentation field. Values come from
elastic_constants_fits.json -- nothing is copied by hand -- so re-running fit_elastic.py and then
this script is the whole update path.
"""
import argparse, collections, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("--family", default=None,
                    help="force one fit family for all constants (default: each constant's recommended)")
    a = ap.parse_args()

    fits = json.loads((HERE / "elastic_constants_fits.json").read_text())
    path = pathlib.Path(a.input_json)
    cfg = json.loads(path.read_text(), object_pairs_hook=collections.OrderedDict)
    sec = cfg["MaterialParameters"]["LinearElasticityParameters"]

    for n, e in fits["constants"].items():
        fam = a.family or e["recommended"]
        m, fit = e[fam]["model_ready"], e[fam]
        lo, hi = fit["valid_T_K"]
        tag = (f"Polewczyk et al. 2023, LCBOP_II MD, {fam} fit valid {lo}-{hi} K "
               f"(max rel err {100*fit['max_rel_error']:.1f}%), DIGITIZED FROM A FIGURE. "
               f"Source: materialData/graphite_elastic_MD_Polewczyk2023/.")
        sec[f"{n}_0"]["value"] = f"{m['Cij_0_Pa']:.6e}"
        sec[f"{n}_0"]["documentation"] = f"{n} at T_ref = {m['T_ref_K']:g} K. " + tag
        sec[f"{n}_functionOfTemperature"]["value"] = m["Cij_functionOfTemperature"]
        sec[f"{n}_functionOfTemperature"]["documentation"] = \
            f"f(T) normalised to 1 at {m['T_ref_K']:g} K. " + tag
        print(f"  {n:4s} {fam:15s} {n}_0 = {m['Cij_0_Pa']/1e9:9.5g} GPa   f(T) = {m['Cij_functionOfTemperature']}")

    path.write_text(json.dumps(cfg, indent=1))
    print(f"wrote {path}")

if __name__ == "__main__":
    main()
