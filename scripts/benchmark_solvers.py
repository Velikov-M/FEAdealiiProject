#!/usr/bin/env python3
import argparse
import itertools
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

import verify

ROOT = verify.ROOT
TARGET = "solver-benchmark"


def set_value(config, path, value):
    section, entry = path.rsplit(".", 1)
    node = config
    for part in section.split("."):
        node = node[part]
    node[entry]["value"] = str(value)


def execute_child(argv):
    report = Path(argv[0])
    started = time.monotonic()
    returncode = subprocess.call(argv[1:])
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    verify.dump_json(report, {
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
    })
    return returncode


def parse_profile(text):
    profile = {"timer_sections": {}}
    total = re.search(r"\| Total wallclock time elapsed since start\s*\|\s*([0-9.eE+-]+)s", text)
    if total:
        profile["timer_total_wall_seconds"] = float(total.group(1))
    pattern = re.compile(
        r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)s\s*\|", re.MULTILINE)
    for name, calls, seconds in pattern.findall(text):
        name = name.strip()
        if name != "Section":
            profile["timer_sections"][name] = {"calls": int(calls), "wall_seconds": float(seconds)}
    totals = re.search(
        r"Linear solver totals: solves=(\d+), CG iterations=(\d+), Cbar rebuilds=(\d+)", text)
    if totals:
        profile.update({
            "linear_solves": int(totals.group(1)),
            "total_cg_iterations": int(totals.group(2)),
            "cbar_rebuilds": int(totals.group(3)),
        })
    profile["outer_iterations"] = len(re.findall(
        r"^Refinement cycle \d+, temperature step \d+, iteration \d+", text, re.MULTILINE))
    return profile


def environment_text(executable):
    commands = [
        ["uname", "-a"], ["lscpu"], ["free", "-h"],
        [sys.executable, "--version"], ["git", "rev-parse", "HEAD"],
    ]
    sections = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        sections.append("$ " + " ".join(command) + "\n" + result.stdout)
    sections.append(f"executable sha256: {verify.sha256(executable)}\n")
    cache = ROOT / "resolvedGrainCooling/build/CMakeCache.txt"
    if cache.is_file():
        selected = [line for line in cache.read_text(errors="replace").splitlines()
                    if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:",
                                        "DEAL_II_DIR:", "deal.II_DIR:"))]
        sections.append("CMake cache selections:\n" + "\n".join(selected) + "\n")
    return "\n".join(sections)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        return execute_child(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Isolated resolved-grain linear-solver and threading benchmark matrix.")
    parser.add_argument("--solvers", nargs="+", required=True, choices=("cg", "amg", "direct"))
    parser.add_argument("--threads", nargs="+", required=True, type=int)
    parser.add_argument("--refinements", nargs="+", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--final-temperature", type=float, default=1960.0)
    parser.add_argument("--stress-threshold", type=float)
    parser.add_argument("--require-damage", action="store_true")
    parser.add_argument("--coupling-acceleration", choices=("none", "aitken"), default="aitken")
    parser.add_argument("--exact-cbar", action="store_true")
    parser.add_argument("--amg-smoother-sweeps", type=int)
    parser.add_argument("--base-input", type=Path,
                        default=ROOT / "resolvedGrainCooling/input.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "verification-runs")
    parser.add_argument("--memory-gib", type=float, default=6.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--label", default="resolved-grain solver benchmark")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in args.threads) or args.repetitions < 1:
        parser.error("thread counts and repetitions must be positive")
    if any(value < 0 for value in args.refinements):
        parser.error("refinement levels must be nonnegative")
    if args.timeout < 1 or args.memory_gib <= 0:
        parser.error("timeout and memory limit must be positive")

    executable = ROOT / "resolvedGrainCooling/build/main"
    if not executable.is_file() and not args.dry_run:
        parser.error(f"missing executable: {executable}; build resolvedGrainCooling first")
    base_path = args.base_input.resolve()
    base = json.loads(base_path.read_text())
    threshold = (args.stress_threshold if args.stress_threshold is not None else
                 float(base["MaterialParameters"]["DamageParameters"]["StressThreshold"]["value"]))
    initial_temperature = float(base["ModelParameters"]["InitialTemperature"]["value"])
    increment = float(base["ModelParameters"]["TemperatureIncrement"]["value"])
    expected_steps = int(math.ceil(
        (initial_temperature - args.final_temperature) / increment - 1e-9))
    if expected_steps < 1:
        parser.error("final temperature must require at least one cooling step")

    runner_args = argparse.Namespace(
        run_root=args.run_root, reference_run=None, jobs=1, threads=1,
        memory_gib=args.memory_gib, timeouts={TARGET: args.timeout})
    runner = verify.Runner(runner_args, [TARGET])
    runner.manifest["resources"]["solver_threads"] = "varies per case; see each isolated input"
    runner.manifest["benchmark"] = {
        "label": args.label,
        "base_input": str(base_path),
        "base_input_sha256": verify.sha256(base_path),
        "solvers": args.solvers,
        "threads": args.threads,
        "refinements": args.refinements,
        "repetitions": args.repetitions,
        "final_temperature_k": args.final_temperature,
        "stress_threshold_pa": threshold,
        "coupling_acceleration": args.coupling_acceleration,
        "frozen_cbar_quasi_newton": not args.exact_cbar,
        "amg_smoother_sweeps": args.amg_smoother_sweeps,
        "linear_solver_tolerance": base["SolverParameters"]["LinearSolverTolerance"]["value"],
        "timeout_seconds_per_case": args.timeout,
        "dry_run": args.dry_run,
    }
    verify.dump_json(runner.run_dir / "manifest.json", runner.manifest)
    if executable.is_file():
        (runner.run_dir / "environment.txt").write_text(environment_text(executable))

    cases = []
    for refinement, solver, threads, repetition in itertools.product(
            args.refinements, args.solvers, args.threads, range(1, args.repetitions + 1)):
        case_id = f"r{refinement}-{solver}-t{threads}-rep{repetition}"
        case_dir = runner.run_dir / "cases" / case_id
        workspace = case_dir / "workspace"
        output = workspace / "output"
        workspace.mkdir(parents=True)
        for name in ("cube10cellsPerSide.msh", "groupFile"):
            shutil.copy2(ROOT / "resolvedGrainCooling" / name, workspace / name)
        config = json.loads(base_path.read_text())
        overrides = {
            "MeshRefinementParameters.InitialGlobalRefinements": refinement,
            "MeshRefinementParameters.NumberOfRefinementCycles": 1,
            "ModelParameters.FinalTemperature": args.final_temperature,
            "MaterialParameters.DamageParameters.StressThreshold": threshold,
            "BoundaryConditions.Type": "periodic",
            "SolverParameters.LinearSolverType": solver,
            "SolverParameters.FrozenCbarQuasiNewton": "false" if args.exact_cbar else "true",
            "SolverParameters.CouplingAcceleration": args.coupling_acceleration,
            "ParallelizationParameters.NumberOfThreads": threads,
            "OutputParameters.OutputFrequency": 1000000,
            "OutputParameters.WriteFullStressTensor": "false",
            "OutputParameters.WriteFullStrainTensor": "false",
            "OutputParameters.OutputDirectory": output,
            "OutputParameters.OutputFileNamePrefix": "benchmark",
        }
        if args.amg_smoother_sweeps is not None:
            overrides["SolverParameters.AmgSmootherSweeps"] = args.amg_smoother_sweeps
        for path, value in overrides.items():
            set_value(config, path, value)
        run_input = workspace / "input.json"
        verify.dump_json(run_input, config)
        summary = {
            "case": case_id, "solver": solver, "threads": threads,
            "initial_global_refinements": refinement,
            "expected_cells": 1000 * 8 ** refinement, "repetition": repetition,
            "input_sha256": verify.sha256(run_input),
            "artifacts": str(workspace.relative_to(runner.run_dir)),
        }
        if args.dry_run:
            summary["status"] = "DRY_RUN"
            cases.append(summary)
            verify.dump_json(case_dir / "summary.json", summary)
            continue

        runner.args.threads = threads
        resource_report = case_dir / "resources.json"
        record = runner.command(TARGET, case_id, [
            sys.executable, Path(__file__).resolve(), "--execute", resource_report,
            executable, run_input,
        ], workspace, timeout=args.timeout)
        log = runner.run_dir / record["log"]
        shutil.copy2(log, case_dir / "run.log")
        summary["command"] = record
        if log.is_file():
            summary.update(parse_profile(log.read_text(errors="replace")))
        if resource_report.is_file():
            summary["resources"] = json.loads(resource_report.read_text())
        if record["status"] != "PASS":
            summary["status"] = record["status"]
        else:
            try:
                production = runner.check_production(
                    workspace, "resolvedGrainCooling", args.final_temperature, "benchmark",
                    expected_steps, False, 1000 * 8 ** refinement)
                if args.require_damage and (production["final_mean_damage"] <= 0 or
                                            production["final_max_damage"] <= 0):
                    raise ValueError("benchmark completed without required active damage")
                summary.update(production)
                history = output / "benchmark_macroscopic_history.csv"
                summary["macroscopic_history_sha256"] = verify.sha256(history)
                if "linear_solves" not in summary or "timer_total_wall_seconds" not in summary:
                    raise ValueError("missing profiling totals in production log")
                summary["status"] = "PASS"
            except Exception as error:
                summary["status"] = "FAIL"
                summary["reason"] = str(error)
        cases.append(summary)
        verify.dump_json(case_dir / "summary.json", summary)
        verify.dump_json(runner.run_dir / "cases.json", cases)
        print(f"[{case_id}] {summary['status']}", flush=True)

    protected_after = runner.protected_hashes()
    changed = sorted(path for path in set(runner.protected_before) | set(protected_after)
                     if runner.protected_before.get(path) != protected_after.get(path))
    runner.manifest.update({
        "finished": verify.utc_now(),
        "protected_files_changed": changed,
        "children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    })
    verify.dump_json(runner.run_dir / "manifest.json", runner.manifest)
    expected_status = "DRY_RUN" if args.dry_run else "PASS"
    overall = not changed and all(case["status"] == expected_status for case in cases)
    verify.dump_json(runner.run_dir / "summary.json", {
        "status": "PASS" if overall else "FAIL",
        "label": args.label,
        "case_counts": {
            "total": len(cases),
            "passed": sum(case["status"] == "PASS" for case in cases),
            "failed": sum(case["status"] not in ("PASS", "DRY_RUN") for case in cases),
        },
        "protected_files_changed": changed,
        "cases": cases,
    })
    print(f"Evidence: {runner.run_dir}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
