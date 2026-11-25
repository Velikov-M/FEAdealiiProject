#!/usr/bin/env python3
import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
BUILD_TARGETS = {
    "build-mms": "mmsSandbox",
    "build-uniaxial": "uniaxialDamageTest",
    "build-resolved": "resolvedGrainCooling",
    "build-hopg": "hopgCooling",
}
RESOLVED_AB_TARGETS = [
    "smoke-resolved", "damage-resolved", "serial-resolved",
    "damage-serial-resolved", "exact-resolved", "zero-resolved",
    "free-resolved", "uniform-resolved", "refine-resolved",
]
RESOLVED_ANALYTICAL_TARGETS = [
    "uniform-resolved", "clamped-uniform-resolved", "free-uniform-resolved",
    "laminate-resolved",
    "zero-response-periodic-resolved", "zero-response-free-resolved",
    "zero-response-clamped-resolved",
]
CHECK_TARGETS = [
    "env", "build-mms", "build-uniaxial", "build-resolved", "build-hopg",
    "unit-resolved", "production-mms-resolved", "mms-small", "uniaxial-small", *RESOLVED_ANALYTICAL_TARGETS,
    "damage-resolved", "adaptive-resolved", "smoke-hopg",
]
TARGET_ORDER = [
    "env", *BUILD_TARGETS, "unit-resolved", "production-mms-resolved", "mms-small", "uniaxial-small",
    "smoke-resolved", "smoke-hopg", "damage-resolved", "adaptive-resolved",
    *(name for name in RESOLVED_AB_TARGETS
      if name not in ("smoke-resolved", "damage-resolved")),
    *(name for name in RESOLVED_ANALYTICAL_TARGETS if name not in RESOLVED_AB_TARGETS),
]
DEFAULT_TIMEOUTS = {
    "env": 120,
    **{name: 600 for name in BUILD_TARGETS},
    "mms-small": 600,
    "uniaxial-small": 900,
    "unit-resolved": 120,
    "production-mms-resolved": 300,
    "smoke-resolved": 600,
    "smoke-hopg": 600,
    "damage-resolved": 600,
    "adaptive-resolved": 600,
    **{name: 600 for name in RESOLVED_AB_TARGETS},
    **{name: 600 for name in RESOLVED_ANALYTICAL_TARGETS},
}
ARCHIVED_MMS_ERRORS = {
    1000: {"L2": 1.23566e-6, "H1": 2.70120e-5, "Linfty": 2.79240e-6},
    8000: {"L2": 3.09003e-7, "H1": 1.33240e-5, "Linfty": 7.14046e-7},
}
ARCHIVED_UNIAXIAL_DAMAGE = {0.1: 0.687177, 0.05: 0.688889, 0.025: 0.689688}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_temperature_expression(expression, temperature):
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "T":
            return temperature
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.Pow: lambda: left ** right,
            }
            for operation, apply in operations.items():
                if isinstance(node.op, operation):
                    return apply()
        raise ValueError(f"unsupported temperature expression: {expression}")

    return evaluate(tree)


class Runner:
    def __init__(self, args, targets):
        self.args = args
        self.targets = targets
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True
        ).strip()
        self.run_dir = args.run_root.resolve() / f"{stamp}-{revision}-{os.getpid()}"
        self.run_dir.mkdir(parents=True)
        self.results = {}
        self.commands = []
        self.deferred = []
        self.protected_before = self.protected_hashes()
        self.manifest = {
            "started": utc_now(),
            "working_directory": str(ROOT),
            "run_directory": str(self.run_dir),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "targets": targets,
            "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
            "resources": {
                "build_jobs": args.jobs,
                "solver_threads": args.threads,
                "memory_limit_gib_per_process": args.memory_gib,
                "timeouts_seconds": args.timeouts,
                "tests": "sequential",
                "timeout_kill_grace_seconds": 30,
            },
        }
        dump_json(self.run_dir / "manifest.json", self.manifest)
        (self.run_dir / "git-status.txt").write_text(subprocess.check_output(
            ["git", "status", "--short", "--branch", "--untracked-files=all"],
            cwd=ROOT, text=True,
        ))
        with (self.run_dir / "git-diff.patch").open("w") as stream:
            subprocess.run(["git", "diff", "--no-ext-diff"], cwd=ROOT, stdout=stream, check=False)
        dump_json(self.run_dir / "protected-hashes-before.json", self.protected_before)
        source_files = [
            *ROOT.glob("*/main.cpp"), *ROOT.glob("*/CMakeLists.txt"),
            *ROOT.glob("*/input.json"),
            *ROOT.glob("resolvedGrainCooling/include/**/*.h"),
            *ROOT.glob("resolvedGrainCooling/src/*.cpp"),
            *ROOT.glob("resolvedGrainCooling/tests/**/*.cpp"),
            *ROOT.glob("resolvedGrainCooling/tests/**/*.py"),
            ROOT / "resolvedGrainCooling/Doxyfile",
            ROOT / "scripts/verify.sh", *ROOT.glob("scripts/*.py"),
        ]
        lines = [f"{sha256(path)}  {path.relative_to(ROOT)}"
                 for path in sorted(set(source_files)) if path.is_file()]
        (self.run_dir / "source-hashes.sha256").write_text("\n".join(lines) + "\n")

    def protected_hashes(self):
        patterns = [
            "*/input.json", "mmsSandbox/mms_convergence_data.json",
            "mmsSandbox/mms_verification.png", "mmsSandbox/mms_verification.pdf",
            "uniaxialDamageTest/convergence_data.json",
            "uniaxialDamageTest/damage_verification.png",
            "uniaxialDamageTest/damage_verification.pdf", "*/results/**",
        ]
        files = set()
        for pattern in patterns:
            files.update(ROOT.glob(pattern))
        return {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(files) if path.is_file()
        }

    def timeout_for(self, target):
        return self.args.timeouts.get(target, DEFAULT_TIMEOUTS[target])

    def command(self, target, name, argv, cwd, timeout=None):
        timeout = timeout or self.timeout_for(target)
        log = self.run_dir / target / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        actual = [
            "timeout", "--signal=TERM", "--kill-after=30s", str(timeout),
            *[str(item) for item in argv],
        ]
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(self.args.threads),
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        memory = int(self.args.memory_gib * 1024 ** 3)

        def limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        record = {
            "target": target,
            "name": name,
            "argv": actual,
            "cwd": str(cwd),
            "started": utc_now(),
            "timeout_seconds": timeout,
            "log": str(log.relative_to(self.run_dir)),
        }
        print(f"[{target}] {name}: {' '.join(actual)}", flush=True)
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                actual, cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, preexec_fn=limits,
            )
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise
        elapsed = time.monotonic() - started
        record.update({
            "finished": utc_now(),
            "returncode": returncode,
            "wall_seconds": round(elapsed, 3),
            "children_peak_rss_kib_after": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        })
        if returncode == 0:
            record["status"] = "PASS"
        elif returncode == 124 or (returncode == 137 and elapsed >= timeout):
            record["status"] = "TIMEOUT"
        else:
            record["status"] = "FAIL"
        self.commands.append(record)
        dump_json(self.run_dir / "commands.json", self.commands)
        print(f"[{target}] {name}: {record['status']} ({elapsed:.1f} s)", flush=True)
        return record

    def finish(self, target, status, **details):
        result = {"status": status, **details}
        self.results[target] = result
        dump_json(self.run_dir / target / "summary.json", result)
        dump_json(self.run_dir / "summary.json", {
            "status": "PASS" if all(r["status"] == "PASS" for r in self.results.values()) else "FAIL",
            "results": self.results,
            "deferred": self.deferred,
        })
        print(f"[{target}] {status}", flush=True)

    def run(self):
        if "env" in self.targets:
            self.environment()
        for target in BUILD_TARGETS:
            if target in self.targets:
                self.build(target)
        for target in self.targets:
            if target == "unit-resolved":
                self.unit()
            elif target == "production-mms-resolved":
                self.production_mms()
            elif target == "mms-small":
                self.mms()
            elif target == "uniaxial-small":
                self.uniaxial()
            elif target == "smoke-resolved":
                self.production(target, "resolvedGrainCooling", 1960)
            elif target == "smoke-hopg":
                self.production(target, "hopgCooling", 2400)
            elif target == "damage-resolved":
                self.production(target, "resolvedGrainCooling", 1800,
                                output_prefix="damage", require_damage=True,
                                preserve_baseline=True, damage_threshold_pa=60e6)
            elif target == "adaptive-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                output_prefix="adaptive", require_damage=True,
                                preserve_baseline=True, damage_threshold_pa=5e6,
                                overrides={
                                    "MeshRefinementParameters.RefinementStrategy": "adaptive",
                                    "MeshRefinementParameters.AdaptiveRefinementInterval": "1",
                                    "MeshRefinementParameters.FractionToCoarsen": "0",
                                }, require_adaptive_refinement=True)
            elif target == "serial-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"ParallelizationParameters.UseWorkStream": "false"})
            elif target == "damage-serial-resolved":
                self.production(target, "resolvedGrainCooling", 1800,
                                output_prefix="damage", require_damage=True,
                                preserve_baseline=True, damage_threshold_pa=60e6,
                                overrides={"ParallelizationParameters.UseWorkStream": "false"})
            elif target == "exact-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"SolverParameters.FrozenCbarQuasiNewton": "false"})
            elif target == "zero-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"BoundaryConditions.Type": "zeroDisplacement"})
            elif target == "free-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"BoundaryConditions.Type": "freeSurface"})
            elif target == "uniform-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"BoundaryConditions.UniformOrientation": "true"},
                                analytical_response="uniform")
            elif target == "clamped-uniform-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"BoundaryConditions.Type": "zeroDisplacement",
                                           "BoundaryConditions.UniformOrientation": "true"},
                                analytical_response="clamped")
            elif target == "free-uniform-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"BoundaryConditions.Type": "freeSurface",
                                           "BoundaryConditions.UniformOrientation": "true",
                                           "SolverParameters.LinearSolverType": "direct"},
                                analytical_response="free")
            elif target == "laminate-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"SolverParameters.FrozenCbarQuasiNewton": "false",
                                           "SolverParameters.LinearSolverType": "direct"},
                                analytical_response="laminate")
            elif target in ("zero-response-periodic-resolved", "zero-response-free-resolved",
                            "zero-response-clamped-resolved"):
                boundary = {
                    "zero-response-periodic-resolved": "periodic",
                    "zero-response-free-resolved": "freeSurface",
                    "zero-response-clamped-resolved": "zeroDisplacement",
                }[target]
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={
                                    "BoundaryConditions.Type": boundary,
                                    "MaterialParameters.ThermoElasticParameters.alpha11_0": "0",
                                    "MaterialParameters.ThermoElasticParameters.alpha33_0": "0",
                                }, analytical_response="zero")
            elif target == "refine-resolved":
                self.production(target, "resolvedGrainCooling", 1960,
                                overrides={"MeshRefinementParameters.InitialGlobalRefinements": "1"},
                                expected_cells=8000)
        protected_after = self.protected_hashes()
        changed = sorted(
            path for path in set(self.protected_before) | set(protected_after)
            if self.protected_before.get(path) != protected_after.get(path)
        )
        self.manifest.update({
            "finished": utc_now(),
            "protected_files_changed": changed,
            "children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        })
        dump_json(self.run_dir / "manifest.json", self.manifest)
        overall = not changed and all(r["status"] == "PASS" for r in self.results.values())
        target_counts = {
            "total": len(self.results),
            "passed": sum(r["status"] == "PASS" for r in self.results.values()),
            "failed": sum(r["status"] not in ("PASS", "SKIPPED") for r in self.results.values()),
            "skipped": sum(r["status"] == "SKIPPED" for r in self.results.values()),
        }
        dump_json(self.run_dir / "summary.json", {
            "status": "PASS" if overall else "FAIL",
            "results": self.results,
            "deferred": self.deferred,
            "protected_files_changed": changed,
            "target_counts": target_counts,
            "run_directory": str(self.run_dir),
        })
        self.write_junit()
        print(f"Artifacts: {self.run_dir}")
        print(f"Targets: {target_counts['passed']} passed, {target_counts['failed']} failed, "
              f"{target_counts['skipped']} skipped")
        print(f"Overall: {'PASS' if overall else 'FAIL'}")
        return 0 if overall else 1

    def write_junit(self):
        root = ET.Element("testsuites", name="fea-verification")
        for target, result in self.results.items():
            if target == "unit-resolved" and "xml" in result:
                xml_path = self.run_dir / result["xml"]
                if xml_path.is_file():
                    for suite in ET.parse(xml_path).getroot().findall("testsuite"):
                        root.append(ET.fromstring(ET.tostring(suite)))
                    continue
            failures = int(result["status"] in ("FAIL", "TIMEOUT"))
            skipped = int(result["status"] == "SKIPPED")
            suite = ET.SubElement(root, "testsuite", name=target, tests="1",
                                  failures=str(failures), errors="0", skipped=str(skipped))
            case = ET.SubElement(suite, "testcase", classname="verify", name=target)
            if failures:
                failure = ET.SubElement(case, "failure", message=result["status"])
                failure.text = json.dumps(result, indent=2, sort_keys=True)
            elif skipped:
                ET.SubElement(case, "skipped", message=result.get("reason", "skipped"))
        root.set("tests", str(sum(int(suite.attrib["tests"]) for suite in root.findall("testsuite"))))
        root.set("failures", str(sum(int(suite.attrib["failures"]) for suite in root.findall("testsuite"))))
        root.set("errors", "0")
        root.set("skipped", str(sum(int(suite.attrib["skipped"]) for suite in root.findall("testsuite"))))
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(self.run_dir / "junit.xml", encoding="utf-8", xml_declaration=True)

    def environment(self):
        target = "env"
        lines = [
            f"recorded: {utc_now()}", f"cwd: {ROOT}", f"platform: {platform.platform()}",
            f"machine: {platform.machine()}",
        ]
        for path in (Path("/etc/os-release"), Path("/proc/1/cgroup"), Path("/usr/local/summary.log")):
            if path.exists():
                lines.extend([f"--- {path}", path.read_text().strip()])
        config = self.find_dealii_config()
        features = {}
        if config:
            text = config.read_text()
            for key in (
                "DEAL_II_PACKAGE_VERSION", "DEAL_II_WITH_MPI", "DEAL_II_WITH_TBB",
                "DEAL_II_WITH_TRILINOS", "DEAL_II_TRILINOS_WITH_ML", "DEAL_II_WITH_UMFPACK",
            ):
                match = re.search(rf"set\({key}\s+([^\)]+)\)", text)
                features[key] = match.group(1).strip().strip('"') if match else "UNKNOWN"
            lines.append(f"deal.II config: {config}")
            lines.extend(f"{key}={value}" for key, value in features.items())
        lines.append("Container image metadata available inside the container is limited to the records above.")
        log = self.run_dir / target / "environment.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(lines) + "\n")
        records = [
            self.command(target, "compiler", ["c++", "--version"], ROOT),
            self.command(target, "cmake", ["cmake", "--version"], ROOT),
            self.command(target, "python", ["python3", "--version"], ROOT),
            self.command(target, "python-packages", ["python3", "-c",
                "import numpy, scipy, matplotlib, sympy; "
                "print('numpy', numpy.__version__); print('scipy', scipy.__version__); "
                "print('matplotlib', matplotlib.__version__); "
                "print('sympy', sympy.__version__)"], ROOT),
            self.command(target, "dealii-discovery", [
                "cmake", "--find-package", "-DNAME=deal.II", "-DCOMPILER_ID=GNU",
                "-DLANGUAGE=CXX", "-DMODE=EXIST",
            ], self.run_dir / target),
        ]
        required = ["DEAL_II_WITH_UMFPACK", "DEAL_II_WITH_TRILINOS", "DEAL_II_TRILINOS_WITH_ML"]
        okay = bool(config) and all(features.get(key) == "ON" for key in required)
        okay = okay and all(record["status"] == "PASS" for record in records)
        self.finish(target, "PASS" if okay else "FAIL", dealii=features,
                    environment_file=str(log.relative_to(self.run_dir)))

    def find_dealii_config(self):
        candidates = []
        if self.args.deal_ii_dir:
            candidates.append(Path(self.args.deal_ii_dir) / "deal.IIConfig.cmake")
        candidates.extend([
            Path("/usr/local/lib/cmake/deal.II/deal.IIConfig.cmake"),
            Path("/usr/lib/x86_64-linux-gnu/cmake/deal.II/deal.IIConfig.cmake"),
        ])
        return next((path for path in candidates if path.is_file()), None)

    def build(self, target):
        source = ROOT / BUILD_TARGETS[target]
        build = source / "build"
        configure = ["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release"]
        if self.args.deal_ii_dir:
            configure.append(f"-DDEAL_II_DIR={self.args.deal_ii_dir}")
        first = self.command(target, "configure", configure, source)
        if first["status"] != "PASS":
            self.finish(target, first["status"], program=BUILD_TARGETS[target])
            return
        second = self.command(target, "build", [
            "cmake", "--build", build, "--parallel", self.args.jobs,
        ], source)
        executable = build / "main"
        okay = second["status"] == "PASS" and executable.is_file()
        details = {"program": BUILD_TARGETS[target], "executable": str(executable)}
        cache = build / "CMakeCache.txt"
        if cache.exists():
            for line in cache.read_text(errors="replace").splitlines():
                if line.startswith(("deal.II_DIR:", "DEAL_II_DIR:", "CMAKE_BUILD_TYPE:")):
                    details[line.split(":", 1)[0]] = line.split("=", 1)[-1]
        self.finish(target, "PASS" if okay else second["status"], **details)

    def require_executable(self, target, program):
        executable = ROOT / program / "build/main"
        if executable.is_file():
            return executable
        self.finish(target, "SKIPPED", reason=f"missing executable: {executable}; select its build target first")
        return None

    def prepare_verification_workspace(self, target, program, driver):
        executable = self.require_executable(target, program)
        if executable is None:
            return None
        workspace = self.run_dir / target / "workspace"
        (workspace / "build").mkdir(parents=True)
        source = ROOT / program
        for name in (driver, "input.json", "cube10cellsPerSide.msh", "groupFile"):
            shutil.copy2(source / name, workspace / name)
        for name in ("cube10cellsPerSide.msh", "groupFile"):
            shutil.copy2(source / name, workspace / "build" / name)
        shutil.copy2(executable, workspace / "build/main")
        return workspace

    def unit(self):
        target = "unit-resolved"
        executable = ROOT / "resolvedGrainCooling/build/resolved_grain_tests"
        if not executable.is_file():
            self.finish(target, "SKIPPED",
                        reason=f"missing unit-test executable: {executable}; select build-resolved first")
            return
        workspace = self.run_dir / target / "workspace"
        workspace.mkdir(parents=True)
        xml_path = self.run_dir / target / "gtest.xml"
        record = self.command(target, "run", [
            executable, f"--gtest_output=xml:{xml_path}",
        ], workspace)
        if record["status"] != "PASS":
            self.finish(target, record["status"], command=record)
            return
        try:
            root = ET.parse(xml_path).getroot()
            tests = int(root.attrib["tests"])
            failures = int(root.attrib["failures"])
            skipped = int(root.attrib.get("skipped", "0"))
            if tests < 1 or failures != 0 or skipped != 0:
                raise ValueError(
                    f"unexpected GoogleTest totals: tests={tests}, failures={failures}, skipped={skipped}")
            self.finish(target, "PASS", tests=tests, failures=failures, skipped=skipped,
                        xml=str(xml_path.relative_to(self.run_dir)))
        except Exception as error:
            self.finish(target, "FAIL", reason=str(error))

    def production_mms(self):
        target = "production-mms-resolved"
        executable = ROOT / "resolvedGrainCooling/build/resolved_grain_mms"
        if not executable.is_file():
            self.finish(target, "SKIPPED",
                        reason=f"missing production MMS executable: {executable}; select build-resolved first")
            return
        workspace = self.run_dir / target / "workspace"
        workspace.mkdir(parents=True)
        output = workspace / "mms.json"
        record = self.command(target, "run", [executable, output], workspace)
        if record["status"] != "PASS":
            self.finish(target, record["status"], command=record)
            return
        try:
            data = json.loads(output.read_text())
            rows = data["rows"]
            cells = [row["cells"] for row in rows]
            if cells != [64, 512, 4096]:
                raise ValueError(f"unexpected production MMS cells: {cells}")
            rates = {}
            for norm, bounds in (("L2", (1.8, 2.2)), ("H1", (0.8, 1.2))):
                errors = [float(row[norm]) for row in rows]
                if any(not finite(value) or value <= 0 for value in errors):
                    raise ValueError(f"non-finite or non-positive production MMS {norm} error")
                rates[norm] = [math.log(errors[i] / errors[i + 1]) / math.log(2)
                               for i in range(len(errors) - 1)]
                if any(not bounds[0] <= value <= bounds[1] for value in rates[norm]):
                    raise ValueError(f"production MMS {norm} rates outside {bounds}: {rates[norm]}")
            adaptive = data["adaptive_rows"]
            adaptive_cells = [row["cells"] for row in adaptive]
            adaptive_dofs = [row["dofs"] for row in adaptive]
            if adaptive_cells != [64, 197, 610, 1891]:
                raise ValueError(f"unexpected adaptive MMS cells: {adaptive_cells}")
            if adaptive_dofs != [375, 1029, 2901, 7434]:
                raise ValueError(f"unexpected adaptive MMS DoFs: {adaptive_dofs}")
            adaptive_errors = {}
            for norm in ("L2", "H1", "indicator"):
                values = [float(row[norm]) for row in adaptive]
                if any(not finite(value) or value <= 0 for value in values):
                    raise ValueError(f"non-finite or non-positive adaptive MMS {norm}")
                if any(new >= old for old, new in zip(values, values[1:])):
                    raise ValueError(f"adaptive MMS {norm} is not strictly decreasing: {values}")
                adaptive_errors[norm] = values
            for norm in ("L2", "H1"):
                if abs(adaptive_errors[norm][0] - float(rows[0][norm])) > 1e-14:
                    raise ValueError("initial adaptive and global 64-cell solutions differ")
            comparisons = []
            for row in adaptive:
                reference = min(rows, key=lambda candidate: abs(candidate["dofs"] - row["dofs"]))
                comparisons.append({
                    "adaptive_cells": row["cells"], "adaptive_dofs": row["dofs"],
                    "nearest_global_cells": reference["cells"],
                    "nearest_global_dofs": reference["dofs"],
                    "dof_ratio": row["dofs"] / reference["dofs"],
                    "L2_ratio": row["L2"] / reference["L2"],
                    "H1_ratio": row["H1"] / reference["H1"],
                })
            self.finish(target, "PASS", cells=cells,
                        errors={norm: [float(row[norm]) for row in rows] for norm in ("L2", "H1")},
                        observed_rates=rates,
                        adaptive_cells=adaptive_cells, adaptive_dofs=adaptive_dofs,
                        adaptive_errors=adaptive_errors,
                        adaptive_nearest_global_comparisons=comparisons,
                        artifacts=str(workspace.relative_to(self.run_dir)))
        except Exception as error:
            self.finish(target, "FAIL", reason=str(error))

    def mms(self):
        target = "mms-small"
        self.deferred.append({"target": target, "case": "64000-cell MMS refinement level",
                              "status": "SKIPPED", "reason": "approved host resource limit"})
        workspace = self.prepare_verification_workspace(target, "mmsSandbox", "plot_mms_verification.py")
        if workspace is None:
            return
        input_path = workspace / "input.json"
        config = json.loads(input_path.read_text())
        config["MeshRefinementParameters"]["NumberOfRefinementCycles"]["value"] = "2"
        config["ParallelizationParameters"]["NumberOfThreads"]["value"] = str(self.args.threads)
        config["OutputParameters"]["OutputFrequency"]["value"] = "1000000"
        dump_json(input_path, config)
        driver = workspace / "plot_mms_verification.py"
        text = driver.read_text()
        text = text.replace(
            "        out = proc.stdout\n",
            "        out = proc.stdout\n"
            "        with open(os.path.join(HERE, 'mms_fea.log'), 'w') as raw_log:\n"
            "            raw_log.write(out)\n",
        )
        driver.write_text(text)
        record = self.command(target, "run", [sys.executable, driver.name, "--run"], workspace)
        if record["status"] != "PASS":
            self.finish(target, record["status"], command=record)
            return
        try:
            summary = self.check_mms(workspace)
            self.finish(target, "PASS", **summary)
        except Exception as error:
            self.finish(target, "FAIL", reason=str(error))

    def check_mms(self, workspace):
        data_path = workspace / "mms_convergence_data.json"
        data = json.loads(data_path.read_text())
        rows = data["rows"]
        expected = {f"{cycle}|{step}" for cycle in range(2) for step in range(1, 11)}
        missing = sorted(expected - set(rows))
        if missing:
            raise ValueError(f"missing MMS rows: {missing}")
        final = [rows[f"{cycle}|10"] for cycle in range(2)]
        cells = [row["cells"] for row in final]
        if cells != [1000, 8000]:
            raise ValueError(f"unexpected MMS cell counts: {cells}")
        errors = {norm: [float(row[norm]) for row in final] for norm in ("L2", "H1", "Linfty")}
        if any(not finite(value) or value < 0 for values in errors.values() for value in values):
            raise ValueError("non-finite or negative MMS error")
        rates = {
            norm: math.log(values[0] / values[1]) / math.log(2)
            for norm, values in errors.items()
        }
        for norm in rates:
            lower, upper = (0.8, 1.2) if norm == "H1" else (1.8, 2.2)
            if not lower <= rates[norm] <= upper:
                raise ValueError(f"MMS {norm} rate {rates[norm]:.6g} outside [{lower}, {upper}]")
        relative_changes = {}
        for index, count in enumerate(cells):
            for norm in errors:
                change = abs(errors[norm][index] - ARCHIVED_MMS_ERRORS[count][norm]) / ARCHIVED_MMS_ERRORS[count][norm]
                relative_changes[f"{count}:{norm}"] = change
                if change > 0.01:
                    raise ValueError(f"MMS {count} {norm} differs from archive by {change:.3%}")
        for name in ("mms_verification.png", "mms_verification.pdf", "mms_fea.log"):
            if not (workspace / name).is_file() or (workspace / name).stat().st_size == 0:
                raise ValueError(f"missing or empty {name}")
        return {"cells": cells, "final_errors": errors, "observed_rates": rates,
                "relative_change_from_archive": relative_changes,
                "completed_rows": len(expected), "artifacts": str(workspace.relative_to(self.run_dir))}

    def uniaxial(self):
        target = "uniaxial-small"
        for step_size, steps in ((0.0125, 80), (0.00625, 160)):
            self.deferred.append({"target": target, "case": f"dT={step_size} ({steps} steps)",
                                  "status": "SKIPPED", "reason": "approved host resource limit"})
        workspace = self.prepare_verification_workspace(
            target, "uniaxialDamageTest", "plot_verification.py")
        if workspace is None:
            return
        input_path = workspace / "input.json"
        config = json.loads(input_path.read_text())
        config["ParallelizationParameters"]["NumberOfThreads"]["value"] = str(self.args.threads)
        config["OutputParameters"]["OutputFrequency"]["value"] = "1000000"
        dump_json(input_path, config)
        driver = workspace / "plot_verification.py"
        text = driver.read_text().replace(
            "STEP_SIZES = [0.1, 0.05, 0.025, 0.0125, 0.00625]",
            "STEP_SIZES = [0.1, 0.05, 0.025]",
        )
        text = text.replace(
            "            # Keep the LAST displacement-damage iteration per step (the converged one).\n",
            "            with open(os.path.join(HERE, f'uniaxial_dT_{dT}.log'), 'w') as raw_log:\n"
            "                raw_log.write(proc.stdout)\n"
            "            # Keep the LAST displacement-damage iteration per step (the converged one).\n",
        )
        driver.write_text(text)
        record = self.command(target, "run", [sys.executable, driver.name, "--run"], workspace)
        if record["status"] != "PASS":
            self.finish(target, record["status"], command=record)
            return
        try:
            summary = self.check_uniaxial(workspace)
            self.finish(target, "PASS", **summary)
        except Exception as error:
            self.finish(target, "FAIL", reason=str(error))

    def uniaxial_reference(self, workspace):
        from scipy.integrate import solve_ivp
        config = json.loads((workspace / "input.json").read_text())
        elastic = config["MaterialParameters"]["LinearElasticityParameters"]
        damage = config["MaterialParameters"]["DamageParameters"]
        model = config["ModelParameters"]
        c11 = float(elastic["C11_0"]["value"])
        c12 = float(elastic["C12_0"]["value"])
        c13 = float(elastic["C13_0"]["value"])
        c33 = float(elastic["C33_0"]["value"])
        threshold = float(damage["StressThreshold"]["value"])
        a = float(damage["a_kineticParam"]["value"])
        m = float(damage["m_kineticParam"]["value"])
        rate = float(model["AppliedAxialStrainRate"]["value"])
        transverse = 2 * c13 ** 2 / (c11 + c12)

        def rhs(s, y):
            omega = y[0]
            stress = rate * s * (c33 * (1 - omega) - transverse)
            acting = max(stress - threshold, 0.0)
            return [a * (acting / threshold) ** m * (1 / (1 - omega)) ** m]

        solution = solve_ivp(rhs, [0, 1], [0], rtol=1e-12, atol=1e-14)
        if not solution.success:
            raise ValueError(f"independent uniaxial reference failed: {solution.message}")
        return float(solution.y[0, -1]), threshold

    def check_uniaxial(self, workspace):
        data = json.loads((workspace / "convergence_data.json").read_text())
        expected = {0.1: 10, 0.05: 20, 0.025: 40}
        reference, threshold = self.uniaxial_reference(workspace)
        final_damage, errors, max_spread, max_algebraic = {}, {}, 0.0, 0.0
        for step_size, steps in expected.items():
            key = str(step_size)
            if key not in data:
                raise ValueError(f"missing uniaxial dT={key}")
            row = data[key]
            if len(row["s"]) != steps or not math.isclose(row["s"][-1], 1.0, abs_tol=1e-12):
                raise ValueError(f"uniaxial dT={key} did not complete {steps} steps to pseudo-time 1")
            if any(not finite(value) for field in ("damage", "spread", "sigma", "sigma_alg_err") for value in row[field]):
                raise ValueError(f"non-finite uniaxial output at dT={key}")
            final_damage[key] = row["damage"][-1]
            errors[key] = abs(final_damage[key] - reference)
            archived_error = abs(ARCHIVED_UNIAXIAL_DAMAGE[step_size] - reference)
            if errors[key] > archived_error * 1.01:
                raise ValueError(f"uniaxial dT={key} final error exceeds archived error tolerance")
            max_spread = max(max_spread, *row["spread"])
            max_algebraic = max(max_algebraic, *row["sigma_alg_err"])
            raw_log = workspace / f"uniaxial_dT_{step_size}.log"
            if not raw_log.is_file():
                raise ValueError(f"missing {raw_log.name}")
            if re.search(r"Aborting|did not converge|failed to converge", raw_log.read_text(), re.I):
                raise ValueError(f"non-convergence marker in {raw_log.name}")
        step_sizes = list(expected)
        order = __import__("numpy").polyfit(
            [math.log(value) for value in step_sizes],
            [math.log(errors[str(value)]) for value in step_sizes], 1,
        )[0]
        if not 0.8 <= order <= 1.3:
            raise ValueError(f"uniaxial observed order {order:.6g} outside [0.8, 1.3]")
        if max_spread > 1e-9:
            raise ValueError(f"uniaxial spatial spread {max_spread:.3e} exceeds 1e-9")
        if max_algebraic > 1e-8 * threshold:
            raise ValueError(f"uniaxial algebraic stress mismatch {max_algebraic:.3e} Pa is too large")
        for name in ("damage_verification.png", "damage_verification.pdf"):
            if not (workspace / name).is_file() or (workspace / name).stat().st_size == 0:
                raise ValueError(f"missing or empty {name}")
        return {"reference_final_damage": reference, "final_damage": final_damage,
                "final_errors": errors, "least_squares_order": float(order),
                "max_spatial_spread": max_spread,
                "max_algebraic_stress_mismatch_pa": max_algebraic,
                "completed_steps": expected, "artifacts": str(workspace.relative_to(self.run_dir))}

    def production(self, target, program, final_temperature, output_prefix="smoke",
                   require_damage=False, preserve_baseline=False, damage_threshold_pa=None,
                   overrides=None, expected_cells=1000, analytical_response=None,
                   require_adaptive_refinement=False):
        executable = self.require_executable(target, program)
        if executable is None:
            return
        workspace = self.run_dir / target / "workspace"
        output = workspace / "output"
        workspace.mkdir(parents=True)
        source = ROOT / program
        for name in ("cube10cellsPerSide.msh", "groupFile"):
            shutil.copy2(source / name, workspace / name)
        config = json.loads((source / "input.json").read_text())
        if damage_threshold_pa is not None:
            config["MaterialParameters"]["DamageParameters"]["StressThreshold"]["value"] = str(
                damage_threshold_pa)
        model = config["ModelParameters"]
        initial_temperature = float(model["InitialTemperature"]["value"])
        increment = float(model["TemperatureIncrement"]["value"])
        expected_steps = (initial_temperature - final_temperature) / increment
        if expected_steps < 1 or not expected_steps.is_integer():
            raise ValueError(f"invalid temperature-step count: {expected_steps}")
        expected_steps = int(expected_steps)
        if not preserve_baseline:
            config["MeshRefinementParameters"]["InitialGlobalRefinements"]["value"] = "0"
            config["MeshRefinementParameters"]["NumberOfRefinementCycles"]["value"] = "1"
            config["OutputParameters"]["OutputFrequency"]["value"] = "1000000"
        model["FinalTemperature"]["value"] = str(final_temperature)
        for path, value in (overrides or {}).items():
            section, entry = path.rsplit(".", 1)
            node = config
            for part in section.split("."):
                node = node[part]
            node[entry]["value"] = value
        config["ParallelizationParameters"]["NumberOfThreads"]["value"] = str(self.args.threads)
        config["OutputParameters"]["OutputDirectory"]["value"] = str(output)
        config["OutputParameters"]["OutputFileNamePrefix"]["value"] = output_prefix
        if analytical_response == "laminate":
            orientation_path = workspace / "laminate_orientations.txt"
            with orientation_path.open("w") as stream:
                for grain in range(1000):
                    matrix = ((1, 0, 0, 0, 1, 0, 0, 0, 1) if grain % 10 < 5 else
                              (0, 1, 0, 0, 0, 1, 1, 0, 0))
                    stream.write(" ".join(str(value) for value in matrix) + "\n")
            config["BoundaryConditions"]["UseOrientationFile"] = {"value": "true"}
            config["InputFiles"]["OrientationFile"]["value"] = str(orientation_path)
        run_input = workspace / "input.json"
        dump_json(run_input, config)
        record = self.command(target, "run", [executable, run_input], workspace)
        if record["status"] != "PASS":
            self.finish(target, record["status"], command=record)
            return
        try:
            summary = self.check_production(
                workspace, program, final_temperature, output_prefix,
                expected_steps, require_damage, expected_cells, analytical_response,
                require_adaptive_refinement)
            if self.args.reference_run:
                reference_workspace = (self.args.reference_run.resolve() / target / "workspace")
                if reference_workspace.is_dir() or target in RESOLVED_AB_TARGETS:
                    summary["ab_comparison"] = self.compare_with_reference(target, workspace)
                else:
                    summary["ab_comparison"] = {
                        "status": "not_available",
                        "reason": "target did not exist in the accepted reference run",
                    }
            self.finish(target, "PASS", **summary)
        except Exception as error:
            self.finish(target, "FAIL", reason=str(error))

    def read_commented_csv(self, path):
        with path.open() as stream:
            return list(csv.DictReader(line for line in stream if not line.startswith("#")))

    def check_production(self, workspace, program, final_temperature, output_prefix,
                         expected_steps, require_damage, expected_cells=1000,
                         analytical_response=None, require_adaptive_refinement=False):
        output = workspace / "output"
        log = workspace.parent / "run.log"
        text = log.read_text(errors="replace")
        if re.search(r"Aborting|did not converge|failed to converge|WARNING", text, re.I):
            raise ValueError("production log contains a non-convergence or warning marker")
        if f"Cells: {expected_cells}" not in text:
            raise ValueError(f"production log does not confirm {expected_cells} cells")
        adapted_cells = None
        if require_adaptive_refinement:
            matches = re.findall(r"Adaptive refinement completed: (\d+) cells", text)
            if len(matches) != 1:
                raise ValueError(f"expected one in-history adaptation, found {len(matches)}")
            adapted_cells = int(matches[0])
            if adapted_cells <= expected_cells:
                raise ValueError(f"adaptive refinement did not increase cells: {adapted_cells}")
            if f"Cells: {adapted_cells}" not in text:
                raise ValueError("post-adaptation solve does not confirm the adapted cell count")
        history_path = output / f"{output_prefix}_macroscopic_history.csv"
        history = self.read_commented_csv(history_path)
        if len(history) != expected_steps:
            raise ValueError(f"expected {expected_steps} macroscopic rows, found {len(history)}")
        self.check_history_rows(history, final_temperature)
        damages = {
            key: [float(row[key]) for row in history]
            for key in ("meanDamage", "maxDamage")
        }
        for key, values in damages.items():
            if any(value < 0 or value > 0.999 for value in values):
                raise ValueError(f"{key} outside [0, 0.999]: {values}")
            if any(new < old - 1e-12 for old, new in zip(values, values[1:])):
                raise ValueError(f"{key} is not monotone: {values}")
        if require_damage and (damages["meanDamage"][-1] <= 0 or damages["maxDamage"][-1] <= 0):
            raise ValueError("resolved-grain baseline completed without active damage")
        if program == "hopgCooling" and any(value != 0 for value in damages["maxDamage"]):
            raise ValueError("HOPG smoke unexpectedly evolved damage")
        pvd = output / f"{output_prefix}.pvd"
        tree = ET.parse(pvd)
        fields = [node.attrib["file"] for node in tree.iter("DataSet")]
        if not fields:
            raise ValueError("PVD contains no field files")
        if require_damage and len(fields) != expected_steps:
            raise ValueError(f"expected {expected_steps} PVD field files, found {len(fields)}")
        for name in fields:
            field = output / name
            if not field.is_file() or field.stat().st_size == 0:
                raise ValueError(f"missing or empty PVD field: {name}")
        activation = next(
            (float(row["T"]) for row in history if float(row["maxDamage"]) > 0), None)
        if analytical_response:
            if analytical_response == "uniform":
                self.check_uniform_crystal(workspace, history)
            elif analytical_response in ("free", "clamped"):
                self.check_homogeneous_boundary_response(workspace, history, analytical_response)
            elif analytical_response == "laminate":
                self.check_laminate_response(workspace, history)
            elif analytical_response == "zero":
                self.check_zero_response(history)
            else:
                raise ValueError(f"unknown analytical response check: {analytical_response}")
        summary = {
            "cells": adapted_cells if adapted_cells is not None else expected_cells,
            "completed_temperature_steps": len(history),
            "initial_temperature_k": float(history[0]["T"]) + float(history[0]["cooling"]),
            "final_temperature_k": float(history[-1]["T"]),
            "damage_activation_temperature_k": activation,
            "final_mean_damage": damages["meanDamage"][-1],
            "final_max_damage": damages["maxDamage"][-1],
            "final_mean_stress_norm_pa": float(history[-1]["meanStressNorm"]),
            "pvd_field_files": fields,
            "artifacts": str(workspace.relative_to(self.run_dir)),
        }
        if analytical_response:
            summary["analytical_response"] = analytical_response
        if adapted_cells is not None:
            summary["initial_cells"] = expected_cells
            summary["adapted_cells"] = adapted_cells
        if program == "hopgCooling":
            statistics = self.read_commented_csv(output / f"{output_prefix}_stress_statistics.csv")
            if len(statistics) != expected_steps:
                raise ValueError(
                    f"expected {expected_steps} HOPG stress-statistics rows, found {len(statistics)}")
            self.check_history_rows(statistics, final_temperature)
            summary["stress_statistics_rows"] = len(statistics)
        return summary

    def check_homogeneous_boundary_response(self, workspace, history, response):
        config = json.loads((workspace / "input.json").read_text())
        elastic = config["MaterialParameters"]["LinearElasticityParameters"]
        thermal = config["MaterialParameters"]["ThermoElasticParameters"]
        initial_temperature = float(config["ModelParameters"]["InitialTemperature"]["value"])
        for row in history:
            temperature = float(row["T"])
            actual_norm = float(row["meanStressNorm"])
            if response == "free":
                if actual_norm > 1e-3:
                    raise ValueError(
                        f"traction-free uniform crystal has mean-stress norm {actual_norm} Pa")
                continue

            constants = {
                name: float(elastic[f"{name}_0"]["value"])
                * evaluate_temperature_expression(
                    elastic[f"{name}_functionOfTemperature"]["value"], temperature)
                for name in ("C11", "C12", "C13", "C33")
            }
            alpha11 = float(thermal["alpha11_0"]["value"]) * evaluate_temperature_expression(
                thermal["alpha11_functionOfTemperature"]["value"], temperature)
            alpha33 = float(thermal["alpha33_0"]["value"]) * evaluate_temperature_expression(
                thermal["alpha33_functionOfTemperature"]["value"], temperature)
            delta_t = temperature - initial_temperature
            transverse = -((constants["C11"] + constants["C12"]) * alpha11
                           + constants["C13"] * alpha33) * delta_t
            axial = -(2 * constants["C13"] * alpha11
                      + constants["C33"] * alpha33) * delta_t
            expected_norm = math.sqrt(2 * transverse ** 2 + axial ** 2)
            if not math.isclose(actual_norm, expected_norm, rel_tol=1e-7, abs_tol=1e-3):
                raise ValueError(
                    f"clamped uniform crystal mean-stress norm {actual_norm} Pa, "
                    f"expected {expected_norm} Pa")

    def check_laminate_response(self, workspace, history):
        import sympy as sp

        config = json.loads((workspace / "input.json").read_text())
        elastic = config["MaterialParameters"]["LinearElasticityParameters"]
        thermal = config["MaterialParameters"]["ThermoElasticParameters"]
        initial_temperature = float(config["ModelParameters"]["InitialTemperature"]["value"])
        Ey, Ez = sp.symbols("Ey Ez")
        for row in history:
            temperature = float(row["T"])
            constants = {
                name: float(elastic[f"{name}_0"]["value"])
                * evaluate_temperature_expression(
                    elastic[f"{name}_functionOfTemperature"]["value"], temperature)
                for name in ("C11", "C12", "C13", "C33")
            }
            C11, C12, C13, C33 = (constants[name] for name in ("C11", "C12", "C13", "C33"))
            alpha11 = float(thermal["alpha11_0"]["value"]) * evaluate_temperature_expression(
                thermal["alpha11_functionOfTemperature"]["value"], temperature)
            alpha33 = float(thermal["alpha33_0"]["value"]) * evaluate_temperature_expression(
                thermal["alpha33_functionOfTemperature"]["value"], temperature)
            delta_t = temperature - initial_temperature
            transverse = ((C11 + C12) * alpha11 + C13 * alpha33) * delta_t
            axial = (2 * C13 * alpha11 + C33 * alpha33) * delta_t

            # Layer normal is global x. Phase A has local axes (x,y,z); phase B has
            # local axes (y,z,x). Ey/Ez are compatible; ex jumps at the interface.
            ex_a = (transverse - C12 * Ey - C13 * Ez) / C11
            ex_b = (axial - C13 * (Ey + Ez)) / C33
            sigma_yy_a = C12 * ex_a + C11 * Ey + C13 * Ez - transverse
            sigma_yy_b = C13 * ex_b + C11 * Ey + C12 * Ez - transverse
            sigma_zz_a = C13 * (ex_a + Ey) + C33 * Ez - axial
            sigma_zz_b = C13 * ex_b + C12 * Ey + C11 * Ez - transverse
            solution = sp.solve((sigma_yy_a + sigma_yy_b, sigma_zz_a + sigma_zz_b),
                                (Ey, Ez), dict=True)[0]
            expected = [0.5 * (ex_a + ex_b), Ey, Ez]
            for direction, expression in zip(("11", "22", "33"), expected):
                value = float(expression.subs(solution))
                actual = float(row[f"Ebar_{direction}"])
                if not math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-11):
                    raise ValueError(
                        f"laminate Ebar_{direction}={actual}, symbolic reference={value}")
            for direction in ("23", "13", "12"):
                if abs(float(row[f"Ebar_{direction}"])) > 1e-11:
                    raise ValueError("laminate has unexpected macroscopic shear")
            if float(row["meanStressNorm"]) > 1e-3:
                raise ValueError("laminate zero-average-stress condition was not met")

    def check_zero_response(self, history):
        for row in history:
            if float(row["meanStressNorm"]) > 1e-6:
                raise ValueError(f"zero-CTE response has nonzero mean stress: {row['meanStressNorm']}")
            if float(row["maxDamage"]) != 0.0 or float(row["meanDamage"]) != 0.0:
                raise ValueError("zero-CTE response unexpectedly evolved damage")
            for direction in ("11", "22", "33", "23", "13", "12"):
                if abs(float(row[f"Ebar_{direction}"])) > 1e-12:
                    raise ValueError("zero-CTE response has nonzero macroscopic strain")

    def check_uniform_crystal(self, workspace, history):
        config = json.loads((workspace / "input.json").read_text())
        elastic = config["MaterialParameters"]["LinearElasticityParameters"]
        thermal = config["MaterialParameters"]["ThermoElasticParameters"]
        initial_temperature = float(config["ModelParameters"]["InitialTemperature"]["value"])
        for row in history:
            temperature = float(row["T"])
            if float(row["meanStressNorm"]) > 1e-3:
                raise ValueError("uniform-orientation response has nonzero mean stress")
            constants = {
                name: float(elastic[f"{name}_0"]["value"])
                * evaluate_temperature_expression(
                    elastic[f"{name}_functionOfTemperature"]["value"], temperature)
                for name in ("C11", "C12", "C13", "C33", "C44")
            }
            expected_cbar = [
                constants["C11"], constants["C11"], constants["C33"],
                2 * constants["C44"], 2 * constants["C44"],
                constants["C11"] - constants["C12"],
            ]
            for index, expected in enumerate(expected_cbar, 1):
                actual = float(row[f"CbarDiag_{index}"])
                if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-3):
                    raise ValueError(
                        f"uniform-orientation CbarDiag_{index}={actual}, expected {expected}")
            for direction in ("11", "22", "33"):
                parameter_direction = "11" if direction == "22" else direction
                alpha0 = float(thermal[f"alpha{parameter_direction}_0"]["value"])
                multiplier = evaluate_temperature_expression(
                    thermal[f"alpha{parameter_direction}_functionOfTemperature"]["value"],
                    temperature)
                expected = alpha0 * multiplier * (temperature - initial_temperature)
                actual = float(row[f"Ebar_{direction}"])
                if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-11):
                    raise ValueError(
                        f"uniform-orientation Ebar_{direction}={actual}, expected {expected}")
            for direction in ("23", "13", "12"):
                if abs(float(row[f"Ebar_{direction}"])) > 1e-11:
                    raise ValueError(f"uniform-orientation Ebar_{direction} is nonzero")

    def compare_with_reference(self, target, workspace):
        reference_root = self.args.reference_run.resolve() / target / "workspace"
        if not reference_root.is_dir():
            raise ValueError(f"missing reference workspace: {reference_root}")
        reference_output = reference_root / "output"
        candidate_output = workspace / "output"
        reference_histories = list(reference_output.glob("*_macroscopic_history.csv"))
        candidate_histories = list(candidate_output.glob("*_macroscopic_history.csv"))
        if len(reference_histories) != 1 or len(candidate_histories) != 1:
            raise ValueError("A/B comparison requires exactly one macroscopic CSV per run")
        reference = self.read_commented_csv(reference_histories[0])
        candidate = self.read_commented_csv(candidate_histories[0])
        if len(reference) != len(candidate) or reference[0].keys() != candidate[0].keys():
            raise ValueError("A/B macroscopic CSV schemas or row counts differ")
        exact_columns = {"cycle", "step", "cgIterations"}
        max_scaled_difference = 0.0
        for row_number, (before, after) in enumerate(zip(reference, candidate), 1):
            for column in before:
                if column in exact_columns:
                    if before[column] != after[column]:
                        raise ValueError(
                            f"A/B {column} differs at row {row_number}: "
                            f"{before[column]} -> {after[column]}")
                    continue
                left, right = float(before[column]), float(after[column])
                scale = max(1.0, abs(left), abs(right))
                max_scaled_difference = max(max_scaled_difference, abs(left - right) / scale)
                if not math.isclose(left, right, rel_tol=1e-10,
                                    abs_tol=1e-12 * max(1.0, abs(left))):
                    raise ValueError(
                        f"A/B {column} differs at row {row_number}: {left} -> {right}")

        reference_pvd = next(reference_output.glob("*.pvd"))
        candidate_pvd = next(candidate_output.glob("*.pvd"))
        reference_fields = [node.attrib["file"] for node in ET.parse(reference_pvd).iter("DataSet")]
        candidate_fields = [node.attrib["file"] for node in ET.parse(candidate_pvd).iter("DataSet")]
        if reference_fields != candidate_fields:
            raise ValueError("A/B PVD field sequences differ")
        def normalized_vtu(path):
            return re.sub(
                rb"(?m)^#This file was generated by the deal\.II library on [^\r\n]*$",
                b"# deal.II generation timestamp normalized",
                path.read_bytes(),
            )

        for name in reference_fields:
            if normalized_vtu(reference_output / name) != normalized_vtu(candidate_output / name):
                raise ValueError(f"A/B VTU contents differ beyond the generation timestamp: {name}")

        progress = re.compile(
            r"Refinement cycle (\d+), temperature step (\d+), iteration (\d+) \(T = ([^)]+)\):"
            r"\n\s+Cells: (\d+), DoFs: (\d+), CG iterations = (\d+)")
        reference_log = (reference_root.parent / "run.log").read_text()
        candidate_log = (workspace.parent / "run.log").read_text()
        if progress.findall(reference_log) != progress.findall(candidate_log):
            raise ValueError("A/B temperature, convergence, cell/DoF, or CG histories differ")
        macro = re.compile(r"Ebar iterations this pass = (\d+), Cbar rebuilds so far = (\d+)")
        if macro.findall(reference_log) != macro.findall(candidate_log):
            raise ValueError("A/B macroscopic iteration or Cbar rebuild histories differ")
        return {
            "reference_run": str(self.args.reference_run.resolve()),
            "macroscopic_rows": len(candidate),
            "max_scaled_csv_difference": max_scaled_difference,
            "vtu_files_identical_except_timestamp": len(reference_fields),
            "convergence_history": "identical",
        }

    def check_history_rows(self, rows, final_temperature):
        if not math.isclose(float(rows[-1]["T"]), final_temperature, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"final temperature is {rows[-1]['T']}, expected {final_temperature}")
        for row in rows:
            for key, value in row.items():
                if key not in ("cycle", "step") and not finite(value):
                    raise ValueError(f"non-finite CSV value in {key}: {value}")


def parse_timeout_overrides(items):
    values = dict(DEFAULT_TIMEOUTS)
    for item in items:
        try:
            target, raw = item.split("=", 1)
            seconds = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid --set-timeout value: {item}")
        if target not in DEFAULT_TIMEOUTS or seconds <= 0:
            raise argparse.ArgumentTypeError(f"invalid --set-timeout value: {item}")
        values[target] = seconds
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Isolated, resource-limited verification for the FEA crystallite repository.")
    parser.add_argument("targets", nargs="*", help="explicit targets; use --list-targets to inspect")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-root", type=Path, default=ROOT / "verification-runs")
    parser.add_argument("--reference-run", type=Path,
                        help="previous verification run for full resolved-grain A/B comparison")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=8.0)
    parser.add_argument("--deal-ii-dir", default=os.environ.get("DEAL_II_DIR"))
    parser.add_argument("--set-timeout", action="append", default=[], metavar="TARGET=SECONDS")
    args = parser.parse_args()
    args.timeouts = parse_timeout_overrides(args.set_timeout)
    if args.list_targets:
        print("\n".join(["check", "build", "ab-resolved", "analytical-resolved",
                         "zero-response-resolved", *TARGET_ORDER]))
        return 0
    selected = set(args.targets)
    if "check" in selected:
        selected.remove("check")
        selected.update(CHECK_TARGETS)
    if "analytical-resolved" in selected:
        selected.remove("analytical-resolved")
        selected.update(RESOLVED_ANALYTICAL_TARGETS)
    if "zero-response-resolved" in selected:
        selected.remove("zero-response-resolved")
        selected.update(name for name in RESOLVED_ANALYTICAL_TARGETS
                        if name.startswith("zero-response-"))
    if "build" in selected:
        selected.remove("build")
        selected.update(BUILD_TARGETS)
    if "ab-resolved" in selected:
        selected.remove("ab-resolved")
        selected.update(RESOLVED_AB_TARGETS)
    unknown = sorted(selected - set(TARGET_ORDER))
    if unknown or not selected:
        parser.error(f"select at least one valid target; unknown: {unknown}")
    if args.jobs <= 0 or args.threads <= 0 or args.memory_gib <= 0:
        parser.error("--jobs, --threads, and --memory-gib must be positive")
    if args.reference_run and not args.reference_run.is_dir():
        parser.error(f"reference run does not exist: {args.reference_run}")
    targets = [target for target in TARGET_ORDER if target in selected]
    if args.dry_run:
        print(json.dumps({"targets": targets, "resources": {
            "jobs": args.jobs, "threads": args.threads,
            "memory_gib": args.memory_gib, "timeouts": args.timeouts,
        }}, indent=2))
        return 0
    return Runner(args, targets).run()


if __name__ == "__main__":
    sys.exit(main())
