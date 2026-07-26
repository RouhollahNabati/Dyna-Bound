#!/usr/bin/env python3
"""Run DynaCol-GNN smoke / grid trials into dynacol-gnn/evaluation/results/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IFOG = ROOT / "ifogsim2"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "simulator"
DEFAULT_GRID = ROOT / "dynacol-gnn" / "configs" / "grid_a2_smoke.json"

# CLI scenario key -> CSV scenario label written by the Java runner.
SCENARIO_CSV = {
    "normal": "Normal Load",
    "burst": "Burst Load",
    "churn": "Churn",
}


def java_bin() -> str:
    home = os.environ.get("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    candidate = Path(home) / "bin" / "java"
    if candidate.is_file():
        return str(candidate)
    return "java"


def classpath() -> str:
    out = IFOG / "out" / "production" / "iFogSim"
    jars = [str(out)]
    for pattern in (IFOG / "jars").glob("*.jar"):
        jars.append(str(pattern))
    commons = IFOG / "jars" / "commons-math3-3.5"
    if commons.is_dir():
        jars.extend(str(p) for p in commons.glob("*.jar"))
    return ":".join(jars)


def compile_ifog() -> None:
    subprocess.check_call(["bash", str(IFOG / "compile.sh"), "incremental"], cwd=IFOG)


def run_trial(policy: str, nodes: int, scenario: str, trial: int, seed: int,
              out_dir: Path, java_props: dict[str, str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"trials_{policy}.csv"
    cmd = [java_bin()]
    if java_props:
        for key, value in java_props.items():
            cmd.append(f"-D{key}={value}")
    cmd.extend([
        "-cp",
        classpath(),
        "org.fog.test.dynacol.DynaColEvaluationRunner",
        policy,
        str(nodes),
        scenario,
        str(trial),
        str(seed),
        str(csv_path),
    ])
    subprocess.check_call(cmd, cwd=IFOG)
    return csv_path


def load_done(out_dir: Path, policy: str) -> set[tuple[int, str, int]]:
    """Return completed (nodes, csv_scenario, trial) for a policy CSV."""
    path = out_dir / f"trials_{policy}.csv"
    done: set[tuple[int, str, int]] = set()
    if not path.is_file():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((
                    int(float(row["nodes"])),
                    row.get("scenario", ""),
                    int(float(row["trial"])),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override CSV output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip trials already present in CSVs; do not clear files")
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not args.skip_compile:
        compile_ifog()

    out_dir = args.out_dir
    if out_dir is None:
        rel = grid.get("out_subdir")
        out_dir = (Path(__file__).resolve().parent / "results" / rel) if rel else DEFAULT_OUT
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    java_props = grid.get("java_props") or {}
    seed_base = int(grid.get("seed_base", 42))
    trials = int(grid.get("trials", 1))

    if not args.resume:
        for policy in grid["policies"]:
            stale = out_dir / f"trials_{policy}.csv"
            if stale.exists():
                stale.unlink()
                print(f"Cleared {stale.name}", flush=True)

    done_by_policy = {
        policy: load_done(out_dir, policy) if args.resume else set()
        for policy in grid["policies"]
    }

    skipped = 0
    for nodes in grid["nodes"]:
        for scenario in grid["scenarios"]:
            csv_scenario = SCENARIO_CSV.get(scenario, scenario)
            for policy in grid["policies"]:
                for trial in range(1, trials + 1):
                    key = (int(nodes), csv_scenario, trial)
                    if args.resume and key in done_by_policy[policy]:
                        skipped += 1
                        print(
                            f"SKIP policy={policy} N={nodes} scenario={scenario} trial={trial}",
                            flush=True,
                        )
                        continue
                    seed = seed_base + nodes + trial * 17
                    print(
                        f"RUN policy={policy} N={nodes} scenario={scenario} trial={trial}",
                        flush=True,
                    )
                    run_trial(policy, int(nodes), scenario, trial, seed, out_dir, java_props)
                    if args.resume:
                        done_by_policy[policy].add(key)
    print(f"Done. CSVs under {out_dir} (skipped={skipped})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
