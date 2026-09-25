#!/usr/bin/env python3
"""Collect multipipe trajectories for offline GIN/PPO training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IFOG = ROOT / "ifogsim2"
OUT_DIR = Path(__file__).resolve().parent / "data"
TRAJ = OUT_DIR / "traj_multipipe.jsonl"


def java_bin() -> str:
    home = os.environ.get("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    c = Path(home) / "bin" / "java"
    return str(c) if c.is_file() else "java"


def classpath() -> str:
    out = IFOG / "out" / "production" / "iFogSim"
    jars = [str(out)]
    for pattern in (IFOG / "jars").glob("*.jar"):
        jars.append(str(pattern))
    commons = IFOG / "jars" / "commons-math3-3.5"
    if commons.is_dir():
        jars.extend(str(p) for p in commons.glob("*.jar"))
    return ":".join(jars)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["normal", "burst", "churn"],
        help="Scenarios to collect (default: all three)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRAJ.exists() and not args.append:
        TRAJ.unlink()

    if not args.skip_compile:
        subprocess.check_call(["bash", str(IFOG / "compile.sh"), "incremental"], cwd=IFOG)

    runs = []
    for scenario in args.scenarios:
        for trial in range(1, args.trials + 1):
            runs.append((scenario, trial, args.seed_base + args.nodes + trial * 17 + hash(scenario) % 100))

    for scenario, trial, seed in runs:
        csv = OUT_DIR / f"collect_{scenario}_{trial}.csv"
        print(f"COLLECT N={args.nodes} scenario={scenario} trial={trial}", flush=True)
        cmd = [
            java_bin(),
            "-Ddynacolgnn.appVariant=multipipe",
            "-Ddynacolgnn.learnBlend=0.60",
            "-Ddynacolgnn.slaDeadlineScale=10.0",
            "-Ddynacolgnn.collectEpsilon=0.15",
            f"-Ddynacolgnn.trajScenario={scenario}",
            f"-Ddynacolgnn.trajLog={TRAJ.resolve()}",
            "-cp",
            classpath(),
            "org.fog.test.dynacol.DynaColEvaluationRunner",
            "colony_gnn",
            str(args.nodes),
            scenario,
            str(trial),
            str(seed),
            str(csv),
        ]
        subprocess.check_call(cmd, cwd=IFOG)

    n = sum(1 for _ in TRAJ.open(encoding="utf-8"))
    print(f"Done. {n} decisions -> {TRAJ}", flush=True)
    meta = {"traj": str(TRAJ), "decisions": n, "runs": len(runs), "nodes": args.nodes}
    (OUT_DIR / "collect_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
