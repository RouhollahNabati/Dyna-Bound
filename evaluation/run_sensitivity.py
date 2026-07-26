#!/usr/bin/env python3
"""One-factor sensitivity sweeps for JoS revision (Burst, N=100, 5 trials).

Sweeps learnBlend, slaDeadlineScale, and epsilon around the champion lock while
holding other props fixed. Writes per-cell trial CSVs under distinct out dirs
and a combined summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
CFG = ROOT / "dynacol-gnn" / "configs"
RES = EVAL / "results"

WEIGHTS = str(ROOT / "dynacol-gnn" / "offline" / "data" / "gnn_weights_s23.json")

BASE_PROPS = {
    "dynacolgnn.appVariant": "multipipe",
    "dynacolgnn.learnBlend": "0.60",
    "dynacolgnn.slaDeadlineScale": "10.0",
    "dynacolgnn.epsilon": "0.02",
    "dynacolgnn.gnnFineTune": "true",
    "dynacolgnn.gnnWeights": WEIGHTS,
}

SWEEPS = {
    "blend": ("dynacolgnn.learnBlend", ["0.40", "0.60", "0.80"]),
    "deadline": ("dynacolgnn.slaDeadlineScale", ["8.0", "10.0", "12.0"]),
    "epsilon": ("dynacolgnn.epsilon", ["0", "0.02"]),
}


def run_grid(grid_path: Path, resume: bool, skip_compile: bool) -> None:
    cmd = [sys.executable, str(EVAL / "run_grid.py"), "--grid", str(grid_path)]
    if skip_compile:
        cmd.append("--skip-compile")
    if resume:
        cmd.append("--resume")
    subprocess.check_call(cmd)


def summarize_dir(sim: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(sim.glob("trials_*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def aggregate(rows: list[dict], factor: str, value: str) -> list[dict]:
    from collections import defaultdict
    import statistics

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    oh: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if int(float(r["nodes"])) != 100:
            continue
        key = (r["method"], r["scenario"])
        buckets[key].append(float(r["sla_pct"]))
        oh[key].append(float(r.get("overhead_norm") or 0.0))
    out = []
    for (method, scenario), vals in sorted(buckets.items()):
        out.append({
            "factor": factor,
            "value": value,
            "method": method,
            "scenario": scenario,
            "n_trials": len(vals),
            "sla_mean": round(statistics.mean(vals), 4),
            "sla_std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "overhead_mean": round(statistics.mean(oh[(method, scenario)]), 2)
            if oh[(method, scenario)] else "",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--factors",
        nargs="+",
        default=list(SWEEPS.keys()),
        choices=list(SWEEPS.keys()),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RES / "sensitivity_burst5_summary.csv",
    )
    args = parser.parse_args()

    tmp = CFG / "_sensitivity_generated"
    tmp.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    first = True

    for factor in args.factors:
        prop, values = SWEEPS[factor]
        for value in values:
            props = dict(BASE_PROPS)
            props[prop] = value
            tag = f"{factor}_{value.replace('.', 'p')}"
            out_subdir = f"simulator_sens_{tag}"
            grid = {
                "name": f"sens_{tag}",
                "description": f"Sensitivity {factor}={value} Burst N=100 × 5",
                "nodes": [100],
                "scenarios": ["burst"],
                "policies": ["colony_gnn", "colony_rl", "dcbo"],
                "trials": 5,
                "seed_base": 42,
                "out_subdir": out_subdir,
                "java_props": props,
            }
            grid_path = tmp / f"grid_{tag}.json"
            grid_path.write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
            print(f"=== sensitivity {factor}={value} ===", flush=True)
            run_grid(grid_path, resume=args.resume, skip_compile=args.skip_compile or not first)
            first = False
            sim = RES / out_subdir
            all_rows.extend(aggregate(summarize_dir(sim), factor, value))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "factor", "value", "method", "scenario", "n_trials",
        "sla_mean", "sla_std", "overhead_mean",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Wrote {args.out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
