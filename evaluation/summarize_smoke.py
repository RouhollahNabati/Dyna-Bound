#!/usr/bin/env python3
"""Summarize DynaCol-GNN trial CSVs into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM = ROOT / "results" / "simulator"
DEFAULT_OUT = ROOT / "results" / "smoke_summary.csv"


def load_rows(sim: Path):
    """Load trial rows; keep the last row per (method, nodes, scenario, trial)."""
    uniq = {}
    for path in sorted(sim.glob("trials_*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get("method", ""),
                    row.get("nodes", ""),
                    row.get("scenario", ""),
                    row.get("trial", ""),
                )
                uniq[key] = row
    return list(uniq.values())


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = load_rows(args.sim)
    if not rows:
        print(f"No CSVs under {args.sim}")
        return 1

    groups = defaultdict(list)
    for row in rows:
        key = (row.get("method", ""), int(float(row["nodes"])), row.get("scenario", ""))
        groups[key].append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "nodes", "scenario", "n_trials",
        "sla_mean", "sla_std", "p95_mean", "p95_std",
        "overhead_mean", "colonies_mean",
    ]
    lines = []
    print(f"{'method':<14} {'N':>4} {'scenario':<12} {'SLA%':>10} {'P95':>10} {'oh':>10} {'col':>5}")
    print("-" * 72)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups, key=lambda k: (k[1], k[2], k[0])):
            method, nodes, scenario = key
            g = groups[key]
            sla = [x for x in (fnum(r, "sla_pct") for r in g) if x is not None]
            p95 = [x for x in (fnum(r, "p95_ms") for r in g) if x is not None]
            oh = [x for x in (fnum(r, "overhead_norm") for r in g) if x is not None]
            col = [x for x in (fnum(r, "colonies") for r in g) if x is not None]
            rec = {
                "method": method,
                "nodes": nodes,
                "scenario": scenario,
                "n_trials": len(g),
                "sla_mean": round(statistics.mean(sla), 3) if sla else "",
                "sla_std": round(statistics.stdev(sla), 3) if len(sla) > 1 else 0.0,
                "p95_mean": round(statistics.mean(p95), 3) if p95 else "",
                "p95_std": round(statistics.stdev(p95), 3) if len(p95) > 1 else 0.0,
                "overhead_mean": round(statistics.mean(oh), 3) if oh else "",
                "colonies_mean": round(statistics.mean(col), 2) if col else "",
            }
            writer.writerow(rec)
            print(
                f"{method:<14} {nodes:>4} {scenario:<12} "
                f"{rec['sla_mean']:>10} {rec['p95_mean']:>10} "
                f"{rec['overhead_mean']:>10} {rec['colonies_mean']:>5}"
            )
            lines.append(rec)
    print(f"\nWrote {args.out} ({len(lines)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
