#!/usr/bin/env python3
"""Paired SLA tests with Holm–Bonferroni correction for Dyna-Bound grids.

Compares Dyna-Bound and DynaCol-RL against DynaCol/DCBO (and GNN vs Vector)
per (nodes, scenario) using paired Student t on matched trial seeds.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM = ROOT / "results" / "simulator_gnn_lock_s23_ft5"
DEFAULT_OUT = ROOT / "results" / "holm_s23_ft5.csv"

METHOD_GNN = "Dyna-Bound"
METHOD_VEC = "DynaCol-RL"
METHOD_DCBO = "DynaCol/DCBO"
METHOD_DRL = "DRL-based"


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (m - rank)
        adj = min(1.0, max(raw, prev))
        adjusted[idx] = adj
        prev = adj
    return adjusted


def load_trial_map(sim: Path) -> dict[tuple[str, int, str, int], float]:
    """(method, nodes, scenario, trial) -> sla_pct; last row wins."""
    out: dict[tuple[str, int, str, int], float] = {}
    for path in sorted(sim.glob("trials_*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    key = (
                        row["method"],
                        int(float(row["nodes"])),
                        row["scenario"],
                        int(float(row["trial"])),
                    )
                    out[key] = float(row["sla_pct"])
                except (KeyError, TypeError, ValueError):
                    continue
    return out


def paired_vectors(
    data: dict[tuple[str, int, str, int], float],
    a: str,
    b: str,
    nodes: int,
    scenario: str,
) -> tuple[list[float], list[float]]:
    trials_a = {t for (m, n, s, t) in data if m == a and n == nodes and s == scenario}
    trials_b = {t for (m, n, s, t) in data if m == b and n == nodes and s == scenario}
    trials = sorted(trials_a & trials_b)
    xa = [data[(a, nodes, scenario, t)] for t in trials]
    xb = [data[(b, nodes, scenario, t)] for t in trials]
    return xa, xb


def paired_t_pvalue(xa: list[float], xb: list[float]) -> tuple[float, float, float]:
    """Two-sided paired t; returns (mean_diff_a_minus_b, t_stat, p_approx)."""
    n = len(xa)
    if n < 2:
        return float("nan"), float("nan"), 1.0
    diffs = [a - b for a, b in zip(xa, xb)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        # identical paired samples
        return mean, float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    se = math.sqrt(var / n)
    t = mean / se
    # Normal approximation for two-sided p (adequate for draft; n≈10).
    # Φ(|z|) via erfc.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return mean, t, min(1.0, max(0.0, p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    data = load_trial_map(args.sim)
    if not data:
        print(f"No trial rows under {args.sim}")
        return 1

    cells = sorted({(n, s) for (_, n, s, _) in data})
    comparisons = [
        ("GNN_vs_DCBO", METHOD_GNN, METHOD_DCBO),
        ("Vector_vs_DCBO", METHOD_VEC, METHOD_DCBO),
        ("GNN_vs_Vector", METHOD_GNN, METHOD_VEC),
    ]
    methods_present = {m for (m, _, _, _) in data}
    if METHOD_DRL in methods_present:
        comparisons.extend([
            ("GNN_vs_DRL", METHOD_GNN, METHOD_DRL),
            ("Vector_vs_DRL", METHOD_VEC, METHOD_DRL),
            ("DRL_vs_DCBO", METHOD_DRL, METHOD_DCBO),
        ])
    # Drop comparisons whose methods are absent.
    comparisons = [
        c for c in comparisons if c[1] in methods_present and c[2] in methods_present
    ]

    rows = []
    for nodes, scenario in cells:
        for label, a, b in comparisons:
            xa, xb = paired_vectors(data, a, b, nodes, scenario)
            mean_diff, t_stat, p_raw = paired_t_pvalue(xa, xb)
            rows.append({
                "nodes": nodes,
                "scenario": scenario,
                "comparison": label,
                "n_pairs": len(xa),
                "mean_sla_a": round(sum(xa) / len(xa), 4) if xa else "",
                "mean_sla_b": round(sum(xb) / len(xb), 4) if xb else "",
                "mean_diff_a_minus_b": round(mean_diff, 4) if xa else "",
                "t_stat": round(t_stat, 4) if xa and t_stat == t_stat else "",
                "p_raw": round(p_raw, 6),
            })

    # Holm within each nodes group (all scenarios × comparisons)
    by_nodes: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_nodes[int(r["nodes"])].append(r)
    for node_rows in by_nodes.values():
        ps = [float(r["p_raw"]) for r in node_rows]
        for r, p_h in zip(node_rows, holm_adjust(ps)):
            r["p_holm"] = round(p_h, 6)
            r["significant_holm"] = "Yes" if p_h < args.alpha else "No"

    fields = [
        "nodes", "scenario", "comparison", "n_pairs",
        "mean_sla_a", "mean_sla_b", "mean_diff_a_minus_b",
        "t_stat", "p_raw", "p_holm", "significant_holm",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["nodes"]), x["scenario"], x["comparison"])):
            w.writerow(r)
            print(
                f"N={r['nodes']:<4} {r['scenario']:<12} {r['comparison']:<14} "
                f"Δ={r['mean_diff_a_minus_b']!s:>8}  "
                f"p={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f}  {r['significant_holm']}"
            )
    print(f"\nWrote {args.out} ({len(rows)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
