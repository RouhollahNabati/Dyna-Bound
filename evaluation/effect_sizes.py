#!/usr/bin/env python3
"""Paired bootstrap CIs and Cohen's d for DynaCol-GNN journal grids.

Writes one CSV row per (nodes, scenario, comparison) with mean difference,
bootstrap 95% CI, paired Cohen's d, and Holm-adjusted p from the same paired t
used in holm_tests.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from holm_tests import (
    METHOD_DCBO,
    METHOD_GNN,
    METHOD_VEC,
    holm_adjust,
    load_trial_map,
    paired_t_pvalue,
    paired_vectors,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM = ROOT / "results" / "simulator_gnn_lock_s23_ft20"
DEFAULT_OUT = ROOT / "results" / "effect_s23_ft20.csv"

METHOD_DRL = "DRL-based"

COMPARISONS = [
    ("GNN_vs_DCBO", METHOD_GNN, METHOD_DCBO),
    ("Vector_vs_DCBO", METHOD_VEC, METHOD_DCBO),
    ("GNN_vs_Vector", METHOD_GNN, METHOD_VEC),
    ("GNN_vs_DRL", METHOD_GNN, METHOD_DRL),
    ("Vector_vs_DRL", METHOD_VEC, METHOD_DRL),
    ("DRL_vs_DCBO", METHOD_DRL, METHOD_DCBO),
]


def paired_cohen_d(xa: list[float], xb: list[float]) -> float:
    n = len(xa)
    if n < 2:
        return float("nan")
    diffs = [a - b for a, b in zip(xa, xb)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        return 0.0 if mean == 0 else (float("inf") if mean > 0 else float("-inf"))
    return mean / math.sqrt(var)


def bootstrap_ci(
    xa: list[float],
    xb: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(a-b). Returns (mean, lo, hi)."""
    n = len(xa)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = [a - b for a, b in zip(xa, xb)]
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int(math.floor((alpha / 2) * n_boot))
    hi_i = int(math.ceil((1 - alpha / 2) * n_boot)) - 1
    lo_i = max(0, min(n_boot - 1, lo_i))
    hi_i = max(0, min(n_boot - 1, hi_i))
    return mean, means[lo_i], means[hi_i]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_trial_map(args.sim)
    if not data:
        print(f"No trial rows under {args.sim}")
        return 1

    methods_present = {m for (m, _, _, _) in data}
    comparisons = [
        c for c in COMPARISONS if c[1] in methods_present and c[2] in methods_present
    ]
    # Always include the core three even if partially missing (rows will be empty).
    if not comparisons:
        comparisons = COMPARISONS[:3]

    cells = sorted({(n, s) for (_, n, s, _) in data})
    rows: list[dict] = []
    for nodes, scenario in cells:
        for label, a, b in comparisons:
            xa, xb = paired_vectors(data, a, b, nodes, scenario)
            if len(xa) < 2:
                continue
            mean_diff, t_stat, p_raw = paired_t_pvalue(xa, xb)
            d = paired_cohen_d(xa, xb)
            mean_b, lo, hi = bootstrap_ci(
                xa, xb, n_boot=args.boot, alpha=args.alpha, seed=args.seed + nodes
            )
            rows.append({
                "nodes": nodes,
                "scenario": scenario,
                "comparison": label,
                "n_pairs": len(xa),
                "mean_diff": round(mean_diff, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "cohen_d": round(d, 4) if d == d and abs(d) < 1e6 else "",
                "t_stat": round(t_stat, 4) if t_stat == t_stat else "",
                "p_raw": round(p_raw, 6),
            })

    by_nodes: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_nodes[int(r["nodes"])].append(r)
    for node_rows in by_nodes.values():
        ps = [float(r["p_raw"]) for r in node_rows]
        for r, p_h in zip(node_rows, holm_adjust(ps)):
            r["p_holm"] = round(p_h, 6)
            r["significant_holm"] = "Yes" if p_h < args.alpha else "No"
            r["ci_excludes_zero"] = (
                "Yes" if (r["ci_lo"] > 0 and r["ci_hi"] > 0)
                or (r["ci_lo"] < 0 and r["ci_hi"] < 0)
                else "No"
            )

    fields = [
        "nodes", "scenario", "comparison", "n_pairs",
        "mean_diff", "ci_lo", "ci_hi", "cohen_d",
        "t_stat", "p_raw", "p_holm", "significant_holm", "ci_excludes_zero",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["nodes"]), x["scenario"], x["comparison"])):
            w.writerow(r)
            print(
                f"N={r['nodes']:<4} {r['scenario']:<12} {r['comparison']:<14} "
                f"Δ={r['mean_diff']!s:>8}  CI=[{r['ci_lo']},{r['ci_hi']}]  "
                f"d={r['cohen_d']!s:>7}  p_holm={r['p_holm']:.4f}  "
                f"CI≠0={r['ci_excludes_zero']}"
            )
    print(f"\nWrote {args.out} ({len(rows)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
