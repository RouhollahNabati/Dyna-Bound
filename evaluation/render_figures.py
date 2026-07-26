#!/usr/bin/env python3
"""Render JoS draft figures from locked summary CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_n(rows, n: int):
    return [r for r in rows if int(float(r["nodes"])) == n]


def bar_grouped(path: Path, title: str, series: dict[str, list[float]], scenarios: list[str], ylabel: str):
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    # Fig: surveillance N=100
    surv = load(RES / "medium_v2_summary.csv")
    s100 = rows_n(surv, 100)
    order = ["Normal Load", "Burst Load", "Churn"]
    labels = ["Normal", "Burst", "Churn"]

    def pick(rows, method):
        out = []
        for sc in order:
            hit = [r for r in rows if r["scenario"] == sc and r["method"] == method]
            out.append(float(hit[0]["sla_mean"]) if hit else float("nan"))
        return out

    # On surveillance medium_v2, GNN==RL; plot Hybrid vs DCBO
    bar_grouped(
        FIG / "fig_sla_n100_surveillance.png",
        "Surveillance DAG — SLA violation at N=100",
        {
            "Hybrid": pick(s100, "DynaCol-GNN"),
            "DCBO": pick(s100, "DynaCol/DCBO"),
        },
        labels,
        "SLA violation %",
    )

    # Multipipe champion lock (offline+FT s23)
    champ = load(RES / "s23_ft5_summary.csv")
    c100 = rows_n(champ, 100)
    bar_grouped(
        FIG / "fig_sla_n100_multipipe.png",
        "Multipipe DAG — champion GNN vs Vector vs DCBO (N=100)",
        {
            "GNN": pick(c100, "DynaCol-GNN"),
            "Vector": pick(c100, "DynaCol-RL"),
            "DCBO": pick(c100, "DynaCol/DCBO"),
        },
        labels,
        "SLA violation %",
    )

    # Earlier online multipipe (context)
    mp = load(RES / "multipipe_medium_summary.csv")
    m100 = rows_n(mp, 100)
    bar_grouped(
        FIG / "fig_sla_n100_online_context.png",
        "Multipipe DAG — online-only GNN vs Vector vs DCBO (N=100)",
        {
            "Online-GNN": pick(m100, "DynaCol-GNN"),
            "Vector": pick(m100, "DynaCol-RL"),
            "DCBO": pick(m100, "DynaCol/DCBO"),
        },
        labels,
        "SLA violation %",
    )

    # Legacy offline medium (pre-champion)
    off = load(RES / "offline_gnn_medium_summary.csv")
    o100 = rows_n(off, 100)
    bar_grouped(
        FIG / "fig_sla_n100_offline.png",
        "Multipipe DAG — earlier offline GNN (pre-champion, N=100)",
        {
            "Offline-GNN": pick(o100, "DynaCol-GNN"),
            "Vector": pick(o100, "DynaCol-RL"),
            "DCBO": pick(o100, "DynaCol/DCBO"),
        },
        labels,
        "SLA violation %",
    )

    # Scalability surveillance (hybrid vs dcbo, average over scenarios)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for method, label in (("DynaCol-GNN", "Hybrid"), ("DynaCol/DCBO", "DCBO")):
        xs, ys = [], []
        for n in (100, 300, 500):
            rr = [r for r in surv if int(float(r["nodes"])) == n and r["method"] == method]
            if not rr:
                continue
            xs.append(n)
            ys.append(float(np.mean([float(r["sla_mean"]) for r in rr])))
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Target nodes N")
    ax.set_ylabel("Mean SLA violation % (avg scenarios)")
    ax.set_title("Surveillance — SLA vs scale")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "fig_sla_scalability.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
