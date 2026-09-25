#!/usr/bin/env python3
"""Render JoS-quality figures (300 dpi PNG + PDF) for Dyna-Bound draft."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, FancyBboxPatch as FBP, Rectangle

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly discrete palette (Wong-inspired)
C_GNN = "#0072B2"
C_VEC = "#E69F00"
C_DCBO = "#009E73"
C_DRL = "#CC79A7"
C_OTHER = ["#56B4E9", "#D55E00", "#F0E442", "#999999"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("wrote", out)
    plt.close(fig)


def rows_n(rows, n: int):
    return [r for r in rows if int(float(r["nodes"])) == n]


def pick_mean_std(rows, method: str, scenarios: list[str]):
    means, stds = [], []
    for sc in scenarios:
        hit = [r for r in rows if r["scenario"] == sc and r["method"] == method]
        if not hit:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        means.append(float(hit[0]["sla_mean"]))
        stds.append(float(hit[0].get("sla_std") or 0.0))
    return means, stds


def grouped_bars(stem: str, title: str, series: dict[str, tuple[list[float], list[float]]],
                 scenarios: list[str], ylabel: str, colors: dict[str, str] | None = None):
    x = np.arange(len(scenarios))
    n = len(series)
    width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (name, (means, stds)) in enumerate(series.items()):
        color = (colors or {}).get(name)
        ax.bar(
            x + (i - (n - 1) / 2) * width, means, width,
            yerr=stds, capsize=2.5, label=name, color=color,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(4, n), loc="upper right")
    fig.tight_layout()
    save(fig, stem)


def _box(ax, xy, w, h, text, fc="#E8EEF7", ec="#2F4A6E", fontsize=8):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.1, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1A2433")


def _arrow(ax, a, b, color="#4A607A"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color=color, shrinkA=1, shrinkB=1,
    ))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title("Colony-bounded hybrid placement stack")
    ax.add_patch(Rectangle((0.3, 0.3), 11.4, 1.45, linewidth=1.0,
                           edgecolor="#8AA0B8", facecolor="#F4F7FB", linestyle="--"))
    ax.text(0.45, 1.55, "DynaCol control plane (reused)", fontsize=8, color="#4A607A")
    _box(ax, (0.5, 0.45), 2.5, 0.95, "Cold-start\ncolony formation", fc="#F7F1E8")
    _box(ax, (3.2, 0.45), 2.4, 0.95, "FCM election\n& handover", fc="#F7F1E8")
    _box(ax, (5.8, 0.45), 2.5, 0.95, "CRT / GRT\ntop-K views", fc="#F7F1E8")
    _box(ax, (8.5, 0.45), 2.9, 0.95, "Bounded messaging\n& membership", fc="#F7F1E8")
    _box(ax, (0.7, 3.25), 2.7, 1.25, "L1 Sticky\nreuse host if feasible", fc="#D9E8D8", ec="#3F6B45")
    _box(ax, (4.2, 2.95), 3.6, 1.85,
         "L2 Dyna-Bound\nCRT candidates ≤ K\n"
         "s = u_DCBO + β · f_θ(φ)\nβ → 0 at large N",
         fc="#D6E4F5", ec="#2F4A6E", fontsize=7.5)
    _box(ax, (8.5, 3.25), 2.8, 1.25, "L3 Cloud\nfallback", fc="#F0D9D9", ec="#7A3E3E")
    _arrow(ax, (3.4, 3.85), (4.2, 3.85))
    _arrow(ax, (7.8, 3.85), (8.5, 3.85))
    _arrow(ax, (6.0, 2.95), (6.0, 1.75))
    ax.text(6.15, 2.25, "actions ⊂ CRT/GRT", fontsize=8, color="#2F4A6E")
    ax.text(6.0, 5.55, "Service / module placement request", ha="center", fontsize=9)
    _arrow(ax, (6.0, 5.35), (6.0, 4.9))
    fig.tight_layout()
    save(fig, "fig01_architecture")


def fig_encoders():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title("Vector ablation vs GraphSAGE encoder")
    _box(ax, (3.8, 4.85), 4.4, 0.7, "CRT candidates + service request", fc="#EEF2F7")
    ax.text(2.2, 4.4, "Vector (ablation)", ha="center", fontsize=9, color="#3F6B45")
    _box(ax, (0.5, 3.25), 3.4, 0.85, "Local host + DAG scalars\n(no message passing)",
         fc="#D9E8D8", ec="#3F6B45", fontsize=7.5)
    _box(ax, (0.7, 1.95), 3.0, 0.75, "Linear scorer f_θ", fc="#D9E8D8", ec="#3F6B45")
    _arrow(ax, (2.2, 4.85), (2.2, 4.1))
    _arrow(ax, (2.2, 3.25), (2.2, 2.7))
    ax.text(9.6, 4.4, "GNN (named method)", ha="center", fontsize=9, color="#2F4A6E")
    _box(ax, (7.8, 3.25), 3.6, 0.85, "RTT-affinity CRT graph\n+ producer / fan / coloc",
         fc="#D6E4F5", ec="#2F4A6E", fontsize=7.5)
    _box(ax, (8.0, 1.95), 3.2, 0.75, "2-layer GraphSAGE HID=16", fc="#D6E4F5", ec="#2F4A6E", fontsize=7.5)
    _arrow(ax, (9.6, 4.85), (9.6, 4.1))
    _arrow(ax, (9.6, 3.25), (9.6, 2.7))
    _box(ax, (3.4, 0.35), 5.2, 0.95,
         "Hybrid score  s(i)=u_DCBO(i)+β·score_φ(i)\nε-greedy / argmax over ≤K",
         fc="#F7F1E8", ec="#7A5A2E", fontsize=7.5)
    _arrow(ax, (2.2, 1.95), (4.2, 1.2))
    _arrow(ax, (9.6, 1.95), (7.8, 1.2))
    fig.tight_layout()
    save(fig, "fig02_encoders")


def fig_multipipe_dag():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Multipipe stress application DAG")
    nodes = {
        "cam": (0.4, 2.2, "CAMERA"),
        "md": (2.4, 2.2, "motion_\ndetector"),
        "fe": (4.5, 3.3, "feature_\nextractor"),
        "od": (6.6, 3.3, "object_\ndetector"),
        "ot": (8.8, 3.8, "object_\ntracker"),
        "ba": (11.0, 3.8, "behavior_\nanalyzer"),
        "fu": (8.8, 1.0, "fusion_\nengine"),
        "al": (11.0, 1.0, "alert_\ndispatcher"),
    }
    for _, (x, y, lab) in nodes.items():
        _box(ax, (x, y), 1.9, 0.85, lab, fc="#E8EEF7", fontsize=7)
    edges = [
        ("cam", "md"), ("md", "fe"), ("fe", "od"), ("od", "ot"), ("ot", "ba"),
        ("od", "fu"), ("fu", "al"), ("md", "fu"),
    ]
    pos = {k: (v[0] + 0.95, v[1] + 0.42) for k, v in nodes.items()}
    for a, b in edges:
        _arrow(ax, pos[a], pos[b])
    ax.text(7.0, 0.25, "fan-out / skip / join; SLA deadline scale = 10", ha="center",
            fontsize=8, color="#4A607A")
    fig.tight_layout()
    save(fig, "fig03_multipipe_dag")


def fig_sla_from_summary(summary: Path, stem: str, title: str, methods: list[tuple[str, str]],
                         n: int = 100):
    if not summary.is_file():
        print("skip missing", summary)
        return
    rows = rows_n(load(summary), n)
    order = ["Normal Load", "Burst Load", "Churn"]
    labels = ["Normal", "Burst", "Churn"]
    colors = {"GNN": C_GNN, "Dyna-Bound": C_GNN, "Hybrid": C_GNN,
              "Vector": C_VEC, "DynaCol-RL": C_VEC,
              "DCBO": C_DCBO, "DynaCol/DCBO": C_DCBO}
    series = {}
    color_map = {}
    for label, method in methods:
        means, stds = pick_mean_std(rows, method, order)
        series[label] = (means, stds)
        color_map[label] = colors.get(label, colors.get(method, C_OTHER[0]))
    grouped_bars(stem, title, series, labels, "SLA violation %", color_map)


def fig_scalability():
    path = RES / "medium_v2_summary.csv"
    if not path.is_file():
        print("skip scalability")
        return
    surv = load(path)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for method, label, color in (
        ("Dyna-Bound", "Hybrid", C_GNN),
        ("DynaCol/DCBO", "DCBO", C_DCBO),
    ):
        xs, ys, es = [], [], []
        for n in (100, 300, 500):
            rr = [r for r in surv if int(float(r["nodes"])) == n and r["method"] == method]
            if not rr:
                continue
            vals = [float(r["sla_mean"]) for r in rr]
            xs.append(n)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=es, marker="o", label=label, color=color, capsize=3)
    ax.set_xlabel("Target nodes N")
    ax.set_ylabel("Mean SLA violation % (avg scenarios)")
    ax.set_title("Surveillance — SLA vs scale")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig07_scalability")


def fig_overhead(trial_dir: Path, stem: str = "fig08_overhead"):
    if not trial_dir.is_dir():
        print("skip overhead", trial_dir)
        return
    # mean overhead by method/scenario at N=100
    buckets: dict[tuple[str, str], list[float]] = {}
    for path in trial_dir.glob("trials_*.csv"):
        for row in load(path):
            if int(float(row["nodes"])) != 100:
                continue
            key = (row["method"], row["scenario"])
            buckets.setdefault(key, []).append(float(row["overhead_norm"]))
    methods = ["Dyna-Bound", "DynaCol-RL", "DynaCol/DCBO"]
    labels_m = ["GNN", "Vector", "DCBO"]
    scenarios = ["Normal Load", "Burst Load", "Churn"]
    labels = ["Normal", "Burst", "Churn"]
    series = {}
    colors = {"GNN": C_GNN, "Vector": C_VEC, "DCBO": C_DCBO}
    for m, lab in zip(methods, labels_m):
        means, stds = [], []
        for sc in scenarios:
            vals = buckets.get((m, sc), [])
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        series[lab] = (means, stds)
    grouped_bars(stem, "Matched control overhead at N=100 (multipipe champion)",
                 series, labels, "Normalized control overhead", colors)


def fig_holm_heatmap(path: Path, stem: str = "fig11_holm"):
    if not path.is_file():
        print("skip holm", path)
        return
    rows = load(path)
    comps = sorted({r["comparison"] for r in rows})
    scens = ["Normal Load", "Burst Load", "Churn"]
    mat = np.full((len(comps), len(scens)), np.nan)
    for r in rows:
        if int(float(r["nodes"])) != 100:
            continue
        i = comps.index(r["comparison"])
        j = scens.index(r["scenario"])
        mat[i, j] = float(r["p_holm"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.imshow(mat, cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(scens)))
    ax.set_xticklabels(["Normal", "Burst", "Churn"])
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(comps)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black", fontsize=8)
    ax.set_title("Holm-adjusted p-values (N=100, journal trials)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, stem)


def fig_baselines():
    base = RES / "baselines_ft5_summary.csv"
    champ = RES / "s23_ft5_summary.csv"
    if not base.is_file() or not champ.is_file():
        print("skip baselines fig (need summaries)")
        return
    order = ["Normal Load", "Burst Load", "Churn"]
    labels = ["Normal", "Burst", "Churn"]
    # merge methods of interest
    series = {}
    colors = {}
    for label, method, color, src in [
        ("GNN", "Dyna-Bound", C_GNN, champ),
        ("Vector", "DynaCol-RL", C_VEC, champ),
        ("DCBO", "DynaCol/DCBO", C_DCBO, champ),
    ]:
        rows = rows_n(load(src), 100)
        series[label] = pick_mean_std(rows, method, order)
        colors[label] = color
    brows = rows_n(load(base), 100)
    # discover baseline method names in CSV
    methods = sorted({r["method"] for r in brows})
    for i, m in enumerate(methods):
        short = m.replace("DynaCol/", "").replace("FogPlan", "FogPlan").split("/")[-1]
        if len(short) > 14:
            short = short[:14]
        series[short] = pick_mean_std(brows, m, order)
        colors[short] = C_OTHER[i % len(C_OTHER)]
    grouped_bars("fig10_baselines", "Multipipe N=100 — baselines vs Dyna-Bound (5 trials)",
                 series, labels, "SLA violation %", colors)


def fig_pareto():
    """SLA vs overhead scatter: colony plane + flat baselines (Burst preferred)."""
    colony = RES / "s23_ft20_summary.csv"
    if not colony.is_file():
        colony = RES / "s23_ft10_summary.csv"
    base = RES / "baselines_ft5_summary.csv"
    if not colony.is_file() or not base.is_file():
        print("skip Pareto (need colony + baseline summaries)")
        return
    scen = "Burst Load"
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    points = []
    for src, mapping in [
        (colony, {
            "Dyna-Bound": ("GNN", C_GNN),
            "DynaCol-RL": ("Vector", C_VEC),
            "DynaCol/DCBO": ("DCBO", C_DCBO),
            "DRL-based": ("Tabular-DRL", C_DRL),
        }),
        (base, None),
    ]:
        for r in load(src):
            if int(float(r["nodes"])) != 100 or r["scenario"] != scen:
                continue
            method = r["method"]
            if mapping is not None:
                if method not in mapping:
                    continue
                label, color = mapping[method]
                marker = "o"
            else:
                label = method.replace("DynaCol/", "")[:12]
                color = C_OTHER[len(points) % len(C_OTHER)]
                marker = "s"
            x = float(r["overhead_mean"])
            y = float(r["sla_mean"])
            ax.scatter([x], [y], s=70, c=color, marker=marker, zorder=3)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
            points.append((x, y, label))
    ax.set_xlabel("Normalized control overhead")
    ax.set_ylabel("SLA violation %")
    ax.set_title("Pareto view — multipipe Burst at N=100")
    fig.tight_layout()
    save(fig, "fig12_pareto")


def fig_sensitivity():
    path = RES / "sensitivity_burst5_summary.csv"
    if not path.is_file():
        print("skip sensitivity fig")
        return
    rows = [r for r in load(path) if r["method"] in ("Dyna-Bound", "DynaCol-RL", "DynaCol/DCBO")]
    factors = ["blend", "deadline", "epsilon"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
    method_color = {"Dyna-Bound": C_GNN, "DynaCol-RL": C_VEC, "DynaCol/DCBO": C_DCBO}
    method_label = {"Dyna-Bound": "GNN", "DynaCol-RL": "Vector", "DynaCol/DCBO": "DCBO"}
    for ax, factor in zip(axes, factors):
        sub = [r for r in rows if r["factor"] == factor]
        values = sorted({r["value"] for r in sub}, key=lambda v: float(v))
        x = np.arange(len(values))
        for i, (method, color) in enumerate(method_color.items()):
            means, stds = [], []
            for v in values:
                hit = [r for r in sub if r["value"] == v and r["method"] == method]
                if hit:
                    means.append(float(hit[0]["sla_mean"]))
                    stds.append(float(hit[0]["sla_std"]))
                else:
                    means.append(float("nan"))
                    stds.append(0.0)
            width = 0.25
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=2,
                   color=color, label=method_label[method],
                   error_kw={"elinewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_xlabel(factor)
        ax.set_title(factor)
        if factor == "blend":
            ax.set_ylabel("SLA violation %")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("One-factor sensitivity — Burst N=100 (5 trials)", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig13_sensitivity")


def main() -> int:
    fig_architecture()
    fig_encoders()
    fig_multipipe_dag()

    fig_sla_from_summary(
        RES / "medium_v2_summary.csv", "fig04_sla_surveillance",
        "Surveillance DAG — SLA at N=100 (± std)",
        [("Hybrid", "Dyna-Bound"), ("DCBO", "DynaCol/DCBO")],
    )
    fig_sla_from_summary(
        RES / "s23_ft5_summary.csv", "fig05_sla_champion",
        "Multipipe tuning lock — SLA at N=100 (± std, 5 trials)",
        [("GNN", "Dyna-Bound"), ("Vector", "DynaCol-RL"), ("DCBO", "DynaCol/DCBO")],
    )
    journal = RES / "s23_ft20_summary.csv"
    journal_n = 20 if journal.is_file() else 10
    if not journal.is_file():
        journal = RES / "s23_ft10_summary.csv"
    journal_series = [
        ("GNN", "Dyna-Bound"), ("Vector", "DynaCol-RL"), ("DCBO", "DynaCol/DCBO"),
    ]
    if journal.is_file():
        methods = {r["method"] for r in load(journal)}
        if "DRL-based" in methods:
            journal_series.append(("Tabular-DRL", "DRL-based"))
    fig_sla_from_summary(
        journal, "fig06_sla_journal10",
        f"Multipipe journal grid — SLA at N=100 (± std, {journal_n} trials)",
        journal_series,
    )
    fig_scalability()
    oh_dir = RES / "simulator_gnn_lock_s23_ft20"
    if not oh_dir.is_dir():
        oh_dir = RES / "simulator_gnn_lock_s23_ft5"
    fig_overhead(oh_dir)
    if (RES / "s23_n1000_ft10_summary.csv").is_file():
        fig_sla_from_summary(
            RES / "s23_n1000_ft10_summary.csv", "fig09_sla_n1000",
            "Multipipe large-N parity — SLA at N=1000 (± std, 10 trials)",
            [("GNN", "Dyna-Bound"), ("Vector", "DynaCol-RL"), ("DCBO", "DynaCol/DCBO")],
            n=1000,
        )
    else:
        print("defer fig09 until N=1000 summary exists")
    fig_baselines()
    holm = RES / "holm_s23_ft20.csv"
    if not holm.is_file():
        holm = RES / "holm_s23_ft10.csv"
    fig_holm_heatmap(holm)
    fig_pareto()
    fig_sensitivity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
