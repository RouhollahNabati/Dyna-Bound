#!/usr/bin/env python3
"""Render schematic Figs 1–2 for the Dyna-Bound manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, text, fc="#E8EEF7", ec="#2F4A6E", fontsize=9, lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1A2433", wrap=True)
    return patch


def _arrow(ax, a, b, color="#4A607A"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.3, color=color, shrinkA=2, shrinkB=2,
    ))


def fig1_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Fig. 1 — Colony-bounded hybrid placement stack", fontsize=12, pad=8)

    # Overlay plane
    ax.add_patch(Rectangle((0.3, 0.35), 11.4, 1.55, linewidth=1.0,
                           edgecolor="#8AA0B8", facecolor="#F4F7FB", linestyle="--"))
    ax.text(0.5, 1.7, "DynaCol control plane (reused)", fontsize=8, color="#4A607A")
    _box(ax, (0.55, 0.55), 2.4, 1.0, "Cold-start\ncolony formation", fc="#F7F1E8")
    _box(ax, (3.2, 0.55), 2.4, 1.0, "FCM election\n& handover", fc="#F7F1E8")
    _box(ax, (5.85, 0.55), 2.5, 1.0, "CRT / GRT\ntop-K views", fc="#F7F1E8")
    _box(ax, (8.6, 0.55), 2.7, 1.0, "Bounded messaging\n& membership", fc="#F7F1E8")

    # Placement stack
    _box(ax, (0.8, 3.5), 2.6, 1.35, "L1 Sticky\nreuse current host\nif still feasible",
         fc="#D9E8D8", ec="#3F6B45")
    _box(ax, (4.3, 3.2), 3.5, 1.95,
         "L2 Dyna-Bound (this paper)\nCRT candidates ≤ K\n"
         "s = u_DCBO + β · f_θ(φ)\nscale-aware β → DCBO @ large N",
         fc="#D6E4F5", ec="#2F4A6E", fontsize=8.5)
    _box(ax, (8.6, 3.5), 2.6, 1.35, "L3 Cloud\nfallback when\nCRT/GRT fail",
         fc="#F0D9D9", ec="#7A3E3E")

    _arrow(ax, (3.4, 4.15), (4.3, 4.15))
    _arrow(ax, (7.8, 4.15), (8.6, 4.15))
    _arrow(ax, (6.05, 3.2), (6.05, 1.9))
    ax.text(6.2, 2.45, "actions ⊂ CRT/GRT", fontsize=8, color="#2F4A6E")

    ax.text(6.0, 5.7, "Service request / module placement", ha="center",
            fontsize=9, color="#1A2433")
    _arrow(ax, (6.0, 5.5), (6.0, 5.15))

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig2_encoders(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Fig. 2 — Vector ablation vs GraphSAGE encoder", fontsize=12, pad=8)

    # Shared input
    _box(ax, (4.0, 5.2), 4.0, 0.85, "CRT candidates + service request", fc="#EEF2F7")

    # Vector path
    ax.text(2.2, 4.7, "Vector (ablation)", ha="center", fontsize=9, color="#3F6B45")
    _box(ax, (0.5, 3.55), 3.4, 0.9, "Local host channels\n+ DAG scalars\n(no message passing)",
         fc="#D9E8D8", ec="#3F6B45", fontsize=8)
    _box(ax, (0.7, 2.15), 3.0, 0.85, "Linear scorer f_θ", fc="#D9E8D8", ec="#3F6B45")
    _arrow(ax, (2.2, 5.2), (2.2, 4.45))
    _arrow(ax, (2.2, 3.55), (2.2, 3.0))

    # GNN path
    ax.text(9.6, 4.7, "GNN (named method)", ha="center", fontsize=9, color="#2F4A6E")
    _box(ax, (7.9, 3.55), 3.6, 0.9, "RTT-affinity CRT graph\n+ producer / fan / coloc ctx",
         fc="#D6E4F5", ec="#2F4A6E", fontsize=8)
    _box(ax, (8.0, 2.15), 3.4, 0.85, "2-layer GraphSAGE\nHID=16 + readout",
         fc="#D6E4F5", ec="#2F4A6E", fontsize=8)
    _arrow(ax, (9.6, 5.2), (9.6, 4.45))
    _arrow(ax, (9.6, 3.55), (9.6, 3.0))

    # Hybrid merge
    _box(ax, (3.6, 0.55), 4.8, 1.05,
         "Hybrid score  s(i)=u_DCBO(i)+β·score_φ(i)\nε-greedy / argmax over ≤K candidates",
         fc="#F7F1E8", ec="#7A5A2E", fontsize=8.5)
    _arrow(ax, (2.2, 2.15), (4.4, 1.35))
    _arrow(ax, (9.6, 2.15), (7.6, 1.35))

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    fig1_architecture(FIG / "fig01_architecture.png")
    fig2_encoders(FIG / "fig02_encoders.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
