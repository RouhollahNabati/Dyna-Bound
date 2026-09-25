# Dyna-Bound

Colony-bounded **hybrid learning** (Dyna-Bound) for fog service placement on the DynaCol
control plane (cold-start colonies, FCM handover, CRT/GRT), without changing
the main-paper evaluation tree.

**Working title:** *Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement*

## Public repositories

- JoS artifact: https://github.com/RouhollahNabati/Dyna-Bound
- Parent DynaCol runtime (separate, unchanged): https://github.com/RouhollahNabati/DynaCol

**Target venue:** *The Journal of Supercomputing*, Section — Artificial Intelligence  
https://link.springer.com/journal/11227

> Manuscript brand: **Dyna-Bound**. Vector is the flat-encoder ablation.
> Primary empirical claim vs DCBO; GraphSAGE ≈ Vector (honest).

## Relation to main DynaCol

| | Main paper (`evaluation/`, `org.fog.dynacol`) | This line (`dyna-bound/`) |
|---|---|---|
| Control plane | Colony formation, FCM, handover, CRT/GRT | Reused as-is |
| L1 / L3 | Sticky host / cloud fallback | Reused as-is |
| L2 placement | DCBO (deterministic + light attractiveness) | **New:** colony-bounded hybrid learner |
| Results | `evaluation/results/` | `dyna-bound/evaluation/results/` only |

Do not write CSVs, LaTeX, or figures into the top-level `evaluation/` tree.

## Layout

```
dyna-bound/
├── README.md
├── notes/          # concept, MDP, abstract, outline, related work
├── configs/        # experiment grids
├── evaluation/     # runners + results (this line only)
└── Java: ifogsim2/src/org/fog/dynacolgnn/
```

## Variants

| ID | CLI | Role in paper |
|----|-----|----------------|
| Hybrid-Vector | `colony_rl` | Flat linear ablation |
| Hybrid-GNN | `colony_gnn` | **Online 2-layer GraphSAGE** on CRT + service context |
| DCBO | `dcbo` | Control-plane baseline (same L1/L3) |

Primary **claim** = colony-bounded hybrid (Dyna-Bound) vs DCBO.
Vector ablation shows encoder parity; do not oversell GraphSAGE≻Vector.

## Status (locked evidence)

**Surveillance `medium_v2`:** N=100 learning beats DCBO; N≥300 parity.

## Offline GIN/PPO (path 2)

Pipeline: [`offline/`](offline/) — collect → PyTorch train → import.

**Medium locked** ([`offline_gnn_medium_summary.csv`](evaluation/results/offline_gnn_medium_summary.csv),
2634 traj decisions):

| N=100 | Offline GraphSAGE | Vector | DCBO |
|-------|-------------|--------|------|
| Normal | 10.54 | **10.06** | 11.07 |
| Burst | 24.20 | **24.12** | 26.64 |
| Churn | 21.45 | **20.62** | 22.83 |

Hybrid learning ≪ DCBO; Vector ≳ Offline GraphSAGE. Keep GraphSAGE as encoder, not title hero.

Details: [`notes/results_narrative.md`](notes/results_narrative.md) ·
[`notes/abstract_en.md`](notes/abstract_en.md) ·
[`notes/outline.md`](notes/outline.md) ·
**[`notes/manuscript_draft.md`](notes/manuscript_draft.md)** (full JoS draft) ·
[`evaluation/figures/`](evaluation/figures/) (SLA bars + scalability)

## Quick start

```bash
bash ifogsim2/compile.sh incremental
python3 dyna-bound/evaluation/run_grid.py
# multipipe stress workload:
python3 dyna-bound/evaluation/run_grid.py \
  --grid dyna-bound/configs/grid_a2_multipipe_medium.json
```

## Quick links

- Concept: [`notes/concept.md`](notes/concept.md)
- Manuscript: [`notes/manuscript_draft.md`](notes/manuscript_draft.md)
- Outline: [`notes/outline.md`](notes/outline.md)
- MDP: [`notes/mdp.md`](notes/mdp.md)
- Abstract: [`notes/abstract_en.md`](notes/abstract_en.md)
- Related work: [`notes/related_work.md`](notes/related_work.md)
- Java: [`../ifogsim2/src/org/fog/dynacolgnn/`](../ifogsim2/src/org/fog/dynacolgnn/)
