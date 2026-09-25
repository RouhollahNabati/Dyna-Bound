# Results narrative — Dyna-Bound (locked, JoS-complete)

## Primary claim

**Colony-bounded hybrid learning beats DCBO at N=100** on SLA under
normal/burst/churn at matched overhead (surveillance + multipipe champion),
with near-parity at N=1000. Named GNN encoder wins the multipipe **5-trial
champion** lock vs Vector; **10-trial** journal ranking is mixed; Holm at
n=10 is non-significant.

## Canonical tables

### Champion — `s23_ft5` (HID=16, offline+FT, ε=0.02)

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | **10.64** | 10.98 | 11.48 |
| Burst | **24.89** | 25.35 | 26.13 |
| Churn | **22.31** | 22.43 | 23.21 |

### Journal — `s23_ft10`

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | **11.09** | 11.34 | 11.60 |
| Burst | 25.89 | **25.62** | 26.71 |
| Churn | 22.91 | **22.61** | 23.23 |

Holm: `holm_s23_ft10.csv` — no cell significant at α=0.05 after correction.

### N=1000 — `s23_n1000_ft10`

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | 0.82 | **0.80** | 0.81 |
| Burst | **2.53** | 3.13 | 3.59 |
| Churn | **1.94** | 2.27 | 2.70 |

Overhead matched (~2112–2140). Holm: non-significant after correction.

### Baselines — `baselines_ft5` (FogPlan / Greedy / Static-DCBO)

Lower absolute SLA on multipipe N=100 is possible, but overhead ≪ colony plane
(FogPlan ~253, Greedy ~659, Static ~760 vs DynaCol ~1769) and colonies≠matched.
Primary claim stays within GNN/Vector/DCBO. Edgeward omitted (runtime).

## Contributions

- **C1:** Colony-bounded MDP over CRT/GRT after cold-start
- **C2:** Hybrid L2 + DCBO + scale-aware blend / large-N parity
- **C3:** GraphSAGE GNN (named) vs Vector ablation; champion GNN edge
- **C4:** iFogSim2 study + journal grids + baselines + figures fig01–fig11

## Artifacts

- Manuscript: [`manuscript_draft.md`](manuscript_draft.md)
- Figures: `../evaluation/figures/fig01`–`fig11` (+ captions)
- Refs: [`refs.bib`](refs.bib)
- Render: `../evaluation/render_jos_figures.py`
- Resume grids: `../evaluation/run_grid.py --resume`
