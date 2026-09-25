# org.fog.dynacolgnn

Colony-bounded hybrid L2 placement for DynaCol-Hybrid.

| Encoder | Class | Role |
|---------|-------|------|
| VECTOR | `OnlineLinearPolicy` | Flat hybrid ablation (`colony_rl`) |
| GNN | `OnlineColonyGnn` | 2-layer GraphSAGE on structure adj (RTT+coloc+DAG pull) + service context (`colony_gnn`) |

Shared: `ColonyBoundedPlacement`, `ColonyFeatureEncoder`, `ServiceDagFeatures`.

GNN-v2 structure adjacency (default on): `-Ddynacolgnn.structureAdj=true`.
RTT-only ablation: `-Ddynacolgnn.structureAdj=false`.
CRT ablation (global top-K hosts ≫ CRT): `-Ddynacolgnn.crtAblate=true`
(`-Ddynacolgnn.crtAblateK=50` default).
