# org.fog.dynacolgnn

Colony-bounded hybrid L2 placement for DynaCol-Hybrid.

| Encoder | Class | Role |
|---------|-------|------|
| VECTOR | `OnlineLinearPolicy` | Flat hybrid ablation (`colony_rl`) |
| GNN | `OnlineColonyGnn` | 2-layer GraphSAGE on CRT + service context (`colony_gnn`) |

Shared: `ColonyBoundedPlacement`, `ColonyFeatureEncoder`, `ServiceDagFeatures`.
