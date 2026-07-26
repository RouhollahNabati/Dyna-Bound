# Configs

Experiment grids and feature flags for DynaCol-GNN only.

Suggested files (to add when implementation starts):

| File | Purpose |
|------|---------|
| `grid_a2.json` | N × workload × policy × trials |
| `ablation_a2.json` | CRT/GRT/graph ablations |
| `train_a2.yaml` | learner hyperparameters (lr, γ, K, GNN depth) |

Keep paths pointing at `dynacol-gnn/evaluation/results/`, never at
`../../evaluation/results/`.

## Files

| File | Purpose |
|------|---------|
| `grid_a2_smoke.json` | Small runnable smoke grid |
| `grid_a2_full.json` | Target journal grid (heavy) |
| `train_a2.yaml` | Placeholder for phase-2 trainer hyperparams |
