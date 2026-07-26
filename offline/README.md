# Offline GIN trainer (path 2)

Train a graph policy outside iFogSim, then import weights into Java
`OnlineColonyGnn` (online fine-tune optional).

The trainer uses **reward-advantage behavioral cloning + structure ranking**
(not classic PPO), despite the historical script name `train_gin_ppo.py`.

## Setup

```bash
cd dynacol-gnn/offline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
python3 collect_traj.py --trials 5 --skip-compile
python3 collect_traj.py --trials 5 --append --skip-compile --seed-base 1000
python3 train_gin_ppo.py --traj data/traj_multipipe.jsonl --out data/gnn_weights.json --epochs 60
python3 ../evaluation/run_grid.py --grid ../configs/grid_offline_gnn_medium.json --skip-compile
```

## Locked medium (5 trials, N∈{100,300})

Source: `evaluation/results/offline_gnn_medium_summary.csv` (2634 training decisions).

| N=100 | Offline-GNN | Vector | DCBO |
|-------|-------------|--------|------|
| Normal | 10.54 | **10.06** | 11.07 |
| Burst | 24.20 | **24.12** | 26.64 |
| Churn | 21.45 | **20.62** | 22.83 |

N=300 ≈ parity. Primary win remains hybrid vs DCBO, not GNN≻Vector.

## Properties

| Property | Role |
|----------|------|
| `dynacolgnn.trajLog` | JSONL collection |
| `dynacolgnn.collectEpsilon` | Exploration while collecting |
| `dynacolgnn.gnnWeights` | Absolute path to JSON weights |
