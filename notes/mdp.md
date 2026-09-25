# Formal MDP and architecture (Dyna-Bound)

## MDP

At each placement (or reconcile) decision on an FCM:

| Symbol | Definition |
|--------|------------|
| \(S_t\) | Local control view: CRT entries (≤K), optional GRT summaries, service features, optional service-DAG stats |
| \(A_t\) | Index of a feasible CRT host, or escalate to GRT search, or defer to L3 cloud |
| \(R_t\) | SLA margin reward in \([0,1]\) minus optional cloud/migration penalties |
| γ | Discount for multi-step episodes (online updates use ρ-style trace; full PPO later) |

**Constraint:** \(|A_t|\) is bounded by CRT/GRT top-K — the colony control plane is part of the method.

## Stack

```
L1 sticky (deterministic SLA thresholds)     ← shared with DCBO
L2 colony-bounded learner                   ← VECTOR (A1) or GNN (A2)
L3 cloud fallback                           ← shared with DCBO
```

## Encoders

### A1 — VECTOR
Feature vector per candidate host = concat(host_local, service, bias) without message passing.
Score with hybrid: inverted DCBO \(J\) + blend × online linear score + ε-greedy.

### A2 — GNN (online GraphSAGE)
1. Host nodes = CRT candidates; edges = RTT affinity.
2. Two layers: \(h'=\tanh(W_{\mathrm{self}}h + W_{\mathrm{neigh}}Ah)\) with online-updated weights.
3. Context = service/DAG scalars + producer RTT + fan-in + co-location.
4. Readout score hybridized with inverted DCBO \(J\) via scale-aware \(\beta\).
5. Online delta-rule update on readout + both layers (cached activations).

Class: `org.fog.dynacolgnn.learn.OnlineColonyGnn`.

## Policies (CLI)

| CLI | Enum | Encoder | Paper role |
|-----|------|---------|------------|
| `colony_rl` | `COLONY_RL` | Vector (A1) | Hybrid ablation (often strongest) |
| `colony_gnn` | `COLONY_GNN` | Local graph (A2) | Hybrid ablation |
| `dcbo` | `DYNACOL_DCBO` | — | Baseline |

Both learners use cold-start colony overlay like `dcbo`. Manuscript brand:
**DynaCol-Hybrid** (not GNN-primary).
