# Abstract draft (English) — Dyna-Bound

**Working title:** Colony-Bounded GNN Hybrid Learning for Cold-Start Fog Service Placement

**Target:** *The Journal of Supercomputing*, Section — Artificial Intelligence

**Method name in prose:** Dyna-Bound (Vector encoder as ablation)

---

Self-organizing fog colonies can bootstrap from a zero-colony state and keep
control state bounded in colony/global resource tables (CRT/GRT), but most
learning-based placement methods either assume a ready control plane or act
over the full set of fog nodes. This paper presents **Dyna-Bound**, a
colony-bounded hybrid placer that reuses DynaCol’s cold-start formation,
manager handover, and L1 sticky / L3 cloud stack, while replacing only the L2
host choice with an online **GraphSAGE GNN** over CRT candidates with
service-graph context. The learner is hybridized with DCBO and uses a
scale-aware blend that falls back to exact DCBO at large scale. A flat vector
encoder is the ablation of the same MDP. On iFogSim2 under normal/burst/churn,
colony-bounded hybrid learning reduces mean SLA violations at N=100 versus DCBO
at matched control overhead on surveillance and multipipe DAGs. On the
multipipe champion lock (offline weights + fine-tune, 5 trials), Dyna-Bound
also edges the vector ablation under all three scenarios. A 10-trial journal
grid confirms directional gains versus DCBO (mixed encoder ranking; Holm
non-significant at n=10). At N=1000, methods remain near parity with matched
overhead.

**Keywords:** fog computing, service placement, graph neural networks, hybrid
learning, self-organizing systems, cold start, CRT/GRT, iFogSim2

---

## Locked numbers (canonical)

### Champion multipipe — `s23_ft5` (5 trials, N=100)

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | **10.64** | 10.98 | 11.48 |
| Burst | **24.89** | 25.35 | 26.13 |
| Churn | **22.31** | 22.43 | 23.21 |

### Journal multipipe — `s23_ft10` (10 trials, N=100)

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | **11.09** | 11.34 | 11.60 |
| Burst | 25.89 | **25.62** | 26.71 |
| Churn | 22.91 | **22.61** | 23.23 |

### Large-N — `s23_n1000_ft10` (10 trials, N=1000)

| Scenario | GNN | Vector | DCBO |
|----------|-----|--------|------|
| Normal | 0.82 | **0.80** | 0.81 |
| Burst | **2.53** | 3.13 | 3.59 |
| Churn | **1.94** | 2.27 | 2.70 |

Full prose: [`manuscript_draft.md`](manuscript_draft.md). Figures: `fig01`–`fig11`.
