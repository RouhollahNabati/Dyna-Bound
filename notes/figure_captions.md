# Figure captions — Dyna-Bound (JoS)

Files under `dyna-bound/evaluation/figures/` (PNG 300 dpi + PDF).

**Fig. 1.** `fig01_architecture` — Colony-bounded hybrid placement stack: reused
DynaCol control plane (cold start, FCM, CRT/GRT) with L1 sticky → L2
Dyna-Bound → L3 cloud.

**Fig. 2.** `fig02_encoders` — Vector ablation versus GraphSAGE encoder
pipelines feeding the same hybrid score \(s(i)=u_{\mathrm{DCBO}}+\beta\cdot
f_\theta(\phi)\).

**Fig. 3.** `fig03_multipipe_dag` — Multipipe stress application DAG (fan-out,
skip, and join) used for champion and journal grids.

**Fig. 4.** `fig04_sla_surveillance` — Surveillance DAG mean SLA violation at
\(N{=}100\) (± std) for hybrid vs DCBO (`medium_v2`).

**Fig. 5.** `fig05_sla_champion` — Multipipe tuning lock (`s23_ft5`, 5 trials):
GNN vs Vector vs DCBO at \(N{=}100\) (hyperparameter/weight lock, not primary stats).

**Fig. 6.** `fig06_sla_journal10` — Multipipe journal grid (`s23_ft20`, 20
trials) at \(N{=}100\), including tabular DRL when present.

**Fig. 7.** `fig07_scalability` — Surveillance mean SLA versus target scale
\(N\in\{100,300,500\}\).

**Fig. 8.** `fig08_overhead` — Normalized control overhead at \(N{=}100\)
(multipipe colony-plane trials); methods remain matched.

**Fig. 9.** `fig09_sla_n1000` — Multipipe large-\(N\) parity at \(N{=}1000\)
(10 trials).

**Fig. 10.** `fig10_baselines` — Named baselines (FogPlan, Greedy-Nearest,
Static-DCBO) vs Dyna-Bound / Vector / DCBO on multipipe \(N{=}100\) (5 trials).
Edgeward omitted due to prohibitive multipipe runtime.

**Fig. 11.** `fig11_holm` — Holm-adjusted pairwise \(p\)-values for SLA at
\(N{=}100\) (journal grid).

**Fig. 12.** `fig12_pareto` — SLA vs normalized control overhead (Burst,
\(N{=}100\)): colony-plane methods vs flat baselines.

**Fig. 13.** `fig13_sensitivity` — One-factor sweeps of blend \(\beta\),
deadline scale, and \(\varepsilon\) on Burst \(N{=}100\) (5 trials).
