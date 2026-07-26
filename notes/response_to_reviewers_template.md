# Response to Reviewers — JoS Major Revision

Manuscript: *Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement*
(retitled per `notes/revision_gate.md`)

We thank the reviewers for the careful assessment. Below we map each major
concern to the revision artifact.

## Major comments

### M1 — Statistical power / GNN claim vs journal evidence
**Action.** Primary table is now the **n=20** multipipe journal grid
(`s23_ft20_summary.csv`) with paired bootstrap 95% CIs and Cohen's *d*
(`effect_s23_ft20.csv`) plus Holm adjustment (`holm_s23_ft20.csv`). Hybrid vs
DCBO is **Holm-significant on all three loads** with CIs excluding zero. The
5-trial lock is retained only as a **tuning / weight lock**. Title retitled to
emphasize colony-bounded *hybrid* learning; GraphSAGE remains the named encoder
without a universal GNN≻Vector claim (encoder gaps non-significant).

### M2 — Offline import drives GNN edge
**Action.** Method § documents the offline objective as
**reward-advantage BC + structure ranking** (not PPO). Online-only ablation
table remains. Claims distinguish encoder structure from the offline+fine-tune
pipeline.

### M3 — Incremental novelty / opaque DynaCol prior
**Action.** Added BibTeX `@unpublished{nabati_dynacol_cluster2026}` (Cluster
Computing, under review) and **Appendix** reprinting cold-start, handover, and
DCBO L1–L3 algorithms for self-containment without re-claiming them.

### M4 — Same-plane learning baseline
**Action.** Added tabular ε-greedy attractiveness learner (`drl` /
DRL-based) on the **same cold-start CRT/GRT plane** (overlay fixed to
COLD_START). Named honestly (not DQN). On the n=20 lock it is a strong SLA
control; hybrids remain the DCBO-anchored contribution with matched overhead vs
DCBO. Flat FogPlan/Greedy/Static-DCBO remain for absolute SLA context with
Pareto Fig. 12.

### M5 — Practical scope of learning (gate at 450)
**Action.** Sensitivity Fig. 13 sweeps blend β, deadline scale, and ε around
the lock. Discussion clarifies that large-*N* parity via exact DCBO
delegation is by design.

### M6 — Thin method for AI section
**Action.** Expanded MDP features (dims 8/8), reward shaping equation, online
updates, offline loss, complexity \(O(K_{\mathrm{crt}})\), and default
\(K_{\mathrm{crt}}{=}5\), \(K_{\mathrm{grt}}{=}3\).

## Minor comments
- Simulation-only: kept as explicit limitation; testbed listed as future work.
- Edgeward omitted: stated with runtime rationale.
- Surveillance GNN=Vector: emphasized that structured encoder evidence is
  multipipe/offline-path.
- Table baselines: Pareto Fig. 12 pairs SLA with overhead.
- Contributions restored in Introduction (C1–C4).

## Checklist of new artifacts
- [x] `evaluation/results/s23_ft20_summary.csv`
- [x] `evaluation/results/holm_s23_ft20.csv`
- [x] `evaluation/results/effect_s23_ft20.csv`
- [x] `evaluation/results/sensitivity_burst5_summary.csv`
- [x] `evaluation/figures/fig12_pareto.{png,pdf}`
- [x] `evaluation/figures/fig13_sensitivity.{png,pdf}`
- [x] `notes/revision_gate.md`
- [x] Appendix DynaCol algorithms in `latex/main.tex`
- [x] Declarations (funding, competing interests, contributions, data/code)
- [x] `latex/main.pdf` on Springer Nature `sn-jnl` (9 pp., two-column `iicol`);
      article backup: `latex/main_article.tex`
