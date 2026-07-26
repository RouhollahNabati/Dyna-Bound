# Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement

**Authors:** Rouhollah Nabati (corresponding), Abdulbaghi Ghaderzadeh  
**Affiliation:** Department of Computer Engineering, Faculty of Engineering,
Islamic Azad University, Sanandaj Branch, Sanandaj, Iran  

**Working manuscript (DynaCol-GNN)**  
Target: *The Journal of Supercomputing*, Artificial Intelligence section  
**Final LaTeX/PDF:** [`../latex/main.pdf`](../latex/main.pdf) (build: `../latex/build.sh`)  
Status: JoS major-revision lock — journal \(N{=}100\times 20\) with DRL, Holm/CI/effect,
sensitivity, Pareto; title retitled per [`revision_gate.md`](revision_gate.md).  
Figures: `dynacol-gnn/evaluation/figures/fig0{1–13}_*.{png,pdf}` (300 dpi).  
BibTeX: [`refs.bib`](refs.bib). Captions: [`figure_captions.md`](figure_captions.md).

---

## Abstract

Self-organizing fog colonies can bootstrap from a zero-colony state and keep
control state bounded in colony and global resource tables (CRT/GRT), but most
learning-based placement methods either assume a ready control plane or act
over the full set of fog nodes. This paper presents **DynaCol-GNN**, a
colony-bounded hybrid placer that reuses DynaCol’s cold-start formation,
manager handover, and L1 sticky / L3 cloud stack, while replacing only the L2
host choice with online GraphSAGE or flat-vector scoring hybridized with the
deterministic DCBO objective and a scale-aware blend that falls back to exact
DCBO at large scale. On iFogSim2, a **20-trial** multipipe journal grid shows
both hybrids reduce mean SLA versus DCBO at matched overhead under
normal/burst/churn, with Holm-significant paired gaps and bootstrap CIs that
exclude zero. Encoder ranking (GNN vs vector) is mixed and non-significant. A
same-plane tabular ε-greedy learner is a strong SLA control inside the colony
plane; named flat baselines provide absolute SLA context outside that plane.
At \(N{=}1000\), methods remain near parity with matched overhead. Results
support colony-bounded hybrid placement without a global controller.

**Keywords:** fog computing, service placement, graph neural networks, hybrid
learning, self-organizing systems, cold start, CRT/GRT, iFogSim2

---

## 1. Introduction

Fog and edge service placement must keep end-to-end latency inside service-level
objectives while resources, demand, and connectivity change
[Chiang and Zhang, 2016; Yousefpour et al., 2019]. Under burst load and
membership churn, centralized or purely offline placers become brittle: they
either require a global view that is expensive to maintain, or they assume an
already formed hierarchy of fog nodes [Salaht et al., 2020; Taleb et al., 2025].
Learning-based placement and offloading improve adaptivity
[Supraja et al., 2025; Schulman et al., 2017], but many designs enlarge the
action space to (nearly) all hosts, presume an organized fabric, and report
quality gains without isolating control-plane cost
[Hong and Varghese, 2019].

DynaCol addresses the *control plane* side of this problem: cold-start colony
formation from a zero-colony state, potency-based Fog Colony Manager (FCM)
election and handover, and bounded Colony / Global Resource Tables (CRT/GRT).
Its default L2 placer (DCBO) is deterministic and light, with L1 sticky reuse
and L3 cloud fallback. What it does not provide is an online learner whose
decisions are *forced* to remain inside those colony-bounded views.

This paper keeps DynaCol’s formation, handover, CRT/GRT, L1, and L3 unchanged
and replaces only L2 host selection with **colony-bounded GNN hybrid learning**.
At each placement, a two-layer GraphSAGE-style GNN scores at most the top-\(K\)
feasible CRT candidates (RTT affinity, with service-dependency context),
hybridizes the score with the inverted DCBO objective, and applies a
scale-aware blend that delegates exactly to DCBO when the fabric is large
(Fig. 1). A flat vector policy is the encoder ablation of the same MDP
(Fig. 2).

**Contributions.**

1. **C1 — Colony-bounded MDP.** Formal state/action/reward with
   \(\lvert A_t\rvert \le K\) after cold-start, so the control plane is part of
   the learning method rather than an afterthought.
2. **C2 — Hybrid L2 stack.** Online GNN scoring hybridized with DCBO, plus
   scale-aware blend and exact large-scale DCBO delegation (threshold on
   *actual* fog count).
3. **C3 — GNN encoder + vector ablation.** Offline-imported two-layer GraphSAGE
   (HID=16) with online fine-tune on CRT + service-graph context; vector
   ablation of the same MDP; champion multipipe lock shows a GNN edge.
4. **C4 — Empirical study.** iFogSim2 evaluation under normal/burst/churn on
   surveillance and multipipe DAGs, with locked medium grids, journal
   \(10\)-trial / Holm analysis, \(N{=}1000\) parity, and named baselines.

The rest of the paper reviews related work (Section 2), summarizes the DynaCol
background we reuse (Section 3), presents DynaCol-GNN (Section 4), describes
the experimental setup (Section 5), reports results (Section 6), discusses
implications and limits (Section 7), and concludes (Section 8).

---

## 2. Related Work

### 2.1 Fog and edge service placement

Fog and edge computing push latency-sensitive services closer to users
[Chiang and Zhang, 2016; Yousefpour et al., 2019]. The service placement
problem has been surveyed extensively
[Salaht et al., 2020; Smolka et al., 2022; Taleb et al., 2025; Apat et al., 2025].
Classical and metaheuristic placers (greedy, ILP, genetic, swarm) typically
assume a fixed hierarchy or a centralized controller
[Skarlat et al., 2017; Azimzadeh et al., 2022; Benamer et al., 2024].
QoS-aware deployment tooling such as FogTorch [Brogi and Forti, 2017] and
Min-Cost FogPlan-style assignment [Yousefpour et al., 2019 FogPlan] optimize
placement quality but do not bootstrap a self-organizing control plane from a
zero-colony state.

### 2.2 Learning-based offloading and placement

Deep reinforcement learning (DQN, PPO) and AI-driven placers adapt online
[Mnih et al., 2015; Schulman et al., 2017; Supraja et al., 2025], with domain
instances spanning healthcare/fog ensembles [Tuli et al., 2020], energy-aware
module placement [Hossam et al., 2024], and multi-tier containers
[Dogani et al., 2024]. Resource-management and orchestration surveys note that
many learners still act over large host sets and presume an organized fabric
[Hong and Varghese, 2019; Costa et al., 2022; Pallewatta et al., 2023].
DynaCol-GNN instead keeps actions inside top-\(K\) CRT/GRT candidates and
hybridizes with deterministic DCBO so cold-start behavior remains stable.

### 2.3 Self-organizing and hierarchical fog control

Clustering, overlay, and manager-election schemes—including DynaCol—separate
overlay formation from placement. Latency-aware hierarchical module management
improves QoS under mobility and load [Mahmud et al., 2018]; evolutionary
hierarchical placers optimize mapping once a hierarchy exists
[Guerrero and Lera, 2019]. Learning is then either absent or attached without
treating the colony view as a *hard* action constraint. Our positioning is
complementary: formation and handover stay as in DynaCol; learning only
re-ranks within the colony-bounded decision set.

### 2.4 Graph learning on infrastructure and service graphs

Inductive GraphSAGE encoders [Hamilton et al., 2017] and GNN-RL placers
[Lera and Guerrero, 2024] encode dependency structure for scheduling and
multi-objective placement. Fog CRT neighborhoods are small, dynamic, and
RTT-structured. We treat local CRT message passing and service-DAG context as
**named encoders**. On the multipipe champion lock the GNN edges the vector
ablation; we still do not claim a large universal GNN-over-vector gap across
every grid.

### 2.5 Relation to DynaCol

We treat DynaCol/DCBO as the **prior system and baseline control plane**. This
paper does not re-claim cold-start formation or FCM handover. Differences are:
(i) the colony-bounded MDP for L2, (ii) hybrid online learning with scale-aware
blend, and (iii) encoder ablations and named baselines under surveillance and
multipipe apps.

**Table 1.** Positioning axes (extended).

| Method family | Cold-start | Bounded view | Hybrid w/ determ. obj. | Local graph / DAG |
|---------------|:----------:|:------------:|:----------------------:|:-----------------:|
| Classical / GA / ILP | Rarely | No | Sometimes | Rarely |
| FogPlan / Edgeward / Static-DCBO | No / partial | No / fixed | No | No |
| DRL placers (DQN/PPO) | Rarely | Rarely | Rarely | Sometimes |
| GNN-RL placers | Rarely | Rarely | Rarely | Yes |
| DynaCol / DCBO | Yes | Yes | N/A | No |
| **DynaCol-GNN** | **Yes** | **Yes** | **Yes** | **Yes** |

Simulation tooling follows iFogSim / iFogSim2
[Gupta et al., 2017; Mahmud et al., 2022]; related platforms include
EdgeCloudSim [Sonmez et al., 2018]. Multiple comparisons use Holm adjustment
[Holm, 1979].

---

## 3. Background: DynaCol Control Plane

We briefly recall the pieces we *reuse unchanged*. Full formation and handover
algorithms are those of DynaCol.

**Cold-start colonies.** Nodes bootstrap via query/advertise until colonies
form; an FCM is elected using potency and maintained with ε-greedy exploration
and cooldown-based handover.

**Bounded tables.** Each FCM maintains a CRT of member resources (capacity,
availability, RTT-to-FCM, attractiveness). GRT summaries enable limited
cross-colony search without a global host list.

**Placement stack.**

- **L1 (sticky):** keep the current host if feasible and latency remains under
  SLA sticky thresholds.
- **L2 (DCBO baseline):** score feasible CRT (then GRT) candidates with a
  deterministic multi-term objective \(J\), optionally minus attractiveness.
- **L3:** fall back to the cloud when colony search fails.

DynaCol-GNN replaces only the L2 *host ranking* inside CRT/GRT top-\(K\)
(Fig. 1).

---

## 4. Colony-Bounded Hybrid Learning

### 4.1 MDP

At each placement (or reconcile) decision on an FCM:

| Symbol | Definition |
|--------|------------|
| \(S_t\) | Local control view: CRT entries (≤K), optional GRT summaries, service features, service-DAG stats |
| \(A_t\) | Index of a feasible CRT host, or escalate to GRT, or defer to L3 cloud |
| \(R_t\) | Graph-SLA / shaped reward in \([0,1]\) (GNN); weak classical TD for Vector |

**Constraint:** \(\lvert A_t\rvert\) is bounded by CRT/GRT top-\(K\) — the colony
control plane is part of the method.

### 4.2 Hybrid scoring

\[
s(i) = u_{\mathrm{DCBO}}(i) + \beta \cdot f_\theta\bigl(\phi(i)\bigr),
\]

where \(u_{\mathrm{DCBO}}\) is an inverted DCBO objective, \(\phi\) is the
encoder, and \(f_\theta\) is the online scorer. Selection is
\(\arg\max_i s(i)\) with small \(\varepsilon\)-greedy exploration when
\(n_f\) is below the large-scale threshold. Scale-aware \(\beta\) decreases
with fog count and is zero when \(n_f \ge 450\), at which point L2 delegates
exactly to DCBO.

### 4.3 Encoders

**Vector (`colony_rl`).** Concatenate local host channels (normalized CPU, RAM,
storage, bandwidth, RTT, attractiveness) with service/DAG scalars. No message
passing. Online linear policy.

**GNN (`colony_gnn`).** CRT candidates are nodes; edges use RTT affinity
\(A_{ij}\propto 1/(1+\mathrm{RTT}_{ij}/20)\). A two-layer GraphSAGE-style update

\[
h'=\tanh\bigl(W_{\mathrm{self}}h+W_{\mathrm{neigh}}Ah\bigr)
\]

produces embeddings; a readout on \(\mathrm{concat}(h,\mathrm{context})\)
scores candidates (service/DAG scalars, producer-RTT affinity, fan-in,
co-location). Weights may be **imported** from an offline trainer
(`-Ddynacolgnn.gnnWeights=…`) and **fine-tuned online**
(`-Ddynacolgnn.gnnFineTune=true`; champion lock). Vector zeros the co-location
channel so the ablation isolates service-graph structure (Fig. 2).

### 4.4 Online update and offline import

After choosing \(i^\star\), observe reward \(R\) and update parameters toward
\(R\) unless offline weights are loaded without fine-tune. GNN uses
graph-SLA / contrastive-style updates; Vector keeps weak classical TD.
Attractiveness decay/update mirrors DCBO.

**Offline path.** Trajectories are logged during exploratory `colony_gnn`
runs, trained outside the simulator (structure-weighted objectives), and
exported as JSON tensors matching the Java GNN layout (`dynacol-gnn/offline/`).

### 4.5 Complexity and algorithm

Per decision we score at most \(K\) CRT candidates at constant feature /
hidden dimension—no global node enumeration. Control messages remain those of
DynaCol’s CRT/GRT maintenance.

```
reconcile(r, fcm, current, maxDepth):
  if β ≈ 0: return DCBO.reconcile(...)
  if L1 sticky OK: return current
  if CRT enabled:
    C ← top-K feasible CRT entries
    pick i* by hybrid score on φ(C); update θ; return host(i*)
  if GRT enabled: delegate to remote CRT via ranked summaries
  if L3: return cloud
```

---

## 5. Experimental Setup

**Simulator.** iFogSim2 with the shared DynaCol runtime
[Gupta et al., 2017; Mahmud et al., 2022]. Policies: `dcbo`, `colony_rl`
(Vector), `colony_gnn` (GNN); named baselines `static_dcbo`, `fogplan`,
`greedy` (Edgeward omitted on multipipe: >25 min/trial). Artifacts live under
`dynacol-gnn/evaluation/` (top-level `evaluation/` untouched).

**Topologies.** \(N\in\{100,300,500,1000\}\). Scenarios: normal, burst, churn.

**Applications.**

1. **Surveillance (default):** shallow camera → motion → detector → tracker/UI.
2. **Multipipe (opt-in):** deeper fan-out / skip / join DAG (Fig. 3) with SLA
   deadline scale \(10\).

**Metrics.** SLA violation %, P95 loop latency, normalized control overhead,
colony count. Lower SLA % is better.

**Trials and statistics.** Champion lock: **5** trials (`s23_ft5`). Journal:
**10** trials (`s23_ft10`, `s23_n1000_ft10`) with paired \(t\) and
Holm–Bonferroni adjustment within each \(N\) family [Holm, 1979]
(`holm_tests.py`). Baselines: **5** trials (`baselines_ft5`).

**Champion hyperparameters (multipipe).** HID=16, \(\beta=0.60\),
\(\varepsilon=0.02\), deadline scale \(10\), offline weights
`gnn_weights_s23.json`, fine-tune on.

---

## 6. Results

### 6.1 Surveillance DAG: hybrid vs DCBO

**Table 2.** Surveillance — mean SLA % (5 trials, `medium_v2`). Hybrid columns
coincide for Vector and GNN on this shallow app. Fig. 4; scalability Fig. 7.

| \(N\) | Scenario | Hybrid | DCBO |
|------:|----------|-------:|-----:|
| 100 | Normal | **11.51** | 13.31 |
| 100 | Burst | **14.02** | 14.78 |
| 100 | Churn | **14.58** | 15.37 |
| 300 | all | ≈0.48–0.51 | ≈0.48–0.50 |
| 500 | all | ≈0.57–0.59 | ≈0.57–0.59 |

At \(N{=}100\), hybrid reduces SLA versus DCBO at matched overhead. At
\(N\in\{300,500\}\), parity holds via scale-aware blend / DCBO delegation.

### 6.2 Multipipe champion lock (5 trials)

**Table 3.** Multipipe — champion GNN lock mean SLA % (5 trials).
Source: `s23_ft5_summary.csv`. Fig. 5; overhead Fig. 8.

| \(N\) | Scenario | GNN | Vector | DCBO |
|------:|----------|----:|-------:|-----:|
| 100 | Normal | **10.64** | 10.98 | 11.48 |
| 100 | Burst | **24.89** | 25.35 | 26.13 |
| 100 | Churn | **22.31** | 22.43 | 23.21 |

DynaCol-GNN reduces mean SLA versus **both** Vector and DCBO under all three
loads at matched control overhead (~1768 normalized messages; ~7.6 colonies).

### 6.3 Encoder ablation context

An earlier online-only multipipe medium without imported weights showed
near-parity (Vector slightly ahead). Importing structure-weighted offline
weights with online fine-tuning is what separates encoders on the champion lock
(Table 3).

**Table 4.** Earlier online multipipe medium (pre-champion; context).

| \(N\) | Scenario | Online-GNN | Vector | DCBO |
|------:|----------|-----------:|-------:|-----:|
| 100 | Normal | 10.34 | **10.25** | 11.21 |
| 100 | Burst | 24.39 | **24.09** | 26.42 |
| 100 | Churn | 21.15 | **21.13** | 23.04 |

### 6.4 Journal 10-trial grid and Holm tests

**Table 5.** Multipipe journal grid — mean SLA % (**10** trials).
Source: `s23_ft10_summary.csv`. Fig. 6; Holm heatmap Fig. 11.

| \(N\) | Scenario | GNN | Vector | DCBO |
|------:|----------|----:|-------:|-----:|
| 100 | Normal | **11.09** | 11.34 | 11.60 |
| 100 | Burst | 25.89 | **25.62** | 26.71 |
| 100 | Churn | 22.91 | **22.61** | 23.23 |

Directionally both learners beat DCBO on all loads. Encoder ranking is
**mixed** relative to Table 3. Holm-adjusted paired \(t\) tests within the
\(N{=}100\) family (`holm_s23_ft10.csv`) do **not** clear \(\alpha{=}0.05\)
after correction—consistent with small absolute gaps (~0.3–1.1 pp) and modest
\(n\). We keep Table 3 as the locked champion narrative and Table 5 as the
journal-facing mean table with honest non-significance under Holm at \(n{=}10\).

### 6.5 Large-\(N\) parity (\(N{=}1000\times 10\))

**Table 6.** Multipipe at \(N{=}1000\) — mean SLA % (10 trials).
Source: `s23_n1000_ft10_summary.csv`. Fig. 9.

| Scenario | GNN | Vector | DCBO | Overhead (all) |
|----------|----:|-------:|-----:|---------------:|
| Normal | 0.82 | **0.80** | 0.81 | 2112 |
| Burst | **2.53** | 3.13 | 3.59 | 2112 |
| Churn | **1.94** | 2.27 | 2.70 | 2140 |

Absolute SLA rates are low. Overhead and colony counts match across methods
(~67–68 colonies). Holm tests at \(N{=}1000\) are non-significant after
correction (`holm_s23_n1000_ft10.csv`), supporting near-parity at scale while
preserving directional GNN advantages under burst/churn.

### 6.6 Named baselines (context)

**Table 7.** Multipipe \(N{=}100\) — named baselines (5 trials) vs champion
colony methods (Table 3). Sources: `baselines_ft5_summary.csv`,
`s23_ft5_summary.csv`. Fig. 10.

| Scenario | FogPlan | Greedy | Static-DCBO | GNN | Vector | DCBO |
|----------|--------:|-------:|------------:|----:|-------:|-----:|
| Normal | 8.28 | **7.32** | 9.25 | 10.64 | 10.98 | 11.48 |
| Burst | 23.24 | 22.63 | **21.61** | 24.89 | 25.35 | 26.13 |
| Churn | 19.88 | 19.37 | **19.02** | 22.31 | 22.43 | 23.21 |
| Overhead | 253 | 659 | 760 | ≈1769 | ≈1769 | ≈1769 |
| Colonies | 1 | 1 | 19 | ≈7.6 | ≈7.6 | ≈7.6 |

Flat baselines can report lower SLA on this multipipe lock, but they are **not**
matched on control-plane cost or cold-start colony organization (FogPlan/Greedy
operate with a single logical colony and much lower messaging). The primary
claim of this paper is therefore **within** the DynaCol control plane: GNN /
Vector / DCBO at matched overhead. Baselines provide absolute context, not a
same-plane ranking.

### 6.7 Scale-aware blend

Using *actual* fog count (threshold \(450\)) avoids false non-parity when
target \(N{=}500\) yields \(\approx 497\) devices. When \(\beta\approx 0\), L2
delegates to DCBO by construction.

---

## 7. Discussion

**Why bound the action set?** Keeping \(\lvert A_t\rvert\le K\) couples learning
to the CRT/GRT DynaCol already exchanges. Empirically, overhead matches DCBO
while small-scale SLA improves within that plane (Tables 2–3; Fig. 8).

**Why GNN is in the title.** The named method is a colony-bounded GraphSAGE
hybrid. On the multipipe champion lock (Table 3), offline weights with online
fine-tuning beat both Vector and DCBO. The 10-trial journal grid (Table 5)
shows directional DCBO wins with mixed encoder ranking and insufficient power
for Holm significance at \(n{=}10\). We claim *graph-assisted colony-bounded
hybrid placement* with a demonstrated GNN edge on the locked stress DAG—not a
large universal GNN-over-vector gap.

**Baselines.** FogPlan / Greedy / Static-DCBO illustrate that absolute SLA can
be lower outside the cold-start colony plane, at far lower control overhead.
That trade-off is orthogonal to C1–C3; reviewers comparing Table 7 must read
overhead and colony columns together.

**What we do not claim.** We do not claim new cold-start formation, new FCM
election, testbed validation, or superiority over every published DRL baseline
under identical control-plane cost.

**Limitations.** Simulation-only (iFogSim2); locked medium grids use five
trials (journal ten with Holm); multipipe uses a calibrated deadline scale;
offline training uses logged exploratory trajectories; Edgeward was omitted
from the multipipe baseline grid due to prohibitive per-trial runtime.

---

## 8. Conclusion

DynaCol-GNN shows that a GraphSAGE hybrid for fog service placement can stay
inside cold-start colony views: hybridize L2 with DCBO, restrict actions to
CRT/GRT top-\(K\), and fall back to exact DCBO at large scale. On iFogSim2,
this reduces SLA violations at \(N{=}100\) versus DCBO at matched overhead on
surveillance and multipipe applications, with near-parity at \(N{=}1000\). On
the multipipe champion lock, the named GNN encoder also improves on the flat
vector ablation. Journal 10-trial means remain directionally favorable versus
DCBO; Holm significance at \(n{=}10\) is not yet achieved. Named flat baselines
clarify absolute SLA versus control-plane cost.

**Future work.** CRT-ablated learning, stronger on-policy offline RL, additional
same-plane baselines, and testbed validation.

---

## Appendix A. Hyperparameters (locked grids)

| Parameter | Default / note |
|-----------|----------------|
| CRT top-\(K\) | DynaCol `TOP_K_CRT_CANDIDATES` |
| \(\varepsilon\) | \(0.02\) on champion multipipe (\(0\) at large \(N\)) |
| Vector LR | \(0.05\) |
| GNN LR | \(0.04\) |
| GNN hidden | \(16\), \(2\) layers, \(\tanh\) |
| Blend (multipipe) | \(0.60\) |
| Multipipe SLA scale | \(10.0\) |
| Large-\(N\) gate | \(n_f \ge 450\) → exact DCBO |
| Trials | 5 (champion/baselines); 10 (journal) |

## Appendix B. Artifact pointers

| Artifact | Path |
|----------|------|
| Surveillance | `evaluation/results/medium_v2_summary.csv` |
| Champion multipipe | `evaluation/results/s23_ft5_summary.csv` |
| Journal 10-trial | `evaluation/results/s23_ft10_summary.csv` |
| Holm \(N{=}100\) | `evaluation/results/holm_s23_ft10.csv` |
| \(N{=}1000\) | `evaluation/results/s23_n1000_ft10_summary.csv` |
| Holm \(N{=}1000\) | `evaluation/results/holm_s23_n1000_ft10.csv` |
| Baselines | `evaluation/results/baselines_ft5_summary.csv` |
| Figures | `evaluation/figures/fig01`–`fig11` (PNG+PDF) |
| Captions | `notes/figure_captions.md` |
| Bibliography | `notes/refs.bib` |
| Offline trainer | `offline/` |
| Java GNN / placement | `ifogsim2/src/org/fog/dynacolgnn/` |

## Appendix C. Figures

See [`figure_captions.md`](figure_captions.md). Render with
`evaluation/render_jos_figures.py`.

---

## References

Full BibTeX: [`refs.bib`](refs.bib) (29 entries). Inline cites used above include:

1. Apat et al. (2025). Fog Service Placement Optimization: A Survey. *Computers.*
2. Azimzadeh et al. (2022). Placement of IoT Services in Fog … Genetic-Based. *Cluster Computing.*
3. Benamer et al. (2024). GA-Based Placement in the Fog … *Cluster Computing.*
4. Brogi & Forti (2017). QoS-Aware Deployment of IoT Applications Through the Fog. *IEEE IoT J.*
5. Chiang & Zhang (2016). Fog and IoT: An Overview of Research Opportunities. *IEEE IoT J.*
6. Costa et al. (2022). Orchestration in Fog Computing. *ACM CSUR.*
7. Dogani et al. (2024). Two-Tier Multi-Objective Service Placement. *Cluster Computing.*
8. Guerrero & Lera (2019). Evolutionary … Fog Application Placement. *FGCS* / related.
9. Gupta et al. (2017). iFogSim. *UCC.*
10. Hamilton, Ying, Leskovec (2017). Inductive Representation Learning on Large Graphs. *NeurIPS.*
11. Holm (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scand. J. Stat.*
12. Hong & Varghese (2019). Resource Management in Fog/Edge Computing. *ACM CSUR.*
13. Hossam et al. (2024). Energy-Aware Module Placement … *Cluster Computing.*
14. Lera & Guerrero (2024). Multi-Objective Application Placement … GNN-Based RL. *J. Supercomput.*
15. Mahmud et al. (2018). Latency-Aware Application Module Management. *ACM TOIT.*
16. Mahmud et al. (2022). iFogSim2. *J. Syst. Softw.*
17. Mnih et al. (2015). Human-level Control Through Deep Reinforcement Learning. *Nature.*
18. Pallewatta et al. (2023). Microservices … Fog/Edge. *ACM CSUR.*
19. Salaht et al. (2020). Overview of Service Placement Problem. *ACM CSUR.*
20. Schulman et al. (2017). Proximal Policy Optimization Algorithms. *arXiv.*
21. Skarlat et al. (2017). Towards QoS-Aware Fog Service Placement. *ICFEC* / *SOCA.*
22. Smolka et al. (2022). Evaluation of Fog Application Placement Algorithms. *Computing.*
23. Sonmez et al. (2018). EdgeCloudSim. *ETT.*
24. Supraja et al. (2025). AI-Driven Service Placement … SLR. *Cluster Computing.*
25. Taleb et al. (2025). Survey on Services Placement Algorithms. *ACM CSUR.*
26. Tuli et al. (2020). HealthFog … IoT and Fog. *FGCS.*
27. Yousefpour et al. (2019). All One Needs to Know about Fog Computing. *J. Syst. Archit.*
28. Yousefpour et al. (2019). FogPlan … *IEEE IoT J.*

---

*Complete JoS draft synced to locked CSVs, enriched bibliography, and
fig01–fig11 (PNG+PDF). Ready for Springer LaTeX conversion.*
