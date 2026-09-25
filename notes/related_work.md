# Related Work — Dyna-Bound

Venue: *The Journal of Supercomputing*, Artificial Intelligence section.
BibTeX: [`refs.bib`](refs.bib) (~29 entries).

## 1. Fog / edge service placement

Fog and edge computing push latency-sensitive services closer to users
[Chiang and Zhang, 2016; Yousefpour et al., 2019]. The service placement
problem—where to host application modules under capacity, latency, and cost
constraints—has been surveyed extensively
[Salaht et al., 2020; Smolka et al., 2022; Taleb et al., 2025; Apat et al., 2025].
Classical and metaheuristic placers (greedy, ILP, genetic, swarm) typically
assume a fixed hierarchy or a centralized controller
[Skarlat et al., 2017; Skarlat et al., 2017 SOCA; Azimzadeh et al., 2022;
Benamer et al., 2024]. QoS-aware deployment tooling such as FogTorch
[Brogi and Forti, 2017] and Min-Cost FogPlan-style assignment
[Yousefpour et al., 2019 FogPlan] optimize placement quality but do not
bootstrap a self-organizing control plane from a zero-colony state.

**Gap:** quality under cold-start colony formation with *bounded* CRT/GRT
decision views, not only offline assignment on a static topology.

## 2. Learning-based offloading and placement

Deep RL (DQN/PPO) and AI-driven placers adapt online
[Mnih et al., 2015; Schulman et al., 2017; Supraja et al., 2025]. Domain
instances include healthcare/fog ensembles [Tuli et al., 2020], energy-aware
module placement [Hossam et al., 2024], and multi-tier container placement
[Dogani et al., 2024]. Resource-management surveys highlight that many
learners still act over large host sets and presume an organized fabric
[Hong and Varghese, 2019]. Orchestration surveys similarly separate control
planes from learning plugins [Costa et al., 2022; Pallewatta et al., 2023].

**Positioning:** Dyna-Bound keeps actions inside top-\(K\) CRT/GRT candidates
and hybridizes with deterministic DCBO so cold start remains stable. Learning
is the L2 plug-in, not a replacement for formation/handover.

## 3. Self-organizing / hierarchical fog control

Clustering, overlay, and manager-election schemes—including DynaCol—separate
overlay formation from placement. Latency-aware hierarchical module management
improves QoS under mobility and load [Mahmud et al., 2018]. Evolutionary and
application-aware hierarchical placers [Guerrero and Lera, 2019] optimize
mapping once a hierarchy exists. Learning is then either absent or attached
without treating the colony view as a *hard* action constraint.

**Positioning:** formation/handover stay as in DynaCol; learning only re-ranks
within the colony-bounded decision set.

## 4. Graph learning on infrastructure / service graphs

Inductive GraphSAGE encoders [Hamilton et al., 2017] and GNN-RL placers
[Lera and Guerrero, 2024] encode dependency structure for scheduling and
multi-objective placement. Fog CRT neighborhoods are small, dynamic, and
RTT-structured—well matched to local message passing, but only if the action
set stays colony-bounded. On our multipipe champion lock the GNN edges the
vector ablation; we still do not claim a large universal GNN-over-vector gap
on every grid.

## 5. Evaluation tooling and statistics

iFogSim / iFogSim2 provide the shared simulation substrate
[Gupta et al., 2017; Mahmud et al., 2022]; EdgeCloudSim is a related
alternative [Sonmez et al., 2018]. Multiple-comparison control uses Holm’s
sequentially rejective procedure [Holm, 1979].

## 6. Relation to DynaCol

Treat DynaCol/DCBO as prior system / baseline control plane. This paper does
not re-claim cold-start formation or FCM handover. Differences: (i) colony-
bounded MDP for L2, (ii) hybrid GraphSAGE / vector learners with scale-aware
blend, (iii) multipipe stress + named baselines + journal-scale grids.

## Comparison table (extended)

| Method family | Cold-start overlay | Bounded decision view | Hybrid w/ determ. obj. | Local graph / DAG |
|---------------|:------------------:|:---------------------:|:----------------------:|:-----------------:|
| Classical / GA / ILP placers | Rarely | No (often global) | Sometimes | Rarely |
| FogPlan / Edgeward / Static-DCBO | No / partial | No / fixed hierarchy | No | No |
| DRL placers (DQN/PPO) | Rarely | Rarely | Rarely | Sometimes |
| GNN-RL placers | Rarely | Rarely | Rarely | Yes |
| DynaCol / DCBO | Yes | Yes (CRT/GRT) | N/A | No |
| **Dyna-Bound** | **Yes (reused)** | **Yes (≤K CRT)** | **Yes (DCBO blend)** | **Yes (GraphSAGE)** |
