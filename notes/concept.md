# Concept — DynaCol-Hybrid (Idea A)

## One-sentence claim

Keep DynaCol’s self-organizing control plane; replace only the L2 host
choice with a **colony-bounded hybrid learner** whose action space is the
pruned CRT (and optional GRT), hybridized with DCBO and scale-aware fallback.

## What stays frozen (main paper)

- Cold-start query/advertise formation
- Potency-based FCM election and handover (ε + cooldown)
- Bounded CRT / GRT
- L1 sticky path and L3 cloud fallback
- Top-level `evaluation/` results and journal scripts

## What is new here

1. MDP with actions restricted to CRT/GRT candidates after cold-start
2. Hybrid stack: L1 deterministic → L2 online learner → L3 cloud
3. Scale-aware blend (exact DCBO delegation at large fog counts)
4. Encoder ablations: flat vector (A1) vs local graph (A2)
5. Separate evaluation artifacts under `dyna-bound/evaluation/`

## Hypotheses (evidence-aligned)

- **H1 (primary):** At small/mid scale, hybrid learning ≤ DCBO on SLA under
  burst/churn, matched control overhead — **supported** (surveillance + multipipe N=100)
- **H2:** Control overhead stays bounded (actions only over top-K) — **supported**
- **H3:** Large-N blend→0 yields DCBO parity — **supported** (surveillance N≥300)
- **H4:** CRT-ablated learning is weaker → colony bound is part of the method — pending dedicated ablation
- **H5 (secondary):** Local graph encoder improves over flat vector on
  non-trivial DAGs — **not established**; report as ablation (A1 ≳ A2 on multipipe)

## Experiment sketch

- **N:** 100, 300, 500, (journal: 1000)
- **Loads:** normal, burst, churn
- **Apps:** surveillance (default); multipipe (structural stress)
- **Policies:** DCBO, Colony-Vector (A1), Colony-Graph (A2); journal adds baselines
- **Metrics:** SLA %, P95, control msgs, colonies
- **Stats:** ≥10 trials + Holm for submission

## Venue fit

Journal of Supercomputing — Artificial Intelligence section.
Lead with **bounded-view hybrid learning under cold-start**, not “yet another GNN.”

## Design lock

- [x] Formal state / action / reward in `mdp.md`
- [x] Online-on-FCM linear head (phase-1); offline PPO/GIN deferred
- [x] **Primary story:** hybrid colony-bounded learning vs DCBO
- [x] **A1 / A2:** encoder ablations (A1 slightly stronger on multipipe)
- [x] Folder name `dyna-bound/` kept for history; prose uses DynaCol-Hybrid
