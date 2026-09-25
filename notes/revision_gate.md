# Revision gate — JoS major revision (n=20 lock)

**Date:** 2026-07-25  
**Primary artifacts:** `s23_ft20_summary.csv`, `holm_s23_ft20.csv`, `effect_s23_ft20.csv`,
`sensitivity_burst5_summary.csv`

## 1. Hybrid vs DCBO (n=20)

| Scenario | GNN mean | Vector mean | DCBO mean | GNN−DCBO CI | Holm sig? |
|----------|----------|-------------|-----------|-------------|-----------|
| Normal | 10.56 | 10.79 | 11.50 | [−1.49, −0.43] | **Yes** |
| Burst | 25.26 | 25.19 | 26.56 | [−1.89, −0.73] | **Yes** |
| Churn | 22.28 | 22.04 | 23.19 | [−1.52, −0.31] | **Yes** |

Both hybrids beat DCBO on all loads with Holm significance and CI excluding zero
at \(n{=}20\). **Primary claim PASS.**

## 2. GNN vs Vector (title rule)

Rule: GNN mean better on ≥2/3 scenarios **and** bootstrap CI for
`mean(GNN−Vector)` does not lie entirely on the Vector-favoring side.

| Scenario | GNN−Vector mean | CI | GNN better? |
|----------|-----------------|-----|-------------|
| Normal | −0.23 | [−0.77, 0.34] | Yes (mean) |
| Burst | +0.07 | [−0.75, 0.89] | No |
| Churn | +0.24 | [−0.27, 0.83] | No |

GNN better on **1/3** scenarios; no CI excludes zero. Holm GNN_vs_Vector = No
everywhere.

**Title decision: RETITLE** to
**“Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement”**.
Keep GraphSAGE as the named encoder / method brand **Dyna-Bound** in the body;
do **not** claim a universal GNN≻Vector gap.

## 3. Tabular DRL (same-plane)

DRL-based mean SLA is lowest on all loads (e.g. Burst 22.47 vs GNN 25.26) with
Holm-significant wins vs DCBO/GNN/Vector. Overhead is lower (~1501 vs ~1770)
at matched colony count (~7.65).

**Interpretation for claims:** DRL is a strong same-plane learning *control*,
not a failure of the hybrid. Hybrid contribution = DCBO-anchored scoring +
scale-aware exact DCBO fallback + offline structure import. Absolute SLA
leadership inside the plane belongs to tabular DRL on this lock; hybrids beat
deterministic DCBO with statistical support.

## 4. Pareto / cold-start plane

Fig. 12: flat FogPlan/Greedy still lower SLA + much lower overhead (different
plane). Primary ranking remains within-plane.

## 5. Sensitivity

- **β:** GNN SLA improves mildly as blend rises 0.40→0.80 (25.04→24.52); lock
  0.60 is interior, not an extreme cherry-pick.
- **Deadline scale:** absolute SLA highly sensitive (8→42%, 10→25%, 12→9%);
  relative hybrid≻DCBO direction preserved at 8 and 12. Scale=10 is the
  calibrated operating point (stated as such).
- **ε:** 0 vs 0.02 nearly tied; 0.02 remains the lock.

## 6. Locked manuscript decisions

| Item | Lock |
|------|------|
| Title | Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement |
| Primary table | n=20 journal (GNN, Vector, DCBO, DRL) |
| Champion 5-trial | Tuning/weight lock only |
| C3 wording | Encoders + same-plane tabular learner; GNN edge not universal |
| Abstract | Lead with hybrid≻DCBO (significant); note encoder parity; mention DRL control |
| Holm narrative | Significant at n=20 for hybrid vs DCBO |
