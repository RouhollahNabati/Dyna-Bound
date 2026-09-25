# Method draft (§4) — Colony-Bounded Hybrid Learning

## 4.1 Problem setting

After DynaCol cold-start, each Fog Colony Manager (FCM) maintains a bounded
Colony Resource Table (CRT) and may consult Global Resource Table (GRT)
summaries. A service/module placement request \(r\) arrives with demand vector
and deadline. The control stack is unchanged:

- **L1:** sticky host if still feasible and under SLA thresholds;
- **L2:** choose a feasible CRT host (this paper);
- **L3:** cloud fallback.

Learning must not expand the action set beyond CRT/GRT top-\(K\).

## 4.2 MDP

| Symbol | Definition |
|--------|------------|
| \(S_t\) | Feasible CRT entries (resources, RTT-to-FCM, attractiveness), optional GRT summaries, service features, optional DAG stats / placement memory |
| \(A_t\) | Index of a feasible CRT host, or escalate to GRT search / L3 |
| \(R_t\) | Shaped reward in \([0,1]\): advantage vs DCBO top pick, plus optional co-location terms |
| Constraint | \(\lvert A_t\rvert \le K\) (colony bound is part of the method) |

## 4.3 Hybrid score

For each feasible candidate \(i\), let \(u_{\mathrm{DCBO}}(i) = 1/(1+J(i))\) be the
inverted DCBO objective. The hybrid score is

\[
s(i) = u_{\mathrm{DCBO}}(i) + \beta \cdot f_\theta(\phi(i)),
\]

where \(f_\theta\) is an online linear scorer and \(\phi\) is the encoder.
Selection is \(\arg\max_i s(i)\) with \(\varepsilon\)-greedy exploration at small scale.

**Scale-aware blend.** Let \(n_f\) be the actual fog-node count (not the grid
target \(N\)). Default \(\beta\):

- \(n_f < 250\): \(\beta = 0.40\)
- \(250 \le n_f < 450\): \(\beta = 0.25\)
- \(n_f \ge 450\): \(\beta = 0\) and L2 **delegates exactly to DCBO**

Override: system property `dynacolgnn.learnBlend`. Multipipe stress grids use
\(\beta = 0.60\) unless noted.

## 4.4 Encoders \(\phi\) (ablations)

**Vector (CLI `colony_rl`).** Concatenate local host channels (normalized CPU,
RAM, storage, bandwidth, RTT, attractiveness) with service/DAG scalars
(demand, deadline, in/out degree, depth, endpoint flags) and a bias. No
message passing.

**Local graph / GNN (CLI `colony_gnn`).** Two-layer online GraphSAGE over CRT
with RTT affinity, service-graph context (producer RTT, fan-in, co-location),
and online weight updates (`OnlineColonyGnn`). Competitive with Vector; not
required for the primary hybrid-vs-DCBO claim.

## 4.5 Online update

After choosing host \(i^\star\), observe shaped reward \(R\) and update

\[
\theta \leftarrow \theta + \eta\,(R - \sigma(f_\theta(\phi(i^\star))))\,\phi(i^\star).
\]

Attractiveness decay/update mirrors DCBO so hybrid and baseline share the same
colony-side learning hooks where enabled.

## 4.6 Complexity

Per decision: score at most \(K\) CRT candidates (constant feature dim). No
global node enumeration. Control messages remain those of DynaCol’s CRT/GRT
maintenance; learning adds negligible decision-time overhead at fixed \(K\).

## Algorithm (sketch)

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
