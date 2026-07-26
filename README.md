# DynaCol-GNN

Colony-bounded hybrid learning for cold-start fog service placement
(GraphSAGE / vector encoders + DCBO hybridization on CRT/GRT views).

Companion artifact for the Journal of Supercomputing manuscript
**Colony-Bounded Hybrid Learning for Cold-Start Fog Service Placement**.

## Relation to DynaCol

This repository packages the **DynaCol-GNN** extension (configs, evaluation,
offline trainer, LaTeX, and Java `org.fog.dynacolgnn` sources).

Full simulation runtime still depends on the parent **DynaCol / iFogSim2**
codebase:

- Parent project: [RouhollahNabati/DynaCol](https://github.com/RouhollahNabati/DynaCol)
- DynaCol control-plane paper: *Cluster Computing* (under review)

Drop / merge `ifogsim2/src/org/fog/dynacolgnn/` into a DynaCol checkout, then
run grids from this tree (or symlink paths). See `ifogsim2/patches/` for a
reference `DynaColBootstrap.java` (tabular `drl` on cold-start overlay).

## Layout

| Path | Contents |
|------|----------|
| `configs/` | Locked journal / sensitivity / smoke JSON grids |
| `evaluation/` | `run_grid.py`, Holm / effect-size scripts, summaries, figures |
| `offline/` | Trajectory collect + GIN BC trainer + locked `gnn_weights_s23.json` |
| `latex/` | Springer Nature `sn-jnl` manuscript (`bash build.sh`) |
| `notes/` | Draft notes, revision gate, response-to-reviewers template |
| `ifogsim2/src/org/fog/dynacolgnn/` | Java colony-bounded placers |

## Quick start

```bash
# Python tools (offline + evaluation)
cd offline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build PDF
cd ../latex && bash build.sh

# Journal-scale revision runner (needs compiled iFogSim2 + DynaCol)
cd ../evaluation && bash run_jos_revision.sh
```

Primary journal lock: multipipe \(N{=}100\times 20\) trials
(`evaluation/results/s23_ft20_summary.csv`, Holm + bootstrap CIs).

## License / citation

Please cite the JoS manuscript when available. Until then, cite this repository
and the under-review DynaCol paper.
