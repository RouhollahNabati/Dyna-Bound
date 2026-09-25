# Dyna-Bound

Colony-bounded hybrid learning for cold-start fog service placement
(GraphSAGE / vector encoders + DCBO hybridization on CRT/GRT views).

Companion JoS artifact (method brand: **Dyna-Bound**).

## Relation to DynaCol

- Parent (unchanged, use separately): https://github.com/RouhollahNabati/DynaCol
- This artifact: https://github.com/RouhollahNabati/Dyna-Bound

Merge `ifogsim2/src/org/fog/dynacolgnn/` into a DynaCol checkout, then run grids from this tree.

## Layout

| Path | Contents |
|------|----------|
| `configs/` | Journal / sensitivity / smoke grids |
| `evaluation/` | Runners, Holm/effect scripts, summaries, figures |
| `offline/` | Trajectory collect + trainer + weights |
| `latex/` | Springer sn-jnl manuscript |
| `ifogsim2/src/org/fog/dynacolgnn/` | Java colony-bounded placers |

## Quick start

```bash
cd offline && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd ../latex && bash build.sh
```
