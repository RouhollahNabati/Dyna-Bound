# Evaluation (Dyna-Bound only)

All A2/A1 trial CSVs, summaries, figures, and LaTeX fragments for this line
belong here:

```
dyna-bound/evaluation/
├── results/          # trials + locked summary CSVs
├── figures/          # rendered bars/scalability from summaries
├── render_figures.py # matplotlib renderer (use offline/.venv)
├── run_grid.py       # grid runner for this line
└── latex/            # optional table fragments
```

## Isolation rule

| Allowed | Forbidden |
|---------|-----------|
| Write under `dyna-bound/evaluation/` | Overwrite `../../evaluation/results/` |
| Read main baselines for comparison | Regenerate main-paper LaTeX from A2 runs |
| Call `ifogsim2` binaries / classes | Fork a second full `ifogsim2` tree |

Main-paper harness remains `../../evaluation/run_ifogsim2.py` and
`../../evaluation/generate_results.py`. Add A2-specific scripts in this
directory when ready.
