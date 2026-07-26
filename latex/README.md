# LaTeX draft — DynaCol-GNN (Journal of Supercomputing)

## Files

| File | Role |
|------|------|
| `main.tex` | Manuscript on Springer Nature `sn-jnl` (two-column `iicol`) |
| `main_article.tex` | Portable `article`-class backup |
| `sn-jnl.cls` | Compatibility wrapper (same as Cluster Computing submission) |
| `sn-mathphys-num.bst` | Numbered Springer math/phys bibliography style |
| `refs.bib` | Bibliography |
| `figures/` | Symlinks to `../evaluation/figures/fig01`–`fig13` PDFs |
| `build.sh` | `pdflatex` → `bibtex` → `pdflatex` ×2 |

## Build

```bash
cd dynacol-gnn/latex
bash build.sh          # → main.pdf (sn-jnl, two-column)
bash build.sh article  # → main_article.pdf (single-column article)
```

## Authors

- **Rouhollah Nabati** — corresponding (`rnabati@gmail.com`)
- **Abdulbaghi Ghaderzadeh** (`b.ghaderzadeh80@gmail.com`)
- **Affiliation:** Department of Computer Engineering, Islamic Azad University,
  Sanandaj Branch, Sanandaj, Iran

## Springer notes

This build uses the project’s `sn-jnl` compatibility class (API-compatible with
Springer Nature template options). For Editorial Manager upload you may still
prefer the official publisher zip from Springer Nature LaTeX author support /
Overleaf if the journal requires the full v3.1 package; body content is the same.

Declarations (funding, competing interests, contributions, data/code) are in
`main.tex` before the appendix.
