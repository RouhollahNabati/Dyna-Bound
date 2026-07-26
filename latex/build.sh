# Build DynaCol-GNN Springer sn-jnl PDF (iicol two-column).
# Article-class backup: bash build.sh article  → main_article.pdf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-snjnl}"

mkdir -p figures
for f in ../evaluation/figures/fig0{1,2,3,4,5,6,7,8,9}_*.pdf \
         ../evaluation/figures/fig1{0,1,2,3}_*.pdf; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  ln -sfn "../../evaluation/figures/$base" "figures/$base"
done

[[ -f refs.bib ]] || cp ../notes/refs.bib refs.bib
[[ -f sn-jnl.cls ]] || cp ../../submission/cluster-computing/latex/sn-jnl.cls .
[[ -f sn-mathphys-num.bst ]] || cp ../../submission/cluster-computing/latex/sn-mathphys-num.bst .

if [[ "$MODE" == "article" ]]; then
  TEX=main_article.tex
  JOB=main_article
else
  TEX=main.tex
  JOB=main
fi

pdflatex -interaction=nonstopmode -jobname="$JOB" "$TEX"
bibtex "$JOB"
pdflatex -interaction=nonstopmode -jobname="$JOB" "$TEX"
pdflatex -interaction=nonstopmode -jobname="$JOB" "$TEX"

echo "Built: $ROOT/${JOB}.pdf"
pdfinfo "${JOB}.pdf" 2>/dev/null | grep -E 'Pages|Page size' || ls -la "${JOB}.pdf"