# Push to GitHub

Local release tree is ready (commits on `main`). From this directory:

```bash
cd /home/dispro/Projects/Dynacol/_publish/DynaCol-GNN-release

# 1) Authenticate if needed
gh auth login -h github.com

# 2) Create public repo and push
gh repo create RouhollahNabati/DynaCol-GNN --public \
  --description "Colony-bounded hybrid learning for cold-start fog service placement (JoS artifact)" \
  --source=. --remote=origin --push

# Or SSH push if the empty repo already exists:
# git remote add origin git@github.com:RouhollahNabati/DynaCol-GNN.git
# git push -u origin main
```

Expected URL: https://github.com/RouhollahNabati/DynaCol-GNN
