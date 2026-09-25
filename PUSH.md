# Rename GitHub artifact: DynaCol-GNN → Dyna-Bound

Do **not** change [RouhollahNabati/DynaCol](https://github.com/RouhollahNabati/DynaCol).
Only the JoS companion repo is renamed.

```bash
# 1) Re-authenticate if gh token is expired
gh auth login -h github.com

# 2) Rename the existing public artifact (keeps history; old URL redirects)
gh repo rename Dyna-Bound --repo RouhollahNabati/DynaCol-GNN --yes

# 3) Point this local release tree at the new name
cd "/media/dispro/New Volume/Ph.D/Research/Dynacol/_publish/Dyna-Bound-release"
git remote set-url origin git@github.com:RouhollahNabati/Dyna-Bound.git
git remote -v

# 4) Push updated README / latex after local edits
git add -A
git status
# commit + push when ready:
# git commit -m "Rebrand artifact to Dyna-Bound; keep DynaCol parent separate"
# git push -u origin main
```

Expected URL: https://github.com/RouhollahNabati/Dyna-Bound  
Parent (unchanged): https://github.com/RouhollahNabati/DynaCol
