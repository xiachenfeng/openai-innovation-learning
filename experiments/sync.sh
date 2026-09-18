#!/usr/bin/env bash
# 云主机上用：把 experiments/ 下的改动（代码 + 结果 CSV）提交并推送到 GitHub。
#   bash experiments/sync.sh "E1: 填空并跑通 16 组"
# 本机（教练）下次 session 开始时 git pull 即可拿到。checkpoint（*.pt）和缓存不会被提交。
set -e
cd "$(dirname "$0")/.."
msg="${1:-E: update from cloud host}"
git pull --rebase origin main
git add experiments
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -m "$msg"
fi
git push origin main
echo "synced: $(git log --oneline -1)"
