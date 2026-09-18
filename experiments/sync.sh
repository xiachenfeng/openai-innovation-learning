#!/usr/bin/env bash
# 云主机上用：把 experiments/ 下的改动（代码 + 结果 CSV）提交并推送到 GitHub。
#   bash experiments/sync.sh "E1: 填空并跑通 16 组"
# 顺序：先本地 commit（不需要网络）→ pull --rebase → push。checkpoint（*.pt）和缓存不会被提交。
set -e
cd "$(dirname "$0")/.."
msg="${1:-E: update from cloud host}"

# AutoDL 容器：开学术加速，否则 GitHub 常常连不上
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true

git add experiments
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -m "$msg"
fi
echo "pulling..."
git pull --rebase origin main
echo "pushing..."
git push origin main
echo "synced: $(git log --oneline -1)"
