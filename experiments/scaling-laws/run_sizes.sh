#!/usr/bin/env bash
# E2：四个尺寸，跨两个数量级（12·L·d² 约 2.5 万 → 470 万），数据固定为 tinyshakespeare。
# 头数按 d/h = 16 或 32 取。学习率各尺寸相同（3e-4），这正是 Kaplan 被 Chinchilla 批评的做法，先照做，之后可以改。
# 第一轮（09-20）s1～s3 在 4000 步上限处仍在下降（best_step = 4000），没有收敛，所以第二轮把上限提到 12000、耐心提到 8。
set -e
cd "$(dirname "$0")"
# 旧结果改名保留，避免 fit.py 读到重复的 tag
if [ -f outputs/e2_sizes.csv ]; then
  mv outputs/e2_sizes.csv "outputs/e2_sizes_$(date +%m%d-%H%M).csv"
fi
COMMON="--max_steps 12000 --patience 8"
python scaling.py --n_layer 2 --n_embd 32  --n_head 2 --tag s1 $COMMON
python scaling.py --n_layer 3 --n_embd 64  --n_head 4 --tag s2 $COMMON
python scaling.py --n_layer 4 --n_embd 128 --n_head 4 --tag s3 $COMMON
python scaling.py --n_layer 6 --n_embd 256 --n_head 8 --tag s4 $COMMON
# 可选第 5 个点（约 1400 万参数，4090 上几分钟）：
# python scaling.py --n_layer 8 --n_embd 384 --n_head 8 --tag s5 $COMMON
python fit.py
