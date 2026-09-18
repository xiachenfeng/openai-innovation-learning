#!/usr/bin/env bash
# E1 第二阶段补充：关键对照换 3 个随机种子，判断差异是否超过噪声（测试集 278 条，±0.03 是一个标准差）。
# 固定 n_train=1000、lam=0.5，比较 init × lr，共 2×2×3 = 12 组。
set -e
cd "$(dirname "$0")"
for seed in 1 2 3; do
  for init in pretrained scratch; do
    for lr in 6e-5 3e-4; do
      python sft.py --init "$init" --n_train 1000 --lam 0.5 --lr "$lr" --seed "$seed"
    done
  done
done
python sft.py --summary
