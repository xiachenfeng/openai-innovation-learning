#!/usr/bin/env bash
# E1 第二阶段：一次跑完全部对照（16 组，4090 上每组约十几秒）。
set -e
cd "$(dirname "$0")"
[ -f outputs/e1_tiny_gpt.pt ] || python train.py --steps 2000
rm -f outputs/e1_stage2_results.csv
for init in pretrained scratch; do
  for n in 100 1000; do
    for lam in 0 0.5; do
      for lr in 6e-5 3e-4; do
        python sft.py --init "$init" --n_train "$n" --lam "$lam" --lr "$lr"
      done
    done
  done
done
python sft.py --summary
