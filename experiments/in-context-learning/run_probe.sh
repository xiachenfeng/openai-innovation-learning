#!/usr/bin/env bash
# E3：GPT-2 四个尺寸 × K ∈ {0,1,4,8} × 5 组示例，SST-2 前 200 条验证集查询。4090 上约 10～15 分钟（gpt2-xl 占大半）。
# 首次运行会从 Hugging Face 下载模型（gpt2-xl 约 6GB）与 SST-2；AutoDL 上先 source /etc/network_turbo，
# 仍连不上就取消下一行的注释走镜像。
# export HF_ENDPOINT=https://hf-mirror.com
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt

# 旧结果改名保留，避免汇总时重复
if [ -f outputs/e3_icl.csv ]; then
  mv outputs/e3_icl.csv "outputs/e3_icl_$(date +%m%d-%H%M).csv"
fi

# 冒烟：最小模型、两个 K、两组示例、50 条查询（约 1 分钟）。跑通后再跑下面的正式版。
python icl_probe.py --models gpt2 --shots 0,4 --seeds 2 --n_eval 50 --out_csv outputs/e3_smoke.csv

# 正式：四个尺寸
python icl_probe.py --models gpt2,gpt2-medium,gpt2-large,gpt2-xl --shots 0,1,4,8 --seeds 5 --n_eval 200

# 可选：两位数加法（生成式读法），预期接近 0
# python icl_probe.py --task addition --models gpt2,gpt2-xl --shots 0,4,8 --seeds 2 --n_eval 100
