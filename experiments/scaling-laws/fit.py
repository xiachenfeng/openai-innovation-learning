"""
E2 拟合脚本：读 outputs/e2_sizes.csv，在双对数坐标下拟合 L(N) 的幂律，并检查最大模型是否偏离直线。

Kaplan 2020：L(N) = (N_c / N)^α_N  ⇔  log L = α_N·log N_c − α_N·log N，
所以 log L 对 log N 做线性回归，斜率的相反数就是 α_N。
论文的 α_N ≈ 0.076 是 WebText2 上的值；这里是字符级、数据固定，得到的指数只能和自己比。

带 `TODO(you)` 的两处留给你填，每处一行。
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs", "e2_sizes.csv")

rows = list(csv.DictReader(open(path)))
rows.sort(key=lambda r: int(r["N_kaplan"]))
N = np.array([int(r["N_kaplan"]) for r in rows], dtype=float)
L = np.array([float(r["best_val"]) for r in rows])

print(f"{'tag':>4} {'N_kaplan':>10} {'best_val':>9} {'best_step':>9} {'epochs':>7} {'flops':>10}")
for r in rows:
    print(f"{r['tag']:>4} {int(r['N_kaplan']):>10,} {float(r['best_val']):>9.4f} {int(r['best_step']):>9} "
          f"{float(r['epochs_at_best']):>7.2f} {r['flops_at_best']:>10}")


def fit_alpha(N_sub, L_sub):
    """在双对数坐标下做一次线性回归，返回 (alpha, intercept)。"""
    # TODO(you) (1): 用 np.polyfit 对 log N 与 log L 拟合一次多项式，斜率 s，截距 b；alpha = -s。
    #   提示：np.polyfit(x, y, 1) 返回 [斜率, 截距]。
    s, b = np.polyfit(np.log(N_sub), np.log(L_sub), 1)
    return -s, b


def predict(alpha, b, N_query):
    """用拟合结果预测 N_query 处的 loss。"""
    # TODO(you) (2): log L = b − alpha·log N，返回 L（不是 log L）。
    return np.exp(b - alpha*np.log(N_query))


alpha_all, b_all = fit_alpha(N, L)
print(f"\n全部 {len(N)} 个点拟合：alpha_N = {alpha_all:.3f}")

if len(N) >= 4:
    alpha_3, b_3 = fit_alpha(N[:-1], L[:-1])
    pred = predict(alpha_3, b_3, N[-1])
    print(f"前 {len(N) - 1} 个点拟合：alpha_N = {alpha_3:.3f}，外推到 N = {int(N[-1]):,} 预测 L = {pred:.4f}，实际 {L[-1]:.4f}")
    gap = L[-1] - pred
    print(f"实际 − 预测 = {gap:+.4f}  → {'最大模型偏离直线，数据瓶颈或欠训练' if gap > 0.02 else '仍在直线上'}")

# 供知识卡 §6 表格用：每翻 10 倍 N，loss 乘以 10^(-alpha)
print(f"\n按 alpha = {alpha_all:.3f}，N 翻 10 倍 loss 变为原来的 {10 ** (-alpha_all):.3f} 倍（Kaplan：{10 ** (-0.076):.3f}）")
