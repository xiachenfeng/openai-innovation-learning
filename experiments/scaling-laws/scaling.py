"""
E2 训练脚本：复用 E1 的 model.py / train.py，训一个尺寸，记录"最低 val loss"及当时的 token 数与算力估计。

用法（在云主机上，experiments/scaling-laws/ 目录下）：
  python scaling.py --n_layer 2 --n_embd 32 --n_head 2 --tag s1
  bash run_sizes.sh          # 四个尺寸一次跑完并拟合

每个尺寸训到 val loss 连续 --patience 次评估不再下降就停（早停），
这对应 Kaplan 测 L(N) 时"数据给足、训到收敛"的做法；但我们的数据只有 tinyshakespeare，
所以大模型会先遇到数据瓶颈（知识卡 A4 §6 的 L(N, D)）。

带 `TODO(you)` 的两处留给你填，每处一行：
  (1) Kaplan 口径的非 embedding 参数量 N；
  (2) 训练算力估计 C ≈ 6·N·D。
"""
import argparse
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pretraining", "tiny_gpt"))
from model import GPT, GPTConfig          # noqa: E402
from train import estimate_loss, get_batch, load_text  # noqa: E402


def kaplan_params(model: GPT) -> int:
    """Kaplan 2020 的 N：排除 token embedding 与 position embedding（lm_head 与 wte 共享，只算一次）。"""
    total = sum(p.numel() for p in model.parameters())
    # TODO(you) (1): 从 total 里减去两个 embedding 矩阵的元素数。
    #   提示：model.wte.weight.numel() 与 model.wpe.weight.numel()。
    #   结果应接近 12·L·d² 加上 LayerNorm 与 bias（每层 13·d，最后再加一个 LN 的 2d）。
    raise NotImplementedError("TODO(you) (1)")  # ← 替换这一行，形如 return total - ... - ...


def flops_estimate(n_params: int, tokens: int) -> float:
    """Kaplan 2020 的算力估计 C ≈ 6·N·D（前向 2N/token，反向 4N/token）。"""
    # TODO(you) (2): 一行。
    raise NotImplementedError("TODO(you) (2)")  # ← 替换这一行


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, required=True)
    p.add_argument("--n_embd", type=int, required=True)
    p.add_argument("--n_head", type=int, required=True)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--patience", type=int, default=6, help="val 连续多少次评估不降就早停")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_csv", type=str, default=os.path.join(HERE, "outputs", "e2_sizes.csv"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据：与 E1 相同的 tinyshakespeare，缓存复用 E1 的下载 ----
    text = load_text(os.path.join(HERE, "..", "pretraining", "tiny_gpt", "outputs", "cache"))
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size,
                    n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head)
    model = GPT(cfg).to(device)
    n_kaplan = kaplan_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device {device}  L={cfg.n_layer} d={cfg.n_embd} h={cfg.n_head}  "
          f"N_kaplan={n_kaplan:,}  12·L·d²={12 * cfg.n_layer * cfg.n_embd ** 2:,}  total={n_total:,}  "
          f"train tokens={n_train:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokens_per_step = args.batch_size * args.block_size
    best_val, best_step, bad = float("inf"), 0, 0
    t0 = time.time()
    step = 0
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0:
            val = estimate_loss(model, val_data, cfg, args.batch_size, device)
            improved = val < best_val - 1e-3
            if improved:
                best_val, best_step, bad = val, step, 0
            else:
                bad += 1
            print(f"step {step:5d}  train {loss.item():.4f}  val {val:.4f}  best {best_val:.4f}@{best_step}  {time.time() - t0:.0f}s")
            if bad >= args.patience:
                print(f"early stop: val 连续 {args.patience} 次未降")
                break

    tokens_at_best = best_step * tokens_per_step
    row = {
        "tag": args.tag, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
        "N_kaplan": n_kaplan, "N_total": n_total,
        "best_val": round(best_val, 4), "best_step": best_step,
        "tokens_at_best": tokens_at_best, "epochs_at_best": round(tokens_at_best / n_train, 2),
        "flops_at_best": f"{flops_estimate(n_kaplan, tokens_at_best):.3e}",
        "steps_run": step, "final_train": round(loss.item(), 4),
        "lr": args.lr, "batch": args.batch_size, "seed": args.seed, "elapsed_s": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("wrote", args.out_csv, row)


if __name__ == "__main__":
    main()
