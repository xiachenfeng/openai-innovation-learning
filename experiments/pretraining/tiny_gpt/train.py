"""
E1 训练脚本（字符级 tiny GPT）。

用法：
  python train.py --smoke            # 只跑一次前向 + 一次反向，打印 shape 与 loss（本周目标）
  python train.py --steps 2000       # 正式训练（下周 E1 完成时用）

数据：默认下载 tinyshakespeare（约 1MB 公开文本）；下载失败则退回内置的一小段文本。
"""
import argparse
import math
import os
import time
import urllib.request

import torch

from model import GPT, GPTConfig

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FALLBACK_TEXT = (
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles "
    "and by opposing end them. To die, to sleep; no more; and by a sleep to say we end "
    "the heart-ache and the thousand natural shocks that flesh is heir to. "
) * 200


def load_text(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "input.txt")
    if not os.path.exists(path):
        try:
            print(f"downloading {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as e:  # noqa: BLE001
            print(f"download failed ({e}); using built-in fallback text")
            with open(path, "w") as f:
                f.write(FALLBACK_TEXT)
    with open(path) as f:
        return f.read()


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """随机切 batch_size 段，每段长 block_size + 1；input 取前 T 个，target 取右移一位的 T 个。"""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])          # [B, T]
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # [B, T]，作业第 1 题
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, cfg, batch_size, device, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="只跑一次前向 + 反向")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- 数据：字符级词表 ----
    text = load_text(os.path.join(args.out_dir, "cache"))
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"text chars: {len(text):,}  vocab: {len(chars)}  train tokens: {n_train:,}")

    # ---- 模型 ----
    cfg = GPTConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    approx = 12 * cfg.n_layer * cfg.n_embd**2
    print(f"params (non-embedding): {n_params:,}   12·L·d² ≈ {approx:,}")

    # ---- smoke：一次前向 + 一次反向 ----
    x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
    print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}")
    logits, loss = model(x, y)
    print(f"logits: {tuple(logits.shape)}  loss: {loss.item():.4f}   "
          f"(随机初始化时应接近 ln V = {math.log(cfg.vocab_size):.4f})")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    print(f"backward ok, grad norm: {grad_norm:.4f}")
    model.zero_grad(set_to_none=True)
    if args.smoke:
        print("smoke test passed")
        return

    # ---- 训练循环 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)                       # 前向：T 个位置同时算 loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                             # 反向
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                            # AdamW 一步更新
        if step % args.eval_every == 0 or step == args.steps:
            val = estimate_loss(model, val_data, cfg, args.batch_size, device)
            print(f"step {step:5d}  train {loss.item():.4f}  val {val:.4f}  {time.time() - t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "stoi": stoi}, os.path.join(args.out_dir, "e1_tiny_gpt.pt"))

    # ---- 采样看看 ----
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=300)[0].tolist()
    print("---- sample ----")
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
