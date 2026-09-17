"""
E1 第二阶段：模拟 GPT-1 的"生成式预训练 → 判别式微调"（知识卡 A2 图 2、图 3）。

任务：只看一段台词片段（≤126 字符），猜它是 tinyshakespeare 里出场最多的 K 个角色中谁说的（K=5）。
数据：5 个角色共 876 段台词，切成约 1480 个片段；按整段台词分 80/20 做训练池/测试集，防止泄漏。
输入拼法照 GPT-1：<s> 台词 <e>，取 <e> 位置的隐状态过新增线性层 W_y。

对照组（GPT-1 论文表 5 的消融）：
  --init pretrained | scratch     从 train.py 的 checkpoint 起 vs 随机起
  --n_train 100 | 1000            标注量（训练池约 1200 条）
  --lam 0 | 0.5                   辅助 LM 损失权重 λ
  --lr 6e-5 | 3e-4                微调学习率（论文是预训练的 1/40）

带 `TODO(you)` 的三处留给你填，每处一到两行：
  (1) 输入拼法：把一段文本变成 [<s>] + 字符 id + [<e>]
  (2) 取 <e> 位置的隐状态：[B, T, d] → [B, d]
  (3) 总损失：L3 = L2 + λ·L1
其余代码完整。先跑 `python train.py --steps 2000` 得到 checkpoint，再跑本脚本；
一次跑全部对照用 `bash run_stage2.sh`。
"""
import argparse
import csv
import os
import random
import re
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig
from train import load_text

RESULTS = "outputs/e1_stage2_results.csv"


# ---------------------------------------------------------------- 数据 ----
def parse_speeches(text: str):
    """tinyshakespeare 的格式是 "NAME:\\n台词\\n台词\\n\\nNAME:\\n..."，切成 (角色, 台词) 对。"""
    out = []
    for block in text.split("\n\n"):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        head = lines[0].strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,30}:", head):
            continue
        out.append((head[:-1], " ".join(l.strip() for l in lines[1:])))
    return out


def build_dataset(speeches, k: int, max_chars: int, seed: int, test_frac: float = 0.2, min_chars: int = 20):
    """取出场最多的 k 个角色；每段台词切成 ≤ max_chars 的片段，每个片段一个样本。
    先按"整段台词"切训练/测试，再切片段，避免同一段台词的片段同时出现在两边（泄漏）。"""
    cnt = Counter(s for s, _ in speeches)
    top = [s for s, _ in cnt.most_common(k)]
    label = {s: i for i, s in enumerate(top)}
    kept = [(body, label[s]) for s, body in speeches if s in label]
    random.Random(seed).shuffle(kept)
    n_test_sp = int(len(kept) * test_frac)

    def chunks(subset):
        out = []
        for body, lab in subset:
            for i in range(0, len(body), max_chars):
                piece = body[i : i + max_chars]
                if len(piece) >= min_chars:
                    out.append((piece, lab))
        random.Random(seed + 1).shuffle(out)
        return out

    return chunks(kept[n_test_sp:]), chunks(kept[:n_test_sp]), top


def encode_example(text: str, stoi: dict, S: int, E: int) -> list[int]:
    """GPT-1 的 task-aware input transformation（分类任务那一行）。

    TODO(you) (1): 返回 token id 列表：先 <s>，然后每个字符的 id，最后 <e>。
      提示：stoi 是字符 → id 的字典，S、E 是两个新 token 的 id；
            语料里没见过的字符用 stoi[" "] 兜底：stoi.get(c, stoi[" "])。
      知识卡 A2 §2.3 的第一行就是这个拼法。
    """
    raise NotImplementedError("TODO(you) (1)")


def collate(batch, stoi, S, E, device):
    """把长度不一的样本右填充成 [B, T]。
    因为 causal mask，<e> 之后的填充不会影响 <e> 位置的 h；LM 目标用 -100 忽略填充位。"""
    ids = [encode_example(t, stoi, S, E) for t, _ in batch]
    T = max(len(x) for x in ids)
    x = torch.zeros(len(ids), T, dtype=torch.long)
    lm_y = torch.full((len(ids), T), -100, dtype=torch.long)
    e_pos = torch.zeros(len(ids), dtype=torch.long)
    for i, seq in enumerate(ids):
        n = len(seq)
        x[i, :n] = torch.tensor(seq)
        lm_y[i, : n - 1] = torch.tensor(seq[1:])      # 右移一位，只在真实 token 上算（作业第 1 题）
        e_pos[i] = n - 1                              # <e> 的位置
    y = torch.tensor([lab for _, lab in batch], dtype=torch.long)
    return x.to(device), e_pos.to(device), y.to(device), lm_y.to(device)


# ---------------------------------------------------------------- 模型 ----
class GPTClassifier(nn.Module):
    """GPT-1 微调模型：整个 decoder（参数 Θ）+ 一个新的线性层 W_y。全部参数都更新。"""

    def __init__(self, gpt: GPT, n_classes: int):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg.n_embd, n_classes)   # W_y: [d, K]，随机初始化
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx, e_pos, labels=None, lm_targets=None, lam: float = 0.0):
        h = self.gpt.hidden(idx)                            # [B, T, d]，共享的 Θ 算出来的
        B = h.size(0)

        # TODO(you) (2): 取每个样本 <e> 位置的隐状态，得到 h_e，形状 [B, d]。
        #   提示：e_pos 是 [B] 的位置下标；h[torch.arange(B), e_pos] 就是"第 i 个样本取第 e_pos[i] 个位置"。
        #   为什么取 <e> 而不是 <s>：见知识卡 A2 §2.3 末尾（causal mask）。
        h_e = None  # ← 替换这一行
        if h_e is None:
            raise NotImplementedError("TODO(you) (2)")

        logits = self.head(h_e)                             # [B, K]，softmax(h_l^m W_y)
        if labels is None:
            return logits, None, None, None

        loss_cls = F.cross_entropy(logits, labels)          # L2：任务交叉熵
        loss_lm = torch.zeros((), device=h.device)
        if lam > 0:
            lm_logits = self.gpt.lm_head(h)                 # [B, T, V]，复用预训练的 LM head
            loss_lm = F.cross_entropy(                      # L1(C)：在标注数据的文本上继续 next-token
                lm_logits.reshape(-1, lm_logits.size(-1)), lm_targets.reshape(-1), ignore_index=-100
            )

        # TODO(you) (3): 总损失 L3 = L2 + λ·L1（知识卡 A2 §3 第三个公式）。
        loss = None  # ← 替换这一行
        if loss is None:
            raise NotImplementedError("TODO(you) (3)")
        return logits, loss, loss_cls, loss_lm


def build_gpt(init: str, ckpt_path: str, n_new_tokens: int):
    """从 train.py 的 checkpoint 建模型。词表扩 n_new_tokens 个（<s>、<e>），新行随机初始化。
    init="pretrained" 时载入预训练的 Θ；init="scratch" 时同样结构但全部随机。"""
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    V_old = cfg.vocab_size
    cfg.vocab_size = V_old + n_new_tokens
    gpt = GPT(cfg)
    if init == "pretrained":
        sd = dict(ck["model"])
        old_wte = sd.pop("wte.weight")
        sd.pop("lm_head.weight", None)                     # 与 wte 绑定，只需拷一次
        gpt.load_state_dict(sd, strict=False)
        with torch.no_grad():
            gpt.wte.weight[:V_old] = old_wte                # 老 token 用预训练向量，新 token 保持随机
    return gpt, ck["stoi"], V_old


# ---------------------------------------------------------------- 训练 ----
@torch.no_grad()
def evaluate(model, items, stoi, S, E, device, batch_size=64):
    model.eval()
    correct = 0
    for i in range(0, len(items), batch_size):
        x, e_pos, y, _ = collate(items[i : i + batch_size], stoi, S, E, device)
        logits, *_ = model(x, e_pos)
        correct += (logits.argmax(-1) == y).sum().item()
    model.train()
    return correct / len(items)


def summary():
    if not os.path.exists(RESULTS):
        print("no results yet")
        return
    rows = list(csv.DictReader(open(RESULTS)))
    print(f"{'init':10s} {'n_train':>7s} {'lam':>4s} {'lr':>7s}  {'test_acc':>8s}  {'majority':>8s}  {'train_loss':>10s}")
    for r in sorted(rows, key=lambda r: (r["init"], int(r["n_train"]), float(r["lam"]), float(r["lr"]))):
        print(f"{r['init']:10s} {r['n_train']:>7s} {r['lam']:>4s} {r['lr']:>7s}  {float(r['test_acc']):8.3f}  "
              f"{float(r['majority']):8.3f}  {float(r['final_train_loss']):10.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--n_train", type=int, default=100, help="100 或 1000；总池约 1200 条")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--k", type=int, default=5, help="角色数")
    p.add_argument("--ckpt", type=str, default="outputs/e1_tiny_gpt.pt")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--summary", action="store_true", help="只打印 results.csv 汇总")
    args = p.parse_args()
    if args.summary:
        summary()
        return

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 模型与词表 ----
    gpt, stoi, V_old = build_gpt(args.init, args.ckpt, n_new_tokens=2)
    S, E = V_old, V_old + 1                                 # <s>、<e> 的 id
    model = GPTClassifier(gpt, args.k).to(device)
    T_max = gpt.cfg.block_size

    # ---- 数据 ----
    text = load_text(os.path.join(os.path.dirname(args.ckpt) or ".", "cache"))
    pool, test, top = build_dataset(parse_speeches(text), args.k, max_chars=T_max - 2, seed=args.seed)
    train = pool[: args.n_train]                            # pool 约 1200 条，n_train 超过则取全部
    majority = max(Counter(l for _, l in test).values()) / len(test)
    print(f"device: {device}  init: {args.init}  n_train: {len(train)}  n_test: {len(test)}  "
          f"lam: {args.lam}  lr: {args.lr}")
    print(f"classes: {top}")
    print(f"majority baseline on test: {majority:.3f}   chance: {1/args.k:.3f}")

    # ---- 微调：全部参数（Θ 和 W_y）一起更新 ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    order = list(range(len(train)))
    ptr = 0
    for step in range(1, args.steps + 1):
        if ptr + args.batch_size > len(order):
            random.shuffle(order)
            ptr = 0
        batch = [train[i] for i in order[ptr : ptr + args.batch_size]]
        ptr += args.batch_size
        x, e_pos, y, lm_y = collate(batch, stoi, S, E, device)
        _, loss, loss_cls, loss_lm = model(x, e_pos, y, lm_y, lam=args.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps:
            acc = evaluate(model, test, stoi, S, E, device)
            print(f"step {step:4d}  loss {loss.item():.3f}  (cls {loss_cls.item():.3f}  lm {loss_lm.item():.3f})  "
                  f"test acc {acc:.3f}  {time.time() - t0:.0f}s")

    acc = evaluate(model, test, stoi, S, E, device)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["init", "n_train", "lam", "lr", "steps", "seed", "test_acc", "majority", "final_train_loss"])
        w.writerow([args.init, len(train), args.lam, args.lr, args.steps, args.seed,
                    f"{acc:.4f}", f"{majority:.4f}", f"{loss.item():.4f}"])
    print(f"final test acc {acc:.3f}  (majority {majority:.3f})  → appended to {RESULTS}")


if __name__ == "__main__":
    main()
