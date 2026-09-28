"""
E3 in-context learning 探针：GPT-2 四个开源尺寸上的 zero / one / few-shot 曲线（知识卡 A5 §6 图 1.2 的缩小版）。

用法（云主机，experiments/in-context-learning/ 目录下）：
  python icl_probe.py --models gpt2 --shots 0,4 --seeds 2 --n_eval 50      # 冒烟，1 分钟
  bash run_probe.sh                                                          # 四个尺寸全跑并汇总
  python icl_probe.py --summarize                                            # 只汇总已有 CSV

任务一（默认）：SST-2 情感分类，打分式读法。
  prompt = K 条 "Review: …\\nSentiment: positive/negative" + 查询 "Review: …\\nSentiment:"
  只做一次前向，比较末位 " positive" 与 " negative" 两个 token 的 logit，大者为预测。
  这就是 GPT-1 §5 / GPT-3 多数基准用的"比较候选概率"读法（知识卡 A3 §2.3、A5 §2.2）。
任务二（可选，--task addition）：两位数加法，生成式读法（贪心解码后解析整数）。
  预期 GPT-2 尺寸上接近 0，对应 GPT-3 论文图 3.10 里 13B 以下的水平。

与 GPT-3 论文的对应与差别：
  相同：示例从训练集随机抽、参数冻结、K 是唯一变量、打分式读法；
  不同：模型小 100～1000 倍；每个 K 换 5 组示例并报告均值 ± 标准差（论文只报单次）。

带 `TODO(you)` 的三处留给你填，每处一行：
  (1) 拼 prompt：示例段落 + 查询段落；
  (2) 打分式读法：比较两个标签 token 的 logit；
  (3) 汇总时的样本标准差。
其余代码完整。
"""
import argparse
import csv
import os
import random
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "outputs", "e3_icl.csv")

LABELS = {0: " negative", 1: " positive"}          # SST-2 的 label id → 标签词（前面带空格，GPT-2 BPE 里各是一个 token）
FIELDS = ["model", "n_params", "task", "k", "seed", "n_eval", "acc", "majority", "pred_pos_rate",
          "prompt_tokens_mean", "elapsed_s"]


# ---------------------------------------------------------------- 数据 ----
def load_sst2():
    """SST-2：train 6.7 万条做示例池，validation 872 条做查询集。需要 `datasets` 包与网络。"""
    from datasets import load_dataset
    try:
        ds = load_dataset("stanfordnlp/sst2")
    except Exception:                                  # noqa: BLE001  旧版路径
        ds = load_dataset("glue", "sst2")
    train = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"]]
    val = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"]]
    return train, val


def sst2_line(sentence: str, label: int | None) -> str:
    """一条样本写成两行；label 为 None 时是查询，以 'Sentiment:' 结尾等模型补词。"""
    tail = "" if label is None else LABELS[label]
    return f"Review: {sentence}\nSentiment:{tail}"


def build_prompt(demos, query_sentence: str) -> str:
    """K 条示例（可为空）+ 查询，示例之间用空行分隔（GPT-3 论文用 1～2 个换行分隔示例）。"""
    demo_block = "\n\n".join(sst2_line(s, y) for s, y in demos)
    query_block = sst2_line(query_sentence, None)
    # TODO(you) (1): 把示例块和查询块拼成最终 prompt。K = 0 时 demo_block 是空串，不要多出前导空行。
    #   提示：demo_block 非空时 demo_block + "\n\n" + query_block，否则只有 query_block。
    return (demo_block + "\n\n" + query_block) if demo_block else query_block  # ← 替换这一行


def make_addition(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        out.append((f"Q: What is {a} plus {b}?", a + b))
    return out


def addition_prompt(demos, query: str) -> str:
    demo_block = "\n\n".join(f"{q}\nA: {ans}" for q, ans in demos)
    query_block = f"{query}\nA:"
    return (demo_block + "\n\n" + query_block) if demo_block else query_block


# ---------------------------------------------------------------- 模型 ----
def load_model(name: str, device: str):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return tok, model


def label_token_ids(tok):
    ids = {}
    for y, word in LABELS.items():
        enc = tok.encode(word)
        assert len(enc) == 1, f"{word!r} 不是单个 token: {enc}"
        ids[y] = enc[0]
    return ids


def run_sst2(tok, model, device, train, val, k: int, seed: int, n_eval: int):
    """一个 (k, seed) 设置：抽 K 条示例，对固定的 n_eval 条查询逐条前向，返回 acc 等。"""
    import torch
    rng = random.Random(seed)
    demos = rng.sample(train, k) if k > 0 else []
    ids = label_token_ids(tok)
    neg_id, pos_id = ids[0], ids[1]

    correct, pred_pos, n_tok = 0, 0, 0
    with torch.no_grad():
        for sentence, y in val[:n_eval]:
            prompt = build_prompt(demos, sentence)
            enc = tok(prompt, return_tensors="pt").to(device)
            n_tok += enc["input_ids"].shape[1]
            logits = model(**enc).logits                       # [1, T, V]
            # TODO(you) (2): 打分式读法。取最后一个位置的 logits，比较 pos_id 与 neg_id 两项，大者为预测（1 = positive）。
            #   提示：last = logits[0, -1]；pred = int(last[pos_id] > last[neg_id])
            pred = int(logits[0, -1, pos_id] > logits[0, -1, neg_id])  # ← 替换这一行
            correct += int(pred == y)
            pred_pos += pred
    n = min(n_eval, len(val))
    majority = max(sum(1 for _, y in val[:n] if y == 1), sum(1 for _, y in val[:n] if y == 0)) / n
    return {"acc": correct / n, "majority": majority, "pred_pos_rate": pred_pos / n,
            "prompt_tokens_mean": n_tok / n, "n_eval": n}


def run_addition(tok, model, device, k: int, seed: int, n_eval: int):
    """生成式读法：贪心解码 4 个 token，正则抽第一个整数。"""
    import torch
    demos = make_addition(k, seed=1000 + seed)
    queries = make_addition(n_eval, seed=7)                  # 查询集固定，与 seed 无关
    correct, n_tok = 0, 0
    with torch.no_grad():
        for q, ans in queries:
            prompt = addition_prompt(demos, q)
            enc = tok(prompt, return_tensors="pt").to(device)
            n_tok += enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:])
            m = re.search(r"-?\d+", text)
            correct += int(m is not None and int(m.group()) == ans)
    return {"acc": correct / n_eval, "majority": 0.0, "pred_pos_rate": float("nan"),
            "prompt_tokens_mean": n_tok / n_eval, "n_eval": n_eval}


# ---------------------------------------------------------------- 汇总 ----
def summarize(csv_path: str):
    """按 model × k 汇总：acc 的均值 ± 样本标准差（跨 seed），并打印 few-shot 与 zero-shot 的差距。"""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("CSV 为空")
        return
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        models = sorted({r["model"] for r in sub}, key=lambda m: int(next(r for r in sub if r["model"] == m)["n_params"]))
        ks = sorted({int(r["k"]) for r in sub})
        print(f"\n== task: {task}   acc 均值 ± 样本标准差（跨 seed），majority = {float(sub[0]['majority']):.3f}")
        print("model".ljust(14) + "N".rjust(8) + "".join(f"k={k}".rjust(16) for k in ks) + "  gap(max k − 0)")
        for m in models:
            cells, means = [], {}
            for k in ks:
                accs = [float(r["acc"]) for r in sub if r["model"] == m and int(r["k"]) == k]
                mean = float(np.mean(accs))
                # TODO(you) (3): 跨 seed 的样本标准差（ddof=1）；只有一个样本时记 0。
                #   提示：np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0  # ← 替换这一行
                means[k] = mean
                cells.append(f"{mean:.3f}±{std:.3f}".rjust(16))
            gap = means[max(ks)] - means[min(ks)]
            n_params = next(r for r in sub if r["model"] == m)["n_params"]
            print(m.ljust(14) + f"{int(n_params) / 1e6:6.0f}M" + "".join(cells) + f"  {gap:+.3f}")
        print("读法：gap 为正且大于对应格子的标准差，few-shot 才算有效；gap 随 N 变大，才是图 1.2 的形状。")


# ---------------------------------------------------------------- 主程序 ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="gpt2,gpt2-medium,gpt2-large,gpt2-xl")
    p.add_argument("--shots", type=str, default="0,1,4,8")
    p.add_argument("--seeds", type=int, default=5, help="每个 K>0 换几组随机示例；K=0 只跑一次")
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--task", type=str, default="sst2", choices=["sst2", "addition"])
    p.add_argument("--out_csv", type=str, default=DEFAULT_CSV)
    p.add_argument("--summarize", action="store_true", help="只汇总已有 CSV")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out_csv)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shots = [int(s) for s in args.shots.split(",")]
    train, val = (None, None)
    if args.task == "sst2":
        train, val = load_sst2()
        rng = random.Random(0)
        rng.shuffle(val)                                       # 固定打乱一次，取前 n_eval 条，所有设置共用
        print(f"SST-2: train {len(train):,}  val {len(val)}  用前 {args.n_eval} 条查询")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for name in args.models.split(","):
            tok, model = load_model(name, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n[{name}] params {n_params:,}  device {device}")
            for k in shots:
                for seed in (range(1) if k == 0 else range(args.seeds)):
                    t0 = time.time()
                    if args.task == "sst2":
                        r = run_sst2(tok, model, device, train, val, k, seed, args.n_eval)
                    else:
                        r = run_addition(tok, model, device, k, seed, args.n_eval)
                    row = {"model": name, "n_params": n_params, "task": args.task, "k": k, "seed": seed,
                           "n_eval": r["n_eval"], "acc": round(r["acc"], 4), "majority": round(r["majority"], 4),
                           "pred_pos_rate": round(r["pred_pos_rate"], 4),
                           "prompt_tokens_mean": round(r["prompt_tokens_mean"], 1),
                           "elapsed_s": round(time.time() - t0, 1)}
                    w.writerow(row)
                    f.flush()
                    print(f"  k={k} seed={seed}  acc {r['acc']:.3f}  pred_pos {r['pred_pos_rate']:.2f}  "
                          f"tokens {r['prompt_tokens_mean']:.0f}  {row['elapsed_s']}s")
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
    print("\nwrote", args.out_csv)
    summarize(args.out_csv)


if __name__ == "__main__":
    main()
