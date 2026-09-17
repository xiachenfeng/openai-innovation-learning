# Experiment

- Name: E1 tiny GPT（字符级 decoder-only 语言模型）
- Date: 2026-09-16 启动
- Related concept: [[transformer-decoder-next-token]]（A1）、[[sentiment-neuron-gpt-1]]（A2，第二阶段）
- Status: 骨架已写，教练参考解在本地 CPU 验证通过；待用户填空后在 4090 上跑 smoke test

## Question

1. 本周（A1）：一个按知识卡 13 步实现的 decoder block 叠 4 层，能否跑通一次前向和一次反向，初始 loss 是否接近 ln V？
2. 下周（A2）：同一模型先预训练再 SFT，下游指标是否优于从零训练？

## Hypothesis

- 随机初始化时 loss ≈ ln V（V=65 时约 4.17），因为 softmax 近似均匀；
- 训练 2000 步后 val loss 应降到 1.5 左右（字符级 tinyshakespeare 的常见水平）。

## Environment

- 机器：用户的 NVIDIA 4090 云主机（CUDA），本地 Mac 无 torch；
- 依赖：Python ≥ 3.10，torch；
- 数据：tinyshakespeare（约 1MB 公开文本，脚本自动下载，失败退回内置文本）。

## Implementation

- `tiny_gpt/model.py`：GPTConfig、CausalSelfAttention（步 2～9）、MLP（步 12）、Block（步 1～13）、GPT（图 2）。两处 `TODO(you)`：causal mask、交叉熵；
- `tiny_gpt/train.py`：字符级词表、get_batch（input 与右移一位的 target）、smoke 模式、AdamW 训练循环、采样。

默认配置：L=4，d=128，h=4，T=128，非 embedding 参数 ≈ 12·4·128² ≈ 786k。

## Variables

本周无变量，只验证可运行。下周加：pretrain 步数、SFT 数据、从零训练对照。

## Metrics

- smoke：logits shape 为 [B, T, V]，初始 loss 与 ln V 的差；
- 训练：train/val loss 曲线；采样文本是否像英文。

## Reproduction Commands

```bash
cd experiments/pretraining/tiny_gpt
python train.py --smoke
python train.py --steps 2000
```

## Results

### 教练参考解验证（2026-09-16，本地 Mac CPU，torch CPU 版）

```text
vocab: 65  train tokens: 1,003,854
params (non-embedding): 801,664   12·L·d² ≈ 786,432
x: (8, 128)  y: (8, 128)
logits: (8, 128, 65)  loss: 4.2323   (ln V = 4.1744)
backward ok, grad norm: 4.5389
smoke test passed
```

100 步训练（batch 16）：step 50 val 2.849，step 100 val 2.645，4 秒。采样文本已有英文词形但无意义，符合预期。

### 用户 4090 运行

待填：粘贴 smoke 输出。

## Interpretation

待填。

## Limitations

- 字符级、<1M 参数、单一语料，只用于验证机制，不代表任何 OpenAI 模型的行为；
- 无 dropout、无学习率调度，训练效率不是目标。

## Cost

- 算力：4090 单卡，smoke < 1 分钟；2000 步预计 2～5 分钟；
- API：无。

## Knowledge Changes

待 smoke 通过后在 [[transformer-decoder-next-token]] 的"掌握证据"里记录。

## Next Experiment

E1 第二阶段（A2）：pretrain → SFT vs from-scratch。之后 E2 复用同一代码训 3～4 个尺寸。
