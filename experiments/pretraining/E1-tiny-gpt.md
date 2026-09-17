# Experiment

- Name: E1 tiny GPT（字符级 decoder-only 语言模型）
- Date: 2026-09-16 启动
- Related concept: [[transformer-decoder-next-token]]（A1）、[[sentiment-neuron-gpt-1]]（A2，第二阶段）
- Status: 第一阶段完成（2026-09-17，用户填空 + 4090 smoke 通过）；第二阶段（pretrain → SFT vs from-scratch）待 A2

## Question

1. 本周（A1）：一个按知识卡 13 步实现的 decoder block 叠 4 层，能否跑通一次前向和一次反向，初始 loss 是否接近 ln V？
2. 下周（A2）：同一模型先预训练再 SFT，下游指标是否优于从零训练？

## Hypothesis

- 随机初始化时 loss ≈ ln V（V=65 时约 4.17），因为 softmax 近似均匀；
- 训练 2000 步后 val loss 应降到 1.5 左右（字符级 tinyshakespeare 的常见水平）。

## Environment

- 机器：用户的 NVIDIA 4090 云主机（CUDA），本地 Mac 无 torch；
- 依赖：Python ≥ 3.10（类型标注用了 `X | None` 语法），torch ≥ 2.0（教练用 2.14.0 CPU 版验证）；4090 上需 CUDA 构建，用 `torch.cuda.is_available()` 确认；无其他第三方依赖；
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

### 用户 4090 运行（2026-09-17）

用户自行填了 causal mask 与交叉熵两处空。过程中两次出错并自行修正：`cross_enrtopy` 拼写；`logits.reshape(-1, x.size(-1))` 用了隐状态维 d 而非词表维 V，改为 `logits.size(-1)`。

```text
device: cuda
text chars: 1,115,394  vocab: 65  train tokens: 1,003,854
params (non-embedding): 801,664   12·L·d² ≈ 786,432
x: (64, 128)  y: (64, 128)
logits: (64, 128, 65)  loss: 4.2245   (随机初始化时应接近 ln V = 4.1744)
backward ok, grad norm: 4.2311
smoke test passed
```

与 CPU 参考解结果一致（loss 4.22 vs 4.23，差异来自随机初始化与 batch 采样）。

## Interpretation

- 初始 loss 4.22 ≈ ln 65 = 4.17：随机初始化下 LM head 对 65 个字符打分近似均匀，交叉熵等于均匀分布的熵。略高于 ln V 是因为随机 logits 并非严格相等，softmax 稍有偏离均匀，损失只会更大；
- 非 embedding 参数 801,664 与 12·L·d² = 786,432 的差是 LayerNorm 与 bias（约 15k），推导公式成立；
- 用户第二次错误（用 d 而非 V 摊平 logits）恰好对应知识卡追问 7 的混淆点：隐状态维 d 与词表维 V 是两个不同的轴。

## Limitations

- 字符级、<1M 参数、单一语料，只用于验证机制，不代表任何 OpenAI 模型的行为；
- 无 dropout、无学习率调度，训练效率不是目标。

## Cost

- 算力：4090 单卡，smoke < 1 分钟（已用）；2000 步预计 2～5 分钟（待用）；本地 CPU 验证约 1 分钟；
- API：无。

## Knowledge Changes

[[transformer-decoder-next-token]] "掌握证据"记录 E1 第一阶段完成；mastery 保持 1（升 2 需要复测时 mechanism ≥ 3）。

## Next Experiment

E1 第二阶段（A2）：pretrain → SFT vs from-scratch。之后 E2 复用同一代码训 3～4 个尺寸。
