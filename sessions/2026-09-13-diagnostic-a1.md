# Learning Session

- Date: 2026-09-13（讨论延续至 09-15、09-16）
- Topic: 前置诊断（8 题）+ A1 Transformer decoder 与 next-token objective
- Phase: Track A · A1
- Duration: 约 3 个 40 分钟块（诊断 / A1 讲解与追问 / 作业与 E1）

## Learning Goals

1. 完成 8 项前置诊断，写入 mastery；
2. 讲 A1，配 decoder block 数据流图；
3. 手推 4 token causal mask 与 cross entropy；
4. 启动 E1 骨架。

## Source Freshness Check

沿用 2026-09-13 的核验结果（GPT-6 Astra 为最新；openai.com 403）。A1 卡引用 Attention Is All You Need（非 OpenAI，前置）与 GPT-1 论文，均为经典公开来源。

## 诊断结果（2026-09-13）

| 题 | 领域 | 结果 | mastery |
|---|---|---|---|
| 1 | A1 decoder / fine-tune | 识别交叉熵与 causal mask；不知 block 结构；fine-tune 差异答成省算力 | A1 → 1 |
| 2 | A4/A5 ICL、scaling | 说出 N/D/C 幂律；不知 ICL 定义 | A4 → 1 |
| 3 | B1–B3 RLHF | 全部不知 | 0 |
| 4 | C1–C3 reasoning | 只知推理算力占比高；best-of-N 与 GRPO 混淆 | 0 |
| 5 | D2–D3 tool / agent | 知道模型输出调用、外部执行；知道 compact 一词 | D2 → 1 |
| 6 | S1 multimodal | 不知 | 0 |
| 7 | PyTorch | 不能独立写训练循环；有 4090 云主机；无 API key | 0 |
| 8 | 时间目标 | 多次 40 分钟；转岗；API key 第 5 周前办 | 不计 |

评分特点：诚实说"不知道"，evidence_discipline 稳定 2；自评信心与实际接近（第 2 题 3 分）。

## A1 讲解与追问

讲了 13 步 decoder block 数据流（shape 逐步表）、causal mask、目标函数、推理、scaling 维度、局限、公开边界。用户追问六个，全部写入知识卡：

1. 切 head 是沿特征维切 Q/K/V；
2. 步 2 是逐 token 独立映射 → 13 步分成"逐 token / 跨 token"两类；
3. head 切分 vs token mixer（MLP-Mixer、MetaFormer，非 OpenAI 对照）；
4. 一个 head 的 Q、K 只在本 head 内打分，W_O 才混合；
5. W_O "混合 head"的含义；
6. 为什么需要逐 token 的非线性 MLP。

用户自己总结出"沿 token 维搬运 / 逐 token 非线性加工"的物理图景，已记入卡片并链接到 A4、D3、D4。

## 作业（2026-09-16）

4 token 手推 5 题：3 对 2 错。错在 target 右移与 softmax 算成 hardmax；重做 softmax + CE 通过。评分 correctness 2、mechanism 2，A1 mastery 保持 1，7 天后复测 block 结构。

## 实验

E1 骨架写入 `experiments/pretraining/tiny_gpt/`，两处填空（causal mask、交叉熵）。教练参考解在本地 CPU 验证：smoke 通过，初始 loss 4.23 ≈ ln 65；100 步 val 2.64。待用户在 4090 上跑。

## Files Updated

- state/answer-history.csv（+9 行）、state/mastery.md、state/current.md、state/mistakes.md、inbox.md
- knowledge/A-pretraining-scaling/transformer-decoder-next-token.md（新建，canonical，mastery 1）
- knowledge/A-pretraining-scaling/A-pretraining-scaling.md（A1 行）
- experiments/pretraining/E1-tiny-gpt.md、tiny_gpt/model.py、tiny_gpt/train.py（新建）

## Canonical Knowledge Changes

新增 canonical：transformer-decoder-next-token（依据：Vaswani 2017 非 OpenAI 前置 + GPT-1 论文）。

## Next Step

1. 用户填 model.py 两处空并在 4090 跑 smoke，粘贴输出；
2. 下一块：A2 Sentiment Neuron → GPT-1，E1 加 SFT 阶段；
3. 2026-09-23 前后复测 A1 block 结构（review queue）。
