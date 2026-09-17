---
title: Track A · Pretraining & Scaling
track: A
level: 1
updated: 2026-09-13
tags: [openai, track/A, moc]
---

# Track A · Pretraining & Scaling

主线问题：如何在没有标注的情况下，用越来越大的模型学到可迁移的通用能力？

## 主线因果图

### 我的版本

学完本线后在 session 里填因果表（第一档），教练转成图写在这里。复测时用下面参考版的骨架做填空（第二档）。

| 上一代 | 下一代 | 上一代的瓶颈 |
|---|---|---|
| | | |

### 教练参考版

> [!example]- 学完并交了因果表之后再展开对照
> ```mermaid
> flowchart LR
>   T[Transformer decoder<br/>next-token objective]:::model -->|"任务专用模型无法共享表征"| G1[Sentiment Neuron → GPT-1<br/>pretrain + fine-tune]:::model
>   G1 -->|"每个任务仍需 fine-tune"| G2[GPT-2<br/>zero-shot, prompt 即任务]:::model
>   G2 -->|"下一次该训多大？"| SL[Scaling Laws<br/>幂律 + compute-efficient]:::data
>   SL -->|"zero-shot 不稳定，few-shot 要梯度"| G3[GPT-3<br/>in-context learning]:::model
>   classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
>   classDef data fill:#EEF3FA,stroke:#3B6BB3,color:#111
> ```

## 概念卡

| # | 卡片 | 状态 | Mastery |
|---|---|---|---|
| A1 | [[transformer-decoder-next-token]] | canonical | 1 |
| A2 | [[sentiment-neuron-gpt-1]] | 未写 | 0 |
| A3 | [[gpt-2-zero-shot]] | 未写 | 0 |
| A4 | [[scaling-laws]] | 未写 | 0 |
| A5 | [[gpt-3-in-context-learning]] | 未写 | 0 |

## 实验

E1 tiny GPT → E2 scaling 拟合 → E3 ICL 探针，共用一套代码。

## 对照（非 OpenAI）

Attention Is All You Need（A1 前置）、Chinchilla（A4 对照）。

← [[index]]
