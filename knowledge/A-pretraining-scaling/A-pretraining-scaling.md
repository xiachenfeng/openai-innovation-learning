---
title: Track A · Pretraining & Scaling
track: A
level: 1
updated: 2026-10-11
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

| #   | 卡片                                 | 状态        | Mastery |
| --- | ---------------------------------- | --------- | ------- |
| A1  | [[A1-transformer-decoder-next-token]] | canonical | 1       |
| A2  | [[A2-sentiment-neuron-gpt-1]]         | canonical | 2       |
| A3  | [[A3-gpt-2-zero-shot]]                | canonical | 1       |
| A4  | [[A4-scaling-laws]]                   | canonical | 2       |
| A5  | [[A5-gpt-3-in-context-learning]]      | canonical | 1       |

## 图板与图解页

每节的 Artifact 页面（私有链接，需登录 claude.ai），全表见 [[artifacts]]。

| 节点 | 页面 |
|---|---|
| A1 | [困惑度是几面骰子](https://claude.ai/artifact/KYwj7isSTGng4vsLbwBdYz)（ELI5 图解）；[一个 nat 有多惊讶](https://claude.ai/artifact/Psi2yZUrFa4kVJccehLfhh)（ELI5 图解）；[一个 token 碰到多少参数](https://claude.ai/artifact/RvHqmEJq4pzTaAbv55BnJ2)（ELI5 图解）；[Softmax 数几次](https://claude.ai/artifact/99E9zJgrQDsNnMLvgKyoDN)（ELI5 图解）；[注意力头是什么](https://claude.ai/artifact/Hf8HWThP6TwHa6dDF9g5nu)（ELI5 图解） |
| A2 | [A2 GPT-1 图板](https://claude.ai/artifact/JA8AYXzSxbitKPRk1qqtXa)（本节图板） |
| A3 | [A3 GPT-2 图板](https://claude.ai/artifact/GzWJbEhPea5v3HpWm5oLSS)（本节图板）；[字节级 BPE 分词器](https://claude.ai/artifact/2ouTn5rzu1mgUTJdXA7tz6)（ELI5 图解）；[Zero-shot 与 Few-shot](https://claude.ai/artifact/57G8vt5QYDAqwCTxvkRB6B)（ELI5 图解） |
| A4 | [A4 Scaling Laws 图板](https://claude.ai/artifact/E4C59fa4YoEvaWBgcTohCJ)（本节图板）；[Scaling Laws 图画书](https://claude.ai/artifact/2g6By9u5K8WFcvXet23Ek9)（ELI5 图解）；[对数尺度图画书](https://claude.ai/artifact/D1D9TsrBmwka6mbDXHS85k)（ELI5 图解）；[Loss 的地板](https://claude.ai/artifact/YEqy5DigiFqQ2VdCw5SyHG)（ELI5 图解）；[临界 Batch 图画书](https://claude.ai/artifact/2oP8DVbtx17BFt83nyYbE8)（ELI5 图解）；[6ND 算力账本](https://claude.ai/artifact/AtC9n7P2nMaBuEgm2a6NwL)（ELI5 图解）；[KKT 小球与墙](https://claude.ai/artifact/GgtwRJP9EVyPFdzigvLgr9)（ELI5 图解）；[Robbins–Monro 步长](https://claude.ai/artifact/2mPAdELufA9km8DoNWSMk5)（ELI5 图解） |
| A5 | [GPT-3 In-context Learning 图板](https://claude.ai/artifact/Y83YHgaWwLJeEfaCwShnuX)（本节图板）；[GPT 一二三代](https://claude.ai/artifact/BbXMezmJyrEVJuUiyuXUmG)（ELI5 图解）；[Induction Heads 图画书](https://claude.ai/artifact/Upa1wVEfQ4LBn2ahwG3aNh)（ELI5 图解）；[MAML 小画书](https://claude.ai/artifact/XNdkqWJ1ezvEF5oFaBiH66)（ELI5 图解） |

## 实验

E1 tiny GPT → E2 scaling 拟合（共用 E1 代码）→ E3 ICL 探针（换用 GPT-2 四个开源尺寸，因为 ICL 随规模出现，E1 的 80 万参数字符模型上没有信号）。

## 对照（非 OpenAI）

Attention Is All You Need（A1 前置）、Chinchilla（A4 对照）。A5 用到的 SST-2 数据集（Stanford，非 OpenAI）只作 E3 的探针任务。

← [[index]]
