---
title: Track C · Reasoning
track: C
level: 1
updated: 2026-09-13
tags: [openai, track/C, moc]
---

# Track C · Reasoning

主线问题：如何让模型在推理时花更多计算换更高正确率，并把这种行为训练出来？

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
>   PS[Process Supervision<br/>PRM vs ORM, 2023]:::data -->|"prompt 出来的 CoT 不随训练变强"| O1[o1<br/>RL on CoT, 2024]:::model
>   O1 -->|"怎样把'多想一会儿'变成可控参数？"| TT[Test-time Compute<br/>pass@k · verifier · effort]:::data
>   TT -->|"纯文本推理无法执行和查证"| O3[o3 / o4-mini<br/>tool-integrated reasoning]:::model
>   O3 -->|"用户不该自己选模型"| G5[GPT-5 Unified System<br/>router · effort 分层]:::model
>   classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
>   classDef data fill:#EEF3FA,stroke:#3B6BB3,color:#111
> ```

## 概念卡

| # | 卡片 | 状态 | Mastery |
|---|---|---|---|
| C1 | [[process-vs-outcome-supervision]] | 未写 | 0 |
| C2 | [[o1-rl-on-cot]] | 未写 | 0 |
| C3 | [[test-time-compute]] | 未写 | 0 |
| C4 | [[o3-tool-integrated-reasoning]] | 未写 | 0 |
| C5 | [[gpt-5-unified-router]] | 未写 | 0 |

## 实验

E5 ORM vs PRM + pass@k；可选 E7 router。

## 对照（非 OpenAI）

Self-Consistency（C3 对照）。

← [[index]]
