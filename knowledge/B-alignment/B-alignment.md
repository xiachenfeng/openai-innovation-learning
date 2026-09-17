---
title: Track B · Alignment
track: B
level: 1
updated: 2026-09-13
tags: [openai, track/B, moc]
---

# Track B · Alignment

主线问题：一个只会续写的模型，如何变成按人的意图行事的模型？

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
>   P[PPO + Human Preferences<br/>2017]:::model -->|"用在语言模型上会 reward hacking 吗？"| LS[Learning to Summarize<br/>RM + PPO + KL, 2020]:::model
>   LS -->|"GPT-3 不听指令"| IG[InstructGPT / ChatGPT<br/>SFT → RM → PPO, 2022]:::model
>   IG -->|"模型超过标注者后怎么监督？"| SO[Scalable Oversight<br/>CriticGPT · PVG · Weak-to-strong]:::human
>   SO -->|"有了推理链，安全训练能否利用它？"| DA[Deliberative Alignment · CoT Monitoring · GPT-Red]:::model
>   classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
>   classDef human fill:#EAF7EE,stroke:#2E8B57,color:#111
> ```

B5 建立在 [[C-reasoning]] 之上，学完 Track C 再回来。

## 概念卡

| # | 卡片 | 状态 | Mastery |
|---|---|---|---|
| B1 | [[ppo-human-preferences-2017]] | 未写 | 0 |
| B2 | [[rlhf-learning-to-summarize]] | 未写 | 0 |
| B3 | [[instructgpt-chatgpt]] | 未写 | 0 |
| B4 | [[scalable-oversight]] | 未写 | 0 |
| B5 | [[deliberative-alignment-gpt-red]] | 未写 | 0 |

## 实验

E4 reward model + 策略优化；可选 E8 无害环境自博弈。

## 对照（非 OpenAI）

DPO（B3 对照）。

← [[index]]
