---
title: OpenAI 创新知识库
level: 0
updated: 2026-10-15
tags: [openai, hub]
---

# OpenAI 创新知识库

四条因果主线，两条辅线。每条主线的 MOC 里有主线因果图和概念卡索引。

| 主线 | MOC | 主线问题 |
|---|---|---|
| A | [[A-pretraining-scaling]] | 如何在没有标注的情况下学到可迁移的通用能力？ |
| B | [[B-alignment]] | 只会续写的模型如何变成按意图行事的模型？ |
| C | [[C-reasoning]] | 如何用推理时的计算换正确率，并把它训练出来？ |
| D | [[D-agents-systems]] | 模型如何在环境中多步行动，系统如何保证可靠？ |
| S | [[S-survey]] | Multimodal 与 Voice 的范式识别（浅层） |
| F | [[F-frontier]] | 最新发布的证据分类（每月滚动） |

## 学习顺序

```mermaid
flowchart LR
  A[A · Pretraining & Scaling]:::model --> B1[B1–B4 · Alignment]:::model
  B1 --> C[C · Reasoning]:::model
  C --> B5[B5 · Reasoning 时代的对齐]:::model
  B5 --> D[D · Agents & Systems]:::model
  D --> S[S · Survey]:::frozen
  F[F · Frontier 每月]:::data
  classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
  classDef frozen fill:#F2F2F2,stroke:#888,color:#333,stroke-dasharray:4 3
  classDef data fill:#EEF3FA,stroke:#3B6BB3,color:#111
```

## 论文台账

课程用到的每篇论文读到哪、哪张卡引用了它的什么：[[papers|论文台账]]。

## 图板与图解页

每次讲解发布的 Artifact 页面（本节图板、追问图板、ELI5 图解）的登记表：[[artifacts|图板与图解页台账]]。链接私有，需登录 claude.ai。

## 术语表

LLM 时代新造或改义的专有名词，按使用频率分档：[[glossary|LLM 时代术语表]]。

## 规范

写卡之前先读 [[README|知识库规范]]。
