---
title: Track S · Survey
track: S
level: 1
updated: 2026-09-13
tags: [openai, track/S, moc]
---

# Track S · Survey

目标 mastery 1～2：能识别时间、术语和范式差异，不做实验。

## 范式对比

```mermaid
flowchart TB
  subgraph Contrastive
    CLIP[CLIP<br/>图文对比 · zero-shot 分类]:::model
  end
  subgraph Generative
    DE[DALL·E 系列<br/>自回归 → 扩散]:::model
    WH[Whisper<br/>弱监督语音]:::model
    SO[Sora<br/>space-time patches]:::model
  end
  subgraph Omni
    G4[GPT-4 → GPT-4o<br/>单模型多模态实时交互]:::model
    GL[GPT-Live<br/>full-duplex voice]:::model
  end
  classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
```

## 概念卡

| # | 卡片 | 状态 | Mastery |
|---|---|---|---|
| S1 | [[multimodal-paradigms]] | 未写 | 0 |
| S2 | [[gpt-live-full-duplex]] | 未写 | 0 |

← [[index]]
