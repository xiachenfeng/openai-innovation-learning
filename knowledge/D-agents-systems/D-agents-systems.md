---
title: Track D · Agents & Systems
track: D
level: 1
updated: 2026-09-13
tags: [openai, track/D, moc]
---

# Track D · Agents & Systems

主线问题：模型如何在环境里多步行动，系统如何让这种行动可靠？

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
>   C21[Codex 2021 · WebGPT<br/>代码 fine-tune · 浏览器动作]:::model -->|"工具需要结构化接口"| FC[Function Calling · Responses API · Deep Research]:::data
>   FC -->|"长任务撑爆上下文，动作有副作用"| AL[Codex 2025<br/>agent loop · compaction · harness]:::model
>   AL -->|"闭源架构不可审计"| OSS[gpt-oss<br/>公开 MoE 架构]:::model
>   AL -->|"单 agent 串行，难以人工审查"| MA[Multi-agent<br/>subagents · Symphony · ultra]:::human
>   classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
>   classDef data fill:#EEF3FA,stroke:#3B6BB3,color:#111
>   classDef human fill:#EAF7EE,stroke:#2E8B57,color:#111
> ```

## 概念卡

| # | 卡片 | 状态 | Mastery |
|---|---|---|---|
| D1 | [[codex-2021-webgpt]] | 未写 | 0 |
| D2 | [[function-calling-deep-research]] | 未写 | 0 |
| D3 | [[codex-agent-loop-compaction]] | 未写 | 0 |
| D4 | [[gpt-oss-architecture]] | 未写 | 0 |
| D5 | [[multi-agent-orchestration]] | 未写 | 0 |

## 实验

E6 agent loop + compaction；可选 E7 multi-agent、E9 gpt-oss 架构分析。

← [[index]]
