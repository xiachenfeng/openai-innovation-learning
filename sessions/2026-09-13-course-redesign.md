# Learning Session

- Date: 2026-09-13
- Topic: 课程设计评审与重构（非学习 session）
- Phase: Track A 之前
- Duration: 约 1 小时

## Learning Goals

评审课程设计，找出不合理处，按用户决定重构。

## Source Freshness Check

- 39 条原始链接：30 条为 2025 年中前的经典页面（知识确认），9 条前沿链接经限定域名搜索确认存在；
- 发现 GPT-6 Astra 已发布，source baseline（GPT-5.6）过期；
- openai.com 对 curl / WebFetch / 内置浏览器返回 403。

## 用户决定

1. 每周 3～4 小时；
2. Multimodal 与 Voice 只做 survey；
3. 核验链接；
4. 按四条主线重组课程；
5. 多用图例教学。

## 主要发现

- 课程按产品顺序排列，四条主线被拆散；
- 范围过宽，Week 1 计划与"每周最多三个主题"矛盾；
- 前沿内容硬编码进课程；
- 缺 2017 年 PPO、human preferences、sentiment neuron 三个源头；
- answer-history 列缺 historical_causality 和 evidence_discipline；
- 诊断清单有三个不同版本；
- 知识卡模板与十个 lens 不对应；
- 实验彼此独立且无算力预算。

## Files Updated

- curriculum.md（重写：Track A/B/C/D/S/F）
- timeline.md（加 2017 起点、Codex 2021、GPT-6 Astra、Track 列）
- WEEK1_PLAN.md（按 3～4 小时重写）
- AGENTS.md（图例规则、分数到 mastery 映射、review 间隔、非 OpenAI 参考政策、统一诊断清单）
- CLAUDE.md → 符号链接到 AGENTS.md
- state/answer-history.csv（列名对齐七维度）
- state/mastery.md（按 Track 分组、重新分配 importance）
- state/current.md（baseline 更新）
- templates/knowledge-card.md（与十个 lens 一一对应，加图解段）
- FIRST_SESSION_PROMPT.md（统一 8 项清单）
- experiments/README.md（实验总表）
- sources/official-sources.md（按 Track 分组、核验状态列、新增源头与非 OpenAI 参考）
- sources/source-status.md（GPT-6 Astra、访问限制）
- knowledge/ 重排为两级（index + 六个主线 MOC + 概念卡），status 移入 frontmatter；knowledge/README.md 写 Obsidian/mermaid/LaTeX 规范；knowledge/F-frontier/gpt-6-astra.md（新建）
- templates/knowledge-card.md 改为 Obsidian 格式；templates/knowledge-card-example.md 新建示范

## Canonical Knowledge Changes

无。本次不产生知识卡。

## 补充：画图分工

用户问自己画图难不难。定为三档：填因果表（默认）、填空骨架（复测）、自己写 mermaid（自愿）。评分看内容不看语法。已写入 AGENTS.md、curriculum.md、knowledge/README.md、templates/session.md；四个主线 MOC 的教练参考图改为折叠 callout，新增"我的版本"空表。

## Next Step

按 WEEK1_PLAN.md 开始：前置诊断 + A1。
