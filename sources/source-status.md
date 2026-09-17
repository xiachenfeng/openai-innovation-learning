# Source Status

Last checked: 2026-09-13（限定 openai.com 域名的搜索；页面未逐条打开）

## Current Public Frontier

- Latest public GPT model: **GPT-6 Astra**（2026-08-01 之后发布；有 system card、safety overview 和 "Path to Astra" 文章）
- 状态：未分类，已建 `knowledge/F-frontier/gpt-6-astra.md`
- 上一代：GPT-5.6 系列（Sol / Terra / Luna），已纳入 Track C、D
- GPT-5.6 公开系统机制：max reasoning effort；ultra mode 用 subagents 并行 workstream；显式 prompt cache breakpoint
- GPT-5.6 公开效率披露：load balancing、speculative decoding、caching、kernel optimization、harness 层减少 context bloat
- 完整底层网络架构：全部未公开

## Other Current Tracks

- GPT-Live：full-duplex voice；有 system card 和 realtime system 工程博客；API 版本 GPT-Live-1
- GPT-Red：self-play red teaming；有论文 PDF；官方称已接入生产模型训练（Public system behavior）
- Codex：agent loop、App Server harness、subagents 文档、Symphony
- gpt-oss：唯一可深读的公开架构

## Watch List

- GPT-6 Astra 的 system card 中是否披露新的训练或推理机制
- 新 reasoning-training 出版物
- Agent orchestration 与 compaction 研究
- Open-weight 模型更新
- Voice 架构论文

## 访问限制

openai.com 对 curl、WebFetch、内置浏览器返回 403。可用方式：限定域名的 WebSearch，或用户在自己浏览器打开后粘贴。

## Update Protocol

1. 只搜官方来源；
2. 记录日期、URL 和核验方式；
3. 建 candidate；
4. 区分 product behavior 与 algorithmic disclosure；
5. 不从 benchmark 推断闭源架构。
