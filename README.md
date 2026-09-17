# OpenAI Innovation Algorithms Learning Project

Last verified: 2026-09-13

这是一个按时间线学习 OpenAI 从早期研究到最新公开创新的项目。

## 学习原则

OpenAI 的许多前沿模型不会公开完整网络结构、训练数据和超参数。因此，本项目重点学习：

1. 已公开的论文与方法；
2. 官方明确披露的系统机制；
3. 能被实验验证的算法思想；
4. 产品背后的 Agent、推理、对齐和效率范式；
5. 明确标记“公开事实”和“未知内部细节”。

## 主线

课程按四条因果主线组织（见 `curriculum.md`），每周 3～4 小时，约 6 个月：

- **Track A Pretraining & Scaling**：Sentiment Neuron → GPT-1 → GPT-2 → Scaling Laws → GPT-3
- **Track B Alignment**：PPO / Human Preferences（2017）→ Learning to Summarize → InstructGPT → Scalable Oversight → Deliberative Alignment / GPT-Red
- **Track C Reasoning**：Process Supervision → o1 → Test-time Compute → o3/o4 → GPT-5 Router
- **Track D Agents & Systems**：Codex 2021 / WebGPT → Function Calling / Deep Research → Codex Agent Loop → gpt-oss → Multi-agent
- **Track S Survey**（浅层）：CLIP、DALL·E、Whisper、GPT-4o、Sora、GPT-Live
- **Track F Frontier**（每月滚动）：最新 system card 与发布，先进 candidates

## 最终能力

完成后能够：

- 解释 OpenAI 技术路线的关键范式变化；
- 区分 pretraining scaling、alignment scaling、test-time scaling 和 agent scaling；
- 实现 RLHF、process reward、router、agent loop 等 toy experiments；
- 对未公开的模型内部结构保持严格证据纪律；
- 输出《OpenAI 公开算法创新演进报告》。

## 快速开始

```text
初始化这个 OpenAI 创新算法学习项目。
先只读检查 AGENTS.md、curriculum.md、timeline.md、state/ 和 sources/，
然后逐题评估我的前置知识。
```

每次开始：

```text
开始学习。
```

指定主题：

```text
今天学习从 GPT-3 到 InstructGPT 的范式变化。
```

结束：

```text
结束学习，保存进度。
```
