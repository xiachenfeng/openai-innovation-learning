# Mastery

Importance：5 = 主线因果链的关键节点；4 = 主线内重要但可替代；3 = Survey 或可选。
Mastery 变化规则见 `AGENTS.md` Mastery Scale。

## Track A：Pretraining & Scaling

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| A1 Transformer decoder / next-token objective | 5 | 1 | 2026-09-13 | 诊断：识别交叉熵与 causal mask，不知 block 结构与 shape |
| A2 Sentiment Neuron → GPT-1 pretrain + fine-tune | 5 | 0 | 2026-09-13 | 诊断：fine-tune 差异答成省算力 |
| A3 GPT-2 zero-shot / prompt conditioning | 4 | 0 | - | 未测试 |
| A4 Scaling Laws（Kaplan、Chinchilla 对照、predictable scaling） | 5 | 1 | 2026-09-13 | 诊断：说出 N/D/C 与 loss 幂律，Chinchilla 方向对但无依据 |
| A5 GPT-3 in-context learning | 5 | 0 | 2026-09-13 | 诊断：不知 in-context learning 定义 |

## Track B：Alignment

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| B1 PPO / learning from human preferences（2017） | 5 | 0 | 2026-09-13 | 诊断：不知 |
| B2 RM + PPO + KL on LMs（Learning to Summarize） | 4 | 0 | 2026-09-13 | 诊断：不知 RM 与 KL |
| B3 InstructGPT 三阶段 / ChatGPT / DPO 对照 | 5 | 0 | 2026-09-13 | 诊断：不知三阶段 |
| B4 Scalable oversight（CriticGPT、Prover-Verifier、Weak-to-strong） | 4 | 0 | - | 未测试 |
| B5 Deliberative Alignment / Model Spec / CoT Monitoring / GPT-Red | 4 | 0 | - | 未测试 |

## Track C：Reasoning

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| C1 Outcome vs process supervision / PRM | 5 | 0 | 2026-09-13 | 诊断：不知 outcome vs process |
| C2 o1 RL on CoT / 两条 scaling 曲线 | 5 | 0 | 2026-09-13 | 诊断：只知推理时算力占比高 |
| C3 Test-time compute 机制（pass@k、verifier、effort 参数） | 5 | 0 | 2026-09-13 | 诊断：best-of-N 与 GRPO 混淆 |
| C4 o3/o4-mini tool-integrated reasoning | 4 | 0 | - | 未测试 |
| C5 GPT-5 unified router / effort 分层 | 4 | 0 | - | 未测试 |

## Track D：Agents & Systems

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| D1 Codex 2021 / WebGPT | 4 | 0 | - | 未测试 |
| D2 Function calling / Responses API / Deep Research | 4 | 1 | 2026-09-13 | 诊断：知道模型输出调用、外部执行 |
| D3 Codex agent loop / compaction / harness | 5 | 0 | 2026-09-13 | 诊断：不知循环停止条件，只知 compact 一词 |
| D4 gpt-oss 公开架构 | 4 | 0 | - | 未测试 |
| D5 Multi-agent / Symphony / ultra | 5 | 0 | - | 未测试 |

## Track S：Survey（目标 mastery 1～2）

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| S1 Multimodal（CLIP、DALL·E、Whisper、GPT-4o、Sora） | 3 | 0 | 2026-09-13 | 诊断：不知 CLIP 与 4o |
| S2 GPT-Live full-duplex voice | 3 | 0 | - | 未测试 |

## 横向能力

| Knowledge Point | Importance | Mastery | Last Tested | Evidence |
|---|---:|---:|---|---|
| Public-vs-undisclosed discipline（按每题 evidence_discipline 维度累计） | 5 | 0 | - | 未测试 |
| Python/PyTorch 实验能力 | 4 | 0 | 2026-09-13 | 自报：不能独立写训练循环；有 4090 云主机 |
