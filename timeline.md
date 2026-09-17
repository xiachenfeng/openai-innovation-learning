# OpenAI Public Innovation Timeline

Last verified: 2026-09-13

| Year | Milestone | Track | Core public innovation |
|---|---|---|---|
| 2017 | Sentiment Neuron | A | 无监督生成模型学到可线性读出的语义特征 |
| 2017 | PPO / Deep RL from Human Preferences | B | Clipped policy objective；从成对比较学 reward |
| 2018 | GPT-1 | A | Generative pretraining + supervised fine-tuning |
| 2019 | GPT-2 | A | Zero-shot task behavior and scaled language modeling |
| 2020 | Scaling Laws / GPT-3 | A | Power-law predictability and in-context few-shot learning |
| 2020 | Learning to Summarize | B | RM + PPO + KL penalty 用于语言模型 |
| 2021 | CLIP / DALL·E | S | Natural-language supervision, text-to-image |
| 2021 | Codex (paper) / WebGPT | D | Code fine-tune + pass@k；browser actions + citation RLHF |
| 2022 | InstructGPT / ChatGPT / Whisper | B, S | SFT → RM → PPO；dialogue deployment；weakly supervised speech |
| 2023 | GPT-4 / Process Supervision / Weak-to-strong | A, C, B | Predictable scaling；step-level rewards；scalable oversight |
| 2024 | GPT-4o / o1 / CriticGPT / Prover-Verifier / Instruction Hierarchy | S, C, B | Omni interaction；reasoning RL；critique；legibility；layered instructions |
| 2024 | Deliberative Alignment | B | Safety spec as reasoning training material |
| 2025 | o3/o4-mini / Deep Research / Codex (agent) / GPT-5 / gpt-oss | C, D | Tool-integrated reasoning；agents；unified router；public MoE |
| 2025 | CoT Monitoring / Codex agent loop & harness / Symphony | B, D | Monitorability；open agent loop；orchestration spec |
| 2026 | GPT-5.4 / 5.5 / 5.6 (Sol, Terra, Luna) | C, D | Long-horizon work；token efficiency；max effort；ultra multi-agent |
| 2026 | GPT-Red / GPT-Live | B, S | Self-play red teaming；full-duplex voice |
| 2026 | GPT-6 Astra | F | 尚未分类，见 `knowledge/F-frontier/gpt-6-astra.md` |

Track 列：A = Pretraining & Scaling，B = Alignment，C = Reasoning，D = Agents & Systems，S = Survey，F = Frontier 待分类。

注意：时间线只记录公开信息，不代表完整内部研发顺序。2017 年两项是整条 alignment 线的起点，原时间线把 human feedback 标为 2020 起是错误的。
