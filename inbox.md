# Inbox

- [ ] 第 5 周前申请模型 API key（E3、E5、E6 需要），记录预算上限。
- [x] 确认 4090 云主机的 PyTorch/CUDA 环境可用（2026-09-17 E1 smoke 通过）。
- [ ] WEEK1_PLAN 按『多次 40 分钟』切分：诊断（已完成）/ A1 讲解 / A1 手推 / E1 骨架。

## 未来计划：Anthropic 十篇论文对照线（2026-10-11 记录）

- [ ] 主课程四条线走完后（或 Track B 前半结束时先学其中的对齐五篇），开一条"非 OpenAI 对照"小线，学 Anthropic 引用最高的十篇 LLM 论文。规则沿用 AGENTS.md：只用于解释通用机制和做对比，不推断 OpenAI 内部实现；知识卡写明"非 OpenAI"。开始前决定存放位置（建议新建 `knowledge/X-contrast/`，需先在 `knowledge/README.md` 登记）。
- [ ] 每篇 1 个 40 分钟块，顺序按与主线的对应关系排：

| # | 论文 | 年 | 对照的主线节点 | 看什么 |
|---|---|---|---|---|
| 1 | Training a Helpful and Harmless Assistant with RLHF (Bai 2022) | 2022 | B3 InstructGPT | 与 InstructGPT 同期的另一套 RLHF 配方；helpful/harmless 冲突怎么处理 |
| 2 | Constitutional AI (Bai 2022) | 2022 | B3 / B4 | 用模型自评替代人标（RLAIF），对照 OpenAI 的 Deliberative Alignment |
| 3 | A General Language Assistant as a Laboratory for Alignment (Askell 2021) | 2021 | B1 / B3 | 偏好模型预训练、prompt 蒸馏；对齐税的提法 |
| 4 | Language Models (Mostly) Know What They Know (Kadavath 2022) | 2022 | B4 / C3 | 校准与自评 P(IK)，对照 OpenAI 的 confidence 评测 |
| 5 | Towards Understanding Sycophancy (Sharma 2023) | 2023 | B3 / B4 | RLHF 偏好数据如何奖励迎合；对照 InstructGPT 局限段 |
| 6 | Discovering LM Behaviors with Model-Written Evaluations (Perez 2022) | 2022 | B4 | 用模型生成评测集，对照 OpenAI evals |
| 7 | A Mathematical Framework for Transformer Circuits (Elhage 2021) | 2021 | A1 / A5 | 残差流、QK/OV 回路、K-composition；已在 A5 §11.5 浅涉 |
| 8 | In-context Learning and Induction Heads (Olsson 2022) | 2022 | A5 / E3 | 已在 A5 §11.5 与 ELI5 图画书覆盖，届时只做复测 |
| 9 | Toy Models of Superposition (Elhage 2022) | 2022 | A1 / A4 | 特征多于神经元时的叠加，为 10 做前置 |
| 10 | Towards Monosemanticity / Scaling Monosemanticity (Bricken 2023 / Templeton 2024) | 2023–24 | D4 gpt-oss | 稀疏自编码器拆特征；可用 gpt-oss 做 E9 的延伸实验 |

- [ ] 引用数来自 Semantic Scholar 2026-10-11（见 `sources/official-sources.md` R-6、R-7 及 A5 §11.5），开始前重新核一次名次。
- [ ] 可选延伸（引用增长快、与 Track B5/C 相关）：Sleeper Agents (2024)、Alignment Faking (2024)、Reasoning Models Don't Always Say What They Think (2025)。
