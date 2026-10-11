---
title: LLM 时代术语表
aliases: [术语表, glossary, LLM 术语]
level: 0
status: candidate
created: 2026-10-11
updated: 2026-10-11
tags: [openai, glossary, status/candidate]
---

# LLM 时代术语表

收录规则：只收 2017 年 Transformer 之后在 LLM 语境里**新造的词**，或**老词获得了新的专门含义**（如 alignment、agent）。普通深度学习词汇（softmax、dropout、batch）不收。

排序规则：按**使用频率**分三档，档内大致由高到低。频率是教练按论文、文档和工业界口头使用的普遍程度做的判断，不是统计计数；同档之间差别不大。

每条给：一句话说明、首出处（机构 / 年），有概念卡的加 wikilink。非 OpenAI 出处照规范标明，只用于解释通用机制。

---

## 一档：日常高频（读任何 LLM 材料都会碰到）

| 术语 | 一句话说明 | 首出处 | 关联卡 |
|---|---|---|---|
| LLM（大语言模型） | 参数量达数十亿以上、靠 next-token 预训练得到通用能力的语言模型；"大"没有硬阈值 | 约 2020 年起通用，GPT-3 之后固定 | [[gpt-3-in-context-learning]] |
| Prompt / prompting | 给模型的输入文本；把任务写进输入而非改权重的做法。衍生 prompt engineering、system prompt | GPT-2 / GPT-3，OpenAI 2019–2020 | [[gpt-2-zero-shot]] |
| In-context learning（ICL） | 把示例写进 prompt，模型在一次前向里识别模式并续写，不更新参数；按示例数分 zero / one / few-shot | GPT-3，OpenAI 2020 | [[gpt-3-in-context-learning]] |
| Token / 上下文窗口 | token 是模型的最小输入单位（子词）；上下文窗口是一次前向能看的最大 token 数 | BPE 早于 LLM；"上下文窗口"作为产品参数自 GPT-3 起 | [[transformer-decoder-next-token]] |
| Scaling law | 损失随参数量、数据量、算力按幂律下降的经验规律；衍生 compute-optimal、Chinchilla | OpenAI 2020（Kaplan）；Chinchilla 为 DeepMind 2022，非 OpenAI | [[scaling-laws]] |
| RLHF | 用人类偏好训练奖励模型，再用 RL 微调策略；ChatGPT 的训练范式 | OpenAI 2017（Christiano）、2022（InstructGPT） | [[instructgpt-chatgpt]] |
| Alignment（对齐） | 让模型按人的意图与价值行事；老词，LLM 时代专指这一训练目标。衍生 alignment tax、misalignment | 2016 年前后 AI 安全社区；OpenAI / Anthropic 普及 | [[B-alignment]] |
| Chain-of-thought（CoT） | 让模型先写推理步骤再给答案；后来成为 reasoning model 的训练对象 | Google 2022（Wei），非 OpenAI | [[C-reasoning]] |
| Hallucination（幻觉） | 模型流畅地生成与事实或输入不符的内容 | NLG 领域 2018 前后，LLM 时代泛化 | — |
| Instruction tuning / SFT | 用"指令 → 回答"数据做监督微调，让模型听指令 | Google FLAN 2021（非 OpenAI）；OpenAI InstructGPT 2022 | [[instructgpt-chatgpt]] |
| Reward model（奖励模型） | 用偏好数据训练的打分器，替代人在 RL 循环里给分 | OpenAI 2017 / 2020 | [[rlhf-learning-to-summarize]] |
| Agent / agentic | 模型在循环里调用工具、观察结果、再决策，直到完成任务；衍生 agent loop、harness | 老词，LLM 语境约 2023 年起 | [[D-agents-systems]] |
| Tool use / function calling | 模型输出结构化调用请求，由外部执行并把结果回填 | OpenAI 2023 | [[function-calling-deep-research]] |
| Reasoning model / test-time compute | 用 RL 训练出长思考过程的模型；推理时多算换正确率 | OpenAI 2024（o1） | [[o1-rl-on-cot]] |
| Jailbreak / prompt injection | 绕过安全训练让模型违规输出；注入指把指令藏在模型读到的数据里 | 社区 2022–2023 | — |
| Emergent abilities（涌现） | 小模型没有、到某个规模突然出现的能力；是否真"突然"有争议（度量方式影响） | Google 2022（Wei），非 OpenAI | [[scaling-laws]] |
| RAG（检索增强生成） | 先检索相关文档塞进 prompt 再生成 | Meta 2020（Lewis），非 OpenAI | — |

## 二档：研究常用（读论文会碰到）

| 术语 | 一句话说明 | 首出处 | 关联卡 |
|---|---|---|---|
| Foundation model | 在大规模数据上预训练、可适配多种下游任务的模型 | Stanford 2021（Bommasani），非 OpenAI | — |
| Zero-shot / few-shot | 不给示例 / 给少量示例；原为 CV 术语，GPT-2/3 后指 prompt 里的示例数 | OpenAI 2019 / 2020 | [[gpt-2-zero-shot]] |
| Process vs outcome supervision（PRM / ORM） | 奖励给每一步推理 vs 只给最终答案 | OpenAI 2023（Let's Verify Step by Step） | [[process-vs-outcome-supervision]] |
| DPO | 跳过奖励模型，直接用偏好对优化策略 | Stanford 2023，非 OpenAI，B3 对照 | [[instructgpt-chatgpt]] |
| Constitutional AI / RLAIF | 用一组原则让模型自评自改，用 AI 反馈替代部分人类标注 | Anthropic 2022，非 OpenAI | — |
| Red teaming | 主动找模型的有害输出；老词（军事 / 安全），LLM 语境指对抗性评测 | OpenAI、Anthropic 2022 | — |
| Reward hacking | 策略利用奖励函数漏洞拿高分而不做对事；RL 老词，LLM 时代专指奖励模型被钻空子 | RL 社区；OpenAI 2016 起讨论 | [[B-alignment]] |
| Sycophancy（迎合） | 模型顺着用户观点说而非说真话；RLHF 偏好数据的副作用 | Anthropic 2023，非 OpenAI | — |
| Self-consistency | 多次采样推理链再投票 | Google 2022，非 OpenAI，C3 对照 | [[test-time-compute]] |
| Grokking | 训练 loss 早已收敛后，泛化在很久之后突然出现 | OpenAI 2022（Power） | — |
| Phase change / 相变 | 训练中某能力或电路在窄窗口内突然出现，借自物理 | Anthropic 2022（induction heads 论文用法），非 OpenAI | [[gpt-3-in-context-learning]] §11.5 |
| Scalable oversight | 当任务超出人类评判能力时，如何仍能监督模型 | Anthropic 2022 提出术语；OpenAI 2023 weak-to-strong | [[scalable-oversight]] |
| Mixture of experts（MoE） | 每个 token 只激活部分专家子网络；1991 年老词，LLM 时代复兴 | 1991；LLM 用法自 Google 2021、gpt-oss 2025 | [[gpt-oss-architecture]] |
| Data contamination | 评测题混进训练数据导致分数虚高 | GPT-3 论文已讨论 | [[gpt-3-in-context-learning]] |
| Synthetic data / self-play | 用模型自己生成训练数据；自博弈借自 AlphaGo | 2023 年起普遍 | — |
| Distillation（蒸馏） | 用大模型输出训练小模型；2015 年老词，LLM 时代指用 API 输出造数据 | Hinton 2015（Google） | — |
| Long-context / needle in a haystack | 超长上下文能力；大海捞针是其检索测试 | 2023 社区 | — |
| LoRA / PEFT | 只训练少量附加参数的微调 | Microsoft 2021，非 OpenAI | — |

## 三档：细分领域专有

### 可解释性（mechanistic interpretability，主要是 Anthropic，非 OpenAI）

| 术语 | 一句话说明 | 首出处 | 关联卡 |
|---|---|---|---|
| Mechanistic interpretability | 把网络权重逆向成可读的算法 / 电路 | Olah 等，OpenAI 时期（2020）起，后在 Anthropic | — |
| Circuit（电路） | 一组协作完成某功能的神经元 / 注意力头 | Olah 等 2020 | — |
| Residual stream（残差流） | 把残差连接看作各层读写的共享通道 | Anthropic 2021（Elhage） | [[gpt-3-in-context-learning]] §11.5 |
| QK / OV 回路 | 注意力头拆成"看哪里"（QK）和"搬什么"（OV）两部分 | Anthropic 2021 | 同上 |
| Induction head / previous-token head | 两个跨层配合的头，实现"找上次出现、抄后面"；ICL 的机制假设 | Anthropic 2021 / 2022 | 同上 |
| Superposition（叠加） | 特征数多于神经元数时，一个神经元混装多个特征 | Anthropic 2022 | — |
| Monosemanticity / SAE 特征 | 用稀疏自编码器把叠加拆成单义特征 | Anthropic 2023 / 2024 | — |
| Logit lens | 把中间层残差直接投到词表看"此刻在猜什么" | nostalgebraist 2020（社区） | — |
| Steering vector / activation steering | 在激活上加一个方向向量改变行为 | 2023 社区、Anthropic | — |

### 对齐与安全

| 术语 | 一句话说明 | 首出处 | 关联卡 |
|---|---|---|---|
| Deliberative alignment | 让 reasoning model 在思考时显式引用安全规范 | OpenAI 2024 | [[deliberative-alignment-gpt-red]] |
| CoT monitoring / faithfulness | 用思维链监控模型意图；忠实性指思维链是否反映真实计算 | OpenAI 2025；Anthropic 2023 | [[deliberative-alignment-gpt-red]] |
| Instruction hierarchy | system > developer > user 的指令优先级 | OpenAI 2024 | — |
| Sleeper agent / alignment faking | 植入后门能挺过安全训练；模型在训练时假装对齐 | Anthropic 2024，非 OpenAI | — |
| Weak-to-strong generalization | 用弱模型监督强模型，看强模型能否超过弱监督 | OpenAI 2023 | [[scalable-oversight]] |
| Refusal | 模型拒答有害请求的行为及其训练 | 2022 年起 | — |

### 系统与推理

| 术语 | 一句话说明 | 首出处 | 关联卡 |
|---|---|---|---|
| KV cache | 缓存已算过 token 的 K、V，避免重复计算 | 工程实践，2019 年起 | [[transformer-decoder-next-token]] |
| Speculative decoding | 小模型先猜、大模型并行验证，加速生成 | Google / DeepMind 2022，非 OpenAI | — |
| Compaction / context management | agent 长会话中压缩历史以腾出上下文 | OpenAI Codex 2025、各家 agent 框架 | [[codex-agent-loop-compaction]] |
| Router（路由） | 按请求难度把流量分给不同模型 | OpenAI 2025（GPT-5 系统） | [[gpt-5-unified-router]] |
| MCP | 模型与工具之间的开放协议 | Anthropic 2024，非 OpenAI | [[D-agents-systems]] |
| Model collapse | 反复用模型生成数据训练后分布退化 | 2023 学术界 | — |

---

## 维护

- 追问中遇到新词，加进对应档，补首出处；不确定出处写"待核"。
- 频率判断每季度顺一遍，升降档不留痕。
- 关联卡尚未写的用 `[[...]]` 占位，Obsidian 显示为未解析链接。
