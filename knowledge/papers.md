---
title: 论文台账
level: 1
updated: 2026-10-11
tags: [openai, hub, papers]
---

# 论文台账

课程里用到的每篇论文一条记录：读到了论文的哪些章节、哪张概念卡的哪一段引用了它的什么信息。维护规则见 `AGENTS.md`「论文台账」；URL 与核验方式在 `sources/official-sources.md`（编号 A-x / R-x），本页只记"用在哪"。

首版于 2026-10-11 按已有概念卡整理；"已读章节"指课程讲解覆盖到的论文章节，未列出的即未读。状态：已学 = 对应课程节点讲完；部分 = 只学了其中一段；仅引用 = 追问或对照时引用过，未系统读；未学 = 课程计划内但尚未到。

## 总表

| # | 论文 | 年份 | 来源级别 | 课程节点 | 已读章节 | 状态 |
|---|---|---|---|---|---|---|
| P-1 | Attention Is All You Need | 2017 | 非 OpenAI，前置（R-1） | A1 | §3.2 | 部分 |
| P-2 | Learning to Generate Reviews and Discovering Sentiment（Sentiment Neuron） | 2017 | OpenAI（A-2） | A2 | 主结果与开源代码 | 已学 |
| P-3 | Improving Language Understanding by Generative Pre-Training（GPT-1） | 2018 | OpenAI（A-3） | A1、A2 | §3、§5、表 2～5 | 已学 |
| P-4 | Language Models are Unsupervised Multitask Learners（GPT-2） | 2019 | OpenAI（A-4） | A3 | §2、§3、§4、图 1、表 3 | 已学 |
| P-5 | Scaling Laws for Neural Language Models（Kaplan） | 2020 | OpenAI（A-5） | A4 | §1～§6、附录 lr 拟合 | 已学 |
| P-6 | Language Models are Few-Shot Learners（GPT-3） | 2020 | OpenAI（A-6） | A5 | §1、§2、§3.1/3.3/3.9、§4、§5、附录 B | 已学（检查题待答） |
| P-7 | GPT-4 Technical Report | 2023 | OpenAI（A-7） | A4 | §2 predictable scaling | 部分 |
| P-8 | Training Compute-Optimal Large Language Models（Chinchilla） | 2022 | 非 OpenAI，对照（R-2） | A4 | §3、§4 结论与拟合式 | 部分 |
| P-9 | An Empirical Model of Large-Batch Training（McCandlish） | 2018 | OpenAI（A-9） | A4、A5 | 梯度噪声尺度定义 | 仅引用 |
| P-10 | Generating Long Sequences with Sparse Transformers | 2019 | OpenAI | A5 | 无 | 仅提及 |
| P-11 | Tensor Programs V: µTransfer（µP） | 2022 | 非 OpenAI，对照（R-5，OpenAI 合著） | A5 | 摘要与 GPT-3 6.7B 实验 | 仅引用 |
| P-12 | In-context Learning and Induction Heads；A Mathematical Framework for Transformer Circuits | 2022；2021 | 非 OpenAI（Anthropic），对照 | A1、A5 | induction head 定义 | 仅引用 |
| P-13 | Rethinking the Role of Demonstrations（Min 等） | 2022 | 非 OpenAI，对照 | A5 | 标签打乱实验结论 | 仅引用 |
| P-14 | Lost in the Middle（Liu 等） | 2023 | 非 OpenAI，对照 | A5 | 主结论 | 仅引用 |
| P-15 | Scaling Laws for Autoregressive Generative Modeling（Henighan） | 2020 | OpenAI | A4 | 无（凭记忆） | 仅提及 |
| P-16 | Scaling Laws for Transfer（Hernandez） | 2021 | OpenAI | A4 | 无（凭记忆） | 仅提及 |
| P-17 | Deep Learning Scaling is Predictable, Empirically（Hestness，百度） | 2017 | 非 OpenAI，前史 | A3 | 无 | 仅提及 |
| P-18 | WebGPT | 2021 | OpenAI | A5 提及，D1 待学 | 无 | 未学 |
| P-19 | Direct Preference Optimization | 2023 | 非 OpenAI，对照（R-3） | B3 | 无 | 未学 |
| P-20 | Self-Consistency Improves Chain of Thought Reasoning | 2022 | 非 OpenAI，对照（R-4） | C3 | 无 | 未学 |
| P-21 | Training language models to follow instructions with human feedback（InstructGPT） | 2022 | OpenAI | A5 §11.8 提及，B3 待学 | 无 | 仅提及 |
| P-22 | Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank（SST，Socher 等） | 2013 | 非 OpenAI，数据集来源 | A2 评测、E3 任务 | 无 | 仅提及 |
| P-23 | Are Emergent Abilities of Large Language Models a Mirage?（Schaeffer 等） | 2023 | 非 OpenAI，对照（R-6） | A5 §11.12 | 主结论 | 仅引用 |
| P-24 | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks（MAML，Finn 等） | 2017 | 非 OpenAI，对照（R-8） | A5 §11.15 | §1、§2、Algorithm 1、式 (1)～(3)、图 1 | 仅引用 |

## 逐篇

### P-1 Attention Is All You Need（Vaswani 等，Google，2017）

- 来源级别：非 OpenAI，前置。只用来解释通用机制。
- 课程节点：A1（2026-09-15～16）
- 已读章节：§3.2 scaled dot-product attention 与 multi-head。未读：encoder 部分、位置编码细节、§5 训练设置。

| 知识点 | 引用了什么 |
|---|---|
| [[A1-transformer-decoder-next-token]] §3 | 注意力公式 softmax(QKᵀ/√d_k + M)V，课程加了 causal mask M |
| [[A1-transformer-decoder-next-token]] §7 | 注意力可替代循环结构并并行：机器翻译 BLEU 达 SOTA 且训练成本更低 |

### P-2 Learning to Generate Reviews and Discovering Sentiment（Radford 等，OpenAI，2017）

- 来源级别：OpenAI 官方（A-2），论文与代码、权重开源。
- 课程节点：A2（2026-09-17）
- 已读章节：主结果（字节级 mLSTM 隐状态中的单个情感单元、SST 线性探针结果）。未读：训练细节。

| 知识点 | 引用了什么 |
|---|---|
| [[A2-sentiment-neuron-gpt-1]] §2.1 | 无监督 LM 的隐状态里出现单个"情感神经元"，冻结模型加线性探针可读出情感 |
| [[A2-sentiment-neuron-gpt-1]] §7 | SST 上超过当时 SOTA |
| [[A2-sentiment-neuron-gpt-1]] Unknown callout | 论文没有解释为何集中到单个单元；"与 L1 正则有关"是课程推断 |
| [[A2-sentiment-neuron-gpt-1]] §10 | linear probe 作为与全参数微调的对照 |
| [[A2-sentiment-neuron-gpt-1]] §11.2 | "表征里已有特征，微调只是读出"是 GPT-1 少标注前提的证据 |

### P-3 Improving Language Understanding by Generative Pre-Training（Radford 等，OpenAI，2018）

- 来源级别：OpenAI 官方（A-3），代码与权重开源（finetune-transformer-lm）。
- 课程节点：A1（目标函数）、A2（流水线与消融）
- 已读章节：§3.1 LM 目标式 (1)、§3.2 微调目标与辅助损失、§3.3 task-aware input transformation、§5 分析（迁移层数、zero-shot 行为）、表 2～5。未读：§4 实验设置细节。

| 知识点 | 引用了什么 |
|---|---|
| [[A1-transformer-decoder-next-token]] §3 | 式 (1)：L₁(U) = −Σ log P(u_i ∣ u_{i−k}…u_{i−1}; Θ) |
| [[A1-transformer-decoder-next-token]] §7 | 表 2：12 个 NLU 任务 9 个 SOTA；图 2 左：迁移层数越多下游越好 |
| [[A2-sentiment-neuron-gpt-1]] §2.3 | 四种 input transformation 拼法与取 `<e>` 位置隐状态的规则 |
| [[A2-sentiment-neuron-gpt-1]] §3 | 微调目标 L₃ = L₂ + λ·L₁ |
| [[A2-sentiment-neuron-gpt-1]] §4 | 微调学习率约为预训练的 1/40 |
| [[A2-sentiment-neuron-gpt-1]] §7 | 表 5 消融：不预训练平均掉 14.8 个百分点；换 LSTM 掉 5.6；辅助 LM 损失大数据集有益、小数据集略有损害 |
| [[A3-gpt-2-zero-shot]] §2.3 | §5：用 "very" + positive/negative 的打分式读法做 zero-shot 情感分类 |
| [[A5-gpt-3-in-context-learning]] §2.3、§10 | 作为 fine-tune 路径的对照：全参数更新、新增 W_y |

### P-4 Language Models are Unsupervised Multitask Learners（Radford 等，OpenAI，2019）

- 来源级别：OpenAI 官方（A-4），论文、代码、四个尺寸权重开源。
- 课程节点：A3（2026-09-19）
- 已读章节：§1 动机、§2 数据与模型（WebText、尺寸表、pre-LN、残差初始化）、§3 各任务结果、§4 记忆与泛化（8-gram 重叠）。未读：§5 相关工作、§6 讨论。

| 知识点 | 引用了什么 |
|---|---|
| [[A3-gpt-2-zero-shot]] §2.1 | p(y ∣ x, task)：任务写进文本，同一 LM 不改参数 |
| [[A3-gpt-2-zero-shot]] §2.4 | WebText 约 800 万文档 40GB、karma ≥ 3 筛选；尺寸表 117M→1.5B；pre-LN；残差初始化乘 1/√N |
| [[A3-gpt-2-zero-shot]] §7 | 表 3：8 个 LM 数据集 7 个 SOTA，LAMBADA 63.2%，CoQA F1 55；图 1 四尺寸单调；§2 WebText 自然含翻译对；§4 8-gram 重叠率低；§3.6 去掉 TL;DR 掉约 6 分（待核对） |
| [[A3-gpt-2-zero-shot]] §11.4 | 图 1 是 scaling law 的雏形：四尺寸 log N 近似线性且未饱和、最大模型仍 underfit |
| [[A1-transformer-decoder-next-token]] §2 | E1 的 pre-LN block 写法照 GPT-2 |
| [[A4-scaling-laws]] §1 | 四个点无法外推，是 Kaplan 2020 的承接瓶颈 |
| [[A5-gpt-3-in-context-learning]] §4 | GPT-3 架构沿用 GPT-2（pre-LN、修改初始化、可逆分词） |

### P-5 Scaling Laws for Neural Language Models（Kaplan 等，OpenAI，2020）

- 来源级别：OpenAI 官方（A-5）。
- 课程节点：A4（2026-09-19～20）
- 已读章节：§1 结论摘要、§2 方法（非 embedding 参数、数据集）、§3 单变量幂律、§4 L(N, D) 与 L(N, S)、§5 临界 batch、§6 算力最优分配、附录学习率拟合。未读：§7 相关工作、附录各图细节。

| 知识点 | 引用了什么 |
|---|---|
| [[A4-scaling-laws]] §2 | 实验设计：一次只让一个量成瓶颈 |
| [[A4-scaling-laws]] §3 | L(N)、L(D)、L(C_min) 幂律与常数（α_N 0.076、α_D 0.095、α_C 0.050）；L(N, D)、L(N, S) 联合式；C ≈ 6ND |
| [[A4-scaling-laws]] §4、§11.6 | 临界 batch size B_crit(L) = B\*/L^(1/α_B)，B\* ≈ 2×10⁸，α_B ≈ 0.21 |
| [[A4-scaling-laws]] §6 | N ∝ C^0.73、D ∝ C^0.27；形状影响在几个百分点内；大模型样本效率高；B ∝ C^0.24、S ∝ C^0.03 |
| [[A4-scaling-laws]] §7 | 图 1 三条幂律跨七个数量级；图 2、图 4 样本效率 |
| [[A4-scaling-laws]] §8、§10 | 各尺寸共用学习率计划；≤1.5B；单数据集 |
| [[A4-scaling-laws]] §11.1 | L(C) 是 min over 6ND = C 的约束最优曲线 |
| [[A4-scaling-laws]] §11.2 | 图 6：按总参数画各深度分开，按非 embedding 参数画收敛到一条线 |
| [[A4-scaling-laws]] §11.14 | 6N 的来源：前向 2N、反向 4N |
| [[A5-gpt-3-in-context-learning]] §11.4 | 附录经验拟合 lr(N) ≈ 0.003239 − 0.0001395·ln N（凭记忆待核对）；B_crit 用于解释 GPT-3 的 batch 表 |
| [[A3-gpt-2-zero-shot]] §11.4 | Kaplan 引用 Hestness 2017 为前史 |

### P-6 Language Models are Few-Shot Learners（Brown 等，OpenAI，2020）

- 来源级别：OpenAI 官方（A-6）；权重未公开，github.com/openai/gpt-3 只有数据污染统计。
- 课程节点：A5（2026-09-28 讲解；检查题待答）
- 已读章节：§1 引言（图 1.1、图 1.2）、§2 方法（2.1 尺寸表 2.1、2.2 数据表 2.2、2.3 训练、2.4 评测协议）、§3.1 语言建模与 LAMBADA、§3.3 TriviaQA、§3.9 合成任务与新闻生成、§4 数据污染、§5 局限、附录 B 训练细节。未读：§3.2、§3.4～3.8 各基准细节、§6 broader impacts、附录 C 以后。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §2.1 | 图 1.1：外循环（跨文档 SGD）与内循环（上下文内识别模式）的 meta-learning 类比 |
| [[A5-gpt-3-in-context-learning]] §2.2、§11.1 | zero / one / few-shot 的定义，K 以 2048 token 为限（10～100），全部不更新参数，未做 fine-tune |
| [[A5-gpt-3-in-context-learning]] §3 | §2.4 打分式归一化：每 token 平均 log P；ARC/OpenBookQA/RACE 用 P(y∣c)/P(y∣"Answer: ")；beam 4、长度惩罚 0.6；表 2.2 五个数据源的 token 数、权重、epoch |
| [[A5-gpt-3-in-context-learning]] §4、§11.4 | 表 2.1 八个尺寸的层数、宽度、头数、batch、lr；注意力 dense / locally banded sparse 交替；3000 亿 token；175B 3.14×10²³ FLOPs；batch 从 3.2 万 token 线性升；cosine 衰减到 10% |
| [[A5-gpt-3-in-context-learning]] §5 | LAMBADA few-shot 的提升归因于示例修正格式；新闻生成人类判别约 52% |
| [[A5-gpt-3-in-context-learning]] §6 | 图 1.2 差距随 N 变宽；图 3.1 L(C) 延续到 175B；LAMBADA 76.2/72.5/86.4；TriviaQA 64.3/68.0/71.2；SuperGLUE 71.8；算术图 3.10（两位数加法 100、三位 80.4、四位 25.5、五位 9.3）；WiC 49.4 |
| [[A5-gpt-3-in-context-learning]] §7 | §4 去污染后分数变化小 |
| [[A5-gpt-3-in-context-learning]] §8 | §5 自列局限：ANLI、WiC、单向、目标对 token 一视同仁、样本效率、"学还是认"分不清 |
| [[A4-scaling-laws]] Inference callout、§7、§11.9 | 175B 只训 3000 亿 token 是 Kaplan 分配的产物；图 3.1 平滑曲线；按 Chinchilla 标准训练不足 |
| [[A3-gpt-2-zero-shot]] §2.3、§11.10、§11.11 | 多数基准用打分式读法；few-shot 一词在 GPT-3 定义为上下文示例；示例不留痕迹 |

### P-7 GPT-4 Technical Report（OpenAI，2023）

- 来源级别：OpenAI 官方（A-7）。
- 课程节点：A4（只用 predictable scaling 部分）
- 已读章节：§2 predictable scaling。未读：其余（留给 B、C、F 线）。

| 知识点 | 引用了什么 |
|---|---|
| [[A4-scaling-laws]] Public system behavior callout、§9 | 用 ≤ 1/10000 算力的小模型拟合 L(C) 预测 GPT-4 最终 loss；1/1000 算力预测 HumanEval 通过率；N、D、C 与数据点未公开 |
| [[A4-scaling-laws]] §3、§11.13 | 拟合式 L(C) = a·C^b + c，比 Kaplan 多一个不可约项 |
| [[A4-scaling-laws]] §8 | hindsight neglect：loss 平滑不等于能力平滑 |

### P-8 Training Compute-Optimal Large Language Models（Hoffmann 等，DeepMind，2022）

- 来源级别：非 OpenAI，对照（R-2）。只用于对比，不推断 OpenAI 内部。
- 课程节点：A4
- 已读章节：§3 三种方法的结论、§4 Chinchilla 70B vs Gopher 280B、拟合式 E + A/N^α + B/D^β。未读：各方法实验细节、下游评测。

| 知识点 | 引用了什么 |
|---|---|
| [[A4-scaling-laws]] §4、§11.19 | 对 Kaplan"各尺寸共用学习率计划"的批评；cosine 长度按 token 数调 |
| [[A4-scaling-laws]] §6、§10 | N ∝ C^0.5、D ∝ C^0.5，约每参数 20 token；拟合常数 E ≈ 1.69、α ≈ 0.34、β ≈ 0.28；70M～16B 400 多次训练 |
| [[A4-scaling-laws]] §7 | Chinchilla 70B 反超 Gopher 280B |
| [[A4-scaling-laws]] §11.2 | 计入 embedding 参数，常数不可与 Kaplan 直接比较 |
| [[A4-scaling-laws]] §11.9 | "每参数 20 token"的含义 |
| [[A5-gpt-3-in-context-learning]] 图板图 6 | 按 Chinchilla 标准 GPT-3 是"大模型、少数据" |

### P-9 An Empirical Model of Large-Batch Training（McCandlish 等，OpenAI，2018）

- 来源级别：OpenAI 官方（A-9）。
- 课程节点：A4 §11.6、A5 §11.4
- 已读章节：梯度噪声尺度的定义与"低于临界 batch 时 batch 翻倍 ≈ 步数减半"的结论。未系统读。

| 知识点 | 引用了什么 |
|---|---|
| [[A4-scaling-laws]] §11.6 | 梯度噪声尺度是临界 batch 的来源 |
| [[A5-gpt-3-in-context-learning]] §11.4 | batch 随尺寸递增的机制链：loss 低 → 噪声尺度大 → B_crit 大 |

### P-10 Generating Long Sequences with Sparse Transformers（Child 等，OpenAI，2019）

- 来源级别：OpenAI 官方。
- 课程节点：A5 §4 提及
- 已读章节：无。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §4 | GPT-3 的 locally banded sparse 注意力来自此文 |

### P-11 Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer（Yang 等，微软主导，OpenAI 合著，2022）

- 来源级别：非 OpenAI，对照（R-5）。前作 Tensor Programs IV（2020）提出 µP。
- 课程节点：A5 §11.4、§11.6
- 已读章节：摘要、GPT-3 6.7B 迁移实验结论。未读：推导。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.4 | µP 下 Adam 隐层最优 lr ∝ 1/宽度，是 GPT-3 时代"逐尺寸缩小 lr"的事后解释 |
| [[A5-gpt-3-in-context-learning]] §11.6 | 用约 4000 万参数代理模型调参迁移到 GPT-3 6.7B，超过原版，开销约一次预训练的 7% |

### P-12 In-context Learning and Induction Heads（Olsson 等，2022）与 A Mathematical Framework for Transformer Circuits（Elhage 等，2021），Anthropic

- 来源级别：非 OpenAI，对照。只解释通用机制，不推到 GPT-3 内部。
- 课程节点：A1 §11.13、A5 §11.2、§11.5
- 已读章节：induction head 的定义与"出现时刻与 ICL 能力上升重合"的结论。未读：电路分析细节。

| 知识点 | 引用了什么 |
|---|---|
| [[A1-transformer-decoder-next-token]] §11.13 | 用位置规则给 head 假设打分，induction 头看 j+1（x_j = x_t 的上次出现） |
| [[A5-gpt-3-in-context-learning]] §11.2、§11.5 | "找上次出现、抄后面的 token"的两头配合电路；与 ICL 出现时刻重合 |

### P-13 Rethinking the Role of Demonstrations（Min 等，2022）

- 来源级别：非 OpenAI，对照。
- 课程节点：A5 §11.2
- 已读章节：主结论。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.2 | 示例标签随机打乱，few-shot 准确率掉得很少；示例主要提供格式、标签空间与输入分布 |
| [[E3-icl-probe]] 可选扩展 | 标签打乱对照的实验设计来源 |

### P-14 Lost in the Middle（Liu 等，2023）

- 来源级别：非 OpenAI，对照。
- 课程节点：A5 §11.7
- 已读章节：主结论。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.7 | 长上下文中间位置的内容利用率低 |

### P-15、P-16 Scaling Laws for Autoregressive Generative Modeling（Henighan 等，2020）；Scaling Laws for Transfer（Hernandez 等，2021），OpenAI

- 来源级别：OpenAI 官方。
- 课程节点：A4 §9、§11.13
- 已读章节：无，凭记忆写入，待核对。

| 知识点 | 引用了什么 |
|---|---|
| [[A4-scaling-laws]] §9 | 幂律扩展到图像、视频、数学与迁移学习（待核对） |
| [[A4-scaling-laws]] §11.13 | 不可约项的讨论（待核对） |

### P-17 Deep Learning Scaling is Predictable, Empirically（Hestness 等，百度，2017）

- 来源级别：非 OpenAI，前史。
- 课程节点：A3 §11.4
- 已读章节：无。

| 知识点 | 引用了什么 |
|---|---|
| [[A3-gpt-2-zero-shot]] §11.4 | 翻译、语音上已报告幂律学习曲线，Kaplan 引用了它 |

### P-18 WebGPT: Browser-assisted question-answering with human feedback（Nakano 等，OpenAI，2021）

- 来源级别：OpenAI 官方。
- 课程节点：A5 §11.7 提及；D1 [[D1-codex-2021-webgpt]] 待学
- 已读章节：无。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.7 | GPT-3 微调后操作文本浏览器，引文进上下文后写带引用的答案；模仿学习 + RM/RL |

### P-21 Training language models to follow instructions with human feedback（Ouyang 等，OpenAI，2022）

- 来源级别：OpenAI 官方（B-4）。
- 课程节点：A5 §11.8 提及；B3 [[B3-instructgpt-chatgpt]] 待学
- 已读章节：无。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.8 | 按指令行事的能力来自 SFT + RLHF 后训练；目标是让 zero-shot 指令直接可用 |

### P-22 Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank（Socher 等，斯坦福，2013）

- 来源级别：非 OpenAI，数据集来源（SST / SST-2）。
- 课程节点：A2 评测数据、E3 探针任务
- 已读章节：无，只用数据集。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.10 | SST-2 的构成与规模；E3 选它的理由 |
| [[A2-sentiment-neuron-gpt-1]] §7 | Sentiment Neuron 与 GPT-1 在 SST 上的分数所用的数据集 |
| [[E3-icl-probe]] Environment | 示例池与查询集 |

### P-23 Are Emergent Abilities of Large Language Models a Mirage?（Schaeffer 等，斯坦福，2023）

- 来源级别：非 OpenAI，对照（R-6）。
- 课程节点：A5 §11.12
- 已读章节：主结论。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.12 | 精确匹配等阈值指标会把每 token 概率的平滑上升显示成陡变，一部分"涌现"是打分方式造成 |

### P-24 Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks（Finn、Abbeel、Levine，UC Berkeley / OpenAI 兼职，2017）

- 来源级别：非 OpenAI，对照（R-8）。作者之一当时兼任 OpenAI，但论文不是 OpenAI 官方发布，按非 OpenAI 处理。
- 课程节点：A5 §11.15（ELI5 讲解）
- 已读章节：§1、§2、Algorithm 1、式 (1)～(3)、图 1。

| 知识点 | 引用了什么 |
|---|---|
| [[A5-gpt-3-in-context-learning]] §11.15 | 图 1：一个初始化到多个任务最优点都只差几步；Algorithm 1 与式 (1)～(3)：内循环 support 集一步梯度、外循环 query 集损失对初始化求导（二阶）；first-order 近似；分类 / 回归 / RL 三类实验说明 model-agnostic |

### P-19 Direct Preference Optimization（Rafailov 等，2023）

- 来源级别：非 OpenAI，对照（R-3）。
- 课程节点：B3 待学。

### P-20 Self-Consistency Improves Chain of Thought Reasoning（Wang 等，2022）

- 来源级别：非 OpenAI，对照（R-4）。
- 课程节点：C3 待学。

← [[index]]
