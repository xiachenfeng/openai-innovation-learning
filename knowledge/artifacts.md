---
title: 图板与图解页台账
level: 0
updated: 2026-10-21
tags: [openai, hub, artifacts]
---

# 图板与图解页台账

课程里每次讲解发布的 Artifact 页面（本节图板、追问图板、ELI5 图解页）都登记在这里，按 V-编号递增，不重排。每张概念卡的「图板与图解页」一节只列自己的，全表以本文件为准。

**打开方式。** 链接是 claude.ai 的私有 Artifact，只有本账号登录后能打开（Obsidian 里点链接会跳到默认浏览器）。要分享给别人需在页面的 Share 菜单里开权限。页面源码不在仓库里，仓库只存链接；mermaid 图的源码在知识卡里另有一份，由 Obsidian 渲染。

**类型。** 本节图板 = 讲一节课时教练画的全部教学图；追问图板 = 回答某个追问时的图；ELI5 图解 = `/eli5` 生成的少字大图绘本，只做直觉，不替代卡片的机制描述。

## 总表

| 编号 | 页面 | 类型 | 节点 | 关联卡与章节 | 日期 | 内容 |
|---|---|---|---|---|---|---|
| V-1 | [困惑度是几面骰子](https://claude.ai/artifact/KYwj7isSTGng4vsLbwBdYz) | ELI5 图解 | A1 | [[A1-transformer-decoder-next-token]] §11.6 | 2026-09-28 | 可拖动概率的骰子演示，E1 / GPT-2 词表尺子 |
| V-2 | [一个 nat 有多惊讶](https://claude.ai/artifact/Psi2yZUrFa4kVJccehLfhh) | ELI5 图解 | A1 | [[A1-transformer-decoder-next-token]] §11.6、§11.9 | 2026-09-24 | nat / bit / 困惑度三种刻度 |
| V-3 | [一个 token 碰到多少参数](https://claude.ai/artifact/RvHqmEJq4pzTaAbv55BnJ2) | ELI5 图解 | A1 | [[A1-transformer-decoder-next-token]] §11.7 | 2026-09-28 | GPT-2 small 按 wte / wpe / 12 层记账 |
| V-4 | [Softmax 数几次](https://claude.ai/artifact/99E9zJgrQDsNnMLvgKyoDN) | ELI5 图解（交互） | A1 | [[A1-transformer-decoder-next-token]] §11.8 | 2026-09-28 | H × T 次；T、H、D 滑块记账 |
| V-5 | [注意力头是什么](https://claude.ai/artifact/Hf8HWThP6TwHa6dDF9g5nu) | ELI5 图解 | A1 | [[A1-transformer-decoder-next-token]] §11.10 | 2026-10-11 | head 与 Q/K/V 的关系 |
| V-6 | [A2 GPT-1 图板](https://claude.ai/artifact/JA8AYXzSxbitKPRk1qqtXa) | 本节图板 | A2 | [[A2-sentiment-neuron-gpt-1]] 全卡 | 2026-09-17 | 探针冻结图、两阶段参数共享图、训练循环图 |
| V-7 | [A3 GPT-2 图板](https://claude.ai/artifact/GzWJbEhPea5v3HpWm5oLSS) | 本节图板 | A3 | [[A3-gpt-2-zero-shot]] 全卡 | 2026-09-19 | fine-tune vs prompt、任务写进文本、四尺寸 scaling |
| V-8 | [字节级 BPE 分词器](https://claude.ai/artifact/2ouTn5rzu1mgUTJdXA7tz6) | ELI5 图解 | A3 | [[A3-gpt-2-zero-shot]] §11.5～11.9 | 2026-09-23 | BPE 合并过程与 Huffman 对照 |
| V-9 | [Zero-shot 与 Few-shot](https://claude.ai/artifact/57G8vt5QYDAqwCTxvkRB6B) | ELI5 图解 | A3 | [[A3-gpt-2-zero-shot]] §11.10～11.11（A5 §11.1 对照） | 2026-09-28 | GPT-1 培训班 / GPT-2 zero-shot / GPT-3 few-shot 三种教法 |
| V-10 | [A4 Scaling Laws 图板](https://claude.ai/artifact/E4C59fa4YoEvaWBgcTohCJ) | 本节图板（交互） | A4 | [[A4-scaling-laws]] 全卡；图 4 滑块对应 §11.7、§11.9 | 2026-09-19 | 拖 C 读出 Kaplan 与 Chinchilla 两套分配下的 N、D |
| V-11 | [Scaling Laws 图画书](https://claude.ai/artifact/2g6By9u5K8WFcvXet23Ek9) | ELI5 图解 | A4 | [[A4-scaling-laws]] 全卡 | 2026-09-23 | 幂律的直觉版 |
| V-12 | [对数尺度图画书](https://claude.ai/artifact/D1D9TsrBmwka6mbDXHS85k) | ELI5 图解 | A4 | [[A4-scaling-laws]] §11.12 | 2026-09-23 | 为什么用双对数坐标 |
| V-13 | [Loss 的地板](https://claude.ai/artifact/YEqy5DigiFqQ2VdCw5SyHG) | ELI5 图解 | A4 | [[A4-scaling-laws]] §11.13 | 2026-09-23 | 不可约项 E |
| V-14 | [临界 Batch 图画书](https://claude.ai/artifact/2oP8DVbtx17BFt83nyYbE8) | ELI5 图解 | A4 | [[A4-scaling-laws]] §11.6 | 2026-09-23 | 问路人的比喻，S/S_min 与 E/E_min |
| V-15 | [6ND 算力账本](https://claude.ai/artifact/AtC9n7P2nMaBuEgm2a6NwL) | ELI5 图解 | A4 | [[A4-scaling-laws]] §11.14 | 2026-09-24 | 正向 2N、反向 4N |
| V-16 | [KKT 小球与墙](https://claude.ai/artifact/GgtwRJP9EVyPFdzigvLgr9) | ELI5 图解（交互） | A4 | [[A4-scaling-laws]] §11.5 推导（拉格朗日乘子）的数学前置，推测 | 2026-09-23 | 约束优化四条件；卡内正文未引用 |
| V-17 | [Robbins–Monro 步长](https://claude.ai/artifact/2mPAdELufA9km8DoNWSMk5) | ELI5 图解 | A4 | [[A4-scaling-laws]] §11.20 | 2026-09-26 | 步长必须趋于 0 的两个条件 |
| V-18 | [GPT-3 In-context Learning 图板](https://claude.ai/artifact/Y83YHgaWwLJeEfaCwShnuX) | 本节图板 | A5 | [[A5-gpt-3-in-context-learning]] 全卡（7 张，4 张入卡） | 2026-09-28 | meta-learning 框架、few-shot 一次前向、fine-tune vs in-context、主线因果位置、E3 设计 |
| V-19 | [GPT 一二三代](https://claude.ai/artifact/BbXMezmJyrEVJuUiyuXUmG) | ELI5 图解 | A5 | [[A5-gpt-3-in-context-learning]] §11.9 | 2026-10-11 | 脑子大小、数据量、三种用法、参数表 |
| V-20 | [Induction Heads 图画书](https://claude.ai/artifact/Upa1wVEfQ4LBn2ahwG3aNh) | ELI5 图解 | A5 | [[A5-gpt-3-in-context-learning]] §11.5 | 2026-10-11 | 两个 head 配合完成「抄前文」的电路 |
| V-21 | [MAML 小画书](https://claude.ai/artifact/XNdkqWJ1ezvEF5oFaBiH66) | ELI5 图解 | A5 | [[A5-gpt-3-in-context-learning]] §11.15 | 2026-10-13 | 显式梯度式 meta-learning 与 ICL 对照（非 OpenAI） |
| V-22 | [PPO 与 Human Preferences 图板](https://claude.ai/artifact/U8pfXG1BpHr83eC6BJyTbg) | 本节图板 | B1 | [[B1-ppo-human-preferences-2017]] 全卡（9 张，3 张入卡；衔接图入 MOC；参数分区图另见 V-27） | 2026-10-15 | 三进程闭环、PPO 训练循环、clip 四种情况、两个 KL、时序图 |
| V-23 | [PPO in Pictures](https://claude.ai/artifact/W62d69PXW35bCnih2LQWAN) | ELI5 图解（英文） | B1 | [[B1-ppo-human-preferences-2017]] §2.2 | 2026-10-15 | 小狗学把戏六张图：treat、advantage、cliff、fence、loop、RLHF |
| V-24 | [图解 PPO](https://claude.ai/artifact/WvB6r5iggPxUCXeBm4RqbS) | ELI5 图解（中文） | B1 | [[B1-ppo-human-preferences-2017]] §2.2 | 2026-10-15 | 上一页的中文版，术语括注英文 |
| V-25 | [PPO 基线图板](https://claude.ai/artifact/WLVGDQAYB6fWJuiCTsaHNx) | 追问图板 | B1 | [[B1-ppo-human-preferences-2017]] §11.3～§11.8 | 2026-10-15～16 | 基线 V(s) 与 GAE 的数据流，λ 两端对照表；PPO 论文与 RLHF 两种网络结构（value head）与 head 形状表；InstructGPT 三步流程与一轮 PPO 循环；逐 token 量合成总 loss 的图与 T=4 数值例子；rollout / update 交替与概率比 ρ_t 的重要性采样本质 |
| V-26 | [PPO × Next-token 训练图板](https://claude.ai/artifact/KcFkoErwZppVLoMdy22Ho3) | 追问图板（交互） | B1 | [[B1-ppo-human-preferences-2017]] §11.9～§11.12 | 2026-10-16～17 | InstructGPT 第 3 步的实体与数据流（少公式版）：三步定位、4 模型 + 3 种数据 + 旧策略快照的实体卡、同一词表头的 SFT / PPO 两种来源、rollout → 账本 → update 的数据流转大图、「一个 token 的一生」时序图、账本每列的生产者与消费者血缘图、新旧概率对比为何必要、四模型泳道、六节点压缩图、公开边界四格；06b 价值模型两条路、各位置目标与估值、带 γ 与整条分双滑块的四行账交互图（对数比 → 偏离罚 → r_t → G_t 递推表，数值自洽）、γ 的物理意义与 γ/λ 分工、整条分的归一化与分差单位、每步奖励 r_t 的来路与正负号来源、末尾信号向前衰减 |
| V-27 | [Decoder 参数分区图](https://claude.ai/artifact/GmGabuBSj1Aca5xJhwQgNy) | 追问图板 | A1 | [[A1-transformer-decoder-next-token]] §11.14 | 2026-10-21 | GPT-2 small 的 decoder 数据流图：每个组件一个节点，只标组件名 + 参数量（含无参数的 mask / softmax / GELU / 残差），合计 124.4M |
| V-28 | [Next-token 推理一步](https://claude.ai/artifact/7c9XT52iZQhCDfJiuQsVg3) | 追问图板 | A1 | [[A1-transformer-decoder-next-token]] §11.15 | 2026-10-21 | "I have a" 三个 token 经 GPT-2 small 各层到 " cat" 的张量轨迹（第 10 版 = 回滚到第 5 版：参数方框在张量旁边、标签用 `<br/>`，线上会把换行吃掉但用户接受；单列最简版已于 2026-10-21 删除），圆柱 = 数据张量、方框 = 参数、边标 = 无参数计算；3×3 因果 mask；三个位置各预测什么；最后一步 h₂ @ wteᵀ 与 KV cache；LM head 与 ln_f 的区别表 |

## 登记规则

1. 发布一个新页面后立刻在总表追加一行，并在关联卡的「图板与图解页」一节加同一条链接（写明对应 §）；
2. 同一 session 内后续的图追加到同一个图板重新发布，不新建，台账不加行；
3. 页面与课程无关的（别的学习项目、生活用途）不登记；
4. 关联章节写「推测」的，表示卡片正文没有引用该页面，待讲解时核对。

← [[index]]
