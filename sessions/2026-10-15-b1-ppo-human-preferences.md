# Learning Session

- Date: 2026-10-15
- Topic: B1 PPO 与 Learning from Human Preferences（2017）讲解
- Phase: Track B 第 1 周（用户决定在 A5 复测与 E3 之前先开 Track B）
- Duration: 约 40 分钟（讲解部分；检查题待答）

## Learning Goals

1. 建立 2017 年两篇论文各自承接的瓶颈：reward 写不出来；策略优化器难用；
2. 说清偏好学习三进程闭环为何必须在线，以及 Bradley–Terry 损失的形式；
3. 说清 PPO clip 的四种情况与 min 的作用；
4. 预埋 B2：PPO 的 KL 与 RLHF 的 KL 对象不同。

## Source Freshness Check

2026-10-15 打开 arXiv 核对：1706.03741（HTML v4，公式 (1)、损失、三处修正、700 / 5,500 / 900 / 1,300 等数字全部核对）、1707.06347（abs 页 200，HTML 版不可用，PDF 未能抽取文本；公式与超参凭记忆，表 1 小数值标待核对）、1909.08593（abs 200）、1502.05477（abs 200）、github.com/openai/baselines（200）。OpenAI 博客 B-1、B-2 仍 403，未打开。

## Prerequisite Check

B1 mastery 0（09-13 诊断：不知 PPO、RM、RLHF）。A5 mastery 1，E3 未跑，10-14 复测块未做；用户明确选择先进 Track B，A5 相关事项并行保留在 review queue。

## 主线衔接（Track A → Track B）

- 上一条线终点留下的问题：GPT-3 的目标是模仿语料不是用户意图，输出靠 prompt 技巧、会编造与有害；放大 N 解决不了，要换目标函数（即本线主线问题）。
- 对象对应表写入的 MOC：[[B-alignment]]「与前序主线的衔接」，七行（A1 目标 → 策略梯度；LM head → 动作分布；A2 加头微调 → RM；A2 小 lr → KL；prompt/ICL → 改权重；scaling → 对齐与规模两条轴；few-shot 分数 → 偏好胜率）。
- 顺序说明：B1 的 2017 早于 A5 的 2020，两篇论文在机器人与游戏上；用到 LM 要等值得对齐的预训练模型（2019 GPT-2）与 A2 的加头微调配方。课程按因果排。
- 用户 2026-10-15 要求补这段并抽象成规则，规则已写入 AGENTS.md「主线启动：先讲衔接，再讲第一节」，模板 `templates/session.md` 加了「主线衔接」段。

## Questions Asked

检查题四道（Q0 衔接 + Q1～Q3，见 Next Step），待答。
追问：11.1 PPO 用到 LM 是否"从机器人跨到 LLM"；11.2 next-token 逐字 vs PPO 轨迹级偏好；11.3（另一 session，/eli5 PPO 绘本之后）"比平时高分就多做"的"平时"怎么定义 → 基线 V(s_t) 与 GAE；11.4 PPO 的网络图与 value head → 两种接法（共享躯干 / 独立价值模型），图追加到基线图板。 10-16 追问 11.5 LLM 里 PPO 的整体流程（InstructGPT 三步 + 一轮循环），两张图再追加到基线图板。 追问 11.6 逐 token 的 RM / KL / value 分数怎么合成总 loss（图 6 + T=4 数值例子，追加到基线图板）。 追问 11.7 PPO 是否在 OpenAI 引用前十（Semantic Scholar API，2026-10-16，第 3）。 追问 11.8 rollout 与 update 的交替、概率比 ρ_t 的本质（图 7、图 8 追加到基线图板）。 10-21 追问：LM head 具体是什么组件 → 写入 A1 §11.14，B1 §11.13 链接；同日追问"全模型参数分几块"，参数分区图（GPT-2 small 账本）写入 A1 §11.14，并按用户要求单独发布为图解页 https://claude.ai/artifact/GmGabuBSj1Aca5xJhwQgNy（V-27），已从本节图板移除；再追问"I have a → cat 的推理过程"，写入 A1 §11.15 并发布图解页 https://claude.ai/artifact/7c9XT52iZQhCDfJiuQsVg3（V-28）。

## My Answers

待答。

## Evaluation

待评。

## New Knowledge

- 新建 canonical 卡 [[B1-ppo-human-preferences-2017]]：十个 lens、三张 mermaid 图、两组公式与符号表、clip 四情况表、三张对照表（reward 来源、优化器、两个 KL）；
- MOC [[B-alignment]]：B1 状态 canonical，mastery 0。

## Mistakes and Gaps

待检查题。

## 因果表（用户填写，主线结束或复测时）

| 上一代 | 下一代 | 上一代的瓶颈 |
|---|---|---|
| | | |

## 图解产出

- 教练画的教学图：9 张（含衔接图写入 MOC、追问 11.2 对照图入卡），图板 https://claude.ai/artifact/U8pfXG1BpHr83eC6BJyTbg ；追问 11.3 另出 PPO 基线图板 https://claude.ai/artifact/WLVGDQAYB6fWJuiCTsaHNx ；同日 /eli5 PPO 绘本两页（英 https://claude.ai/artifact/W62d69PXW35bCnih2LQWAN 、中 https://claude.ai/artifact/WvB6r5iggPxUCXeBm4RqbS ），全部登记在 [[artifacts]]；其中 3 张（三进程闭环、PPO 训练循环、主线因果位置）写入 [[B1-ppo-human-preferences-2017]]，clip 情况表与两个 KL 对照表以表格入卡，时序图只留图板
- 用户因果表转成的图：无
- 复测填空结果与参考版差异：无

## Candidate Knowledge

无。

## Canonical Knowledge Changes

新增 canonical：B1-ppo-human-preferences-2017（两篇 OpenAI 官方论文，PPO 代码开源；博客页面未打开的部分已标待核对）。

## 论文台账更新

新增 P-25 Deep RL from Human Preferences（已学）、P-26 PPO（已学）、P-27 TRPO（非 OpenAI 前置，仅提及）、P-28 Fine-Tuning LMs from Human Preferences（Ziegler 2019，仅提及，B2 待学；追问 11.1 加引用处）、P-29 MIXER（非 OpenAI 前史，仅提及）。追问 11.3 新增 P-30 GAE（非 OpenAI 前置，仅引用，R-11），P-26 补 §5 式 (10)～(12) 引用处，P-21 InstructGPT 补附录 C.4 value 初始化（待核对）。`sources/official-sources.md` 新增 B-16～B-19、R-9、R-10。

## Experiment Progress

本节不挂实验。E3 仍待用户在云主机跑；E4 在 B2 启动。

## Mastery Changes

无（检查题待答）。

## Review Tasks

沿用 review-queue：A5 / A3 / A2 / A4 / A1 复测块仍过期待做，与 Track B 并行。

## 待办更新（state/todo.md）

2026-10-21 建清单：T-1 E3、T-2～T-5 过期复测、T-6 Track A 因果表、T-7 B1 复述、T-8 B1 四道检查题、T-9 删根目录空文件；全部未完成。规则写入 AGENTS.md「待办练习清单与及时反馈」。

## Files Updated

- knowledge/B-alignment/B1-ppo-human-preferences-2017.md（新建）
- knowledge/B-alignment/B-alignment.md（B1 行、「与前序主线的衔接」段与衔接图）
- AGENTS.md（新规则「主线启动」）、templates/session.md
- knowledge/papers.md
- sources/official-sources.md
- state/current.md
- sessions/2026-10-15-b1-ppo-human-preferences.md（本文件）

## Next Step

1. 用户答四道检查题（每题先写信心 0～100）：
   - Q0（因果 / 衔接）：A2 的 fine-tune 配方和 B 线的 reward model 在结构上哪里相同、训练信号哪里不同？A2 的小学习率和 B2 的 KL 惩罚在目的上有什么共同点？
   - Q1（机制）：偏好论文为什么要让 reward model 的训练数据不断从**当前策略**的轨迹里抽，而不是一开始收一批比较然后固定？如果固定会发生什么？
   - Q2（机制 + 推导）：ε = 0.2，某动作 Â = −1，当前 r_t = 0.7。写出 PPO 目标项的值和梯度是否为零；再把 r_t 换成 1.3 做一遍。为什么两种情况不对称？
   - Q3（因果 + 迁移）：2017 年偏好论文用的是 TRPO 和 A2C 而不是 PPO。请说明原因，并说出 2019 年合体时除了换 PPO 还加了什么、为什么要加。
2. 检查题评完后再决定 B2 时间；云主机 E3 与 10-14 复测块并行。
