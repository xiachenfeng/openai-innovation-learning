# Learning Session

- Date: 2026-09-18 ～ 2026-09-19
- Topic: E1 第二阶段结果 + A3 GPT-2
- Phase: Track A 第 2～3 周
- Duration: 分散多次，合计约 2 小时

## Learning Goals

1. 检查 sft.py 填空，解读 16 组对照；
2. A3：建立"任务写进文本 → 不改参数做任务"的机制，与 A2 的因果衔接。

## Source Freshness Check

沿用 2026-09-13。A3 引用 GPT-2 论文、代码、发布博客与 6 个月跟进，均为经典公开来源；论文数字凭记忆，卡片标待核对。

## Prerequisite Check

A2 mastery 2 → 3（E1 第二阶段完成）。

## Questions Asked

1. （09-19）E1：n=100 时训练损失趋零、测试低于多数类基线，用参数量/样本数/W_y 形状解释，预训练组为何没避免？
2. （09-19）A3：不用 W_y，从 LM head 读出分类结果的两种办法，哪种在 GPT-1 §5 出现过？

## My Answers

1. "不能解释"。
2. "一种是 few-shot，另一个不知道"。

## Evaluation

| 题 | correctness | mechanism | 备注 |
|---|---|---|---|
| E1 解读 | 0 | 0 | 诚实说不会 |
| A3 读法 | 1 | 1 | few-shot 是条件不是读法；第四次未给自评信心 |

sft.py 三处填空：correctness 4（含一次自我纠错）。

## New Knowledge

- 新建 canonical 卡 [[A3-gpt-2-zero-shot]]（十个 lens、三张图、两个追问）；
- [[A2-sentiment-neuron-gpt-1]] 新增 tip callout（"表征里已有特征，微调只是读出来"）、§11.1～11.3；
- [[A1-transformer-decoder-next-token]] §11.5（随机采样与重复）。

## Mistakes and Gaps

- 看实验数据反推机制的能力是短板（E1 解读 0 分）；
- 混淆"给不给例子"（条件）与"怎么读出分类"（机制）；
- 连续四题未给自评信心。

## 因果表（用户填写，主线结束或复测时）

Track A 结束时填。

## 图解产出

- 教练图：A3 三张（fine-tune vs prompt、任务写进文本、四尺寸 scaling）写入 [[A3-gpt-2-zero-shot]]，图板 https://claude.ai/artifact/GzWJbEhPea5v3HpWm5oLSS
- 用户因果表转图：无
- 复测填空：无

## Candidate Knowledge

无。

## Canonical Knowledge Changes

新增 canonical：gpt-2-zero-shot（论文 + 代码 + 权重公开）。

## Experiment Progress

E1 第二阶段完成：16 组结果与解读入记录（n=100 纯记忆；n=1000 大学习率下预训练组领先 0.1+，推翻教练"大学习率缩小优势"的预测；预训练是正则化；λ 方向与论文一致但单种子）。新增 run_seeds.sh 待用户跑。记录了哪些部分照论文、哪些是 toy 设计。

## Mastery Changes

A2：2 → 3（实验完成并有记录）。A3：0（检查题未过线）。

## Review Tasks

- 09-22：A2 实验解读复测（用 run_seeds 结果）、A3 两种读法识别题；
- 09-23：A1 block 结构填空。

## 工具与规则变更

- 云主机工作流：experiments/sync.sh（先 commit 再 rebase，AutoDL 开加速）、.gitignore 放行 outputs/*.csv；
- AGENTS.md：Session Start 先 pull；实验代码自动提交推送；代码只在云主机编辑；追问入卡；
- 仓库 git 身份改为 xiachenfeng。

## Files Updated

- knowledge/A-pretraining-scaling/A3-gpt-2-zero-shot.md（新建）、A2-sentiment-neuron-gpt-1.md、A1-transformer-decoder-next-token.md、A-pretraining-scaling.md
- state/answer-history.csv（+4 行）、mastery.md、review-queue.md、current.md
- experiments/pretraining/E1-tiny-gpt.md、tiny_gpt/run_seeds.sh、experiments/sync.sh、.gitignore（已推送）
- AGENTS.md

## Next Step

1. 云主机 `bash experiments/pretraining/tiny_gpt/run_seeds.sh` 后 sync；
2. 09-22 复测；
3. A4 Scaling Laws + E2（复用 E1 代码训 3～4 个尺寸拟合幂律）。
