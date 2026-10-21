# 待办练习清单

学习者需要做、还没做的习题 / 代码 / 复测 / 练习。规则见 `AGENTS.md`「待办练习清单与及时反馈」：教练每布置一项立刻加一行，每次 session 开始先把未完成项列出来；完成后填「完成日期」，不删行。按「布置日期」递增，不重排。

| # | 布置日期 | 类型 | 节点 | 要做什么 | 在哪 | 完成日期 |
|---|---|---|---|---|---|---|
| T-1 | 2026-09-28 | 实验 | A5 | E3：云主机 `git pull` → 填 `icl_probe.py` 三处空 → `bash run_probe.sh` → `bash experiments/sync.sh "E3: ..."` | `experiments/in-context-learning/E3-icl-probe.md` | |
| T-2 | 2026-09-30 | 复测 | A2 / A3 | A2 实验解读（新均值表判断显著性）、A3 条件 × 读法 2×2 表 | `state/review-queue.md` | |
| T-3 | 2026-10-01 | 复测 | A1 | 多头拆分五步 shape（另一组 B/T/d/n_head）、softmax + CE 手算出数值 | `state/review-queue.md` | |
| T-4 | 2026-09-23 | 复测 | A4 | 补答第 4 问（s1/s2 未收敛与 Chinchilla 批评）、换 C 用 Kaplan 指数算 N、D，先写信心 | `state/review-queue.md` | |
| T-5 | 2026-10-14 | 复测 | A5 | 四个新任务归入 2×2 表；口头说示例 → 注意力 → 查询位置 → logits 路径与 KV cache 生命周期；Q2 补三个实验设计条件 | `state/review-queue.md` | |
| T-6 | 2026-10-11 | 因果表 | Track A | 第一档因果表：A1～A5 上一代 / 下一代 / 瓶颈 | `knowledge/A-pretraining-scaling/A-pretraining-scaling.md`「我的版本」 | |
| T-7 | 2026-10-15 | 复述 | B1 | 不看卡用 ≤150 字复述 B1 核心机制（三进程闭环为何在线、clip 的 min 在做什么） | 写进 B1 卡「我的复述」 | |
| T-8 | 2026-10-15 | 检查题 | B1 | Q0 衔接、Q1 在线 RM、Q2 clip 计算、Q3 2017→2019；每题第一行「信心：NN%」 | `sessions/2026-10-15-b1-ppo-human-preferences.md` Next Step | |
| T-9 | 2026-09-28 | 维护 | — | 手动删除根目录空文件 `gpt-3-in-context-learning.md`（与知识卡同名，Obsidian 链接会解析错） | 仓库根目录 | |
