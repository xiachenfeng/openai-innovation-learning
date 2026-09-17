# Experiments

实验是各 Track 的完成标准，不是独立阶段。前后复用同一代码和模型。

| 编号 | 目录 | 挂在 | 复用 | 必做 | 完成标准 |
|---|---|---|---|---|---|
| E1 | pretraining/ | A1–A2 | 起点 | 是 | tiny GPT 预训练后 SFT 的下游指标优于从零训练 |
| E2 | scaling-laws/ | A4 | E1 代码 | 是 | 3～4 个尺寸拟合出幂律指数，并与 Kaplan/Chinchilla 讨论 |
| E3 | in-context-learning/ | A5 | E1 模型或 GPT-2 small | 是 | zero/one/few-shot 曲线 |
| E4 | rlhf/ | B2–B3 | 小型 pretrained LM | 是 | RM 准确率 + 一轮策略优化的 reward/KL 曲线 |
| E5 | process-supervision/ + reasoning-scaling/ | C1, C3 | 同一题集 | 是 | ORM vs PRM 重排 + pass@k budget 曲线 |
| E6 | agent-loop/ | D3 | 起点 | 是 | 带 compaction 的 loop 在长任务上的 token 与成功率 |
| E7 | router/ + multi-agent/ | C5, D5 | E6 | 可选 | 成本/正确率/时延三项对比 |
| E8 | safety-self-play/ | B5 | E4 | 可选 | 无害文本分类环境下的 attacker/defender 曲线 |
| E9 | gpt-oss/ | D4 | 无 | 可选 | 架构参数表与一层结构图 |

每个实验用 `templates/experiment.md` 记录，写明算力与 API 花费。
