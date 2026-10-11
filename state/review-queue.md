# Review Queue

| Knowledge Point | Reason | Priority | Review After | Required Test | Status |
|---|---|---:|---|---|---|
| A1 Transformer decoder | 09-28 复测：target 右移与 softmax 式子已纠正；多头拆分 shape 错（Q [2,4,8,16]、分数 [2,4,8,8]），⑥ 不知是 MLP；calibration 1 | 4 | 2026-10-01 | 给另一组 B/T/d/n_head 填多头拆分五步 shape；CE 算出数值 | 待复测 |
| A2 Sentiment Neuron → GPT-1 | 首次达到 mastery 2；学习率与梯度流混淆 | 4 | 2026-09-24 | 口头：四种 input transformation 拼法与取位规则；区分 linear probe 与 full fine-tune；E1 第二阶段数据支持 | 待复测 |
| A2 实验解读 | 09-27 复测仍 ≤1：单种子结论当成立、答不出预训练为何无效；mastery 3 → 2 | 5 | 2026-09-30 | 给新均值/标准差表，按"差值 vs 均值标准误"判断显著性（考虑样本是否独立）；解释表征—任务错配 | 待复测 |
| A3 GPT-2 | 09-27 复测两种读法对 → mastery 1；仍把 few-shot 当第三种读法 | 4 | 2026-09-30 | 填"条件 × 读法"2×2 表，并给 GPT-3 论文里的一个例子归格 | 待复测 |
| A4 Scaling Laws | 09-20 复测升 2；第 4 问（s1/s2 未收敛对拟合的影响、与固定 lr 计划批评的关系）不清楚；六次未写自评信心 | 4 | 2026-09-23 | 口头补答第 4 问，每条先写信心；再换一个 C 用 Kaplan 指数算 N、D；09-27 常规复测 L(N,D) 与 compute-optimal 推导 | 待复测 |
| A5 GPT-3 ICL | 10-11 检查题：Q3 条件 × 读法二维表 0 分（A3 同一短板第三次）；Q1 信息路径未说到注意力、KV cache 生命周期不清；第十次未写自评信心 | 5 | 2026-10-14 | 给四个新任务归入 2×2 表（与 A3 复测合并）；口头说示例 → 注意力 → 查询位置 → logits 的路径及 KV cache 何时丢弃；Q2 补三个实验设计条件；每题先写信心 | 待复测 |
