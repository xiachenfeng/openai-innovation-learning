# Mistakes and Misconceptions

当前暂无记录。

每条记录使用：

## YYYY-MM-DD — Knowledge Point

- Question:
- My answer:
- What was correct:
- Main mistake:
- Correct understanding:
- Root cause:
- Required retest:
- Retest status:

## 2026-09-16 — A1 Transformer decoder / next-token

- Question: 4 token 手推作业 Q1（target 序列）与 Q4（softmax 与交叉熵）
- My answer: target = 4 2 3 0；softmax([2,0,0,2,0]) = [0.5, 0, 0, 0.5, 0]，CE = 0.3466
- What was correct: input 序列、causal mask 形状、Q3 的 softmax 计算、Q5 的"每个位置都有 next-token"
- Main mistake: (1) target 不是重排，是 input 右移一位；(2) softmax 里 e^0 = 1 不是 0，五项都有概率
- Correct understanding: target = [4, 1, 3, 0]；softmax = [0.416, 0.056, 0.056, 0.416, 0.056]，CE = −ln 0.416 = 0.878
- Root cause: 把 softmax 当成了"只保留最大值"的 hardmax；target 构造没有对照 input 逐位检查
- Required retest: 换一组 logits 重算 softmax + CE；写出 input/target 对齐
- Retest status: 2026-09-16 重做通过（logits [1,0,3,0,0] → softmax 与 CE 全对）

## 2026-09-17 — A2 GPT-1 微调

- Question: 微调为什么把学习率降到预训练的 1/40？不降会怎样？
- My answer: 不想对底层参数影响太大，只想把梯度传到末层参数（信心 50）
- What was correct: 小学习率是为了保护预训练参数
- Main mistake: 把"学习率小"理解成"梯度只到末层"
- Correct understanding: 反向传播的梯度流经全部层，GPT-1 更新全部参数；学习率只缩放所有参数的步长。W_y 随机初始化，早期传回 Θ 的梯度是噪声，大学习率会用噪声覆盖预训练特征，预训练组退化到接近从零组。只更新末层是另一种方法（linear probe，冻结 Θ）
- Root cause: 混淆了"梯度是否流到"与"参数改动有多大"两个概念
- Required retest: E1 第二阶段用 lr 3e-4 vs 6e-5 跑预训练组，用数据验证；复测时口头区分 linear probe 与 full fine-tune
- Retest status: 待复测

## 2026-09-20 — A4 Scaling Laws（E2 预测题）

- Question: 数据固定、四个尺寸训到 val 不再下降，双对数图上是什么形状？最大模型 val 反而更高时用哪条公式解释、先怀疑什么？
- My answer: 四个点是一条向下的直线；用 L(N, D) 解释，因为它不是单调的
- What was correct: 选对了 L(N, D)；前三个点近似直线的判断对
- Main mistake: (1) 漏掉数据固定时第四点走平，这是本节图 3 与 §6 的核心；(2) 把 L(N, D) 说成非单调
- Correct understanding: L(N, D) 对 N 单调递减，数据固定时 D_c/D 项占主导，曲线走平而不是上翘；早停下的最低 val 若反而升高，原因在公式之外：所有尺寸同一学习率、max_steps 上限、早停耐心过短、val 只用 20 个 batch 估计的噪声，其次才是大模型在达到小模型 loss 之前就开始记训练集
- Root cause: 把"过拟合会让 loss 变差"直接套到 L(N, D) 上，没区分"公式描述的是早停后的最低 loss"与"训练过程中 loss 回升"
- Required retest: 09-23 用 E2 真实结果口头解读
- Retest status: 待复测

## 2026-09-20 — A4 E2 解读第 4 问

- Question: s1、s2 在步数上限处仍未收敛，会让哪些判断偏向哪个方向？和 Chinchilla 对 Kaplan 的批评有什么关系？
- My answer: 不清楚
- What was correct: 其余四问都答对了方向
- Main mistake: 没有把"训练配方对所有尺寸相同"和"拟合结果被扭曲"连起来
- Correct understanding: s1、s2 的 best_val 偏高 → 前三点的弯曲一部分是假象，前三点的 alpha 0.074 与外推预测都不可信；若小模型训到收敛，前三点会更接近直线，第四点的偏离会更明显。所有尺寸共用 lr 3e-4 和同一步数上限，正是 Kaplan 被 Chinchilla 批评的"固定学习率计划"在 toy 尺度上的重演；Chinchilla 的做法是按每个尺寸的 token 预算单独调计划
- Root cause: 把实验设置当成中性背景，没有问"哪些点的数值本身不可信"
- Required retest: 09-23 口头补答；可选跑 s1/s2 换 lr 1e-3 的第三轮，用数据验证
- Retest status: 待复测

## 2026-09-27 — A2 实验解读（种子复验）

- Question: run_seeds 7 样本均值表，判断 init × lr 差异是否超过噪声
- My answer: 单种子"预训练领先 0.1"仍成立；小 lr 可以；预训练为何无效不清楚；同 seed 两次不同是未收敛
- What was correct: 注意到差值与标准差同量级
- Main mistake: 用单次结果下结论；把代码非确定性当成收敛问题
- Correct understanding: 差值要和均值差的标准误比（σ·√(2/n)），重复同 seed 不是独立样本；预训练有效的前提是表征含任务相关特征，字符级小模型不含说话人特征
- Root cause: 缺少"噪声底线"这一统计习惯；GPT-1 前提只记了结论没记条件
- Required retest: 2026-09-30
- Retest status: 待复测

## 2026-09-27 — A3 条件与读法

- Question: few-shot 为什么不是第三种读法
- My answer: few-shot 本质也是输出对应词的 logits
- Main mistake: 把输入端（条件）与输出端（读法）混为一个维度
- Correct understanding: 条件（zero/few-shot）× 读法（打分式/生成式）是正交的 2×2
- Required retest: 2026-09-30
- Retest status: 待复测

## 2026-09-28 — A1 多头注意力 shape

- Question: 每头 Q 与注意力分数矩阵的 shape
- My answer: [2,8,64]、[2,8,8]
- Correct understanding: 拆头后 [B,H,T,d/H] = [2,4,8,16]，分数 [B,H,T,T] = [2,4,8,8]；block 第二个子层是 MLP；targets 是 [B,T] 整数索引
- Root cause: 把多头当单头理解
- Required retest: 2026-10-01
- Retest status: 待复测

## 2026-10-11 — A5 GPT-3 ICL（检查题 1）

- Question: few-shot 为何叫 learning 却不更新参数；任务信息从哪条路径进入模型，前向结束后去了哪里
- My answer: 从 context 学回答方式、不更新参数；任务信息从 token 进入模型；前向结束后进入 KV cache？
- What was correct: 学的是上下文里的模式而非权重；KV cache 确实是示例信息在生成期间的存放处
- Main mistake: 路径只说到"token 进入"，没说示例位置的 K、V 被查询位置的注意力读取；把 KV cache 当成最终去向
- Correct understanding: 示例 token → 每层算出 K、V → 查询位置用 Q 读取 → 改写查询位置的残差流 → logits。KV cache 只是把这些 K、V 在本次生成期间缓存起来，回答生成完即丢弃，不进权重；下一个查询要重新给示例
- Root cause: 把"推理时的缓存"与"学到的东西存在哪"混在一起
- Required retest: 2026-10-14 口头复述路径与 KV cache 生命周期
- Retest status: 待复测

## 2026-10-11 — A5 / A3 条件 × 读法（检查题 3，第三次）

- Question: 把 E3 的三个设置放进"条件 × 读法"二维表，补第四格并举 GPT-3 论文的例子
- My answer: 不懂
- What was correct: 无
- Main mistake: 两个维度没有分开。09-19 把 few-shot 当读法，09-27 仍然，10-11 完全答不出
- Correct understanding: 条件 = 上下文里给不给示例（zero / one / few-shot），读法 = 答案怎么从 LM head 读出（打分式比较候选概率 / 生成式续写）。E3：打分 × 8-shot、打分 × zero-shot、生成 × 4-shot（加法）；第四格生成 × zero-shot = GPT-3 的 zero-shot TriviaQA（64.3，自由生成答案）或 zero-shot 翻译
- Root cause: "few-shot"在日常语境里连带着"让模型自己写答案"，输入端与输出端被绑在一起
- Required retest: 2026-10-14 给四个新任务归格
- Retest status: 待复测
