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
