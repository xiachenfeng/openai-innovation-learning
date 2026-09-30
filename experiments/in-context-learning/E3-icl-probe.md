# Experiment

- Name: E3 in-context learning 探针（GPT-2 四个开源尺寸，SST-2 zero / one / few-shot）
- Date: 2026-09-28 启动
- Related concept: [[gpt-3-in-context-learning]]（A5）；读法承接 [[gpt-2-zero-shot]] §2.3
- Status: 骨架已推送，待用户填三处空并在 4090 上跑

## Question

1. 参数冻结、只在上下文里放 K 条示例，GPT-2 四个尺寸（124M / 355M / 774M / 1.5B）在 SST-2 上的准确率随 K 怎么变？
2. few-shot 与 zero-shot 的差距是否随模型尺寸变宽（GPT-3 论文图 1.2 的形状）？
3. 换一组随机示例带来的波动有多大？K 的效应是否超过这个噪声？

## Hypothesis

- zero-shot 四个尺寸都接近多数类基线（约 0.5），因为模型不知道 "Sentiment:" 后该接什么词，且 GPT-2 对 " positive" 有先验偏向；
- K = 4、8 时 gpt2-xl 明显高于 zero-shot（差距 0.1 以上），gpt2 小模型差距接近 0；差距随尺寸单调变宽；
- 跨 seed 的标准差在 0.03～0.10 之间，小模型上与差距同量级，也就是"小模型的 few-shot 效应分不清是不是噪声"。
- 这三条是教练的预期，不是论文结论；GPT-3 论文没有在 GPT-2 尺寸上报告 SST-2。

## Environment

- 机器：用户的 4090 云主机；本地 Mac 无 torch / transformers，教练只做了语法检查与 pyflakes；
- 依赖：`requirements.txt`（torch、transformers、datasets、numpy）；
- 数据与模型：SST-2（Hugging Face `stanfordnlp/sst2`，回退 `glue/sst2`）；GPT-2 四个尺寸的官方权重（OpenAI 2019 公开，经 Hugging Face 分发）。首次下载约 7GB。

## Implementation

- `icl_probe.py`：对每个 (model, k, seed)，从 SST-2 train 抽 K 条示例，对固定的 200 条 validation 查询逐条前向，取末位 " positive" 与 " negative" 两个 token 的 logit 比大小。记录 acc、多数类基线、预测为 positive 的比例（看先验偏向）、平均 prompt 长度。`--summarize` 按 model × k 打印均值 ± 样本标准差与 gap。可选 `--task addition`：两位数加法，贪心生成 4 个 token 解析整数。
- `run_probe.sh`：先冒烟（gpt2，K ∈ {0,4}，2 组示例，50 条），再正式跑四个尺寸。
- 三处 `TODO(you)`：(1) 拼 prompt；(2) 打分式读法比较两个 logit；(3) 跨 seed 样本标准差。

## Variables

| 变量 | 取值 | 对应论文 |
|---|---|---|
| model | gpt2 124M / gpt2-medium 355M / gpt2-large 774M / gpt2-xl 1.5B | 表 2.1 前四行的量级 |
| K | 0 / 1 / 4 / 8 | zero / one / few-shot；论文 K 到 10～100，GPT-2 上下文 1024 放不下 |
| seed | 0～4（K > 0） | 论文只报单次随机抽样 |
| 查询 | validation 固定打乱后前 200 条 | 全部设置共用，差异只来自 model、K、示例 |

固定：打分式读法、fp16、无任务描述（zero-shot 只有 "Review: … Sentiment:" 格式本身）。

## Metrics

- acc：200 条查询的准确率；
- majority：多数类基线；
- pred_pos_rate：预测为 positive 的比例，偏离 0.5 越多说明标签先验越强；
- gap：同一模型 K = 8 与 K = 0 的 acc 均值之差；
- std：同一 (model, K) 跨 5 组示例的样本标准差，作为噪声底线。

## Reproduction Commands

```bash
cd ~/openai-innovation-learning && git pull --ff-only origin main
cd experiments/in-context-learning
bash run_probe.sh
bash ../sync.sh "E3: 填空并跑四个尺寸"
```

只汇总：`python icl_probe.py --summarize`。

## Results

（待填）

## Interpretation

解读顺序：

1. zero-shot 四个尺寸是否都贴着 majority；pred_pos_rate 偏向哪边；
2. 每个模型 gap 是否为正，是否大于该格的 std（差值 vs 噪声，A2 复测的同一道题）；
3. gap 是否随 N 单调变宽；如果 gpt2-xl 的 gap 与 gpt2 的差不超过两者 std 之和，只能说"看不出趋势"；
4. K = 1 是否有时低于 K = 0（GPT-3 论文 LAMBADA 上 one-shot 低于 zero-shot）；
5. 哪些结论受 200 条查询的抽样误差影响：二项分布下 acc 0.7 的标准误约 0.032。

## Limitations

- 模型比 GPT-3 小 100～1000 倍，ICL 信号弱，可能四个尺寸都看不出明显 gap；
- 上下文 1024 限制 K ≤ 8；
- 单一任务、单一 prompt 格式；GPT-3 论文的结论来自 42 个基准；
- 打分式读法只比较两个 token，没做论文 §2.4 的归一化（两个标签都是单 token，不需要）；
- 这不是对 GPT-3 的复现，只是在开源小模型上重做"K 与尺寸两个变量"的实验设计。

## Knowledge Changes

（待填：结果写入 [[gpt-3-in-context-learning]] 的"最小示例或实验"段）

## 可选扩展（2026-10-01 追问引出）

示例标签随机打乱的 K = 8 对照（Min 等 2022 的设计，非 OpenAI）：准确率接近正常示例 → 示例主要激活格式与标签空间；掉到 zero-shot 水平 → 标签映射被用到。正式四尺寸跑完后再加开关。见 [[gpt-3-in-context-learning]] §11.2。

## Next Experiment

E4（Track B）：reward model + 策略优化，换用小型 pretrained LM。
