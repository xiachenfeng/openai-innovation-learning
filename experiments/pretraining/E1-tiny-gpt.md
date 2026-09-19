# Experiment

- Name: E1 tiny GPT（字符级 decoder-only 语言模型）
- Date: 2026-09-16 启动
- Related concept: [[transformer-decoder-next-token]]（A1）、[[sentiment-neuron-gpt-1]]（A2，第二阶段）
- Status: 第一阶段完成（2026-09-17）；第二阶段 16 组跑完（2026-09-18，用户填空三处全对，4090）；种子复验待跑

## Question

1. 本周（A1）：一个按知识卡 13 步实现的 decoder block 叠 4 层，能否跑通一次前向和一次反向，初始 loss 是否接近 ln V？
2. 下周（A2）：同一模型先预训练再 SFT，下游指标是否优于从零训练？

## Hypothesis

- 随机初始化时 loss ≈ ln V（V=65 时约 4.17），因为 softmax 近似均匀；
- 训练 2000 步后 val loss 应降到 1.5 左右（字符级 tinyshakespeare 的常见水平）。

## Environment

- 机器：用户的 NVIDIA 4090 云主机（CUDA），本地 Mac 无 torch；
- 依赖：Python ≥ 3.10（类型标注用了 `X | None` 语法），torch ≥ 2.0（教练用 2.14.0 CPU 版验证）；4090 上需 CUDA 构建，用 `torch.cuda.is_available()` 确认；无其他第三方依赖；
- 数据：tinyshakespeare（约 1MB 公开文本，脚本自动下载，失败退回内置文本）。

## Implementation

- `tiny_gpt/model.py`：GPTConfig、CausalSelfAttention（步 2～9）、MLP（步 12）、Block（步 1～13）、GPT（图 2）。两处 `TODO(you)`：causal mask、交叉熵；
- `tiny_gpt/train.py`：字符级词表、get_batch（input 与右移一位的 target）、smoke 模式、AdamW 训练循环、采样；
- `tiny_gpt/sft.py`（第二阶段）：解析台词、GPT-1 式输入拼法、词表扩 `<s>`/`<e>` 两个 token、GPTClassifier（Θ + W_y）、L3 = L2 + λ·L1、结果写 CSV。三处 `TODO(you)`：输入拼法、取 `<e>` 位置的 h、总损失；
- `tiny_gpt/run_stage2.sh`：16 组对照一次跑完并汇总。

默认配置：L=4，d=128，h=4，T=128，非 embedding 参数 ≈ 12·4·128² ≈ 786k。

### 与 GPT-1 论文方法的对应（2026-09-19 追问）

照论文：两阶段全参数微调、start/delimiter/extract 式输入拼法、末位隐状态过新增线性层、L3 = L2 + λ·L1（λ=0.5）、"不预训练"与"去掉辅助 LM"两项消融（论文表 5）。

教练自行设计：任务（台词猜说话人，论文用 SST-2/MNLI/RACE 等 12 个基准）、规模（80 万参数字符级 vs 1.17 亿 BPE）、微调日程（固定 600 步无 warmup，论文 3 epoch + warmup + 余弦）、标注量与学习率两个扫描、多种子复验。未做：LSTM 对照、迁移层数消融。结论只说明机制在小尺度下的行为，不能反推论文数字。可选补充：把微调日程改成 3～10 epoch 加 warmup，看 n=100 是否仍纯记忆。

## Variables

第一阶段无变量。第二阶段（2026-09-17 设计，对应 GPT-1 论文表 5 的消融）：

| 项 | 设计 |
|---|---|
| 预训练 | tinyshakespeare 2000 步（`python train.py --steps 2000`） |
| 下游任务 | 从语料抽 "SPEAKER:\n台词" 对，取出现最多的 5 个角色（GLOUCESTER、DUKE VINCENTIO、MENENIUS、ROMEO、PETRUCHIO），台词切成 ≤126 字符的片段，只看片段猜说话人。共 876 段 / 约 1480 片段，按整段分 80/20 防泄漏；测试集多数类基线约 0.25 |
| 输入拼法 | `<s>` 台词 `<e>`，字符级词表加 2 个新 token；取 `<e>` 位置的 h 过 W_y |
| 对照 | 同一 SFT 数据与步数：预训练权重起 vs 随机起 |
| 变量 | 标注量 100 vs 1000；λ = 0 vs 0.5；微调学习率 6e-5 vs 3e-4；共 2×2×2×2 = 16 组，`bash run_stage2.sh` 一次跑完 |
| 预期 | 预训练组在 100 条时优势最大；λ 在 1000 条时才有帮助；大学习率缩小预训练组优势。任务对 80 万参数的字符级模型偏难，绝对准确率可能只有 0.3～0.5，看的是组间差 |

用户填三处空：输入拼法函数、取 `<e>` 位置 h 的那一行、总损失合成。

## Metrics

- smoke：logits shape 为 [B, T, V]，初始 loss 与 ln V 的差；
- 训练：train/val loss 曲线；采样文本是否像英文。

## Reproduction Commands

```bash
cd experiments/pretraining/tiny_gpt
python train.py --smoke
python train.py --steps 2000
bash run_stage2.sh          # 第二阶段 16 组对照
bash run_seeds.sh           # 关键对照换 3 个种子
python sft.py --summary     # 汇总表
```

## Results

### 教练参考解验证（2026-09-16，本地 Mac CPU，torch CPU 版）

```text
vocab: 65  train tokens: 1,003,854
params (non-embedding): 801,664   12·L·d² ≈ 786,432
x: (8, 128)  y: (8, 128)
logits: (8, 128, 65)  loss: 4.2323   (ln V = 4.1744)
backward ok, grad norm: 4.5389
smoke test passed
```

100 步训练（batch 16）：step 50 val 2.849，step 100 val 2.645，4 秒。采样文本已有英文词形但无意义，符合预期。

### 用户 4090 运行（2026-09-17）

用户自行填了 causal mask 与交叉熵两处空。过程中两次出错并自行修正：`cross_enrtopy` 拼写；`logits.reshape(-1, x.size(-1))` 用了隐状态维 d 而非词表维 V，改为 `logits.size(-1)`。

```text
device: cuda
text chars: 1,115,394  vocab: 65  train tokens: 1,003,854
params (non-embedding): 801,664   12·L·d² ≈ 786,432
x: (64, 128)  y: (64, 128)
logits: (64, 128, 65)  loss: 4.2245   (随机初始化时应接近 ln V = 4.1744)
backward ok, grad norm: 4.2311
smoke test passed
```

与 CPU 参考解结果一致（loss 4.22 vs 4.23，差异来自随机初始化与 batch 采样）。

### 第二阶段：16 组对照（2026-09-18，用户在 4090 上跑，单种子 1337）

测试集 278 条，多数类基线 0.295，随机 0.200。二项噪声一个标准差约 0.03，差距小于 0.06 不算数。

| init       | n_train |   λ |   lr |  test acc | final train loss |
| ---------- | ------: | --: | ---: | --------: | ---------------: |
| pretrained |     100 |   0 | 6e-5 |     0.237 |            0.006 |
| pretrained |     100 |   0 | 3e-4 |     0.248 |            0.001 |
| pretrained |     100 | 0.5 | 6e-5 |     0.255 |            0.127 |
| pretrained |     100 | 0.5 | 3e-4 |     0.219 |            0.035 |
| pretrained |    1000 |   0 | 6e-5 |     0.335 |            0.338 |
| pretrained |    1000 |   0 | 3e-4 |     0.374 |            0.424 |
| pretrained |    1000 | 0.5 | 6e-5 |     0.324 |            1.344 |
| pretrained |    1000 | 0.5 | 3e-4 | **0.414** |            1.121 |
| scratch    |     100 |   0 | 6e-5 |     0.237 |            0.010 |
| scratch    |     100 |   0 | 3e-4 |     0.270 |            0.001 |
| scratch    |     100 | 0.5 | 6e-5 |     0.252 |            1.271 |
| scratch    |     100 | 0.5 | 3e-4 |     0.273 |            0.173 |
| scratch    |    1000 |   0 | 6e-5 |     0.349 |            1.113 |
| scratch    |    1000 |   0 | 3e-4 |     0.306 |            0.324 |
| scratch    |    1000 | 0.5 | 6e-5 |     0.360 |            2.335 |
| scratch    |    1000 | 0.5 | 3e-4 |     0.255 |            1.457 |

## Interpretation

- 初始 loss 4.22 ≈ ln 65 = 4.17：随机初始化下 LM head 对 65 个字符打分近似均匀，交叉熵等于均匀分布的熵。略高于 ln V 是因为随机 logits 并非严格相等，softmax 稍有偏离均匀，损失只会更大；
- 非 embedding 参数 801,664 与 12·L·d² = 786,432 的差是 LayerNorm 与 bias（约 15k），推导公式成立；
- 用户第二次错误（用 d 而非 V 摊平 logits）恰好对应知识卡追问 7 的混淆点：隐状态维 d 与词表维 V 是两个不同的轴。

### 第二阶段解读

1. **n=100 全军覆没**：16 组里 8 组 n=100 的测试准确率全部低于多数类基线，而训练损失降到 0.001～0.1，是纯记忆（600 步 × 32 ÷ 100 ≈ 192 epoch，GPT-1 只跑 3 epoch）。测试 0.22～0.27 等于五类瞎猜，瞎猜本来就低于多数类 0.295。与 GPT-1"标注少时预训练优势最大"的预期相反。原因：80 万参数、2000 步字符级预训练学到的是拼写和词形，没有"谁在说话"这种篇章级特征，100 条片段不足以从头建立它；测试集 278 条的噪声也接近组间差。
2. **n=1000 时预训练的优势只在大学习率下出现**：lr 3e-4 时预训练组 0.37～0.41 对从零组 0.26～0.31，差 0.1 以上，超过噪声；lr 6e-5 时两组 0.32～0.36 打平。与课上"大学习率会缩小预训练优势"的预测**相反**。解释：GPT-1 的 6.25e-5 是给 1.17 亿参数、上千步微调调的；这里 600 步 × 6e-5 总更新量太小，Θ 来不及从"字符 LM"适配到"说话人分类"，预训练特征还没被用上。学习率的绝对值不跨尺度迁移，要看 lr × 步数的总预算。
3. **预训练是正则化**：同样 n=1000、λ=0、lr 3e-4，从零组训练损失 0.32 / 测试 0.31（记住了训练集），预训练组训练损失 0.42 / 测试 0.37（记得少、泛化好）。
4. **辅助 LM 损失**：n=1000、lr 3e-4 时 λ=0.5 把预训练组从 0.374 提到 0.414，与论文"大数据集上有帮助"一致；lr 6e-5 时没有差别；从零组加 λ 反而更差，因为它要同时从零学两个目标。方向与论文一致但只有单种子，不能下结论。
5. **教练的预测错了一条**（大学习率缩小优势），实验推翻了它，这正是做 toy 实验的价值。下一步用 `run_seeds.sh` 换 3 个种子复验第 2、4 条。

## Limitations

- 字符级、<1M 参数、单一语料，只用于验证机制，不代表任何 OpenAI 模型的行为；
- 无 dropout、无学习率调度，训练效率不是目标；
- 第二阶段单种子、测试集 278 条，一个标准差约 0.03，只有大于 0.06 的差距可信；
- 说话人分类任务对字符级小模型偏难，绝对准确率 0.2～0.4，只看组间差。

## Cost

- 算力：4090 单卡，smoke < 1 分钟；2000 步预训练 + 第二阶段 16 组已跑（用户未报耗时，估计 10 分钟内）；种子复验 12 组待跑；本地 CPU 验证约 3 分钟（教练，300 步预训练 + 3 组 200 步 SFT）；
- API：无。

## Knowledge Changes

[[transformer-decoder-next-token]] "掌握证据"记录 E1 第一阶段完成；mastery 保持 1（升 2 需要复测时 mechanism ≥ 3）。

## Next Experiment

`run_seeds.sh` 复验 n=1000 下 init × lr 的差异是否稳定。之后 E2 复用同一代码训 3～4 个尺寸。
