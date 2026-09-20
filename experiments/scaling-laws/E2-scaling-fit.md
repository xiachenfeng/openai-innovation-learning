# Experiment

- Name: E2 scaling-law 拟合（字符级 tiny GPT，4 个尺寸）
- Date: 2026-09-19 启动
- Related concept: [[scaling-laws]]（A4）；复用 [[transformer-decoder-next-token]] 的 E1 代码
- Status: 骨架已写，待用户填空并在 4090 上跑

## Question

1. 数据固定（tinyshakespeare，约 100 万字符）时，4 个尺寸（Kaplan 口径 N 约 2.5 万 → 470 万）的最低 val loss 在双对数坐标下是不是一条直线？斜率 α_N 是多少？
2. 最大模型是否偏离前三个点的直线，也就是数据成为瓶颈（知识卡 §6 的 L(N, D) 走平）？

## Hypothesis

- 前三个点近似直线，α_N 在 0.1～0.3 之间（字符级、小模型，指数通常比 Kaplan 的 0.076 大；这条是教练的猜测，不是论文结论）；
- 第四个点（470 万参数）的 best_val 高于直线外推值 0.02 以上：0.9M 训练 token 对 470 万参数不够，模型很快开始记训练集，val 早停在较高位置；
- best_step 随尺寸增大而提前（大模型更快过拟合），epochs_at_best 递减。

## Environment

- 机器：用户的 4090 云主机；本地 Mac 无 torch，教练只做了语法检查与 fit.py 的合成数据验证；
- 依赖：与 E1 相同（torch ≥ 2.0，numpy）；
- 数据：复用 E1 缓存 `experiments/pretraining/tiny_gpt/outputs/cache/input.txt`。

## Implementation

- `scaling.py`：从 E1 的 model.py / train.py 导入 GPT、get_batch、estimate_loss、load_text；训一个尺寸，每 100 步评估 val，连续 6 次不降就早停；记录 best_val、best_step、tokens_at_best、flops_at_best 到 `outputs/e2_sizes.csv`。两处 `TODO(you)`：Kaplan 口径的 N（总参数减两个 embedding）、C ≈ 6·N·D；
- `run_sizes.sh`：四个尺寸顺序跑完后调用 fit.py；
- `fit.py`：log L 对 log N 线性回归；先用全部点拟合，再用前三个点拟合并外推到第四个点，看偏离。两处 `TODO(you)`：np.polyfit 拟合、按拟合值预测。

## Variables

| tag | L | d | h | 12·L·d² | 备注 |
|---|---:|---:|---:|---:|---|
| s1 | 2 | 32 | 2 | 24,576 | |
| s2 | 3 | 64 | 4 | 147,456 | |
| s3 | 4 | 128 | 4 | 786,432 | E1 的模型 |
| s4 | 6 | 256 | 8 | 4,718,592 | 预期偏离直线 |
| s5（可选） | 8 | 384 | 8 | 14,155,776 | 脚本里注释掉了 |

固定：batch 64，T=128，lr 3e-4（所有尺寸相同，这是 Kaplan 的做法，也是 Chinchilla 批评的点），AdamW weight decay 0.1，seed 1337，max_steps 4000。

## Metrics

- best_val：早停时的最低 val loss（nats/token），对应 Kaplan 的 test loss；
- best_step、tokens_at_best、epochs_at_best：到达最低点时看了多少 token；
- flops_at_best = 6·N·tokens_at_best；
- α_N：双对数斜率的相反数。

## Reproduction Commands

```bash
cd ~/openai-innovation-learning && git pull --ff-only origin main
cd experiments/scaling-laws
bash run_sizes.sh                    # 四个尺寸 + 拟合，4090 上约 5～10 分钟
bash ../sync.sh "E2: 填空并跑四个尺寸"
```

单独重跑一个尺寸：`python scaling.py --n_layer 6 --n_embd 256 --n_head 8 --tag s4`；重新拟合：`python fit.py`。

## 怎么读 fit.py 的输出（2026-09-20 追问）

下面是教练用合成数据跑参考解得到的样例（不是真实结果，只看格式）：

```text
 tag   N_kaplan  best_val best_step  epochs      flops
  s1     27,576    1.0261      1000    9.10  1.355e+12
  s2    153,456    0.8351      1000    9.10  7.543e+12
  s3    801,432    0.6849      1000    9.10  3.939e+13
  s4  4,758,592    0.6331      1000    9.10  2.339e+14

全部 4 个点拟合：alpha_N = 0.096
前 3 个点拟合：alpha_N = 0.120，外推到 N = 4,758,592 预测 L = 0.5531，实际 0.6331
实际 − 预测 = +0.0800  → 最大模型偏离直线，数据瓶颈或欠训练

按 alpha = 0.096，N 翻 10 倍 loss 变为原来的 0.802 倍（Kaplan：0.839）
```

表格五列：

| 列 | 含义 | 看什么 |
|---|---|---|
| N_kaplan | 非 embedding 参数量，横轴 | 四个点应跨两个数量级 |
| best_val | 早停时的最低 val loss，纵轴 | 应随 N 单调下降；若 s4 ≥ s3 先查训练配方 |
| best_step | 到达最低点的步数 | 预期随 N 增大而提前（大模型样本效率高，也更早过拟合） |
| epochs | 到最低点时把训练集过了几遍 | 大于 1 说明数据在重复，D 不是论文意义上的 D |
| flops | 6·N·tokens_at_best | 用来画 L 对 C 的图，本轮不拟合 |

三行拟合结果：

1. **全部点的 alpha_N**：整体斜率。字符级 toy 的值不能和 Kaplan 的 0.076 比大小，只记录。
2. **前三点的 alpha_N 与外推**：关键行。用前三个点画直线，预测第四个点该在哪，再和实际比。"实际 − 预测"为正且超过 0.02，说明第四个点在直线上方，也就是走平：数据固定成了瓶颈（知识卡 §6 的 L(N, D)）或大模型欠训练。样例里 +0.08 就是这种情况。若差值接近 0，四个点仍在直线上，说明 100 万字符对 470 万参数还够，可以加跑 s5 找走平点。
3. **N 翻 10 倍 loss 乘以多少**：把指数换成直觉：0.80 表示参数放大十倍 loss 只降两成。

解读时按这个顺序答：四个点是否单调 → 前三点是否共线 → 第四点偏离多少、往哪个方向 → best_step 与 epochs 的趋势是否符合"大模型更早到最低点" → 哪些结论受单种子、20 个 batch 估计 val 的噪声影响。

## Results

（待用户跑完填写：把 fit.py 的输出表和两条 alpha 贴进来）

## Interpretation

（待用户先口头解读，教练再补：哪些点在直线上、α_N 与 Kaplan 的 0.076 为何不同、第四个点偏离说明什么、best_step 的趋势对应论文哪条结论）

## Limitations

- 数据只有 0.9M 训练 token 且重复多轮（epochs_at_best 会大于 1）；Kaplan 的 D 是不重复的 token，所以这里的 tokens_at_best 不是论文意义上的 D；
- 字符级、词表 65、模型 ≤ 500 万参数，指数与常数都不能和论文比，只能比形状；
- 四个点拟合一条直线，误差大；单种子；
- 所有尺寸同一学习率，小模型可能欠训练（Chinchilla 对 Kaplan 的批评在这里同样成立），可作为后续对照：给 s1、s2 换 lr 1e-3 再跑。

## Cost

- 算力：4090，预计 5～10 分钟；本地 CPU 未跑（无 torch）；
- API：无。

## Knowledge Changes

（待结果：写入 [[scaling-laws]] §6 的 toy 数据行与"掌握证据"）

## Next Experiment

E3 in-context learning 探针（A5），复用 s3 或 s4 的 checkpoint。
