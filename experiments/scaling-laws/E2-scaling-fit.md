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
