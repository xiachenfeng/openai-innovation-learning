# 知识库规范

把仓库根目录当作 Obsidian vault 打开即可浏览。不需要任何插件；图用 mermaid、数学用 LaTeX，Obsidian 原生渲染，仓库里不放图片文件。

## 层级（最多两级）

```text
knowledge/
  index.md                       ← 总入口（hub）
  papers.md                      ← 论文台账（hub）：每篇论文读到哪、哪张卡引用了什么
  A-pretraining-scaling/
    A-pretraining-scaling.md     ← 第 1 级：主线 MOC（Map of Content）
    A5-gpt-3-in-context-learning.md ← 第 2 级：概念卡
  B-alignment/ … C-reasoning/ … D-agents-systems/ … S-survey/ … F-frontier/
```

- 第 1 级文件名与文件夹同名，负责索引和主线因果图；
- 第 2 级是概念卡，一个概念一个文件，不再建子文件夹；
- canonical / candidate 不再用文件夹区分，改用 frontmatter 的 `status` 字段。

## 文件命名

- 概念卡文件名 = `<节点>-<概念>.md`，节点是 curriculum 的编号（A1～A5、B1～B5、C1～C5、D1～D5、S1、S2），F 线没有固定节点，用 `F-`；例如 `B3-instructgpt-chatgpt.md`。节点在前，Obsidian 文件列表自然按课程顺序排；
- 概念部分全小写 kebab-case 英文；MOC、index、papers、glossary 这些 hub 文件不加节点前缀；
- 全库唯一，因为 `[[wikilink]]` 按文件名解析，不看路径；
- 中文名和常用缩写放进 frontmatter 的 `aliases`，这样 `[[InstructGPT]]` 也能跳转；2026-10-14 加前缀改名时，旧文件名也放进了 `aliases`，旧链接仍可跳转。

## Frontmatter

```yaml
---
title: InstructGPT 与 ChatGPT
aliases: [InstructGPT, ChatGPT, RLHF 三阶段]
track: B
level: 2            # 1 = 主线 MOC，2 = 概念卡
status: candidate   # candidate | canonical
mastery: 0
importance: 5
created: 2026-09-13
updated: 2026-09-13
source_check: 2026-09-13
sources:
  - https://openai.com/index/instruction-following/
tags: [openai, track/B, status/candidate]
---
```

`tags` 里的 `track/X` 和 `status/X` 是嵌套标签，Obsidian 标签面板可以按层折叠。

## 链接

- 概念之间用 `[[文件名]]` 或 `[[文件名|显示文字]]`；
- 每张概念卡的"承接的瓶颈"和"后续影响"两段必须各含至少一个 wikilink，这样图谱视图能显示因果链；
- 指向尚未写的卡片也可以先写 `[[...]]`，Obsidian 显示为灰色未解析链接，等于待办；
- 引用官方来源用普通 markdown 链接，不用 wikilink。

## 公开边界用 callout

```markdown
> [!success] Public algorithm
> 论文明确披露的部分。

> [!info] Public system behavior
> 官方说明了机制但未公开训练细节。

> [!question] Inference
> 根据证据的推断，写明依据。

> [!danger] Unknown
> 架构、数据或算法未公开，不得猜测。
```

## 图：mermaid 规范

1. 训练流程和因果链用 `flowchart TB`（竖向，见第 8 条），多阶段用一层 `subgraph`；agent loop 和 tool call 用 `sequenceDiagram`；router 和状态机用 `stateDiagram-v2`；
2. 节点按语义着色，统一用下面这套 `classDef`，浅底深字，亮暗主题都可读：

```text
classDef data  fill:#EEF3FA,stroke:#3B6BB3,color:#111
classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
classDef frozen fill:#F2F2F2,stroke:#888,color:#333,stroke-dasharray:4 3
classDef loss  fill:#FDECEC,stroke:#C0392B,color:#111
classDef human fill:#EAF7EE,stroke:#2E8B57,color:#111
```

   含义：`data` 数据集，`model` 可训练模型，`frozen` 冻结模型，`loss` 目标函数，`human` 人工环节。

3. 因果链的边上写"解决了什么瓶颈"，用引号包住：`A -->|"每个任务仍需 fine-tune"| B`；
4. 节点标签里用 Unicode 写符号（π_θ、r_φ、β·KL），mermaid 节点内不渲染 LaTeX；
5. 数据节点用圆柱 `[( )]`，目标函数用六边形 `{{ }}`，决策用菱形 `{ }`；
6. 一张图只讲一件事，超过 12 个节点就拆两张；
7. 曲线关系（scaling、pass@k）优先用表格给数值，再用文字描述形状；`xychart-beta` 依赖 Obsidian 内置 mermaid 版本，可试用但不作为唯一表达。
8. 版式硬约束（2026-09-20 起，详见 `AGENTS.md` Teaching with Diagrams 第 8 条）：一律 `flowchart TB`；节点与边标签每行 ≤ 20 个汉字，用 `<br/>` 拆行；≤ 10 层；不嵌套 subgraph；subgraph 标题留空，用框内第一个无边框节点（`classDef label fill:none,stroke:none`）当标题并用 `~~~` 接到下一节点；源码开头加统一的 init 指令。目标是 16 寸屏一屏看全、A4 整张打印不裁切。示例见 `A-pretraining-scaling/A4-scaling-laws.md` §11.3。左右并列对照时（如 fine-tune vs prompt）：两个 subgraph 之间没有连线时，框内必须写 `direction TB`，否则会被排成一条横带；mermaid 11 会把先声明的框放在右边，所以想放在左边的那个要后声明。示例见 `A-pretraining-scaling/A3-gpt-2-zero-shot.md` §2.2。

## 数学：LaTeX 规范

- 行内 `$...$`，独立公式 `$$` 各占一行，多行用 `\begin{aligned}`；
- 表格单元格内的公式不要用 `|`，用 `\mid` 或 `\vert`；
- 统一记号：

| 记号 | 含义 |
|---|---|
| $\pi_\theta$ | 待训练策略 |
| $\pi_{\mathrm{ref}}$ / $\pi_{\mathrm{SFT}}$ | 参考策略 |
| $r_\phi$ | reward model |
| $\mathcal{L}$ / $\mathcal{J}$ | 最小化的损失 / 最大化的目标 |
| $\mathbb{E}$, $\mathrm{KL}$, $\operatorname{clip}$, $\sigma$ | 期望、KL 散度、截断、sigmoid |
| $N, D, C$ | 参数量、数据量、计算量 |

- 每个公式后面用一句话说明每个符号代表训练流程中的哪一步；
- 只写论文里有的公式；自己推导的部分标"推导"。

## 自己画图怎么做

教学图由教练画，你只画"主线因果图"。三档任选，默认第一档：

1. **填表**：在 session 里给出"上一代 / 下一代 / 上一代的瓶颈"三列表，教练转成图写入 MOC 的"我的版本"。
2. **填空**：复测时教练给节点已摆好、边标签写成 `|"???"|` 的骨架，你只改引号里的字。在编辑模式改，切阅读模式看图。
3. **自己写**：flowchart 只要四个语法点：

```text
flowchart TB
  A[节点文字]:::model --> B[另一个节点]:::data
  A -->|"边上的标签，有问号或空格要加引号"| C[第三个]
  classDef model fill:#FFF4E0,stroke:#D98E04,color:#111
  classDef data  fill:#EEF3FA,stroke:#3B6BB3,color:#111
```

写错时图的位置会显示红色报错和行号。语法错误不扣分。

## 可选：Dataview 查询

装了 Dataview 插件后，在任意笔记里可以列出全部概念卡：

```text
TABLE track, status, mastery, updated
FROM "knowledge"
WHERE level = 2
SORT track ASC, file.name ASC
```

## 示范

完整示范见 [[knowledge-card-example]]，模板见 [[knowledge-card]]。
