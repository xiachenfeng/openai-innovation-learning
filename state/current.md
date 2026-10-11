# Current Learning State

Last updated: 2026-10-11（A5 检查题评完，mastery 1；论文台账建立）

## Long-term Goal

沿四条主线掌握 OpenAI 的公开算法创新：Pretraining & Scaling、Alignment、Reasoning、Agents & Systems。Multimodal 与 Voice 只做 survey。

## Time Budget

每周 3～4 小时，用户偏好拆成多次 40 分钟。目标：转岗。算力：一台 NVIDIA 4090 云主机。模型 API key：第 5 周前再办。

时间分配规则（幂律原则，见 curriculum.md 设计原则 2）：
- 每周最多一个课程节点，不因为"感觉慢"加速；
- importance 5 节点未到 mastery 3 不进入下一节，可以为它多花一个 40 分钟块；
- importance 4 节点复测一次后无论分数直接走，不追加课时；
- 周报里记录"本周耗时最多的节点及其 importance"，连续两周不是 5 就要调整。

## Current Phase

Track A 进行中：前置诊断已完成（8 题，见 answer-history 与 mastery）。

## Current Focus

A5 GPT-3 与 ICL 讲解完成（Track A 最后一节）：卡片 canonical，图板 10 张，八个追问入卡；E3 骨架推送，用 GPT-2 四个开源尺寸而非 E1 模型。10-11 检查题：Q1 2 分、Q2 3 分、Q3 0 分 → mastery 1。A5 importance 5 未到 3，按规则先复测（10-14）再进 Track B。之后：Track A 因果表（第一档）、E3 跑完解读、09-30 与 10-01 复测块。

上一节：A4 Scaling Laws 完成：卡片 canonical，十个追问入卡；E2 两轮跑完，用户解读复测 correctness 3 / mechanism 3 → mastery 2。短板：欠训练假象与固定 lr 计划批评的联系（09-23 补答）；连续六次未写自评信心。下一步：E2 结果解读、09-22～23 五项复测（A1 block、A2 解读、A2 机制、A3 读法、A4 预测）。诊断结论：A 线能识别术语（A1、A4 mastery 1），B/C/S 线从零起步，D 线能识别 tool calling（D2 mastery 1）。PyTorch 不能独立写训练循环，E1 骨架由教练提供。

## Source Baseline

截至 2026-09-13：GPT 主线最新公开模型为 GPT-6 Astra（2026-08 之后发布，尚未纳入课程，见 knowledge/F-frontier）。GPT-5.6 系列（Sol/Terra/Luna）为课程 Track C、D 引用的最新已分类版本。所有前沿模型的完整底层架构均未公开。

## Recent Progress

2026-09-13：完成课程设计评审与重构（见 sessions/2026-09-13-course-redesign.md）。
2026-09-13：完成 8 题前置诊断，进入 A1。
2026-09-15～16：A1 讲解、六个追问写入知识卡；作业 3/5 对，softmax 重做通过；E1 骨架写好并在本地 CPU 验证参考解。
2026-09-17：追问 7、8（logits vs 注意力分数、大词表 LM head）写入知识卡；用户填空并在 4090 跑通 E1 smoke。
2026-09-17（下午）：A2 讲解完成，两道检查题（3/2 分），A2 mastery 0 → 2；新增防乱码规则 + Stop hook、图板 Artifact 规则。
2026-09-18：用户在 AutoDL 4090 填完 sft.py、跑通 16 组并推送；云主机 git 工作流（sync.sh）建立。
2026-09-19：E1 结果解读（用户答不出，讲解入卡）；A2 → 3；A3 GPT-2 讲解完成，四个追问入卡。
2026-09-19（晚）：用户跑完 run_seeds（24 行，脚本跑了两遍，seed 不完全确定）；第 1 周周报写好；source status 更新（GPT-6 Astra 09-03 发布，新增两条 watch 线索）。
2026-09-19～20：A4 Scaling Laws 讲解（图板 7 张图），九个追问入卡；E2 骨架推送；知识卡图例规范固化到 AGENTS.md（竖向、A4 可打印）；检查题两道（3/1 分），A4 保持 1。
2026-09-20：E2 两轮跑完，A4 解读复测 → mastery 2。
2026-09-27～28：复测块（A2 解读、A3 读法、A1 多头 shape、CE 手算），A2 降到 2、A3 升到 1；session 文件未保存。
2026-09-28：A5 GPT-3 与 ICL 讲解，canonical 卡 + 4 张图入卡；E3 骨架推送（commit b3f3f54）。
2026-09-29～10-11：A5 八个追问入卡（§11.1～11.8），图板加到 10 张；来源表加 A-9、R-5。
2026-10-11：A5 检查题评分 → mastery 1；新建论文台账 knowledge/papers.md（P-1～P-21）并把规则写入 AGENTS.md。

## Active Experiment

E3 骨架已推送，待用户填三处空并在 4090 上跑（GPT-2 四尺寸 × K × 5 组示例，SST-2）。E1、E2 完成；E2 可选第三轮（s1/s2 换 lr 1e-3）未跑。

## Next Recommended Action

1. 新 session（10-14 起）：复测块一个 40 分钟，合并 A5（2×2 表、注意力路径、Q2 三条件）、A3、A2 解读、A4 第 4 问、A1 多头 shape，全部过期项一次清；每题先写信心；2. 云主机跑 E3 并 sync，结果解读计入 A5；3. Track A 因果表（第一档），教练转图写入 MOC；4. 补第 2、3 周周报；5. 手动删除根目录空文件 gpt-3-in-context-learning.md；6. 以上完成且 A5 ≥ 2 后进 Track B B1。
