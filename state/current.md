# Current Learning State

Last updated: 2026-09-17（A1 完成，E1 第一阶段 smoke 通过）

## Long-term Goal

沿四条主线掌握 OpenAI 的公开算法创新：Pretraining & Scaling、Alignment、Reasoning、Agents & Systems。Multimodal 与 Voice 只做 survey。

## Time Budget

每周 3～4 小时，用户偏好拆成多次 40 分钟。目标：转岗。算力：一台 NVIDIA 4090 云主机。模型 API key：第 5 周前再办。

## Current Phase

Track A 进行中：前置诊断已完成（8 题，见 answer-history 与 mastery）。

## Current Focus

A1 Transformer decoder 与 next-token objective。诊断结论：A 线能识别术语（A1、A4 mastery 1），B/C/S 线从零起步，D 线能识别 tool calling（D2 mastery 1）。PyTorch 不能独立写训练循环，E1 骨架由教练提供。

## Source Baseline

截至 2026-09-13：GPT 主线最新公开模型为 GPT-6 Astra（2026-08 之后发布，尚未纳入课程，见 knowledge/F-frontier）。GPT-5.6 系列（Sol/Terra/Luna）为课程 Track C、D 引用的最新已分类版本。所有前沿模型的完整底层架构均未公开。

## Recent Progress

2026-09-13：完成课程设计评审与重构（见 sessions/2026-09-13-course-redesign.md）。
2026-09-13：完成 8 题前置诊断，进入 A1。
2026-09-15～16：A1 讲解、六个追问写入知识卡；作业 3/5 对，softmax 重做通过；E1 骨架写好并在本地 CPU 验证参考解。
2026-09-17：追问 7、8（logits vs 注意力分数、大词表 LM head）写入知识卡；用户填空并在 4090 跑通 E1 smoke。

## Active Experiment

E1 tiny GPT：第一阶段完成（4090 smoke 通过）。第二阶段（pretrain → SFT vs from-scratch）随 A2 进行。

## Next Recommended Action

1. 进入 A2 Sentiment Neuron → GPT-1，E1 加 SFT 阶段；2. 09-23 复测 A1 block 结构；3. 可选：先跑 `python train.py --steps 2000` 看采样效果。
