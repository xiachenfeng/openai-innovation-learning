# Current Learning State

Last updated: 2026-09-16（A1 讲解与作业完成，E1 骨架待用户运行）

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

## Active Experiment

E1 tiny GPT：骨架在 experiments/pretraining/tiny_gpt/，用户需填两处空并在 4090 跑 `python train.py --smoke`。

## Next Recommended Action

1. 填 E1 两处空，跑 smoke，粘贴输出；2. 进入 A2 Sentiment Neuron → GPT-1；3. 09-23 复测 A1。
