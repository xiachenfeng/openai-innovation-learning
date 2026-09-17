# Week 1 Plan（按每周 3～4 小时设计）

## Goal

完成前置诊断，启动 Track A，写出 tiny GPT 的骨架。不做全景扫描；全景由四张主线因果图在各 Track 结束时逐步画出。

## 主 session（约 2 小时）

1. 前置诊断（8 项清单，每次一个主问题，约 40 分钟）
2. A1：Transformer decoder 与 next-token objective
   - 教练画 decoder block 数据流图（标 tensor shape），存入 A1 概念卡
   - 你用一个 4 token 的例子手推 causal mask 和 cross entropy，不需要画图
3. 更新 `state/mastery.md` 与 `state/current.md`

## 实验时间（约 1.5 小时）

- E1 启动：写 tiny GPT（字符级，≤2M 参数），跑通一次前向和一步反向
- 记录到 `experiments/pretraining/` 的 experiment 文件

## Weekly Output

- 一张 decoder 数据流图（教练画）
- 诊断结果写入 mastery 表
- E1 骨架代码可运行
- 一次"公开事实 vs 推断"小测（3 题）

## 后续周次概览

| 周 | 内容 | 实验 |
|---|---|---|
| 2 | A2 Sentiment Neuron → GPT-1 | E1 完成 |
| 3 | A3 GPT-2 | 复习 |
| 4 | A4 Scaling Laws + Chinchilla | E2 |
| 5 | A5 GPT-3 ICL | E3 |
| 6～9 | B1～B4 | E4 |
| 10～14 | C1～C5 | E5 |
| 15 | B5 | 可选 E8 |
| 16～20 | D1～D5 | E6 |
| 21～22 | S1～S2 | 无 |
| 每月 | Track F 滚动 | 无 |
