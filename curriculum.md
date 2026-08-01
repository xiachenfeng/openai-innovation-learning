# OpenAI Innovation Algorithms Curriculum

Last verified: 2026-08-01

## 总目标

建立四条贯穿时间线：

1. **Pretraining scaling**：GPT-1 → GPT-3；
2. **Alignment scaling**：Human Feedback → InstructGPT → Deliberative Alignment；
3. **Reasoning scaling**：Process supervision → o1/o3/o4 → GPT-5.x；
4. **Agent and system scaling**：WebGPT/Codex → Deep Research → Codex harness → multi-agent ultra。

---

# Phase 0：前置知识

- Transformer
- Autoregressive language modeling
- Cross entropy
- Fine-tuning
- Prompting and in-context learning
- PPO and reward models
- Contrastive learning
- Multimodal tokenization
- Tool calling and agent loops

---

# Phase 1：生成式预训练的建立

## 1.1 Early Generative Models

- Representation learning
- Unsupervised feature discovery

## 1.2 GPT-1

- Generative pretraining
- Supervised fine-tuning
- Task-aware input transformations
- Transfer learning

完成标准：

- 能解释为什么“先预训练再适配”改变 NLP；
- 能写一个小型 pretrain/fine-tune 实验。

---

# Phase 2：零样本、Scaling 与 In-context Learning

## 2.1 GPT-2

- Larger unsupervised LM
- Zero-shot task behavior
- Prompt conditioning
- Staged release

## 2.2 Scaling Laws

- Loss vs model/data/compute
- Power-law fitting
- Compute-efficient allocation
- Predictability

## 2.3 GPT-3

- Few-shot / one-shot / zero-shot
- No gradient updates at task time
- Natural-language programming
- Emergent capability and limitations

完成标准：

- 能比较 fine-tuning 与 in-context learning；
- 能拟合简单 scaling law；
- 能设计 few-shot 实验。

---

# Phase 3：Human Feedback 与可用性

## 3.1 Learning from Human Feedback

- Preference comparisons
- Reward model
- PPO
- KL control

## 3.2 WebGPT

- Browser actions
- Evidence and citations
- RLHF for factual question answering
- Tool-use precursor

## 3.3 InstructGPT

- SFT
- Preference ranking
- Reward model
- PPO
- Alignment tax and generalization

## 3.4 ChatGPT

- Dialogue fine-tuning
- Iterative deployment
- Multi-turn behavior

完成标准：

- 画出 RLHF pipeline；
- 实现 toy reward model；
- 比较 base GPT 与 instruction model。

---

# Phase 4：Multimodal Foundation Models

## 4.1 CLIP

- Contrastive language-image pretraining
- Natural-language supervision
- Zero-shot classification

## 4.2 DALL·E and Image Generation

- Text-conditioned image generation
- Prompt expansion and controllability
- Diffusion/autoregressive evolution按官方公开资料学习

## 4.3 Whisper

- Weak supervision at scale
- Multilingual speech recognition
- Robust transfer

## 4.4 GPT-4 and GPT-4o

- Large multimodal model
- Image + text reasoning
- GPT-4o real-time audio/vision/text
- Native multimodal interaction

## 4.5 Sora and World Simulation

- Video generation
- Space-time patches
- Scaling video models
- World-simulator hypothesis

完成标准：

- 能比较 contrastive、generative 和 omni-modal paradigms；
- 区分公开机制与未公开 GPT-4 internals。

---

# Phase 5：监督推理过程

## 5.1 Outcome vs Process Supervision

- Final-answer reward
- Step-level reward
- Process reward models
- Alignment benefits and costs

## 5.2 Critic Models

- CriticGPT
- AI-assisted human supervision
- Bug insertion and critique
- Scalable oversight

## 5.3 Prover-Verifier Games

- Legibility
- Verification
- Generator-verifier dynamics

完成标准：

- 实现 toy process reward；
- 设计 critic-assisted evaluation；
- 比较 outcome/process supervision。

---

# Phase 6：Reasoning Models

## 6.1 o1

- Large-scale RL on chain of thought
- Train-time RL scaling
- Test-time compute scaling
- Productive reasoning traces
- Limitations and hidden CoT

## 6.2 Deliberative Alignment

- Safety specification as training material
- Reasoning over policies
- Generalization and robustness

## 6.3 o3 and o4-mini

- Stronger reasoning
- Tool-integrated reasoning
- Visual reasoning in chain of thought
- Python and web tools

完成标准：

- 区分 pretraining compute、RL compute、test-time compute；
- 设计 pass@k / budget scaling 实验；
- 解释为什么公开“机制”不等于公开全部算法。

---

# Phase 7：Unified Models and Open Weights

## 7.1 GPT-5 Unified System

- Fast model
- Deeper reasoning model
- Real-time router
- Minimal reasoning
- Parallel test-time compute
- Safe-completions

## 7.2 gpt-oss

- MoE Transformer
- Total vs active parameters
- Alternating dense and locally banded sparse attention
- Grouped multi-query attention
- SFT + high-compute RL
- Tool use and reasoning

完成标准：

- 实现一个 fast/reasoning router；
- 分析 gpt-oss 公开架构；
- 区分 ChatGPT system routing 与 API model access。

---

# Phase 8：Agentic Systems

## 8.1 From WebGPT to Deep Research

- Search
- Browsing
- Citation
- Multi-step synthesis
- Long-running research

## 8.2 Codex

- Sandbox
- Repository context
- Tool loop
- Test-and-iterate
- RL on real-world coding tasks

## 8.3 Agent Loop and Harness

- Prompt construction
- Tools
- Responses API
- Context management
- Sandboxing
- Compaction
- Verification

## 8.4 Orchestration

- Subagents
- Parallel workstreams
- Symphony-style orchestration
- Human review control plane
- Harness engineering

完成标准：

- 实现最小 agent loop；
- 实现 context compaction；
- 设计单 Agent 与多 Agent 对比实验。

---

# Phase 9：GPT-5.4 → GPT-5.6 当前创新

## 9.1 Long-horizon Capability

- Professional work
- Agentic coding
- Computer use
- Long context
- Tool coordination

## 9.2 Efficiency

- More task success per token
- Load balancing
- Speculative decoding
- Caching
- Kernel optimization
- Workload-specific tuning

## 9.3 GPT-5.6 ultra

- Multiple agents
- Parallel workstreams
- Maximum capability setting
- Coordination overhead
- Verification and merging

## 9.4 Public Boundary

- Officially disclosed behavior and systems
- Undisclosed base network architecture
- Evaluation evidence vs causal claims

完成标准：

- 能解释 GPT-5.6 的公开创新重点；
- 不猜测未公开架构；
- 设计 token-efficiency 与 multi-agent 实验。

---

# Phase 10：Safety and Self-improvement

## 10.1 CoT Monitoring

- Detect reward hacking
- Monitorability
- Risks of directly optimizing hidden reasoning

## 10.2 Confessions and Instruction Hierarchy

- Behavioral transparency
- Layered safeguards
- Prompt-injection defense

## 10.3 GPT-Red

- Self-play RL
- Attacker/defender population
- Adversarial training
- Prompt-injection robustness

## 10.4 Model Spec

- Explicit behavioral specification
- Deliberative alignment
- Limits of specification-to-behavior mapping

完成标准：

- 解释 safety training 与 capability training 的耦合；
- 实现安全的 toy attacker/defender simulation，只用于无害文本分类环境；
- 评价 self-play 的覆盖与过拟合问题。

---

# Phase 11：Voice and Full-duplex Interaction

## GPT-Live

- Cascaded vs end-to-end voice
- Turn-based vs full-duplex
- Simultaneous listening and speaking
- Delegation to frontier model
- Latency, interruption and orchestration

完成标准：

- 画出三代 voice architecture；
- 分析 full-duplex 对 streaming、state 和 scheduling 的要求。

---

# Phase 12：实验

- `experiments/pretraining/`
- `experiments/scaling-laws/`
- `experiments/in-context-learning/`
- `experiments/rlhf/`
- `experiments/process-supervision/`
- `experiments/reasoning-scaling/`
- `experiments/gpt-oss/`
- `experiments/agent-loop/`
- `experiments/router/`
- `experiments/multi-agent/`
- `experiments/safety-self-play/`

---

# Phase 13：毕业项目

最终交付：

1. OpenAI 技术时间线；
2. GPT pretraining 与 scaling 知识卡；
3. RLHF 与 InstructGPT 知识卡；
4. Process supervision 与 reasoning scaling 知识卡；
5. Multimodal 演进知识卡；
6. GPT-5 router 与 gpt-oss 架构知识卡；
7. Agent loop、compaction 与 multi-agent 知识卡；
8. Safety/self-play 知识卡；
9. 至少五个 toy experiments；
10. `final-report.md`。
