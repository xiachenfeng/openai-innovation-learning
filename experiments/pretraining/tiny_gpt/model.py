"""
E1 tiny GPT —— 与知识卡 A1 "图 1 逐步说明" 的 13 步一一对应。

约定：B = batch，T = 序列长度，d = n_embd，h = n_head，d_k = d // h，V = vocab_size。
每一步的注释标出 tensor shape，和知识卡里的表格一致。

带 `TODO(you)` 的三处留给你填，每处一行，都是作业里手推过的内容：
  (1) causal mask
  (2) 交叉熵
其余代码是完整的。
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # 字符级词表，运行时由数据决定
    block_size: int = 128     # T_max，position embedding 的行数
    n_layer: int = 4          # L
    n_embd: int = 128         # d
    n_head: int = 4           # h
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """步 2 ～ 步 9：投影 → 切 head → 打分 → mask → softmax → 加权求和 → 合并 → W_O"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.n_embd // cfg.n_head
        # 步 2：W_Q, W_K, W_V 合并成一个 [d, 3d] 的矩阵，一次乘完再切开（等价于三个 [d, d]）
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # 步 9：W_O
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        # 步 5 用的 causal mask，注册为 buffer（不是参数，不参与训练，但会随模型搬到 GPU）。
        # TODO(you) (1): 生成一个 [T_max, T_max] 的下三角矩阵，主对角线及以下为 1，其余为 0。
        #   提示：torch.tril(torch.ones(T, T))
        #   作业第 2 题就是 T=4 时的这个矩阵。
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))  # ← 替换这一行
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape                                           # 步 0（已过 LN1）：[B, T, d]

        qkv = self.qkv(x)                                           # 步 2：[B, T, 3d]，逐 token 独立
        q, k, v = qkv.split(d, dim=2)                               #       各 [B, T, d]

        # 步 3：切 head。[B, T, d] → [B, T, h, d_k] → [B, h, T, d_k]，沿特征维切，不沿 token 维切
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 步 4：打分。每个 head 内 Q @ Kᵀ，沿 head 维批量做，head 之间零交流
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)       # [B, h, T, T]

        # 步 5：causal mask，j > i 的位置填 −∞
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 步 6：softmax，每行和为 1，被 mask 的项恰好为 0（作业第 3 题）
        att = F.softmax(att, dim=-1)                                # [B, h, T, T]
        att = self.dropout(att)

        # 步 7：加权求和，唯一跨 token 搬运信息的一步
        y = att @ v                                                 # [B, h, T, d_k]

        # 步 8：合并 head。[B, h, T, d_k] → [B, T, h, d_k] → [B, T, d]
        y = y.transpose(1, 2).contiguous().view(B, T, d)

        # 步 9：W_O，把不同 head 的通道混合（仍是逐 token）
        y = self.dropout(self.proj(y))                              # [B, T, d]
        return y


class MLP(nn.Module):
    """步 12：逐 token 的非线性加工，d → 4d → GELU → d"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)                                              # [B, T, 4d]
        x = F.gelu(x)
        x = self.proj(x)                                            # [B, T, d]
        return self.dropout(x)


class Block(nn.Module):
    """一个 decoder block（pre-LN 写法）：步 1 ～ 步 13"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)     # 步 1
        self.attn = CausalSelfAttention(cfg)     # 步 2 ～ 9
        self.ln_2 = nn.LayerNorm(cfg.n_embd)     # 步 11
        self.mlp = MLP(cfg)                      # 步 12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))          # 步 1 → 2～9 → 步 10 残差
        x = x + self.mlp(self.ln_2(x))           # 步 11 → 12 → 步 13 残差
        return x                                 # [B, T, d]，形状与输入相同


class GPT(nn.Module):
    """知识卡"图 2"：token ids → embedding → L 个 block → LN → LM head → logits → CE"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # token embedding   [V, d]
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)     # position embedding [T_max, d]
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # [d, V]
        # weight tying：LM head 与 token embedding 共享矩阵（GPT-2 做法，减少参数）
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()  # wte 与 lm_head 共享，已只算一次
        return n

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """token ids → 最后一层 LN 之后的隐状态 h，[B, T, d]。
        E1 第二阶段（sft.py）用它：GPT-1 微调取的就是这个 h 的最后一个位置，再过新增的 W_y。"""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} 超过 block_size={self.cfg.block_size}"
        pos = torch.arange(0, T, device=idx.device)                 # [T]

        x = self.wte(idx) + self.wpe(pos)                           # [B, T, d]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)                                            # [B, T, d]，L 次
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.hidden(idx)                                        # [B, T, d]
        logits = self.lm_head(x)                                    # [B, T, V]

        loss = None
        if targets is not None:
            # TODO(you) (2): 对 T 个位置同时算交叉熵取平均（作业第 4 题的批量版）。
            #   提示：F.cross_entropy 接受 [N, V] 的 logits 和 [N] 的 targets，
            #   所以要先把 [B, T, V] 摊成 [B*T, V]，targets 摊成 【】[B*T]。
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))  # ← 替换这一行
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """推理是串行的：每次前向只取最后一个位置的 logits，采样一个 token，拼上去再前向。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                 # [B, V]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)       # [B, 1]
            idx = torch.cat([idx, next_id], dim=1)
        return idx
