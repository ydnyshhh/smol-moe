import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.moe import SparseMoE, Router

# -------------------------------
# Single Self-Attention Head
# -------------------------------
class Head(nn.Module):
    """
    A single self-attention head.
    Projects input to key, query, value and computes attention.
    """
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Lower-triangular mask to preserve autoregressive property
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**-0.5  # [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  # weighted sum of values
        return out

# -------------------------------
# Multi-Head Self-Attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads in parallel.
    """
    def __init__(self, num_heads: int, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embed, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# -------------------------------
# Transformer Block with MoE
# -------------------------------
class Block(nn.Module):
    """
    A single Transformer block with:
    - Multi-head self-attention
    - Sparse Mixture of Experts (MoE) in the feedforward path
    """
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        num_experts: int,
        top_k: int,
        block_size: int,
        router_class: Router,
    ) -> None:
        super().__init__()
        head_size = n_embed // n_head

        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k, router_class)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention (with residual)
        x = x + self.sa(self.ln1(x))

        # MoE feedforward layer (with residual)
        moe_output = self.smoe(self.ln2(x))
        x = x + moe_output

        return x

# -------------------------------
# SparseMoE Transformer Language Model
# -------------------------------
class SparseMoELanguageModel(nn.Module):
    """
    Full Transformer-based language model with Sparse MoE layers.
    """
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        num_experts: int,
        top_k: int,
        router_class: Router,
    ) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head, num_experts, top_k, block_size, router_class)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        # Token + Positional Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Language modeling loss (cross-entropy)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int) -> torch.Tensor:
        """
        Autoregressive generation using sampling.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to last block_size tokens
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]  # focus on last time step
            probs = F.softmax(logits, dim=-1)  # convert to probabilities

            idx_next = torch.multinomial(probs, num_samples=1)  # sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # append
        return idx
