import torch
import torch.nn as nn
from typing import Tuple
from torch.nn import functional as F

from src.model.moe import Router


class NoisyTopKRouter(Router):
    """
    Implements a Noisy Top-k Router for Mixture of Experts with load balancing.
    """

    def __init__(self, n_embed: int, num_experts: int, top_k: int) -> None:
        super().__init__(n_embed, num_experts, top_k)
        self.top_k_route_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, n_embed]

        Returns:
            router_output: Softmax over sparse top-k logits, shape [batch, seq_len, num_experts]
            indices: Top-k expert indices for each token, shape [batch, seq_len, top_k]
            aux_loss: Load balancing loss (scalar tensor)
        """
        
        # B --> Batch Size
        # S --> Sequence Length
        # K --> Number of Active Experts
        # E --> Number of Experts
        
        logits = self.top_k_route_linear(x)  # [B, S, E]
        noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)  # [B, S, K]
        zeros = torch.full_like(noisy_logits, float("-inf"))  # [B, S, E]
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # Only top-k retained
        router_output = F.softmax(sparse_logits, dim=-1)  # [B, S, E]

        # Compute auxiliary load balancing loss
        aux_loss = self.load_balancing_loss(router_output)

        return router_output, indices, aux_loss

    def load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourages balanced expert usage via entropy-based loss.

        Args:
            router_probs: Routing probabilities [batch, seq_len, num_experts]

        Returns:
            aux_loss: Scalar tensor, higher if experts are imbalanced
        """
        # Average router probabilities across batch and seq
        # shape: [num_experts]
        expert_prob = router_probs.mean(dim=(0, 1))  # [E]

        # Encourage uniform distribution → minimize squared deviation
        # (same as KL divergence from uniform + constant)
        balance_loss = torch.sum(expert_prob * expert_prob) * self.num_experts

        return balance_loss
