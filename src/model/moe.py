import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod

# ---------------------------
# Abstract Router Interface
# ---------------------------

class Router(nn.Module, ABC):
    """
    Abstract Router class that computes routing decisions for the Mixture of Experts.
    It returns a gating output (weights) and indices of selected experts for each input.
    """
    def __init__(self, n_embed: int, num_experts: int, top_k: int) -> None:
        super(Router, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given input x, returns:
        - gating_output: Tensor of shape [batch, top_k], representing soft scores.
        - indices: Tensor of shape [batch, top_k], representing top-k expert indices.
        """
        raise NotImplementedError

# ---------------------------
# Mixture of Experts Module
# ---------------------------

class SparseMoE(nn.Module):
    """
    Implements a sparse Mixture of Experts layer using a given router and multiple experts.
    Only a top-k subset of experts process each token.
    """
    def __init__(
        self, 
        n_embed: int, 
        num_experts: int, 
        top_k: int, 
        router_class: Router
    ) -> None:
        super(SparseMoE, self).__init__()

        # Initialize the router (dispatches inputs to top-k experts)
        self.router = router_class(
            n_embed=n_embed, 
            num_experts=num_experts, 
            top_k=top_k
        )

        # Create a list of expert networks (typically FFNs)
        self.experts = nn.ModuleList(
            [Expert(n_embed=n_embed) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer:
        1. Route inputs to top-k experts.
        2. Process relevant inputs through each expert.
        3. Combine expert outputs using gating scores.
        """
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten input tensors for easier indexing
        flat_x = x.view(-1, x.size(-1))  # [batch * seq, n_embed]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # [batch * seq, top_k]

        for i, expert in enumerate(self.experts):
            # Mask: for each position, is expert i among its top-k?
            expert_mask = (indices == i).any(dim=-1)  # shape: [batch, seq]
            flat_mask = expert_mask.view(-1)  # [batch * seq]

            if flat_mask.any():
                # Select inputs routed to this expert
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Get corresponding gating scores (for expert i only)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)

                # Weight expert output using gating score
                weighted_output = expert_output * gating_scores

                # Add contribution back into final output
                final_output.view(-1, final_output.size(-1))[flat_mask] += weighted_output

        return final_output

# ---------------------------
# Expert Network
# ---------------------------

class Expert(nn.Module):
    """
    Each expert is a simple 2-layer MLP (FeedForward network) with ReLU and dropout.
    """
    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Expand hidden size
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # Project back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
