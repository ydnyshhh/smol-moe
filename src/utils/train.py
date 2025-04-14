import time
import torch
import wandb
import torch.nn as nn
from torch.nn import init
from typing import Callable, List, TypedDict, NamedTuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.data.loader import Loader
from src.data.tokenizer import Tokenizer
from src.model.transformer import SparseMoELanguageModel
from src.model.moe import Router
from src.model.router import NoisyTopKRouter
from src.utils.utils import set_seed

set_seed(42)


class CheckpointLoss(TypedDict):
    train: float
    val: float


@torch.no_grad()
def estimate_loss(
    model: SparseMoELanguageModel, get_batch: Callable, eval_iters: int = 100
) -> CheckpointLoss:
    device = next(model.parameters()).device
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.mean().item()
        out[split] = losses.mean()
    model.train()
    return CheckpointLoss(train=out["train"], val=out["val"])


def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)


def get_model(
    vocab_size: int,
    block_size: int,
    router_class: Router,
    n_embed: int = 128,
    n_layer: int = 8,
    n_head: int = 8,
    num_experts: int = 8,
    top_k: int = 2,
) -> SparseMoELanguageModel:
    model = SparseMoELanguageModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        num_experts=num_experts,
        top_k=top_k,
        router_class=router_class,
    )
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    model.apply(kaiming_init_weights)
    return model


class TrainerResults(NamedTuple):
    model: SparseMoELanguageModel
    train_losses: List[float]
    val_losses: List[float]


def train_loop(
    experiment_name: str,
    experiment_group: str,
    lr: float = 1e-3,
    max_iters: int = 100,
    eval_interval: int = 10,
    batch_size: int = 16,
    block_size: int = 32,
    top_k: int = 2,
    n_experts: int = 8,
    n_embed: int = 128,
    n_layer: int = 8,
    n_head: int = 8,
    device: str = "cuda",
) -> TrainerResults:
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="smol-moe",
        name=experiment_name,
        group=experiment_group,
        config={
            "lr": lr,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "batch_size": batch_size,
            "block_size": block_size,
            "top_k": top_k,
            "n_experts": n_experts,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
        },
    )

    tokenizer = Tokenizer()
    loader = Loader(tokenizer)

    model = get_model(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        top_k=top_k,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        num_experts=n_experts,
        router_class=NoisyTopKRouter,  # type: ignore
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model=model,
                get_batch=lambda split: loader.get_batch(split, batch_size, block_size),
            )
            print(f"Step {iter}: train_loss={losses['train']:.4f} | val_loss={losses['val']:.4f}")
            wandb.log({"train_loss": losses["train"], "val_loss": losses["val"]}, step=iter)
            val_losses.append(losses["val"])

        xb, yb = loader.get_batch("train", batch_size=batch_size, block_size=block_size)
        xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        loss_value = loss.mean().item()
        train_losses.append(loss_value)

        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        optimizer.step()

        wandb.log({"train_loss_step": loss_value}, step=iter)

    return TrainerResults(model=model, train_losses=train_losses, val_losses=val_losses)
