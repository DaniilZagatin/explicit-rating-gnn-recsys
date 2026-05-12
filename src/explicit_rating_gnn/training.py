from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import move_to_device


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_pairwise_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    score_fn: Callable[[torch.nn.Module, dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pos_scores, neg_scores = score_fn(model, batch)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return {"loss": total_loss / max(total_batches, 1)}


@torch.inference_mode()
def recommend_topk(
    scores: torch.Tensor,
    k: int,
    seen_items: dict[int, set[int]] | None = None,
    users: list[int] | None = None,
) -> dict[int, list[int]]:
    scores = scores.detach().clone()
    if seen_items and users:
        for row, user in enumerate(users):
            seen = list(seen_items.get(int(user), set()))
            if seen:
                scores[row, seen] = -torch.inf
    topk = torch.topk(scores, k=min(k, scores.size(1)), dim=1).indices.cpu().numpy()
    if users is None:
        users = list(range(scores.size(0)))
    return {int(user): [int(x) for x in items] for user, items in zip(users, topk)}
