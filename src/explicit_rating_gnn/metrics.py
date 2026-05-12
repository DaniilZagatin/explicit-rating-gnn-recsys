from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def gain_from_rating(ratings: np.ndarray, gain_type: str = "binary_ge_4", beta: float = 0.2, gamma: float = 2.0) -> np.ndarray:
    ratings = np.asarray(ratings, dtype=np.float32)
    if gain_type in {"binary", "binary_ge_4"}:
        return (ratings >= 4.0).astype(np.float32)
    if gain_type == "raw":
        return ratings
    if gain_type == "centered_3":
        return np.maximum(ratings - 3.0, 0.0)
    if gain_type == "power":
        return np.maximum(ratings - 3.0 + beta, 0.0) ** gamma
    raise ValueError(f"Unknown gain type: {gain_type}")


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    gains = np.asarray(gains, dtype=np.float64)[:k]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains * discounts))


def _average_precision(hit_flags: np.ndarray, k: int) -> float:
    hit_flags = hit_flags[:k].astype(np.float32)
    total_hits = hit_flags.sum()
    if total_hits == 0:
        return 0.0
    precision_at_i = np.cumsum(hit_flags) / (np.arange(k) + 1)
    return float((precision_at_i * hit_flags).sum() / total_hits)


def compute_ranking_metrics(
    ranked_items: Mapping[int, Sequence[int]],
    targets: Mapping[int, Mapping[int, float]],
    ks: Iterable[int] = (10, 20, 50),
) -> dict[str, float]:
    stats: dict[str, list[float]] = defaultdict(list)
    for user, recs in ranked_items.items():
        target = targets.get(user, {})
        if not target:
            continue
        target_items = set(target)
        ideal_gains = sorted(target.values(), reverse=True)
        recs = list(recs)
        gains = np.array([float(target.get(item, 0.0)) for item in recs], dtype=np.float32)
        hits = np.array([item in target_items for item in recs], dtype=bool)
        for k in ks:
            top_hits = hits[:k]
            stats[f"precision@{k}"].append(float(top_hits.sum() / k))
            stats[f"recall@{k}"].append(float(top_hits.sum() / len(target_items)))
            stats[f"hitrate@{k}"].append(float(top_hits.any()))
            if top_hits.any():
                stats[f"mrr@{k}"].append(float(1.0 / (np.argmax(top_hits) + 1)))
            else:
                stats[f"mrr@{k}"].append(0.0)
            stats[f"map@{k}"].append(_average_precision(hits, min(k, len(hits))))
            ideal = dcg_at_k(ideal_gains, k)
            ndcg = dcg_at_k(gains, k) / ideal if ideal > 0 else 0.0
            stats[f"ndcg@{k}"].append(float(ndcg))
    return {key: float(np.mean(values)) for key, values in stats.items() if values}


def build_targets(
    interactions,
    user_col: str,
    item_col: str,
    rating_col: str = "rating",
    gain_type: str = "binary_ge_4",
) -> dict[int, dict[int, float]]:
    ratings = interactions[rating_col].to_numpy(dtype=np.float32)
    gains = gain_from_rating(ratings, gain_type=gain_type)
    targets: dict[int, dict[int, float]] = defaultdict(dict)
    for user, item, gain in zip(interactions[user_col], interactions[item_col], gains):
        if gain > 0:
            targets[int(user)][int(item)] = float(gain)
    return dict(targets)
