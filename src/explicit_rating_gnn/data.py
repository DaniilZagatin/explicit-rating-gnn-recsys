from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_interactions(path: str | Path, required: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported interaction file: {path}")


def infer_user_item_columns(df: pd.DataFrame) -> tuple[str, str]:
    user_candidates = ["warm_user_idx", "user_idx", "cold_user_idx", "userId"]
    item_candidates = ["warm_item_idx", "item_idx", "all_item_idx", "cold_item_idx", "movieId"]
    user_col = next((c for c in user_candidates if c in df.columns), None)
    item_col = next((c for c in item_candidates if c in df.columns), None)
    if user_col is None or item_col is None:
        raise ValueError("Cannot infer user/item index columns")
    return user_col, item_col


def build_observed_items(df: pd.DataFrame, user_col: str, item_col: str) -> dict[int, set[int]]:
    observed: dict[int, set[int]] = defaultdict(set)
    for user, item in zip(df[user_col].to_numpy(), df[item_col].to_numpy()):
        observed[int(user)].add(int(item))
    return observed


def build_user_histories(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str = "rating",
    time_col: str = "timestamp",
    max_len: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    histories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sort_cols = [user_col, time_col] if time_col in df.columns else [user_col]
    for user, part in df.sort_values(sort_cols).groupby(user_col, sort=False):
        items = part[item_col].to_numpy(dtype=np.int64)
        ratings = part[rating_col].to_numpy(dtype=np.float32) if rating_col in part else np.ones(len(part), dtype=np.float32)
        if max_len is not None and len(items) > max_len:
            items = items[-max_len:]
            ratings = ratings[-max_len:]
        histories[int(user)] = (items, ratings)
    return histories


class PairwiseInteractionDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str = "rating",
        positive_threshold: float = 4.0,
        seed: int = 42,
    ) -> None:
        if user_col is None or item_col is None:
            user_col, item_col = infer_user_item_columns(interactions)
        positives = interactions[interactions[rating_col] >= positive_threshold].copy()
        if positives.empty:
            raise ValueError("No positive interactions for pairwise training")
        self.users = positives[user_col].to_numpy(dtype=np.int64)
        self.pos_items = positives[item_col].to_numpy(dtype=np.int64)
        self.ratings = positives[rating_col].to_numpy(dtype=np.float32)
        self.num_items = int(num_items)
        self.observed = build_observed_items(interactions, user_col, item_col)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, user: int) -> int:
        seen = self.observed.get(user, set())
        for _ in range(100):
            item = int(self.rng.integers(0, self.num_items))
            if item not in seen:
                return item
        return int(self.rng.integers(0, self.num_items))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user = int(self.users[idx])
        pos_item = int(self.pos_items[idx])
        neg_item = self._sample_negative(user)
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_item": torch.tensor(neg_item, dtype=torch.long),
            "rating": torch.tensor(float(self.ratings[idx]), dtype=torch.float32),
        }


class SequencePairwiseDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        histories: dict[int, tuple[np.ndarray, np.ndarray]],
        num_items: int,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str = "rating",
        positive_threshold: float = 4.0,
        max_history_len: int = 50,
        seed: int = 42,
    ) -> None:
        if user_col is None or item_col is None:
            user_col, item_col = infer_user_item_columns(interactions)
        positives = interactions[interactions[rating_col] >= positive_threshold].copy()
        self.users = positives[user_col].to_numpy(dtype=np.int64)
        self.pos_items = positives[item_col].to_numpy(dtype=np.int64)
        self.ratings = positives[rating_col].to_numpy(dtype=np.float32)
        self.histories = histories
        self.num_items = int(num_items)
        self.max_history_len = int(max_history_len)
        self.observed = build_observed_items(interactions, user_col, item_col)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, user: int) -> int:
        seen = self.observed.get(user, set())
        for _ in range(100):
            item = int(self.rng.integers(0, self.num_items))
            if item not in seen:
                return item
        return int(self.rng.integers(0, self.num_items))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user = int(self.users[idx])
        items, ratings = self.histories.get(user, (np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
        if len(items) > self.max_history_len:
            items = items[-self.max_history_len:]
            ratings = ratings[-self.max_history_len:]
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "history_items": torch.tensor(items, dtype=torch.long),
            "history_ratings": torch.tensor(ratings, dtype=torch.float32),
            "pos_item": torch.tensor(int(self.pos_items[idx]), dtype=torch.long),
            "neg_item": torch.tensor(self._sample_negative(user), dtype=torch.long),
            "rating": torch.tensor(float(self.ratings[idx]), dtype=torch.float32),
        }


def sequence_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max((x["history_items"].numel() for x in batch), default=0)
    max_len = max(max_len, 1)
    history_items = torch.zeros(len(batch), max_len, dtype=torch.long)
    history_ratings = torch.zeros(len(batch), max_len, dtype=torch.float32)
    history_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["history_items"].numel()
        if length:
            history_items[row, -length:] = item["history_items"]
            history_ratings[row, -length:] = item["history_ratings"]
            history_mask[row, -length:] = True
    return {
        "user": torch.stack([x["user"] for x in batch]),
        "history_items": history_items,
        "history_ratings": history_ratings,
        "history_mask": history_mask,
        "pos_item": torch.stack([x["pos_item"] for x in batch]),
        "neg_item": torch.stack([x["neg_item"] for x in batch]),
        "rating": torch.stack([x["rating"] for x in batch]),
    }
