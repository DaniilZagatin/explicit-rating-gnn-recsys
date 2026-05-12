from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class PathsConfig:
    dataset_dir: str = ""
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ModelConfig:
    name: str = "two_tower"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    max_history_len: int = 50
    num_heads: int = 4
    num_ratings: int = 5
    use_item_features: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 1024
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 2
    device: str = "auto"
    grad_clip_norm: float | None = 5.0


@dataclass
class EvalConfig:
    ks: tuple[int, ...] = (10, 20, 50)
    positive_threshold: float = 4.0
    main_metric: str = "ndcg@20"
    gain_type: str = "binary_ge_4"


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dataclass(instance: Any, values: Mapping[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            if key == "ks" and isinstance(value, list):
                value = tuple(value)
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExperimentConfig()
    return _merge_dataclass(cfg, raw)
