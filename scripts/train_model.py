from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explicit_rating_gnn import MFModel, TwoTowerModel, load_config, train_pairwise_epoch
from explicit_rating_gnn.data import PairwiseInteractionDataset, SequencePairwiseDataset, build_user_histories, infer_user_item_columns, sequence_collate_fn
from explicit_rating_gnn.utils import ensure_dir, resolve_device, set_seed


def build_model(cfg, train_df: pd.DataFrame, num_users: int, num_items: int) -> torch.nn.Module:
    name = cfg.model.name.lower()
    if name in {"mf", "matrix_factorization"}:
        return MFModel(num_users, num_items, cfg.model.embedding_dim)
    if name in {"two_tower", "sasrec", "two_tower_sasrec"}:
        return TwoTowerModel(
            num_items=num_items,
            embedding_dim=cfg.model.embedding_dim,
            feature_dim=0,
            hidden_dim=cfg.model.hidden_dim,
            max_history_len=cfg.model.max_history_len,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    raise ValueError(f"Unsupported model for script training: {cfg.model.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", default="train_warm_interactions.parquet")
    parser.add_argument("--checkpoint-name", default="model.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    dataset_dir = Path(cfg.paths.dataset_dir)
    train_df = pd.read_parquet(dataset_dir / args.train_file)
    user_col, item_col = infer_user_item_columns(train_df)
    num_users = int(train_df[user_col].max()) + 1
    num_items = int(train_df[item_col].max()) + 1

    model = build_model(cfg, train_df, num_users, num_items).to(device)
    name = cfg.model.name.lower()

    if name in {"two_tower", "sasrec", "two_tower_sasrec"}:
        histories = build_user_histories(train_df, user_col, item_col, max_len=cfg.model.max_history_len)
        dataset = SequencePairwiseDataset(
            train_df,
            histories=histories,
            num_items=num_items,
            user_col=user_col,
            item_col=item_col,
            positive_threshold=cfg.eval.positive_threshold,
            max_history_len=cfg.model.max_history_len,
            seed=cfg.train.seed,
        )
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=sequence_collate_fn)

        def score_fn(model, batch):
            pos = model.score_batch(batch["history_items"], batch["history_ratings"], batch["history_mask"], batch["pos_item"])
            neg = model.score_batch(batch["history_items"], batch["history_ratings"], batch["history_mask"], batch["neg_item"])
            return pos, neg
    else:
        dataset = PairwiseInteractionDataset(
            train_df,
            num_items=num_items,
            user_col=user_col,
            item_col=item_col,
            positive_threshold=cfg.eval.positive_threshold,
            seed=cfg.train.seed,
        )
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)

        def score_fn(model, batch):
            return model(batch["user"], batch["pos_item"]), model(batch["user"], batch["neg_item"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    for epoch in range(1, cfg.train.epochs + 1):
        metrics = train_pairwise_epoch(model, loader, optimizer, device, score_fn, cfg.train.grad_clip_norm)
        print(f"epoch={epoch} loss={metrics['loss']:.6f}")

    checkpoint_dir = ensure_dir(cfg.paths.checkpoint_dir)
    torch.save({"model_state_dict": model.state_dict(), "config": args.config}, checkpoint_dir / args.checkpoint_name)
    print(checkpoint_dir / args.checkpoint_name)


if __name__ == "__main__":
    main()
