from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explicit_rating_gnn import MFModel, TwoTowerModel, load_config
from explicit_rating_gnn.data import build_observed_items, build_user_histories, infer_user_item_columns, sequence_collate_fn
from explicit_rating_gnn.metrics import build_targets, compute_ranking_metrics
from explicit_rating_gnn.training import recommend_topk
from explicit_rating_gnn.utils import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-file", default="train_warm_interactions.parquet")
    parser.add_argument("--eval-file", default="warm_test_interactions.parquet")
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    dataset_dir = Path(cfg.paths.dataset_dir)
    train_df = pd.read_parquet(dataset_dir / args.train_file)
    eval_df = pd.read_parquet(dataset_dir / args.eval_file)
    user_col, item_col = infer_user_item_columns(train_df)
    eval_user_col, eval_item_col = infer_user_item_columns(eval_df)

    num_users = int(max(train_df[user_col].max(), eval_df[eval_user_col].max())) + 1
    num_items = int(max(train_df[item_col].max(), eval_df[eval_item_col].max())) + 1

    if cfg.model.name.lower() in {"mf", "matrix_factorization"}:
        model = MFModel(num_users, num_items, cfg.model.embedding_dim).to(device)
    else:
        model = TwoTowerModel(num_items=num_items, embedding_dim=cfg.model.embedding_dim, max_history_len=cfg.model.max_history_len).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    users = sorted(int(u) for u in eval_df[eval_user_col].unique())
    seen = build_observed_items(train_df, user_col, item_col)

    if isinstance(model, MFModel):
        user_tensor = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.score_all_items(user_tensor)
    else:
        histories = build_user_histories(train_df, user_col, item_col, max_len=cfg.model.max_history_len)
        rows = []
        for user in users:
            items, ratings = histories.get(user, ([], []))
            rows.append({
                "user": torch.tensor(user),
                "history_items": torch.tensor(items, dtype=torch.long),
                "history_ratings": torch.tensor(ratings, dtype=torch.float32),
                "pos_item": torch.tensor(0),
                "neg_item": torch.tensor(0),
                "rating": torch.tensor(0.0),
            })
        batch = sequence_collate_fn(rows)
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = model.score_all_items_for_histories(batch["history_items"], batch["history_ratings"], batch["history_mask"])

    ranked = recommend_topk(scores, k=args.topk, seen_items=seen, users=users)
    targets = build_targets(eval_df, eval_user_col, eval_item_col, gain_type=cfg.eval.gain_type)
    metrics = compute_ranking_metrics(ranked, targets, ks=cfg.eval.ks)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
