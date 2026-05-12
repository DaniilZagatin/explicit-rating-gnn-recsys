from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u = self.user_embeddings(user)
        v = self.item_embeddings(item)
        score = (u * v).sum(dim=-1)
        return score + self.user_bias(user).squeeze(-1) + self.item_bias(item).squeeze(-1)

    def score_all_items(self, user: torch.Tensor) -> torch.Tensor:
        u = self.user_embeddings(user)
        scores = u @ self.item_embeddings.weight.t()
        return scores + self.user_bias(user) + self.item_bias.weight.t()


class FeatureAwareItemEncoder(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, feature_dim: int = 0, hidden_dim: int = 256) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.feature_dim = int(feature_dim)
        if feature_dim > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embedding_dim),
            )
        else:
            self.feature_proj = None
        nn.init.normal_(self.item_embeddings.weight, std=0.02)

    def forward(self, item: torch.Tensor, item_features: torch.Tensor | None = None) -> torch.Tensor:
        out = self.item_embeddings(item)
        if self.feature_proj is not None and item_features is not None:
            out = out + self.feature_proj(item_features)
        return out

    def all_vectors(self, item_features: torch.Tensor | None = None) -> torch.Tensor:
        ids = torch.arange(self.item_embeddings.num_embeddings, device=self.item_embeddings.weight.device)
        return self.forward(ids, item_features)


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(x).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        empty = ~mask.any(dim=1)
        if empty.any():
            pooled[empty] = 0.0
        return pooled


class SASRecSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 50, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.max_len = max_len
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, item_vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = item_vectors.size(1)
        positions = torch.arange(length, device=item_vectors.device).unsqueeze(0)
        x = item_vectors + self.position_embeddings(positions.clamp_max(self.max_len - 1))
        causal = torch.triu(torch.ones(length, length, device=item_vectors.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=~mask)
        x = self.norm(x)
        last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
        return x[torch.arange(x.size(0), device=x.device), last_idx]


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        feature_dim: int = 0,
        hidden_dim: int = 256,
        max_history_len: int = 50,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_encoder = FeatureAwareItemEncoder(num_items, embedding_dim, feature_dim, hidden_dim)
        self.rating_proj = nn.Linear(1, embedding_dim)
        self.sequence_encoder = SASRecSequenceEncoder(embedding_dim, max_history_len, num_heads, num_layers, dropout)
        self.user_norm = nn.LayerNorm(embedding_dim)
        self.item_norm = nn.LayerNorm(embedding_dim)

    def encode_history(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
        history_mask: torch.Tensor,
        all_item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = None
        if all_item_features is not None:
            features = all_item_features[history_items]
        item_vectors = self.item_encoder(history_items, features)
        item_vectors = item_vectors + self.rating_proj(history_ratings.unsqueeze(-1))
        return self.user_norm(self.sequence_encoder(item_vectors, history_mask))

    def encode_item(self, item: torch.Tensor, item_features: torch.Tensor | None = None) -> torch.Tensor:
        return self.item_norm(self.item_encoder(item, item_features))

    def score_batch(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
        history_mask: torch.Tensor,
        item: torch.Tensor,
        item_features: torch.Tensor | None = None,
        all_item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u = self.encode_history(history_items, history_ratings, history_mask, all_item_features)
        v = self.encode_item(item, item_features)
        return (u * v).sum(dim=-1)

    def score_all_items_for_histories(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
        history_mask: torch.Tensor,
        all_item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u = self.encode_history(history_items, history_ratings, history_mask, all_item_features)
        v = self.item_norm(self.item_encoder.all_vectors(all_item_features))
        return u @ v.t()


class RelationGCMLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_relations: int = 5) -> None:
        super().__init__()
        self.transforms = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(num_relations)])
        self.norm_user = nn.LayerNorm(embedding_dim)
        self.norm_item = nn.LayerNorm(embedding_dim)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor, edge_index: torch.Tensor, relation_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_user, dst_item = edge_index
        user_msg = torch.zeros_like(user_x)
        item_msg = torch.zeros_like(item_x)
        for rel, transform in enumerate(self.transforms):
            mask = relation_id == rel
            if not mask.any():
                continue
            users = src_user[mask]
            items = dst_item[mask]
            user_msg.index_add_(0, users, transform(item_x[items]))
            item_msg.index_add_(0, items, transform(user_x[users]))
        return self.norm_user(user_x + user_msg), self.norm_item(item_x + item_msg)


class BilinearRatingDecoder(nn.Module):
    def __init__(self, embedding_dim: int, num_relations: int = 5) -> None:
        super().__init__()
        self.relation_matrices = nn.Parameter(torch.empty(num_relations, embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.relation_matrices)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bd,rde,be->br", user_x, self.relation_matrices, item_x)
        return scores


class RelationGCMCModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128, num_relations: int = 5, num_layers: int = 2) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.layers = nn.ModuleList([RelationGCMLayer(embedding_dim, num_relations) for _ in range(num_layers)])
        self.decoder = BilinearRatingDecoder(embedding_dim, num_relations)
        self.num_relations = num_relations
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)

    def encode(self, edge_index: torch.Tensor, relation_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_x = self.user_embeddings.weight
        item_x = self.item_embeddings.weight
        for layer in self.layers:
            user_x, item_x = layer(user_x, item_x, edge_index, relation_id)
        return user_x, item_x

    def forward(self, user: torch.Tensor, item: torch.Tensor, edge_index: torch.Tensor, relation_id: torch.Tensor) -> torch.Tensor:
        user_x, item_x = self.encode(edge_index, relation_id)
        logits = self.decoder(user_x[user], item_x[item])
        probs = torch.softmax(logits, dim=-1)
        values = torch.arange(1, self.num_relations + 1, device=probs.device, dtype=probs.dtype)
        return (probs * values).sum(dim=-1)


class RatingWeightedNGCFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.layers = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)

    def encode(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_user, dst_item = edge_index
        user_x = self.user_embeddings.weight
        item_x = self.item_embeddings.weight
        user_states = [user_x]
        item_states = [item_x]
        weights = edge_weight.float().unsqueeze(-1)
        for layer in self.layers:
            user_msg = torch.zeros_like(user_x)
            item_msg = torch.zeros_like(item_x)
            user_msg.index_add_(0, src_user, item_x[dst_item] * weights)
            item_msg.index_add_(0, dst_item, user_x[src_user] * weights)
            user_x = F.normalize(layer(user_msg), dim=-1)
            item_x = F.normalize(layer(item_msg), dim=-1)
            user_x = self.dropout(user_x)
            item_x = self.dropout(item_x)
            user_states.append(user_x)
            item_states.append(item_x)
        return torch.stack(user_states).mean(dim=0), torch.stack(item_states).mean(dim=0)

    def forward(self, user: torch.Tensor, item: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        user_x, item_x = self.encode(edge_index, edge_weight)
        return (user_x[user] * item_x[item]).sum(dim=-1)


class EnsembleTwoTowerGCMC(nn.Module):
    def __init__(self, two_tower: TwoTowerModel, gcmc: RelationGCMCModel, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.two_tower = two_tower
        self.gcmc = gcmc
        value = torch.logit(torch.tensor(float(init_alpha)).clamp(1e-4, 1 - 1e-4))
        self.alpha_logit = nn.Parameter(value)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, tt_score: torch.Tensor, gcmc_score: torch.Tensor) -> torch.Tensor:
        return self.alpha * tt_score + (1.0 - self.alpha) * gcmc_score
