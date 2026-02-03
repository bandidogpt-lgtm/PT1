"""Two-Tower recommendation model in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn


@dataclass
class TowerConfig:
    embedding_dim: int = 32
    hidden_dims: List[int] | None = None
    dropout: float = 0.1


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        numeric_user_dim: int,
        numeric_item_dim: int,
        config: TowerConfig,
    ):
        super().__init__()
        self.config = config
        # Embeddings for user-side categorical features.
        self.user_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, config.embedding_dim)
                for name, cardinality in user_cardinalities.items()
            }
        )
        # Embeddings for item-side categorical features.
        self.item_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, config.embedding_dim)
                for name, cardinality in item_cardinalities.items()
            }
        )

        user_input_dim = config.embedding_dim * len(user_cardinalities) + numeric_user_dim
        item_input_dim = config.embedding_dim * len(item_cardinalities) + numeric_item_dim

        hidden = config.hidden_dims or [64, 32]
        # Separate towers learn representations for users and items.
        self.user_tower = MLP(user_input_dim, hidden, config.dropout)
        self.item_tower = MLP(item_input_dim, hidden, config.dropout)

        # Projection layers align both towers into the same embedding space.
        self.user_proj = nn.Linear(hidden[-1], config.embedding_dim)
        self.item_proj = nn.Linear(hidden[-1], config.embedding_dim)

    def encode_user(self, user_cat: Dict[str, torch.Tensor], user_num: torch.Tensor) -> torch.Tensor:
        embeddings = [self.user_embeddings[name](tensor) for name, tensor in user_cat.items()]
        features = torch.cat(embeddings + [user_num], dim=1)
        user_hidden = self.user_tower(features)
        return self.user_proj(user_hidden)

    def encode_item(self, item_cat: Dict[str, torch.Tensor], item_num: torch.Tensor) -> torch.Tensor:
        embeddings = [self.item_embeddings[name](tensor) for name, tensor in item_cat.items()]
        features = torch.cat(embeddings + [item_num], dim=1)
        item_hidden = self.item_tower(features)
        return self.item_proj(item_hidden)

    def forward(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
        item_cat: Dict[str, torch.Tensor],
        item_num: torch.Tensor,
    ) -> torch.Tensor:
        # Dot-product similarity is the affinity score used for ranking.
        user_vec = self.encode_user(user_cat, user_num)
        item_vec = self.encode_item(item_cat, item_num)
        score = (user_vec * item_vec).sum(dim=1)
        return score
