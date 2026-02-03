"""Train Two-Tower model with BCE or BPR loss."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.two_tower import TowerConfig, TwoTowerModel

DATA_DIR = Path("/workspace/PT1/data/processed")
MODEL_DIR = Path("/workspace/PT1/data/models")


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    loss_type: str = "bce"  # or "bpr"
    device: str = "cpu"


USER_CAT_COLS = ["cliente_id", "rubro_cliente", "canal_venta"]
ITEM_CAT_COLS = ["producto_id", "categoria_producto"]
USER_NUM_COLS = ["frecuencia_compra", "ticket_promedio", "mes", "semana_anio", "cerca_feriado_7d"]
ITEM_NUM_COLS = [
    "precio_unitario",
    "costo_unitario",
    "stock",
    "dias_para_vencer",
    "indice_proximo_vencer",
    "rotacion_producto",
    "indice_baja_rotacion",
]


class RecDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        # Package categorical and numerical features for the two-tower model.
        sample = {
            "user_cat": torch.tensor(row[USER_CAT_COLS].values.astype(np.int64)),
            "item_cat": torch.tensor(row[ITEM_CAT_COLS].values.astype(np.int64)),
            "user_num": torch.tensor(row[USER_NUM_COLS].values.astype(np.float32)),
            "item_num": torch.tensor(row[ITEM_NUM_COLS].values.astype(np.float32)),
            "label": torch.tensor(row["label_compra"], dtype=torch.float32),
        }
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    user_cat = torch.stack([b["user_cat"] for b in batch])
    item_cat = torch.stack([b["item_cat"] for b in batch])
    user_num = torch.stack([b["user_num"] for b in batch])
    item_num = torch.stack([b["item_num"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    # Build dicts so the model can map each categorical feature to its embedding.
    user_cat_dict = {col: user_cat[:, idx] for idx, col in enumerate(USER_CAT_COLS)}
    item_cat_dict = {col: item_cat[:, idx] for idx, col in enumerate(ITEM_CAT_COLS)}

    return {
        "user_cat": user_cat_dict,
        "item_cat": item_cat_dict,
        "user_num": user_num,
        "item_num": item_num,
        "labels": labels,
    }


def _load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    data = pd.read_csv(DATA_DIR / "dataset.csv")
    with open(DATA_DIR / "encoders.json", "r", encoding="utf-8") as f:
        encoders = json.load(f)

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]
    return train, val, test, encoders


def _build_model(encoders: Dict[str, Dict[str, int]]) -> TwoTowerModel:
    user_cardinalities = {col: len(encoders[col]) for col in USER_CAT_COLS}
    item_cardinalities = {col: len(encoders[col]) for col in ITEM_CAT_COLS}

    model = TwoTowerModel(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        numeric_user_dim=len(USER_NUM_COLS),
        numeric_item_dim=len(ITEM_NUM_COLS),
        config=TowerConfig(),
    )
    return model


def train_model(cfg: TrainConfig) -> None:
    train_df, val_df, _, encoders = _load_dataset()

    model = _build_model(encoders)
    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(
        RecDataset(train_df),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        RecDataset(val_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    # Training loop supports BCE (pointwise) or BPR (pairwise) loss.
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            user_cat = {k: v.to(device) for k, v in batch["user_cat"].items()}
            item_cat = {k: v.to(device) for k, v in batch["item_cat"].items()}
            scores = model(
                user_cat,
                batch["user_num"].to(device),
                item_cat,
                batch["item_num"].to(device),
            )
            labels = batch["labels"].to(device)
            # BPR optimizes ranking by separating positive and negative scores.
            if cfg.loss_type == "bpr":
                pos_scores = scores[labels == 1]
                neg_scores = scores[labels == 0]
                loss = -torch.log(torch.sigmoid(pos_scores[:, None] - neg_scores[None, :]) + 1e-8).mean()
            else:
                loss = bce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                user_cat = {k: v.to(device) for k, v in batch["user_cat"].items()}
                item_cat = {k: v.to(device) for k, v in batch["item_cat"].items()}
                scores = model(
                    user_cat,
                    batch["user_num"].to(device),
                    item_cat,
                    batch["item_num"].to(device),
                )
                labels = batch["labels"].to(device)
                if cfg.loss_type == "bpr":
                    pos_scores = scores[labels == 1]
                    neg_scores = scores[labels == 0]
                    loss = -torch.log(torch.sigmoid(pos_scores[:, None] - neg_scores[None, :]) + 1e-8).mean()
                else:
                    loss = bce_loss(scores, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "two_tower.pt")
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"user_cat_cols": USER_CAT_COLS, "item_cat_cols": ITEM_CAT_COLS}, f, indent=2)


if __name__ == "__main__":
    train_model(TrainConfig())
