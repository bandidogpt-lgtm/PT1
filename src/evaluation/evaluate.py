"""Evaluate Two-Tower model with Recall@K and NDCG@K."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.models.two_tower import TowerConfig, TwoTowerModel

DATA_DIR = Path("/workspace/PT1/data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

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


def _load_encoders() -> Dict[str, Dict[str, int]]:
    with open(PROCESSED_DIR / "encoders.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model(encoders: Dict[str, Dict[str, int]]) -> TwoTowerModel:
    user_cardinalities = {col: len(encoders[col]) for col in USER_CAT_COLS}
    item_cardinalities = {col: len(encoders[col]) for col in ITEM_CAT_COLS}
    return TwoTowerModel(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        numeric_user_dim=len(USER_NUM_COLS),
        numeric_item_dim=len(ITEM_NUM_COLS),
        config=TowerConfig(),
    )


def evaluate(k: int = 5) -> None:
    data = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    test = data[data["split"] == "test"]
    productos = data.drop_duplicates("producto_id")[ITEM_CAT_COLS + ITEM_NUM_COLS]
    encoders = _load_encoders()

    # Load trained Two-Tower weights.
    model = _build_model(encoders)
    state_dict = torch.load(MODEL_DIR / "two_tower.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    recalls: List[float] = []
    ndcgs: List[float] = []

    # Evaluate per client to compute ranking metrics.
    for cliente_id, group in test.groupby("cliente_id"):
        user_row = group.iloc[0]
        user_cat = {col: torch.tensor([int(user_row[col])]) for col in USER_CAT_COLS}
        user_num = torch.tensor([user_row[USER_NUM_COLS].values.astype(np.float32)])

        scores = []
        for _, prod in productos.iterrows():
            item_cat = {col: torch.tensor([int(prod[col])]) for col in ITEM_CAT_COLS}
            item_num = torch.tensor([prod[ITEM_NUM_COLS].values.astype(np.float32)])
            with torch.no_grad():
                score = model(user_cat, user_num, item_cat, item_num).item()
            scores.append((prod["producto_id"], score))

        # Rank products by model score and compare to actual purchases.
        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [prod_id for prod_id, _ in scores]
        relevant = group[group["label_compra"] == 1]["producto_id"].tolist()

        recalls.append(recall_at_k(relevant, recommended, k))
        ndcgs.append(ndcg_at_k(relevant, recommended, k))

    print(f"Recall@{k}: {np.mean(recalls):.4f}")
    print(f"NDCG@{k}: {np.mean(ndcgs):.4f}")


if __name__ == "__main__":
    evaluate(k=5)
