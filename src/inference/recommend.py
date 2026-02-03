"""Inference utilities and business-aligned re-ranking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.models.two_tower import TowerConfig, TwoTowerModel

DATA_DIR = Path("/workspace/PT1/data")
MODEL_DIR = Path("/workspace/PT1/data/models")
PROCESSED_DIR = DATA_DIR / "processed"

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


@dataclass
class RerankConfig:
    alpha: float = 0.2
    beta: float = 0.3


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


def _prepare_user_features(cliente_id: str, data: pd.DataFrame) -> pd.Series:
    user_row = data[data["cliente_id"] == cliente_id].iloc[0]
    return user_row


def recomendar(cliente_id: str, top_k: int = 5, rerank_cfg: RerankConfig | None = None) -> pd.DataFrame:
    rerank_cfg = rerank_cfg or RerankConfig()
    encoders = _load_encoders()
    dataset = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    productos = pd.read_csv(DATA_DIR / "productos.csv")
    productos = productos.merge(dataset.drop_duplicates("producto_id")[ITEM_NUM_COLS + ITEM_CAT_COLS], on="producto_id", how="left")

    # Use a representative user row (latest seen in dataset) to form features.
    user_row = _prepare_user_features(cliente_id, dataset)

    model = _build_model(encoders)
    state_dict = torch.load(MODEL_DIR / "two_tower.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    user_cat = {col: torch.tensor([int(user_row[col])]) for col in USER_CAT_COLS}
    user_num = torch.tensor([user_row[USER_NUM_COLS].values.astype(np.float32)])

    # Score every candidate product and apply business re-ranking.
    scores = []
    for _, prod in productos.iterrows():
        item_cat = {col: torch.tensor([int(prod[col])]) for col in ITEM_CAT_COLS}
        item_num = torch.tensor([prod[ITEM_NUM_COLS].values.astype(np.float32)])
        with torch.no_grad():
            score = model(user_cat, user_num, item_cat, item_num).item()
        # Combine model affinity with low-rotation and near-expiry incentives.
        score_final = score + rerank_cfg.alpha * prod["indice_baja_rotacion"] + rerank_cfg.beta * prod["indice_proximo_vencer"]
        scores.append({"producto_id": prod["producto_id"], "score_dl": score, "score_final": score_final})

    ranking = pd.DataFrame(scores).sort_values("score_final", ascending=False).head(top_k)
    ranking = ranking.merge(productos[["producto_id", "categoria_producto", "precio_unitario"]], on="producto_id", how="left")
    return ranking


if __name__ == "__main__":
    print(recomendar("C001", top_k=5))
