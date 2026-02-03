"""Build training dataset with engineered features and negative sampling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("/workspace/PT1/data")
OUTPUT_DIR = DATA_DIR / "processed"


@dataclass
class DatasetConfig:
    future_window_days: int = 30
    train_end_date: str = "2024-03-01"
    val_end_date: str = "2024-04-01"
    negative_samples_per_positive: int = 3
    random_seed: int = 42


def _parse_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df


def _compute_client_features(ventas: pd.DataFrame) -> pd.DataFrame:
    # Aggregates user history to capture purchase frequency and average ticket size.
    resumen = ventas.groupby("cliente_id").agg(
        frecuencia_compra=("venta_id", "nunique"),
        ticket_promedio=("monto_total", "mean"),
    )
    return resumen.reset_index()


def _compute_context_features(ventas: pd.DataFrame) -> pd.DataFrame:
    # Adds temporal signals to model seasonality and weekly patterns.
    ventas = ventas.copy()
    ventas["mes"] = ventas["fecha_venta"].dt.month
    ventas["semana_anio"] = ventas["fecha_venta"].dt.isocalendar().week.astype(int)
    ventas["cerca_feriado_7d"] = ventas["fecha_venta"].dt.dayofweek.isin([4, 5, 6]).astype(int)
    return ventas[["venta_id", "mes", "semana_anio", "cerca_feriado_7d"]]


def _compute_product_features(productos: pd.DataFrame, detalle: pd.DataFrame) -> pd.DataFrame:
    # Derives freshness and rotation metrics for business-aware ranking.
    productos = productos.copy()
    productos["fecha_min_caducidad"] = pd.to_datetime(productos["fecha_min_caducidad"])
    productos["dias_para_vencer"] = (productos["fecha_min_caducidad"] - pd.Timestamp.today()).dt.days
    productos["indice_proximo_vencer"] = (
        1 - (productos["dias_para_vencer"] - productos["dias_para_vencer"].min())
        / (productos["dias_para_vencer"].max() - productos["dias_para_vencer"].min() + 1)
    ).clip(0, 1)

    detalle_agg = detalle.groupby("producto_id").agg(cogs=("subtotal", "sum"))
    inventario_promedio = productos.set_index("producto_id")["stock"].replace(0, 1)
    rotacion = detalle_agg["cogs"].div(inventario_promedio, fill_value=0)
    productos = productos.set_index("producto_id")
    productos["rotacion_producto"] = rotacion
    productos["rotacion_producto"] = productos["rotacion_producto"].fillna(0)
    productos["indice_baja_rotacion"] = (
        1 - (productos["rotacion_producto"] - productos["rotacion_producto"].min())
        / (productos["rotacion_producto"].max() - productos["rotacion_producto"].min() + 1)
    ).clip(0, 1)

    return productos.reset_index()


def _build_labels(ventas: pd.DataFrame, detalle: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    # Label positive interactions and apply a temporal split.
    base = ventas[[\"venta_id\", \"cliente_id\", \"fecha_venta\", \"canal_venta\"]].copy()
    base = base.merge(detalle[["venta_id", "producto_id"]], on="venta_id", how="left")

    train_end = pd.Timestamp(cfg.train_end_date)
    val_end = pd.Timestamp(cfg.val_end_date)

    base["split"] = np.where(
        base["fecha_venta"] <= train_end,
        "train",
        np.where(base["fecha_venta"] <= val_end, "val", "test"),
    )
    base["label_compra"] = 1
    return base


def _negative_sampling(
    positives: pd.DataFrame,
    productos: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    # Sample negatives from products not purchased by the client.
    rng = np.random.default_rng(cfg.random_seed)
    all_products = productos["producto_id"].unique()
    negativos = []

    for cliente_id, group in positives.groupby("cliente_id"):
        comprados = set(group["producto_id"].unique())
        candidatos = np.array([p for p in all_products if p not in comprados])
        if candidatos.size == 0:
            continue
        for _, row in group.iterrows():
            n_samples = cfg.negative_samples_per_positive
            sampled = rng.choice(candidatos, size=min(n_samples, len(candidatos)), replace=False)
            for prod_id in sampled:
                negativos.append(
                    {
                        "cliente_id": cliente_id,
                        "producto_id": prod_id,
                        "fecha_venta": row["fecha_venta"],
                        "split": row["split"],
                        "label_compra": 0,
                    }
                )
    return pd.DataFrame(negativos)


def build_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    clientes = pd.read_csv(DATA_DIR / "clientes.csv")
    productos = pd.read_csv(DATA_DIR / "productos.csv")
    ventas = pd.read_csv(DATA_DIR / "ventas.csv")
    detalle = pd.read_csv(DATA_DIR / "detalle_venta.csv")

    ventas = _parse_dates(ventas, ["fecha_venta"])

    client_features = _compute_client_features(ventas)
    context_features = _compute_context_features(ventas)
    product_features = _compute_product_features(productos, detalle)

    positivos = _build_labels(ventas, detalle, cfg)
    negativos = _negative_sampling(positivos, productos, cfg)
    data = pd.concat([positivos, negativos], ignore_index=True)

    data = data.merge(clientes, on="cliente_id", how="left")
    data = data.merge(client_features, on="cliente_id", how="left")
    data = data.merge(context_features, on="venta_id", how="left")
    data = data.merge(product_features, on="producto_id", how="left")

    data["fecha_venta"] = pd.to_datetime(data["fecha_venta"]).dt.date.astype(str)

    categorical_cols = ["cliente_id", "rubro_cliente", "producto_id", "categoria_producto", "canal_venta"]
    numerical_cols = [
        "frecuencia_compra",
        "ticket_promedio",
        "mes",
        "semana_anio",
        "cerca_feriado_7d",
        "precio_unitario",
        "costo_unitario",
        "stock",
        "dias_para_vencer",
        "indice_proximo_vencer",
        "rotacion_producto",
        "indice_baja_rotacion",
    ]

    encoders: Dict[str, Dict[str, int]] = {}
    # Encode categorical variables to integer IDs for embeddings.
    for col in categorical_cols:
        categories = sorted(data[col].fillna("desconocido").unique())
        encoders[col] = {cat: idx for idx, cat in enumerate(categories)}
        data[col] = data[col].fillna("desconocido").map(encoders[col])

    # Fill missing numeric values to keep training stable.
    data[numerical_cols] = data[numerical_cols].fillna(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "dataset.csv", index=False)
    with open(OUTPUT_DIR / "encoders.json", "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2, ensure_ascii=False)

    return data, encoders


if __name__ == "__main__":
    build_dataset(DatasetConfig())
