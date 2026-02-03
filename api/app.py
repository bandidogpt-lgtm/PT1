"""Simple API for product recommendations."""
from __future__ import annotations

from fastapi import FastAPI

from src.inference.recommend import RerankConfig, recomendar

app = FastAPI(title="Recomendador HORECA")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{cliente_id}")
def recommend(cliente_id: str, top_k: int = 5, alpha: float = 0.2, beta: float = 0.3) -> dict:
    ranking = recomendar(cliente_id, top_k=top_k, rerank_cfg=RerankConfig(alpha=alpha, beta=beta))
    return {"cliente_id": cliente_id, "recomendaciones": ranking.to_dict(orient="records")}
