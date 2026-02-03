"""Ranking metrics for recommendation evaluation."""
from __future__ import annotations

from typing import List

import numpy as np


def recall_at_k(relevant: List[int], recommended: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / max(len(relevant_set), 1)


def ndcg_at_k(relevant: List[int], recommended: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(idx + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
