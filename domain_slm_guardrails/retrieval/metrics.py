from __future__ import annotations

import math
from typing import List, Optional


def dcg(scores: List[float], k: Optional[int] = None) -> float:
    """
    Discounted Cumulative Gain (DCG) at k.
    """
    if k is None:
        k = len(scores)
    scores = scores[:k]
    return sum(score / math.log2(i + 2) for i, score in enumerate(scores))


def ndcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    if not relevance_scores:
        return 0.0
    ideal_scores = sorted(relevance_scores, reverse=True)
    dcg_value = dcg(relevance_scores, k)
    idcg_value = dcg(ideal_scores, k)
    return dcg_value / idcg_value if idcg_value > 0 else 0.0


def mrr(relevance: List[int]) -> float:
    """
    Mean Reciprocal Rank (MRR) of first relevant item.
    """
    for i, is_relevant in enumerate(relevance, 1):
        if is_relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall at k: fraction of relevant items that are in top-k retrieved.
    """
    if not relevant_ids:
        return 1.0
    retrieved_top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(retrieved_top_k) & set(relevant_ids))
    return relevant_in_top_k / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision at k: fraction of top-k retrieved items that are relevant.
    """
    if k == 0:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(retrieved_top_k) & set(relevant_ids))
    return relevant_in_top_k / k


def map_score(relevance_lists: List[List[int]]) -> float:
    """
    Mean Average Precision (MAP) over multiple queries.
    """
    average_precisions = []
    for relevance in relevance_lists:
        precisions = []
        num_relevant = 0
        for i, is_relevant in enumerate(relevance, 1):
            if is_relevant:
                num_relevant += 1
                precisions.append(num_relevant / i)
        if precisions:
            average_precisions.append(sum(precisions) / len(precisions))
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
