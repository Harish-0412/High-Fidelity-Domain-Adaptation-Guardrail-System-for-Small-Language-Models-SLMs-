from __future__ import annotations

import re
from typing import List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)


def mmr_rerank(
    chunks: List[dict],
    query: str,
    lambda_param: float = 0.6,
    top_k: int = 5,
) -> List[dict]:
    """
    Maximal Marginal Relevance (MMR) reranking to balance relevance and diversity.
    """
    if not chunks:
        return []

    query_tokens = set(_tokenize(query))
    chunk_tokens_list = [set(_tokenize(chunk.get("text", ""))) for chunk in chunks]
    scores = [chunk.get("score", 1.0) for chunk in chunks]
    selected = []
    remaining = list(range(len(chunks)))

    for _ in range(min(top_k, len(chunks))):
        best_score = -1.0
        best_idx = -1

        for idx in remaining:
            relevance = scores[idx]
            max_similarity = max(
                _jaccard_similarity(chunk_tokens_list[idx], chunk_tokens_list[s])
                for s in selected
            ) if selected else 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected]
