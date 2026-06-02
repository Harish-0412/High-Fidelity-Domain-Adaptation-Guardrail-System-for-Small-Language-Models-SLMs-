from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from domain_slm_guardrails.core.domain_registry import get_domain_config
from domain_slm_guardrails.retrieval.bm25 import BM25Index
from domain_slm_guardrails.retrieval.embeddings import load_embedding_model
from domain_slm_guardrails.retrieval.vector_store import DenseResult, LocalDenseIndex


@dataclass
class HybridResult:
    chunk: dict[str, object]
    score: float
    dense_score: float
    bm25_score: float


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def merge_results(
    dense_results: list[DenseResult],
    bm25_results,
    top_k: int = 5,
    dense_weight: float = 0.6,
) -> list[HybridResult]:
    dense_norm = _normalize([result.score for result in dense_results])
    bm25_norm = _normalize([result.score for result in bm25_results])
    by_chunk_id: dict[str, HybridResult] = {}

    for result, score in zip(dense_results, dense_norm):
        chunk_id = str(result.chunk["chunk_id"])
        by_chunk_id[chunk_id] = HybridResult(
            chunk=result.chunk,
            score=dense_weight * score,
            dense_score=result.score,
            bm25_score=0.0,
        )

    for result, score in zip(bm25_results, bm25_norm):
        chunk_id = str(result.chunk["chunk_id"])
        lexical_score = (1.0 - dense_weight) * score
        if chunk_id in by_chunk_id:
            existing = by_chunk_id[chunk_id]
            by_chunk_id[chunk_id] = HybridResult(
                chunk=existing.chunk,
                score=existing.score + lexical_score,
                dense_score=existing.dense_score,
                bm25_score=result.score,
            )
        else:
            by_chunk_id[chunk_id] = HybridResult(
                chunk=result.chunk,
                score=lexical_score,
                dense_score=0.0,
                bm25_score=result.score,
            )

    merged = list(by_chunk_id.values())
    merged.sort(key=lambda result: result.score, reverse=True)
    return merged[:top_k]


class HybridRetriever:
    def __init__(
        self,
        dense_index: LocalDenseIndex,
        bm25_index: BM25Index,
        embedding_model_name: str = "local-hashing",
        embedding_dimension: int = 384,
    ):
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedding_model = load_embedding_model(
            embedding_model_name,
            dimension=embedding_dimension,
        )

    def search(self, query: str, top_k: int = 5, candidate_multiplier: int = 5) -> list[HybridResult]:
        candidate_k = max(top_k, top_k * candidate_multiplier)
        query_vector = self.embedding_model.encode([query])[0]
        dense_results = self.dense_index.search(query_vector, top_k=candidate_k)
        bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        return merge_results(dense_results, bm25_results, top_k=top_k)


@lru_cache(maxsize=16)
def load_hybrid_retriever(domain_id: str) -> HybridRetriever:
    domain = get_domain_config(domain_id)
    bm25 = BM25Index.load(domain.bm25_path)
    dense = LocalDenseIndex.load(domain.dense_index_path)
    return HybridRetriever(
        dense_index=dense,
        bm25_index=bm25,
        embedding_model_name=str(domain.settings["embedding_model"]),
        embedding_dimension=int(domain.settings["dense_vector_size"]),
    )
