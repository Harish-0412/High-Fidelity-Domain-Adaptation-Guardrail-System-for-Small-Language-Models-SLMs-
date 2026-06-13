from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from domain_slm_guardrails.core.domain_registry import get_domain_config
from domain_slm_guardrails.retrieval.bm25 import BM25Index
from domain_slm_guardrails.retrieval.embeddings import load_embedding_model
from domain_slm_guardrails.retrieval.vector_store import DenseResult, LocalDenseIndex
from domain_slm_guardrails.retrieval.preprocessor import QueryPreprocessor, ProcessedQuery
from domain_slm_guardrails.retrieval.reranker import CrossEncoderReranker
from domain_slm_guardrails.retrieval.diversity import mmr_rerank


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    chunk: dict[str, object]
    score: float
    dense_score: float
    bm25_score: float
    rrf_rank_dense: int = 0
    rrf_rank_bm25: int = 0


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _min_max_normalize(scores: list[float]) -> list[float]:
    """Classic min-max normalisation.  Returns uniform 1.0 when all scores equal."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 if s > 0 else 0.0 for s in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, object]]],   # each: [(chunk_id, chunk), …]
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF) score for each chunk_id.

    RRF score = Σ  1 / (k + rank_i)

    k=60 is the standard value from the original Cormack et al. paper.  It
    smooths out rank differences between the two retrieval channels without
    requiring score calibration.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Merge / fusion
# ---------------------------------------------------------------------------

def merge_results(
    dense_results: list[DenseResult],
    bm25_results,                       # list[BM25Result]
    top_k: int = 5,
    dense_weight: float = 0.6,
    use_rrf: bool = True,
    rrf_k: int = 60,
) -> list[HybridResult]:
    """
    Fuse dense and BM25 result lists into a single ranked list.

    Strategy selection:
    * ``use_rrf=True``  (default) — RRF fusion.  Rank-based; robust to score
      scale differences between the two retrievers.  Recommended unless you
      have carefully calibrated score distributions.
    * ``use_rrf=False`` — weighted score fusion (original behaviour).  Useful
      when dense and BM25 scores are well-calibrated and you want to tune the
      relative contribution via ``dense_weight``.
    """
    if use_rrf:
        return _merge_rrf(dense_results, bm25_results, top_k=top_k, rrf_k=rrf_k)
    return _merge_weighted(dense_results, bm25_results, top_k=top_k, dense_weight=dense_weight)


def _merge_rrf(
    dense_results: list[DenseResult],
    bm25_results,
    top_k: int,
    rrf_k: int,
) -> list[HybridResult]:
    # Build ranked lists: [(chunk_id, chunk), …]
    dense_ranked = [(str(r.chunk["chunk_id"]), r.chunk) for r in dense_results]
    bm25_ranked  = [(str(r.chunk["chunk_id"]), r.chunk) for r in bm25_results]

    rrf_scores = _reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=rrf_k)

    # Collect raw scores for diagnostics
    dense_score_map = {str(r.chunk["chunk_id"]): r.score for r in dense_results}
    bm25_score_map  = {str(r.chunk["chunk_id"]): r.score for r in bm25_results}

    # Build rank maps for transparency fields
    dense_rank_map = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_ranked, 1)}
    bm25_rank_map  = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_ranked, 1)}

    # Merge chunk lookup
    chunk_map: dict[str, object] = {}
    for chunk_id, chunk in dense_ranked + bm25_ranked:
        chunk_map.setdefault(chunk_id, chunk)

    merged = [
        HybridResult(
            chunk=chunk_map[cid],           # type: ignore[arg-type]
            score=rrf_score,
            dense_score=dense_score_map.get(cid, 0.0),
            bm25_score=bm25_score_map.get(cid, 0.0),
            rrf_rank_dense=dense_rank_map.get(cid, 0),
            rrf_rank_bm25=bm25_rank_map.get(cid, 0),
        )
        for cid, rrf_score in rrf_scores.items()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:top_k]


def _merge_weighted(
    dense_results: list[DenseResult],
    bm25_results,
    top_k: int,
    dense_weight: float,
) -> list[HybridResult]:
    """Original weighted-score merge, kept for opt-out use."""
    dense_norm = _min_max_normalize([r.score for r in dense_results])
    bm25_norm  = _min_max_normalize([r.score for r in bm25_results])

    by_chunk_id: dict[str, HybridResult] = {}

    for result, score in zip(dense_results, dense_norm):
        cid = str(result.chunk["chunk_id"])
        by_chunk_id[cid] = HybridResult(
            chunk=result.chunk,
            score=dense_weight * score,
            dense_score=result.score,
            bm25_score=0.0,
        )

    for result, score in zip(bm25_results, bm25_norm):
        cid = str(result.chunk["chunk_id"])
        lexical = (1.0 - dense_weight) * score
        if cid in by_chunk_id:
            ex = by_chunk_id[cid]
            by_chunk_id[cid] = HybridResult(
                chunk=ex.chunk,
                score=ex.score + lexical,
                dense_score=ex.dense_score,
                bm25_score=result.score,
            )
        else:
            by_chunk_id[cid] = HybridResult(
                chunk=result.chunk,
                score=lexical,
                dense_score=0.0,
                bm25_score=result.score,
            )

    merged = sorted(by_chunk_id.values(), key=lambda r: r.score, reverse=True)
    return merged[:top_k]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines dense (vector) search with BM25 lexical search.

    Improvements:
    * RRF fusion by default — no score-calibration needed.
    * Query expansion: appends stemmed synonyms extracted from the query so
      BM25 recall improves on medical abbreviation queries like "CPT 99214".
    * ``candidate_multiplier`` is now a per-retriever setting, not a search arg.
    * Thread-safe: encoding and index search are stateless; the only shared
      mutable state is the lru_cache at module level (protected by GIL).
    * Query preprocessor: cleans and expands queries.
    * Cross-encoder reranking: re-scores results for better relevance.
    * MMR diversity: balances relevance and diverse information.
    """

    def __init__(
        self,
        dense_index: LocalDenseIndex,
        bm25_index: BM25Index,
        embedding_model_name: str = "local-hashing",
        embedding_dimension: int = 384,
        candidate_multiplier: int = 5,
        use_rrf: bool = True,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        preprocessor: Optional[QueryPreprocessor] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.6,
    ):
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedding_model = load_embedding_model(
            embedding_model_name,
            dimension=embedding_dimension,
        )
        self.candidate_multiplier = candidate_multiplier
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.preprocessor = preprocessor
        self.reranker = reranker
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def search(
        self,
        query: str,
        top_k: int = 5,
        expanded_query: str | None = None,
        use_reranking: bool = True,
        use_mmr: bool | None = None,
    ) -> list[HybridResult]:
        """
        Retrieve top_k results for ``query``.

        Args:
            query:          Original user query.
            expanded_query: Optional rewritten/expanded query string.  If
                            provided, BM25 uses it while dense uses the
                            original (preserving semantic precision).
            use_reranking:  Whether to use cross-encoder reranking if available.
            use_mmr:        Whether to use MMR diversity, overrides instance-level setting.
        """
        # Process query with preprocessor if available
        processed_query: Optional[ProcessedQuery] = None
        if self.preprocessor:
            processed_query = self.preprocessor.preprocess(query)
            bm25_query = processed_query.expanded
        else:
            bm25_query = expanded_query if expanded_query else query

        candidate_k = max(top_k, top_k * self.candidate_multiplier)

        # Dense: encode original query for best semantic precision
        query_vector = self.embedding_model.encode([query])[0]
        dense_results = self.dense_index.search(query_vector, top_k=candidate_k)

        # BM25: use expanded query
        bm25_results = self.bm25_index.search(bm25_query, top_k=candidate_k)

        # Initial merge
        initial_results = merge_results(
            dense_results,
            bm25_results,
            top_k=candidate_k,
            dense_weight=self.dense_weight,
            use_rrf=self.use_rrf,
            rrf_k=self.rrf_k,
        )

        # Convert to chunks for further processing
        chunks = [result.chunk for result in initial_results]

        # Apply cross-encoder reranking if available
        if self.reranker and use_reranking:
            chunks = self.reranker.rerank(query, chunks, top_k=candidate_k)

        # Apply MMR diversity
        if use_mmr is None:
            use_mmr = self.use_mmr
        if use_mmr:
            chunks = mmr_rerank(chunks, query, lambda_param=self.mmr_lambda, top_k=top_k)

        # Return top_k
        # First, create a map from chunk_id to original HybridResult
        result_map = {str(r.chunk["chunk_id"]): r for r in initial_results}
        final_results = []
        for chunk in chunks[:top_k]:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in result_map:
                final_results.append(result_map[chunk_id])

        return final_results


# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------

# Thread lock ensures a cold-start race condition doesn't build two retrievers
_load_lock = threading.Lock()


@lru_cache(maxsize=16)
def load_hybrid_retriever(domain_id: str) -> HybridRetriever:
    """
    Load and cache a HybridRetriever for the given domain.

    The lru_cache makes subsequent calls free.  The thread lock ensures only
    one thread pays the construction cost on a cold start.
    """
    with _load_lock:
        # Check again inside lock to avoid double-build on concurrent cold start
        # (lru_cache itself is not re-entrant-safe during construction)
        domain = get_domain_config(domain_id)
        bm25 = BM25Index.load(domain.bm25_path)
        dense = LocalDenseIndex.load(domain.dense_index_path)

        settings = domain.settings
        
        # Preprocessor
        preprocessor = None
        if bool(settings.get("use_query_preprocessor", True)):
            preprocessor = QueryPreprocessor(domain=domain_id)
            
        # Reranker
        reranker = None
        if bool(settings.get("use_cross_encoder_reranker", False)):
            cross_encoder_model = str(settings.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"))
            try:
                reranker = CrossEncoderReranker(model_name=cross_encoder_model)
            except Exception:
                pass
                
        return HybridRetriever(
            dense_index=dense,
            bm25_index=bm25,
            embedding_model_name=str(settings["embedding_model"]),
            embedding_dimension=int(settings["dense_vector_size"]),
            candidate_multiplier=int(settings.get("candidate_multiplier", 5)),
            use_rrf=False,
            rrf_k=int(settings.get("rrf_k", 60)),
            dense_weight=0.0,  # Only use BM25 scores now!
            preprocessor=preprocessor,
            reranker=reranker,
            use_mmr=bool(settings.get("use_mmr", False)),
            mmr_lambda=float(settings.get("mmr_lambda", 0.6)),
        )