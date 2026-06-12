from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback structures if flashrank is missing
@dataclass
class RerankResult:
    chunk: dict[str, object]
    score: float
    original_rank: int


class CrossEncoderReranker:
    """
    Reranker that uses FlashRank to avoid the heavy PyTorch dependency 
    in the runtime container. FlashRank uses ONNXRuntime under the hood.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or "/tmp/flashrank_cache"
        self._ranker = None
        self._initialize_ranker()

    def _initialize_ranker(self):
        try:
            from flashrank import Ranker
            # ensure cache directory exists
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        except ImportError:
            logger.warning("FlashRank not installed. Reranking will act as a pass-through.")
            self._ranker = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Re-score and re-order the given candidate chunks using the cross-encoder.
        """
        if not candidates:
            return []
            
        # Pass-through if ranker failed to load
        if self._ranker is None:
            return [
                RerankResult(chunk=c, score=1.0 - (i * 0.01), original_rank=i+1) 
                for i, c in enumerate(candidates[:top_k])
            ]
            
        from flashrank import RerankRequest

        # Format candidates for FlashRank
        passages = []
        for i, chunk in enumerate(candidates):
            passages.append({
                "id": str(chunk.get("chunk_id", i)),
                "text": str(chunk.get("text", "")),
                "meta": chunk
            })
            
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        
        # Results are returned in sorted order by FlashRank
        # format: [{"id": "...", "text": "...", "meta": {...}, "score": 0.98}, ...]
        reranked = []
        for i, res in enumerate(results[:top_k]):
            reranked.append(
                RerankResult(
                    chunk=res["meta"],
                    score=float(res.get("score", 0.0)),
                    original_rank=i+1 # Not strictly original rank, but final rank
                )
            )
            
        return reranked
