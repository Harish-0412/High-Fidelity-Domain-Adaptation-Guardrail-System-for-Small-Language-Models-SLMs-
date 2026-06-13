from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CrossEncoder = None
CROSS_ENCODER_AVAILABLE = False

try:
    # Try importing with keras compatibility fix
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except (ImportError, ValueError):
    try:
        # Try without the fix
        from sentence_transformers import CrossEncoder
        CROSS_ENCODER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"CrossEncoder not available: {e}. Install via pip install sentence-transformers")
        pass


@dataclass
class RerankedResult:
    chunk: dict[str, object]
    score: float
    original_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder reranker that re-scores retrieval results using semantic similarity
    between the query and each retrieved chunk.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
    ):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("CrossEncoder not available. Install via pip install sentence-transformers")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """
        Rerank chunks based on cross-encoder similarity to the query.
        """
        if not chunks:
            return []

        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored[:top_k]]
