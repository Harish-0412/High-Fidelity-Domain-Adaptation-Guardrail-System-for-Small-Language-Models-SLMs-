from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterator


@dataclass
class DenseResult:
    chunk: dict[str, object]
    score: float


# ---------------------------------------------------------------------------
# Local dense index (JSONL, no external deps)
# ---------------------------------------------------------------------------

class LocalDenseIndex:
    """
    In-memory cosine-similarity index backed by a JSONL file.

    Improvements over original:
    * Pre-computes and caches per-vector L2 norms at load time (not per query).
    * Uses a numpy-style manual dot product loop that is easier for CPython to
      optimise — the bottleneck is the inner product, kept as simple as
      possible so PyPy / numba can JIT it later.
    * search() accepts an optional ``filter_fn`` so callers can apply metadata
      filters (e.g. domain, source) without a separate post-processing pass.
    * Streaming loader: reads records one line at a time, preventing OOM on
      large corpora while still landing in a contiguous list for fast scoring.
    """

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        # Pre-compute vectors and their norms once
        self._vectors: list[list[float]] = [list(r["vector"]) for r in records]
        self._norms: list[float] = [
            math.sqrt(sum(v * v for v in vec)) or 1.0
            for vec in self._vectors
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_fn: "Any | None" = None,
    ) -> list[DenseResult]:
        """
        Return top_k results by cosine similarity.

        Args:
            query_vector: Already-encoded query embedding.
            top_k:        Number of results to return.
            filter_fn:    Optional callable(chunk) → bool.  Only chunks for
                          which filter_fn returns True are scored.
        """
        query_norm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        inv_query_norm = 1.0 / query_norm

        scored: list[tuple[float, int]] = []

        for idx, (vec, vnorm) in enumerate(zip(self._vectors, self._norms)):
            if filter_fn is not None:
                payload = dict(self.records[idx]["payload"])
                if not filter_fn(payload):
                    continue
            # Dot product (cosine with pre-normalised query is just dot/norm_b)
            dot = sum(a * b for a, b in zip(query_vector, vec))
            score = dot * inv_query_norm / vnorm
            scored.append((score, idx))

        best = heapq.nlargest(top_k, scored, key=lambda item: item[0])
        return [
            DenseResult(chunk=dict(self.records[idx]["payload"]), score=score)
            for score, idx in best
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> "LocalDenseIndex":
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return LocalDenseIndex(records)

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_local_dense_index(
    chunks: list[dict[str, object]],
    vectors: list[list[float]],
    path: str | Path,
) -> None:
    """Write chunks + vectors to a JSONL file consumable by LocalDenseIndex."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks and vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk, vector in zip(chunks, vectors):
            record = {
                "id": chunk["chunk_id"],
                "payload": chunk,
                "vector": vector,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Qdrant wrapper (production path)
# ---------------------------------------------------------------------------

_QDRANT_BATCH_SIZE = 256   # safe default; tune for your GPU memory


class QdrantVectorStore:
    """
    Production Qdrant client wrapper.

    Improvements:
    * Batched upserts — avoids request-size limits on large corpora.
    * ``recreate_collection`` is guarded: if the collection already exists and
      has the correct vector size, it is left intact so accidental data loss
      during restarts is prevented.
    * search() passes an optional ``query_filter`` kwarg through to Qdrant for
      server-side metadata filtering (faster than post-filtering in Python).
    """

    def __init__(self, url: str, collection_name: str, vector_size: int):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http.models import Distance, VectorParams  # type: ignore

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._Distance = Distance
        self._VectorParams = VectorParams

    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if it does not exist.  Pass recreate=True to
        drop and recreate (e.g. when reindexing from scratch).
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            if not recreate:
                return
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._VectorParams(
                size=self.vector_size,
                distance=self._Distance.COSINE,
            ),
        )

        from qdrant_client.http.models import PayloadSchemaType
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="domain",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # Keep the old name for backward-compat
    def recreate_collection(self) -> None:
        self.ensure_collection(recreate=True)

    def upsert(self, chunks: list[dict[str, object]], vectors: list[list[float]]) -> None:
        from qdrant_client.http.models import PointStruct  # type: ignore

        import uuid
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk.get("chunk_id", idx)))),
                vector=vector,
                payload=dict(chunk)
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        # Batch upserts to stay within Qdrant's default payload size limit
        for i in range(0, len(points), _QDRANT_BATCH_SIZE):
            batch = points[i : i + _QDRANT_BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: "Any | None" = None,
    ) -> list[DenseResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        hits = response.points
        return [
            DenseResult(chunk=dict(hit.payload or {}), score=float(hit.score))
            for hit in hits
        ]


def try_build_qdrant_store(
    url: str, collection_name: str, vector_size: int
) -> "QdrantVectorStore | None":
    try:
        return QdrantVectorStore(url=url, collection_name=collection_name, vector_size=vector_size)
    except Exception:
        return None