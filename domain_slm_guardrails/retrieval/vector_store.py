from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any

from domain_slm_guardrails.retrieval.embeddings import cosine_similarity


@dataclass
class DenseResult:
    chunk: dict[str, object]
    score: float


class LocalDenseIndex:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self._vectors = [list(record["vector"]) for record in records]
        self._norms = [math.sqrt(sum(value * value for value in vector)) or 1.0 for vector in self._vectors]

    def search(self, query_vector: list[float], top_k: int = 5) -> list[DenseResult]:
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        scored = []
        for idx, vector in enumerate(self._vectors):
            score = sum(left * right for left, right in zip(query_vector, vector)) / (
                query_norm * self._norms[idx]
            )
            scored.append((score, idx))
        best = heapq.nlargest(top_k, scored, key=lambda item: item[0])
        return [
            DenseResult(chunk=dict(self.records[idx]["payload"]), score=score)
            for score, idx in best
        ]

    @staticmethod
    def load(path: str | Path) -> "LocalDenseIndex":
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return LocalDenseIndex(records)


def write_local_dense_index(
    chunks: list[dict[str, object]],
    vectors: list[list[float]],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk, vector in zip(chunks, vectors):
            handle.write(
                json.dumps(
                    {
                        "id": chunk["chunk_id"],
                        "payload": chunk,
                        "vector": vector,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, vector_size: int):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http.models import Distance, VectorParams  # type: ignore

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.Distance = Distance
        self.VectorParams = VectorParams

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=self.VectorParams(size=self.vector_size, distance=self.Distance.COSINE),
        )

    def upsert(self, chunks: list[dict[str, object]], vectors: list[list[float]]) -> None:
        from qdrant_client.http.models import PointStruct  # type: ignore

        points = [
            PointStruct(
                id=idx,
                vector=vector,
                payload=dict(chunk),
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[DenseResult]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [DenseResult(chunk=dict(hit.payload or {}), score=float(hit.score)) for hit in hits]


def try_build_qdrant_store(url: str, collection_name: str, vector_size: int) -> QdrantVectorStore | None:
    try:
        return QdrantVectorStore(url=url, collection_name=collection_name, vector_size=vector_size)
    except Exception:
        return None
