from __future__ import annotations

from retrieval.bm25 import build_bm25_index
from retrieval.embeddings import HashingEmbeddingModel
from retrieval.hybrid import merge_results
from retrieval.vector_store import DenseResult, LocalDenseIndex


def _chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "modifier 25 separately identifiable evaluation management service",
            "source_id": "manual",
            "page": 1,
        },
        {
            "chunk_id": "c2",
            "text": "claim review documentation supports cpt 99214",
            "source_id": "manual",
            "page": 2,
        },
    ]


def test_bm25_returns_results():
    index = build_bm25_index(_chunks())
    results = index.search("modifier 25", top_k=1)
    assert results
    assert results[0].chunk["chunk_id"] == "c1"


def test_dense_retrieval_returns_results():
    chunks = _chunks()
    embedder = HashingEmbeddingModel(dimension=64)
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    dense = LocalDenseIndex(
        [
            {"id": chunk["chunk_id"], "payload": chunk, "vector": vector}
            for chunk, vector in zip(chunks, vectors)
        ]
    )
    query_vector = embedder.encode(["modifier 25"])[0]
    results = dense.search(query_vector, top_k=1)
    assert results
    assert results[0].chunk["chunk_id"] == "c1"


def test_hybrid_merges_duplicate_chunk_ids():
    chunks = _chunks()
    dense_results = [DenseResult(chunk=chunks[0], score=0.8), DenseResult(chunk=chunks[1], score=0.1)]
    bm25 = build_bm25_index(chunks)
    bm25_results = bm25.search("modifier 25", top_k=2)

    merged = merge_results(dense_results, bm25_results, top_k=2)
    assert len({result.chunk["chunk_id"] for result in merged}) == len(merged)
    assert merged[0].chunk["chunk_id"] == "c1"

