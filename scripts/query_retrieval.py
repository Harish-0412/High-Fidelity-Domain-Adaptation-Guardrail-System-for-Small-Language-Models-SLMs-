from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.core.domain_registry import get_domain_config
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import LocalDenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a domain's hybrid retrieval index.")
    parser.add_argument("--domain", required=True, help="Domain id, for example medical_prescription")
    parser.add_argument("--query", required=True, help="User query to retrieve evidence for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    args = parser.parse_args()

    domain = get_domain_config(args.domain)
    bm25 = BM25Index.load(domain.bm25_path)
    dense = LocalDenseIndex.load(domain.dense_index_path)
    retriever = HybridRetriever(
        dense_index=dense,
        bm25_index=bm25,
        embedding_model_name=str(domain.settings["embedding_model"]),
        embedding_dimension=int(domain.settings["dense_vector_size"]),
    )

    results = []
    for result in retriever.search(args.query, top_k=args.top_k):
        chunk = result.chunk
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": round(result.score, 4),
                "dense_score": round(result.dense_score, 4),
                "bm25_score": round(result.bm25_score, 4),
                "source_id": chunk["source_id"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
