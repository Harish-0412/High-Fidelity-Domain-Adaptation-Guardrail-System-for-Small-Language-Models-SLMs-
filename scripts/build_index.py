from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.core.domain_registry import get_domain_config
from ingestion.pipeline import read_chunks_jsonl
from retrieval.bm25 import build_bm25_index
from retrieval.embeddings import load_embedding_model
from retrieval.vector_store import try_build_qdrant_store, write_local_dense_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and dense indexes for a domain.")
    parser.add_argument("--domain", required=True, help="Domain id, for example medical_prescription")
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Only write the local dense index and BM25 index; skip Qdrant upsert.",
    )
    args = parser.parse_args()

    domain = get_domain_config(args.domain)
    chunks = read_chunks_jsonl(domain.chunks_path)
    domain.index_path.mkdir(parents=True, exist_ok=True)

    bm25 = build_bm25_index(chunks)
    bm25.save(domain.bm25_path)

    embedding_model_name = str(domain.settings["embedding_model"])
    vector_size = int(domain.settings["dense_vector_size"])
    embedder = load_embedding_model(embedding_model_name, dimension=vector_size)
    vectors = embedder.encode([str(chunk["text"]) for chunk in chunks])
    write_local_dense_index(chunks, vectors, domain.dense_index_path)

    qdrant_status = "skipped"
    if not args.no_qdrant:
        store = try_build_qdrant_store(
            url=str(domain.settings["qdrant_url"]),
            collection_name=domain.index_name,
            vector_size=len(vectors[0]) if vectors else vector_size,
        )
        if store is not None:
            try:
                store.recreate_collection()
                store.upsert(chunks, vectors)
                qdrant_status = "upserted"
            except Exception as exc:
                qdrant_status = f"unavailable ({exc.__class__.__name__})"
        else:
            qdrant_status = "unavailable"

    print(f"Built BM25 index: {domain.bm25_path}")
    print(f"Built local dense index: {domain.dense_index_path}")
    print(f"Qdrant status: {qdrant_status}")


if __name__ == "__main__":
    main()
