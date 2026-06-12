#!/usr/bin/env python3
"""
End-to-end indexing pipeline.
Workflow: medical documents -> load_document() -> chunk_page() -> BAAI/bge-small-en-v1.5 -> QdrantVectorStore.upsert()
"""

import argparse
import sys
from pathlib import Path

# Add project root to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from services.core.domain_registry import get_domain_config
from ingestion.pipeline import run_ingestion
from retrieval.embeddings import load_embedding_model
from retrieval.vector_store import try_build_qdrant_store

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Indexing Pipeline for Medical Documents")
    parser.add_argument(
        "--domain", 
        default="medical_prescription", 
        help="Domain to index (default: medical_prescription)"
    )
    parser.add_argument(
        "--qdrant-url", 
        default="http://localhost:6333", 
        help="Qdrant URL (default: http://localhost:6333)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64, 
        help="Embedding batch size (default: 64)"
    )
    parser.add_argument(
        "--recreate", 
        action="store_true", 
        help="Recreate the Qdrant collection if it exists"
    )
    args = parser.parse_args()

    print(f"==================================================")
    print(f" 🚀 Medical Document Indexing Pipeline")
    print(f"==================================================")

    print(f"\n[1/4] Loading configuration for domain '{args.domain}'...")
    try:
        domain = get_domain_config(args.domain)
    except Exception as e:
        print(f"Error loading domain: {e}")
        sys.exit(1)
        
    print(f"      Corpus Path: {domain.corpus_path}")

    print(f"\n[2/4] Running document ingestion (Load & Chunk)...")
    chunks = run_ingestion(domain)
    if not chunks:
        print(f"      No documents found in {domain.corpus_path}. Exiting.")
        return
    print(f"      Successfully extracted {len(chunks)} chunks.")

    print(f"\n[3/4] Initializing Qdrant Vector Store...")
    store = try_build_qdrant_store(
        url=args.qdrant_url,
        collection_name=domain.index_name,
        vector_size=384
    )
    if not store:
        print("      Failed to connect to Qdrant Vector Store.")
        sys.exit(1)
        
    print(f"      Ensuring collection '{domain.index_name}' exists (recreate={args.recreate})...")
    store.ensure_collection(recreate=args.recreate)

    print(f"\n[4/4] Generating embeddings and upserting to Qdrant...")
    print(f"      Model: BAAI/bge-small-en-v1.5")
    
    model = load_embedding_model("BAAI/bge-small-en-v1.5", dimension=384)
    # The batching for Qdrant is already handled inside store.upsert (default 256 chunks).
    # The embedding model batches using its own batch_size.
    # To show a single progress bar for BOTH encoding and upserting, we will batch manually here.
    
    # We will process chunks in multiples of args.batch_size
    batch_size = args.batch_size
    
    with tqdm(total=len(chunks), desc="Indexing Chunks", unit="chunk") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch_chunks]
            
            # Encode texts using the model
            # Note: The underlying SentenceTransformer might also batch internally, 
            # but since we send args.batch_size elements, it processes them in one go.
            vectors = model.encode(batch_texts)
            
            # Prepare payload for Qdrant
            payloads = [c.to_dict() for c in batch_chunks]
            
            # Upsert into vector store
            store.upsert(payloads, vectors)
            
            pbar.update(len(batch_chunks))

    print(f"\n✅ Indexing Pipeline Complete! Indexed {len(chunks)} chunks into '{domain.index_name}'.")

    print(f"\n[5/5] Building and saving BM25 lexical index...")
    from retrieval.bm25 import build_bm25_index
    chunks_dicts = [c.to_dict() for c in chunks]
    bm25_idx = build_bm25_index(chunks_dicts)
    bm25_idx.save(domain.bm25_path)
    print(f"      BM25 Index saved to {domain.bm25_path}.")

if __name__ == "__main__":
    main()
