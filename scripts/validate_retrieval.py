#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.core.domain_registry import get_domain_config
from retrieval.vector_store import try_build_qdrant_store
from retrieval.embeddings import load_embedding_model

def main():
    domain = get_domain_config("medical_prescription")
    store = try_build_qdrant_store("http://localhost:6333", domain.index_name, 384)
    if not store:
        print("Failed to connect to Qdrant")
        sys.exit(1)
        
    print("Loading embedding model BAAI/bge-small-en-v1.5...")
    model = load_embedding_model("BAAI/bge-small-en-v1.5", dimension=384)

    queries = [
        "What is the dosage of amoxicillin?",
        "What are the side effects of aspirin?",
        "What are warfarin drug interactions?",
        "How is insulin glargine administered?",
        "Contraindications of semaglutide."
    ]

    print("\n==========================================")
    print(" 🔎 WEEK 3 RETRIEVAL VALIDATION ")
    print("==========================================\n")

    for idx, query in enumerate(queries, 1):
        print(f"QUERY {idx}: \"{query}\"")
        print("-" * 60)
        
        # BGE models usually benefit from an instruction prefix for retrieval, but we will pass the raw query 
        # as it was encoded without prefix. Actually, BGE small en v1.5 requires "Represent this sentence for searching relevant passages: " 
        # for queries. Let's see if we should just use the raw query. Let's use raw query as standard RAG pipeline typically uses.
        query_vector = model.encode([query])[0]
        
        results = store.search(query_vector=query_vector, top_k=5)
        
        for rank, res in enumerate(results, 1):
            chunk = res.chunk
            print(f"  [Rank {rank}] Score: {res.score:.4f}")
            print(f"  Source Document: {chunk.get('source_id')} (Page {chunk.get('page')})")
            print(f"  Chunk ID: {chunk.get('chunk_id')}")
            # print snippet
            text = chunk.get('text', '')
            # replace newlines to make it compact
            snippet = " ".join(text.split())
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"  Text: {snippet}")
            print()
            
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
