#!/usr/bin/env python3
from domain_slm_guardrails.retrieval.hybrid import load_hybrid_retriever

retriever = load_hybrid_retriever("medications")
results = retriever.search(query="What is Aspirin?", top_k=5, expanded_query="aspirin")

print("Top 5 retrieval results:")
for idx, r in enumerate(results, 1):
    print(f"\nResult {idx} (score: {r.score:.4f})")
    print(f"Chunk ID: {r.chunk.get('chunk_id')}")
    print(f"Text: {r.chunk.get('text')[:400]}")

