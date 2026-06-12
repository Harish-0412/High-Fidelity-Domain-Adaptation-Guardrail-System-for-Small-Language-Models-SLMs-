#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.core.domain_registry import get_domain_config
from retrieval.hybrid import load_hybrid_retriever
from retrieval.rerank import RerankResult

def main():
    domain_id = "medical_prescription"
    domain = get_domain_config(domain_id)
    
    # We force the settings to enable MMR and Reranking for this test
    domain.settings["use_mmr"] = True
    domain.settings["mmr_lambda"] = 0.5
    domain.settings["use_rerank"] = True
    domain.settings["candidate_multiplier"] = 5
    
    print("Loading Hybrid Retriever (Dense + BM25 + MMR + Reranker)...")
    retriever = load_hybrid_retriever(domain_id)
    
    queries = [
        "What is the dosage of amoxicillin?",
        "What are the side effects of aspirin?",
        "What are warfarin drug interactions?",
    ]

    print("\n==========================================")
    print(" 🔬 HYBRID RETRIEVAL PIPELINE VALIDATION ")
    print("==========================================\n")

    for idx, query in enumerate(queries, 1):
        print(f"QUERY {idx}: \"{query}\"")
        print("-" * 60)
        
        # BGE models usually benefit from an instruction prefix, but here we just use the original pipeline
        results = retriever.search(query=query, top_k=5)
        
        for rank, res in enumerate(results, 1):
            if isinstance(res, RerankResult):
                chunk = res.chunk
                score_str = f"Rerank Score: {res.score:.4f} (Original Rank: {res.original_rank})"
            else:
                chunk = res.chunk
                score_str = f"Hybrid Score: {res.score:.4f}"
                
            print(f"  [Rank {rank}] {score_str}")
            print(f"  Source Document: {chunk.get('source_id')} (Page {chunk.get('page')})")
            
            text = chunk.get('text', '')
            snippet = " ".join(text.split())
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"  Text: {snippet}")
            print()
            
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
