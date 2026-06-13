#!/usr/bin/env python3

from domain_slm_guardrails.retrieval.preprocessor import QueryPreprocessor
from domain_slm_guardrails.retrieval.diversity import mmr_rerank
from domain_slm_guardrails.retrieval.metrics import ndcg, mrr, recall_at_k, precision_at_k, map_score


def test_preprocessor():
    print("=== Testing Query Preprocessor ===")
    preprocessor = QueryPreprocessor("medical_prescription")
    query = "What's the dose of paracetamol?"
    processed = preprocessor.preprocess(query)
    print(f"Original query: {processed.original}")
    print(f"Cleaned query: {processed.cleaned}")
    print(f"Expanded query: {processed.expanded}")
    print()


def test_metrics():
    print("=== Testing Metrics ===")
    
    # NDCG
    relevance_scores = [3, 2, 3, 0, 1, 2]
    ndcg_at_6 = ndcg(relevance_scores, k=6)
    print(f"NDCG@6: {ndcg_at_6:.4f}")
    
    # MRR
    mrr_result = mrr([0, 0, 1, 0, 1, 0])
    print(f"MRR: {mrr_result:.4f}")
    
    # Recall/Precision
    retrieved_ids = ["id1", "id2", "id3", "id4"]
    relevant_ids = ["id2", "id4", "id6"]
    recall = recall_at_k(retrieved_ids, relevant_ids, 3)
    precision = precision_at_k(retrieved_ids, relevant_ids, 3)
    print(f"Recall@3: {recall:.4f}")
    print(f"Precision@3: {precision:.4f}")
    print()


def test_mmr():
    print("=== Testing MMR Reranking ===")
    chunks = [
        {"text": "Paracetamol is used for pain relief and fever.", "score": 0.9, "chunk_id": "1"},
        {"text": "Paracetamol dosage is 500-1000mg every 4-6 hours.", "score": 0.85, "chunk_id": "2"},
        {"text": "Paracetamol should not be taken with alcohol.", "score": 0.8, "chunk_id": "3"},
    ]
    
    query = "What's the dosage of paracetamol?"
    reranked = mmr_rerank(chunks, query, lambda_param=0.7, top_k=3)
    print("MMR reranked results:")
    for i, chunk in enumerate(reranked, 1):
        print(f"  {i}: {chunk.get('text', '')[:50]}...")
    print()


def test_retriever():
    print("=== Testing Hybrid Retriever with Query Preprocessing ===")
    from domain_slm_guardrails.retrieval.hybrid import load_hybrid_retriever
    retriever = load_hybrid_retriever("medical_prescription")
    query = "What's the dose of paracetamol?"
    results = retriever.search(query, top_k=3)
    print(f"Search query: {query}")
    print(f"Number of results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Chunk ID: {result.chunk['chunk_id']}")
        print(f"  Text preview: {result.chunk['text'][:100]}...")


if __name__ == "__main__":
    test_preprocessor()
    test_metrics()
    test_mmr()
    test_retriever()
    print("All components working!")
