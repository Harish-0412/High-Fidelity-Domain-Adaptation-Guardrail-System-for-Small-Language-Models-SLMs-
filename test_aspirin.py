#!/usr/bin/env python3
from domain_slm_guardrails.api.rag import answer_query, RAGConfig

print("Testing 'What is Aspirin?' query")
result = answer_query(
    domain='medications',
    query='What is Aspirin?',
    config=RAGConfig(use_llm_generation=False)
)
print(f"\nQuery: {result.query}")
print(f"\nAnswer:\n{result.answer}")
print()
print("Citations:")
for idx, c in enumerate(result.citations, 1):
    print(f"\n  [{c.citation_id}] Score: {c.score:.4f}")
    print(f"  Chunk ID: {c.chunk_id}")
    print(f"  Text: {c.text}")
