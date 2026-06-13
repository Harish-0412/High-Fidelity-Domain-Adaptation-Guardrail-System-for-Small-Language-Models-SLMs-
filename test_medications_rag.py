#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain_slm_guardrails.api.rag import answer_query, RAGConfig
from domain_slm_guardrails.core.config import load_base_config


def main() -> None:
    print("Testing Medications RAG System")
    print("=" * 50)

    test_queries = [
        "What are the side effects of paracetamol?",
        "What should I do for a fever?",
        "What is metformin used for?",
        "How do I treat a headache?",
        "What are the uses of amoxicillin?"
    ]

    config = RAGConfig(use_llm_generation=False)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        try:
            result = answer_query(
                domain="medications",
                query=query,
                config=config
            )
            print(f"Answer: {result.answer}")
            print(f"\nCitations Found: {len(result.citations)}")
            print(f"Latency: {result.latency_ms:.2f}ms")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
