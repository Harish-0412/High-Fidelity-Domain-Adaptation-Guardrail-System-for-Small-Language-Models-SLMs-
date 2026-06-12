#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.rag import answer_query

queries = [
    "What is the dosage of amoxicillin?",
    "What are warfarin drug interactions?",
    "How is insulin glargine administered?",
    "Contraindications of semaglutide.",
    "What are aspirin side effects?"
]

def main():
    print("==============================================")
    print("WEEK 5 END-TO-END VALIDATION")
    print("==============================================\n")
    
    for q in queries:
        print(f"QUERY: {q}")
        print("-" * 40)
        
        # We use a raw string output format for easy reading in the console, but Ollama might output json since we set it up to do so if 'answer_with_citations' is used.
        # Wait, if output_format="answer_with_citations", Ollama outputs JSON. We can parse it.
        resp = answer_query(
            domain="medical_prescription", 
            query=q, 
            top_k=5, 
            output_format="answer_with_citations"
        )
        
        print("OOD Guardrail Decision:")
        if resp.guardrail_status.fallback_used and resp.guardrail_status.reason == "ood_threshold_failed":
            print(f"  --> TRIGGERED! (Score: {resp.guardrail_status.critic_score})")
        else:
            print(f"  --> PASSED (Fallback: {resp.guardrail_status.fallback_used})")
        
        print("\nTop Retrieved Chunks (from Citations):")
        for i, c in enumerate(resp.citations):
            # Print first 100 chars
            snippet = c.text[:150].replace('\n', ' ') + '...'
            print(f"  [{c.citation_id}] Score: {c.score:.4f} | Source: {c.source_id}")
            print(f"        Text: {snippet}")
        
        print("\nOllama Generated Answer:")
        print(resp.answer)
        
        print("\nCitation IDs Injected: ", end="")
        import re
        citations_found = re.findall(r'\[C\d+\]', resp.answer)
        print(list(set(citations_found)))
        
        print(f"Source documents used: {list(set([c.source_id for c in resp.citations]))}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
